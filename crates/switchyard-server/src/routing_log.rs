// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable per-request routing records and session snapshots.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use humantime::format_rfc3339_millis;
use serde::{Deserialize, Serialize};
use switchyard_llm_client::SelectionTrace;
use switchyard_protocol::{Metadata, ModelId, Usage};

use crate::usage_metrics::token_usage;
use crate::{ServerError, ServerResult};

const LEGACY_SESSION_ID_HEADER: &str = "proxy_x_session_id";
const ORIGIN_HEADER: &str = "x-switchyard-origin";
const TASK_HEADER: &str = "x-switchyard-intake-task";
const TRIAL_ID_HEADER: &str = "x-switchyard-trial-id";

/// Append-only writer for one routing JSONL file, plus the decision log beside it.
pub(crate) struct RoutingLog {
    attempts: fs::File,
    decisions: fs::File,
}

impl RoutingLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| routing_log_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| routing_log_error(&path, error))?;
        // The decision stream is a separate file on purpose: an attempt record carries
        // usage, a decision record carries the candidates that were never attempted, and a
        // session snapshot must not have to tell two shapes apart to sum tokens.
        let decision_path = path.with_extension("decisions.jsonl");
        let decisions = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&decision_path)
            .map_err(|error| routing_log_error(&decision_path, error))?;
        Ok(Self {
            attempts: file,
            decisions,
        })
    }

    /// One request's selection decision, written to the decision log.
    pub(crate) fn append_decision(
        &mut self,
        context: RoutingLogContext,
        trace: &SelectionTrace,
    ) -> std::io::Result<()> {
        let record = RoutingDecisionRecord::from_trace(context, trace);
        let mut line = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.decisions.write_all(&line)
    }

    /// Composition facts for one request, appended to the same record so the served
    /// model and the child that produced it are read together rather than joined later.
    pub(crate) fn append_with(
        &mut self,
        context: RoutingLogContext,
        model: &str,
        tier: Option<&str>,
        usage: &Usage,
        composition: Option<&switchyard_runner::composition::CompositionMetadata>,
    ) -> std::io::Result<()> {
        let usage = token_usage(usage);
        let record = RoutingRecord {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            route_id: context.route_id.into(),
            algorithm: context.algorithm.into(),
            origin: context.origin.map(Cow::Owned),
            task: context.task.map(Cow::Owned),
            trial_id: context.trial_id.map(Cow::Owned),
            session_id: context.session_id.map(Cow::Owned),
            model: model.into(),
            tier: tier.unwrap_or("").into(),
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.cached_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            completion_tokens: usage.completion_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens: usage.prompt_tokens.saturating_add(usage.completion_tokens),
            child_route: composition
                .map(|composition| composition.child_route.clone().into())
                .unwrap_or_else(|| "".into()),
            final_target: composition
                .map(|composition| composition.final_target.clone().into())
                .unwrap_or_else(|| "".into()),
            cross_child_reason: composition
                .and_then(|composition| composition.cross_child_reason())
                .unwrap_or_default()
                .into(),
        };
        let mut line = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        line.push(b'\n');

        self.attempts.write_all(&line)
    }
}

/// Reads complete records without synchronizing with the writer.
pub(crate) fn snapshot(
    path: &Path,
    session_id: &str,
) -> std::io::Result<Option<SessionStatsSnapshot>> {
    let mut reader = BufReader::with_capacity(64 * 1024, fs::File::open(path)?);
    let mut line = Vec::new();
    let mut snapshot = SessionStatsSnapshot::new(session_id);
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if !line.ends_with(b"\n") {
            break;
        }
        let Ok(record) = serde_json::from_slice::<RoutingRecord>(&line) else {
            continue;
        };
        snapshot.add_record(&record, session_id);
    }
    snapshot.sum_totals();
    Ok((snapshot.total_calls > 0).then_some(snapshot))
}

/// Holds request metadata until route resolution attaches the durable route identity.
#[derive(Clone)]
pub(crate) struct RoutingLogContext {
    route_id: String,
    algorithm: String,
    origin: Option<String>,
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
}

impl RoutingLogContext {
    /// Captures the normalized session ID, with the legacy log-only header as a fallback.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        let headers = metadata.http_headers.as_ref();
        Self {
            route_id: String::new(),
            algorithm: String::new(),
            origin: headers
                .and_then(|headers| nonempty_header(headers, ORIGIN_HEADER))
                .map(str::to_string),
            task: headers
                .and_then(|headers| nonempty_header(headers, TASK_HEADER))
                .map(str::to_string),
            trial_id: headers
                .and_then(|headers| nonempty_header(headers, TRIAL_ID_HEADER))
                .map(str::to_string),
            session_id: metadata.session_id.clone().or_else(|| {
                headers
                    .and_then(|headers| nonempty_header(headers, LEGACY_SESSION_ID_HEADER))
                    .map(str::to_string)
            }),
        }
    }

    /// Attaches the resolved route identity shared by every log entry for this request.
    pub(crate) fn with_route(mut self, route_id: impl Into<String>, algorithm: &str) -> Self {
        self.route_id = route_id.into();
        self.algorithm = algorithm.to_string();
        self
    }
}

/// One appended routing record, and the read schema [`snapshot`] parses back,
/// so the written and expected shapes cannot drift apart. Missing fields
/// default so a record from an older schema still contributes what it has.
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct RoutingRecord<'a> {
    ts: Cow<'a, str>,
    route_id: Cow<'a, str>,
    algorithm: Cow<'a, str>,
    #[serde(borrow)]
    origin: Option<Cow<'a, str>>,
    #[serde(borrow)]
    task: Option<Cow<'a, str>>,
    #[serde(borrow)]
    trial_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    model: Cow<'a, str>,
    tier: Cow<'a, str>,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
    reasoning_tokens: u64,
    /// Child route that served the request, when a composite served it.
    child_route: Cow<'a, str>,
    /// The model that actually answered, as recorded by the composition.
    final_target: Cow<'a, str>,
    /// Why the composite moved on from an earlier child, empty when it did not.
    cross_child_reason: Cow<'a, str>,
    total_tokens: u64,
}

/// One request's selection decision, written to the decision log beside the routing log.
///
/// The routing log records attempts that completed and produced usage. This record names the
/// candidates that were *never* attempted and why, which is the half of the story the
/// attempt log cannot carry: a request whose Google leg was skipped on privacy grounds has no
/// usage line at all, so without this the exclusion is indistinguishable from a route that
/// never declared Google.
#[derive(Serialize)]
struct RoutingDecisionRecord<'a> {
    ts: Cow<'a, str>,
    route_id: Cow<'a, str>,
    algorithm: Cow<'a, str>,
    /// `served`, `no_eligible_candidate` or `failed`.
    outcome: &'static str,
    served: Option<Cow<'a, str>>,
    /// Bytes the conversation was costed at, including any output reservation.
    demand_tokens: u64,
    /// Input tokens the hard gates judged the payload by.
    input_tokens: u64,
    /// Declared order, so a reader can tell a demotion from a declaration.
    considered: Vec<String>,
    /// The order actually handed to dispatch.
    dispatched: Vec<String>,
    /// Candidate to the rule that excluded it before any call.
    skipped: BTreeMap<String, &'static str>,
    /// Candidate to the measurement that moved it down the order.
    demoted: BTreeMap<String, &'static str>,
    /// Leg the context-pressure rule offered first, when it did.
    promotion: Option<String>,
    attempts: Vec<AttemptDecision>,
}

#[derive(Serialize)]
struct AttemptDecision {
    model: String,
    /// `routing` or `answer`.
    phase: &'static str,
    /// Stable outcome code, `served` when the call completed.
    outcome: &'static str,
    duration_ms: u64,
}

impl<'a> RoutingDecisionRecord<'a> {
    fn from_trace(context: RoutingLogContext, trace: &SelectionTrace) -> Self {
        let mut skipped = BTreeMap::new();
        let mut demoted = BTreeMap::new();
        for candidate in &trace.candidates {
            if let Some(code) = candidate.exclusion {
                skipped.insert(candidate.model.clone(), code);
            }
            if let Some(code) = candidate.demotion {
                demoted.insert(candidate.model.clone(), code);
            }
        }
        Self {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            route_id: context.route_id.into(),
            algorithm: context.algorithm.into(),
            outcome: trace.outcome,
            served: trace.served.clone().map(Cow::Owned),
            demand_tokens: trace.demand_tokens,
            input_tokens: trace.estimated_input_tokens,
            considered: trace.considered.clone(),
            dispatched: trace.dispatched.clone(),
            skipped,
            demoted,
            promotion: trace.promotion.clone(),
            attempts: trace
                .attempts
                .iter()
                .map(|attempt| AttemptDecision {
                    model: attempt.model.clone(),
                    phase: attempt.phase,
                    outcome: attempt.outcome,
                    duration_ms: attempt.duration_ms,
                })
                .collect(),
        }
    }
}

/// Session totals returned by the routing stats endpoint.
#[derive(Serialize)]
pub(crate) struct SessionStatsSnapshot {
    session_id: String,
    total_calls: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
    total_cache_creation_tokens: u64,
    total_completion_tokens: u64,
    models: BTreeMap<ModelId, SessionModelStats>,
}

#[derive(Default, Serialize)]
struct SessionModelStats {
    calls: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
}

impl SessionStatsSnapshot {
    fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            total_calls: 0,
            total_prompt_tokens: 0,
            total_cached_tokens: 0,
            total_cache_creation_tokens: 0,
            total_completion_tokens: 0,
            models: BTreeMap::new(),
        }
    }

    fn add_record(&mut self, record: &RoutingRecord<'_>, session_id: &str) {
        if record.session_id.as_deref() != Some(session_id) {
            return;
        }
        let model = match record.model.as_ref() {
            "" => "unknown",
            model => model,
        };
        let stats = self.models.entry(ModelId::from(model)).or_default();
        stats.calls = stats.calls.saturating_add(1);
        stats.prompt_tokens = stats.prompt_tokens.saturating_add(record.prompt_tokens);
        stats.cached_tokens = stats.cached_tokens.saturating_add(record.cached_tokens);
        stats.cache_creation_tokens = stats
            .cache_creation_tokens
            .saturating_add(record.cache_creation_tokens);
        stats.completion_tokens = stats
            .completion_tokens
            .saturating_add(record.completion_tokens);
    }

    /// Session totals are exactly the sum of the per-model stats, so they are
    /// derived once rather than accumulated alongside them.
    fn sum_totals(&mut self) {
        for stats in self.models.values() {
            self.total_calls = self.total_calls.saturating_add(stats.calls);
            self.total_prompt_tokens = self.total_prompt_tokens.saturating_add(stats.prompt_tokens);
            self.total_cached_tokens = self.total_cached_tokens.saturating_add(stats.cached_tokens);
            self.total_cache_creation_tokens = self
                .total_cache_creation_tokens
                .saturating_add(stats.cache_creation_tokens);
            self.total_completion_tokens = self
                .total_completion_tokens
                .saturating_add(stats.completion_tokens);
        }
    }
}

fn nonempty_header<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .filter(|value| !value.is_empty())
        .and_then(|v| v.to_str().ok())
}

fn routing_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize routing log {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_llm_client::{AttemptTrace, CandidateTrace};

    /// Missing, empty, and non-text origin headers remain absent even with a User-Agent.
    #[test]
    fn unusable_origin_is_not_inferred_from_user_agent() {
        for origin in [
            None,
            Some(http::HeaderValue::from_static("")),
            Some(http::HeaderValue::from_bytes(b"\xff").expect("header")),
        ] {
            let mut headers = http::HeaderMap::new();
            headers.insert(
                "user-agent",
                http::HeaderValue::from_static("codex-cli/1.0"),
            );
            if let Some(origin) = origin {
                headers.insert(ORIGIN_HEADER, origin);
            }
            let metadata = Metadata {
                http_headers: Some(headers),
                ..Default::default()
            };
            assert!(RoutingLogContext::from_metadata(&metadata).origin.is_none());
        }
        assert!(
            RoutingLogContext::from_metadata(&Metadata::default())
                .origin
                .is_none()
        );
    }

    /// Only the requested session is counted, absent fields fall back to zero
    /// and `unknown`, and an unparseable line does not abort the scan.
    #[test]
    fn snapshot_counts_only_the_requested_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        fs::write(
            &path,
            concat!(
                r#"{"session_id":"a","model":"m1","fallback_reason":"unavailable","prompt_tokens":10,"completion_tokens":2}"#,
                "\n",
                r#"{"session_id":"b","model":"m1","prompt_tokens":99,"completion_tokens":99}"#,
                "\n",
                "not json\n",
                r#"{"session_id":"a","origin":"custom-agent","prompt_tokens":5}"#,
                "\n",
            ),
        )
        .expect("write log");

        let stats = snapshot(&path, "a").expect("read log").expect("session a");
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.total_prompt_tokens, 15);
        assert_eq!(stats.total_completion_tokens, 2);
        assert_eq!(stats.models["m1"].calls, 1);
        assert_eq!(stats.models["unknown"].prompt_tokens, 5);
        assert!(snapshot(&path, "missing").expect("read log").is_none());
    }

    /// A decision record states the two things the attempt log cannot: which candidate was
    /// never contacted and why, and which one was moved down after being measured.
    #[test]
    fn a_decision_record_names_skipped_and_demoted_candidates() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        let mut log = RoutingLog::new(&path).expect("routing log");
        let trace = SelectionTrace {
            algorithm: "passthrough".to_string(),
            considered: vec!["gated".to_string(), "healthy".to_string()],
            candidates: vec![
                CandidateTrace {
                    model: "gated".to_string(),
                    eligible: false,
                    exclusion: Some("privacy_refused"),
                    tier: "free",
                    rank: None,
                    remaining_today: Some(14_400),
                    pressure_permille: 120,
                    demotion: None,
                },
                CandidateTrace {
                    model: "healthy".to_string(),
                    eligible: true,
                    exclusion: None,
                    tier: "free",
                    rank: Some(0),
                    remaining_today: Some(500),
                    pressure_permille: 40,
                    demotion: None,
                },
            ],
            dispatched: vec!["healthy".to_string()],
            attempts: vec![AttemptTrace {
                model: "healthy".to_string(),
                phase: "answer",
                outcome: "served",
                duration_ms: 412,
                prompt_tokens: 900,
                completion_tokens: 20,
            }],
            served: Some("healthy".to_string()),
            outcome: "served",
            demand_tokens: 1_234,
            estimated_input_tokens: 1_000,
            promotion: None,
        };
        log.append_decision(
            RoutingLogContext {
                route_id: "gemini-lite".to_string(),
                algorithm: "passthrough".to_string(),
                origin: None,
                task: None,
                trial_id: None,
                session_id: None,
            },
            &trace,
        )
        .expect("append decision");

        let written = fs::read_to_string(path.with_extension("decisions.jsonl"))
            .expect("decision log written beside the routing log");
        let record: serde_json::Value =
            serde_json::from_str(written.lines().next().expect("one line")).expect("valid json");
        assert_eq!(record["route_id"], "gemini-lite");
        assert_eq!(record["outcome"], "served");
        assert_eq!(record["served"], "healthy");
        assert_eq!(record["skipped"]["gated"], "privacy_refused");
        assert_eq!(record["demoted"], serde_json::json!({}));
        assert_eq!(record["dispatched"][0], "healthy");
        assert_eq!(record["attempts"][0]["outcome"], "served");
        assert_eq!(record["attempts"][0]["duration_ms"], 412);
        // Control: the attempt stream is a different file, so a decision line can never be
        // summed into token totals by a session snapshot.
        assert!(
            !path.exists() || fs::read_to_string(&path).expect("read").is_empty(),
            "an attempt record must not be written by a decision append"
        );
    }
}
