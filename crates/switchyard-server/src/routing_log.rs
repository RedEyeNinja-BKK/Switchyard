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
use switchyard_protocol::{Metadata, ModelId, Usage};

use crate::usage_metrics::token_usage;
use crate::{ServerError, ServerResult};

const LEGACY_SESSION_ID_HEADER: &str = "proxy_x_session_id";
const TASK_HEADER: &str = "x-switchyard-intake-task";
const TRIAL_ID_HEADER: &str = "x-switchyard-trial-id";

/// Append-only writer for one routing JSONL file.
pub(crate) struct RoutingLog(fs::File);

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
        Ok(Self(file))
    }

    pub(crate) fn append(
        &mut self,
        context: RoutingLogContext,
        model: &str,
        tier: Option<&str>,
        usage: &Usage,
    ) -> std::io::Result<()> {
        let usage = token_usage(usage);
        let record = RoutingRecord {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            task: context.task.map(Cow::Owned),
            trial_id: context.trial_id.map(Cow::Owned),
            session_id: context.session_id.map(Cow::Owned),
            requested_route: context.requested_route.map(Cow::Owned),
            agent_id: context.agent_id.map(Cow::Owned),
            task_id: context.task_id.map(Cow::Owned),
            task_kind: context.task_kind.map(Cow::Owned),
            turn_id: context.turn_id.map(Cow::Owned),
            correlation_id: context.correlation_id.map(Cow::Owned),
            agent_kind: context.agent_kind.map(Cow::Owned),
            agent_role: context.agent_role.map(Cow::Owned),
            is_subagent: context.is_subagent,
            is_delegated_work: context.is_delegated_work,
            routing_principal: context.routing_principal.map(Cow::Owned),
            policy_domain: context.policy_domain.map(Cow::Owned),
            work_shape: context.work_shape.map(Cow::Owned),
            reasoning_intent: context.reasoning_intent.map(Cow::Owned),
            tool_required: context.tool_required.map(Cow::Owned),
            model: model.into(),
            tier: tier.unwrap_or("").into(),
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.cached_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            completion_tokens: usage.completion_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens: usage.prompt_tokens.saturating_add(usage.completion_tokens),
        };
        let mut line = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        line.push(b'\n');

        self.0.write_all(&line)
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

/// Request fields retained until terminal usage and routing are available.
///
/// Serialization contract for the durable routing JSONL:
/// - String identity/envelope fields are `Option`: when the caller supplied no
///   value, the record omits (or nulls) the field rather than fabricating one.
/// - `is_subagent` / `is_delegated_work` are plain booleans derived from inbound
///   metadata (absent = `false`); they are always present in a record.
/// - These are caller-DECLARED metadata for observability only. Their presence
///   is NOT evidence of trusted/authenticated authority (declared != trusted).
/// - `requested_route` is stamped after route resolution so the caller's route
///   alias is preserved independently of the resolved physical `model`, and stays
///   absent when no route alias was supplied.
/// - Records are forward/backward compatible: `#[serde(default)]` lets older
///   records (missing new fields) and newer records parse through unchanged.
#[derive(Clone, Default)]
pub(crate) struct RoutingLogContext {
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
    requested_route: Option<String>,
    agent_id: Option<String>,
    task_id: Option<String>,
    task_kind: Option<String>,
    turn_id: Option<String>,
    correlation_id: Option<String>,
    agent_kind: Option<String>,
    agent_role: Option<String>,
    is_subagent: bool,
    is_delegated_work: bool,
    // Narrow allowlisted subset of `extra_metadata`, present only when supplied.
    routing_principal: Option<String>,
    policy_domain: Option<String>,
    work_shape: Option<String>,
    reasoning_intent: Option<String>,
    tool_required: Option<String>,
}

impl RoutingLogContext {
    /// Captures the normalized session ID, with the legacy log-only header as a fallback.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        let headers = metadata.http_headers.as_ref();
        let envelope = extra_metadata_allowlist(metadata);
        Self {
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
            requested_route: None,
            agent_id: metadata.agent_id.clone(),
            task_id: metadata.task_id.clone(),
            task_kind: metadata.task_kind.clone(),
            turn_id: metadata.turn_id.clone(),
            correlation_id: metadata.correlation_id.clone(),
            agent_kind: metadata.agent_kind.clone(),
            agent_role: metadata.agent_role.clone(),
            is_subagent: metadata.is_subagent,
            is_delegated_work: metadata.is_delegated_work,
            routing_principal: envelope.routing_principal,
            policy_domain: envelope.policy_domain,
            work_shape: envelope.work_shape,
            reasoning_intent: envelope.reasoning_intent,
            tool_required: envelope.tool_required,
        }
    }

    /// Records the caller's pre-resolution route alias, kept independent of the
    /// resolved physical model. Stamped once route resolution succeeds.
    pub(crate) fn with_requested_route(mut self, requested_route: String) -> Self {
        self.requested_route = Some(requested_route);
        self
    }
}

/// The allowlisted `extra_metadata` routing-envelope subset.
///
/// Reads only the keys below out of `Metadata::extra_metadata`. Any other key,
/// header, secret, or arbitrary metadata map is intentionally NOT captured.
/// Each `(key, setter)` pair is the single authoritative definition of which
/// `extra_metadata` key maps to which record field, so there is no separate
/// string list to drift.
struct EnvelopeAllowlist {
    routing_principal: Option<String>,
    policy_domain: Option<String>,
    work_shape: Option<String>,
    reasoning_intent: Option<String>,
    tool_required: Option<String>,
}

impl EnvelopeAllowlist {
    fn new() -> Self {
        Self {
            routing_principal: None,
            policy_domain: None,
            work_shape: None,
            reasoning_intent: None,
            tool_required: None,
        }
    }
}

fn extra_metadata_allowlist(metadata: &Metadata) -> EnvelopeAllowlist {
    let mut envelope = EnvelopeAllowlist::new();
    let Some(map) = metadata.extra_metadata.as_ref() else {
        return envelope;
    };
    // One authoritative source of the allowlist key -> target field mapping.
    for (key, setter) in ENVELOPE_FIELD_MAP {
        if let Some(nonempty) = map.get(*key).filter(|value| !value.is_empty()) {
            setter(&mut envelope, nonempty.clone());
        }
    }
    envelope
}

/// Setter for one allowlisted field inside [`EnvelopeAllowlist`].
type EnvelopeSetter = fn(&mut EnvelopeAllowlist, String);

/// Maps each allowlisted `extra_metadata` key to the record field it feeds.
static ENVELOPE_FIELD_MAP: &[(&str, EnvelopeSetter)] = &[
    ("routing_principal", |e, v| e.routing_principal = Some(v)),
    ("policy_domain", |e, v| e.policy_domain = Some(v)),
    ("work_shape", |e, v| e.work_shape = Some(v)),
    ("reasoning_intent", |e, v| e.reasoning_intent = Some(v)),
    ("tool_required", |e, v| e.tool_required = Some(v)),
];

/// One appended routing record, and the read schema [`snapshot`] parses back,
/// so the written and expected shapes cannot drift apart. Missing fields
/// default so a record from an older schema still contributes what it has.
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct RoutingRecord<'a> {
    ts: Cow<'a, str>,
    #[serde(borrow)]
    task: Option<Cow<'a, str>>,
    #[serde(borrow)]
    trial_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    requested_route: Option<Cow<'a, str>>,
    #[serde(borrow)]
    agent_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    task_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    task_kind: Option<Cow<'a, str>>,
    #[serde(borrow)]
    turn_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    correlation_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    agent_kind: Option<Cow<'a, str>>,
    #[serde(borrow)]
    agent_role: Option<Cow<'a, str>>,
    #[serde(borrow)]
    routing_principal: Option<Cow<'a, str>>,
    #[serde(borrow)]
    policy_domain: Option<Cow<'a, str>>,
    #[serde(borrow)]
    work_shape: Option<Cow<'a, str>>,
    #[serde(borrow)]
    reasoning_intent: Option<Cow<'a, str>>,
    #[serde(borrow)]
    tool_required: Option<Cow<'a, str>>,
    is_subagent: bool,
    is_delegated_work: bool,
    model: Cow<'a, str>,
    tier: Cow<'a, str>,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
    reasoning_tokens: u64,
    total_tokens: u64,
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
                r#"{"session_id":"a","prompt_tokens":5}"#,
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

    /// Only the allowlisted `extra_metadata` routing-envelope keys are captured.
    /// Arbitrary or sensitive keys are never propagated into the routing log.
    #[test]
    fn from_metadata_allowlists_only_envelope_keys() {
        let mut map = BTreeMap::new();
        map.insert(
            "routing_principal".to_string(),
            "stephen-remote-openclaw".to_string(),
        );
        map.insert("policy_domain".to_string(), "managed-external".to_string());
        map.insert("work_shape".to_string(), "bounded".to_string());
        map.insert("reasoning_intent".to_string(), "none".to_string());
        map.insert("tool_required".to_string(), "false".to_string());
        map.insert("sensitive_secret".to_string(), "must-not-leak".to_string());
        map.insert("user_content".to_string(), "must-not-leak".to_string());
        let metadata = Metadata {
            agent_id: Some("openclaw-remote".to_string()),
            extra_metadata: Some(map.clone()),
            ..Metadata::default()
        };

        let context = RoutingLogContext::from_metadata(&metadata);
        // Allowlisted envelope keys present.
        assert_eq!(
            context.routing_principal.as_deref(),
            Some("stephen-remote-openclaw")
        );
        assert_eq!(context.policy_domain.as_deref(), Some("managed-external"));
        assert_eq!(context.work_shape.as_deref(), Some("bounded"));
        assert_eq!(context.reasoning_intent.as_deref(), Some("none"));
        assert_eq!(context.tool_required.as_deref(), Some("false"));
        // Declared identity captured.
        assert_eq!(context.agent_id.as_deref(), Some("openclaw-remote"));
        // Sensitive / unrecognized keys are NOT captured; the full map is not stored.
        let record = RoutingRecord {
            requested_route: context.requested_route.map(Cow::Owned),
            agent_id: context.agent_id.map(Cow::Owned),
            task_id: context.task_id.map(Cow::Owned),
            task_kind: context.task_kind.map(Cow::Owned),
            turn_id: context.turn_id.map(Cow::Owned),
            correlation_id: context.correlation_id.map(Cow::Owned),
            agent_kind: context.agent_kind.map(Cow::Owned),
            agent_role: context.agent_role.map(Cow::Owned),
            routing_principal: context.routing_principal.map(Cow::Owned),
            policy_domain: context.policy_domain.map(Cow::Owned),
            work_shape: context.work_shape.map(Cow::Owned),
            reasoning_intent: context.reasoning_intent.map(Cow::Owned),
            tool_required: context.tool_required.map(Cow::Owned),
            is_subagent: context.is_subagent,
            is_delegated_work: context.is_delegated_work,
            ..Default::default()
        };
        let serialized = serde_json::to_string(&record).expect("serialize");
        assert!(!serialized.contains("sensitive_secret"));
        assert!(!serialized.contains("user_content"));
        assert!(!serialized.contains("must-not-leak"));
    }
}
