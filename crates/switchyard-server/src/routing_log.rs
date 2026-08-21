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
            declared_is_subagent: context.declared_is_subagent,
            is_delegated_work: context.is_delegated_work,
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
/// Serialization contract for the durable routing JSONL (categorized by
/// normalized-metadata vs explicit-declaration vs harness-derived):
/// - **Normalized request metadata** (`agent_id`, `task_id`, `task_kind`,
///   `turn_id`, `correlation_id`, `agent_kind`, `agent_role`,
///   `is_subagent`): the aggregated upstream `Metadata` values as Switchyard
///   already normalizes them (for `is_subagent`, all supported signals). These
///   are an upstream classification, not a caller credential.
/// - **Explicit caller declaration** (`declared_is_subagent`): `Option<bool>`
///   read from the explicit `x-switchyard-is-subagent` header via
///   `Metadata::declared_is_subagent()` — absent => `null`, explicit true/false
///   => `Some(true)`/`Some(false)`, unparseable => `null`.
/// - **Harness-derived classification** (`is_delegated_work`): the
///   upstream-normalized `Metadata.is_delegated_work`, computed from raw harness
///   signals (no caller declaration exists for it, so no presence is claimed).
/// - **Trust boundary:** `normalized metadata != authenticated principal !=
///   governance authority != effective routing authorization`. `agent_id`,
///   `is_subagent`, `declared_is_subagent`, and `is_delegated_work` are
///   observability facts only.
/// - `requested_route` is stamped after route resolution so the caller's route
///   alias is preserved independently of the resolved physical `model`, and stays
///   absent when no route alias was supplied.
/// - Records are forward/backward compatible: `#[serde(default)]` lets older
///   records (missing new fields) and newer records parse through unchanged.
/// - Durable telemetry applies to normal answer-serving request paths that
///   generate terminal routing/usage records (not the `/v1/decision` path).
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
    /// Upstream normalized `Metadata.is_subagent` (all supported signals).
    is_subagent: bool,
    /// Explicit `x-switchyard-is-subagent` declaration only (absent => `None`).
    declared_is_subagent: Option<bool>,
    /// Upstream harness-derived `Metadata.is_delegated_work`.
    is_delegated_work: bool,
}

impl RoutingLogContext {
    /// Captures the normalized session ID, with the legacy log-only header as a fallback.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        let headers = metadata.http_headers.as_ref();
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
            declared_is_subagent: metadata.declared_is_subagent(),
            is_delegated_work: metadata.is_delegated_work,
        }
    }

    /// Records the caller's pre-resolution route alias, kept independent of the
    /// resolved physical model. Stamped once route resolution succeeds.
    pub(crate) fn with_requested_route(mut self, requested_route: String) -> Self {
        self.requested_route = Some(requested_route);
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
    is_subagent: bool,
    declared_is_subagent: Option<bool>,
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

    /// The durable record preserves the distinct concepts `is_subagent` (upstream
    /// normalized, bool), `declared_is_subagent` (explicit header, Option<bool>),
    /// and `is_delegated_work` (harness-derived, bool). Covers absent / explicit
    /// false / explicit true / native-harness-child-without-override / malformed.
    /// Arbitrary `extra_metadata` is never propagated into the routing record.
    #[test]
    fn preserves_normalized_and_declared_subagent_distinctly() {
        fn context_from(
            value: Option<&str>,
            normalized_subagent: bool,
            delegated_work: bool,
        ) -> RoutingLogContext {
            let mut map = BTreeMap::new();
            map.insert("sensitive_secret".to_string(), "must-not-leak".to_string());
            map.insert("user_content".to_string(), "must-not-leak".to_string());
            let mut metadata = Metadata {
                agent_id: Some("openclaw-remote".to_string()),
                // The normalizer and the harness-derived flag are independent
                // facts; the fixture supplies each separately (test-data only,
                // no semantic coupling implied between the two fields).
                is_subagent: normalized_subagent,
                is_delegated_work: delegated_work,
                extra_metadata: Some(map.clone()),
                ..Metadata::default()
            };
            if let Some(header_value) = value {
                let mut h = http::HeaderMap::new();
                h.insert(
                    "x-switchyard-is-subagent",
                    header_value.parse().expect("header value"),
                );
                metadata.http_headers = Some(h);
            }
            RoutingLogContext::from_metadata(&metadata)
        }

        // Case A — no subagent signal: normalized false, no declaration.
        let ctx = context_from(None, false, false);
        assert!(!ctx.is_subagent);
        assert_eq!(ctx.declared_is_subagent, None);
        assert!(!ctx.is_delegated_work);

        // Case B — explicit false: both false.
        let ctx = context_from(Some("false"), false, false);
        assert!(!ctx.is_subagent);
        assert_eq!(ctx.declared_is_subagent, Some(false));

        // Case C — explicit true: both true.
        let ctx = context_from(Some("true"), true, true);
        assert!(ctx.is_subagent);
        assert_eq!(ctx.declared_is_subagent, Some(true));

        // Case D — native harness child, NO explicit Switchyard override:
        // normalized is_subagent = true (derived), declared = None (no header),
        // and it IS delegated work.
        let ctx = context_from(None, true, true);
        assert!(ctx.is_subagent);
        assert_eq!(ctx.declared_is_subagent, None);
        assert!(ctx.is_delegated_work);

        // Case D-2 — native harness child lineage fact WITHOUT delegated-work
        // (e.g. a harness kind that is a sub-agent lineage fact but not routed
        // as work): is_subagent and is_delegated_work can differ.
        let ctx = context_from(None, true, false);
        assert!(ctx.is_subagent);
        assert_eq!(ctx.declared_is_subagent, None);
        assert!(!ctx.is_delegated_work);

        // Case E — malformed explicit override: declared = None, but the
        // normalized upstream classification is independent of the malformed
        // header (still reflects the underlying derived value).
        let ctx = context_from(Some("banana"), false, false);
        assert_eq!(ctx.declared_is_subagent, None);
        assert!(!ctx.is_subagent);

        // Declared identity still captured.
        assert_eq!(
            context_from(None, false, false).agent_id.as_deref(),
            Some("openclaw-remote")
        );

        // Arbitrary extra_metadata and the removed LocalClaw envelope fields are
        // never serialized into the durable record.
        let context = context_from(Some("false"), false, false);
        let record = RoutingRecord {
            requested_route: context.requested_route.map(Cow::Owned),
            agent_id: context.agent_id.map(Cow::Owned),
            task_id: context.task_id.map(Cow::Owned),
            task_kind: context.task_kind.map(Cow::Owned),
            turn_id: context.turn_id.map(Cow::Owned),
            correlation_id: context.correlation_id.map(Cow::Owned),
            agent_kind: context.agent_kind.map(Cow::Owned),
            agent_role: context.agent_role.map(Cow::Owned),
            is_subagent: context.is_subagent,
            declared_is_subagent: context.declared_is_subagent,
            is_delegated_work: context.is_delegated_work,
            ..Default::default()
        };
        let serialized = serde_json::to_string(&record).expect("serialize");
        assert!(!serialized.contains("sensitive_secret"));
        assert!(!serialized.contains("user_content"));
        assert!(!serialized.contains("must-not-leak"));
        for removed in [
            "routing_principal",
            "policy_domain",
            "work_shape",
            "reasoning_intent",
            "tool_required",
        ] {
            assert!(
                !serialized.contains(removed),
                "removed LocalClaw envelope field {removed} must not be serialized"
            );
        }
    }
}
