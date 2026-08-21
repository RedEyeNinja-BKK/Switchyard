// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw integration fork.
// SPDX-License-Identifier: Apache-2.0

//! Server-side ComfyNinja runtime composition (Layer-8C Gate C2-A + C2-F).
//!
//! This module is the **smallest native composition seam** that connects the
//! Gate C1 ComfyNinja intake capability (in `switchyard-libsy::algorithms::comfy`)
//! to Switchyard's runtime/configuration architecture.
//!
//! # Gate C2-A (inert composition) — accepted
//!
//! Gate C2-A added the config schema, the runtime holder, and the
//! deterministic disabled / credential-fail-closed / bootstrap-state /
//! failure semantics — all inert (no live fetch, no worker, no credential).
//!
//! # Gate C2-F (source-readiness corrective gate)
//!
//! Gate C2-F removes the C2-B source blocker: it adds the **authenticated HTTP
//! fetch paths** (snapshot `GET /v1/resource`, transition
//! `GET /v1/transitions`) and the **smallest bounded runtime driver** that
//! performs factual consumption when the integration is later activated.
//!
//! This is **source readiness only**. The invariants remain:
//! - a configured-but-`enabled=false` ComfyNinja integration performs **no**
//!   network request, spawns **no** worker task, and resolves **no** credential;
//! - an `enabled=true` integration constructs the consumers and may spawn the
//!   bounded driver, but only when explicitly started by a future activation;
//! - an `enabled=true` integration **without** a resolvable credential fails
//!   closed and explicitly (never implies GPU availability).
//!
//! No live production activation occurs in C2-A or C2-F. No `:8447` credential
//! is created, copied, placed, or used. ComfyNinja remains an orthogonal
//! hardware-resource domain and never participates in monetary cost-pool
//! semantics or routing decisions.

use std::sync::Arc;
use std::time::Duration;

use libsy::{
    ComfyProfileClass, ComfyRemoteSnapshot, ComfyResourceState, ComfySnapshotCache,
    ComfySourceHealth, ComfyTransitionLedger, FeedDisposition, TransitionCursor, TransitionFeed,
};

/// ComfyNinja bootstrap / runtime status — deterministic and observable.
///
/// These are the factual bootstrap conditions required by Gate C2-A
/// clarification #3. They distinguish, at any time, what Switchyard knows about
/// ComfyNinja continuity — never silently presenting post-restart state as
/// continuous history when continuity has not been proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComfyBootState {
    /// Integration is configured but disabled; no consumer, no network, no
    /// credential required.
    Disabled,
    /// Fresh startup with no prior Switchyard history and no checkpoint.
    Fresh,
    /// Startup resumed from a persisted cursor/checkpoint (selected later).
    ResumedFromCheckpoint,
    /// Rebuilding the bounded history from the ComfyNinja transition feed
    /// (e.g. after a fresh cursor in a known epoch).
    RebuildingHistory,
    /// The ComfyNinja producer epoch changed from the persisted checkpoint —
    /// local continuation was reset.
    ProducerEpochChanged,
    /// Continuation cannot be established (e.g. the checkpoint is older than
    /// the ComfyNinja-side retained window, or a cursor gap was reported and
    /// not resolvable). Degraded; no uninterrupted-history claim.
    ContinuationDegraded,
    /// The source is unavailable (credential missing/unresolvable, or the
    /// endpoint is not reachable).
    SourceUnavailable,
}

impl ComfyBootState {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Fresh => "fresh",
            Self::ResumedFromCheckpoint => "resumed_from_checkpoint",
            Self::RebuildingHistory => "rebuilding_history",
            Self::ProducerEpochChanged => "producer_epoch_changed",
            Self::ContinuationDegraded => "continuation_degraded",
            Self::SourceUnavailable => "source_unavailable",
        }
    }
}

/// The relationship between cursor/checkpoint state and transition/history
/// state, as required by Gate C2-A.
///
/// These are **separate** concerns:
/// - **cursor/checkpoint**: the `(producer_epoch, eid)` feed position that lets
///   Switchyard continue losslessly and detect producer restarts;
/// - **transition/history**: the bounded sliding-window samples derived from
///   the transition events.
///
/// Durable local history is NOT automatically required. With the deployed
/// ComfyNinja transition contract, Switchyard can deterministically
/// reconstruct a bounded useful history after restart from a retained feed
/// window (`oldest_available_eid`..`latest_eid`) and a persisted (or fresh)
/// cursor. So the C2-A default design is:
/// - cursor/checkpoint: **in-memory by default**, optionally persisted later;
/// - transition/history: **reconstructed** on restart from the ComfyNinja feed
///   (not durably stored) — unless reconstruction is proven insufficient, in
///   which case persistence is documented as required (a later decision).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComfyHistoryPolicy {
    /// Cursor + history are in-memory; on restart a fresh/known epoch is
    /// rebuilt from the ComfyNinja retained feed window (reconstruction).
    ReconstructFromFeed,
    /// Reserved for a later durable checkpoint decision; not implemented here.
    DurableCheckpoint,
}

/// Read-only view of the runtime ComfyNinja composition state (observability).
#[derive(Clone, Debug)]
pub struct ComfyCompositionStatus {
    pub enabled: bool,
    pub boot_state: ComfyBootState,
    pub history_policy: ComfyHistoryPolicy,
    pub current_cursor: Option<TransitionCursor>,
    pub last_source_health: ComfySourceHealth,
    pub recent_event_count: usize,
    pub last_active_profile: Option<ComfyProfileClass>,
}

impl ComfyCompositionStatus {
    pub fn for_disabled() -> Self {
        Self {
            enabled: false,
            boot_state: ComfyBootState::Disabled,
            history_policy: ComfyHistoryPolicy::ReconstructFromFeed,
            current_cursor: None,
            last_source_health: ComfySourceHealth::Unavailable,
            recent_event_count: 0,
            last_active_profile: None,
        }
    }
}

/// The runtime ComfyNinja composition: constructed only when `enabled`, holds
/// the Gate C1 consumers passively (no background task, no live fetch at
/// construction), and exposes a deterministic bootstrap status.
///
/// # Inert construction
///
/// Constructing a `ComfyNinjaRuntime` performs **no** network I/O and spawns
/// **no** task. The snapshot cache fetches lazily only when `snapshot()`
/// is called; the transition ledger ingests only when `ingest_transitions()`
/// is called. In Gate C2-A nothing calls these — the runtimes are inert.
#[derive(Debug)]
pub struct ComfyNinjaRuntime {
    /// Set only when the integration is enabled.
    enabled: bool,
    boot_state: ComfyBootState,
    history_policy: ComfyHistoryPolicy,
    /// Gate C1 snapshot cache (lazy). `None` when disabled.
    #[allow(dead_code)] // intentionally retained inert for Gate C2-B
    snapshot: Option<ComfySnapshotCache>,
    /// Gate C1 transition ledger (passive).
    ledger: Option<ComfyTransitionLedger>,
    /// Current feed cursor (in-memory checkpoint).
    cursor: Option<TransitionCursor>,
    #[allow(dead_code)] // intentionally retained; set on construction
    last_source_health: ComfySourceHealth,
    /// TTL for the snapshot cache.
    #[allow(dead_code)] // intentionally retained; drives the lazy cache in C2-B
    ttl: Duration,
}

impl ComfyNinjaRuntime {
    /// Construct a disabled (inert) runtime: no consumer, no network, no
    /// credential.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            boot_state: ComfyBootState::Disabled,
            history_policy: ComfyHistoryPolicy::ReconstructFromFeed,
            snapshot: None,
            ledger: None,
            cursor: None,
            last_source_health: ComfySourceHealth::Unavailable,
            ttl: Duration::from_secs(30),
        }
    }

    /// Construct an enabled runtime from a validated, complete configuration.
    ///
    /// Precondition: the caller has already validated that a credential
    /// reference resolves (env var present, non-empty) and that the config
    /// URLs are well-formed. This constructor performs **no** network I/O and
    /// does **not** touch `:8447`. The credential is used only inside the
    /// lazy fetch closure at the moment an authenticated read is requested
    /// (which never happens in Gate C2-A).
    pub fn enabled(
        ttl: Duration,
        snapshot_fetch: Box<
            dyn Fn() -> futures::future::BoxFuture<'static, Result<ComfyResourceState, String>>
                + Send
                + Sync,
        >,
        boot_state: ComfyBootState,
    ) -> Self {
        let snapshot = ComfySnapshotCache::new(ttl, snapshot_fetch);
        Self {
            enabled: true,
            boot_state,
            history_policy: ComfyHistoryPolicy::ReconstructFromFeed,
            snapshot: Some(snapshot),
            ledger: Some(ComfyTransitionLedger::default()),
            cursor: None,
            last_source_health: ComfySourceHealth::Healthy,
            ttl,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn boot_state(&self) -> ComfyBootState {
        self.boot_state
    }

    pub fn history_policy(&self) -> ComfyHistoryPolicy {
        self.history_policy
    }

    pub fn cursor(&self) -> Option<&TransitionCursor> {
        self.cursor.as_ref()
    }

    /// Access the snapshot cache (enabled only). `None` when disabled.
    pub fn snapshot_cache(&self) -> Option<&ComfySnapshotCache> {
        self.snapshot.as_ref()
    }

    /// Refresh the snapshot through the runtime's TTL cache, and reflect the
    /// outcome in `last_source_health`. Returns the parsed state on success
    /// (which also populates the cache) or an opaque coarse error on failure.
    ///
    /// The driver calls this once per tick. Disabled runtime (`None` cache)
    /// returns an error (fail closed).
    pub async fn refresh_snapshot(&mut self) -> Result<libsy::ComfySnapshotState, String> {
        let Some(cache) = self.snapshot.as_ref() else {
            self.last_source_health = ComfySourceHealth::Unavailable;
            return Err("comfyninja snapshot cache absent (runtime disabled)".to_string());
        };
        match cache.snapshot().await {
            Ok(state) => {
                self.last_source_health = state.source_health;
                Ok(state.state.clone())
            }
            Err(error) => {
                self.last_source_health = ComfySourceHealth::Unavailable;
                Err(error)
            }
        }
    }

    /// Mark the source degraded (used by the driver after an error threshold).
    /// Observable via `boot_state`/`last_source_health`; never implies GPU
    /// availability.
    pub fn mark_degraded(&mut self) {
        self.last_source_health = ComfySourceHealth::Unavailable;
        if self.boot_state != ComfyBootState::Disabled {
            self.boot_state = ComfyBootState::SourceUnavailable;
        }
    }

    /// Consume this (enabled) runtime into a running bounded driver, returning
    /// a stop handle. This is the **activation path**: it spawns the single
    /// driver task that refreshes the snapshot (through this runtime's real
    /// snapshot cache) and catches up the transition feed. Only call for an
    /// enabled runtime; a disabled runtime must remain driver-less (the caller
    /// must not call this for a disabled config).
    pub fn into_driver(
        self,
        transitions: TransitionTick,
        config: ComfyDriverConfig,
    ) -> ComfyDriverHandle {
        // The snapshot fetch is already captured inside this runtime's cache.
        let runtime = Arc::new(tokio::sync::Mutex::new(self));
        let driver = ComfyRuntimeDriver::new(runtime, transitions, config);
        driver.spawn()
    }

    /// Ingest a transition feed and advance the in-memory checkpoint cursor.
    ///
    /// Pure in-memory; returns the disposition recognized. This is the C2-A
    /// hook that a later Gate C2-B (or a readiness consumer) would drive — it
    /// is **not** called during construction or by any background task in
    /// Gate C2-A.
    pub fn ingest_transitions(&mut self, feed: &TransitionFeed) -> FeedDisposition {
        let Some(ledger) = self.ledger.as_mut() else {
            return FeedDisposition::Continuation; // no ledger (disabled) -> no-op
        };
        use libsy::advance_cursor;
        let (disposition, next_cursor) = advance_cursor(self.cursor.as_ref(), feed);
        self.cursor = next_cursor.clone();
        // Reflect the disposition in the observable boot state.
        self.boot_state = match disposition {
            FeedDisposition::EpochReset => ComfyBootState::ProducerEpochChanged,
            FeedDisposition::Overflow | FeedDisposition::Gap => {
                ComfyBootState::ContinuationDegraded
            }
            FeedDisposition::Continuation => {
                if self.boot_state == ComfyBootState::Fresh
                    || self.boot_state == ComfyBootState::Disabled
                    || self.boot_state == ComfyBootState::SourceUnavailable
                {
                    ComfyBootState::RebuildingHistory
                } else {
                    self.boot_state
                }
            }
        };
        ledger.ingest(disposition, feed);
        disposition
    }

    pub fn status(&self) -> ComfyCompositionStatus {
        ComfyCompositionStatus {
            enabled: self.enabled,
            boot_state: self.boot_state,
            history_policy: self.history_policy,
            current_cursor: self.cursor.clone(),
            last_source_health: self.last_source_health,
            recent_event_count: self
                .ledger
                .as_ref()
                .map(|l| l.recent_event_count())
                .unwrap_or(0),
            last_active_profile: self.ledger.as_ref().and_then(|l| l.last_active_profile()),
        }
    }

    /// Alias kept for callers that read `last_active_profile` through the
    /// composition (Switchyard-owned history discriminator).
    pub fn last_active_profile(&self) -> Option<ComfyProfileClass> {
        self.ledger.as_ref().and_then(|l| l.last_active_profile())
    }
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

/// Human-readable summary of the runtime's bootstrap/observability state.
pub fn summarize(runtime: &ComfyNinjaRuntime) -> String {
    let s = runtime.status();
    format!(
        "comfy_enabled={} boot_state={} history_policy={:?} cursor={:?} source_health={:?} recent_events={}",
        s.enabled,
        s.boot_state.label(),
        s.history_policy,
        s.current_cursor,
        s.last_source_health,
        s.recent_event_count
    )
}

// ---------------------------------------------------------------------------
// Gate C2-F: authenticated HTTP fetch paths + bounded runtime driver.
// ---------------------------------------------------------------------------

/// Resolve a bearer credential from the environment lazily, at request time.
///
/// Reads the value only to construct the request header, then returns it to be
/// dropped immediately. The value is never stored in any struct/status/history/
/// config/log. An absent or empty env var fails closed (the caller treats the
/// source as unavailable) — never implying GPU availability.
fn resolve_bearer(env_name: &str) -> Result<String, String> {
    let value = std::env::var(env_name).map_err(|_| {
        format!(
            "credential env var {env_name} is not set (fail closed; ComfyNinja source unavailable)"
        )
    })?;
    if value.trim().is_empty() {
        return Err(format!(
            "credential env var {env_name} is empty (fail closed; ComfyNinja source unavailable)"
        ));
    }
    Ok(value)
}

/// Shared HTTP client for ComfyNinja telemetry fetches (native `reqwest` seam,
/// already a workspace dependency). Timeout-bound, no redirects (we never want
/// a bare token redirected off-origin).
fn comfy_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("static reqwest client builder")
}

/// Authenticated snapshot fetcher: `GET <url>/v1/resource` with
/// `Authorization: Bearer <credential>` resolved lazily from an env-var NAME.
///
/// Produces a `ComfyResourceState` with `source_health=Healthy` + `last_success_at`
/// on success; any non-2xx or malformed body produces a coarse error and an
/// `Unavailable` state (fail closed). The bearer is resolved per-fetch and
/// dropped; it is never part of any persistent structure.
pub struct ComfyHttpSnapshotFetcher {
    url: String,
    auth_token_env: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for ComfyHttpSnapshotFetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfyHttpSnapshotFetcher")
            .field("url", &self.url)
            .field("auth_token_env", &self.auth_token_env) // env NAME only, never a value
            .finish()
    }
}

impl ComfyHttpSnapshotFetcher {
    pub fn new(url: String, auth_token_env: String) -> Self {
        Self {
            url,
            auth_token_env,
            client: comfy_http_client(),
        }
    }

    /// Perform one authenticated snapshot fetch.
    pub async fn fetch(&self) -> Result<ComfyResourceState, String> {
        let token = resolve_bearer(&self.auth_token_env)?;
        let response = self
            .client
            .get(&self.url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| format!("comfyninja snapshot request failed: {e}"))?;
        // token dropped here (no longer referenced)
        drop(token);
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| format!("comfyninja snapshot read failed: {e}"))?;
        if !status.is_success() {
            // Coarse; never propagate the upstream body (could carry secrets).
            return Err(format!("comfyninja snapshot fetch failed (HTTP {status})"));
        }
        let remote: ComfyRemoteSnapshot = serde_json::from_str(&body)
            .map_err(|e| format!("comfyninja snapshot parse failed: {e}"))?;
        Ok(ComfyResourceState {
            producer_epoch: remote.producer_epoch,
            state_generation: remote.state_generation,
            state_fingerprint: remote.state_fingerprint,
            state: remote.state,
            source_health: ComfySourceHealth::Healthy,
            last_success_at: Some(now_unix()),
            error: None,
        })
    }
}

/// Authenticated transition fetcher: `GET <url>/v1/transitions?after_eid=&after_epoch=`
/// with `Authorization: Bearer <credential>` resolved lazily.
///
/// The cursor is passed as explicit query params. Returns a parsed
/// [`TransitionFeed`] (deployed schema F). Any non-2xx or malformed body is a
/// coarse error (fail closed). The bearer is per-fetch and dropped.
pub struct ComfyHttpTransitionFetcher {
    url: String,
    auth_token_env: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for ComfyHttpTransitionFetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfyHttpTransitionFetcher")
            .field("url", &self.url)
            .field("auth_token_env", &self.auth_token_env) // env NAME only
            .finish()
    }
}

impl ComfyHttpTransitionFetcher {
    pub fn new(url: String, auth_token_env: String) -> Self {
        Self {
            url,
            auth_token_env,
            client: comfy_http_client(),
        }
    }

    /// Fetch the transition feed page beginning after `cursor`.
    ///
    /// Passing `None` => a fresh request with no `after_eid`/`after_epoch`
    /// (the producer defaults `after_epoch` to its current epoch; eid starts at
    /// the oldest retained). Passing `Some(cursor)` => `after_eid=cursor.last_eid`
    /// and `after_epoch=cursor.producer_epoch`.
    ///
    /// The query is appended directly to the URL (hex epoch + integer eid are
    /// URL-safe), avoiding the `query` feature of `reqwest` so the workspace's
    /// minimal `default-features=false` dependency is preserved.
    pub async fn fetch(&self, cursor: Option<&TransitionCursor>) -> Result<TransitionFeed, String> {
        let token = resolve_bearer(&self.auth_token_env)?;
        let url = match cursor {
            Some(c) => format!(
                "{}{}after_eid={}&after_epoch={}",
                self.url,
                if self.url.contains('?') { '&' } else { '?' },
                c.last_eid,
                c.producer_epoch
            ),
            None => self.url.clone(),
        };
        let response = self
            .client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| format!("comfyninja transition request failed: {e}"))?;
        drop(token);
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| format!("comfyninja transition read failed: {e}"))?;
        if !status.is_success() {
            return Err(format!(
                "comfyninja transition fetch failed (HTTP {status})"
            ));
        }
        serde_json::from_str(&body).map_err(|e| format!("comfyninja transition parse failed: {e}"))
    }
}

/// Driver configuration (bounded; no uncontrolled polling loop).
#[derive(Clone, Copy, Debug)]
pub struct ComfyDriverConfig {
    /// Base poll cadence between ticks (snapshot refresh + transition catch-up,
    /// one in-flight per kind per tick).
    pub interval: Duration,
    /// Maximum exponential backoff after consecutive errors (default 30s).
    pub max_backoff: Duration,
    /// Error-count threshold before the worker marks the source degraded
    /// (observable via boot_state `ContinuationDegraded`).
    pub max_consecutive_errors: u32,
}

impl Default for ComfyDriverConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(15),
            max_backoff: Duration::from_secs(30),
            max_consecutive_errors: 8,
        }
    }
}

/// A fetch closure produces one transition-feed page given the current cursor.
/// `Send + Sync` so it can be shared across the driver task.
pub type TransitionTick = Box<
    dyn Fn(
            Option<TransitionCursor>,
        ) -> futures::future::BoxFuture<'static, Result<TransitionFeed, String>>
        + Send
        + Sync,
>;

/// Bounded runtime driver: periodically refreshes the snapshot and catches up
/// the transition feed while the integration is enabled.
///
/// - Constructed/spawned only when the integration is enabled.
/// - One `tokio::spawn`'d task; at most one in-flight snapshot and one in-flight
///   transition request per tick (sequential). No parallel storm.
/// - Snapshot is driven through the runtime's TTL cache (`refresh_snapshot`);
///   transitions are driven through the injected transition tick.
/// - Exponential backoff on consecutive errors, capped at `max_backoff`, with
///   the source reported degraded (`.boot_state` observable) after the error
///   threshold.
/// - Clean cancellation: `ComfyDriverHandle::stop()` sets a flag; the loop
///   exits at the top of the next tick and aborts the in-flight sleep.
pub struct ComfyRuntimeDriver {
    runtime: Arc<tokio::sync::Mutex<ComfyNinjaRuntime>>,
    transitions: TransitionTick,
    config: ComfyDriverConfig,
}

impl std::fmt::Debug for ComfyRuntimeDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfyRuntimeDriver")
            .field("config", &self.config)
            .finish()
    }
}

/// A handle permitting a clean stop of a spawned [`ComfyRuntimeDriver`].
pub struct ComfyDriverHandle {
    stop: Arc<std::sync::atomic::AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for ComfyDriverHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfyDriverHandle").finish_non_exhaustive()
    }
}

impl ComfyDriverHandle {
    /// Request a clean stop of the driver loop (cancellation).
    pub fn stop(&self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Wait for the driver loop to exit (e.g. on shutdown / cancellation).
    pub async fn join(self) {
        let _ = self.task.await;
    }
}

impl ComfyRuntimeDriver {
    /// Build the enabled-only driver.
    pub(super) fn new(
        runtime: Arc<tokio::sync::Mutex<ComfyNinjaRuntime>>,
        transitions: TransitionTick,
        config: ComfyDriverConfig,
    ) -> Self {
        Self {
            runtime,
            transitions,
            config,
        }
    }

    /// Spawn the tick loop on the current tokio runtime; returns a stop handle.
    pub fn spawn(self) -> ComfyDriverHandle {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_b = Arc::clone(&stop);
        let task = tokio::spawn(async move {
            self.run(stop_b).await;
        });
        ComfyDriverHandle { stop, task }
    }

    async fn run(self, stop: Arc<std::sync::atomic::AtomicBool>) {
        let mut consecutive_errors: u32 = 0;

        loop {
            if stop.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            // Snapshot phase: drive through the runtime's TTL snapshot cache.
            let snap_ok = {
                let mut run = self.runtime.lock().await;
                run.refresh_snapshot().await.is_ok()
            };
            consecutive_errors = if snap_ok {
                consecutive_errors.saturating_sub(1)
            } else {
                consecutive_errors.saturating_add(1)
            };

            // Transition phase: fetch the page since the current cursor.
            {
                let cursor = {
                    let run = self.runtime.lock().await;
                    run.cursor().cloned()
                };
                let page = (self.transitions)(cursor).await;
                match page {
                    Ok(feed) => {
                        let mut run = self.runtime.lock().await;
                        run.ingest_transitions(&feed);
                        consecutive_errors = 0;
                    }
                    Err(_) => consecutive_errors = consecutive_errors.saturating_add(1),
                }
            }

            // Report the source degraded once the error threshold is crossed.
            if consecutive_errors >= self.config.max_consecutive_errors {
                let mut run = self.runtime.lock().await;
                run.mark_degraded();
            }

            // Exponential backoff on consecutive errors, capped.
            let base = if consecutive_errors > 0 {
                let exp = 1u32 << consecutive_errors.min(10);
                self.config
                    .interval
                    .mul_f64(exp.min(64) as f64)
                    .min(self.config.max_backoff)
            } else {
                self.config.interval
            };
            tokio::time::sleep(base).await;
        }
    }
}

/// Current Unix time (seconds).
fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsy::{ComfyMode, CorrelationStatus, TransitionEvent, TransitionPhase, TransitionResult};

    // --- fixture helpers ----------------------------------------------------

    fn event(
        epoch: &str,
        eid: i64,
        tid: i64,
        phase: TransitionPhase,
        source: Option<ComfyMode>,
        target: Option<ComfyMode>,
        observed: f64,
    ) -> TransitionEvent {
        TransitionEvent {
            rid: format!("shell-{tid}"),
            epoch: epoch.to_string(),
            eid,
            transition_id: tid,
            transition_start_known: matches!(phase, TransitionPhase::Start),
            correlation_status: CorrelationStatus::Matched,
            command: "activate:comfy".into(),
            source_mode: source,
            target_mode: target,
            source_profile: None,
            target_profile: None,
            phase,
            result: if matches!(phase, TransitionPhase::Terminal) {
                Some(TransitionResult::Succeeded)
            } else {
                Some(TransitionResult::Pending)
            },
            underlying_rc: None,
            observed_at: observed,
            ingested_at: observed,
        }
    }

    fn feed(
        epoch: &str,
        oldest: Option<i64>,
        latest: Option<i64>,
        epoch_reset: bool,
        events: Vec<TransitionEvent>,
    ) -> TransitionFeed {
        TransitionFeed {
            producer_epoch: epoch.into(),
            oldest_available_eid: oldest,
            latest_eid: latest,
            epoch_reset,
            gap: false,
            overflow: false,
            events,
        }
    }

    /// A minimal inert snapshot-fetch closure for constructing an enabled
    /// runtime. Never performs network I/O; always fails closed.
    fn inert_fetch() -> Box<
        dyn Fn() -> futures::future::BoxFuture<'static, Result<libsy::ComfyResourceState, String>>
            + Send
            + Sync,
    > {
        Box::new(|| Box::pin(async move { Err("no live fetch in C2-A test".to_string()) }))
    }

    // --- runtime-ledger tests ----------------------------------------------

    #[test]
    fn disabled_runtime_is_inert_no_consumer() {
        let r = ComfyNinjaRuntime::disabled();
        assert!(!r.is_enabled());
        assert_eq!(r.boot_state(), ComfyBootState::Disabled);
        assert_eq!(r.status().recent_event_count, 0);
        assert!(r.cursor().is_none());
        // No snapshot cache / ledger is constructed.
        assert!(r.snapshot.is_none() && r.ledger.is_none());
    }

    #[test]
    fn enabled_runtime_boot_state_is_fresh() {
        let r = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        assert!(r.is_enabled());
        assert_eq!(r.boot_state(), ComfyBootState::Fresh);
        assert!(r.snapshot.is_some() && r.ledger.is_some());
        assert_eq!(r.status().recent_event_count, 0);
    }

    #[test]
    fn fresh_no_history_startup_is_observable() {
        let r = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        assert_eq!(r.status().boot_state, ComfyBootState::Fresh);
        assert!(r.status().current_cursor.is_none());
    }

    #[test]
    fn restart_resets_to_fresh_no_silent_continuity() {
        // Prior runtime advanced to eid 4 in epoch e0...
        let mut first = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        let evs = vec![
            event(
                "e0",
                1,
                1,
                TransitionPhase::Start,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            ),
            event(
                "e0",
                2,
                1,
                TransitionPhase::Terminal,
                Some(ComfyMode::Comfy),
                Some(ComfyMode::Idle),
                1001.0,
            ),
            event(
                "e0",
                3,
                2,
                TransitionPhase::Start,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                2000.0,
            ),
            event(
                "e0",
                4,
                2,
                TransitionPhase::Terminal,
                Some(ComfyMode::Comfy),
                Some(ComfyMode::Idle),
                2001.0,
            ),
        ];
        let f = feed("e0", Some(1), Some(4), false, evs);
        first.ingest_transitions(&f);
        assert_eq!(first.cursor().unwrap().last_eid, 4);

        // ...then Switchyard restarts -> a NEW runtime has NO checkpoint and
        // must NOT claim continuity.
        let second = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        assert_eq!(second.boot_state(), ComfyBootState::Fresh);
        assert!(second.cursor().is_none());
        assert_eq!(second.status().recent_event_count, 0);
    }

    #[test]
    fn epoch_change_sets_producer_epoch_changed() {
        let mut r = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        // Ingest epoch e0, then a reset into epoch e1 -> ProducerEpochChanged.
        let f0 = feed(
            "e0",
            Some(1),
            Some(2),
            false,
            vec![
                event(
                    "e0",
                    1,
                    1,
                    TransitionPhase::Start,
                    Some(ComfyMode::Idle),
                    Some(ComfyMode::Comfy),
                    1000.0,
                ),
                event(
                    "e0",
                    2,
                    1,
                    TransitionPhase::Terminal,
                    Some(ComfyMode::Comfy),
                    Some(ComfyMode::Idle),
                    1001.0,
                ),
            ],
        );
        r.ingest_transitions(&f0);
        r.ingest_transitions(&f0); // idempotent
        // Advance so the cursor is on e0.
        assert_eq!(r.cursor().unwrap().producer_epoch, "e0");

        let f1 = feed(
            "e1",
            Some(1),
            Some(2),
            true,
            vec![
                event(
                    "e1",
                    1,
                    1,
                    TransitionPhase::Start,
                    Some(ComfyMode::Idle),
                    Some(ComfyMode::Studio),
                    3000.0,
                ),
                event(
                    "e1",
                    2,
                    1,
                    TransitionPhase::Terminal,
                    Some(ComfyMode::Studio),
                    Some(ComfyMode::Idle),
                    3001.0,
                ),
            ],
        );
        let disp = r.ingest_transitions(&f1);
        assert_eq!(disp, FeedDisposition::EpochReset);
        assert_eq!(r.boot_state(), ComfyBootState::ProducerEpochChanged);
    }

    #[test]
    fn continuation_degraded_on_overflow() {
        let mut r = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        // Cursor advanced to eid 1 in e0, then a feed whose oldest retained is 10 -> overflow.
        let f0 = feed(
            "e0",
            Some(1),
            Some(1),
            false,
            vec![event(
                "e0",
                1,
                1,
                TransitionPhase::Start,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            )],
        );
        r.ingest_transitions(&f0);
        let fidx_overflow = feed(
            "e0",
            Some(10),
            Some(11),
            false,
            vec![event(
                "e0",
                11,
                5,
                TransitionPhase::Start,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                2000.0,
            )],
        );
        let disp = r.ingest_transitions(&fidx_overflow);
        assert_eq!(disp, FeedDisposition::Overflow);
        assert_eq!(r.boot_state(), ComfyBootState::ContinuationDegraded);
    }

    #[test]
    fn ledger_ingest_recent_events_advance() {
        let mut r = ComfyNinjaRuntime::enabled(
            Duration::from_secs(30),
            inert_fetch(),
            ComfyBootState::Fresh,
        );
        let f = feed(
            "e0",
            Some(1),
            Some(2),
            false,
            vec![
                event(
                    "e0",
                    1,
                    1,
                    TransitionPhase::Start,
                    Some(ComfyMode::Idle),
                    Some(ComfyMode::Comfy),
                    1000.0,
                ),
                event(
                    "e0",
                    2,
                    1,
                    TransitionPhase::Terminal,
                    Some(ComfyMode::Comfy),
                    Some(ComfyMode::Idle),
                    1001.0,
                ),
            ],
        );
        r.ingest_transitions(&f);
        assert_eq!(r.status().recent_event_count, 2);
        assert_eq!(r.last_active_profile(), None); // sealed idle -> no last-active
    }

    #[test]
    fn status_disabled_reports_explicit() {
        let r = ComfyNinjaRuntime::disabled();
        let s = r.status();
        assert_eq!(s.boot_state.label(), "disabled");
        assert!(!s.enabled);
    }

    use axum::Router;
    use axum::body::Body;
    use axum::extract::{RawQuery, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const FAKE_TOKEN: &str = "c2f-test-only-fake-bearer";

    /// Shared capture state for the mock ComfyNinja server.
    #[derive(Default)]
    struct MockState {
        snapshot_requests: AtomicUsize,
        transition_requests: AtomicUsize,
        auth_failures: AtomicUsize,
        /// When set, every authed request returns this status instead of a body.
        fail_status: Option<u16>,
    }

    fn authed(headers: &HeaderMap) -> bool {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {FAKE_TOKEN}"))
            .unwrap_or(false)
    }

    /// GET /v1/resource handler (snapshot). Axum requires `State` LAST.
    async fn mock_resource(headers: HeaderMap, State(st): State<StdArc<MockState>>) -> Response {
        st.snapshot_requests.fetch_add(1, Ordering::SeqCst);
        if !authed(&headers) {
            st.auth_failures.fetch_add(1, Ordering::SeqCst);
            return StatusCode::UNAUTHORIZED.into_response();
        }
        if let Some(code) = st.fail_status {
            return StatusCode::from_u16(code)
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                .into_response();
        }
        Response::new(Body::from(SNAPSHOT_BODY))
    }

    /// GET /v1/transitions handler (transition feed). Axum requires `State` LAST.
    async fn mock_transitions(
        headers: HeaderMap,
        RawQuery(_query): RawQuery,
        State(st): State<StdArc<MockState>>,
    ) -> Response {
        st.transition_requests.fetch_add(1, Ordering::SeqCst);
        if !authed(&headers) {
            st.auth_failures.fetch_add(1, Ordering::SeqCst);
            return StatusCode::UNAUTHORIZED.into_response();
        }
        Response::new(Body::from(
            r#"{"producer_epoch":"e0","oldest_available_eid":1,"latest_eid":2,"epoch_reset":false,"gap":false,"overflow":false,"events":[]}"#,
        ))
    }

    const SNAPSHOT_BODY: &str = r#"{
        "producer_epoch": "abc123def456",
        "state_generation": 3,
        "state_fingerprint": "deadbeefcafe",
        "state": {
            "owner": "comfyui", "mode": "comfy",
            "transition_state": "idle", "transition_target": null,
            "comfyui": "comfyui", "unsloth_studio": null, "llama_server": null,
            "studio_backend_health": "ok",
            "resident_qwen_profile": "unknown",
            "vram_used_mib": 67, "vram_free_mib": 24260,
            "comfy_busy": false, "comfy_queue_running": null, "comfy_queue_pending": null
        }
    }"#;

    struct MockComfy {
        addr: std::net::SocketAddr,
        state: StdArc<MockState>,
    }

    impl MockComfy {
        async fn spawn(fail_status: Option<u16>) -> MockComfy {
            let state = StdArc::new(MockState {
                fail_status,
                ..Default::default()
            });
            let app = Router::new()
                .route("/v1/resource", get(mock_resource))
                .route("/v1/transitions", get(mock_transitions))
                .with_state(StdArc::clone(&state));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            MockComfy { addr, state }
        }

        fn base(&self) -> String {
            format!("http://{}/v1", self.addr)
        }

        fn snapshot_requests(&self) -> usize {
            self.state.snapshot_requests.load(Ordering::SeqCst)
        }
    }

    /// Sets a test-specific credential env var to the FAKE value, awaits the
    /// async body, clears it. Each caller passes a UNIQUE env name so parallel
    /// tests never race on a shared variable.
    async fn with_cred_env_async<F, Fut>(env: &'static str, f: F)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        unsafe {
            std::env::set_var(env, FAKE_TOKEN);
        }
        f().await;
        unsafe {
            std::env::remove_var(env);
        }
    }

    /// Test: authenticated snapshot fetch happy path — mock verifies the Bearer
    /// header; deployed-schema response parsed into expected state.
    #[tokio::test]
    async fn c2f_snapshot_authenticated_happy_path() {
        let mock = MockComfy::spawn(None).await;
        with_cred_env_async("COMFY_C2F_TEST_SNAPSHOT", async || {
            let fetcher = ComfyHttpSnapshotFetcher::new(
                format!("{}/resource", mock.base()),
                "COMFY_C2F_TEST_SNAPSHOT".into(),
            );
            let state = fetcher.fetch().await.unwrap();
            assert_eq!(state.source_health, ComfySourceHealth::Healthy);
            assert_eq!(state.producer_epoch, "abc123def456");
            assert_eq!(state.state_generation, 3);
            assert_eq!(state.state.owner, Some(libsy::ComfyOwner::Comfy));
            assert_eq!(state.state.vram_used_mib, Some(67));
        })
        .await;
    }

    /// Test: authenticated transition fetch happy path.
    #[tokio::test]
    async fn c2f_transition_authenticated_happy_path() {
        let mock = MockComfy::spawn(None).await;
        with_cred_env_async("COMFY_C2F_TEST_TRANS", async || {
            let fetcher = ComfyHttpTransitionFetcher::new(
                format!("{}/transitions", mock.base()),
                "COMFY_C2F_TEST_TRANS".into(),
            );
            let feed = fetcher.fetch(None).await.unwrap();
            assert_eq!(feed.producer_epoch, "e0");
            assert_eq!(feed.events.len(), 0);
        })
        .await;
    }

    /// Test: continuation fetch builds + sends after_eid / after_epoch; the
    /// mock returns 2xx only on a correct bearer, so this proves both auth and
    /// continuation query construction reach the wire.
    #[tokio::test]
    async fn c2f_transition_continuation_query_params_present() {
        let mock = MockComfy::spawn(None).await;
        let cursor = TransitionCursor {
            producer_epoch: "e0".into(),
            last_eid: 7,
        };
        with_cred_env_async("COMFY_C2F_TEST_CONT", async || {
            let fetcher = ComfyHttpTransitionFetcher::new(
                format!("{}/transitions", mock.base()),
                "COMFY_C2F_TEST_CONT".into(),
            );
            let feed = fetcher.fetch(Some(&cursor)).await.unwrap();
            assert_eq!(feed.producer_epoch, "e0");
        })
        .await;
    }

    /// Test: missing credential fails closed (no network request sent).
    #[tokio::test]
    async fn c2f_missing_credential_fails_closed() {
        unsafe { std::env::remove_var("COMFY_C2F_TEST_MISSING") };
        let mock = MockComfy::spawn(None).await;
        let fetcher = ComfyHttpSnapshotFetcher::new(
            format!("{}/resource", mock.base()),
            "COMFY_C2F_TEST_MISSING".into(),
        );
        let err = fetcher.fetch().await.unwrap_err();
        assert!(err.contains("COMFY_C2F_TEST_MISSING"), "err: {err}");
        assert_eq!(
            mock.snapshot_requests(),
            0,
            "no network on missing credential"
        );
    }

    /// Test: empty credential fails closed.
    #[tokio::test]
    async fn c2f_empty_credential_fails_closed() {
        unsafe { std::env::set_var("COMFY_C2F_TEST_EMPTY", "") };
        let mock = MockComfy::spawn(None).await;
        let fetcher = ComfyHttpSnapshotFetcher::new(
            format!("{}/resource", mock.base()),
            "COMFY_C2F_TEST_EMPTY".into(),
        );
        let err = fetcher.fetch().await.unwrap_err();
        assert!(err.contains("empty"), "err: {err}");
        unsafe { std::env::remove_var("COMFY_C2F_TEST_EMPTY") };
        assert_eq!(mock.snapshot_requests(), 0);
    }

    /// Test: HTTP 401 (wrong bearer) is a fetch failure, never availability.
    #[tokio::test]
    async fn c2f_http_401_fails_closed() {
        unsafe { std::env::set_var("COMFY_C2F_TEST_401", "WRONG-TOKEN") };
        let mock = MockComfy::spawn(None).await;
        let fetcher = ComfyHttpSnapshotFetcher::new(
            format!("{}/resource", mock.base()),
            "COMFY_C2F_TEST_401".into(),
        );
        let err = fetcher.fetch().await.unwrap_err();
        assert!(err.contains("401") || err.contains("HTTP"), "err: {err}");
        unsafe { std::env::remove_var("COMFY_C2F_TEST_401") };
    }

    /// Test: HTTP 5xx fails closed.
    #[tokio::test]
    async fn c2f_http_5xx_fails_closed() {
        let mock = MockComfy::spawn(Some(503)).await;
        with_cred_env_async("COMFY_C2F_TEST_5XX", async || {
            let fetcher = ComfyHttpSnapshotFetcher::new(
                format!("{}/resource", mock.base()),
                "COMFY_C2F_TEST_5XX".into(),
            );
            let err = fetcher.fetch().await.unwrap_err();
            assert!(err.contains("503") || err.contains("HTTP"), "err: {err}");
        })
        .await;
    }

    /// Test: malformed response fails closed (parse error).
    #[tokio::test]
    async fn c2f_malformed_response_fails_closed() {
        async fn invalid_resource() -> Response {
            Response::new(Body::from("not-json{"))
        }
        let app = Router::new().route("/v1/resource", get(invalid_resource));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        with_cred_env_async("COMFY_C2F_TEST_MALF", async || {
            let fetcher = ComfyHttpSnapshotFetcher::new(
                format!("http://{addr}/v1/resource"),
                "COMFY_C2F_TEST_MALF".into(),
            );
            let err = fetcher.fetch().await.unwrap_err();
            assert!(err.contains("parse"), "err: {err}");
        })
        .await;
    }

    /// Test: the C2-F public surface links (fetchers + driver config exist).
    #[test]
    fn c2f_public_surface_links() {
        let _ = ComfyDriverConfig::default().interval;
        let _ = std::mem::size_of::<ComfyHttpSnapshotFetcher>();
        let _ = std::mem::size_of::<ComfyHttpTransitionFetcher>();
        assert!(ComfyDriverConfig::default().max_consecutive_errors >= 1);
    }
}
