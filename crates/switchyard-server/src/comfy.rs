// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw integration fork.
// SPDX-License-Identifier: Apache-2.0

//! Server-side ComfyNinja runtime composition (Layer-8C Gate C2-A, inert).
//!
//! This module is the **smallest native composition seam** that connects the
//! Gate C1 ComfyNinja intake capability (in `switchyard-libsy::algorithms::comfy`)
//! to Switchyard's runtime/configuration architecture.
//!
//! # Inert-only (Gate C2-A)
//!
//! Gate C2-A composes the capability into the runtime architecture in a
//! **completely inert state**:
//! - a configured-but-`enabled=false` ComfyNinja integration performs **no**
//!   network request and instantiates **no** consumer task;
//! - an `enabled=true` integration constructs the Gate C1 consumers (snapshot
//!   cache + transition ledger) **lazily / without any live fetch at
//!   construction**, and requires a resolvable credential reference;
//! - an `enabled=true` integration **without** a resolvable credential fails
//!   closed and explicitly (never implies GPU availability).
//!
//! No `:8447` credential is created, copied, requested, read, or placed in this
//! gate. No live ComfyNinja consumption occurs. ComfyNinja remains an orthogonal
//! hardware-resource domain and never participates in monetary cost-pool
//! semantics or routing decisions.

use std::time::Duration;

use libsy::{
    ComfyProfileClass, ComfyResourceState, ComfySnapshotCache, ComfySourceHealth,
    ComfyTransitionLedger, FeedDisposition, TransitionCursor, TransitionFeed,
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
}
