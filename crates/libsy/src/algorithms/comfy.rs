// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw integration fork.
// SPDX-License-Identifier: Apache-2.0

//! ComfyNinja hardware-resource intake (Layer-8C Gate C1).
//!
//! This module defines the **Switchyard-side consumption** of the ComfyNinja
//! resource-truth surfaces exposed on the protected `:8447` tailnet edge:
//!
//! - `GET /v1/resource` — authoritative current GPU/workload/lifecycle snapshot;
//! - `GET /v1/transitions?after_eid=&after_epoch=` — ordered factual transition
//!   events with explicit producer/epoch/cursor semantics.
//!
//! # Orthogonal-domain boundary (owner architecture)
//!
//! ComfyNinja is a **hardware-resource domain**, deliberately **orthogonal to**
//! the cost/credential pools (`ResourceSnapshot` with `openai`/`deepseek`) that
//! the existing [`super::resource`] module routes on. We therefore do NOT
//! coerce GPU/VRAM/ownership semantics into monetary-balance/cost semantics,
//! do NOT reuse `Pool`, and do NOT reuse the cost router's `Candidate` /
//! `ResourceRouter` eligibility machine. This module is self-contained and
//! cleanly separable for upstream reconciliation: it reuses the same *generic
//! techniques* (async-mutex TTL cache, coalesced refresh, fail-closed
//! freshness) but keeps ComfyNinja-native types.
//!
//! # Gate C1 scope
//!
//! Gate C1 implements the intake surfaces and the Switchyard-owned bounded
//! transition/history ledger AND their unit/fixture tests. It does **not**
//! wire any route/algorithm to consume this state, does **not** perform live
//! consumption, does **not** create/place the `:8447` credential, and does
//! **not** implement a readiness estimator. Those are separate, later gates.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex as AsyncMutex;

// ---------------------------------------------------------------------------
// Domain enums (deployed canonical vocabularies — see 8C.2 section E/F)
// ---------------------------------------------------------------------------

/// Canonical GPU owner vocabulary from the ComfyNinja snapshot.
///
/// Mirrors the deployed `state.owner` vocabulary. `idle` is NEVER an owner;
/// `None` (absent) means factually unowned; `Unknown` means ownership cannot
/// currently be established. `None` and `Unknown` are semantically DISTINCT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComfyOwner {
    #[serde(alias = "comfyui")]
    Comfy,
    Unsloth,
    Conflict,
    Unknown,
}

impl ComfyOwner {
    /// True only for a concrete owner (never `Unknown`). `None` composes at a
    /// higher level via `Option<ComfyOwner>`.
    pub fn is_owner(self) -> bool {
        matches!(self, Self::Comfy | Self::Unsloth | Self::Conflict)
    }
}

/// One canonical MODE vocabulary shared by the snapshot and the transition feed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyMode {
    Idle,
    Comfy,
    Studio,
    Conflict,
    Unknown,
}

/// Transition-state (current open-transition indicator) from the snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyTransitionState {
    Idle,
    InTransition,
}

/// Transition target vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyTransitionTarget {
    Idle,
    Comfy,
    Studio,
}

/// Resident Qwen profile. `FAST`/`LONG` are only set when factually provable
/// from the governed authenticated Studio status seam — never inferred from
/// VRAM, model name, or latency. `Unknown` means "cannot establish".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComfyResidentProfile {
    #[serde(rename = "FAST")]
    Fast,
    #[serde(rename = "LONG")]
    Long,
    #[serde(rename = "unknown")]
    Unknown,
}

/// A stable workload-profile class used as the transition-history key.
///
/// This is the FINITE, FIXED class set (8B.1 correction 1). Transition events
/// carry `source_profile`/`target_profile` as `null`/`unknown` at the
/// gpu-workload seam, so the ledger derives a conservative bounded class from
/// the observed mode rather than an unprovable profile name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ComfyProfileClass {
    /// The sealed-idle boundary (GPU at baseline, Studio healthy, no server).
    SealedIdle,
    Fast,
    Long,
    /// Z-Image Turbo text-to-image (C1).
    WkZimageT2i,
    /// Boogu image generation (C2).
    WkBooguGen,
    /// Boogu image editing (C3).
    WkBooguEdit,
    /// Conservative fallback for any unseen/unknown busy class.
    Generic,
}

impl std::fmt::Display for ComfyProfileClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::SealedIdle => "sealed_idle",
            Self::Fast => "FAST",
            Self::Long => "LONG",
            Self::WkZimageT2i => "WK_ZIMAGE_T2I",
            Self::WkBooguGen => "WK_BOOGU_GEN",
            Self::WkBooguEdit => "WK_BOOGU_EDIT",
            Self::Generic => "WK_GENERIC",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for ComfyProfileClass {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_uppercase().as_str() {
            "SEALED_IDLE" | "IDLE" | "NONE" => Ok(Self::SealedIdle),
            "FAST" => Ok(Self::Fast),
            "LONG" => Ok(Self::Long),
            "WK_ZIMAGE_T2I" | "C1" => Ok(Self::WkZimageT2i),
            "WK_BOOGU_GEN" | "C2" => Ok(Self::WkBooguGen),
            "WK_BOOGU_EDIT" | "C3" => Ok(Self::WkBooguEdit),
            _ => Ok(Self::Generic),
        }
    }
}

// ---------------------------------------------------------------------------
// Transition cursor / event / feed (deployed schema F)
// ---------------------------------------------------------------------------

/// The ordered per-event feed cursor: `(producer_epoch, eid)`.
///
/// Canonical transition identity is `(producer_epoch, transition_id)`, but the
/// **cursor for catching up on the feed is `(producer_epoch, eid)`** because
/// `eid` is the independent monotonic per-event sequence (deployed schemas
/// supersede the earlier `transition_id`-cursor sketch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionCursor {
    /// Opaque producer-lifetime identity (uuid4().hex of the deployed producer).
    pub producer_epoch: String,
    /// Highest contiguous `eid` we have ingested for that epoch.
    pub last_eid: i64,
}

impl TransitionCursor {
    pub fn fresh(producer_epoch: &str) -> Self {
        Self {
            producer_epoch: producer_epoch.to_string(),
            last_eid: 0,
        }
    }
}

/// Correlation outcome of a transition event (deployed schema F).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrelationStatus {
    Matched,
    OrphanTerminal,
    ReusedRid,
}

/// Event phase (deployed schema F).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionPhase {
    Start,
    Terminal,
}

/// Terminal result (deployed schema F).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionResult {
    Pending,
    Succeeded,
    Failed,
    Unknown,
}

/// One factual transition event (deployed schema F). Strictly factual — no
/// readiness projection, no `last_active_profile` (Switchyard-owned).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionEvent {
    /// Shell invocation id; correlation-only, never canonical.
    pub rid: String,
    /// Producer-epoch this event belongs to.
    pub epoch: String,
    /// Ordered per-event feed sequence (the cursor key).
    pub eid: i64,
    /// Canonical transition identity within the epoch.
    pub transition_id: i64,
    pub transition_start_known: bool,
    pub correlation_status: CorrelationStatus,
    pub command: String,
    pub source_mode: Option<ComfyMode>,
    pub target_mode: Option<ComfyMode>,
    /// `null`/`unknown` at the gpu-workload seam (never a concrete FAST/LONG).
    pub source_profile: Option<String>,
    pub target_profile: Option<String>,
    pub phase: TransitionPhase,
    pub result: Option<TransitionResult>,
    pub underlying_rc: Option<i64>,
    pub observed_at: f64,
    pub ingested_at: f64,
}

/// One page of the transition feed (deployed schema F response).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionFeed {
    pub producer_epoch: String,
    pub oldest_available_eid: Option<i64>,
    pub latest_eid: Option<i64>,
    pub epoch_reset: bool,
    pub gap: bool,
    pub overflow: bool,
    pub events: Vec<TransitionEvent>,
}

/// The three-way feed-ahead disposition for a cursor advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedDisposition {
    /// Contiguous continuation; ingest `events`.
    Continuation,
    /// Same epoch but a missing `eid` in range (cursor gap); do not fabricate.
    Gap,
    /// Producer restarted (epoch changed) — reset cursor and ingest the new
    /// epoch's current events.
    EpochReset,
    /// Cursor is below the retained leading edge — bounded-ledger overflow on
    /// the ComfyNinja side; re-sync from `oldest_available_eid`.
    Overflow,
}

// ---------------------------------------------------------------------------
// Snapshot domain types (deployed schema E)
// ---------------------------------------------------------------------------

/// Parsed `state` object of `GET /v1/resource` (deployed schema E).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComfySnapshotState {
    #[serde(default)]
    pub owner: Option<ComfyOwner>,
    #[serde(default)]
    pub mode: Option<ComfyMode>,
    #[serde(default)]
    pub transition_state: Option<ComfyTransitionState>,
    #[serde(default)]
    pub transition_target: Option<ComfyTransitionTarget>,
    pub comfyui: Option<String>,
    pub unsloth_studio: Option<String>,
    pub llama_server: Option<String>,
    pub studio_backend_health: Option<String>,
    #[serde(default)]
    pub resident_qwen_profile: Option<ComfyResidentProfile>,
    pub comfy_busy: Option<bool>,
    pub comfy_queue_running: Option<i64>,
    pub comfy_queue_pending: Option<i64>,
    pub vram_used_mib: Option<i64>,
    pub vram_free_mib: Option<i64>,
}

/// Pulled-through remote snapshot (deployed schema E response).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComfyRemoteSnapshot {
    /// Opaque producer-lifetime identity.
    pub producer_epoch: String,
    /// Monotonic integer; +1 exactly once per authoritative-tuple change.
    pub state_generation: i64,
    /// Separate opaque hex digest of the authoritative tuple.
    pub state_fingerprint: String,
    pub state: ComfySnapshotState,
}

/// Derived source health / freshness, computed on the Switchyard side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfySourceHealth {
    Healthy,
    Stale,
    Unavailable,
}

/// Switchyard-side view of one ComfyNinja snapshot: observed fields from the
/// producer plus Switchyard-derived freshness. Never holds secrets.
#[derive(Clone, Debug)]
pub struct ComfyResourceState {
    pub producer_epoch: String,
    pub state_generation: i64,
    pub state_fingerprint: String,
    pub state: ComfySnapshotState,
    /// Derived on Switchyard. `Unavailable`/`Stale` fail closed (no spill).
    pub source_health: ComfySourceHealth,
    /// Unix seconds of the last successful read (None = never/unknown).
    pub last_success_at: Option<i64>,
    /// Coarse opaque error text for the most recent failed read (no secrets).
    pub error: Option<String>,
}

impl ComfyResourceState {
    /// Freshness evaluation: with a TTL, is this snapshot still usable?
    ///
    /// Mirrors the existing fail-closed posture: unknown / stale / error -> NOT
    /// an authority for readiness (ComfyNinja hardware domain ineligible until
    /// refreshed). It never opens a spill/fallback.
    pub fn healthy(&self, observed_unix: i64, ttl: Duration) -> bool {
        if self.source_health != ComfySourceHealth::Healthy {
            return false;
        }
        self.last_success_at
            .map(|last| observed_unix.saturating_sub(last) <= ttl.as_secs() as i64)
            .unwrap_or(false)
    }

    /// Construct an unavailable (error) state for pool-local failure isolation.
    pub fn unavailable_error(error: String) -> Self {
        Self {
            producer_epoch: String::new(),
            state_generation: 0,
            state_fingerprint: String::new(),
            state: ComfySnapshotState::default(),
            source_health: ComfySourceHealth::Unavailable,
            last_success_at: None,
            error: Some(error),
        }
    }
}

// ---------------------------------------------------------------------------
// Switchyard-owned snapshot TTL cache (reuses the generic async-mutex pattern)
// ---------------------------------------------------------------------------

/// TTL-cached, coalesced-refresh holder for the latest ComfyNinja snapshot.
///
/// Reuses the same control-flow as `ResourceState` (async-mutex held across a
/// single refresh; cache hits return an `Arc` clone) but is typed to the
/// ComfyNinja hardware domain and does not share the cost-pool snapshot. It is
/// a pure intake cache — no routing/eligibility is derived here.
pub struct ComfySnapshotCache {
    fetch: Box<
        dyn Fn() -> futures::future::BoxFuture<'static, Result<ComfyResourceState, String>>
            + Send
            + Sync,
    >,
    ttl: Duration,
    cache: AsyncMutex<Option<(std::time::Instant, Arc<ComfyResourceState>)>>,
}

impl std::fmt::Debug for ComfySnapshotCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfySnapshotCache")
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl ComfySnapshotCache {
    pub fn new(
        ttl: Duration,
        fetch: Box<
            dyn Fn() -> futures::future::BoxFuture<'static, Result<ComfyResourceState, String>>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            fetch,
            ttl,
            cache: AsyncMutex::new(None),
        }
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Return a fresh (non-cached) snapshot if the cache is empty or expired,
    /// holding the lock across the fetch so concurrent callers coalesce.
    pub async fn snapshot(&self) -> Result<Arc<ComfyResourceState>, String> {
        let mut guard = self.cache.lock().await;
        if let Some((at, cached)) = guard.as_ref()
            && at.elapsed() < self.ttl
        {
            return Ok(Arc::clone(cached));
        }
        let fetched = (self.fetch)().await?;
        let arc = Arc::new(fetched);
        *guard = Some((std::time::Instant::now(), Arc::clone(&arc)));
        Ok(arc)
    }
}

// ---------------------------------------------------------------------------
// Transition ledger: idempotent ingestion + bounded history + sliding windows
// ---------------------------------------------------------------------------

/// One measured (source_class -> target_class) transition sample.
///
/// The runtime-ready duration is **Switchyard-derived** by correlating a
/// START and its TERMINAL (same epoch + transition_id): the deployed event
/// carries factual timestamps but no duration field, so the duration is the
/// ComfyNinja-side fact `terminal.observed_at - start.observed_at` authored in
/// the Switchyard-owned ledger (not a ComfyNinja-computed readiness estimate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionSample {
    pub runtime_ready_ms: f64,
    pub transition_id: i64,
    pub observed_at: f64,
}

/// Bounded aggregate for one (source_class -> target_class) cell.
///
/// Keeps `sample_n`, `median_ms`, `observed_min_ms`, `observed_max_ms`. We
/// deliberately do not invent a numeric confidence interval (8C.1 evidence
/// representation): observed-min/max are presented as observed values, not as
/// a statistical CI.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TransitionCellStats {
    pub sample_n: u64,
    pub median_ms: Option<f64>,
    pub observed_min_ms: Option<f64>,
    pub observed_max_ms: Option<f64>,
}

/// Builder that mirrors START/TERMINAL correlation for duration derivation.
#[derive(Debug, Default)]
pub struct CorrelationTable {
    /// (epoch, transition_id) -> (rid, start observed_at).
    starts: std::collections::HashMap<(String, i64), (String, f64)>,
}

impl CorrelationTable {
    pub fn new() -> Self {
        Self {
            starts: std::collections::HashMap::new(),
        }
    }

    /// Correlate an event. Returns a derived (source_class,target_class) +
    /// runtime-ms when a terminal is matched to a remembered start.
    ///
    /// Deterministic; orphan terminals and reused rids never fabricate a match.
    pub fn feed(
        &mut self,
        ev: &TransitionEvent,
    ) -> Option<(ComfyProfileClass, ComfyProfileClass, f64)> {
        match ev.phase {
            TransitionPhase::Start => {
                self.starts.insert(
                    (ev.epoch.clone(), ev.transition_id),
                    (ev.rid.clone(), ev.observed_at),
                );
                None
            }
            TransitionPhase::Terminal => {
                let key = (ev.epoch.clone(), ev.transition_id);
                if ev.correlation_status == CorrelationStatus::OrphanTerminal {
                    return None;
                }
                let start_at = self.starts.remove(&key).map(|(_, t)| t)?;
                let runtime_ready_ms = (ev.observed_at - start_at) * 1000.0;
                if runtime_ready_ms < 0.0 {
                    return None; // clock skew / out-of-order; not measurable
                }
                Some((
                    ComfyProfileClass::from_source_mode(ev.source_mode),
                    ComfyProfileClass::from_target_mode(ev.target_mode),
                    runtime_ready_ms,
                ))
            }
        }
    }
}

impl ComfyProfileClass {
    /// Derive a conservative bounded class from an event source mode.
    pub fn from_source_mode(mode: Option<ComfyMode>) -> Self {
        match mode {
            Some(ComfyMode::Idle) => Self::SealedIdle,
            Some(_) => Self::Generic,
            None => Self::Generic,
        }
    }

    /// Derive a conservative bounded class from an event target mode.
    pub fn from_target_mode(mode: Option<ComfyMode>) -> Self {
        match mode {
            Some(ComfyMode::Idle) => Self::SealedIdle,
            Some(_) => Self::Generic,
            None => Self::Generic,
        }
    }
}

/// The Switchyard-owned bounded transition/history ledger.
///
/// Responsibilities (authority split):
/// - Remove duplicates (idempotency) by `(producer_epoch, eid)`.
/// - Detect cursor gaps (same epoch missing `eid`).
/// - Detect producer restarts (epoch change -> reset).
/// - Detect bounded-ledger overflow (cursor below `oldest_available_eid`).
/// - Maintain a bounded sliding-window of recent samples per
///   `(source_class -> target_class)` cell.
/// - Track `last_active_profile` (Switchyard-owned; never a ComfyNinja event field).
///
/// Boundedness: we keep at most `max_recent_events` transition events and
/// `sliding_window_size` samples per cell. This is a pure in-memory ledger
/// (see the module/package note; restart semantics below).
#[derive(Debug)]
pub struct ComfyTransitionLedger {
    max_recent_events: usize,
    sliding_window_size: usize,
    /// Dedup/retention: last `(cursor, eid)` window, oldest first.
    recent: std::collections::VecDeque<(String, i64, TransitionEvent)>,
    /// Sliding-window samples per (source -> target) cell.
    cells:
        std::collections::BTreeMap<(ComfyProfileClass, ComfyProfileClass), Vec<TransitionSample>>,
    /// START/TERMINAL correlation table (bounded; pruned by retention).
    correlation: CorrelationTable,
    /// The profile that was last ACTIVE/resident before the current sealed idle.
    last_active_profile: Option<ComfyProfileClass>,
}

impl Default for ComfyTransitionLedger {
    fn default() -> Self {
        Self::with_limits(4096, 64)
    }
}

impl ComfyTransitionLedger {
    pub fn with_limits(max_recent_events: usize, sliding_window_size: usize) -> Self {
        Self {
            max_recent_events: max_recent_events.max(1),
            sliding_window_size: sliding_window_size.max(1),
            recent: std::collections::VecDeque::new(),
            cells: std::collections::BTreeMap::new(),
            correlation: CorrelationTable::new(),
            last_active_profile: None,
        }
    }

    pub fn last_active_profile(&self) -> Option<ComfyProfileClass> {
        self.last_active_profile
    }

    pub fn recent_event_count(&self) -> usize {
        self.recent.len()
    }

    pub fn cell(
        &self,
        source: ComfyProfileClass,
        target: ComfyProfileClass,
    ) -> Option<&[TransitionSample]> {
        self.cells.get(&(source, target)).map(|v| v.as_slice())
    }

    pub fn cell_stats(
        &self,
        source: ComfyProfileClass,
        target: ComfyProfileClass,
    ) -> TransitionCellStats {
        let Some(samples) = self.cells.get(&(source, target)) else {
            return TransitionCellStats::default();
        };
        let mut vals: Vec<f64> = samples.iter().map(|s| s.runtime_ready_ms).collect();
        let n = vals.len() as u64;
        if n == 0 {
            return TransitionCellStats::default();
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        TransitionCellStats {
            sample_n: n,
            median_ms: Some(vals[(n as usize - 1) / 2]),
            observed_min_ms: vals.first().copied(),
            observed_max_ms: vals.last().copied(),
        }
    }

    /// Ingest an event stream for a given feed disposition. Returns how many
    /// events were newly appended (deduplicated). Deterministic and idempotent.
    ///
    /// Cursor/continuity state (gap/epoch-reset/overflow) is already resolved
    /// by [`advance_cursor`] before this call; this method applies the
    /// disposition to reset retained state when the producer restarts.
    pub fn ingest(&mut self, disposition: FeedDisposition, feed: &TransitionFeed) -> usize {
        if disposition == FeedDisposition::EpochReset {
            // Producer restarted: discard retained dedup window and correlation
            // table from the prior epoch; fresh (bounded) window for the new one.
            self.recent.clear();
            self.correlation = CorrelationTable::new();
        }
        // Overflow on the ComfyNinja side: switch to their oldest_available
        // leading edge by just re-syncing; our own dedup window still prevents
        // re-appending what we already have.

        let mut added = 0usize;
        for ev in &feed.events {
            let dup = self
                .recent
                .iter()
                .any(|(epoch, eid, _)| epoch == &ev.epoch && *eid == ev.eid);
            if dup {
                continue;
            }
            // Derive a duration by correlating START/TERMINAL (Switchyard-owned).
            let derived = self.correlation.feed(ev);
            self.recent
                .push_back((ev.epoch.clone(), ev.eid, ev.clone()));
            added += 1;

            if let Some((src, tgt, runtime_ready_ms)) = derived {
                let sample = TransitionSample {
                    runtime_ready_ms,
                    transition_id: ev.transition_id,
                    observed_at: ev.observed_at,
                };
                let entry = self.cells.entry((src, tgt)).or_default();
                entry.push(sample);
                if entry.len() > self.sliding_window_size {
                    let excess = entry.len() - self.sliding_window_size;
                    entry.drain(0..excess);
                }
                // last_active_profile is Switchyard-owned: a terminal that
                // lands in a non-idle target records that class as the last
                // active profile; a terminal to sealed-idle keeps the prior
                // active (or stays unknown).
                if tgt != ComfyProfileClass::SealedIdle {
                    self.last_active_profile = Some(tgt);
                }
            }

            // Trim the retained dedup window (bounded).
            while self.recent.len() > self.max_recent_events {
                self.recent.pop_front();
            }
        }
        added
    }
}

// ---------------------------------------------------------------------------
// Feed-ahead disposition: pure cursor semantics (deployed feed_meta)
// ---------------------------------------------------------------------------

/// Decide how to advance the cursor given a feed page and the persisted cursor.
///
/// Encodes the deployed `feed_meta` semantics exactly:
/// - `epoch_reset` (after_epoch != current, or reset flag) -> EpochReset;
/// - overflow when the cursor is strictly BELOW the retained leading edge
///   (`after_eid < oldest_available_eid - 1`);
/// - gap when (same epoch) the first returned `eid` is not contiguous with the
///   cursor;
/// - otherwise a contiguous continuation advancing to the latest returned `eid`.
pub fn advance_cursor(
    persisted: Option<&TransitionCursor>,
    feed: &TransitionFeed,
) -> (FeedDisposition, Option<TransitionCursor>) {
    use FeedDisposition::*;

    // A `None` persisted cursor means a FRESH start with no prior history —
    // NOT a producer restart. Begin at eid 0 of the feed's current epoch with
    // a clean continuation (Continuation). Continuity from the past is not
    // claimed because there was none.
    let Some(cursor) = persisted else {
        let last = feed.events.last().map(|e| e.eid).unwrap_or(0);
        return (
            Continuation,
            Some(TransitionCursor {
                producer_epoch: feed.producer_epoch.clone(),
                last_eid: last,
            }),
        );
    };

    // Producer restart is only detectable when we have a prior cursor to
    // compare the epoch against. If we are not aware of an epoch change (and
    // the feed did NOT flag a reset), proceed to overflow/gap/continuation.
    if feed.epoch_reset || cursor.producer_epoch != feed.producer_epoch {
        return (
            EpochReset,
            Some(TransitionCursor::fresh(&feed.producer_epoch)),
        );
    }

    // Overflow: cursor strictly below retained leading edge.
    if let Some(oldest) = feed.oldest_available_eid
        && cursor.last_eid < oldest - 1
    {
        let last = feed.latest_eid.unwrap_or(oldest - 1).max(cursor.last_eid);
        return (
            Overflow,
            Some(TransitionCursor {
                producer_epoch: feed.producer_epoch.clone(),
                last_eid: last,
            }),
        );
    }

    // Gap: same epoch but the first returned event is not contiguous.
    if let Some(first) = feed.events.first() {
        let next_expected = cursor.last_eid + 1;
        if first.eid != next_expected {
            return (Gap, Some(cursor.clone()));
        }
        let new_last = feed.events.last().map(|e| e.eid).unwrap_or(cursor.last_eid);
        return (
            Continuation,
            Some(TransitionCursor {
                producer_epoch: feed.producer_epoch.clone(),
                last_eid: new_last,
            }),
        );
    }

    // Empty page: no advance.
    (Continuation, Some(cursor.clone()))
}

/// A parse error for a ComfyNinja payload.
#[derive(Debug, Error)]
pub enum ComfyParseError {
    #[error("missing field {0}")]
    MissingField(&'static str),
    #[error("invalid value for {0}: {1}")]
    InvalidValue(&'static str, String),
    #[error("malformed JSON: {0}")]
    Json(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // --- fixtures -----------------------------------------------------------

    /// Build a START event for transition `tid` in `epoch` at `eid`, from
    /// `source` mode toward `target` mode, observed at `observed`.
    fn start_event(
        epoch: &str,
        eid: i64,
        tid: i64,
        source: Option<ComfyMode>,
        target: Option<ComfyMode>,
        observed: f64,
    ) -> TransitionEvent {
        TransitionEvent {
            rid: format!("shell-{tid}"),
            epoch: epoch.to_string(),
            eid,
            transition_id: tid,
            transition_start_known: true,
            correlation_status: CorrelationStatus::Matched,
            command: "activate:comfy".into(),
            source_mode: source,
            target_mode: target,
            source_profile: None,
            target_profile: None,
            phase: TransitionPhase::Start,
            result: Some(TransitionResult::Pending),
            underlying_rc: None,
            observed_at: observed,
            ingested_at: observed,
        }
    }

    /// Build a TERMINAL event for transition `tid` in `epoch` at `eid`
    /// observed at `observed`.
    fn terminal_event(
        epoch: &str,
        eid: i64,
        tid: i64,
        source: Option<ComfyMode>,
        target: Option<ComfyMode>,
        observed: f64,
    ) -> TransitionEvent {
        TransitionEvent {
            rid: format!("shell-{tid}"),
            epoch: epoch.to_string(),
            eid,
            transition_id: tid,
            transition_start_known: false,
            correlation_status: CorrelationStatus::Matched,
            command: "activate:comfy".into(),
            source_mode: source,
            target_mode: target,
            source_profile: None,
            target_profile: None,
            phase: TransitionPhase::Terminal,
            result: Some(TransitionResult::Succeeded),
            underlying_rc: Some(0),
            observed_at: observed,
            ingested_at: observed,
        }
    }

    struct FeedBuilder {
        epoch: String,
        oldest: Option<i64>,
        latest: Option<i64>,
        epoch_reset: bool,
        gap: bool,
        overflow: bool,
        events: Vec<TransitionEvent>,
    }

    impl FeedBuilder {
        fn new(epoch: &str) -> Self {
            Self {
                epoch: epoch.into(),
                oldest: None,
                latest: None,
                epoch_reset: false,
                gap: false,
                overflow: false,
                events: Vec::new(),
            }
        }
        fn push(mut self, ev: TransitionEvent) -> Self {
            let eid = ev.eid;
            self.events.push(ev);
            self.latest = Some(self.latest.map_or(eid, |l| l.max(eid)));
            if self.oldest.is_none() {
                self.oldest = Some(eid);
            } else {
                self.oldest = Some(self.oldest.unwrap().min(eid));
            }
            self
        }
        fn epoch_reset(mut self) -> Self {
            self.epoch_reset = true;
            self
        }
        fn overflow_oldest(mut self, oldest: i64) -> Self {
            self.oldest = Some(oldest);
            self
        }
        fn build(self) -> TransitionFeed {
            TransitionFeed {
                producer_epoch: self.epoch,
                oldest_available_eid: self.oldest,
                latest_eid: self.latest,
                epoch_reset: self.epoch_reset,
                gap: self.gap,
                overflow: self.overflow,
                events: self.events,
            }
        }
    }

    /// A single completed transition (START eid n, TERMINAL eid n+1) in epoch "e0"
    /// sealing into idle, with a derived runtime duration of 1000ms.
    fn sealed_idle_transition(tid: i64) -> Vec<TransitionEvent> {
        vec![
            start_event(
                "e0",
                tid * 2 + 1,
                tid,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            ),
            terminal_event(
                "e0",
                tid * 2 + 2,
                tid,
                Some(ComfyMode::Comfy),
                Some(ComfyMode::Idle),
                1001.0,
            ),
        ]
    }

    fn feed_from_events(events: Vec<TransitionEvent>) -> TransitionFeed {
        let mut b = FeedBuilder::new("e0");
        for ev in events {
            b = b.push(ev);
        }
        b.build()
    }

    // --- 1. normal ingestion ------------------------------------------------

    #[tokio::test]
    async fn normal_ingestion_derives_duration_in_sealed_cell() {
        let mut ledger = ComfyTransitionLedger::default();
        let feed = feed_from_events(sealed_idle_transition(1));
        let added = ledger.ingest(FeedDisposition::Continuation, &feed);
        assert_eq!(added, 2, "start+terminal both ingested");
        assert_eq!(ledger.recent_event_count(), 2);
        // START (source idle) -> TERMINAL (target idle) lands in the
        // (source=terminal.source_mode -> target=terminal.target_mode) cell.
        // Terminal source_mode=comfy (prior active) -> Generic; target=idle -> SealedIdle.
        let stats = ledger.cell_stats(ComfyProfileClass::Generic, ComfyProfileClass::SealedIdle);
        assert_eq!(stats.sample_n, 1);
        // Terminal observed at 1001, start at 1000 -> 1000 ms.
        assert_eq!(stats.median_ms, Some(1000.0));
        // Sealing into idle: last_active_profile remains None (nothing was active).
        assert_eq!(ledger.last_active_profile(), None);
    }

    #[tokio::test]
    async fn normal_ingestion_active_target_sets_last_active_profile() {
        let mut ledger = ComfyTransitionLedger::default();
        // Transition that ends ACTIVE in comfy (not sealing into idle).
        let feed = feed_from_events(vec![
            start_event(
                "e0",
                1,
                1,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            ),
            terminal_event(
                "e0",
                2,
                1,
                Some(ComfyMode::Comfy),
                Some(ComfyMode::Comfy),
                1001.0,
            ),
        ]);
        ledger.ingest(FeedDisposition::Continuation, &feed);
        // Terminal target comfy (non-idle) -> last_active_profile = Generic (seam limit).
        assert_eq!(
            ledger.last_active_profile(),
            Some(ComfyProfileClass::Generic)
        );
    }

    // --- 2. duplicate events / idempotency -----------------------------------

    #[tokio::test]
    async fn duplicate_events_are_idempotent() {
        let mut ledger = ComfyTransitionLedger::default();
        let feed = feed_from_events(sealed_idle_transition(1));
        let first = ledger.ingest(FeedDisposition::Continuation, &feed);
        assert_eq!(first, 2);
        // Same feed re-sent (e.g. after a transient fetch retry) is a no-op.
        let second = ledger.ingest(FeedDisposition::Continuation, &feed);
        assert_eq!(second, 0, "duplicate events must not be re-appended");
        assert_eq!(ledger.recent_event_count(), 2);
        let stats = ledger.cell_stats(ComfyProfileClass::Generic, ComfyProfileClass::SealedIdle);
        assert_eq!(stats.sample_n, 1, "duration must not double-count on dup");
    }

    #[tokio::test]
    async fn duplicate_feed_with_cursor_advance_is_noop() {
        let mut ledger = ComfyTransitionLedger::default();
        let feed = feed_from_events(sealed_idle_transition(1));
        let (disp, _) = advance_cursor(None, &feed);
        ledger.ingest(disp, &feed);
        // Re-send the same full feed after a (hypothetical) continuation.
        let (disp2, _) = advance_cursor(Some(&TransitionCursor::fresh("e0")), &feed);
        let added = ledger.ingest(disp2, &feed);
        assert_eq!(added, 0);
    }

    // --- 3. cursor continuation ----------------------------------------------

    #[tokio::test]
    async fn cursor_continuation_advances_to_latest() {
        // Retained from eid 1; cursor already at 2 (contiguous before 3,4) ->
        // a clean continuation, no overflow (2 >= 1-1) and no gap (first==3).
        let mut b = FeedBuilder::new("e0").overflow_oldest(1);
        for ev in sealed_idle_transition(1) {
            b = b.push(ev); // eids 3,4
        }
        let feed = b.build();
        let persisted = TransitionCursor {
            producer_epoch: "e0".into(),
            last_eid: 2,
        };
        let (disp, next) = advance_cursor(Some(&persisted), &feed);
        assert_eq!(disp, FeedDisposition::Continuation);
        let next = next.expect("cursor advances");
        assert_eq!(next.last_eid, 4);
        assert_eq!(next.producer_epoch, "e0");
    }

    #[tokio::test]
    async fn cursor_fresh_start_none_is_continuation_not_epoch_reset() {
        // A None persisted cursor = fresh start with no prior history. It must
        // NOT be misread as a producer restart (epoch reset). Continuity from
        // the past is not claimed because there was none.
        let mut b = FeedBuilder::new("e0").overflow_oldest(1);
        for ev in sealed_idle_transition(1) {
            b = b.push(ev); // eids 3,4
        }
        let feed = b.build();
        let (disp, next) = advance_cursor(None, &feed);
        assert_eq!(
            disp,
            FeedDisposition::Continuation,
            "fresh start is a continuation, not epoch reset"
        );
        let next = next.expect("fresh cursor advances");
        assert_eq!(next.last_eid, 4);
        assert_eq!(next.producer_epoch, "e0");
    }

    #[tokio::test]
    async fn cursor_continuation_empty_page_no_advance() {
        let feed = FeedBuilder::new("e0").build();
        let persisted = TransitionCursor {
            producer_epoch: "e0".into(),
            last_eid: 5,
        };
        let (disp, next) = advance_cursor(Some(&persisted), &feed);
        assert_eq!(disp, FeedDisposition::Continuation);
        assert_eq!(next.unwrap().last_eid, 5);
        assert_eq!(feed.events.len(), 0);
    }

    // --- 4. cursor gaps ------------------------------------------------------

    #[tokio::test]
    async fn cursor_gap_detects_missing_eid() {
        // ComfyNinja retains from eid 1; cursor is at 3 (within retained), but
        // the returned page starts at eid 8 -> a gap (missing eids 4-7).
        let feed = FeedBuilder::new("e0")
            .overflow_oldest(1)
            .push(start_event(
                "e0",
                8,
                2,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                2000.0,
            ))
            .build();
        let persisted = TransitionCursor {
            producer_epoch: "e0".into(),
            last_eid: 3,
        };
        let (disp, next) = advance_cursor(Some(&persisted), &feed);
        assert_eq!(disp, FeedDisposition::Gap);
        // On a gap we do not fabricate; cursor stays (re-fetch advised).
        assert_eq!(next.unwrap().last_eid, 3);
    }

    #[tokio::test]
    async fn gap_ingest_does_not_fabricate_duration() {
        // Even if fed a gap page, the ledger only uses contiguous-start events
        // for duration derivation; a lone orphan terminal (no start) yields no cell.
        let mut ledger = ComfyTransitionLedger::default();
        let feed = feed_from_events(vec![terminal_event(
            "e0",
            9,
            99,
            Some(ComfyMode::Comfy),
            Some(ComfyMode::Idle),
            3000.0,
        )]);
        // Orphan terminal (no matching start in correlation table):
        ledger.ingest(FeedDisposition::Gap, &feed);
        assert_eq!(ledger.recent_event_count(), 1);
        let stats = ledger.cell_stats(ComfyProfileClass::Generic, ComfyProfileClass::SealedIdle);
        assert_eq!(
            stats.sample_n, 0,
            "orphan terminal must not fabricate a duration"
        );
    }

    // --- 5. epoch reset ------------------------------------------------------

    #[tokio::test]
    async fn epoch_reset_resets_cursor_and_clears_recent() {
        let mut ledger = ComfyTransitionLedger::default();
        // Ingest epoch e0, then a feed stating epoch e1 with epoch_reset=true.
        ledger.ingest(
            FeedDisposition::Continuation,
            &feed_from_events(sealed_idle_transition(1)),
        );
        assert_eq!(ledger.recent_event_count(), 2);

        let new_feed = FeedBuilder::new("e1")
            .epoch_reset()
            .push(start_event(
                "e1",
                1,
                1,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Studio),
                5000.0,
            ))
            .push(terminal_event(
                "e1",
                2,
                1,
                Some(ComfyMode::Studio),
                Some(ComfyMode::Comfy),
                5001.0,
            ))
            .build();
        let (disp, cursor) = advance_cursor(
            Some(&TransitionCursor {
                producer_epoch: "e0".into(),
                last_eid: 2,
            }),
            &new_feed,
        );
        assert_eq!(disp, FeedDisposition::EpochReset);
        assert_eq!(cursor.unwrap().producer_epoch, "e1");

        let added = ledger.ingest(disp, &new_feed);
        // Only the two new epoch events land in the retained window.
        assert_eq!(ledger.recent_event_count(), 2);
        assert_eq!(added, 2);
    }

    #[tokio::test]
    async fn epoch_reset_clears_correlation_cross_epoch_orphan() {
        let mut ledger = ComfyTransitionLedger::default();
        // START in e0, then reset to e1 with a TERMINAL that shares transition_id
        // but is in the NEW epoch: must NOT match the old epoch's start.
        ledger.ingest(
            FeedDisposition::Continuation,
            &feed_from_events(vec![start_event(
                "e0",
                1,
                1,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            )]),
        );
        let new_feed = FeedBuilder::new("e1")
            .epoch_reset()
            .push(terminal_event(
                "e1",
                1,
                1,
                Some(ComfyMode::Comfy),
                Some(ComfyMode::Comfy),
                2000.0,
            ))
            .build();
        let (disp, _) = advance_cursor(
            Some(&TransitionCursor {
                producer_epoch: "e0".into(),
                last_eid: 1,
            }),
            &new_feed,
        );
        assert_eq!(disp, FeedDisposition::EpochReset);
        ledger.ingest(disp, &new_feed);
        let stats = ledger.cell_stats(ComfyProfileClass::Generic, ComfyProfileClass::Generic);
        assert_eq!(
            stats.sample_n, 0,
            "cross-epoch start must not satisfy a terminal"
        );
    }

    // --- 6. TTL expiry -------------------------------------------------------

    struct StubFetch {
        counter: std::sync::atomic::AtomicU64,
    }

    impl StubFetch {
        fn call(&self) -> futures::future::BoxFuture<'static, Result<ComfyResourceState, String>> {
            let c = self
                .counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            Box::pin(async move {
                Ok(ComfyResourceState {
                    producer_epoch: format!("e{c}"),
                    state_generation: c as i64,
                    state_fingerprint: format!("fp{c}"),
                    state: ComfySnapshotState::default(),
                    source_health: ComfySourceHealth::Healthy,
                    last_success_at: Some(1_000_000),
                    error: None,
                })
            })
        }
    }

    #[tokio::test]
    async fn ttl_serves_cached_then_refetches() {
        let stub = std::sync::Arc::new(StubFetch {
            counter: std::sync::atomic::AtomicU64::new(0),
        });
        let stub2 = std::sync::Arc::clone(&stub);
        let cache =
            ComfySnapshotCache::new(Duration::from_millis(200), Box::new(move || stub2.call()));
        // First call -> fetch.
        let s1 = cache.snapshot().await.unwrap();
        assert_eq!(s1.state_generation, 1);
        // Within TTL -> cached, no new fetch.
        let s2 = cache.snapshot().await.unwrap();
        assert_eq!(s2.state_generation, 1);
        assert_eq!(
            stub.counter
                .fetch_add(0, std::sync::atomic::Ordering::SeqCst),
            1
        );
        // Sleep past TTL -> refetch.
        tokio::time::sleep(Duration::from_millis(250)).await;
        let s3 = cache.snapshot().await.unwrap();
        assert_eq!(s3.state_generation, 2);
        assert_eq!(
            stub.counter
                .fetch_add(0, std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    // --- 7. stale / unavailable fail-closed -----------------------------------

    #[tokio::test]
    async fn unavailable_fails_closed() {
        let s = ComfyResourceState::unavailable_error("connect failed".into());
        assert_eq!(s.source_health, ComfySourceHealth::Unavailable);
        assert!(!s.healthy(1_000_100, Duration::from_secs(30)));
    }

    #[tokio::test]
    async fn stale_fails_closed() {
        let s = ComfyResourceState {
            producer_epoch: "e".into(),
            state_generation: 1,
            state_fingerprint: "fp".into(),
            state: ComfySnapshotState::default(),
            source_health: ComfySourceHealth::Healthy,
            last_success_at: Some(1_000_000), // read an hour ago
            error: None,
        };
        // 30s TTL, read 3600s ago -> stale.
        assert!(!s.healthy(1_003_600, Duration::from_secs(30)));
        // Within TTL -> healthy.
        assert!(s.healthy(1_000_010, Duration::from_secs(30)));
    }

    #[tokio::test]
    async fn never_read_fails_closed() {
        let s = ComfyResourceState {
            producer_epoch: "e".into(),
            state_generation: 0,
            state_fingerprint: String::new(),
            state: ComfySnapshotState::default(),
            source_health: ComfySourceHealth::Healthy,
            last_success_at: None,
            error: None,
        };
        assert!(!s.healthy(1_000_000, Duration::from_secs(30)));
    }

    // --- 8. ledger bounds / overflow -----------------------------------------

    #[tokio::test]
    async fn recent_event_window_bounded() {
        // max_recent_events = 4, feed 5 events -> recent trimmed to 4.
        let mut ledger = ComfyTransitionLedger::with_limits(4, 64);
        let events = sealed_idle_transition(1)
            .into_iter()
            .chain(sealed_idle_transition(2).into_iter())
            .chain(sealed_idle_transition(3).into_iter()); // 6 events
        let feed = feed_from_events(events.collect());
        ledger.ingest(FeedDisposition::Continuation, &feed);
        assert!(
            ledger.recent_event_count() <= 4,
            "retained window must be bounded"
        );
    }

    #[tokio::test]
    async fn sliding_window_per_cell_bounded() {
        let mut ledger = ComfyTransitionLedger::with_limits(4096, 2);
        // 3 completed transitions all sealing idle -> same cell; only 2 kept.
        for tid in 1..=3 {
            let feed = feed_from_events(sealed_idle_transition(tid));
            ledger.ingest(FeedDisposition::Continuation, &feed);
        }
        let stats = ledger.cell_stats(ComfyProfileClass::Generic, ComfyProfileClass::SealedIdle);
        assert_eq!(stats.sample_n, 2, "sliding window caps at 2");
    }

    #[tokio::test]
    async fn overflow_disposition_detected() {
        // Cursor at eid 1, but ComfyNinja only retains from eid 10 onwards.
        let feed = FeedBuilder::new("e0")
            .overflow_oldest(10)
            .push(start_event(
                "e0",
                11,
                5,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            ))
            .build();
        let persisted = TransitionCursor {
            producer_epoch: "e0".into(),
            last_eid: 1,
        };
        let (disp, _) = advance_cursor(Some(&persisted), &feed);
        assert_eq!(disp, FeedDisposition::Overflow);
    }

    #[tokio::test]
    async fn boundary_cursor_not_overflow() {
        // Cursor at oldest_available_eid - 1 (contiguous start) is NOT overflow.
        let feed = FeedBuilder::new("e0")
            .overflow_oldest(10)
            .push(start_event(
                "e0",
                10,
                5,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                1000.0,
            ))
            .build();
        let persisted = TransitionCursor {
            producer_epoch: "e0".into(),
            last_eid: 9,
        };
        let (disp, _) = advance_cursor(Some(&persisted), &feed);
        assert_eq!(disp, FeedDisposition::Continuation);
    }

    // --- 9. restart / restart semantics via epoch change ----------------------

    #[tokio::test]
    async fn restart_semantics_clear_recent_retain_bounded() {
        let mut ledger = ComfyTransitionLedger::default();
        ledger.ingest(
            FeedDisposition::Continuation,
            &feed_from_events(sealed_idle_transition(1)),
        );
        ledger.ingest(
            FeedDisposition::Continuation,
            &feed_from_events(sealed_idle_transition(2)),
        );
        assert_eq!(ledger.recent_event_count(), 4);

        // Producer restart: epoch e1 carries a fresh feed; after reset only e1 events remain.
        let new_feed = FeedBuilder::new("e1")
            .epoch_reset()
            .push(start_event(
                "e1",
                1,
                1,
                Some(ComfyMode::Idle),
                Some(ComfyMode::Comfy),
                9000.0,
            ))
            .push(terminal_event(
                "e1",
                2,
                1,
                Some(ComfyMode::Comfy),
                Some(ComfyMode::Idle),
                9001.0,
            ))
            .build();
        let (disp, _) = advance_cursor(
            Some(&TransitionCursor {
                producer_epoch: "e0".into(),
                last_eid: 4,
            }),
            &new_feed,
        );
        assert_eq!(disp, FeedDisposition::EpochReset);
        ledger.ingest(disp, &new_feed);
        assert_eq!(
            ledger.recent_event_count(),
            2,
            "old epoch events must be discarded on restart"
        );
    }

    // --- serialization round-trip --------------------------------------------

    #[tokio::test]
    async fn remote_snapshot_parses_deployed_shape() {
        let json = serde_json::json!({
            "producer_epoch": "abc123",
            "state_generation": 3,
            "state_fingerprint": "deadbeef",
            "state": {
                "owner": "comfyui",
                "mode": "comfy",
                "transition_state": "idle",
                "transition_target": null,
                "comfyui": "comfyui",
                "unsloth_studio": null,
                "llama_server": null,
                "studio_backend_health": "ok",
                "resident_qwen_profile": "unknown",
                "vram_used_mib": 67,
                "vram_free_mib": 24260,
                "comfy_busy": false,
                "comfy_queue_running": null,
                "comfy_queue_pending": null
            }
        });
        let snap: ComfyRemoteSnapshot = serde_json::from_value(json).expect("parse deployed shape");
        assert_eq!(snap.producer_epoch, "abc123");
        assert_eq!(snap.state_generation, 3);
        assert_eq!(snap.state.owner, Some(ComfyOwner::Comfy));
        assert_eq!(snap.state.mode, Some(ComfyMode::Comfy));
        assert_eq!(snap.state.vram_used_mib, Some(67));
        assert_eq!(snap.state.vram_free_mib, Some(24260));
        assert_eq!(
            snap.state.resident_qwen_profile,
            Some(ComfyResidentProfile::Unknown)
        );
    }

    #[tokio::test]
    async fn transition_feed_parses_deployed_events() {
        let json = serde_json::json!({
            "producer_epoch": "e0",
            "oldest_available_eid": 1,
            "latest_eid": 2,
            "epoch_reset": false,
            "gap": false,
            "overflow": false,
            "events": [{
                "rid": "shell-5",
                "epoch": "e0",
                "eid": 1,
                "transition_id": 5,
                "transition_start_known": true,
                "correlation_status": "matched",
                "command": "activate:comfy",
                "source_mode": "idle",
                "target_mode": "comfy",
                "source_profile": null,
                "target_profile": null,
                "phase": "start",
                "result": "pending",
                "underlying_rc": null,
                "observed_at": 1000.5,
                "ingested_at": 1000.6
            }]
        });
        let feed: TransitionFeed = serde_json::from_value(json).expect("parse deployed feed");
        assert_eq!(feed.events.len(), 1);
        let ev = &feed.events[0];
        assert_eq!(ev.eid, 1);
        assert_eq!(ev.transition_id, 5);
        assert_eq!(ev.source_mode, Some(ComfyMode::Idle));
        assert_eq!(ev.correlation_status, CorrelationStatus::Matched);
        assert_eq!(ev.phase, TransitionPhase::Start);
    }
}
