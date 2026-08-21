// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fleet routing as a small stateless filter/ranker.
//!
//! [`FleetRouter`] deterministically filters a heterogeneous candidate set using
//! two kinds of inputs that are kept strictly separate:
//!
//! * **candidate profiles** — static Algorithm configuration describing each
//!   candidate target's *capabilities* (`tool_calling`, `reasoning`) and a
//!   deterministic deployment `preference_rank` (lower = more preferred). This
//!   is a stand-in for the already-decided deployment preference ordering; it is
//!   deliberately not an economic model.
//! * **candidate state** — an injected factual *readiness* snapshot
//!   (`ready`, `transition_required`). In this PoC it is an in-memory stub
//!   supplied to the Algorithm at construction; a later gate plugs a real
//!   fleet-fact producer behind the same shape.
//!
//! On each request the Algorithm applies, in order: a **capability filter**
//! (tool / reasoning requirements derived from the normalized request), a
//! **readiness filter** (`ready=false` or `transition_required=true` are never
//! selected for an immediate request), then a deterministic **preference rank**.
//! The highest-ranked eligible candidate becomes [`RoutingOutcome::selected_model_id`]
//! and the remaining eligible candidates become ordered `fallback_models`.
//!
//! The Algorithm is **stateless in the routing sense**: the result depends only
//! on the current request, the static profiles, and the injected snapshot — not
//! on session/cursor/history state — so repeating them yields the same decision.
//! Per-target context-window admission, LLM classifiers, and any real fact
//! producer are explicitly out of scope for this core proof.

use std::sync::Arc;

use switchyard_protocol::ModelId;

use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result, RoutingOutcome};

/// Static capability + preference profile for one candidate target.
///
/// This is Algorithm configuration, not a live fact. `tool_calling` and
/// `reasoning` describe what the target advertises it can do; `preference_rank`
/// is a deterministic deployment ordering (lower = more preferred).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateProfile {
    /// Target model id that will be selected when this candidate wins.
    pub target: ModelId,
    /// Whether the target truthfully advertises tool calling.
    pub tool_calling: bool,
    /// Whether the target truthfully advertises reasoning support.
    pub reasoning: bool,
    /// Deterministic preference ordering; lower is preferred.
    pub preference_rank: u16,
}

impl CandidateProfile {
    /// Creates a candidate profile for `target`.
    pub fn new(
        target: impl Into<ModelId>,
        tool_calling: bool,
        reasoning: bool,
        preference_rank: u16,
    ) -> Self {
        Self {
            target: target.into(),
            tool_calling,
            reasoning,
            preference_rank,
        }
    }
}

/// Live readiness of a candidate, injected as a factual snapshot.
///
/// This is a runtime fact provided by an external fleet-fact producer (stubbed
/// in-memory in this PoC), never part of the static profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CandidateState {
    /// Whether the target is immediately usable.
    pub ready: bool,
    /// Whether the target needs an external transition before it is usable.
    ///
    /// A `transition_required` candidate is never selected for an immediate
    /// request; the Algorithm performs no transition action.
    pub transition_required: bool,
}

impl CandidateState {
    /// A target that is immediately usable with no transition required.
    pub fn ready() -> Self {
        Self {
            ready: true,
            transition_required: false,
        }
    }

    /// A target that is not currently usable.
    pub fn not_ready() -> Self {
        Self {
            ready: false,
            transition_required: false,
        }
    }

    /// A capable target that needs an external transition before use.
    pub fn transition_required() -> Self {
        Self {
            ready: false,
            transition_required: true,
        }
    }

    /// Whether this candidate is immediately eligible for selection.
    fn immediately_eligible(&self) -> bool {
        self.ready && !self.transition_required
    }
}

/// One coherent, immutable readiness snapshot for the whole fleet.
///
/// A routing decision consumes exactly **one** `FleetSnapshot`, so it never mixes
/// readiness generations. It holds only the minimal facts needed to preserve the
/// S2-A readiness semantics: per-target `ready` and `transition_required`.
///
/// Missing state for a target is **fail-closed**: a target with no entry is
/// treated as not immediately eligible.
///
/// Fleet size is small and bounded; the linear `state_for` lookup is intentional
/// (see [`FleetSnapshot::state_for`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FleetSnapshot {
    /// Per-target readiness; a target absent here is treated as not ready.
    states: Vec<(ModelId, CandidateState)>,
}

impl FleetSnapshot {
    /// Builds a snapshot from per-target readiness entries.
    ///
    /// Duplicate target keys are rejected: they would make a decision ambiguous.
    ///
    /// # Errors
    ///
    /// Returns [`LibsyError::AlgorithmError`] when a target key repeats.
    pub fn new(states: Vec<(ModelId, CandidateState)>) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        for (key, _) in &states {
            if !seen.insert(key) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("fleet snapshot key {:?} appears more than once", key),
                });
            }
        }
        Ok(Self { states })
    }

    /// The readiness for `target`, or a fail-closed not-ready state when present
    /// only as an absence.
    ///
    /// The lookup is a linear scan over the few bounded fleet entries; this keeps
    /// the snapshot free of a map/hash dependency for a fleet that is deliberately
    /// small.
    pub fn state_for(&self, target: &ModelId) -> CandidateState {
        self.states
            .iter()
            .find(|(id, _)| id == target)
            .map(|(_, state)| *state)
            .unwrap_or_default()
    }
}

/// Produces one coherent [`FleetSnapshot`] synchronously.
///
/// Implementors do no network I/O in `snapshot`; any external fact producer
/// updates its current snapshot **outside** the routing request path, and the
/// routing Algorithm reads it here as a single immutable view. The snapshot is
/// returned as an `Arc` so a source can hand out a shared immutable snapshot
/// without cloning the whole fleet on every routing decision.
pub trait FleetStateSource: Send + Sync + std::fmt::Debug {
    /// The current coherent fleet readiness snapshot.
    fn snapshot(&self) -> Arc<FleetSnapshot>;
}

/// A fixed readiness source returning the same snapshot every decision.
///
/// Preserves the simple, deterministic behavior that predicate-only routing
/// tests rely on, and keeps `FleetRouter::new` ergonomic.
#[derive(Clone, Debug)]
pub struct StaticFleetState {
    snapshot: Arc<FleetSnapshot>,
}

impl StaticFleetState {
    /// A source that always reports `snapshot`.
    pub fn new(snapshot: FleetSnapshot) -> Self {
        Self {
            snapshot: Arc::new(snapshot),
        }
    }
}

impl FleetStateSource for StaticFleetState {
    fn snapshot(&self) -> Arc<FleetSnapshot> {
        Arc::clone(&self.snapshot)
    }
}

/// A mutable readiness source whose whole snapshot an external writer replaces
/// atomically between decisions.
///
/// The snapshot is stored behind an [`Arc`]; `set` swaps the `Arc` under one
/// write lock (a cheap pointer replacement), and `snapshot` returns that `Arc`
/// under a read lock, so a reader never observes a half-updated fleet and does
/// **not** copy the fleet per routing decision. This is the current factual state,
/// not session/history state.
#[derive(Clone, Debug)]
pub struct SharedFleetState {
    current: Arc<parking_lot::RwLock<Arc<FleetSnapshot>>>,
}

impl SharedFleetState {
    /// A source initially reporting `snapshot`.
    pub fn new(snapshot: FleetSnapshot) -> Self {
        Self {
            current: Arc::new(parking_lot::RwLock::new(Arc::new(snapshot))),
        }
    }

    /// Atomically replaces the whole fleet snapshot.
    pub fn set(&self, snapshot: FleetSnapshot) {
        *self.current.write() = Arc::new(snapshot);
    }
}

impl FleetStateSource for SharedFleetState {
    fn snapshot(&self) -> Arc<FleetSnapshot> {
        // One coherent snapshot: a single read lock yields the whole immutable
        // `Arc`, so the reader cannot observe a half-updated fleet, and sharing
        // it is a cheap `Arc` clone (no per-decision fleet copy).
        Arc::clone(&self.current.read())
    }
}

/// A history-free fleet router with injected readiness state.
///
/// Holds the static candidate profiles and an injected
/// [`FleetStateSource`][state source]; each `route` reads **one** coherent
/// [`FleetSnapshot`] and applies the capability + readiness filters and ranks
/// the survivors. For any single snapshot the decision is deterministic; across
/// decisions the result may change only when an external source replaces the
/// readiness snapshot. The router performs no network I/O, no lifecycle
/// transition, owns no routing history, and owns no session state.
///
/// [state source]: FleetStateSource
#[derive(Clone, Debug)]
pub struct FleetRouter {
    profiles: Vec<CandidateProfile>,
    /// Injected coherent readiness source (static or externally replaceable).
    state: Arc<dyn FleetStateSource>,
}

impl FleetRouter {
    /// Creates a router over the given static profiles and a fixed injected
    /// readiness state.
    ///
    /// This is a convenience that wraps `state` in a [`StaticFleetState`]; it
    /// preserves the S2-A ergonomics. The `state` slice is an injected factual
    /// snapshot keyed by target. Each state key should correspond to a profile
    /// target; a state key with no matching profile is rejected, and duplicate
    /// state keys are rejected. A profile with no state entry is treated as
    /// **fail-closed not ready**.
    ///
    /// For a router whose readiness can change at runtime without reconstruction,
    /// use [`FleetRouter::with_source`] with an externally replaceable source.
    ///
    /// # Errors
    ///
    /// Returns [`LibsyError::AlgorithmError`] when profiles is empty, when two
    /// profiles name the same target, when a state key has no matching profile,
    /// or when a state key repeats.
    pub fn new(
        profiles: Vec<CandidateProfile>,
        state: Vec<(ModelId, CandidateState)>,
    ) -> Result<Self> {
        if profiles.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "fleet router requires at least one candidate profile".to_string(),
            });
        }
        for (i, a) in profiles.iter().enumerate() {
            for b in profiles.iter().skip(i + 1) {
                if a.target == b.target {
                    return Err(LibsyError::AlgorithmError {
                        message: format!(
                            "fleet router profiles must not repeat target {:?}",
                            a.target
                        ),
                    });
                }
            }
        }
        // Validate the injected state: no duplicate keys, and each key must
        // correspond to a profile target.
        let snapshot =
            FleetSnapshot::new(state.clone()).map_err(|error| LibsyError::AlgorithmError {
                message: error.to_string(),
            })?;
        for (key, _) in &snapshot.states {
            if !profiles.iter().any(|p| &p.target == key) {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "fleet router state key {:?} has no matching candidate profile",
                        key
                    ),
                });
            }
        }
        Ok(Self {
            profiles,
            state: Arc::new(StaticFleetState::new(snapshot)),
        })
    }

    /// Creates a router over the given static profiles and an injected readiness
    /// source.
    ///
    /// The source is read once per decision ([`FleetStateSource::snapshot`]),
    /// yielding one coherent snapshot. `new` is a thin convenience over this
    /// with a [`StaticFleetState`]; pass a [`SharedFleetState`] here so an
    /// external writer can replace the whole snapshot while the same
    /// `FleetRouter` instance keeps routing.
    ///
    /// Unlike [`FleetRouter::new`], `with_source` validates **profiles only** —
    /// it cannot validate the keys of a future snapshot that a live source has
    /// not produced yet. The source's snapshots are therefore trusted for key
    /// shape; any snapshot key with no matching profile is simply never selected
    /// (fail-closed, as with a missing entry).
    ///
    /// # Errors
    ///
    /// Returns [`LibsyError::AlgorithmError`] when profiles is empty or two
    /// profiles name the same target.
    pub fn with_source(
        profiles: Vec<CandidateProfile>,
        state: Arc<dyn FleetStateSource>,
    ) -> Result<Self> {
        if profiles.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "fleet router requires at least one candidate profile".to_string(),
            });
        }
        for (i, a) in profiles.iter().enumerate() {
            for b in profiles.iter().skip(i + 1) {
                if a.target == b.target {
                    return Err(LibsyError::AlgorithmError {
                        message: format!(
                            "fleet router profiles must not repeat target {:?}",
                            a.target
                        ),
                    });
                }
            }
        }
        Ok(Self { profiles, state })
    }

    fn snapshot(&self) -> Arc<FleetSnapshot> {
        self.state.snapshot()
    }

    /// The stateless decision core: filter eligible candidates, then rank them
    /// and build `(selected, fallbacks)`.
    fn decide(&self, request: &switchyard_protocol::Request) -> Result<(ModelId, Vec<ModelId>)> {
        let require_tools = !request.llm_request.tools.is_empty();
        let require_reasoning = reasoning_requested(&request.llm_request.reasoning);
        // One request consumes exactly one coherent factual snapshot.
        let snapshot = self.snapshot();

        let mut eligible = Vec::new();
        for profile in &self.profiles {
            // Capability filter: a request that requires tools must only reach
            // candidates that advertise tool calling.
            if require_tools && !profile.tool_calling {
                continue;
            }
            // Capability filter: an explicit reasoning request must only reach
            // candidates that advertise reasoning.
            if require_reasoning && !profile.reasoning {
                continue;
            }
            // Readiness filter: only immediately-eligible (ready, no transition
            // required) candidates are selectable for an immediate request.
            if !snapshot.state_for(&profile.target).immediately_eligible() {
                continue;
            }
            eligible.push(profile);
        }
        // Deterministic preference: lower rank first; ties broken by target id
        // for a total order (still deterministic and stateless).
        eligible.sort_by(|a, b| {
            a.preference_rank
                .cmp(&b.preference_rank)
                .then_with(|| a.target.cmp(&b.target))
        });

        if eligible.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "fleet router found no immediately-eligible candidate for this request"
                    .to_string(),
            });
        }

        let mut ids = eligible.iter().map(|p| p.target.clone());
        let selected = ids.next().expect("non-empty eligible set");
        let fallbacks = ids.collect::<Vec<_>>();
        Ok((selected, fallbacks))
    }
}

/// Whether the request requires a reasoning-capable target, derived from the
/// normalized reasoning controls.
///
/// Reasoning is required only when the normalized `effort` is an actual
/// reasoning level. Semantics:
///
/// * no reasoning controls → `false`
/// * `effort = "none"` → `false` (an explicit non-thinking request)
/// * `effort` is any other string → `true` (a reasoning level such as
///   `low`/`medium`/`high`/`xhigh`/`max` or any provider value preserved in the
///   normalized field)
///
/// `reasoning.raw` is deliberately NOT consulted: it is a container that holds
/// whatever reasoning controls a provider/decoder preserved (the Responses
/// decoder clones the whole `reasoning` object, so `raw` can be present for a
/// deliberate `{"effort":"none"}` request). Presence of `raw` does not mean
/// reasoning is enabled, and there is no generic provider-agnostic shape that
/// reliably means "enabled" in this narrow PoC. Raw-based reasoning admission is
/// therefore **deferred** (not treated as a requirement here); scope reasoning
/// admission to the normalized `effort` field.
fn reasoning_requested(reasoning: &switchyard_protocol::ReasoningParams) -> bool {
    match reasoning.effort.as_deref() {
        None | Some("none") => false,
        Some(_) => true,
    }
}

#[async_trait::async_trait]
impl Algorithm for FleetRouter {
    fn name(&self) -> &str {
        "fleet_router"
    }

    async fn route(
        self: Arc<Self>,
        _driver: Driver,
        request: switchyard_protocol::Request,
    ) -> Result<RoutingOutcome> {
        let (selected, fallbacks) = self.decide(&request)?;
        // DEBUG, not INFO: the selected/fallback target identifiers reveal
        // deployment topology and are routing internals, not user-facing facts.
        tracing::debug!(
            selected = %selected,
            fallbacks = ?fallbacks,
            "fleet_router selected target"
        );
        Ok(RoutingOutcome::route_to(selected, fallbacks, request))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateProfile, CandidateState, FleetRouter, FleetSnapshot, FleetStateSource,
        SharedFleetState,
    };
    use crate::Algorithm;
    use std::sync::Arc;
    use switchyard_protocol::{ModelId, ReasoningParams, Request, text_request};

    fn text_req() -> Request {
        Request {
            llm_request: text_request(None, "hello"),
            raw_request: None,
            metadata: None,
        }
    }

    fn tool_req() -> Request {
        let mut req = text_req();
        req.llm_request
            .tools
            .push(switchyard_protocol::ToolDefinition {
                name: "lookup".to_string(),
                description: Some("test tool".to_string()),
                parameters: serde_json::json!({ "type": "object" }),
                strict: None,
            });
        req
    }

    fn reasoning_req() -> Request {
        let mut req = text_req();
        req.llm_request.reasoning = ReasoningParams {
            effort: Some("high".to_string()),
            raw: None,
        };
        req
    }

    /// An explicit non-thinking request via `effort = "none"` (normalized path).
    fn none_reasoning_req() -> Request {
        let mut req = text_req();
        req.llm_request.reasoning = ReasoningParams {
            effort: Some("none".to_string()),
            raw: None,
        };
        req
    }

    /// The Responses-decoder shape that regressed: `effort = "none"` AND `raw`
    /// carrying the whole reasoning object (including `{"effort":"none"}`).
    /// The decoder clones `body.reasoning` into `raw`, so this is what a
    /// deliberate non-thinking Responses request decodes to.
    fn responses_none_reasoning_req() -> Request {
        let mut req = text_req();
        req.llm_request.reasoning = ReasoningParams {
            effort: Some("none".to_string()),
            raw: Some(serde_json::json!({ "effort": "none" })),
        };
        req
    }

    fn profiles() -> Vec<CandidateProfile> {
        vec![
            CandidateProfile::new("luna", true, true, 1),
            CandidateProfile::new("deepseek-flash", true, true, 2),
            CandidateProfile::new("htpc-qwen3_5", false, false, 3),
            CandidateProfile::new("comfyninja-qwen3_8", true, true, 4),
        ]
    }

    fn all_ready() -> Vec<(ModelId, CandidateState)> {
        vec![
            (ModelId::from("luna"), CandidateState::ready()),
            (ModelId::from("deepseek-flash"), CandidateState::ready()),
            (ModelId::from("htpc-qwen3_5"), CandidateState::ready()),
            (
                ModelId::from("comfyninja-qwen3_8"),
                CandidateState::transition_required(),
            ),
        ]
    }

    #[test]
    fn simple_chooses_top_preference_and_orders_fallbacks() {
        // A: simple text request, ready set. Winner = luna (rank 1); fallbacks
        // ordered by rank among ready, Comfy transition-excluded.
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (selected, fallbacks) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        let fb = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert_eq!(fb, ["deepseek-flash", "htpc-qwen3_5"]);
    }

    #[test]
    fn tool_required_excludes_htpc() {
        // B: tool-required request. HTPC advertises tool_calling=false, so it is
        // excluded from both selected and fallbacks.
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (selected, fallbacks) = router.decide(&tool_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        let fb = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert!(!fb.contains(&"htpc-qwen3_5".to_string()));
        assert!(fb.contains(&"deepseek-flash".to_string()));
    }

    #[test]
    fn reasoning_required_excludes_non_reasoning() {
        // C: reasoning-required request. HTPC advertises reasoning=false, so it
        // is excluded.
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (selected, fallbacks) = router.decide(&reasoning_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        let fb = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert!(!fb.contains(&"htpc-qwen3_5".to_string()));
    }

    #[test]
    fn transition_required_comfy_is_never_selected_or_fallback() {
        // D: Comfy Qwen is transition_required; never selected, never an
        // immediate fallback, and no transition occurs.
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (selected, fallbacks) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        let fb = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert!(!fb.contains(&"comfyninja-qwen3_8".to_string()));
        assert!(fb.contains(&"htpc-qwen3_5".to_string()));
    }

    #[test]
    fn unready_candidate_excluded() {
        // E: a ready=false candidate (HTPC) is excluded.
        let states = vec![
            (ModelId::from("luna"), CandidateState::ready()),
            (ModelId::from("deepseek-flash"), CandidateState::ready()),
            (ModelId::from("htpc-qwen3_5"), CandidateState::not_ready()),
            (
                ModelId::from("comfyninja-qwen3_8"),
                CandidateState::transition_required(),
            ),
        ];
        let router = FleetRouter::new(profiles(), states).unwrap();
        let (selected, fallbacks) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        let fb = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert!(!fb.contains(&"htpc-qwen3_5".to_string()));
        assert!(fb.contains(&"deepseek-flash".to_string()));
    }

    #[test]
    fn no_eligible_fails_closed() {
        // F: no candidate immediately eligible => fail closed with a clear
        // AlgorithmError (never silently pick an incapable/unready target).
        let states = vec![
            (ModelId::from("luna"), CandidateState::not_ready()),
            (ModelId::from("deepseek-flash"), CandidateState::not_ready()),
            (ModelId::from("htpc-qwen3_5"), CandidateState::not_ready()),
            (
                ModelId::from("comfyninja-qwen3_8"),
                CandidateState::transition_required(),
            ),
        ];
        let router = FleetRouter::new(profiles(), states).unwrap();
        let err = router.decide(&text_req()).unwrap_err();
        assert!(
            err.to_string()
                .contains("no immediately-eligible candidate")
        );
    }

    #[test]
    fn fallback_purity() {
        // G: every fallback satisfies the same filters as the selected target
        // (tool-required => all fallbacks advertise tools and are ready).
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (_, fallbacks) = router.decide(&tool_req()).unwrap();
        let fb = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert!(!fb.contains(&"htpc-qwen3_5".to_string()));
        assert!(!fb.contains(&"comfyninja-qwen3_8".to_string()));
    }

    #[test]
    fn stateless_repeatability() {
        // H: same request + profiles + snapshot always yields the same decision
        // (no history/cursor/session state).
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let first = router.decide(&text_req()).unwrap();
        for _ in 0..25 {
            assert_eq!(router.decide(&text_req()).unwrap(), first);
        }
    }

    #[test]
    fn contradictory_state_transition_takes_precedence() {
        // q-2: even if a caller constructs {ready:true, transition_required:true},
        // the transition-required fact wins (never selected for an immediate
        // request).
        let states = vec![
            (ModelId::from("luna"), CandidateState::ready()),
            (
                ModelId::from("comfyninja-qwen3_8"),
                CandidateState {
                    ready: true,
                    transition_required: true,
                },
            ),
        ];
        let profiles_v = vec![
            CandidateProfile::new("luna", true, true, 1),
            CandidateProfile::new("comfyninja-qwen3_8", true, true, 2),
        ];
        let router = FleetRouter::new(profiles_v, states).unwrap();
        let (selected, back) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        assert!(
            back.is_empty(),
            "transition-required Comfy must not be an immediate fallback"
        );
    }

    #[test]
    fn rejects_unknown_or_duplicate_state_keys() {
        // q-3/q-5: state keys that do not match a profile, or that repeat, are
        // rejected at construction.
        let states_unknown = vec![(ModelId::from("does-not-exist"), CandidateState::ready())];
        assert!(
            FleetRouter::new(profiles(), states_unknown).is_err(),
            "state key with no matching profile must be rejected"
        );
        let states_dup = vec![
            (ModelId::from("luna"), CandidateState::ready()),
            (ModelId::from("luna"), CandidateState::not_ready()),
        ];
        assert!(
            FleetRouter::new(profiles(), states_dup).is_err(),
            "duplicate state key must be rejected"
        );
    }

    // ─── Reasoning-mode semantics (review correction) ───────────────────────

    /// The key invariant from the review: an explicit non-thinking request
    /// (`effort = "none"`, with or without `raw`) is NOT a reasoning requirement.
    /// HTPC (reasoning=false) must stay eligible.
    ///
    /// This covers the discovered bug: the Responses decoder preserves the whole
    /// `reasoning` object into `ReasoningParams.raw`, so `{"effort":"none"}`
    /// decodes to `effort="none"` + `raw=Some(reasoning)`. A predicate of
    /// `effort.is_some() || raw.is_some()` would wrongly require reasoning.
    #[test]
    fn effort_none_is_not_a_reasoning_requirement() {
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();

        // effort = "none", raw = None: HTPC stays eligible.
        let (_, fb) = router.decide(&none_reasoning_req()).unwrap();
        let fb = fb.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(fb.contains(&"htpc-qwen3_5".to_string()));

        // Responses-style: effort = "none" + raw = Some(reasoning object).
        // HTPC still eligible (non-thinking != reasoning-required).
        let (_, fb) = router.decide(&responses_none_reasoning_req()).unwrap();
        let fb = fb.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(fb.contains(&"htpc-qwen3_5".to_string()));
    }

    /// Reasoning absent: non-reasoning candidate remains eligible.
    #[test]
    fn reasoning_absent_keeps_non_reasoning_candidate_eligible() {
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (_, fb) = router.decide(&text_req()).unwrap(); // no reasoning controls
        let fb = fb.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(fb.contains(&"htpc-qwen3_5".to_string()));
    }

    /// Positive reasoning level (effort = "high"): non-reasoning candidate excluded.
    #[test]
    fn positive_reasoning_excludes_non_reasoning_candidate() {
        let router = FleetRouter::new(profiles(), all_ready()).unwrap();
        let (_, fb) = router.decide(&reasoning_req()).unwrap();
        let fb = fb.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(
            !fb.contains(&"htpc-qwen3_5".to_string()),
            "a high-effort reasoning request must exclude the non-reasoning candidate"
        );
    }

    // ─── Public Algorithm-path proof (review correction) ────────────────────

    /// Drives a real `FleetRouter` through the public libsy Algorithm path
    /// (`run_stream` → `route` → `RoutingOutcome`) via the exported `drive`, and
    /// asserts the native selected + fallback contract.
    #[tokio::test]
    async fn public_algorithm_path_produces_native_routing_outcome() {
        use crate::{CallModel, drive};
        use std::sync::Arc as StdArc;

        let router: StdArc<dyn Algorithm> =
            StdArc::new(FleetRouter::new(profiles(), all_ready()).unwrap());

        let outcome = drive(
            StdArc::clone(&router),
            text_req(),
            // FleetRouter makes no offloaded calls, so serve is never invoked;
            // provide a stub satisfying the drive contract.
            |_call: CallModel| async { Ok(()) },
        )
        .await
        .unwrap();

        assert_eq!(outcome.selected_model_id.to_string(), "luna");
        let fb = outcome
            .fallback_models
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert_eq!(fb, ["deepseek-flash", "htpc-qwen3_5"]);
        // The selected model is stamped into the request, as upstream requires.
        assert_eq!(outcome.request.llm_request.model.as_deref(), Some("luna"));
    }

    /// The no-eligible fail-closed error propagates through the public path.
    #[tokio::test]
    async fn public_algorithm_path_propagates_fail_closed() {
        use crate::{CallModel, drive};

        let states = vec![
            (ModelId::from("luna"), CandidateState::not_ready()),
            (ModelId::from("htpc-qwen3_5"), CandidateState::not_ready()),
            (
                ModelId::from("comfyninja-qwen3_8"),
                CandidateState::transition_required(),
            ),
        ];
        let router: std::sync::Arc<dyn Algorithm> =
            std::sync::Arc::new(FleetRouter::new(profiles(), states).unwrap());
        let err = match drive(router, text_req(), |_call: CallModel| async { Ok(()) }).await {
            Ok(_) => panic!("expected fail-closed error, got a RoutingOutcome"),
            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("no immediately-eligible candidate")
        );
    }

    // ─── S2-B: dynamic fleet-state snapshot seam ────────────────────────────

    /// A helper that turns a small `(&str, bool, bool)` list into a coherent
    /// `FleetSnapshot` for the S2-B live-update proof.
    fn snapshot(entries: &[(&str, bool, bool)]) -> FleetSnapshot {
        FleetSnapshot::new(
            entries
                .iter()
                .map(|(t, ready, transition)| {
                    (
                        ModelId::from(*t),
                        CandidateState {
                            ready: *ready,
                            transition_required: *transition,
                        },
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    /// The central S2-B proof: the SAME FleetRouter instance consumes one
    /// coherent externally-replaceable snapshot per decision and reacts correctly
    /// to whole-snapshot replacement (cases A → B → C), without reconstruction.
    #[test]
    fn same_router_reacts_to_atomically_replaced_snapshot() {
        let shared = SharedFleetState::new(snapshot(&[
            ("luna", true, false),
            ("deepseek-flash", true, false),
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true), // transition-required
        ]));
        // One router instance over the mutable source; never reconstructed.
        let router = FleetRouter::with_source(profiles(), Arc::new(shared.clone())).unwrap();

        // Case A — initial snapshot: luna (rank 1) wins, HTPC in fallbacks.
        let (selected, fb) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "luna");
        let fb = fb.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(fb, ["deepseek-flash", "htpc-qwen3_5"]);

        // Case B — replace the whole snapshot on the SAME source/instance:
        // luna goes not_ready, so deepseek-flash wins.
        shared.set(snapshot(&[
            ("luna", false, false), // not ready now
            ("deepseek-flash", true, false),
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true),
        ]));
        let (selected, fb) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "deepseek-flash");
        let fb = fb.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(fb, ["htpc-qwen3_5"]);

        // Case C — replace the snapshot again predictably: HTPC only ready.
        shared.set(snapshot(&[
            ("luna", false, false),
            ("deepseek-flash", false, false),
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true),
        ]));
        let (selected, fb) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "htpc-qwen3_5");
        assert!(fb.is_empty());

        // The SAME router instance was used for all three decisions.
    }

    /// Missing snapshot state for a target is fail-closed (not immediately
    /// eligible), and a snapshot with no eligible target fails closed.
    #[test]
    fn missing_state_is_fail_closed() {
        // Only HTPC and Comfy are in the snapshot; luna/deepseek are ABSENT (no
        // entry), so they must NOT be selectable.
        let shared = SharedFleetState::new(snapshot(&[
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true),
        ]));
        let router = FleetRouter::with_source(profiles(), Arc::new(shared)).unwrap();
        let (selected, fb) = router.decide(&text_req()).unwrap();
        // luna and deepseek are absent => fail-closed not ready.
        assert_eq!(selected.to_string(), "htpc-qwen3_5");
        assert!(fb.is_empty());

        // A snapshot where every profiled target is absent/not ready => AlgorithmError.
        let empty = SharedFleetState::new(snapshot(&[]));
        let router = FleetRouter::with_source(profiles(), Arc::new(empty)).unwrap();
        let err = router.decide(&text_req()).unwrap_err();
        assert!(
            err.to_string()
                .contains("no immediately-eligible candidate")
        );
    }

    /// Coherency: even under concurrent whole-snapshot replacement, a reader
    /// only ever observes one of the two complete valid snapshots — never a
    /// mixed generation. The `SharedFleetState` swaps a whole `Arc<FleetSnapshot>`
    /// under a single lock, so this is a property of the implementation.
    #[test]
    fn concurrent_replace_never_exposes_mixed_generation() {
        let shared = std::sync::Arc::new(SharedFleetState::new(snapshot(&[
            ("luna", true, false),
            ("deepseek-flash", true, false),
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true),
        ])));
        let state_a = snapshot(&[
            ("luna", true, false),
            ("deepseek-flash", true, false),
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true),
        ]);
        let state_b = snapshot(&[
            ("luna", false, false),
            ("deepseek-flash", true, false),
            ("htpc-qwen3_5", true, false),
            ("comfyninja-qwen3_8", false, true),
        ]);

        // Writer toggles the whole snapshot between two complete valid states.
        // Clone the states and the Arc into the writer so the originals remain
        // usable by the reader threads below.
        let writer_a = state_a.clone();
        let writer_b = state_b.clone();
        let writer_shared = std::sync::Arc::clone(&shared);
        let writer = std::thread::spawn(move || {
            for i in 0..200 {
                writer_shared.set(if i % 2 == 0 {
                    writer_a.clone()
                } else {
                    writer_b.clone()
                });
            }
        });

        // Readers observe only complete snapshots equal to one of the two valid
        // states — an Arc swap can never yield a half-updated fleet.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let shared = std::sync::Arc::clone(&shared);
            let a = state_a.clone();
            let b = state_b.clone();
            handles.push(std::thread::spawn(move || {
                let mut saw_valid = false;
                for _ in 0..500 {
                    let snap = shared.snapshot();
                    // The coherency guarantee: every observed snapshot is one
                    // complete valid state — never a mixed generation.
                    assert!(
                        *snap == a || *snap == b,
                        "reader observed a snapshot that is neither complete valid state"
                    );
                    saw_valid = true;
                }
                saw_valid
            }));
        }
        writer.join().unwrap();
        let saw = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>();
        // Every reader successfully observed only coherent snapshots.
        assert!(saw.iter().all(|v| *v));
    }
}
