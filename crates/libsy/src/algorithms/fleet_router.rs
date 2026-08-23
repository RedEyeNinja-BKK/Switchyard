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
//! Context-fit admission is applied as a pure eligibility filter (see
//! [`ContextAdmissionPolicy`]); the Algorithm performs no token counting and no
//! network/provider I/O. LLM classifiers and any real fact producer are
//! explicitly out of scope for this core proof.

use std::sync::Arc;

use switchyard_protocol::{ContentBlock, ModelId};

use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result, RoutingOutcome};

/// How FleetRouter treats a candidate's context capacity for preflight admission.
///
/// This is static candidate configuration (part of the profile), not a live fact
/// and not a request-derived count. It deliberately distinguishes *unmanaged*
/// (no preflight context assertion) from *bounded* (requires a trusted
/// candidate-specific input-token fact and an explicit output budget before
/// admission) so that `None` never ambiguously means "unknown / unlimited /
/// legacy".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextAdmissionPolicy {
    /// FleetRouter does not assert preflight context fit for this candidate.
    ///
    /// The candidate remains governed by capability, readiness, and normal
    /// runtime/provider behavior (including the native `ContextWindowExceeded`
    /// fallback). This is the initial state for cloud candidates and for any
    /// candidate config that does not opt in to context admission.
    Unmanaged,
    /// FleetRouter requires a trusted candidate-specific input-token fact and an
    /// explicit output budget before admitting this candidate.
    ///
    /// `usable_context_tokens` is the qualified usable context this deployment
    /// guarantees for the candidate; it is not the theoretical/provider maximum
    /// or route advertisement.
    Bounded {
        /// The qualified usable context this deployment guarantees this candidate
        /// can serve.
        usable_context_tokens: u64,
    },
}

/// Declared work shape of a request, used as a structural eligibility signal.
///
/// This is **declared** in structured request metadata (never inferred from
/// prompt text). It is the smallest generic mechanism FleetRouter uses to keep a
/// bounded-only candidate (e.g. a premium Luna NT lane) off an **agentic**
/// request even when both are non-thinking and the Luna resource is healthy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkShape {
    /// A finite, self-contained task (e.g. a bounded tool call / single response).
    Bounded,
    /// A long-lived, evolving / multi-step agentic session.
    Agentic,
}

/// How FleetRouter learns a request's declared work shape (if at all).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkShapeSource {
    /// No work-shape filtering: every candidate is eligible regardless of shape.
    #[default]
    None,
    /// Read `work_shape` from structured request metadata (extensions) and admit
    /// only candidates whose declared `work_shape` matches (or serves any).
    Request,
}

/// Static capability + preference profile for one candidate target.
///
/// This is Algorithm configuration, not a live fact. `tool_calling` and
/// `reasoning` describe what the target advertises it can do; `preference_rank`
/// is a deterministic deployment ordering (lower = more preferred);
/// `context_policy` says whether and how preflight context admission applies;
/// `usable_context_tokens` is the candidate's STATIC qualified usable context
/// capacity (a configured capability fact, distinct from the exact-request
/// admission policy); `work_shape` (when set) limits this candidate to a
/// declared work shape under a request-sourced work-shape dispatcher.
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
    /// Preflight context admission policy for this candidate.
    pub context_policy: ContextAdmissionPolicy,
    /// Static qualified usable context capacity of this candidate.
    ///
    /// `Some(n)` = this deployment truthfully guarantees/configured the
    /// candidate to serve `n` usable context tokens. `None` = capacity
    /// UNKNOWN. `Some(0)` is invalid configuration (rejected at construction).
    /// This is a capability fact for minimum-context eligibility — it is NOT
    /// the exact-request admission policy, and a candidate may be `Unmanaged`
    /// for exact preflight counting while still carrying a known capacity.
    /// Never inferred from provider/model names.
    pub usable_context_tokens: Option<u64>,
    /// Optional declared work-shape limitation. `None` serves any shape.
    pub work_shape: Option<WorkShape>,
    /// Whether the target truthfully advertises vision (image input) support.
    ///
    /// A candidate with `supports_vision=false` is NEVER selected for a request
    /// that carries image content (fail closed on unknown/undeclared). This is
    /// a static capability fact, separate from readiness and never inferred
    /// from provider/model names.
    pub supports_vision: bool,
}

impl CandidateProfile {
    /// Creates a candidate profile for `target` with no preflight context concern
    /// ([`ContextAdmissionPolicy::Unmanaged`]), UNKNOWN static context capacity,
    /// and no work-shape limitation.
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
            context_policy: ContextAdmissionPolicy::Unmanaged,
            usable_context_tokens: None,
            work_shape: None,
            supports_vision: false,
        }
    }

    /// Sets whether the candidate advertises vision (image input) support
    /// (builder style). Default `false` - undeclared candidates never receive
    /// image-bearing requests (fail closed).
    pub fn with_supports_vision(mut self, supports_vision: bool) -> Self {
        self.supports_vision = supports_vision;
        self
    }

    /// Sets the candidate's preflight context admission policy (builder style).
    pub fn with_context_policy(mut self, policy: ContextAdmissionPolicy) -> Self {
        self.context_policy = policy;
        self
    }

    /// Sets the candidate's static qualified usable context capacity (builder
    /// style). A zero capacity is rejected when the router is constructed. For
    /// a Bounded candidate, an explicitly present capacity must equal the
    /// Bounded policy capacity (mismatch is a construction error).
    pub fn with_usable_context_tokens(mut self, capacity: u64) -> Self {
        self.usable_context_tokens = Some(capacity);
        self
    }

    /// Limits this candidate to a declared work shape (builder style).
    pub fn with_work_shape(mut self, shape: WorkShape) -> Self {
        self.work_shape = Some(shape);
        self
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

/// Shared candidate-profile validation for [`FleetRouter::new`] and
/// [`FleetRouter::with_source`].
///
/// Enforces the invariants that apply to every candidate profile regardless of
/// whether it came from server TOML, tests, another Rust host, or a future
/// integration:
/// - at least one candidate profile is required;
/// - no two profiles name the same target;
/// - a `Bounded` context policy must have a **positive** usable context capacity
///   (a zero-capacity `Bounded` candidate is an invalid context-admission
///   contract and is rejected at construction).
///
/// `Unmanaged` is always valid.
fn validate_profiles(profiles: &[CandidateProfile]) -> Result<()> {
    if profiles.is_empty() {
        return Err(LibsyError::AlgorithmError {
            message: "fleet router requires at least one candidate profile".to_string(),
        });
    }
    for (i, a) in profiles.iter().enumerate() {
        if let ContextAdmissionPolicy::Bounded {
            usable_context_tokens,
        } = a.context_policy
        {
            if usable_context_tokens == 0 {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "fleet router candidate {:?} has an invalid zero usable_context_tokens; \
                         a bounded context policy must declare a positive usable capacity",
                        a.target
                    ),
                });
            }
            // Capacity-coherence invariant (FLEET-1 review): a Bounded candidate's
            // static usable_context_tokens, when explicitly present, MUST equal the
            // Bounded policy capacity. Both represent the same qualified usable
            // capacity; a mismatch is a configuration/construction error, never
            // silently resolved. When the static field is absent, the Bounded
            // policy capacity is the effective known static capacity for
            // min_context_tokens eligibility (min_context_eligible reads it via
            // the fallback below).
            if let Some(static_cap) = a.usable_context_tokens
                && static_cap != usable_context_tokens
            {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "fleet router candidate {:?} has a static usable_context_tokens \
                         ({static_cap}) that disagrees with its Bounded context-policy \
                         capacity ({usable_context_tokens}); they must be equal",
                        a.target
                    ),
                });
            }
        }
        if a.usable_context_tokens == Some(0) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "fleet router candidate {:?} has an invalid zero static \
                     usable_context_tokens; a configured capacity must be positive",
                    a.target
                ),
            });
        }
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
    Ok(())
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
    /// How the router learns a request's declared work shape (default: none).
    work_shape_source: WorkShapeSource,
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
        validate_profiles(&profiles)?;
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
            work_shape_source: WorkShapeSource::None,
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
        validate_profiles(&profiles)?;
        Ok(Self {
            profiles,
            state,
            work_shape_source: WorkShapeSource::None,
        })
    }

    /// Requests that this router read the declared `work_shape` from structured
    /// request metadata and apply candidate `work_shape` eligibility. Building
    /// block for the dynamic (request-sourced) smart routes.
    pub fn with_request_work_shape(mut self) -> Self {
        self.work_shape_source = WorkShapeSource::Request;
        self
    }

    /// Resolves the request's declared work shape from structured metadata
    /// (`work_shape` field in request extensions, or the LEGACY `work_class`
    /// translated), or `None` when no shape is declared. NEVER inferred from
    /// prompt text.
    ///
    /// Legacy `work_class` compatibility (the old ResourceRouter universal smart
    /// contract): `bounded` -> Bounded, `agentic` -> Agentic, `reasoning` ->
    /// (Bounded, deliberate) — the deliberate intent is surfaced separately via
    /// [`Self::declared_legacy_reasoning`] so a thinking candidate is required.
    fn declared_work_shape(&self, request: &switchyard_protocol::Request) -> Option<WorkShape> {
        if self.work_shape_source != WorkShapeSource::Request {
            return None;
        }
        let fields = &request.llm_request.extensions.fields;
        if let Some(shape) = fields
            .get("work_shape")
            .and_then(serde_json::Value::as_str)
            .and_then(|s| match s.to_ascii_lowercase().as_str() {
                "bounded" => Some(WorkShape::Bounded),
                "agentic" => Some(WorkShape::Agentic),
                _ => None,
            })
        {
            return Some(shape);
        }
        // Legacy fallback: work_class=bounded|agentic translate to a shape.
        match fields
            .get("work_class")
            .and_then(serde_json::Value::as_str)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("bounded") => Some(WorkShape::Bounded),
            Some("agentic") => Some(WorkShape::Agentic),
            // work_class=reasoning -> (bounded, deliberate): the deliberate intent
            // is handled by declared_legacy_reasoning, not a shape exclusion.
            Some("reasoning") => None,
            _ => None,
        }
    }

    /// Whether the LEGACY `work_class=reasoning` contract is declared, which
    /// requires a thinking candidate (deliberate intent) regardless of shape.
    fn declared_legacy_reasoning(&self, request: &switchyard_protocol::Request) -> bool {
        self.work_shape_source == WorkShapeSource::Request
            && request
                .llm_request
                .extensions
                .fields
                .get("work_class")
                .and_then(serde_json::Value::as_str)
                .map(|s| s.eq_ignore_ascii_case("reasoning"))
                .unwrap_or(false)
    }

    /// Declared task minimum usable-context requirement from structured request
    /// metadata (the same normalized request-extension mechanism used for
    /// `work_shape`). NEVER inferred from prompt text.
    ///
    /// Absent => `Ok(None)` (no minimum-context requirement; zero regression).
    /// Present => must be a **positive** integer (u64, no overflow). Zero,
    /// negative, non-integer, or malformed => `Err` (hard request/algorithm
    /// failure, never silently ignored).
    fn declared_min_context_tokens(
        &self,
        request: &switchyard_protocol::Request,
    ) -> Result<Option<u64>> {
        let Some(value) = request
            .llm_request
            .extensions
            .fields
            .get("min_context_tokens")
        else {
            return Ok(None);
        };
        let Some(n) = value.as_u64() else {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "declared min_context_tokens must be a positive integer, got {value}"
                ),
            });
        };
        if n == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "declared min_context_tokens must be positive (zero is invalid)"
                    .to_string(),
            });
        }
        Ok(Some(n))
    }

    /// Minimum-context eligibility for one candidate under the declared task
    /// requirement.
    ///
    /// No declared requirement => every candidate eligible. A declared minimum
    /// requires the candidate's STATIC qualified usable context capacity (a
    /// configured capability fact, separate from the exact-request admission
    /// policy) to be known and >= the requirement. Unknown capacity + declared
    /// requirement => excluded (fail closed). Unknown capacity + no requirement
    /// => existing behavior preserved. Once the minimum is satisfied, context
    /// size NEVER affects ranking — this is admission only.
    fn min_context_eligible(profile: &CandidateProfile, declared: Option<u64>) -> bool {
        let Some(min) = declared else {
            return true;
        };
        // Effective known static capacity: the explicit static field when
        // present; for a Bounded candidate WITHOUT an explicit static field, the
        // Bounded policy capacity is the effective known static capacity (the
        // coherence invariant guarantees they would be equal if both present).
        match profile.usable_context_tokens {
            Some(cap) => cap >= min,
            None => match profile.context_policy {
                ContextAdmissionPolicy::Bounded {
                    usable_context_tokens,
                } => usable_context_tokens >= min,
                ContextAdmissionPolicy::Unmanaged => false,
            },
        }
    }

    /// Work-shape eligibility for one candidate under the declared request shape.
    fn work_shape_eligible(profile: &CandidateProfile, declared: Option<WorkShape>) -> bool {
        match declared {
            // No work-shape filtering / no declared shape: every candidate is eligible.
            None => true,
            // A declared shape excludes a candidate limited to the opposite shape;
            // a candidate that serves any shape (None) stays eligible.
            Some(shape) => profile.work_shape.is_none_or(|s| s == shape),
        }
    }

    fn snapshot(&self) -> Arc<FleetSnapshot> {
        self.state.snapshot()
    }

    /// The stateless decision core: filter eligible candidates, then rank them
    /// and build `(selected, fallbacks)`.
    fn decide(&self, request: &switchyard_protocol::Request) -> Result<(ModelId, Vec<ModelId>)> {
        let require_tools = !request.llm_request.tools.is_empty();
        // Multimodal capability filter: a request that carries image content
        // MUST only reach candidates that truthfully advertise vision. This is
        // derived from the normalized content blocks (never from prompt text).
        let require_vision = request_requires_vision(request);
        // Reasoning is required either by an explicit reasoning effort or by the
        // LEGACY `work_class=reasoning` contract (deliberate intent).
        let require_reasoning = reasoning_requested(&request.llm_request.reasoning)
            || self.declared_legacy_reasoning(request);
        let declared_shape = self.declared_work_shape(request);
        let max_output = request.llm_request.output.max_output_tokens;
        // Declared task minimum usable-context requirement. Malformed/zero is a
        // HARD request failure (never silently ignored); absent is no requirement.
        let min_context = self.declared_min_context_tokens(request)?;
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
            // Vision filter: an image-bearing request must only reach candidates
            // that advertise vision; an undeclared/non-vision candidate is
            // excluded (fail closed, never assumed capable).
            if require_vision && !profile.supports_vision {
                continue;
            }
            // Work-shape filter (request-sourced dispatcher): a candidate limited
            // to a different declared work shape is excluded. A candidate that
            // serves any shape (None) remains eligible.
            if !Self::work_shape_eligible(profile, declared_shape) {
                continue;
            }
            // Minimum-context filter (declared task requirement): a candidate with
            // known static usable context capacity >= the minimum may remain
            // eligible; UNKNOWN capacity + declared requirement => excluded
            // (fail closed). Admission only — never a ranking signal.
            if !Self::min_context_eligible(profile, min_context) {
                continue;
            }
            // Readiness filter: only immediately-eligible (ready, no transition
            // required) candidates are selectable for an immediate request.
            if !snapshot.state_for(&profile.target).immediately_eligible() {
                continue;
            }
            // Context-fit filter (pure admission, no counting, no I/O): a
            // BOUNDED candidate is eligible only when the request carries a
            // trusted candidate-specific input-token fact AND an explicit output
            // budget, and input + output fits its qualified usable context with
            // checked arithmetic. Unknown/overflow => excluded (fail closed).
            if !self.context_fits(
                &profile.context_policy,
                &profile.target,
                request,
                max_output,
            ) {
                continue;
            }
            eligible.push(profile);
        }
        // Deterministic preference: lower rank first; ties broken by target id
        // for a total order (still deterministic and stateless). Context size is
        // never used as a ranking signal — it is admission only.
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

    /// Pure preflight context-fit admission for one candidate.
    ///
    /// `Unmanaged` candidates always pass. A `Bounded { usable_context_tokens }`
    /// candidate requires a **candidate-specific** exact input-token fact from the
    /// request (keyed by its own target; never borrowed from another candidate)
    /// and an explicit output budget; it admits only when
    /// `input + output <= usable_context_tokens` under checked arithmetic.
    /// Missing input fact, missing output budget, or arithmetic overflow all fail
    /// closed to non-admission.
    fn context_fits(
        &self,
        policy: &ContextAdmissionPolicy,
        target: &ModelId,
        request: &switchyard_protocol::Request,
        max_output: Option<u64>,
    ) -> bool {
        let ContextAdmissionPolicy::Bounded {
            usable_context_tokens,
        } = policy
        else {
            return true;
        };
        let Some(input) = request.candidate_input_tokens.get(target) else {
            // No trusted candidate-specific input count => unknown => excluded.
            return false;
        };
        let Some(output) = max_output else {
            // Missing explicit output budget => cannot be fully proven => excluded.
            return false;
        };
        match input.checked_add(output) {
            Some(fit) if fit <= *usable_context_tokens => true,
            // Overflow or does-not-fit => fail closed.
            _ => false,
        }
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
/// True when the normalized request carries any image content block.
///
/// Derived from the typed conversation representation (ContentBlock::Image),
/// never from prompt text or model names. A request with an image anywhere in
/// its messages requires a vision-capable candidate.
fn request_requires_vision(request: &switchyard_protocol::Request) -> bool {
    request
        .llm_request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .any(|block| matches!(block, ContentBlock::Image { .. }))
}

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
        SharedFleetState, WorkShape,
    };
    use crate::Algorithm;
    use std::sync::Arc;
    use switchyard_protocol::{ModelId, ReasoningParams, Request, text_request};

    // ------------------------------------------------------------------
    // Vision capability filter (multimodal reconciliation 2026-08-23)
    // ------------------------------------------------------------------

    fn vision_req() -> Request {
        use switchyard_protocol::{ContentBlock, Message, Role};
        let mut req = text_req();
        req.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "what is in this image".into(),
                },
                ContentBlock::Image {
                    source: switchyard_protocol::ImageSource::Url {
                        url: "https://example.com/x.png".into(),
                        detail: None,
                    },
                },
            ],
        });
        req
    }

    #[test]
    fn vision_required_excludes_non_vision_candidates() {
        // Image-bearing request must NOT select a text-only candidate.
        let router = FleetRouter::new(
            vec![
                CandidateProfile::new("text-only", true, true, 1), // supports_vision=false
                CandidateProfile::new("vision", true, true, 2).with_supports_vision(true),
            ],
            vec![
                (ModelId::from("text-only"), CandidateState::ready()),
                (ModelId::from("vision"), CandidateState::ready()),
            ],
        )
        .unwrap();
        let (selected, _) = router.decide(&vision_req()).unwrap();
        assert_eq!(selected.to_string(), "vision");
    }

    #[test]
    fn vision_required_fails_closed_when_only_text_candidates() {
        // No vision-capable candidate => no eligible candidate (fail closed).
        let router = FleetRouter::new(
            vec![CandidateProfile::new("text-only", true, true, 1)],
            vec![(ModelId::from("text-only"), CandidateState::ready())],
        )
        .unwrap();
        let err = router.decide(&vision_req()).unwrap_err();
        assert!(err.to_string().contains("no immediately-eligible candidate"));
    }

    #[test]
    fn text_only_request_unaffected_by_vision_filter() {
        // Text request still selects the text-only candidate (rank 1).
        let router = FleetRouter::new(
            vec![
                CandidateProfile::new("text-only", true, true, 1),
                CandidateProfile::new("vision", true, true, 2).with_supports_vision(true),
            ],
            vec![
                (ModelId::from("text-only"), CandidateState::ready()),
                (ModelId::from("vision"), CandidateState::ready()),
            ],
        )
        .unwrap();
        let (selected, _) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "text-only");
    }

    fn text_req() -> Request {
        Request {
            llm_request: text_request(None, "hello"),
            raw_request: None,
            metadata: None,
            ..Request::default()
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

    // --- S2-E.1: pure context-fit admission -----------------------------------

    use super::ContextAdmissionPolicy;
    use std::collections::BTreeMap;

    /// A request with explicit output budget and candidate input-token facts.
    fn ctx_req(max_output_tokens: Option<u64>, input_tokens: &[(&str, u64)]) -> Request {
        let mut req = text_req();
        req.llm_request.output.max_output_tokens = max_output_tokens;
        req.candidate_input_tokens = input_tokens
            .iter()
            .map(|(m, n)| (ModelId::from(*m), *n))
            .collect::<BTreeMap<_, _>>();
        req
    }

    fn bounded_profile(target: &str, rank: u16, cap: u64) -> CandidateProfile {
        CandidateProfile::new(target, true, true, rank).with_context_policy(
            ContextAdmissionPolicy::Bounded {
                usable_context_tokens: cap,
            },
        )
    }

    #[test]
    fn context_unmanaged_candidate_unchanged() {
        // A: UNMANAGED candidate requires no input-count fact and no output budget;
        // ready/capable => eligible exactly as pre-S2-E.
        let router = FleetRouter::new(
            vec![CandidateProfile::new("cloud", true, true, 1)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "cloud");
    }

    #[test]
    fn bounded_known_fit_eligible() {
        // B: BOUNDED fits (input + output < capacity) => eligible.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router
            .decide(&ctx_req(Some(8_000), &[("local", 50_000)]))
            .unwrap();
        assert_eq!(selected.to_string(), "local");
    }

    #[test]
    fn bounded_exact_boundary_eligible() {
        // C: input + output == capacity => eligible.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router
            .decide(&ctx_req(Some(15_536), &[("local", 50_000)]))
            .unwrap();
        assert_eq!(selected.to_string(), "local");
    }

    #[test]
    fn bounded_overflow_excluded() {
        // D: input + output > capacity => excluded.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        assert!(
            router
                .decide(&ctx_req(Some(15_537), &[("local", 50_000)]))
                .is_err()
        );
    }

    #[test]
    fn bounded_input_fits_but_output_overflows_excluded() {
        // E: input alone fits (65000) but requested output (2000) pushes past
        // capacity (65536) => excluded. FleetRouter protects the requested output
        // budget even though llama.cpp might silently constrain generation.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        assert!(
            router
                .decide(&ctx_req(Some(2_000), &[("local", 65_000)]))
                .is_err()
        );
    }

    #[test]
    fn bounded_missing_candidate_count_excluded() {
        // F: BOUNDED with no candidate-specific input fact => excluded.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        assert!(router.decide(&ctx_req(Some(8_000), &[])).is_err());
    }

    #[test]
    fn bounded_missing_output_budget_excluded() {
        // G: BOUNDED with known input but no max_output_tokens => excluded (not
        // treated as zero).
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        assert!(router.decide(&ctx_req(None, &[("local", 8_000)])).is_err());
    }

    #[test]
    fn bounded_arithmetic_overflow_excluded() {
        // H: checked_add overflow => excluded (fail closed).
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, u64::MAX)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        assert!(
            router
                .decide(&ctx_req(Some(u64::MAX), &[("local", u64::MAX)]))
                .is_err()
        );
    }

    #[test]
    fn candidate_specific_counts_never_borrowed() {
        // I: HTPC has a count, Comfy does not. HTPC may qualify; Comfy must NOT
        // borrow HTPC's count.
        let router = FleetRouter::new(
            vec![
                bounded_profile("htpc", 1, 65_536),
                bounded_profile("comfy", 2, 69_888),
            ],
            vec![
                (ModelId::from("htpc"), CandidateState::ready()),
                (ModelId::from("comfy"), CandidateState::ready()),
            ],
        )
        .unwrap();
        // Only HTPC has a candidate-specific count; both have an output budget.
        let (selected, fallbacks) = router
            .decide(&ctx_req(Some(4_096), &[("htpc", 30_000)]))
            .unwrap();
        assert_eq!(selected.to_string(), "htpc");
        // Comfy must NOT be eligible (no its-own count), so no fallback.
        assert!(fallbacks.is_empty());
    }

    #[test]
    fn context_fit_filters_before_preference() {
        // J: a higher-preference candidate does not fit; a lower-preference one
        // does => the lower-preference candidate is selected.
        let router = FleetRouter::new(
            vec![
                bounded_profile("pref", 1, 65_536),
                bounded_profile("fit", 2, 200_000),
            ],
            vec![
                (ModelId::from("pref"), CandidateState::ready()),
                (ModelId::from("fit"), CandidateState::ready()),
            ],
        )
        .unwrap();
        // 'pref' has no candidate-specific count => excluded despite rank 1.
        let (selected, _) = router
            .decide(&ctx_req(Some(1_000), &[("fit", 50_000)]))
            .unwrap();
        assert_eq!(selected.to_string(), "fit");
    }

    #[test]
    fn fallback_purity_selected_and_fallbacks_admitted() {
        // K: selected AND every fallback must independently pass context admission.
        let router = FleetRouter::new(
            vec![
                bounded_profile("a", 1, 65_536),
                bounded_profile("b", 2, 65_536),
                bounded_profile("c", 3, 65_536),
            ],
            vec![
                (ModelId::from("a"), CandidateState::ready()),
                (ModelId::from("b"), CandidateState::ready()),
                (ModelId::from("c"), CandidateState::ready()),
            ],
        )
        .unwrap();
        // 'a' and 'c' have counts; 'b' (rank 2) does not => 'b' must NOT appear in
        // fallbacks even though it is ranked above 'c'.
        let (selected, fallbacks) = router
            .decide(&ctx_req(Some(1_000), &[("a", 1_000), ("c", 1_000)]))
            .unwrap();
        let fbs = fallbacks
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert_eq!(selected.to_string(), "a");
        assert!(
            !fbs.contains(&"b".to_string()),
            "b must not leak into fallbacks"
        );
        assert_eq!(fbs, ["c"]);
    }

    #[test]
    fn context_fit_does_not_override_readiness() {
        // L: context-fit but not-ready candidate => excluded.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::not_ready())],
        )
        .unwrap();
        assert!(
            router
                .decide(&ctx_req(Some(1_000), &[("local", 1_000)]))
                .is_err()
        );
    }

    #[test]
    fn context_fit_does_not_override_capability() {
        // M: context-fit but tool-ineligible candidate => excluded for a tool request.
        let router = FleetRouter::new(
            vec![
                CandidateProfile::new("local", false, false, 1).with_context_policy(
                    ContextAdmissionPolicy::Bounded {
                        usable_context_tokens: 65_536,
                    },
                ),
            ],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        // A tool-required request: even with a fitting count and output budget, the
        // tool-ineligible candidate must be excluded by the capability filter.
        let mut req = tool_req();
        req.llm_request.output.max_output_tokens = Some(1_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 1_000)]);
        assert!(router.decide(&req).is_err());
    }

    #[test]
    fn context_decision_is_deterministic() {
        // N: same request/facts/snapshot => same decision.
        let router = FleetRouter::new(
            vec![
                bounded_profile("a", 1, 65_536),
                bounded_profile("b", 2, 65_536),
            ],
            vec![
                (ModelId::from("a"), CandidateState::ready()),
                (ModelId::from("b"), CandidateState::ready()),
            ],
        )
        .unwrap();
        let req = ctx_req(Some(1_000), &[("a", 1_000), ("b", 1_000)]);
        let d1 = router.decide(&req).unwrap();
        let d2 = router.decide(&req).unwrap();
        assert_eq!(d1.0, d2.0);
        assert_eq!(d1.1, d2.1);
    }

    #[test]
    fn zero_bounded_capacity_rejected_at_new() {
        // Core invariant: a Bounded context policy with a zero usable capacity is an
        // invalid context-admission contract and must fail construction (new path).
        let profile = CandidateProfile::new("local", true, true, 1).with_context_policy(
            ContextAdmissionPolicy::Bounded {
                usable_context_tokens: 0,
            },
        );
        let err = FleetRouter::new(
            vec![profile],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid zero usable_context_tokens"),
            "error must name the invalid zero capacity: {err}"
        );
    }

    #[test]
    fn zero_bounded_capacity_rejected_at_with_source() {
        // Core invariant via the injected-source constructor path.
        let profile = CandidateProfile::new("local", true, true, 1).with_context_policy(
            ContextAdmissionPolicy::Bounded {
                usable_context_tokens: 0,
            },
        );
        let source = SharedFleetState::new(
            FleetSnapshot::new(vec![(ModelId::from("local"), CandidateState::ready())]).unwrap(),
        );
        let err = FleetRouter::with_source(vec![profile], Arc::new(source)).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid zero usable_context_tokens"),
            "error must name the invalid zero capacity: {err}"
        );
    }

    #[test]
    fn unmanaged_policy_is_valid_and_zero_bounded_is_not() {
        // The explicit-semantic check: Unmanaged is always valid; only Bounded{0} is
        // invalid. Both constructors share one validator.
        let unmanaged = CandidateProfile::new("local", true, true, 1); // Unmanaged
        FleetRouter::with_source(
            vec![unmanaged],
            Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap())),
        )
        .unwrap();
    }

    // --- Request-sourced work-shape dispatch (Fix D) --------------------------

    /// A request carrying a declared `work_shape` in structured extensions.
    fn work_shape_req(shape: &str) -> Request {
        let mut req = text_req();
        req.llm_request.extensions.fields.insert(
            "work_shape".to_string(),
            serde_json::Value::String(shape.to_string()),
        );
        req
    }

    /// A request carrying the LEGACY `work_class` contract (the shape the old
    /// ResourceRouter universal smart route + the live Turnstone aliases send).
    fn work_class_req(class: &str) -> Request {
        let mut req = text_req();
        req.llm_request.extensions.fields.insert(
            "work_class".to_string(),
            serde_json::Value::String(class.to_string()),
        );
        req
    }

    /// A dynamic smart-style router: luna (bounded-only, rank 1) and deepseek
    /// (any shape, non-thinking, rank 2), all ready, request-sourced work shape.
    fn work_shape_router() -> FleetRouter {
        let profiles = vec![
            CandidateProfile::new("luna", true, false, 1).with_work_shape(WorkShape::Bounded),
            CandidateProfile::new("deepseek", true, false, 2),
        ];
        let state = SharedFleetState::new(
            FleetSnapshot::new(vec![
                (ModelId::from("luna"), CandidateState::ready()),
                (ModelId::from("deepseek"), CandidateState::ready()),
            ])
            .unwrap(),
        );
        FleetRouter::with_source(profiles, Arc::new(state))
            .unwrap()
            .with_request_work_shape()
    }

    #[test]
    fn request_work_shape_bounded_prefers_luna() {
        // bounded / non-thinking -> Luna (rank1) both eligible -> luna selected.
        let router = work_shape_router();
        let (selected, _fb) = router.decide(&work_shape_req("bounded")).unwrap();
        assert_eq!(selected.to_string(), "luna");
    }

    #[test]
    fn request_work_shape_agentic_excludes_luna_never_selects_it() {
        // agentic / non-thinking -> Luna (bounded-only) is WORK-SHAPE excluded;
        // deepseek (any-shape) is the only eligible candidate.
        let router = work_shape_router();
        let (selected, fb) = router.decide(&work_shape_req("agentic")).unwrap();
        assert_eq!(selected.to_string(), "deepseek");
        let all: Vec<String> = std::iter::once(selected.to_string())
            .chain(fb.iter().map(ToString::to_string))
            .collect();
        assert!(
            !all.contains(&"luna".to_string()),
            "agentic/non-thinking must NEVER select or fall back to the bounded-only luna: {all:?}"
        );
    }

    #[test]
    fn request_work_shape_missing_metadata_is_not_prompt_guessed() {
        // A request with NO declared work_shape under a request-source dispatcher
        // must NOT be guessed from prompt text; luna (bounded) stays eligible
        // because no shape is declared (deterministic preference wins).
        let router = work_shape_router();
        let (selected, _fb) = router.decide(&text_req()).unwrap();
        // No declared shape -> no work-shape exclusion; preference picks luna.
        assert_eq!(selected.to_string(), "luna");
    }

    // --- Legacy `work_class` compatibility (Fix: cutover regression) ---------
    // The live Turnstone aliases (switchyard-smart-bounded/-agentic/-reasoning)
    // send `work_class` via extra_body, not `work_shape`. The S2-H FleetRouter
    // must translate it exactly like the old ResourceRouter: bounded->Bounded,
    // agentic->Agentic (Luna NEVER), reasoning->(Bounded, deliberate) i.e.
    // thinking-required (Luna NEVER).

    #[test]
    fn legacy_work_class_bounded_prefers_luna() {
        let router = work_shape_router();
        let (selected, _fb) = router.decide(&work_class_req("bounded")).unwrap();
        assert_eq!(selected.to_string(), "luna");
    }

    #[test]
    fn legacy_work_class_agentic_never_selects_luna() {
        let router = work_shape_router();
        let (selected, fb) = router.decide(&work_class_req("agentic")).unwrap();
        assert_eq!(selected.to_string(), "deepseek");
        let all: Vec<String> = std::iter::once(selected.to_string())
            .chain(fb.iter().map(ToString::to_string))
            .collect();
        assert!(
            !all.contains(&"luna".to_string()),
            "legacy work_class=agentic must NEVER select or fall back to the bounded-only luna: {all:?}"
        );
    }

    #[test]
    fn legacy_work_class_reasoning_requires_thinking_never_luna() {
        // A thinking-required router (deepseek thinking candidate exists).
        let profiles = vec![
            CandidateProfile::new("luna", true, false, 1).with_work_shape(WorkShape::Bounded),
            CandidateProfile::new("deepseek", true, true, 2),
        ];
        let state = SharedFleetState::new(
            FleetSnapshot::new(vec![
                (ModelId::from("luna"), CandidateState::ready()),
                (ModelId::from("deepseek"), CandidateState::ready()),
            ])
            .unwrap(),
        );
        let router = FleetRouter::with_source(profiles, Arc::new(state))
            .unwrap()
            .with_request_work_shape();
        // Legacy reasoning contract -> deliberate -> thinking required.
        let (selected, fb) = router.decide(&work_class_req("reasoning")).unwrap();
        assert_eq!(selected.to_string(), "deepseek");
        let all: Vec<String> = std::iter::once(selected.to_string())
            .chain(fb.iter().map(ToString::to_string))
            .collect();
        assert!(
            !all.contains(&"luna".to_string()),
            "legacy work_class=reasoning must select the thinking candidate, never luna: {all:?}"
        );
    }

    // --- FLEET-1: minimum-context requirement (structured ingress) ------------

    /// A request with a declared `min_context_tokens` in structured extensions.
    fn min_ctx_req(min: Option<u64>) -> Request {
        let mut req = text_req();
        if let Some(n) = min {
            req.llm_request
                .extensions
                .fields
                .insert("min_context_tokens".into(), serde_json::json!(n));
        }
        req
    }

    /// A request with a malformed (non-integer) `min_context_tokens`.
    fn bad_min_ctx_req() -> Request {
        let mut req = text_req();
        req.llm_request
            .extensions
            .fields
            .insert("min_context_tokens".into(), serde_json::json!("lots"));
        req
    }

    /// A request with a zero `min_context_tokens`.
    fn zero_min_ctx_req() -> Request {
        let mut req = text_req();
        req.llm_request
            .extensions
            .fields
            .insert("min_context_tokens".into(), serde_json::json!(0));
        req
    }

    /// A candidate with a static qualified usable context capacity (builder).
    fn capacity_profile(target: &str, rank: u16, cap: u64) -> CandidateProfile {
        CandidateProfile::new(target, true, true, rank).with_usable_context_tokens(cap)
    }

    #[test]
    fn min_context_absent_preserves_behavior() {
        // A: no declared requirement => candidate with UNKNOWN capacity stays eligible.
        let router = FleetRouter::new(
            vec![CandidateProfile::new("cloud", true, true, 1)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "cloud");
    }

    #[test]
    fn min_context_below_capacity_eligible() {
        // B: valid minimum below candidate capacity => eligible.
        let router = FleetRouter::new(
            vec![capacity_profile("cloud", 1, 200_000)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router.decide(&min_ctx_req(Some(50_000))).unwrap();
        assert_eq!(selected.to_string(), "cloud");
    }

    #[test]
    fn min_context_exact_boundary_eligible() {
        // C: exact boundary (min == capacity) => eligible.
        let router = FleetRouter::new(
            vec![capacity_profile("cloud", 1, 200_000)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router.decide(&min_ctx_req(Some(200_000))).unwrap();
        assert_eq!(selected.to_string(), "cloud");
    }

    #[test]
    fn min_context_above_capacity_excluded() {
        // D: requirement above capacity => candidate excluded => no eligible candidate.
        let router = FleetRouter::new(
            vec![capacity_profile("cloud", 1, 200_000)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        assert!(router.decide(&min_ctx_req(Some(200_001))).is_err());
    }

    #[test]
    fn min_context_unknown_capacity_declared_excluded() {
        // E: unknown capacity + declared requirement => candidate excluded (fail closed).
        let router = FleetRouter::new(
            vec![CandidateProfile::new("cloud", true, true, 1)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        assert!(router.decide(&min_ctx_req(Some(1_000))).is_err());
    }

    #[test]
    fn min_context_unknown_capacity_no_requirement_preserved() {
        // F: unknown capacity + no requirement => existing behavior preserved.
        let router = FleetRouter::new(
            vec![CandidateProfile::new("cloud", true, true, 1)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let (selected, _) = router.decide(&text_req()).unwrap();
        assert_eq!(selected.to_string(), "cloud");
    }

    #[test]
    fn min_context_malformed_hard_failure() {
        // G: malformed (non-integer) => hard algorithm error, not silently ignored.
        let router = FleetRouter::new(
            vec![capacity_profile("cloud", 1, 200_000)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let err = router.decide(&bad_min_ctx_req()).unwrap_err();
        assert!(
            err.to_string().contains("min_context_tokens"),
            "expected min_context_tokens error, got {err}"
        );
    }

    #[test]
    fn min_context_zero_hard_failure() {
        // H: zero => hard algorithm error (zero is invalid, not "unknown").
        let router = FleetRouter::new(
            vec![capacity_profile("cloud", 1, 200_000)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap();
        let err = router.decide(&zero_min_ctx_req()).unwrap_err();
        assert!(
            err.to_string().contains("positive"),
            "expected positive-integer error, got {err}"
        );
    }

    #[test]
    fn min_context_larger_capacity_does_not_rank() {
        // I: once the minimum is satisfied, capacity NEVER affects ranking.
        // Preference rank wins regardless of which candidate has the larger capacity.
        let profiles = vec![
            capacity_profile("small", 1, 100_000),
            capacity_profile("large", 2, 500_000),
        ];
        let state = SharedFleetState::new(
            FleetSnapshot::new(vec![
                (ModelId::from("small"), CandidateState::ready()),
                (ModelId::from("large"), CandidateState::ready()),
            ])
            .unwrap(),
        );
        let router = FleetRouter::with_source(profiles, Arc::new(state)).unwrap();
        // Both satisfy min 50k; rank 1 (small) wins despite smaller capacity.
        let (selected, _) = router.decide(&min_ctx_req(Some(50_000))).unwrap();
        assert_eq!(selected.to_string(), "small");
        // A large min that only `large` satisfies => large wins (admission only).
        let (selected, _) = router.decide(&min_ctx_req(Some(400_000))).unwrap();
        assert_eq!(selected.to_string(), "large");
    }

    #[test]
    fn min_context_combined_bounded_exact_fit() {
        // Combined admission: a BOUNDED candidate must satisfy BOTH the declared
        // minimum AND the exact request fit (input + output <= capacity).
        let profiles = vec![bounded_profile("local", 1, 65_536).with_usable_context_tokens(65_536)];
        let router = FleetRouter::new(
            profiles.clone(),
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        // (1) minimum satisfied but exact fit fails (60k + 8k > 65,536) => excluded.
        let mut req = min_ctx_req(Some(60_000));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 60_000)]);
        assert!(router.decide(&req).is_err());
        // (2) exact fit OK but minimum not met (min 100k > 65,536) => excluded.
        let mut req = min_ctx_req(Some(100_000));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 10_000)]);
        assert!(router.decide(&req).is_err());
        // (3) both satisfied (60k + 5k <= 65,536; min 60k <= 65,536) => eligible.
        let mut req = min_ctx_req(Some(60_000));
        req.llm_request.output.max_output_tokens = Some(5_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 60_000)]);
        let (selected, _) = router.decide(&req).unwrap();
        assert_eq!(selected.to_string(), "local");
    }

    #[test]
    fn min_context_zero_capacity_config_rejected() {
        // J: static capacity Some(0) is invalid configuration at construction.
        let err = FleetRouter::new(
            vec![CandidateProfile::new("cloud", true, true, 1).with_usable_context_tokens(0)],
            vec![(ModelId::from("cloud"), CandidateState::ready())],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("zero"),
            "expected zero-capacity config error, got {err}"
        );
    }

    // --- FLEET-1 review: capacity-coherence invariant (Bounded vs static) ----

    #[test]
    fn bounded_without_static_uses_policy_capacity_for_min() {
        // 1: Bounded(B) + no static field + min <= B => min eligibility uses B.
        // (Existing `bounded_profile` helper builds exactly this shape.) The
        // Bounded exact-fit admission still requires candidate input facts +
        // an output budget; supply them so only min-context is under test.
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        let mut req = min_ctx_req(Some(65_536));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 50_000)]);
        let (selected, _) = router.decide(&req).unwrap();
        assert_eq!(selected.to_string(), "local");
        // Above B (min 65,537) => excluded (fail closed).
        let mut req = min_ctx_req(Some(65_537));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 50_000)]);
        assert!(router.decide(&req).is_err());
    }

    #[test]
    fn bounded_with_matching_static_accepted() {
        // 2: Bounded(B) + static B => accepted (coherent).
        let router = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536).with_usable_context_tokens(65_536)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        let mut req = min_ctx_req(Some(65_536));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 50_000)]);
        let (selected, _) = router.decide(&req).unwrap();
        assert_eq!(selected.to_string(), "local");
    }

    #[test]
    fn bounded_with_mismatched_static_rejected() {
        // 3: Bounded(B) + static C where C != B => rejected at construction.
        let err = FleetRouter::new(
            vec![bounded_profile("local", 1, 65_536).with_usable_context_tokens(69_888)],
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("disagrees"),
            "expected capacity-coherence error, got {err}"
        );
    }

    #[test]
    fn bounded_combined_min_and_exact_fit_unchanged() {
        // 4: combined minimum + exact-fit behavior unchanged when coherent.
        let profiles = vec![bounded_profile("local", 1, 65_536).with_usable_context_tokens(65_536)];
        let router = FleetRouter::new(
            profiles,
            vec![(ModelId::from("local"), CandidateState::ready())],
        )
        .unwrap();
        // min ok, exact-fit fails (60k + 8k > 65,536) => excluded.
        let mut req = min_ctx_req(Some(60_000));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 60_000)]);
        assert!(router.decide(&req).is_err());
        // exact-fit ok, min fails (min 100k > 65,536) => excluded.
        let mut req = min_ctx_req(Some(100_000));
        req.llm_request.output.max_output_tokens = Some(8_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 10_000)]);
        assert!(router.decide(&req).is_err());
        // both ok => eligible.
        let mut req = min_ctx_req(Some(60_000));
        req.llm_request.output.max_output_tokens = Some(5_000);
        req.candidate_input_tokens = BTreeMap::from([(ModelId::from("local"), 60_000)]);
        let (selected, _) = router.decide(&req).unwrap();
        assert_eq!(selected.to_string(), "local");
    }
}
