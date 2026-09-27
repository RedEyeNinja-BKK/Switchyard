// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fleet routing: the runtime representation of a declared candidate ladder.
//!
//! This module is the **F2 representation layer** and nothing more. It converts a
//! parsed [`FleetCandidateConfig`](crate::FleetCandidateConfig) into a runtime
//! value that preserves the deployment's declarations exactly, in declared order.
//!
//! Deliberately absent, because no live route uses them: work-shape eligibility,
//! a preflight context-admission policy, and any live readiness state. Those are
//! eligibility concerns, owned by the layers that implement them, not
//! representation. Nothing here decides whether a candidate may serve a request.
//!
//! Execution is deliberately non-operational. A constructed router holds its
//! candidate ladder and refuses to route until the selection layers exist; see
//! [`FleetRouter::route`]. Serving "the first candidate" or "the most preferred
//! candidate" would be a selection decision, and making one here would be a silent
//! behavioral guess that later layers would have to unlearn.

use std::sync::Arc;

use switchyard_protocol::{Category, ContentBlock, ModelId, Request};

use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result, RoutingOutcome};

/// One candidate as the runtime holds it: the deployment's declaration, unchanged.
///
/// Field-for-field the live candidate schema. No field is added for a capability
/// the deployment did not declare, and no declared field is defaulted away.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FleetCandidate {
    /// Target model id this candidate would select when chosen.
    pub target: ModelId,
    /// Whether the target declares tool-calling support.
    pub tool_calling: bool,
    /// Whether the target declares reasoning support.
    pub reasoning: bool,
    /// Whether the target declares image-input support.
    pub supports_vision: bool,
    /// Declared preference; lower is preferred. Carried, not yet acted on.
    pub preference_rank: u16,
    /// Declared usable context capacity; `None` when the deployment asserts none.
    pub usable_context_tokens: Option<u64>,
}

/// A constructed fleet route: its candidate ladder in declared order, plus the
/// escalation declaration it carries.
///
/// Construction is faithful and total. It filters nothing, sorts nothing, and
/// deduplicates nothing: a candidate that later proves unusable is still here,
/// because deciding that is the selection layer's job.
#[derive(Clone, Debug)]
pub struct FleetRouter {
    /// Candidates in the order the deployment declared them.
    candidates: Arc<[FleetCandidate]>,
    /// Escalation destination route id, when declared. Carried, not fired.
    escalation: Option<ModelId>,
    /// Escalation input-size threshold, when declared. Carried, not fired.
    escalation_max_input_tokens: Option<u64>,
    /// Injected coherent readiness source (F8). One decision reads it exactly
    /// once, so a ladder can never mix readiness generations.
    state: Arc<dyn FleetStateSource>,
}

impl FleetRouter {
    /// Builds a router over `candidates`, preserving their order.
    ///
    /// Readiness comes from a [`StaticFleetState`] over an EMPTY snapshot, so
    /// every candidate is unobserved and therefore fail-closed. That is the
    /// honest default: a router constructed without an injected readiness
    /// producer has no factual basis for any selection.
    pub fn new(
        candidates: Vec<FleetCandidate>,
        escalation: Option<ModelId>,
        escalation_max_input_tokens: Option<u64>,
    ) -> Self {
        Self {
            candidates: candidates.into(),
            escalation,
            escalation_max_input_tokens,
            state: Arc::new(StaticFleetState::new(FleetSnapshot::empty())),
        }
    }

    /// Builds a router over `candidates` with an injected readiness source.
    pub fn with_source(
        candidates: Vec<FleetCandidate>,
        escalation: Option<ModelId>,
        escalation_max_input_tokens: Option<u64>,
        state: Arc<dyn FleetStateSource>,
    ) -> Self {
        Self {
            candidates: candidates.into(),
            escalation,
            escalation_max_input_tokens,
            state,
        }
    }

    /// The candidate ladder, in declared order.
    pub fn candidates(&self) -> &[FleetCandidate] {
        &self.candidates
    }

    /// The declared escalation destination, if any. Carried only; not fired.
    pub fn escalation(&self) -> Option<&ModelId> {
        self.escalation.as_ref()
    }

    /// The declared escalation input-size threshold, if any. Carried only.
    pub fn escalation_max_input_tokens(&self) -> Option<u64> {
        self.escalation_max_input_tokens
    }

    /// The readiness source this decision reads. Exposed for the host's
    /// readiness monitor and for tests that assert the single-read property.
    pub fn state(&self) -> &Arc<dyn FleetStateSource> {
        &self.state
    }

    /// One coherent readiness snapshot for one decision.
    fn snapshot(&self) -> Arc<FleetSnapshot> {
        self.state.snapshot()
    }

    /// The decision core: filter, rank once, split into `(selected, fallbacks)`.
    ///
    /// Ordering is production's, and each stage is the already-implemented
    /// filter for that layer rather than a new one:
    ///
    /// 1. static capability eligibility (F2/F5) — tools, reasoning, vision;
    /// 2. readiness (F6) — only `immediately_eligible` may be selected;
    /// 3. context admission (row 40) — a pure fail-closed check, no producer;
    /// 4. exactly ONE preference sort (F4) — `compare_preference`.
    ///
    /// Context capacity is admission only and is never a ranking signal.
    /// `escalation` is deliberately excluded from the ladder: escalation is
    /// host-owned and resolved as a separate route, never as a candidate.
    ///
    /// `facts` is passed in by the caller and is empty on every current path.
    /// That is row 40's R40-B dormancy preserved deliberately: this layer
    /// consumes per-candidate input-token facts but wires no producer, so a
    /// `Bounded` candidate fails closed on a missing fact rather than on an
    /// arithmetic verdict. Activating a producer is separate work.
    pub fn decide(
        &self,
        request: &Request,
        facts: &ContextFacts,
    ) -> Result<(ModelId, Vec<ModelId>)> {
        // ONE snapshot for the whole decision. Read before the loop and never
        // again, so every surviving candidate was judged against the same
        // readiness generation.
        let snapshot = self.snapshot();

        let eligibility_facts = EligibilityFacts::from_request(request);
        let eligible = static_eligibility(self, &eligibility_facts);
        let ready = ready_candidates(&eligible, &snapshot);
        let max_output = request_output_budget(request);

        // Readiness survivors projected onto the F4 ordering seam. Readiness can
        // only REMOVE a candidate, never reorder one, so this projection loses
        // nothing the sort could have used.
        let survivors: Vec<CandidateEligibility<'_>> = ready
            .iter()
            .map(|candidate| CandidateEligibility::Eligible(candidate))
            .collect();
        let admitted = context_admission_filter(&survivors, facts, max_output);

        // Exactly one sort. `preference_order` is the F4 comparator seam.
        let ordered = preference_order(&admitted_to_eligibility(&admitted));

        if ordered.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message:
                    "fleet router found no immediately-eligible candidate for this request"
                        .to_string(),
            });
        }

        let mut ids = ordered.into_iter().map(|candidate| candidate.target.clone());
        let selected = ids.next().expect("non-empty ordered set");
        Ok((selected, ids.collect()))
    }
}

/// Projects context-admitted candidates back onto the F4 ordering seam.
///
/// `preference_order` filters on `is_eligible()`, so the admitted set is
/// narrowed to its admitted members and re-projected through the same
/// comparator. Context admission never contributes a ranking signal: it can
/// only remove a candidate, never reorder one.
fn admitted_to_eligibility<'a>(
    admitted: &[ContextAdmittedCandidate<'a>],
) -> Vec<CandidateEligibility<'a>> {
    admitted
        .iter()
        .filter(|entry| entry.is_admitted())
        .map(|entry| CandidateEligibility::Eligible(entry.candidate))
        .collect()
}

// ---------------------------------------------------------------------------
// F8 — readiness source seam.
//
// The routing algorithm reads readiness through this trait exactly once per
// decision, and what it reads is an immutable `Arc`. A reader therefore cannot
// observe a half-updated fleet, and one decision can never mix generations.
// ---------------------------------------------------------------------------

/// A coherent, immutable view of current fleet readiness.
pub trait FleetStateSource: Send + Sync + std::fmt::Debug {
    /// The current coherent fleet readiness snapshot.
    fn snapshot(&self) -> Arc<FleetSnapshot>;
}

/// A fixed readiness source returning the same snapshot every decision.
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

/// A readiness source whose whole snapshot an external writer replaces
/// atomically between decisions.
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
        Arc::clone(&self.current.read())
    }
}

#[async_trait::async_trait]
impl Algorithm for FleetRouter {
    fn name(&self) -> &str {
        "fleet_router"
    }

    /// The F8 decision: one readiness snapshot, the existing filters, one
    /// preference sort, and an immutable ordered ladder.
    ///
    /// Selection happens ONCE. The returned ladder is consumed by the existing
    /// walker in `llm-client`; no candidate attempt re-runs this or re-reads
    /// readiness.
    async fn route(self: Arc<Self>, _driver: Driver, request: Request) -> Result<RoutingOutcome> {
        // R40-B DORMANCY: no context-fact producer is wired on this path, so
        // context admission is not an active filter. The row-40 CONTRACT still
        // exists (candidates may declare `usable_context_tokens`); only the
        // PRODUCER is absent, and its absence must not reject bounded
        // candidates. Activating a producer is separate work.
        let (selected, fallbacks) = self.decide(&request, &ContextFacts::Disabled)?;
        tracing::debug!(
            algorithm = "fleet_router",
            selected = %selected,
            fallbacks = fallbacks.len(),
            "fleet decision"
        );
        Ok(RoutingOutcome::route_to(selected, fallbacks, request))
    }
}

/// Every target this route may reach, in declared order.
///
/// The driver resolves offloaded calls through a category, so a fleet route
/// publishes its whole ladder under `Any` in declared order.
pub fn fleet_category_models(router: &FleetRouter) -> Vec<ModelId> {
    let _ = Category::Any;
    router
        .candidates()
        .iter()
        .map(|c| c.target.clone())
        .collect()
}

/// Why one candidate is not eligible for a request.
///
/// Each variant is a distinct production filter. They are never aggregated into a
/// single verdict with an invented precedence: a caller that cares must see every
/// reason, and F3 deliberately does not choose one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IneligibleReason {
    /// The request carries tool definitions and this candidate does not declare
    /// tool calling.
    MissingToolCalling,
    /// The request asks for reasoning and this candidate does not declare it.
    MissingReasoning,
    /// The request carries image content and this candidate does not declare
    /// vision.
    MissingVision,
}

/// Whether a request requires a tool-calling candidate.
///
/// Production rule: the request carries at least one tool definition. This is
/// the *declaration* check, not whether the request will actually call a tool.
pub fn request_requires_tools(request: &Request) -> bool {
    !request.llm_request.tools.is_empty()
}

/// Whether a request carries image content anywhere in its messages.
///
/// Derived from the typed content blocks, never from prompt text or model names.
pub fn request_requires_vision(request: &Request) -> bool {
    request
        .llm_request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .any(|block| matches!(block, ContentBlock::Image { .. }))
}

/// Whether a request requires a reasoning-capable candidate.
///
/// Production rule, from the normalized `effort` field only:
/// no reasoning controls, or `effort = "none"`, means reasoning is NOT required;
/// any other effort value means it is.
///
/// `reasoning.raw` is deliberately not consulted. It is a container for whatever
/// a provider or decoder preserved, so its presence says nothing about whether
/// reasoning is enabled, and there is no provider-agnostic shape within it that
/// reliably means "enabled". Reading it would admit requests production rejects.
pub fn reasoning_requested(request: &Request) -> bool {
    match request.llm_request.reasoning.effort.as_deref() {
        None | Some("none") => false,
        Some(_) => true,
    }
}

/// The request facts every static eligibility filter reads.
///
/// Extracted once so each filter is a pure function of (facts, candidate) and
/// carries no hidden state, no target identity, and no cross-route cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EligibilityFacts {
    /// The request carries tool definitions.
    pub requires_tools: bool,
    /// The request carries image content.
    pub requires_vision: bool,
    /// The request asks for reasoning.
    pub requires_reasoning: bool,
}

impl EligibilityFacts {
    /// Reads the request facts used by static eligibility.
    pub fn from_request(request: &Request) -> Self {
        Self {
            requires_tools: request_requires_tools(request),
            requires_vision: request_requires_vision(request),
            requires_reasoning: reasoning_requested(request),
        }
    }
}

/// Static eligibility of one candidate for one request.
///
/// This answers exactly one question: may this candidate serve this request on
/// the candidate's own declared capabilities? It never answers which eligible
/// candidate should run - that is ordering, which is a later concern and which
/// `preference_rank` feeds.
/// Deliberately not covered here, because each needs a fact this layer does not
/// own: live readiness and health state, work-shape and legacy-work-class
/// declarations, the declared `min_context_tokens` requirement together with a
/// candidate's usable context capacity, and exact per-candidate input-token
/// counts. A candidate that passes here is *statically* eligible; whether it may
/// actually serve is not yet decided.
pub fn candidate_is_statically_eligible<'a>(
    candidate: &'a FleetCandidate,
    facts: &EligibilityFacts,
) -> CandidateEligibility<'a> {
    if facts.requires_tools && !candidate.tool_calling {
        return CandidateEligibility::Ineligible {
            candidate,
            reason: IneligibleReason::MissingToolCalling,
        };
    }
    if facts.requires_reasoning && !candidate.reasoning {
        return CandidateEligibility::Ineligible {
            candidate,
            reason: IneligibleReason::MissingReasoning,
        };
    }
    if facts.requires_vision && !candidate.supports_vision {
        return CandidateEligibility::Ineligible {
            candidate,
            reason: IneligibleReason::MissingVision,
        };
    }
    CandidateEligibility::Eligible(candidate)
}

/// One candidate's static verdict, always paired with the candidate it is about.
///
/// Keeping the candidate in the verdict means a derived view can never lose its
/// correspondence with the stored ladder, and the caller never has to re-derive
/// which entry a bare reason referred to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateEligibility<'a> {
    /// The candidate's own declared capabilities satisfy the request.
    Eligible(&'a FleetCandidate),
    /// The candidate cannot serve, and here is precisely which filter excluded it.
    Ineligible {
        /// The candidate that was excluded, in its declared position.
        candidate: &'a FleetCandidate,
        /// The single production filter that excluded it.
        reason: IneligibleReason,
    },
}

impl<'a> CandidateEligibility<'a> {
    /// The candidate this verdict is about, eligible or not.
    pub fn candidate(&self) -> &'a FleetCandidate {
        match self {
            Self::Eligible(candidate) | Self::Ineligible { candidate, .. } => candidate,
        }
    }

    /// Whether this candidate passed every implemented filter.
    pub fn is_eligible(&self) -> bool {
        matches!(self, Self::Eligible(_))
    }
}

/// Static eligibility for every candidate, in declared order.
///
/// A derived view. The stored ladder is never reordered, filtered, or rewritten:
/// entries appear in this result in exactly the order they occupy in
/// [`FleetRouter::candidates`], and ineligible ones are marked, not removed.
pub fn static_eligibility<'r>(
    router: &'r FleetRouter,
    facts: &EligibilityFacts,
) -> Vec<CandidateEligibility<'r>> {
    router
        .candidates()
        .iter()
        .map(|candidate| candidate_is_statically_eligible(candidate, facts))
        .collect()
}

/// Production's preference comparator, exactly as production defines it:
///
/// ```text
/// a.preference_rank.cmp(&b.preference_rank).then_with(|| a.target.cmp(&b.target))
/// ```
///
/// Three properties are load-bearing and each is pinned by a test:
///
/// 1. **Direction is lower-rank-first.** `u16` ascending, so rank 1 precedes
///    rank 2. Context size is never a ranking signal.
/// 2. **The tie-break is the target's own `Ord`.** `ModelId` derives `Ord` on a
///    `String` wrapper, so equal ranks are ordered byte-lexically ascending by
///    target id — not by declaration order, and not case-insensitively.
/// 3. **The sort is stable**, so candidates that compare fully equal keep their
///    incoming relative order.
///
/// The comparator is total: `(preference_rank, target)` is unique per candidate
/// in a well-formed ladder, so stability is a safety property rather than a
/// source of ordering.
pub fn compare_preference(a: &FleetCandidate, b: &FleetCandidate) -> std::cmp::Ordering {
    a.preference_rank
        .cmp(&b.preference_rank)
        .then_with(|| a.target.cmp(&b.target))
}

/// Preference order over an already-eligible candidate set.
///
/// **This is a projection, not a final routing decision.** Only static
/// eligibility has been applied so far: context-token admission (row 40), live
/// readiness, and the remaining capability/reasoning constraints are not yet
/// present, so a candidate that appears in this order may still be excluded
/// before it could serve. Callers that need a final order must run the remaining
/// filters first and pass only their survivors through this same seam.
///
/// The returned view is a new allocation. [`FleetRouter::candidates`] keeps
/// exact declaration order and is never reordered or mutated.
pub fn preference_order<'a>(eligible: &[CandidateEligibility<'a>]) -> Vec<&'a FleetCandidate> {
    let mut ordered: Vec<&FleetCandidate> = eligible
        .iter()
        .filter(|verdict| verdict.is_eligible())
        .map(|verdict| verdict.candidate())
        .collect();
    ordered.sort_by(|a, b| compare_preference(a, b));
    ordered
}

// ---------------------------------------------------------------------------
// F6 — operational readiness.
//
// Readiness is a *factual runtime fact*, never a static profile field. It is
// produced outside the router (the host's readiness monitor) and injected as one
// coherent immutable snapshot per decision, so a decision never mixes readiness
// generations.
//
// The state space is deliberately NOT collapsed into one boolean. Production
// distinguishes three cases and this carries the same distinction:
//
// - `ready`               — immediately usable;
// - `not_ready`           — presently unavailable;
// - `transition_required` — valid but needs an external transition first,
//   and therefore not usable for an immediate request. The router never
//   performs the transition.
//
// A candidate with no entry at all is **fail-closed**: absence is treated as
// not immediately eligible. Readiness never makes a candidate "more eligible"
// and never re-ranks, sorts, or reorders anything.
// ---------------------------------------------------------------------------

/// Live operational readiness of one candidate target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CandidateState {
    /// Whether the target is immediately usable.
    pub ready: bool,
    /// Whether the target needs an external transition before it is usable.
    pub transition_required: bool,
}

impl CandidateState {
    /// Immediately usable, no transition needed.
    pub fn ready() -> Self {
        Self {
            ready: true,
            transition_required: false,
        }
    }

    /// Presently unavailable.
    pub fn not_ready() -> Self {
        Self {
            ready: false,
            transition_required: false,
        }
    }

    /// Valid, but an external transition must happen first.
    pub fn transition_required() -> Self {
        Self {
            ready: false,
            transition_required: true,
        }
    }

    /// Whether this candidate may serve an immediate request.
    ///
    /// `transition_required` is a valid, *capable* state that is simply not
    /// immediately available, so it does not survive this filter even though
    /// the candidate is otherwise fine.
    pub fn is_immediately_eligible(&self) -> bool {
        self.ready && !self.transition_required
    }
}

/// How a candidate stands against a readiness snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadinessVerdict {
    /// Presently available for an immediate request.
    Ready,
    /// Observed, and presently unavailable.
    Unready,
    /// Observed, valid, but an external transition is required first.
    TransitionRequired,
    /// No observation exists for this target. Fail-closed: not selectable.
    Unobserved,
}

/// One coherent, immutable readiness snapshot for a whole decision.
///
/// A decision consumes exactly one snapshot, so it cannot mix generations.
/// The snapshot is keyed by **model id** — the same identity production uses —
/// which means two distinct candidate targets that resolve to the same model id
/// deliberately SHARE one observation, while different model ids never do.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FleetSnapshot {
    states: Vec<(ModelId, CandidateState)>,
}

impl FleetSnapshot {
    /// Builds a snapshot from `(model id, state)` entries.
    ///
    /// Duplicate keys are rejected: they would make a decision ambiguous.
    pub fn new(states: Vec<(ModelId, CandidateState)>) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        for (key, _) in &states {
            if !seen.insert(key) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("fleet snapshot key {key:?} appears more than once"),
                });
            }
        }
        Ok(Self { states })
    }

    /// A snapshot with no observations. Every target is unobserved, so every
    /// candidate is fail-closed until the producer publishes its first
    /// generation.
    pub fn empty() -> Self {
        Self { states: Vec::new() }
    }

    /// The recorded state for `model`, or `None` when unobserved.
    ///
    /// Absence is reported distinctly from `not_ready` so the consumer can
    /// preserve production's fail-closed semantics without conflating the two.
    pub fn state_for(&self, model: &ModelId) -> Option<CandidateState> {
        self.states
            .iter()
            .find(|(id, _)| id == model)
            .map(|(_, state)| *state)
    }
    /// Number of observed targets. Small and bounded by the fleet.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether this snapshot carries no observations at all.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// The observed `(model id, state)` pairs, in declared order.
    pub fn entries(&self) -> &[(ModelId, CandidateState)] {
        &self.states
    }
}

/// The readiness standing of one candidate against a snapshot.
pub fn candidate_readiness(
    candidate: &FleetCandidate,
    snapshot: &FleetSnapshot,
) -> ReadinessVerdict {
    match snapshot.state_for(&candidate.target) {
        // Absent is fail-closed, and is NOT the same claim as "observed and
        // unavailable": a producer may simply not cover this target yet.
        None => ReadinessVerdict::Unobserved,
        Some(state) if state.is_immediately_eligible() => ReadinessVerdict::Ready,
        Some(state) if state.transition_required => ReadinessVerdict::TransitionRequired,
        Some(_) => ReadinessVerdict::Unready,
    }
}

/// A candidate paired with its readiness standing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadyCandidate<'a> {
    /// The candidate this standing belongs to.
    pub candidate: &'a FleetCandidate,
    /// Its readiness standing against the snapshot.
    pub verdict: ReadinessVerdict,
}

impl<'a> ReadyCandidate<'a> {
    /// Whether this candidate may serve an immediate request.
    pub fn is_ready(&self) -> bool {
        self.verdict == ReadinessVerdict::Ready
    }
}

/// Applies the readiness filter over an already-statically-eligible set.
///
/// **Pure projection.** It marks candidates; it never deletes, reorders, sorts,
/// or mutates the input, and `preference_rank` plays no part. The surviving
/// order is exactly the order it arrived in, so this stays a strict filter
/// between static eligibility and the single preference-ordering seam.
///
/// This decides availability only. It does not decide context admission
/// (row 40), reasoning/dialect expressibility (F7), preference (F4), or whether
/// anything is executed (F8).
pub fn readiness_filter<'a>(
    eligible: &[CandidateEligibility<'a>],
    snapshot: &FleetSnapshot,
) -> Vec<ReadyCandidate<'a>> {
    eligible
        .iter()
        .filter(|verdict| verdict.is_eligible())
        .map(|verdict| {
            let candidate = verdict.candidate();
            ReadyCandidate {
                candidate,
                verdict: candidate_readiness(candidate, snapshot),
            }
        })
        .collect()
}

/// The subset of candidates that are presently ready, in input order.
pub fn ready_candidates<'a>(
    eligible: &[CandidateEligibility<'a>],
    snapshot: &FleetSnapshot,
) -> Vec<&'a FleetCandidate> {
    readiness_filter(eligible, snapshot)
        .into_iter()
        .filter(|entry| entry.is_ready())
        .map(|entry| entry.candidate)
        .collect()
}

// ---------------------------------------------------------------------------
// Row 40 — exact per-candidate context admission.
//
// A candidate's `usable_context_tokens` is a STATIC qualified capacity. Whether
// a particular request FITS it is a different question, and it is answered here
// using a per-candidate input-token FACT produced outside the router.
//
// The rule is production's exactly: `input + max_output <= capacity`, with
// checked arithmetic and fail-closed handling of every unknown.
//
// Crucially, an ABSENT capacity means the deployment asserts none, so there is
// nothing to enforce and the candidate passes. A present capacity with an
// absent fact is the fail-closed case. Absence of a policy is not a licence to
// guess a capacity, and no capacity is ever inferred from a provider or model
// name.
// ---------------------------------------------------------------------------

/// Why a candidate was excluded by context admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextIneligibleReason {
    /// No trusted input-token fact exists for this candidate.
    NoInputTokenFact,
    /// The request declares no explicit output budget, so a fit cannot be
    /// proven even when the input count is known.
    NoOutputBudget,
    /// `input + output` exceeds the candidate's qualified capacity.
    DoesNotFit,
    /// `input + output` overflows `u64`; treated exactly like a non-fit.
    Overflow,
}

impl std::fmt::Display for ContextIneligibleReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::NoInputTokenFact => "no per-candidate input-token fact",
            Self::NoOutputBudget => "no explicit output budget",
            Self::DoesNotFit => "input plus output exceeds the usable context",
            Self::Overflow => "input plus output overflows",
        };
        formatter.write_str(text)
    }
}

/// The exact input-token facts a host producer published for one decision.
///
/// One map per request, keyed by target model id. A count here is a *fact*: it
/// is the count for that target's own final wire representation, which is why
/// it is per-candidate rather than per-route. A missing key means "no fact",
/// never "zero".
pub type CandidateInputTokens = std::collections::BTreeMap<ModelId, u64>;

/// Whether a context-fact producer is authoritative for this decision, and the
/// facts it produced when it is.
///
/// The row-40 CONTRACT and the row-40 PRODUCER are separate. A deployment may
/// carry the contract (candidates declare `usable_context_tokens`) with no
/// producer wired at all - the current production state, R40-B dormancy. In
/// that state context admission is NOT an active filter: a bounded candidate is
/// not excluded merely for lacking a fact, because no producer exists that
/// could have supplied one.
///
/// When a producer IS authoritative the admission rule is unchanged and still
/// fail-closed: a bounded candidate with no fact of its own is excluded.
///
/// Activation is therefore carried EXPLICITLY, never inferred from the map's
/// contents. An empty map under an enabled producer means "the producer ran and
/// published no fact", which must fail closed; an empty map under a disabled
/// producer means no producer exists, which must not.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ContextFacts {
    /// No context-fact producer is wired for this route/request. Context
    /// admission does not participate.
    #[default]
    Disabled,
    /// A producer is authoritative; the inner map holds the facts it published
    /// (a missing key is a missing fact, never zero).
    Enabled(CandidateInputTokens),
}

impl ContextFacts {
    /// The authoritative fact map, or `None` when no producer is wired.
    pub fn facts(&self) -> Option<&CandidateInputTokens> {
        match self {
            ContextFacts::Disabled => None,
            ContextFacts::Enabled(facts) => Some(facts),
        }
    }

    /// Whether a producer is authoritative for this decision.
    pub fn is_enabled(&self) -> bool {
        matches!(self, ContextFacts::Enabled(_))
    }
}

impl From<CandidateInputTokens> for ContextFacts {
    /// Adopting a map means a producer was authoritative for it.
    fn from(facts: CandidateInputTokens) -> Self {
        ContextFacts::Enabled(facts)
    }
}

/// The context standing of one candidate for one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextVerdict {
    /// No capacity is declared for this candidate, so nothing is enforced.
    Unmanaged,
    /// The request provably fits the declared capacity.
    Fits,
    /// Excluded, with the specific reason.
    Ineligible(ContextIneligibleReason),
}

impl ContextVerdict {
    /// Whether this candidate may still serve the request.
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Unmanaged | Self::Fits)
    }
}

/// Pure context admission for one candidate.
///
/// `max_output` is the request's explicit `output.max_output_tokens`. Every
/// unknown resolves to *excluded*:
/// * no declared capacity -> unmanaged, always admitted;
/// * declared capacity but no input fact -> excluded;
/// * declared capacity and input fact but no output budget -> excluded,
///   because a fit cannot be proven without knowing what must be generated;
/// * `input.checked_add(output)` -> compared against the capacity;
/// * overflow -> excluded, identical to a non-fit.
pub fn candidate_context_verdict(
    candidate: &FleetCandidate,
    facts: &ContextFacts,
    max_output: Option<u64>,
) -> ContextVerdict {
    let Some(capacity) = candidate.usable_context_tokens else {
        return ContextVerdict::Unmanaged;
    };
    // No producer is wired: this admission policy is not participating, so a
    // bounded candidate is NOT rejected for a fact nobody could produce. This
    // preserves the pre-row-40 production behaviour for R40-B deployments.
    let Some(facts) = facts.facts() else {
        return ContextVerdict::Unmanaged;
    };
    let Some(input) = facts.get(&candidate.target) else {
        return ContextVerdict::Ineligible(ContextIneligibleReason::NoInputTokenFact);
    };
    let Some(output) = max_output else {
        return ContextVerdict::Ineligible(ContextIneligibleReason::NoOutputBudget);
    };
    match input.checked_add(output) {
        Some(fit) if fit <= capacity => ContextVerdict::Fits,
        Some(_) => ContextVerdict::Ineligible(ContextIneligibleReason::DoesNotFit),
        None => ContextVerdict::Ineligible(ContextIneligibleReason::Overflow),
    }
}

/// The request's explicit output budget, or `None` when it declares none.
pub fn request_output_budget(request: &Request) -> Option<u64> {
    request.llm_request.output.max_output_tokens
}

/// A candidate paired with its context standing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextAdmittedCandidate<'a> {
    /// The candidate this standing belongs to.
    pub candidate: &'a FleetCandidate,
    /// Its context standing for this request.
    pub verdict: ContextVerdict,
}

impl<'a> ContextAdmittedCandidate<'a> {
    /// Whether this candidate may still serve the request.
    pub fn is_admitted(&self) -> bool {
        self.verdict.is_admitted()
    }
}

/// Applies context admission over an already-eligible, already-ready set.
///
/// **Pure projection**, like every other filter here: it marks, never deletes,
/// and preserves input order. `preference_rank` plays no part, and no capacity
/// is ever used as a ranking signal.
pub fn context_admission_filter<'a>(
    eligible: &[CandidateEligibility<'a>],
    facts: &ContextFacts,
    max_output: Option<u64>,
) -> Vec<ContextAdmittedCandidate<'a>> {
    eligible
        .iter()
        .filter(|verdict| verdict.is_eligible())
        .map(|verdict| {
            let candidate = verdict.candidate();
            ContextAdmittedCandidate {
                candidate,
                verdict: candidate_context_verdict(candidate, facts, max_output),
            }
        })
        .collect()
}

/// Records a count for one target in a fact map.
///
/// This is the producer-side seam the host uses to populate
/// `Request::candidate_input_tokens`. A count is only recorded when the caller
/// actually obtained one; nothing here invents, estimates, or defaults a count
/// to zero.
pub fn record_input_token_fact(facts: &mut CandidateInputTokens, target: &ModelId, count: u64) {
    facts.insert(target.clone(), count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::testing::{category_models, echo, test_drive_with_models};
    use switchyard_protocol::{Category as ProtocolCategory, ReasoningParams, text_request};

    // ---- F3: production-differential eligibility truth table ---------------
    //
    // The predicates below are transcribed from the production
    // `FleetRouter::decide` filters, and the request shapes match the production
    // test fixtures (`text_req`, `tool_req`, `reasoning_req`, `none_reasoning_req`,
    // `vision_req`) so production's own results are the oracle.

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

    fn none_reasoning_req() -> Request {
        let mut req = text_req();
        req.llm_request.reasoning = ReasoningParams {
            effort: Some("none".to_string()),
            raw: None,
        };
        req
    }

    fn raw_only_reasoning_req() -> Request {
        // Production deliberately does NOT read `raw`. A request whose only
        // reasoning signal is inside `raw` must not require reasoning.
        let mut req = text_req();
        req.llm_request.reasoning = ReasoningParams {
            effort: None,
            raw: Some(serde_json::json!({ "effort": "high" })),
        };
        req
    }

    fn vision_req() -> Request {
        let mut req = text_req();
        req.llm_request.messages.push(switchyard_protocol::Message {
            role: switchyard_protocol::Role::User,
            content: vec![switchyard_protocol::ContentBlock::Image {
                source: switchyard_protocol::ImageSource::Url {
                    url: "https://example.invalid/a.png".to_string(),
                    detail: None,
                },
            }],
        });
        req
    }

    fn candidate(tool_calling: bool, reasoning: bool, supports_vision: bool) -> FleetCandidate {
        FleetCandidate {
            target: ModelId::from("c"),
            tool_calling,
            reasoning,
            supports_vision,
            preference_rank: 1,
            usable_context_tokens: None,
        }
    }

    /// Tool dimension: the request's tool declarations drive the requirement.
    #[test]
    fn tool_eligibility_matches_production() {
        let capable = candidate(true, true, true);
        let incapable = candidate(false, true, true);

        // positive: a request with tools, candidate declares tool calling
        assert!(
            candidate_is_statically_eligible(
                &capable,
                &EligibilityFacts::from_request(&tool_req())
            )
            .is_eligible()
        );
        // negative: same request, candidate does not
        assert_eq!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&tool_req())
            ),
            CandidateEligibility::Ineligible {
                candidate: &incapable,
                reason: IneligibleReason::MissingToolCalling,
            },
        );
        // boundary: no tools declared => no requirement, so a non-tool candidate
        // is NOT excluded for lacking tool calling
        assert!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&text_req())
            )
            .is_eligible()
        );
    }

    /// Reasoning dimension: normalized `effort` only; `none` and absent are not
    /// requirements, and `raw` is deliberately never consulted.
    #[test]
    fn reasoning_eligibility_matches_production() {
        let capable = candidate(true, true, true);
        let incapable = candidate(true, false, true);

        assert!(
            candidate_is_statically_eligible(
                &capable,
                &EligibilityFacts::from_request(&reasoning_req())
            )
            .is_eligible()
        );
        assert_eq!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&reasoning_req())
            ),
            CandidateEligibility::Ineligible {
                candidate: &incapable,
                reason: IneligibleReason::MissingReasoning,
            },
        );
        // boundary 1: explicit non-thinking
        assert!(!EligibilityFacts::from_request(&none_reasoning_req()).requires_reasoning);
        assert!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&none_reasoning_req())
            )
            .is_eligible()
        );
        // boundary 2: no reasoning controls at all
        assert!(!EligibilityFacts::from_request(&text_req()).requires_reasoning);
        // boundary 3: `raw` present but normalized effort absent => NOT a requirement
        assert!(!EligibilityFacts::from_request(&raw_only_reasoning_req()).requires_reasoning);
        assert!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&raw_only_reasoning_req())
            )
            .is_eligible()
        );
    }

    /// Vision dimension: image content anywhere in the messages requires it.
    #[test]
    fn vision_eligibility_matches_production() {
        let capable = candidate(true, true, true);
        let incapable = candidate(true, true, false);

        assert!(
            candidate_is_statically_eligible(
                &capable,
                &EligibilityFacts::from_request(&vision_req())
            )
            .is_eligible()
        );
        assert_eq!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&vision_req())
            ),
            CandidateEligibility::Ineligible {
                candidate: &incapable,
                reason: IneligibleReason::MissingVision,
            },
        );
        // boundary: a text-only request must not require vision
        assert!(!EligibilityFacts::from_request(&text_req()).requires_vision);
        assert!(
            candidate_is_statically_eligible(
                &incapable,
                &EligibilityFacts::from_request(&text_req())
            )
            .is_eligible()
        );
    }

    /// Each filter reports its own reason, with no invented precedence between
    /// them: a candidate failing several filters reports the first production
    /// filter that excludes it, in production's own order.
    #[test]
    fn a_candidate_failing_everything_reports_the_first_production_filter() {
        let incapable = candidate(false, false, false);
        let facts = EligibilityFacts::from_request(&tool_req());
        // Production order: tools, then reasoning, then vision. Tools is first.
        assert_eq!(
            candidate_is_statically_eligible(&incapable, &facts),
            CandidateEligibility::Ineligible {
                candidate: &incapable,
                reason: IneligibleReason::MissingToolCalling,
            },
        );
    }

    /// F3 must not reorder the stored ladder, and the derived view must preserve
    /// declared order including ineligible entries.
    #[test]
    fn eligibility_preserves_declared_order_and_marks_rather_than_removes() {
        let router = FleetRouter::new(
            vec![
                candidate(true, true, true),
                {
                    let mut c = candidate(false, true, true);
                    c.target = ModelId::from("no-tools");
                    c.preference_rank = 1; // lowest rank, but ineligible
                    c
                },
                {
                    let mut c = candidate(true, true, false);
                    c.target = ModelId::from("no-vision");
                    c.preference_rank = 1;
                    c
                },
            ],
            None,
            None,
        );
        let facts = EligibilityFacts::from_request(&tool_req());
        let verdicts = static_eligibility(&router, &facts);

        assert_eq!(verdicts.len(), 3, "every declared candidate must appear");
        // Order preserved exactly as declared.
        let order: Vec<String> = verdicts
            .iter()
            .map(|v| v.candidate().target.to_string())
            .collect();
        assert_eq!(order, ["c", "no-tools", "no-vision"]);
        // Ineligible entries are marked, not dropped, and not hoisted. Only the
        // tool filter is armed by this request, so the vision-blind candidate is
        // still statically eligible for it - exclusion is per-request, not a
        // property of the candidate.
        assert!(verdicts[0].is_eligible());
        assert!(!verdicts[1].is_eligible());
        assert!(verdicts[2].is_eligible());
        // The stored ladder is untouched.
        assert_eq!(router.candidates().len(), 3);
        assert_eq!(router.candidates()[1].target, ModelId::from("no-tools"));
    }

    /// `preference_rank` must be inert: two candidates differing ONLY in rank
    /// produce identical eligibility, and a lower-ranked eligible candidate is
    /// never moved ahead of a higher-ranked one in the derived view.
    #[test]
    fn preference_rank_is_inert_in_eligibility() {
        let mut low = candidate(true, true, true);
        low.preference_rank = 1;
        let mut high = candidate(true, true, true);
        high.preference_rank = 9;
        let facts = EligibilityFacts::from_request(&text_req());
        assert_eq!(
            candidate_is_statically_eligible(&low, &facts).is_eligible(),
            candidate_is_statically_eligible(&high, &facts).is_eligible(),
            "rank must not affect eligibility"
        );

        // Declared order is preserved regardless of rank.
        let router = FleetRouter::new(vec![high.clone(), low.clone()], None, None);
        let verdicts = static_eligibility(&router, &facts);
        let order: Vec<u16> = verdicts
            .iter()
            .map(|v| v.candidate().preference_rank)
            .collect();
        assert_eq!(order, [9, 1], "the derived view must not sort by rank");
    }

    /// Two routers over the same target with different candidate metadata must
    /// reach different eligibility, with no state shared between them.
    #[test]
    fn shared_target_is_stateless_across_routes() {
        let facts = EligibilityFacts::from_request(&vision_req());
        let vision_capable = FleetRouter::new(
            vec![{
                let mut c = candidate(true, true, true);
                c.target = ModelId::from("shared");
                c
            }],
            None,
            None,
        );
        let vision_blind = FleetRouter::new(
            vec![{
                let mut c = candidate(true, true, false);
                c.target = ModelId::from("shared");
                c
            }],
            None,
            None,
        );

        assert!(static_eligibility(&vision_capable, &facts)[0].is_eligible());
        assert!(!static_eligibility(&vision_blind, &facts)[0].is_eligible());
        // Re-evaluating the first is unaffected by having evaluated the second.
        assert!(static_eligibility(&vision_capable, &facts)[0].is_eligible());
    }

    // ---- F4: preference order over already-eligible candidates --------------
    //
    // A PROJECTION, not a final routing decision: only static eligibility has
    // run, so context-token admission (row 40), readiness, and the remaining
    // capability/reasoning constraints are still absent.

    fn ranked(target: &str, rank: u16) -> FleetCandidate {
        FleetCandidate {
            target: ModelId::from(target),
            tool_calling: true,
            reasoning: true,
            supports_vision: true,
            preference_rank: rank,
            usable_context_tokens: None,
        }
    }

    fn order_of(candidates: &[FleetCandidate]) -> Vec<String> {
        preference_order(&static_eligibility(
            &FleetRouter::new(candidates.to_vec(), None, None),
            &EligibilityFacts {
                requires_tools: false,
                requires_vision: false,
                requires_reasoning: false,
            },
        ))
        .iter()
        .map(|candidate| candidate.target.to_string())
        .collect()
    }

    /// Rank direction is lower-first, matching production's `u16` ascending
    /// compare. Pinned against production's own fixture ranks (luna 1, qwen 2,
    /// htpc 3, comfy 4), which are distinct and therefore unambiguous.
    #[test]
    fn distinct_ranks_order_lower_first() {
        let ordered = order_of(&[
            ranked("luna", 1),
            ranked("deepseek-flash", 2),
            ranked("htpc-qwen3_5", 3),
            ranked("comfyninja-qwen3_8", 4),
        ]);
        assert_eq!(
            ordered,
            [
                "luna",
                "deepseek-flash",
                "htpc-qwen3_5",
                "comfyninja-qwen3_8"
            ]
        );
    }

    /// Declaration order must NOT be the tie-break. Production breaks equal
    /// ranks by target id, and the live file contains no rank ties, so this arm
    /// is proven by fixtures rather than by the deployment.
    #[test]
    fn equal_ranks_break_ties_by_target_id_ascending() {
        // Declared deliberately in reverse target order, so a comparator that
        // fell back to declaration order would fail.
        let ordered = order_of(&[ranked("zulu", 1), ranked("alpha", 1), ranked("mike", 1)]);
        assert_eq!(ordered, ["alpha", "mike", "zulu"]);
    }

    /// The tie-break is the target's own byte-lexical `Ord`, not a
    /// case-insensitive or numeric-aware comparison.
    #[test]
    fn tie_break_is_byte_lexical_on_the_target_id() {
        let ordered = order_of(&[
            ranked("Beta", 1),
            ranked("alpha", 1),
            ranked("10-numeric", 1),
            ranked("2-numeric", 1),
        ]);
        // Uppercase 'B' (0x42) sorts before lowercase 'a' (0x61); digits sort
        // before both. This is Rust's `String` ordering, not a human collation.
        assert_eq!(ordered, ["10-numeric", "2-numeric", "Beta", "alpha"]);
    }

    /// Rank dominates the target tie-break: a higher-ranked candidate precedes a
    /// lexicographically smaller target.
    #[test]
    fn rank_dominates_the_target_tie_break() {
        let ordered = order_of(&[ranked("alpha", 9), ranked("zulu", 1)]);
        assert_eq!(
            ordered,
            ["zulu", "alpha"],
            "rank 1 must precede rank 9 regardless of name"
        );
    }

    /// Correctly-ordered and reverse-ordered input must both converge on the
    /// same result: the order is a function of the comparator, not of arrival.
    #[test]
    fn ordering_is_independent_of_input_order() {
        let in_order = vec![ranked("alpha", 1), ranked("beta", 2), ranked("gamma", 2)];
        let reversed = vec![ranked("gamma", 2), ranked("beta", 2), ranked("alpha", 1)];
        let shuffled = vec![ranked("beta", 2), ranked("alpha", 1), ranked("gamma", 2)];
        assert_eq!(order_of(&in_order), ["alpha", "beta", "gamma"]);
        assert_eq!(order_of(&reversed), order_of(&in_order));
        assert_eq!(order_of(&shuffled), order_of(&in_order));
    }

    /// Single candidate and empty set are both well defined, and the empty set
    /// must not panic or invent a candidate.
    #[test]
    fn single_and_empty_eligible_sets() {
        assert_eq!(order_of(&[ranked("only", 7)]), ["only"]);
        assert!(order_of(&[]).is_empty());
    }

    /// Ineligible candidates are excluded from the order, and the stored ladder
    /// keeps declaration order regardless of what the order looks like.
    #[test]
    fn preference_order_never_mutates_the_stored_ladder() {
        let declared = vec![
            ranked("zulu", 1),
            {
                let mut c = ranked("alpha", 2);
                c.supports_vision = false;
                c
            },
            ranked("mike", 3),
        ];
        let router = FleetRouter::new(declared.clone(), None, None);
        // Vision request: "alpha" is ineligible and must drop out of the order.
        let facts = EligibilityFacts {
            requires_tools: false,
            requires_vision: true,
            requires_reasoning: false,
        };
        let verdicts = static_eligibility(&router, &facts);
        let ordered: Vec<String> = preference_order(&verdicts)
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        assert_eq!(ordered, ["zulu", "mike"]);

        // The stored ladder is untouched: still declared order, same length, and
        // the ineligible candidate is still present.
        let stored: Vec<String> = router
            .candidates()
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        assert_eq!(stored, ["zulu", "alpha", "mike"]);
        assert_eq!(router.candidates().len(), 3);
    }

    /// `preference_rank` must change the order but never the eligible count.
    #[test]
    fn rank_alters_order_but_never_eligibility() {
        let facts = EligibilityFacts {
            requires_tools: true,
            requires_vision: true,
            requires_reasoning: true,
        };
        let mut candidates = vec![ranked("a", 1), ranked("b", 4), ranked("c", 2)];
        // A candidate lacking every capability is excluded regardless of rank.
        let mut incapable = ranked("d", 0);
        incapable.tool_calling = false;
        incapable.reasoning = false;
        incapable.supports_vision = false;
        candidates.push(incapable);

        let before = FleetRouter::new(candidates.clone(), None, None);
        let eligible_before = static_eligibility(&before, &facts)
            .iter()
            .filter(|v| v.is_eligible())
            .count();

        // Invert every rank; the eligible set must be identical in membership.
        for candidate in &mut candidates {
            candidate.preference_rank = 9 - candidate.preference_rank;
        }
        let after = FleetRouter::new(candidates, None, None);
        let verdicts_after = static_eligibility(&after, &facts);
        let eligible_after = verdicts_after.iter().filter(|v| v.is_eligible()).count();

        assert_eq!(eligible_before, 3);
        assert_eq!(eligible_after, 3, "rank must not change eligibility count");
        // And the lowest-ranked candidate is still the excluded one, at rank 9.
        let excluded = verdicts_after
            .iter()
            .find(|v| !v.is_eligible())
            .expect("the incapable candidate stays excluded");
        assert_eq!(excluded.candidate().target, ModelId::from("d"));
        assert_eq!(excluded.candidate().preference_rank, 9);
    }

    /// Rank 0 is a real, valid value, and it must not be special.
    ///
    /// A `preference_rank > 0` style filter is invisible while every rank is at
    /// least 1, so it would survive any suite drawn only from realistic
    /// configurations. This pins rank 0 as eligible and first-ordered, which is
    /// what makes that class of mutant detectable.
    #[test]
    fn rank_zero_is_eligible_and_orders_first() {
        let facts = EligibilityFacts {
            requires_tools: true,
            requires_vision: true,
            requires_reasoning: true,
        };
        let ladder = vec![ranked("second", 1), ranked("first", 0), ranked("third", 2)];
        let router = FleetRouter::new(ladder, None, None);
        let verdicts = static_eligibility(&router, &facts);
        assert!(
            verdicts.iter().all(|verdict| verdict.is_eligible()),
            "rank 0 is a capability-neutral value, not a disqualifier"
        );
        let ordered: Vec<String> = preference_order(&verdicts)
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        assert_eq!(ordered, ["first", "second", "third"]);
    }

    /// Escalation is not a ranked candidate. It is carried from F2 and is never
    /// part of the ordering input.
    #[test]
    fn escalation_is_never_part_of_the_order() {
        let router = FleetRouter::new(
            vec![ranked("alpha", 1)],
            Some(ModelId::from("switchyard-smart-turnstone")),
            Some(61_440),
        );
        let facts = EligibilityFacts {
            requires_tools: false,
            requires_vision: false,
            requires_reasoning: false,
        };
        let ordered = preference_order(&static_eligibility(&router, &facts));
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].target, ModelId::from("alpha"));
        assert!(
            !ordered
                .iter()
                .any(|c| c.target == ModelId::from("switchyard-smart-turnstone")),
            "an escalation destination must never appear in the candidate order"
        );
        // It is still carried intact.
        assert_eq!(
            router.escalation(),
            Some(&ModelId::from("switchyard-smart-turnstone"))
        );
        assert_eq!(router.escalation_max_input_tokens(), Some(61_440));
    }

    /// Two routes sharing a target must order by their own metadata, with no
    /// state keyed on target identity.
    #[test]
    fn shared_target_ordering_depends_only_on_the_owning_route() {
        let facts = EligibilityFacts {
            requires_tools: false,
            requires_vision: false,
            requires_reasoning: false,
        };
        // The shared target has rank 1 in one route and rank 4 in the other.
        let route_a = FleetRouter::new(vec![ranked("shared", 1), ranked("other", 2)], None, None);
        let route_b = FleetRouter::new(vec![ranked("other", 1), ranked("shared", 4)], None, None);
        let order_a: Vec<String> = preference_order(&static_eligibility(&route_a, &facts))
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        let order_b: Vec<String> = preference_order(&static_eligibility(&route_b, &facts))
            .iter()
            .map(|c| c.target.to_string())
            .collect();
        assert_eq!(order_a, ["shared", "other"]);
        assert_eq!(
            order_b,
            ["other", "shared"],
            "the same target ranks per-route, not globally"
        );
    }

    // ---- F2 construction + execution boundary ------------------------------

    fn router_over(names: &[&str]) -> FleetRouter {
        FleetRouter::new(
            names
                .iter()
                .map(|name| FleetCandidate {
                    target: ModelId::from(*name),
                    tool_calling: false,
                    reasoning: false,
                    supports_vision: false,
                    preference_rank: 0,
                    usable_context_tokens: None,
                })
                .collect(),
            None,
            None,
        )
    }

    // ---- F8: execution boundary -------------------------------------------
    //
    // F8 replaces the old deliberate refusal with the real decision. The tests
    // below are the execution truth table: they assert WHICH candidate is
    // selected and in WHAT ladder order, and they assert the fail-closed cases
    // that must NOT produce a selection at all.

    /// A router whose candidates are all ready, over the given names.
    fn ready_router_over(names: &[&str], ranks: &[u16]) -> FleetRouter {
        let candidates: Vec<FleetCandidate> = names
            .iter()
            .enumerate()
            .map(|(index, name)| FleetCandidate {
                target: ModelId::from(*name),
                tool_calling: true,
                reasoning: true,
                supports_vision: true,
                preference_rank: ranks[index],
                usable_context_tokens: None,
            })
            .collect();
        let states: Vec<(ModelId, CandidateState)> = names
            .iter()
            .map(|name| (ModelId::from(*name), CandidateState::ready()))
            .collect();
        FleetRouter::with_source(
            candidates,
            None,
            None,
            Arc::new(StaticFleetState::new(
                FleetSnapshot::new(states).expect("unique snapshot keys"),
            )),
        )
    }

    /// The decision, without driving a model.
    fn decide_over(router: &FleetRouter, request: &Request) -> Result<(ModelId, Vec<ModelId>)> {
        router.decide(request, &ContextFacts::Disabled)
    }

    /// One ready candidate serves, and it is the one the ladder head names.
    #[tokio::test]
    async fn one_eligible_ready_candidate_is_selected_and_served() {
        let router = ready_router_over(&["alpha"], &[0]);
        let (selected, fallbacks) = decide_over(&router, &text_req()).expect("one candidate decides");
        assert_eq!(selected, ModelId::from("alpha"));
        assert!(fallbacks.is_empty(), "a single candidate has no fallback");

        let algorithm: Arc<dyn Algorithm> = Arc::new(router);
        let (served, _) = test_drive_with_models(
            algorithm,
            text_req(),
            category_models(ProtocolCategory::Any, &["alpha"]),
            echo(),
        )
        .await
        .expect("the selected candidate serves");
        assert_eq!(served, ModelId::from("alpha"));
    }

    /// Preference order decides the head; the rest of the ladder is the tail.
    ///
    /// Declared order is deliberately NOT the order used: `gamma` is declared
    /// first but ranks worst, so the decision must pick `alpha` and carry
    /// `beta` then `gamma` as the immutable fallback order.
    #[tokio::test]
    async fn selection_follows_preference_and_returns_the_whole_ladder() {
        let router = ready_router_over(&["gamma", "beta", "alpha"], &[9, 5, 1]);
        let (selected, fallbacks) = decide_over(&router, &text_req()).expect("three candidates decide");
        assert_eq!(selected, ModelId::from("alpha"), "lowest rank is the head");
        assert_eq!(
            fallbacks,
            vec![ModelId::from("beta"), ModelId::from("gamma")],
            "the tail is the remaining ladder in preference order"
        );
    }

    /// The default constructor has no readiness producer, so it selects nothing.
    ///
    /// This is the honest fail-closed state and is what F8 inherits for every
    /// route that has not been given an injected source: an unobserved
    /// candidate is not a ready candidate.
    #[tokio::test]
    async fn a_router_without_an_injected_readiness_source_selects_nothing() {
        let router = router_over(&["alpha", "beta"]);
        let error = decide_over(&router, &text_req())
            .expect_err("an unobserved candidate must never be selected");
        assert!(
            error.to_string().contains("no immediately-eligible candidate"),
            "fail-closed error must name the missing eligibility, got: {error}"
        );

        let algorithm: Arc<dyn Algorithm> = Arc::new(router);
        match test_drive_with_models(
            algorithm,
            text_req(),
            category_models(ProtocolCategory::Any, &["alpha", "beta"]),
            echo(),
        )
        .await
        {
            Err(_) => {}
            Ok((selected, _)) => panic!(
                "a fleet route with no readiness producer must not serve, but served {selected}"
            ),
        }
    }

    /// An empty ladder refuses rather than inventing a candidate.
    #[tokio::test]
    async fn an_empty_fleet_router_refuses_too() {
        let algorithm: Arc<dyn Algorithm> = Arc::new(ready_router_over(&[], &[]));
        let error = match test_drive_with_models(
            algorithm,
            text_req(),
            category_models(ProtocolCategory::Any, &["alpha"]),
            echo(),
        )
        .await
        {
            Err(error) => error,
            Ok((selected, _)) => {
                panic!("an empty fleet route must refuse rather than serve, but served {selected}")
            }
        };
        assert!(
            error.to_string().contains("no immediately-eligible candidate"),
            "{}",
            error
        );
    }

    /// Readiness is a per-candidate fact, not a route-wide one.
    ///
    /// `alpha` is declared and ranks first but is unready, so the decision must
    /// skip it and select `beta` — and the skipped candidate must NOT appear in
    /// the fallback ladder, because it could not have served.
    #[tokio::test]
    async fn an_unready_first_candidate_is_skipped_and_not_carried_as_a_fallback() {
        let candidates = vec![
            FleetCandidate {
                target: ModelId::from("alpha"),
                tool_calling: true,
                reasoning: true,
                supports_vision: true,
                preference_rank: 1,
                usable_context_tokens: None,
            },
            FleetCandidate {
                target: ModelId::from("beta"),
                tool_calling: true,
                reasoning: true,
                supports_vision: true,
                preference_rank: 2,
                usable_context_tokens: None,
            },
        ];
        let snapshot = FleetSnapshot::new(vec![
            (ModelId::from("alpha"), CandidateState::not_ready()),
            (ModelId::from("beta"), CandidateState::ready()),
        ])
        .expect("unique keys");
        let router = FleetRouter::with_source(
            candidates,
            None,
            None,
            Arc::new(StaticFleetState::new(snapshot)),
        );

        let (selected, fallbacks) = decide_over(&router, &text_req()).expect("beta still decides");
        assert_eq!(selected, ModelId::from("beta"));
        assert!(
            fallbacks.is_empty(),
            "an unready candidate is excluded, not carried, got: {fallbacks:?}"
        );
    }

    /// A candidate in a required transition is not immediately eligible.
    #[tokio::test]
    async fn a_transition_required_candidate_is_not_selectable() {
        let candidates = vec![FleetCandidate {
            target: ModelId::from("alpha"),
            tool_calling: true,
            reasoning: true,
            supports_vision: true,
            preference_rank: 1,
            usable_context_tokens: None,
        }];
        let snapshot =
            FleetSnapshot::new(vec![(ModelId::from("alpha"), CandidateState::transition_required())])
                .expect("unique keys");
        let router =
            FleetRouter::with_source(candidates, None, None, Arc::new(StaticFleetState::new(snapshot)));
        assert!(
            decide_over(&router, &text_req()).is_err(),
            "a candidate mid-transition must not be selected for an immediate request"
        );
    }

    /// Capability filtering happens before the ladder is built, so a request
    /// that needs tools cannot be served by a tool-blind candidate even when it
    /// is the only ready one.
    #[tokio::test]
    async fn capability_failure_excludes_a_candidate_before_selection() {
        let candidates = vec![FleetCandidate {
            target: ModelId::from("blind"),
            tool_calling: false,
            reasoning: true,
            supports_vision: true,
            preference_rank: 1,
            usable_context_tokens: None,
        }];
        let snapshot =
            FleetSnapshot::new(vec![(ModelId::from("blind"), CandidateState::ready())]).expect("key");
        let router =
            FleetRouter::with_source(candidates, None, None, Arc::new(StaticFleetState::new(snapshot)));
        assert!(
            decide_over(&router, &tool_req()).is_err(),
            "a tool request must not be served by a tool-blind candidate"
        );
    }

    /// The decision is computed from ONE snapshot, even when the source changes
    /// between two decisions.
    ///
    /// A shared source lets a test mutate readiness between calls; each call must
    /// reflect the generation current at that call, and must not blend the two.
    #[tokio::test]
    async fn each_decision_reads_exactly_one_snapshot_generation() {
        let candidates = vec![
            FleetCandidate {
                target: ModelId::from("alpha"),
                tool_calling: true,
                reasoning: true,
                supports_vision: true,
                preference_rank: 1,
                usable_context_tokens: None,
            },
            FleetCandidate {
                target: ModelId::from("beta"),
                tool_calling: true,
                reasoning: true,
                supports_vision: true,
                preference_rank: 2,
                usable_context_tokens: None,
            },
        ];
        let shared = Arc::new(SharedFleetState::new(
            FleetSnapshot::new(vec![
                (ModelId::from("alpha"), CandidateState::ready()),
                (ModelId::from("beta"), CandidateState::ready()),
            ])
            .expect("keys"),
        ));
        let router = FleetRouter::with_source(
            candidates,
            None,
            None,
            Arc::clone(&shared) as Arc<dyn FleetStateSource>,
        );

        let (first_selected, _) = decide_over(&router, &text_req()).expect("first decides");
        assert_eq!(first_selected, ModelId::from("alpha"));

        // Replace the whole snapshot between decisions.
        shared.set(
            FleetSnapshot::new(vec![
                (ModelId::from("alpha"), CandidateState::not_ready()),
                (ModelId::from("beta"), CandidateState::ready()),
            ])
            .expect("keys"),
        );

        let (second_selected, fallbacks) = decide_over(&router, &text_req()).expect("second decides");
        assert_eq!(
            second_selected,
            ModelId::from("beta"),
            "a NEW generation is visible to the NEXT decision"
        );
        assert!(fallbacks.is_empty(), "alpha is now excluded, not carried");
    }

    /// Row 40 stays dormant: with NO PRODUCER WIRED, context admission is not
    /// an active filter, so a `Bounded` candidate is admitted on readiness
    /// alone rather than excluded for a fact nobody could produce.
    ///
    /// This is the 0.3.0 cutover defect. The previous expectation here asserted
    /// the opposite - that a bounded candidate with no producer stays excluded -
    /// which is exactly what zeroed every bounded production route and returned
    /// "no immediately-eligible candidate". The row-40 CONTRACT may exist
    /// without the row-40 PRODUCER; a dormant producer means the admission
    /// policy is not participating, not that every candidate lacks evidence.
    #[tokio::test]
    async fn a_bounded_candidate_with_no_producer_is_admitted() {
        let candidates = vec![FleetCandidate {
            target: ModelId::from("bounded"),
            tool_calling: true,
            reasoning: true,
            supports_vision: true,
            preference_rank: 1,
            usable_context_tokens: Some(4096),
        }];
        let snapshot = FleetSnapshot::new(vec![(ModelId::from("bounded"), CandidateState::ready())])
            .expect("key");
        let router =
            FleetRouter::with_source(candidates, None, None, Arc::new(StaticFleetState::new(snapshot)));

        let (selected, _) = decide_over(&router, &text_req())
            .expect("a bounded candidate is admitted while the producer is dormant");
        assert_eq!(selected, ModelId::from("bounded"));
    }

    /// With a producer AUTHORITATIVE, the same bounded candidate fails closed
    /// when its fact is missing. Dormancy and fail-closed are distinct.
    #[tokio::test]
    async fn a_bounded_candidate_with_an_active_producer_and_no_fact_is_excluded() {
        let candidates = vec![FleetCandidate {
            target: ModelId::from("bounded"),
            tool_calling: true,
            reasoning: true,
            supports_vision: true,
            preference_rank: 1,
            usable_context_tokens: Some(4096),
        }];
        let snapshot = FleetSnapshot::new(vec![(ModelId::from("bounded"), CandidateState::ready())])
            .expect("key");
        let router =
            FleetRouter::with_source(candidates, None, None, Arc::new(StaticFleetState::new(snapshot)));

        assert!(
            router
                .decide(&text_req(), &ContextFacts::Enabled(CandidateInputTokens::new()))
                .is_err(),
            "an active producer that published no fact must fail closed, never assume a fit"
        );
    }

    /// An UNMANAGED candidate (no declared capacity) is unaffected by the
    /// dormant producer, so the dormancy does not disable the whole fleet.
    #[tokio::test]
    async fn an_unmanaged_candidate_is_unaffected_by_row_40_dormancy() {
        let router = ready_router_over(&["alpha"], &[0]);
        let (selected, _) = decide_over(&router, &text_req()).expect("unmanaged decides");
        assert_eq!(selected, ModelId::from("alpha"));
    }

    /// Escalation is host-owned: the destination is never part of the ladder.
    #[tokio::test]
    async fn escalation_destination_is_never_a_ladder_entry() {
        let candidates = vec![FleetCandidate {
            target: ModelId::from("alpha"),
            tool_calling: true,
            reasoning: true,
            supports_vision: true,
            preference_rank: 1,
            usable_context_tokens: None,
        }];
        let snapshot =
            FleetSnapshot::new(vec![(ModelId::from("alpha"), CandidateState::ready())]).expect("key");
        let router = FleetRouter::with_source(
            candidates,
            Some(ModelId::from("escape-hatch")),
            Some(8192),
            Arc::new(StaticFleetState::new(snapshot)),
        );

        let (selected, fallbacks) = decide_over(&router, &text_req()).expect("decides");
        assert_eq!(selected, ModelId::from("alpha"));
        assert!(
            fallbacks.is_empty(),
            "the escalation destination is a separate route, never a fallback, got: {fallbacks:?}"
        );
        assert_eq!(router.escalation(), Some(&ModelId::from("escape-hatch")));
        assert_eq!(router.escalation_max_input_tokens(), Some(8192));
    }

    /// Ladder order and metadata are the representation's whole job here.
    #[test]
    fn the_ladder_is_preserved_in_declared_order() {
        let router = FleetRouter::new(
            vec![
                FleetCandidate {
                    target: ModelId::from("first"),
                    tool_calling: true,
                    reasoning: true,
                    supports_vision: true,
                    preference_rank: 2,
                    usable_context_tokens: Some(4096),
                },
                FleetCandidate {
                    target: ModelId::from("second"),
                    tool_calling: false,
                    reasoning: false,
                    supports_vision: false,
                    preference_rank: 1,
                    usable_context_tokens: None,
                },
            ],
            Some(ModelId::from("escape")),
            Some(8192),
        );

        let ladder = router.candidates();
        assert_eq!(ladder.len(), 2);
        // Declared order, not preference order: preference_rank 2 comes first.
        assert_eq!(ladder[0].target, ModelId::from("first"));
        assert_eq!(ladder[1].target, ModelId::from("second"));
        assert!(ladder[0].tool_calling && ladder[0].reasoning && ladder[0].supports_vision);
        assert_eq!(ladder[0].usable_context_tokens, Some(4096));
        assert_eq!(ladder[1].usable_context_tokens, None);
        assert_eq!(router.escalation(), Some(&ModelId::from("escape")));
        assert_eq!(router.escalation_max_input_tokens(), Some(8192));
    }
}
