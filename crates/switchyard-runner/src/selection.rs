// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Choosing among candidates a route has already admitted.
//!
//! This is not a router and deliberately cannot become one. Every hard gate — privacy
//! clearance, media capability, physical context fit, credential scope — runs before a
//! candidate reaches this module. What is left is the choice between interchangeable models
//! that can all serve the request, and that choice is not the manifest's spelling order.
//!
//! A route over a provider's free tier has several models that cost the same and differ in
//! the resource each one makes scarce: one has a large daily request allowance and a small
//! per-minute token budget, another the reverse. Serving every small request with whichever
//! model happens to be typed first spends the scarce resource of the model that was never
//! given a chance. So the order is computed from facts plus measurement:
//!
//! * **Cost tier is structural.** A paid candidate never moves ahead of an eligible free
//!   one, whatever it has measured: the tier is part of the sort key, not a weight, so no
//!   amount of evidence can cross it. Within a tier, the cheapest *appropriate* model wins.
//! * **A request is costed as the conversation it will become.** A small input still carries
//!   its output reservation, and a tool exchange is a conversation that grows, so the first
//!   turn is costed as two. A target whose per-request ceiling would be mostly consumed by
//!   that demand is demoted in favour of one with headroom.
//! * **Daily headroom is the economy signal.** Between two free candidates that fit, the one
//!   with more requests left in its own daily budget is preferred, which is what makes a
//!   14,400-request model the default for small bounded work instead of a 500-request one.
//! * **Measurement can only demote or tie-break.** A candidate this deployment has observed
//!   failing is moved behind one it has observed completing, and latency separates candidates
//!   the economy keys cannot.
//!
//! Three properties keep it from doing harm:
//!
//! * **No data is not bad data.** A candidate with too few observations is never demoted for
//!   being unmeasured; evidence only reorders candidates it can speak about.
//! * **It only reorders.** It never adds, drops or invents a candidate, and it never
//!   overrides a failure class the client's own retry and cooldown logic already decided.
//! * **The declared order is the final tie-break**, so two candidates with identical facts
//!   and identical evidence keep the manifest's order.
//!
//! The evidence is per process and intentionally not persisted: a latency budget measured
//! before a provider incident is not evidence about the provider now. It is shared by every
//! measured route, because a physical model behaves the same whichever alias named it — so
//! real bounded work on one route is evidence for the others.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_llm_client::RunObservation;
use switchyard_protocol::ModelId;

/// Observations before a candidate may be demoted for failing.
///
/// Below this the sample says more about one request than about the model, and reordering
/// on it would make routing order noisy exactly when a deployment is newest.
const MIN_OBSERVATIONS: u32 = 3;

/// How many eligible adaptive opportunities a ready group may be starved behind a healthier
/// peer before it is offered one bounded recovery dispatch.
///
/// An "opportunity" is a real dispatch that ranked this pool (`advance == true`), not a
/// global tick and not a diagnostic preview: the count is scoped to the group's own pool
/// traffic, so two routes over the same groups cannot double-spend it and a preview cannot
/// move it.
///
/// Without this, a group that failed during an outage (e.g. allowance exhaustion) carries
/// its old negative Evidence forever: `completion < 1000` keeps it behind a healthy peer,
/// it is never dispatched, so it never earns the new observations that would restore its
/// ranking, and rotation (the last sort key) never breaks the tie because the groups are
/// not tied. The recovery window offers exactly one ordinary-traffic dispatch after the
/// bound, judged on the evidence that dispatch measures; it does not reset the group's
/// Evidence (other routes share it), is not a background probe, and is not a forced
/// allowance spend.
const RECOVERY_WINDOW: u64 = 8;

/// Weight of the newest latency sample in the average.
///
/// A slow minute should move the ranking without erasing everything measured before it,
/// which is what a plain last-value comparison would do.
const LATENCY_WEIGHT: f64 = 0.25;

/// The share of a candidate's per-request ceiling one conversation may claim before the
/// candidate is treated as high-pressure.
///
/// A request that needs more than half of the target's minute leaves no room for the
/// continuation that answers it, so a cheaper-on-paper model is not the economical choice
/// for it. This is a selection threshold, not an admission rule: the hard per-request gate
/// still refuses a request that does not fit at all.
const PRESSURE_CEILING_PER_MILLE: u64 = 500;

/// What a tool exchange costs relative to a single text turn.
///
/// A tool-calling turn is answered by another turn carrying the result, so budgeting one
/// turn would choose a target for the first turn that the second cannot hold. Costing it as
/// two is the smallest statement of that fact that does not invent a token estimate.
const TOOL_TURN_GROWTH: u64 = 2;

/// Whether a provider charges for a request to a candidate.
///
/// Ordered, because the ordering is the policy: a paid request never precedes a free one.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum CostTier {
    /// Served inside a free allowance. The provider charges nothing per request; the
    /// allowance itself is the scarce resource.
    Free,
    /// A metered request.
    #[default]
    Paid,
}

/// How the comparator breaks a tie, and what it assumes when a candidate's own
/// remaining allowance is unknown.
///
/// The default is the behaviour every existing route already had, so a route that does not
/// ask for anything new keeps it. [`RankingPolicy::PEER`] is what a pool assembled across
/// providers asks for: with the manifest listed provider by provider, the declared order is
/// not a decision, and treating it as one would hand the first-typed provider every tie.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RankingPolicy {
    /// What separates two candidates whose facts and evidence are otherwise equal.
    pub tie_break: TieBreak,
    /// Where a candidate goes when the deployment has not stated its remaining allowance.
    pub unknown_headroom: UnknownHeadroom,
    /// Whether declared groups order the candidates before the leaf keys do.
    pub grouping: Grouping,
}

/// The ranking policy a grouped pool of the given shape uses.
///
/// Kept as a free function so the dispatch path cannot ask for the adaptive policy and then
/// rank with a different one: the [`Grouping`] value the group order was computed with is the
/// same value that selects the leaf keys.
pub fn grouped_policy(grouping: Grouping) -> RankingPolicy {
    match grouping {
        Grouping::Ordered => RankingPolicy::ORDERED_GROUPS,
        Grouping::Flat | Grouping::Adaptive => RankingPolicy::GROUPED,
    }
}

impl RankingPolicy {
    /// The historical policy: declared order breaks ties, and an undeclared budget is
    /// treated as `u64::MAX` claimed requests for ordering purposes.
    ///
    /// The second half is not a claim that the allowance is infinite - it is the behaviour
    /// every route had before a budget could be declared per target, preserved so no owner
    /// route reorders because a stricter option became available.
    pub const LEGACY: Self = Self {
        tie_break: TieBreak::Declared,
        unknown_headroom: UnknownHeadroom::Unbounded,
        grouping: Grouping::Flat,
    };

    /// The provider-neutral policy a mixed free pool uses.
    pub const PEER: Self = Self {
        tie_break: TieBreak::CanonicalId,
        unknown_headroom: UnknownHeadroom::AfterKnown,
        grouping: Grouping::Flat,
    };

    /// The policy a pool of several declared GROUPS uses.
    ///
    /// The leaf keys are unchanged, but they only order candidates *within* one group. Between
    /// groups the comparison is one the leaf keys cannot make honestly: two providers do not
    /// share a request budget, so "more requests left" is not a common yardstick, and a group
    /// with four leaves would win a per-leaf count for reasons that have nothing to do with
    /// which provider is the better choice. The group order is computed separately from
    /// readiness, measured completion rate, measured latency, and a rotation floor so that two
    /// equally-evidenced groups are actually both used.
    pub const GROUPED: Self = Self {
        tie_break: TieBreak::CanonicalId,
        unknown_headroom: UnknownHeadroom::AfterKnown,
        grouping: Grouping::Adaptive,
    };

    /// The policy a pool of ORDERED groups uses: the declared group sequence is the priority
    /// order, and a group is exhausted before the next one is offered.
    ///
    /// This is the cost-efficiency shape rather than the equal-peer shape. Its groups are not
    /// interchangeable - the later ones cost more - so no evidence may let an expensive group
    /// jump ahead of a cheap group that can still answer. Inside a group the leaf keys still
    /// choose the best member for the request, which is what keeps the preference order from
    /// becoming a fixed leaf order.
    pub const ORDERED_GROUPS: Self = Self {
        tie_break: TieBreak::CanonicalId,
        unknown_headroom: UnknownHeadroom::AfterKnown,
        grouping: Grouping::Ordered,
    };
}

/// How declared pool groups participate in the order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Grouping {
    /// One peer set, as every route before groups did: the leaf keys decide everything.
    #[default]
    Flat,
    /// Candidates keep their group; the group is chosen first and the leaf keys order its
    /// members. Equal peers: the group order adapts to measured readiness, completion rate and
    /// latency, and rotates between equals. Only meaningful with a [`GroupPlan`].
    Adaptive,
    /// Candidates keep their group, and the declared group sequence is the order. A group is
    /// exhausted before the next is offered; evidence may reorder leaves inside a group but
    /// never promotes a later group over an earlier one. Only meaningful with a [`GroupPlan`].
    Ordered,
}

impl Grouping {
    /// The manifest spelling, and the inverse of [`Self::from_name`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Adaptive => "adaptive",
            Self::Ordered => "ordered",
        }
    }

    /// Reads a manifest spelling. `None` for an unknown word rather than a default, so a
    /// misspelling is a load refusal instead of a silent behaviour change.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "flat" => Some(Self::Flat),
            "adaptive" => Some(Self::Adaptive),
            "ordered" => Some(Self::Ordered),
            _ => None,
        }
    }
}

/// One declared group of a grouped pool.
#[derive(Clone, Debug)]
pub struct GroupMembers {
    /// The group label, as the manifest and its reports spell it.
    pub name: String,
    /// The models that belong to it, by canonical target identity.
    pub members: Vec<ModelId>,
}

/// The groups a grouped pool is built from.
#[derive(Clone, Debug, Default)]
pub struct GroupPlan {
    /// Declared groups, in the order the route named them.
    pub groups: Vec<GroupMembers>,
}

/// Where each group was last offered first, so two equally-evidenced groups alternate.
///
/// Process-wide and keyed by group label: the label names physical capacity, so two routes
/// over the same groups should not each start their own rotation and double the traffic one
/// of them sees.
#[derive(Default)]
struct GroupRotation {
    /// Monotonic dispatch counter for the rotation floor (`last_served`), so equally-evidenced
    /// groups alternate. This is the tie-break rotation only; it is NOT the starvation clock.
    tick: u64,
    last_served: HashMap<String, u64>,
    /// Per-group count of eligible adaptive opportunities (dispatches that actually ranked
    /// this pool) in which the group was ready but not chosen. When the count reaches
    /// [`RECOVERY_WINDOW`] the group is offered exactly one recovery dispatch, then the
    /// count resets. Scoped to the group's own pool opportunities, never a global tick, and
    /// only advanced on a real dispatch (`advance == true`), so diagnostics and previews are
    /// strictly non-mutating. Keyed by group label, so two routes over the same groups share
    /// one recovery budget rather than each double-spending it.
    starved_opportunities: HashMap<String, u64>,
}

/// What separates two candidates the facts and the evidence cannot order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TieBreak {
    /// The manifest's own order.
    ///
    /// Correct when an operator wrote the list as a decision. Wrong for a pool assembled
    /// from several providers, where the order is an artifact of how the file was written.
    #[default]
    Declared,
    /// The candidate's own stable identity.
    ///
    /// Chosen so that permuting the manifest cannot change the outcome: the same set of
    /// candidates produces the same choice whatever order they were typed in. It is a
    /// fairness floor, not a preference - the name carries no provider advantage beyond
    /// being stable, and it only decides cases the measured facts could not.
    CanonicalId,
}

/// Where a candidate sorts when no remaining allowance is declared for it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UnknownHeadroom {
    /// Ahead of any declared budget, as if it had `u64::MAX` requests left.
    ///
    /// The historical ordering. Kept as the default so owner routes keep their behaviour,
    /// and documented as the assumption it is rather than as a fact.
    #[default]
    Unbounded,
    /// Behind every candidate whose headroom is actually verifiable, ahead of an exhausted one.
    ///
    /// An undeclared budget is not evidence of capacity, so it cannot be allowed to outrank
    /// a candidate the deployment can prove still has allowance. It also must not be treated
    /// as exhausted - that would invent a refusal the deployment never stated - so it sorts
    /// after the known-positive candidates and before a candidate whose own count reached zero.
    AfterKnown,
}

/// A comparable tie key, so the two tie-break policies share one sort.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TieKey {
    /// Position in the manifest. Only compared against another `Declared`.
    Declared(usize),
    /// Canonical model id. Only compared against another `CanonicalId`.
    Canonical(String),
}

/// The deployment facts about one admitted candidate that decide its order.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// The physical model the route would call.
    pub model: ModelId,
    /// Whether a request here is free within an allowance or metered.
    pub tier: CostTier,
    /// Requests this candidate's own daily budget still permits, when it declares one.
    ///
    /// `None` means no local budget is declared, not an unlimited provider allowance:
    /// a paid target is ordered by tier before this is ever consulted.
    pub remaining_today: Option<u64>,
    /// The most input one request may carry here, when the provider caps it below the
    /// model's own window.
    pub max_input_tokens: Option<u64>,
    /// The model's own window, used when no lower per-request ceiling is declared.
    pub context_window: Option<u64>,
}

impl Candidate {
    /// The capacity a single request is measured against, when the deployment states one.
    ///
    /// The per-request ceiling is the number that binds when both are declared: a model can
    /// hold 262,144 tokens and still accept only 16,000 of them in one minute.
    pub fn ceiling(&self) -> Option<u64> {
        self.max_input_tokens.or(self.context_window)
    }

    /// The share of the candidate's ceiling one conversation would claim, in thousandths.
    ///
    /// A target that states no ceiling claims no pressure: the deployment has not said what
    /// would be tight, so this module cannot invent a limit for it.
    fn pressure_per_mille(&self, demand: Demand) -> u64 {
        pressure_permille(demand.required_tokens(), self.ceiling())
    }
}

/// The share of `ceiling` that `required_tokens` would claim, in thousandths.
///
/// Public because the selection trace reports the figure the comparator actually ranked
/// by: computing it twice would let the explanation disagree with the decision.
pub fn pressure_permille(required_tokens: u64, ceiling: Option<u64>) -> u64 {
    let Some(ceiling) = ceiling.filter(|ceiling| *ceiling > 0) else {
        return 0;
    };
    required_tokens
        .saturating_mul(1000)
        .saturating_div(ceiling)
        .min(u64::MAX / 1000)
}

/// What the request will ask of whichever candidate serves it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Demand {
    /// Estimated input tokens plus the caller's output reservation.
    pub required_tokens: u64,
    /// Whether the request carries tools or a tool choice, so the conversation will grow.
    pub needs_tools: bool,
}

impl Demand {
    /// The tokens this conversation is budgeted against a candidate's ceiling.
    ///
    /// Public because the selection trace reports the same budget the comparator ranked
    /// against, including the tool-turn growth factor.
    pub fn required_tokens(self) -> u64 {
        if self.needs_tools {
            self.required_tokens.saturating_mul(TOOL_TURN_GROWTH)
        } else {
            self.required_tokens
        }
    }
}

/// What this deployment has measured about one target.
#[derive(Clone, Copy, Debug, Default)]
struct Evidence {
    calls: u32,
    successes: u32,
    latency_ms: f64,
    /// Calls that returned no usable answer at all.
    ///
    /// Counted separately from a generic failure because it is not noise: the same model
    /// under the same budget reproduces the same non-answer, so a single observation is
    /// enough to stop preferring it. Measured 2026-11-21: hosted Gemma 4 26B spent a
    /// 900-token budget on inline reasoning and answered with nothing on every attempt.
    empty_answers: u32,
}

impl Evidence {
    fn record(&mut self, success: bool, latency_ms: f64, empty_answer: bool) {
        self.calls = self.calls.saturating_add(1);
        if success {
            self.successes = self.successes.saturating_add(1);
        }
        if empty_answer {
            self.empty_answers = self.empty_answers.saturating_add(1);
        }
        self.latency_ms = if self.calls == 1 {
            latency_ms
        } else {
            self.latency_ms + LATENCY_WEIGHT * (latency_ms - self.latency_ms)
        };
    }

    /// Completion rate with a prior, so one lucky call is not a perfect score.
    fn reliability(&self) -> f64 {
        f64::from(self.successes + 1) / f64::from(self.calls + 2)
    }

    /// Whether the sample is large enough to demote the candidate.
    fn is_measured(&self) -> bool {
        self.calls >= MIN_OBSERVATIONS
    }

    /// Whether a measured candidate is failing more often than it completes.
    ///
    /// The threshold is deliberately blunt. A model that answers half the time is still the
    /// wrong first choice when another one is answering every time, and a finer grade would
    /// need a quality signal this layer does not have.
    fn is_unhealthy(&self) -> bool {
        self.is_measured() && self.reliability() < 0.5
    }

    fn latency_millis(&self) -> u64 {
        self.latency_ms.max(0.0) as u64
    }

    /// Whether this target has already answered with nothing at least once.
    ///
    /// Unlike [`Self::is_unhealthy`] this needs no minimum sample: an empty answer is a
    /// measured, repeatable outcome, not one unlucky request.
    fn has_empty_answer(&self) -> bool {
        self.empty_answers > 0
    }
}

/// The ranking facts for one candidate, in dispatch order.
///
/// Every field is a fact the comparator actually used, so a report built from these can
/// state why a candidate ended up where it did rather than describing the manifest.
#[derive(Clone, Debug)]
pub struct CandidateScore {
    /// The candidate.
    pub model: ModelId,
    /// Cost tier - the first and strongest component.
    pub tier: CostTier,
    /// Its own daily allowance is spent.
    pub exhausted: bool,
    /// It has been observed returning no usable answer.
    pub empty_answer: bool,
    /// It has been measured failing more often than completing.
    pub unhealthy: bool,
    /// One conversation would claim more than half its per-request ceiling.
    pub over_pressure: bool,
    /// Requests its own daily budget still permits, when declared.
    pub remaining_today: Option<u64>,
    /// Thousandths of its ceiling this conversation claims.
    pub pressure_permille: u64,
    /// Measured latency, once there is a sample.
    pub latency_ms: Option<u64>,
    /// Where the manifest declared it.
    pub declared_position: usize,
    /// Where it was actually dispatched.
    pub rank: usize,
    /// The first component explaining a fall below `declared_position`.
    pub demotion: Option<&'static str>,
}

/// Process-wide record of how each physical target has actually behaved.
#[derive(Default)]
pub struct MeasuredSelector {
    evidence: Mutex<HashMap<ModelId, Evidence>>,
    /// Rotation floor for grouped pools, so equally-evidenced groups are both used.
    rotation: Mutex<GroupRotation>,
}

impl MeasuredSelector {
    /// A selector with no observations, which therefore only applies the declared policy.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Records one completed call, when the observation describes one.
    ///
    /// Only a call that finished is evidence. A routing-time `LlmCall` is observed on the
    /// same path as the terminal answer, so both are recorded: either one failing is a fact
    /// about the model that was asked.
    pub fn observe(&self, observation: &RunObservation) {
        let call = match observation {
            RunObservation::LlmCall(call) | RunObservation::AnswerCall(call) => call,
            RunObservation::Outcome(_)
            | RunObservation::RoutingOverhead(_)
            | RunObservation::ContextPressurePromotion { .. }
            | RunObservation::Selection(_) => return,
        };
        let mut evidence = self.evidence.lock();
        evidence
            .entry(call.selected_model.clone())
            .or_default()
            .record(
                call.is_success,
                call.duration.as_secs_f64() * 1000.0,
                // The client reports `empty_answer*` codes for a reply that carried no
                // usable content, which is the one failure this layer can act on
                // immediately rather than waiting for a sample. `empty_answer_thinking` is
                // excluded on purpose: on a target that charges its own reasoning against
                // the caller's output reservation, that reply is a fact about the request's
                // budget, not about the model's health. Counting it demoted a working Gemma
                // leg out of every later request - including the 2,048-token judge calls it
                // answers in two to eight seconds - because two sixteen-token probes had
                // spent their whole budget on thinking (measured 2026-11-21: six demotions
                // from `empty_answers`). The pre-dispatch floor keeps that request shape
                // away from the leg instead.
                call.outcome.is_some_and(|code| {
                    code.starts_with("empty_answer") && code != "empty_answer_thinking"
                }),
            );
    }

    /// Orders `candidates` best-first, keeping `declared` order for everything the policy
    /// and the sample cannot separate.
    ///
    /// Returns `None` when the computed order is the order already offered, so the caller
    /// can leave the list untouched rather than rewiring it into an identical order.
    pub fn order(
        &self,
        candidates: &[Candidate],
        demand: Demand,
        declared: &[ModelId],
    ) -> Option<Vec<ModelId>> {
        if candidates.len() < 2 {
            return None;
        }
        let scored = self.explain(candidates, demand, declared);
        let ordered: Vec<ModelId> = scored.into_iter().map(|score| score.model).collect();
        let unchanged = ordered
            .iter()
            .zip(candidates.iter())
            .all(|(left, right)| left == &right.model);
        (!unchanged).then_some(ordered)
    }

    /// The same ordering [`Self::order`] applies, returned with the facts that produced it.
    ///
    /// `order` is expressed through this so the order that is dispatched and the order
    /// that is reported cannot drift apart: there is one comparator, and this exposes
    /// its inputs rather than re-deriving them.
    pub fn explain(
        &self,
        candidates: &[Candidate],
        demand: Demand,
        declared: &[ModelId],
    ) -> Vec<CandidateScore> {
        self.explain_with(candidates, demand, declared, RankingPolicy::LEGACY)
    }

    /// [`Self::explain`] under an explicit ranking policy.
    ///
    /// The policy decides only what the facts leave open: the tie-break and the treatment of
    /// an undeclared budget. Every gate and every measured component stays the same, so a
    /// route opting into [`RankingPolicy::PEER`] changes no eligibility and no score.
    pub fn explain_with(
        &self,
        candidates: &[Candidate],
        demand: Demand,
        declared: &[ModelId],
        policy: RankingPolicy,
    ) -> Vec<CandidateScore> {
        self.explain_with_ranks(candidates, demand, declared, policy, None)
    }

    /// The order a pool of declared GROUPS dispatches in.
    ///
    /// One request, one rotation step: the head group is recorded as served here, which is
    /// the only place the decision is actually made. Diagnostics must use
    /// [`Self::explain_grouped_preview`] instead so that asking what the order *would* be
    /// cannot spend another route's turn.
    pub fn explain_grouped(
        &self,
        candidates: &[Candidate],
        demand: Demand,
        declared: &[ModelId],
        plan: &GroupPlan,
        grouping: Grouping,
    ) -> Vec<CandidateScore> {
        let ranks = self.group_ranks(candidates, plan, grouping, true);
        self.explain_with_ranks(
            candidates,
            demand,
            declared,
            grouped_policy(grouping),
            Some(&ranks),
        )
    }

    /// [`Self::explain_grouped`] without advancing the rotation.
    pub fn explain_grouped_preview(
        &self,
        candidates: &[Candidate],
        demand: Demand,
        declared: &[ModelId],
        plan: &GroupPlan,
        grouping: Grouping,
    ) -> Vec<CandidateScore> {
        let ranks = self.group_ranks(candidates, plan, grouping, false);
        self.explain_with_ranks(
            candidates,
            demand,
            declared,
            grouped_policy(grouping),
            Some(&ranks),
        )
    }

    /// Orders the declared groups and returns each admitted model's group rank.
    ///
    /// The adaptive comparison uses only what is comparable between two providers: whether
    /// the group has anything left to offer at all, whether every one of its measured members
    /// is failing, its measured completion rate, and how fast its measured members have been.
    /// Request budgets are deliberately absent: a declared daily budget is a fact about one
    /// provider's allowance, and using it to order *between* providers would turn "we can count
    /// this one" into "prefer this one", which is the four-leaf-first defect in another
    /// spelling.
    ///
    /// An ordered pool does not compare groups at all: the declared sequence is the priority
    /// order, so a later group is never offered while an earlier one has a member left. That is
    /// the cost-efficiency shape, where the groups differ in price rather than in quality.
    ///
    /// `advance` records the chosen head so the next request starts from the other group when
    /// the evidence cannot separate them. Only an adaptive pool rotates; an ordered pool has
    /// nothing to rotate between.
    fn group_ranks(
        &self,
        candidates: &[Candidate],
        plan: &GroupPlan,
        grouping: Grouping,
        advance: bool,
    ) -> HashMap<ModelId, u32> {
        let evidence = self.evidence.lock();
        let by_model: HashMap<&ModelId, &Candidate> =
            candidates.iter().map(|c| (&c.model, c)).collect();
        let mut rotation = self.rotation.lock();
        let ordered = grouping == Grouping::Ordered;
        let mut order: Vec<(u32, bool, u32, u64, u64, String)> = Vec::new();
        // Groups that are ready but measurably behind a healthier peer this dispatch: the
        // only groups whose starvation counter advances. Collected here and applied after the
        // sort so the counter update is gated on `advance` and never runs for a preview.
        let mut recovery_eligible: Vec<String> = Vec::new();
        for (index, group) in plan.groups.iter().enumerate() {
            let members: Vec<&Candidate> = group
                .members
                .iter()
                .filter_map(|model| by_model.get(model).copied())
                .collect();
            if members.is_empty() {
                continue;
            }
            // A group is ready while any member still has allowance to spend. `None` is an
            // undeclared budget, not a spent one, so it keeps the group ready: refusing on
            // an undeclared number would invent an exhaustion the deployment never stated.
            let ready = members
                .iter()
                .any(|candidate| candidate.remaining_today != Some(0));
            // A group's measured completion rate, as per-mille with the same prior the leaf
            // keys use, averaged over its MEASURED members. A group with no sample at all
            // reads 1000: no data is not bad data, and demoting it would make the first
            // requests of a process a self-fulfilling prophecy.
            //
            // This one key replaces the "every measured member is failing" flag the first
            // draft carried. That flag could never change an order once this number exists:
            // a group whose members all sit below the 500-per-mille unhealthy threshold has a
            // mean below 500 too, so the flag was a dead key that looked like a safeguard.
            // The mean is taken over members rather than calls so a group with more leaves
            // does not get a louder voice for having more leaves.
            let measured: Vec<(u32, u64)> = members
                .iter()
                .filter_map(|candidate| {
                    evidence.get(&candidate.model).filter(|e| e.is_measured()).map(|e| {
                        (
                            ((e.reliability() * 1000.0).round() as u32).min(1000),
                            e.latency_millis(),
                        )
                    })
                })
                .collect();
            let completion = if measured.is_empty() {
                1000
            } else {
                measured.iter().map(|(permille, _)| *permille).sum::<u32>()
                    / measured.len() as u32
            };
            let mean_latency = if measured.is_empty() {
                u64::MAX
            } else {
                measured.iter().map(|(_, latency)| *latency).sum::<u64>()
                    / measured.len() as u64
            };
            // Bounded recovery, computed as a RANKING OVERRIDE for this dispatch only. A ready
            // group whose measured completion is behind a healthier peer is "starved"; once its
            // own pool-scoped opportunity count reaches RECOVERY_WINDOW it is offered exactly one
            // recovery dispatch, ranked as if unmeasured so it can earn fresh evidence. The
            // group's Evidence is never touched (another route may share it), and the counter is
            // only advanced on a real dispatch, so a preview cannot grant or spend recovery.
            let recovering = !ordered
                && ready
                && completion < 1000
                && advance
                && rotation
                    .starved_opportunities
                    .get(&group.name)
                    .copied()
                    .unwrap_or(0)
                    >= RECOVERY_WINDOW;
            let (rank_completion, rank_latency) = if recovering {
                (1000, u64::MAX)
            } else {
                (completion, mean_latency)
            };
            let last_served = if ordered {
                0
            } else {
                rotation.last_served.get(&group.name).copied().unwrap_or(0)
            };
            // An ordered pool keeps the declared index ahead of every evidence key: the
            // groups are a price sequence, and a cheap group that is measurably worse is
            // still the one the deployment chose to spend first. The leaf keys inside the
            // group are what pick a fast member when the group has one.
            let declared_rank = if ordered { index as u32 } else { 0 };
            let readiness = if ordered { false } else { !ready };
            if !ordered && ready && completion < 1000 {
                recovery_eligible.push(group.name.clone());
            }
            order.push((
                declared_rank,
                readiness,
                1000u32.saturating_sub(rank_completion),
                rank_latency,
                last_served,
                group.name.clone(),
            ));
        }
        order.sort();
        if advance && !ordered {
            rotation.tick = rotation.tick.saturating_add(1);
            let tick = rotation.tick;
            if let Some(head) = order.first() {
                rotation.last_served.insert(head.5.clone(), tick);
                // The head group was served, so its starvation episode ends here regardless of
                // whether this dispatch was a recovery: recovery is bounded per episode.
                rotation.starved_opportunities.remove(&head.5);
            }
            // Every other ready-but-behind group spent one more eligible opportunity starved.
            // A group that is no longer eligible (recovered, exhausted, or now healthy) has its
            // counter cleared so a later, separate outage starts a fresh window.
            for name in &recovery_eligible {
                if Some(name) != order.first().map(|head| &head.5) {
                    let count = rotation
                        .starved_opportunities
                        .entry(name.clone())
                        .or_insert(0);
                    *count = count.saturating_add(1);
                }
            }
            // Clear counters only for groups that belong to THIS plan but are no longer eligible
            // (recovered, exhausted, or now healthy), so a later separate outage starts a fresh
            // window. A group from an unrelated pool (e.g. smart-judge while evaluating
            // smart-nt) is not in this plan and its counter must survive untouched: interleaved
            // requests on different pools must not erase each other's accumulated opportunities.
            let plan_groups: std::collections::HashSet<&str> =
                plan.groups.iter().map(|g| g.name.as_str()).collect();
            let eligible: std::collections::HashSet<&str> =
                recovery_eligible.iter().map(String::as_str).collect();
            rotation
                .starved_opportunities
                .retain(|name, _| !plan_groups.contains(name.as_str()) || eligible.contains(name.as_str()));
        }
        let rank_of: HashMap<&str, u32> = order
            .iter()
            .enumerate()
            .map(|(rank, entry)| (entry.5.as_str(), rank as u32))
            .collect();
        plan.groups
            .iter()
            .flat_map(|group| {
                let rank = rank_of.get(group.name.as_str()).copied().unwrap_or(u32::MAX);
                group.members.iter().map(move |model| (model.clone(), rank))
            })
            .collect()
    }

    /// The comparator itself, with an optional group rank as the first leaf key.
    fn explain_with_ranks(
        &self,
        candidates: &[Candidate],
        demand: Demand,
        declared: &[ModelId],
        policy: RankingPolicy,
        group_ranks: Option<&HashMap<ModelId, u32>>,
    ) -> Vec<CandidateScore> {
        let evidence = self.evidence.lock();
        let position = |model: &ModelId| {
            declared
                .iter()
                .position(|candidate| candidate == model)
                .unwrap_or(usize::MAX)
        };
        // Every component is a rank where smaller is better, in this order:
        // tier, exhausted, unhealthy, over-pressure, remaining budget, pressure, latency,
        // and finally the manifest's own order. Tier first is what makes "paid never
        // overtakes eligible free" a property of the shape rather than of the weights.
        let build = |candidate: &Candidate, index: usize| {
            let observed = evidence.get(&candidate.model).copied().unwrap_or_default();
            // An undeclared budget is not capacity. Under `AfterKnown` it sorts behind a
            // verifiable headroom and ahead of a candidate whose own count reached zero, so
            // unknown is neither credited nor treated as an invented refusal.
            let remaining = match (candidate.remaining_today, policy.unknown_headroom) {
                (Some(remaining), _) => remaining,
                (None, UnknownHeadroom::Unbounded) => u64::MAX,
                (None, UnknownHeadroom::AfterKnown) => 0,
            };
            let pressure = candidate.pressure_per_mille(demand);
            let score = CandidateScore {
                model: candidate.model.clone(),
                tier: candidate.tier,
                exhausted: candidate.remaining_today == Some(0),
                empty_answer: observed.has_empty_answer(),
                unhealthy: observed.is_unhealthy(),
                over_pressure: pressure > PRESSURE_CEILING_PER_MILLE,
                remaining_today: candidate.remaining_today,
                pressure_permille: pressure,
                latency_ms: observed.is_measured().then(|| observed.latency_millis()),
                declared_position: index,
                rank: 0,
                demotion: None,
            };
            let key = (
                candidate.tier,
                // A grouped pool chooses its group before any leaf fact: the leaf keys below
                // compare candidates that share an allowance domain, which is the only
                // comparison they can make honestly. `0` for every candidate in a flat pool.
                group_ranks
                    .map(|ranks| {
                        ranks
                            .get(&candidate.model)
                            .copied()
                            // A grouped candidate the plan does not describe is a
                            // configuration bug, and the safe reading is "last", not
                            // "first".
                            .unwrap_or(u32::MAX)
                    })
                    .unwrap_or(0),
                u8::from(score.exhausted),
                // An empty answer outranks a generic unhealthy sample: it is a measured
                // non-answer, so it moves a candidate down on the first observation
                // rather than only after `MIN_OBSERVATIONS` failures.
                u8::from(score.empty_answer),
                u8::from(score.unhealthy),
                u8::from(score.over_pressure),
                u64::MAX - remaining,
                pressure,
                // An unmeasured candidate is not "infinitely fast": it has no latency
                // evidence at all, so it keeps the declared tie-break instead of winning
                // the latency key against a candidate that was actually measured.
                score.latency_ms.unwrap_or(u64::MAX),
                match policy.tie_break {
                    TieBreak::Declared => TieKey::Declared(position(&candidate.model)),
                    TieBreak::CanonicalId => {
                        TieKey::Canonical(candidate.model.as_str().to_string())
                    }
                },
            );
            (key, score)
        };
        let mut ranked: Vec<(_, CandidateScore)> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| build(candidate, index))
            .collect();
        // Not `sort_by_key`: the key carries an owned tie-break, so it cannot be copied out
        // of the element the way `sort_by_key` requires.
        ranked.sort_by(|(left, _), (right, _)| left.cmp(right));
        ranked
            .into_iter()
            .enumerate()
            .map(|(rank, (_, mut score))| {
                score.rank = rank;
                // A candidate that fell below where the manifest declared it gets the
                // first component that explains the fall. `None` means it did not move
                // down, so nothing is claimed about it.
                if rank > score.declared_position {
                    score.demotion = Some(if score.exhausted {
                        "daily_allowance_exhausted"
                    } else if score.empty_answer {
                        "empty_answers"
                    } else if score.unhealthy {
                        "unhealthy"
                    } else if score.over_pressure {
                        "over_pressure"
                    } else if score.remaining_today.is_none()
                        && policy.unknown_headroom == UnknownHeadroom::AfterKnown
                    {
                        // The candidate declared no budget, so it cannot outrank one whose
                        // headroom is verifiable. Reported as the absence it is, rather than
                        // as a small budget the deployment never stated.
                        "unknown_headroom"
                    } else if score.latency_ms.is_some() {
                        "latency"
                    } else {
                        "lower_daily_budget"
                    });
                }
                score
            })
            .collect()
    }

    /// The measured evidence for one model, for tests and diagnostics.
    pub fn evidence_for(&self, model: &ModelId) -> Option<(u32, u32, f64)> {
        self.evidence
            .lock()
            .get(model)
            .map(|value| (value.calls, value.successes, value.latency_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use switchyard_llm_client::LlmCallObservation;

    fn call(model: &str, success: bool, millis: u64) -> RunObservation {
        RunObservation::AnswerCall(LlmCallObservation {
            selected_model: ModelId::from(model),
            is_success: success,
            duration: Duration::from_millis(millis),
            usage: None,
            outcome: None,
            reason: None,
            scope: None,
        })
    }

    fn models(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::from(*name)).collect()
    }

    fn free(model: &str, remaining: u64, ceiling: u64) -> Candidate {
        Candidate {
            model: ModelId::from(model),
            tier: CostTier::Free,
            remaining_today: Some(remaining),
            max_input_tokens: Some(ceiling),
            context_window: Some(262_144),
        }
    }

    fn paid(model: &str) -> Candidate {
        Candidate {
            model: ModelId::from(model),
            tier: CostTier::Paid,
            remaining_today: None,
            max_input_tokens: None,
            context_window: Some(262_144),
        }
    }

    fn small() -> Demand {
        Demand {
            required_tokens: 5_000,
            needs_tools: false,
        }
    }

    #[test]
    fn a_cold_selector_leaves_an_undifferentiated_order_alone() {
        let selector = MeasuredSelector::new();
        let candidates = vec![paid("a"), paid("b")];
        let declared = models(&["a", "b"]);
        assert_eq!(selector.order(&candidates, small(), &declared), None);
    }

    #[test]
    fn a_small_request_prefers_the_model_with_the_least_scarce_daily_budget() {
        // The deployed free pair: a Gemma with 14,400 requests a day and a 16,000-token
        // minute, and a Lite with 500 a day and 250,000 a minute. At 5,000 tokens both fit;
        // the Gemma's daily budget is the cheaper one to spend, so it is offered first and
        // the Lite is reached on a failure instead of on every request.
        let selector = MeasuredSelector::new();
        let candidates = vec![free("lite", 500, 250_000), free("gemma", 14_400, 16_000)];
        let declared = models(&["lite", "gemma"]);
        assert_eq!(
            selector.order(&candidates, small(), &declared),
            Some(models(&["gemma", "lite"]))
        );
    }

    #[test]
    fn a_request_near_the_minute_budget_prefers_the_target_with_headroom() {
        let selector = MeasuredSelector::new();
        let candidates = vec![free("gemma", 14_400, 16_000), free("lite", 500, 250_000)];
        let declared = models(&["gemma", "lite"]);
        // 12,000 against a 16,000-token ceiling is three quarters of the minute: the Gemma
        // is demoted even though its daily budget is larger.
        let demand = Demand {
            required_tokens: 12_000,
            needs_tools: false,
        };
        assert_eq!(
            selector.order(&candidates, demand, &declared),
            Some(models(&["lite", "gemma"]))
        );
    }

    #[test]
    fn a_tool_exchange_is_costed_as_the_conversation_it_becomes() {
        let selector = MeasuredSelector::new();
        let candidates = vec![free("gemma", 14_400, 16_000), free("lite", 500, 250_000)];
        let declared = models(&["gemma", "lite"]);
        // 5,000 tokens is comfortable for the Gemma as a single turn; as a tool exchange it
        // is budgeted at 10,000, which leaves too little of the minute for the result turn.
        let tools = Demand {
            required_tokens: 5_000,
            needs_tools: true,
        };
        assert_eq!(
            selector.order(&candidates, tools, &declared),
            Some(models(&["lite", "gemma"]))
        );
    }

    #[test]
    fn a_paid_candidate_never_overtakes_an_eligible_free_one() {
        let selector = MeasuredSelector::new();
        // The free leg has failed every time; the paid leg has answered every time. The
        // free leg may be demoted within its own tier, never below the paid tier.
        for _ in 0..3 {
            selector.observe(&call("free", false, 900));
            selector.observe(&call("paid", true, 20));
        }
        let candidates = vec![free("free", 100, 250_000), paid("paid")];
        let declared = models(&["free", "paid"]);
        assert_eq!(selector.order(&candidates, small(), &declared), None);
    }

    #[test]
    fn a_measured_unreliable_candidate_is_not_offered_first() {
        let selector = MeasuredSelector::new();
        for _ in 0..3 {
            selector.observe(&call("a", false, 100));
            selector.observe(&call("b", true, 400));
        }
        let candidates = vec![paid("a"), paid("b")];
        let declared = models(&["a", "b"]);
        assert_eq!(
            selector.order(&candidates, small(), &declared),
            Some(models(&["b", "a"]))
        );
    }

    #[test]
    fn an_exhausted_allowance_is_offered_last_within_its_tier() {
        let selector = MeasuredSelector::new();
        // `spent` has more nominal daily budget than `open`, but none of it is left today.
        let candidates = vec![free("spent", 0, 16_000), free("open", 10, 16_000)];
        let declared = models(&["spent", "open"]);
        assert_eq!(
            selector.order(&candidates, small(), &declared),
            Some(models(&["open", "spent"]))
        );
    }

    #[test]
    fn equal_evidence_keeps_the_declared_order() {
        let selector = MeasuredSelector::new();
        for _ in 0..3 {
            selector.observe(&call("a", true, 100));
            selector.observe(&call("b", true, 100));
        }
        let candidates = vec![paid("a"), paid("b")];
        let declared = models(&["a", "b"]);
        assert_eq!(selector.order(&candidates, small(), &declared), None);
    }

    #[test]
    fn latency_breaks_a_tie_the_economy_keys_cannot() {
        let selector = MeasuredSelector::new();
        for _ in 0..4 {
            selector.observe(&call("a", true, 800));
            selector.observe(&call("b", true, 90));
        }
        let candidates = vec![paid("a"), paid("b")];
        let declared = models(&["a", "b"]);
        assert_eq!(
            selector.order(&candidates, small(), &declared),
            Some(models(&["b", "a"]))
        );
    }

    #[test]
    fn the_selector_only_ever_reorders_the_candidates_it_was_given() {
        let selector = MeasuredSelector::new();
        for _ in 0..3 {
            selector.observe(&call("c", true, 10));
        }
        let candidates = vec![paid("a"), paid("b")];
        let declared = models(&["a", "b", "c"]);
        assert_eq!(selector.order(&candidates, small(), &declared), None);
        let ordered = selector
            .order(&[paid("a"), paid("c")], small(), &models(&["a", "b", "c"]))
            .expect("c measured as fast");
        assert_eq!(ordered, models(&["c", "a"]));
        assert!(
            !ordered.contains(&ModelId::from("b")),
            "the selector invented a candidate"
        );
    }

    #[test]
    fn a_latency_sample_does_not_erase_the_history_behind_it() {
        let selector = MeasuredSelector::new();
        for _ in 0..3 {
            selector.observe(&call("a", true, 100));
        }
        selector.observe(&call("a", true, 500));
        let (calls, successes, latency) = selector
            .evidence_for(&ModelId::from("a"))
            .expect("evidence");
        assert_eq!((calls, successes), (4, 4));
        // 100 then one 500 at weight 0.25 moves the average to 200, not to 500.
        assert!((latency - 200.0).abs() < 1.0, "{latency}");
    }

    #[test]
    fn two_observations_are_not_enough_to_demote_a_failing_candidate() {
        let selector = MeasuredSelector::new();
        for _ in 0..2 {
            selector.observe(&call("a", false, 10));
        }
        let candidates = vec![paid("a"), paid("b")];
        let declared = models(&["a", "b"]);
        assert_eq!(selector.order(&candidates, small(), &declared), None);
    }

    /// One observation that ended without a usable answer, tagged with its trace code.
    fn unanswered(model: &str, code: &'static str) -> RunObservation {
        RunObservation::AnswerCall(LlmCallObservation {
            selected_model: ModelId::from(model),
            is_success: false,
            duration: Duration::from_millis(10),
            usage: None,
            outcome: Some(code),
            reason: None,
            scope: None,
        })
    }

    /// A reply whose whole output was hidden reasoning must not count against the leg.
    ///
    /// The pre-dispatch floor is what keeps that request shape away from the target; if the
    /// selector also convicted the target, one sixteen-token probe would remove a leg that
    /// answers the 2,048-token judge calls. One observation isolates the rule: at that
    /// sample size `unhealthy` cannot fire, so the empty-answer rule is the only demoter in
    /// play and the order staying put is the rule's doing.
    #[test]
    fn a_thinking_only_answer_is_not_evidence_against_the_leg() {
        let selector = MeasuredSelector::new();
        selector.observe(&unanswered("a", "empty_answer_thinking"));
        let candidates = vec![paid("a"), paid("b")];
        assert_eq!(
            selector.order(&candidates, small(), &models(&["a", "b"])),
            None,
            "a budget-shaped non-answer must not demote the model that ran out of budget"
        );
    }

    /// The masking partner for the test above: the identical observation, carrying the
    /// ordinary empty-answer code, still demotes immediately. Without this, a selector that
    /// ignored empty answers altogether would also pass.
    #[test]
    fn an_ordinary_empty_answer_still_demotes_immediately() {
        let selector = MeasuredSelector::new();
        selector.observe(&unanswered("a", "empty_answer"));
        let candidates = vec![paid("a"), paid("b")];
        assert_eq!(
            selector.order(&candidates, small(), &models(&["a", "b"])),
            Some(models(&["b", "a"])),
            "a leg that answered with nothing is still the wrong first choice"
        );
    }

    /// A candidate whose provider states no per-request budget at all. The OpenAI
    /// subscription is this shape: the deployment cannot count its allowance, which must not
    /// be read as either unlimited or spent.
    fn free_unknown(model: &str) -> Candidate {
        Candidate {
            model: ModelId::from(model),
            tier: CostTier::Free,
            remaining_today: None,
            max_input_tokens: None,
            context_window: Some(1_050_000),
        }
    }

    fn plan(groups: &[(&str, &[&str])]) -> GroupPlan {
        GroupPlan {
            groups: groups
                .iter()
                .map(|(name, members)| GroupMembers {
                    name: (*name).to_string(),
                    members: models(members),
                })
                .collect(),
        }
    }

    /// The deployed smart-nt primary pool: four Google leaves and one OpenAI leaf.
    fn four_google_one_openai() -> (Vec<Candidate>, GroupPlan) {
        let candidates = vec![
            free("g1", 500, 250_000),
            free("g2", 500, 250_000),
            free("g3", 500, 250_000),
            free("g4", 500, 250_000),
            free_unknown("o1"),
        ];
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1", "g2", "g3", "g4"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        (candidates, plan)
    }

    fn group_of(head: &str) -> &'static str {
        if head.starts_with('g') {
            "google"
        } else {
            "openai"
        }
    }

    #[test]
    fn a_grouped_pool_alternates_groups_while_no_group_is_measured() {
        // The fairness floor. With no evidence the two groups are equal, and a pool of four
        // Google leaves and one OpenAI leaf must not serve Google four times in five: the
        // group is chosen first, so leaf count cannot become provider priority.
        let selector = MeasuredSelector::new();
        let (candidates, plan) = four_google_one_openai();
        let declared = models(&["g1", "g2", "g3", "g4", "o1"]);
        let heads: Vec<&'static str> = (0..6)
            .map(|_| {
                let scores = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
                group_of(scores[0].model.as_str())
            })
            .collect();
        assert_eq!(
            heads,
            vec!["google", "openai", "google", "openai", "google", "openai"]
        );
    }

    #[test]
    fn a_grouped_pool_prefers_the_group_that_has_actually_answered_faster() {
        let selector = MeasuredSelector::new();
        let (candidates, plan) = four_google_one_openai();
        let declared = models(&["g1", "g2", "g3", "g4", "o1"]);
        for _ in 0..3 {
            selector.observe(&call("g1", true, 4_000));
            selector.observe(&call("o1", true, 400));
        }
        let scores = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(
            scores[0].model.as_str(),
            "o1",
            "measurement overrides the rotation floor: the faster group is preferred"
        );
    }

    #[test]
    fn a_group_whose_allowance_is_spent_is_not_offered_while_another_is_ready() {
        // Google is provably out of allowance; OpenAI's allowance is undeclared. Flat peer
        // ranking would keep Google first (its budget is verifiable) until exhaustion, then
        // treat the undeclared group as the last resort. Grouping decides on readiness, so a
        // spent group steps aside for a ready one without inventing an exhaustion claim
        // about the undeclared group.
        let selector = MeasuredSelector::new();
        let candidates = vec![
            free("g1", 0, 250_000),
            free("g2", 0, 250_000),
            free("g3", 0, 250_000),
            free("g4", 0, 250_000),
            free_unknown("o1"),
        ];
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1", "g2", "g3", "g4"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "g2", "g3", "g4", "o1"]);
        let scores = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(scores[0].model.as_str(), "o1");
    }

    #[test]
    fn within_the_chosen_group_the_leaf_economy_still_applies() {
        // The group keys must not flatten the leaf comparison: inside Google, a leaf with
        // almost no daily budget left is still the last of its own group.
        let selector = MeasuredSelector::new();
        let candidates = vec![
            free("g1", 10, 250_000),
            free("g2", 500, 250_000),
            free("g3", 500, 250_000),
            free("o1", 0, 250_000),
        ];
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1", "g2", "g3"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "g2", "g3", "o1"]);
        let scores = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        let position = |model: &str| {
            scores
                .iter()
                .position(|score| score.model.as_str() == model)
                .expect("candidate is ranked")
        };
        assert!(position("g1") > position("g2"));
        assert!(position("g1") > position("g3"));
    }

    #[test]
    fn a_group_becomes_eligible_again_the_moment_its_allowance_recovers() {
        // Recovery is read from the same counter on every request, so no reset or restart is
        // needed, and a spent group does not stay spent once the count moves.
        let selector = MeasuredSelector::new();
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "o1"]);
        for _ in 0..3 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", true, 5_000));
        }
        let fast = vec![free("g1", 5, 250_000), free_unknown("o1")];
        let spent = vec![free("g1", 0, 250_000), free_unknown("o1")];
        assert_eq!(
            selector.explain_grouped(&fast, small(), &declared, &plan, Grouping::Adaptive)[0]
                .model
                .as_str(),
            "g1"
        );
        assert_eq!(
            selector.explain_grouped(&spent, small(), &declared, &plan, Grouping::Adaptive)[0]
                .model
                .as_str(),
            "o1"
        );
        assert_eq!(
            selector.explain_grouped(&fast, small(), &declared, &plan, Grouping::Adaptive)[0]
                .model
                .as_str(),
            "g1",
            "the recovered group is preferred again without any reset"
        );
    }

    #[test]
    fn a_group_measured_failing_is_demoted_below_a_group_with_no_sample() {
        let selector = MeasuredSelector::new();
        let (candidates, plan) = four_google_one_openai();
        let declared = models(&["g1", "g2", "g3", "g4", "o1"]);
        for leaf in ["g1", "g2", "g3", "g4"] {
            for _ in 0..3 {
                selector.observe(&call(leaf, false, 50));
            }
        }
        let scores = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(
            scores[0].model.as_str(),
            "o1",
            "a group with no sample keeps its place ahead of one measured failing"
        );
    }

    // STARVATION + RECOVERY: a group that failed during a genuine outage (e.g. allowance
    // exhaustion) accumulates negative Evidence with no observation-age field. When the outage
    // clears and the group's allowance recovers, the old failures persist and `completion < 1000`
    // would keep the group ranked behind a healthy peer forever, never dispatched, never earning
    // new observations. The fix offers exactly ONE bounded recovery dispatch after
    // RECOVERY_WINDOW eligible pool opportunities, WITHOUT resetting the group's Evidence (other
    // routes share it), and only on a real dispatch (previews never grant or spend it).
    #[test]
    fn a_recovered_group_with_old_failures_gets_one_bounded_recovery_dispatch() {
        let selector = MeasuredSelector::new();
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "o1"]);
        // During the outage: Luna (o1) fails repeatedly; Google (g1) serves successfully.
        for _ in 0..5 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", false, 50));
        }
        let recovered = vec![free("g1", 5, 250_000), free_unknown("o1")];
        // Before the recovery window elapses, healthy Google keeps its place.
        let early = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(early, "g1", "before the window, the healthy peer keeps its place");
        // Drive real dispatches (each advances the pool-scoped opportunity count) up to the
        // window. The group is still behind on every one but the last.
        for _ in 0..(RECOVERY_WINDOW - 1) {
            let head = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
                [0]
                .model
                .as_str()
                .to_string();
            assert_eq!(head, "g1", "still starved before the window elapses");
        }
        // The RECOVERY_WINDOW-th eligible opportunity grants exactly one recovery dispatch.
        let head = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(
            head, "o1",
            "RECOVERY: starved group gets exactly one bounded recovery dispatch after the window"
        );
        // The recovery did NOT reset Evidence: o1's old failures are still on the books, so the
        // very next dispatch is judged on that evidence and the healthy peer retakes the head.
        // Recovery is one dispatch, not a fresh start and not a permanent preference.
        let after = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(
            after, "g1",
            "recovery is one dispatch only; old failures still demote the group next time"
        );
        // A second recovery cannot trigger until the group is starved for a fresh full window.
        for _ in 0..(RECOVERY_WINDOW - 1) {
            let head = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
                [0]
                .model
                .as_str()
                .to_string();
            assert_eq!(head, "g1", "no second recovery before a fresh window");
        }
        let second = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(second, "o1", "a fresh full window grants one more bounded recovery dispatch");
    }

    // The recovery dispatch is an ORDINARY request the route was already going to make, and its
    // real observation must update the real Evidence: a recovery that succeeds restores the group
    // from measured data, not from the override. This is the smart-nt shape (Luna NT recovers).
    #[test]
    fn a_recovery_dispatch_that_succeeds_restores_the_group_from_real_evidence() {
        let selector = MeasuredSelector::new();
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "o1"]);
        for _ in 0..5 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", false, 50));
        }
        let recovered = vec![free("g1", 5, 250_000), free_unknown("o1")];
        for _ in 0..RECOVERY_WINDOW {
            selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive);
        }
        // The window's last dispatch chose o1; the outage is over, so o1 now succeeds. Record the
        // real outcome of that recovery dispatch plus enough successes to clear MIN_OBSERVATIONS.
        selector.observe(&call("o1", true, 80));
        selector.observe(&call("o1", true, 80));
        selector.observe(&call("o1", true, 80));
        // o1's measured completion is now healthy, so it keeps the head on its own evidence.
        let head = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(head, "o1", "a recovered group that now measures healthy keeps the head");
    }

    // A DIAGNOSTIC PREVIEW must never grant or spend recovery: asking what the order would be
    // leaves the starvation count untouched, so a preview cannot manufacture a recovery dispatch.
    #[test]
    fn a_preview_does_not_advance_the_recovery_window() {
        let selector = MeasuredSelector::new();
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "o1"]);
        for _ in 0..5 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", false, 50));
        }
        let recovered = vec![free("g1", 5, 250_000), free_unknown("o1")];
        // Many previews, more than the window, must not grant a recovery.
        for _ in 0..(RECOVERY_WINDOW * 3) {
            let head = selector
                .explain_grouped_preview(&recovered, small(), &declared, &plan, Grouping::Adaptive)[0]
                .model
                .as_str()
                .to_string();
            assert_eq!(head, "g1", "a preview never grants the starved group a recovery");
        }
        // The first REAL dispatch after all those previews is still the first eligible
        // opportunity: the preview did not spend the window. Recovery fires on the
        // (RECOVERY_WINDOW + 1)-th real dispatch from a fresh counter (RECOVERY_WINDOW
        // completed starved opportunities, then one recovery), exactly as if no preview ran.
        let mut head = String::new();
        for _ in 0..=RECOVERY_WINDOW {
            head = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)[0]
                .model
                .as_str()
                .to_string();
        }
        assert_eq!(head, "o1", "previews did not advance the window; real dispatches do");
    }

    // Recovery counters are keyed by GROUP and must survive interleaved requests on an
    // unrelated pool. This uses the actual smart-nt and smart-judge shapes, which SHARE the
    // canonical Luna target (gpt-6-luna): smart-nt = {smart-gemini-bounded, smart-openai-nt},
    // smart-judge = {smart-judge-nt, smart-judge-local}. Interleaving judge dispatches between
    // nt dispatches must not erase the nt group's accumulated starvation opportunities: the
    // starved nt group still reaches RECOVERY_WINDOW and gets exactly one recovery dispatch.
    #[test]
    fn interleaved_unrelated_pool_requests_do_not_erase_recovery_counters() {
        let selector = MeasuredSelector::new();
        // Actual shapes. The Luna leaf o1 is shared by both pools' openai/nt group.
        let nt_plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let nt_declared = models(&["g1", "o1"]);
        let judge_plan = plan(&[
            ("smart-judge-nt", &["o1"]),
            ("smart-judge-local", &["j2"]),
        ]);
        let judge_declared = models(&["o1", "j2"]);
        // Evidence: g1 healthy, o1 failing (during its outage), j2 healthy. o1 is shared, so its
        // failing evidence is real for both pools.
        for _ in 0..5 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", false, 50));
            selector.observe(&call("j2", true, 110));
        }
        let nt_candidates = vec![free("g1", 5, 250_000), free_unknown("o1")];
        let judge_candidates = vec![free_unknown("o1"), free("j2", 5, 250_000)];
        // Interleave: one nt dispatch, then several judge dispatches, repeated. The judge pool
        // is healthy (j2 serves), so its own groups never become recovery-eligible; but the
        // judge dispatches must NOT clear the nt group's counter. Recovery fires on the
        // (RECOVERY_WINDOW + 1)-th nt dispatch from a fresh counter.
        for _ in 0..RECOVERY_WINDOW {
            let nt_head = selector.explain_grouped(&nt_candidates, small(), &nt_declared, &nt_plan, Grouping::Adaptive)
                [0]
                .model
                .as_str()
                .to_string();
            assert_eq!(nt_head, "g1", "nt still starved before its window");
            // Interleave unrelated judge traffic (3 judge dispatches per nt dispatch).
            for _ in 0..3 {
                selector.explain_grouped(&judge_candidates, small(), &judge_declared, &judge_plan, Grouping::Adaptive);
            }
        }
        // The nt group's counter survived all the interleaved judge traffic and now reaches the
        // threshold: the next nt dispatch is the bounded recovery dispatch for o1.
        let nt_head = selector.explain_grouped(&nt_candidates, small(), &nt_declared, &nt_plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(
            nt_head, "o1",
            "interleaved judge requests did not erase the nt group's recovery opportunities"
        );
        // One bounded recovery only: the next nt dispatch is judged on o1's (still failing)
        // real evidence, so g1 retakes the head. Historical evidence was never reset.
        let nt_after = selector.explain_grouped(&nt_candidates, small(), &nt_declared, &nt_plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(nt_after, "g1", "one bounded recovery; old failures still demote o1 next time");
    }

    // The success-recovery witness must observe the ACTUAL recovery dispatch and distinguish a
    // genuine evidence-based ranking from the temporary override. A single success does not
    // erase all previous failures: the group keeps the head only once its real measured
    // completion (with the prior) clears the healthy peer, not because the override is active.
    #[test]
    fn a_success_recovery_witness_distinguishes_real_evidence_from_the_override() {
        let selector = MeasuredSelector::new();
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "o1"]);
        // Outage: o1 fails 5 times, g1 succeeds 5 times.
        for _ in 0..5 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", false, 50));
        }
        let recovered = vec![free("g1", 5, 250_000), free_unknown("o1")];
        // Reach the recovery dispatch.
        for _ in 0..=RECOVERY_WINDOW {
            selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive);
        }
        // The recovery dispatch chose o1. Now observe its ACTUAL outcome: one success. With the
        // Beta prior, one success is NOT enough to clear the 5 prior failures -- o1's measured
        // completion is still below g1's, so the override is gone and g1 retakes the head. This
        // proves the ranking is evidence-based, not override-based, and that one success does
        // not erase all previous failures.
        selector.observe(&call("o1", true, 80));
        let after_one = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(
            after_one, "g1",
            "one success does not erase five failures; the override is gone and evidence rules"
        );
        // The group keeps the head on GENUINE evidence only once its real measured completion
        // (successes+1)/(calls+2) clears g1's 6/7. With five failures fixed, that needs the
        // successes to dominate: 5 failures + 50 successes -> 51/57 = 0.895 > 0.857. Until then
        // the healthy peer stays ahead, which is the honest, evidence-based behaviour -- a single
        // success never flips it and the override is never re-granted without a fresh window.
        for _ in 0..49 {
            selector.observe(&call("o1", true, 80));
        }
        let after_many = selector.explain_grouped(&recovered, small(), &declared, &plan, Grouping::Adaptive)
            [0]
            .model
            .as_str()
            .to_string();
        assert_eq!(
            after_many, "o1",
            "only genuine measured evidence (not the override) restores the group from a real outage"
        );
    }

    // A group that is NOT ready (its own allowance is exhausted) is never offered recovery:
    // quota exhaustion is not "recovered readiness". Neither is an over-pressure or fit-excluded
    // request. Only a ready, eligible group can recover.
    #[test]
    fn an_exhausted_group_is_never_offered_recovery() {
        let selector = MeasuredSelector::new();
        let plan = plan(&[
            ("smart-gemini-bounded", &["g1"]),
            ("smart-openai-nt", &["o1"]),
        ]);
        let declared = models(&["g1", "o1"]);
        for _ in 0..5 {
            selector.observe(&call("g1", true, 100));
            selector.observe(&call("o1", false, 50));
        }
        // o1's allowance is exhausted (remaining_today == Some(0)): not ready, so never recovers.
        let spent = vec![free("g1", 5, 250_000), free("o1", 0, 250_000)];
        for _ in 0..(RECOVERY_WINDOW * 2) {
            let head = selector.explain_grouped(&spent, small(), &declared, &plan, Grouping::Adaptive)[0]
                .model
                .as_str()
                .to_string();
            assert_eq!(head, "g1", "an exhausted group is never offered recovery");
        }
    }
    #[test]
    fn asking_what_the_grouped_order_would_be_does_not_spend_a_turn() {
        let selector = MeasuredSelector::new();
        let (candidates, plan) = four_google_one_openai();
        let declared = models(&["g1", "g2", "g3", "g4", "o1"]);
        let once = selector.explain_grouped_preview(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        let twice = selector.explain_grouped_preview(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(once[0].model, twice[0].model);
        // And the real dispatch path still rotates afterwards.
        let dispatched = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        let next = selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(dispatched[0].model, once[0].model);
        assert_ne!(
            next[0].model, dispatched[0].model,
            "the first real dispatch consumed the rotation step"
        );
    }

    #[test]
    fn an_ordered_pool_exhausts_the_preferred_group_before_the_next() {
        // The cost shape: the groups are a price sequence, not peers. The second group is
        // measurably faster and healthier, and it must still wait, because the deployment
        // chose to spend the cheap group first. Inside the cheap group the leaf keys do
        // decide, so the weaker cheap leaf is not forced to the front.
        let selector = MeasuredSelector::new();
        let candidates = vec![
            free("c1", 500, 250_000),
            free("c2", 500, 250_000),
            free("x1", 500, 250_000),
        ];
        let plan = plan(&[("cheap", &["c1", "c2"]), ("expensive", &["x1"])]);
        let declared = models(&["c1", "c2", "x1"]);
        for _ in 0..4 {
            selector.observe(&call("c1", false, 9_000));
            selector.observe(&call("c2", true, 8_000));
            selector.observe(&call("x1", true, 100));
        }
        let scores =
            selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Ordered);
        let position = |model: &str| {
            scores
                .iter()
                .position(|score| score.model.as_str() == model)
                .expect("candidate is ranked")
        };
        assert!(position("c1") < position("x1"));
        assert!(position("c2") < position("x1"));
        assert!(
            position("c2") < position("c1"),
            "the leaf keys still choose the better member inside the group"
        );
    }

    #[test]
    fn ordered_groups_do_not_rotate_between_equals() {
        // Rotation exists to share traffic between interchangeable groups. A price sequence
        // has nothing to share: repeated dispatches must keep landing on the cheap group.
        let selector = MeasuredSelector::new();
        let candidates = vec![free("c1", 500, 250_000), free("x1", 500, 250_000)];
        let plan = plan(&[("cheap", &["c1"]), ("expensive", &["x1"])]);
        let declared = models(&["c1", "x1"]);
        let heads: Vec<String> = (0..4)
            .map(|_| {
                selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Ordered)
                    [0]
                .model
                .as_str()
                .to_string()
            })
            .collect();
        assert_eq!(heads, vec!["c1", "c1", "c1", "c1"]);
    }

    #[test]
    fn an_ordered_dispatch_does_not_spend_a_rotation_step() {
        // The rotation map is process-wide and keyed by group label, so two routes that name
        // the same groups share it. An ordered route picks the same head every time and has
        // nothing to share: recording it as served would bias an adaptive route over those
        // labels away from a group that was never given a turn it could have used.
        let selector = MeasuredSelector::new();
        let candidates = vec![free("a1", 500, 250_000), free("b1", 500, 250_000)];
        let declared = models(&["a1", "b1"]);
        let plan = plan(&[("a", &["a1"]), ("b", &["b1"])]);
        selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Ordered);
        let preview =
            selector.explain_grouped_preview(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(
            preview[0].model.as_str(),
            "a1",
            "the ordered dispatch left the adaptive rotation untouched"
        );
    }

    #[test]
    fn a_group_with_a_worse_completion_rate_yields_to_a_healthier_one() {
        // Neither group is wholly failing, so the all-unhealthy key cannot separate them.
        // The completion rate has to, or a group that answers two calls in three would keep
        // winning on a rotation floor while a peer answers every call.
        let selector = MeasuredSelector::new();
        let candidates = vec![
            free("g1", 500, 250_000),
            free("g2", 500, 250_000),
            free("o1", 500, 250_000),
        ];
        let plan = plan(&[
            ("google", &["g1", "g2"]),
            ("openai", &["o1"]),
        ]);
        let declared = models(&["g1", "g2", "o1"]);
        for _ in 0..3 {
            selector.observe(&call("g1", false, 100));
            selector.observe(&call("g2", true, 100));
            selector.observe(&call("o1", true, 100));
        }
        let scores =
            selector.explain_grouped(&candidates, small(), &declared, &plan, Grouping::Adaptive);
        assert_eq!(
            scores[0].model.as_str(),
            "o1",
            "the group answering every call outranks the one answering half of them"
        );
        assert!(
            !scores
                .iter()
                .find(|score| score.model.as_str() == "g2")
                .expect("g2 is ranked")
                .unhealthy,
            "the healthy member is not what demoted its group"
        );
    }
}
