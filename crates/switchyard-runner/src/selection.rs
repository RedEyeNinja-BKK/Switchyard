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
    fn ceiling(&self) -> Option<u64> {
        self.max_input_tokens.or(self.context_window)
    }

    /// The share of the candidate's ceiling one conversation would claim, in thousandths.
    ///
    /// A target that states no ceiling claims no pressure: the deployment has not said what
    /// would be tight, so this module cannot invent a limit for it.
    fn pressure_per_mille(&self, demand: Demand) -> u64 {
        let Some(ceiling) = self.ceiling().filter(|ceiling| *ceiling > 0) else {
            return 0;
        };
        let required = demand.required_tokens();
        required
            .saturating_mul(1000)
            .saturating_div(ceiling)
            .min(u64::MAX / 1000)
    }
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
    fn required_tokens(self) -> u64 {
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
}

impl Evidence {
    fn record(&mut self, success: bool, latency_ms: f64) {
        self.calls = self.calls.saturating_add(1);
        if success {
            self.successes = self.successes.saturating_add(1);
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
}

/// Process-wide record of how each physical target has actually behaved.
#[derive(Default)]
pub struct MeasuredSelector {
    evidence: Mutex<HashMap<ModelId, Evidence>>,
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
            | RunObservation::ContextPressurePromotion { .. } => return,
        };
        let mut evidence = self.evidence.lock();
        evidence
            .entry(call.selected_model.clone())
            .or_default()
            .record(call.is_success, call.duration.as_secs_f64() * 1000.0);
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
        let key = |candidate: &Candidate| {
            let observed = evidence.get(&candidate.model).copied().unwrap_or_default();
            let remaining = candidate.remaining_today.unwrap_or(u64::MAX);
            let pressure = candidate.pressure_per_mille(demand);
            (
                candidate.tier,
                u8::from(candidate.remaining_today == Some(0)),
                u8::from(observed.is_unhealthy()),
                u8::from(pressure > PRESSURE_CEILING_PER_MILLE),
                u64::MAX - remaining,
                pressure,
                // An unmeasured candidate is not "infinitely fast": it has no latency
                // evidence at all, so it keeps the declared tie-break instead of winning
                // the latency key against a candidate that was actually measured.
                if observed.is_measured() {
                    observed.latency_millis()
                } else {
                    u64::MAX
                },
                position(&candidate.model),
            )
        };
        let mut ordered: Vec<Candidate> = candidates.to_vec();
        ordered.sort_by_key(key);
        let ordered: Vec<ModelId> = ordered
            .into_iter()
            .map(|candidate| candidate.model)
            .collect();
        let unchanged = ordered
            .iter()
            .zip(candidates.iter())
            .all(|(left, right)| left == &right.model);
        (!unchanged).then_some(ordered)
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
}
