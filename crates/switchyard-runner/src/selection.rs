// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measured ordering among candidates a route has already admitted.
//!
//! This is not a router and deliberately cannot become one. Every hard gate — privacy
//! clearance, media capability, physical context fit, daily quota, credential scope — runs
//! before a candidate reaches this module, so the only decision left is the order in which
//! interchangeable models are tried. For a route built from free-tier models of the same
//! family, declared order is an accident of how the manifest was typed; what an operator
//! actually wants is to try the leg that has been completing and answering fastest.
//!
//! Three properties keep it from doing harm:
//!
//! * **No data, no change.** A candidate is only moved once this deployment has observed
//!   enough calls to say something about it. A cold process, a restart, or a newly added
//!   target keeps the declared order, so the first requests after a deploy behave exactly
//!   as the manifest reads.
//! * **Declared order is the tie-break.** Equal evidence keeps the manifest's order, so
//!   the module can only ever swap candidates it has measured differently.
//! * **It only reorders.** It never adds, drops or invents a candidate, and it never
//!   overrides a failure class the client's own retry and cooldown logic already decided.
//!
//! The evidence is per process and intentionally not persisted: a latency budget measured
//! before a provider incident is not evidence about the provider now.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_llm_client::RunObservation;
use switchyard_protocol::ModelId;

/// Observations before a candidate may be moved.
///
/// Below this the sample says more about one request than about the model, and reordering
/// on it would make routing order noisy exactly when a deployment is newest.
const MIN_OBSERVATIONS: u32 = 3;

/// Weight of the newest sample in the latency average.
///
/// A slow minute should move the ranking without erasing everything measured before it,
/// which is what a plain last-value comparison would do.
const LATENCY_WEIGHT: f64 = 0.25;

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
}

/// Per-route record of how each target has actually behaved.
#[derive(Default)]
pub struct MeasuredSelector {
    evidence: Mutex<HashMap<ModelId, Evidence>>,
}

impl MeasuredSelector {
    /// A selector with no observations, which therefore changes nothing.
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

    /// Orders `candidates` best-first by measured behaviour, keeping `declared` order for
    /// everything the sample cannot separate.
    ///
    /// Returns `None` when there is nothing to say — fewer than two candidates, or no
    /// candidate yet carrying enough observations — so the caller can leave the list
    /// untouched rather than rewiring it into an identical order.
    pub fn order(&self, candidates: &[ModelId], declared: &[ModelId]) -> Option<Vec<ModelId>> {
        if candidates.len() < 2 {
            return None;
        }
        let evidence = self.evidence.lock();
        let measurable = candidates
            .iter()
            .filter(|model| {
                evidence
                    .get(*model)
                    .is_some_and(|e| e.calls >= MIN_OBSERVATIONS)
            })
            .count();
        if measurable < 2 {
            return None;
        }
        let position = |model: &ModelId| {
            declared
                .iter()
                .position(|candidate| candidate == model)
                .unwrap_or(usize::MAX)
        };
        let mut ordered: Vec<ModelId> = candidates.to_vec();
        ordered.sort_by(|left, right| {
            let rank = |model: &ModelId| evidence.get(model).copied().unwrap_or_default();
            let (a, b) = (rank(left), rank(right));
            // A candidate with too little evidence is ordered as if it were perfect but
            // slow: it is not demoted on no evidence, and the declared tie-break keeps it
            // where the manifest put it.
            let score = |value: Evidence| {
                if value.calls < MIN_OBSERVATIONS {
                    (1.0_f64, f64::MAX)
                } else {
                    (value.reliability(), value.latency_ms)
                }
            };
            let ((a_reliability, a_latency), (b_reliability, b_latency)) = (score(a), score(b));
            b_reliability
                .partial_cmp(&a_reliability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a_latency
                        .partial_cmp(&b_latency)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| position(left).cmp(&position(right)))
        });
        let unchanged = ordered
            .iter()
            .zip(candidates.iter())
            .all(|(left, right)| left == right);
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

    #[test]
    fn a_cold_selector_leaves_the_declared_order_alone() {
        let selector = MeasuredSelector::new();
        let declared = models(&["a", "b"]);
        assert_eq!(selector.order(&declared, &declared), None);
    }

    #[test]
    fn one_success_is_not_enough_evidence_to_reorder() {
        let selector = MeasuredSelector::new();
        let declared = models(&["a", "b"]);
        selector.observe(&call("b", true, 10));
        assert_eq!(selector.order(&declared, &declared), None);
    }

    #[test]
    fn a_reliably_faster_candidate_moves_ahead() {
        let selector = MeasuredSelector::new();
        let declared = models(&["a", "b"]);
        for _ in 0..3 {
            selector.observe(&call("a", false, 900));
            selector.observe(&call("b", true, 120));
        }
        assert_eq!(
            selector.order(&declared, &declared),
            Some(models(&["b", "a"]))
        );
    }

    #[test]
    fn equal_evidence_keeps_the_declared_order() {
        let selector = MeasuredSelector::new();
        let declared = models(&["a", "b"]);
        for _ in 0..3 {
            selector.observe(&call("a", true, 100));
            selector.observe(&call("b", true, 100));
        }
        assert_eq!(selector.order(&declared, &declared), None);
    }

    #[test]
    fn latency_breaks_a_tie_in_reliability() {
        let selector = MeasuredSelector::new();
        let declared = models(&["a", "b"]);
        for _ in 0..4 {
            selector.observe(&call("a", true, 800));
            selector.observe(&call("b", true, 90));
        }
        assert_eq!(
            selector.order(&declared, &declared),
            Some(models(&["b", "a"]))
        );
    }

    #[test]
    fn the_selector_only_ever_reorders_the_candidates_it_was_given() {
        let selector = MeasuredSelector::new();
        let declared = models(&["a", "b", "c"]);
        // Both offered candidates carry evidence, so a ranking exists; `c` is measured too
        // but is not offered, and it must not appear in the result.
        for _ in 0..3 {
            selector.observe(&call("a", false, 50));
            selector.observe(&call("b", true, 50));
            selector.observe(&call("c", true, 10));
        }
        let ordered = selector
            .order(&models(&["a", "b"]), &declared)
            .expect("reorder");
        assert_eq!(ordered, models(&["b", "a"]));
        assert!(
            !ordered.contains(&ModelId::from("c")),
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
}
