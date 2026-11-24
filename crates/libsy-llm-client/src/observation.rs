// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped observations emitted while serving an algorithm run.

use std::sync::Arc;
use std::time::Duration;

use switchyard_libsy::OutcomeMetadata;
use switchyard_protocol::{ModelId, Usage};

/// One completed model call observed while serving an algorithm run.
#[derive(Clone, Debug)]
pub struct LlmCallObservation {
    /// Model selected for the completed call.
    pub selected_model: ModelId,
    /// Whether the call completed successfully.
    pub is_success: bool,
    /// Time spent waiting for the model call to resolve.
    pub duration: Duration,
    /// Normalized usage for a buffered successful response.
    pub usage: Option<Usage>,
    /// Stable code naming how the call ended, or `None` when it succeeded.
    ///
    /// `is_success` alone cannot distinguish an empty answer from a timeout, and an
    /// operator diagnosing "why did this route use the wrong model" needs the
    /// difference. The vocabulary is deliberately closed and value-free: it names a
    /// classification, never a payload.
    pub outcome: Option<&'static str>,
}

/// One candidate's verdict in a route's selection decision.
///
/// Recorded for every declared candidate, including the ones excluded before any
/// provider was contacted, so "this model was never called" has a stated reason
/// rather than being an absence.
#[derive(Clone, Debug)]
pub struct CandidateTrace {
    /// The physical model this verdict is about.
    pub model: String,
    /// Whether it survived every hard gate.
    pub eligible: bool,
    /// Stable code naming why it did not survive, when it did not.
    pub exclusion: Option<&'static str>,
    /// `"free"` or `"paid"` - the tier that decides precedence.
    pub tier: &'static str,
    /// Position in the order actually dispatched, when it was dispatched.
    pub rank: Option<u32>,
    /// Requests this candidate's own daily budget still permits, when declared.
    pub remaining_today: Option<u64>,
    /// Thousandths of the candidate's per-request ceiling this conversation claims.
    pub pressure_permille: u64,
    /// Stable code naming why the selector moved it down, when it did.
    pub demotion: Option<&'static str>,
}

/// One upstream attempt made while serving a request.
#[derive(Clone, Debug)]
pub struct AttemptTrace {
    /// The model that was called.
    pub model: String,
    /// `"routing"` for a call the algorithm made, `"answer"` for the terminal call.
    pub phase: &'static str,
    /// Stable outcome code, e.g. `served`, `empty_answer`, `attempt_timeout`.
    pub outcome: &'static str,
    /// Wall time this attempt consumed.
    pub duration_ms: u64,
    /// Prompt tokens the provider reported, when it reported any.
    pub prompt_tokens: u64,
    /// Completion tokens the provider reported, when it reported any.
    pub completion_tokens: u64,
}

/// The selection decision and dispatch record for one request.
///
/// This makes five states distinguishable that the routing log alone conflates:
/// * **considered** - every declared candidate, in declared order (`considered`);
/// * **eligible** - the ones that survived every hard gate (`candidates[].eligible`);
/// * **selected** - the order actually handed to dispatch (`dispatched`, plus `rank`);
/// * **attempted** - the ones a provider was actually called for (`attempts`);
/// * **served** - the one that produced the answer (`served`).
///
/// A candidate can be considered and eligible yet never attempted, and a candidate can
/// be attempted and not serve. The record states which, rather than leaving the
/// difference to be inferred from timestamps.
#[derive(Clone, Debug, Default)]
pub struct SelectionTrace {
    /// Algorithm that made the decision (`"passthrough"`, `"decision_lane"`, ...).
    ///
    /// The addressed route id is named by whoever looked the route up, not by the route:
    /// it is not a property this type can know without inventing one.
    pub algorithm: String,
    /// Every declared candidate, in declared order.
    pub considered: Vec<String>,
    /// One verdict per declared candidate, in the same order.
    pub candidates: Vec<CandidateTrace>,
    /// The order handed to dispatch after gating, measurement and promotion.
    pub dispatched: Vec<String>,
    /// Every upstream call made while serving this request, in order.
    pub attempts: Vec<AttemptTrace>,
    /// The model that produced the answer, when one did.
    pub served: Option<String>,
    /// `"served"`, `"no_eligible_candidate"`, or `"failed"`.
    pub outcome: &'static str,
    /// Tokens the conversation was costed at, including any output reservation.
    pub demand_tokens: u64,
    /// Estimated input tokens the gates judged the payload by.
    pub estimated_input_tokens: u64,
    /// Leg the context-pressure rule promoted, when it promoted one.
    pub promotion: Option<String>,
    /// The target groups this request's candidate pool was built from, when the route
    /// declares a pool.
    ///
    /// Present only for a provider-neutral pool. It is the labels joined for the record, not
    /// a preference order: the groups name membership, and the ranking inside them is
    /// decided per request. Absent means the route is not a pool, which is every route that
    /// states a deliberate lineup.
    pub pool: Option<String>,
    /// Targets the route declares as a terminal leg, in declared order.
    ///
    /// A terminal leg is never ranked with the primary candidates: it is reached only after
    /// every eligible primary is verifiably unavailable. Recording it here keeps the
    /// distinction visible in the trace, so an answer served by the terminal model cannot be
    /// mistaken for a peer that merely ranked last.
    pub terminal: Vec<String>,
    /// The same declared targets, grouped into the ordered tiers that were declared.
    ///
    /// The flat `terminal` field is the union and keeps every existing reader working. This
    /// one exists because the union cannot express the thing that distinguishes a tiered
    /// declaration from a single set: which legs are peers of each other and which are only
    /// reachable after the previous set was proved out. A reader reconstructing escalation
    /// from the union would see one pool and infer that the cheaper member was skipped
    /// first, when in fact it was consulted second.
    pub terminal_tiers: Vec<Vec<String>>,
    /// Whether this request stepped onto the terminal leg.
    pub terminal_crossed: bool,
    /// Stable code naming why the terminal leg was authorized, when it was.
    ///
    /// The crossing carries its reason rather than a bare boolean because "the pool is
    /// exhausted" and "every capable leg is blocked" are different claims about the primary
    /// pool, and a reader deciding whether a paid leg was spent needs to see which was made.
    pub terminal_reason: Option<&'static str>,
    /// Which terminal tier answered, one-based, when the leg was crossed.
    ///
    /// Declared order is the escalation order, so a record that says tier 2 is also a record
    /// that tier 1 was proved out first. It is the index into `terminal_tiers` (one-based, to
    /// match how an operator reads a manifest) rather than a position in the flat union.
    pub terminal_tier: Option<usize>,
    /// Every declared terminal target a gate excluded, with the rule that excluded it.
    ///
    /// `terminal_crossed = false` alone cannot distinguish two very different records: tiers
    /// that were never consulted, and tiers that were consulted and found unable to hold the
    /// request. This field makes the second case visible, so a refusal can be read as "the
    /// wider legs were asked and refused this size" rather than "the wider legs do not
    /// exist". Empty when every terminal target was eligible.
    pub terminal_skipped: Vec<(String, &'static str)>,
}

/// A stable, value-free code naming how one attempt ended.
///
/// Closed vocabulary on purpose: the trace names a classification, never a payload, and
/// an operator can group by it without decoding prose. `None` for a successful call.
pub fn outcome_code(error: &switchyard_protocol::client::LlmClientError) -> &'static str {
    use switchyard_protocol::client::LlmClientError as E;
    match error {
        // A reply whose whole output was hidden reasoning is a different fact from a
        // provider that answered with nothing: the model did work, the caller's reservation
        // was too small to hold the answer. Selection reads the distinction, so the code
        // must carry it rather than flattening both into `empty_answer`.
        E::EmptyAnswer { reason, .. } if *reason == "empty_answer_thinking_only" => {
            "empty_answer_thinking"
        }
        E::EmptyAnswer { .. } => "empty_answer",
        E::AttemptTimeout { .. } => "attempt_timeout",
        E::Timeout { .. } => "timeout",
        E::Transport { .. } => "transport",
        E::TemporarilyUnavailable => "unavailable",
        E::ContextWindowExceeded { .. } => "context_window",
        E::DailyRequestAllowanceExhausted { .. } => "daily_allowance_exhausted",
        E::DailyRequestAllowanceUnavailable { .. } => "daily_allowance_unavailable",
        E::MinuteAllowanceThrottled { .. } => "minute_window",
        E::InputExceedsMinuteWindow { .. } => "minute_window_too_small",
        E::UnsupportedFeature { .. } => "unsupported_feature",
        E::InvalidRequest { .. } => "invalid_request",
        E::RequestTranslation(_) | E::RequestEncoding(_) | E::ResponseTranslation(_) => {
            "translation"
        }
        E::Configuration { .. } => "configuration",
        E::ResponseStateLimitExceeded { .. } | E::ResponseStateConflict => "response_state",
        E::UpstreamHttp { status, .. } => match status.as_u16() {
            400 => "http_400",
            401 => "http_401",
            403 => "http_403",
            404 => "http_404",
            413 => "http_413",
            429 => "http_429",
            code if (500..600).contains(&code) => "http_5xx",
            _ => "http_error",
        },
        // `LlmClientError` is non-exhaustive, so an unrecognised future variant is
        // reported as a failure rather than silently claiming a success.
        _ => "failed",
    }
}

/// [`outcome_code`] for a routing-layer error, which may wrap a client error.
pub fn library_outcome_code(error: &switchyard_libsy::LibsyError) -> &'static str {
    match error {
        switchyard_libsy::LibsyError::ClientCall { source, .. } => outcome_code(source),
        switchyard_libsy::LibsyError::NoTargets => "no_targets",
        switchyard_libsy::LibsyError::TargetNotFound { .. } => "target_not_found",
        switchyard_libsy::LibsyError::MissingFinalResponse => "missing_final_response",
        _ => "failed",
    }
}

/// Events emitted inline while [`crate::run`] serves a routing request.
#[derive(Clone, Debug)]
pub enum RunObservation {
    /// Metadata attached to the completed routing outcome.
    Outcome(OutcomeMetadata),
    /// A completed model call requested by the algorithm for routing work.
    LlmCall(LlmCallObservation),
    /// A completed terminal model call made from the routing outcome.
    AnswerCall(LlmCallObservation),
    /// Routing time recorded by the `switchyard.routing_overhead_ms` metric.
    RoutingOverhead(Duration),
    /// This run reordered its candidates before dispatch, naming the target it promoted
    /// and the estimated input tokens that triggered it.
    ///
    /// A separate variant rather than a metric recorded beside the dispatch: the
    /// promotion happens inside the route, below any composite, so an observer is the
    /// only place that can see it.
    ContextPressurePromotion {
        promoted: ModelId,
        estimated_input_tokens: u64,
    },
    /// The route's full selection decision for this request, emitted once per
    /// execution on every exit path - including a refusal where no upstream call was
    /// ever made.
    Selection(Box<SelectionTrace>),
}

/// Request-scoped callback for algorithm-run observations.
///
/// The runner invokes this callback inline while resolving observations. Several
/// runs may invoke the same observer concurrently, so implementations must be
/// thread-safe, fast, and non-blocking.
pub type RunObserver = Arc<dyn Fn(RunObservation) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::client::LlmClientError;

    fn empty(reason: &'static str) -> LlmClientError {
        LlmClientError::EmptyAnswer {
            model: ModelId::from("gemma-4-26b-a4b-it"),
            reason,
        }
    }

    /// The thinking-only reply must survive the trip to the trace as its own code.
    /// Selection excludes exactly this string from demotion, so flattening it back into
    /// `empty_answer` here would silently re-arm the demotion the distinction exists to stop.
    #[test]
    fn a_thinking_only_answer_is_named_in_the_trace_not_flattened() {
        assert_eq!(
            outcome_code(&empty("empty_answer_thinking_only")),
            "empty_answer_thinking"
        );
    }

    /// Every other empty shape keeps the older code: the trace vocabulary gained one word,
    /// it did not lose any.
    #[test]
    fn the_other_empty_shapes_keep_their_code() {
        for reason in [
            "empty_answer_zero_completion",
            "empty_answer_no_usage",
            "empty_answer_output_but_no_text",
        ] {
            assert_eq!(
                outcome_code(&empty(reason)),
                "empty_answer",
                "reason {reason}"
            );
        }
    }
}
