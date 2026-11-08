// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shadow evaluation of context-window admission.
//!
//! WHAT IS ALREADY ENFORCED, AND WHERE
//!
//! A composite skips a child whose window cannot hold the request, counting the
//! caller's output reservation with the input, before any provider call. That is
//! deployed and correct: a too-small child costs nothing and the next child is
//! tried.
//!
//! What has no equivalent is a LEAF route. `direct-*`, the primitive routes and the
//! tiers are dispatched at whatever size the caller sends, and the provider answers
//! with a rejection. The one production case on record is exactly this shape: a
//! leaf route, 1,077,758 input tokens plus a 32,768-token reserve against a
//! 1,048,576 window, which reached the provider and came back as an opaque 400.
//!
//! WHY SHADOW FIRST
//!
//! Refusing locally is a behaviour change for every caller, so it is measured
//! before it is enforced. This module computes what admission *would* decide for
//! every route -- leaf or composite -- and reports the arithmetic; nothing here
//! refuses a request. Because the estimator is approximate and the reasoning it
//! cannot see is provider-specific, the report separates a firm refusal from a
//! verdict that only the uncertainty band decides.
//!
//! WHAT THE NUMBERS MEAN
//!
//! * `estimated_input_tokens` is the shared estimator used by escalation decisions —
//!   one estimator, so the shadow verdict is comparable with existing behaviour.
//! * `completion_reserve` is the output budget the caller asked for. It consumes the
//!   same window as the input: the observed production failure asked for 32,768
//!   output tokens against a 1,048,576 window with 1,077,758 input tokens.
//! * `reasoning_allowance_tokens` covers reasoning that a provider re-prices on
//!   continuation. It is a floor, not a bound: a provider may charge more for
//!   stored thinking than its visible text suggests, which the uncertainty band and
//!   the `limit_source` field exist to keep visible rather than hide.
//! * `effective_limit` is the *route's declared* window, not the model catalogue's.
//!   A composite resolves to the smallest declared window among its children,
//!   because a larger request makes that child ineligible and the composite cannot
//!   silently exceed it.
//!
//! The estimator is character-based, so it is approximate by construction. The band
//! is asymmetric on purpose: under-estimating input is the dangerous direction.

use serde_json::Value;

/// Which window the evaluation used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitSource {
    /// The route (or composite) declares a `context_window`.
    Declared,
    /// Nothing declares a window, so no local decision is possible.
    Undeclared,
    /// A composite has at least one child with no declared window, so the smallest
    /// window is not knowable.
    CompositeIncomplete,
}

impl LimitSource {
    /// Stable label for logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Undeclared => "undeclared",
            Self::CompositeIncomplete => "composite_incomplete",
        }
    }
}

/// The hypothetical admission verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowDecision {
    /// Comfortably inside the window, including the uncertainty band.
    Admit,
    /// Inside the point estimate but inside the band's upper edge: a future
    /// enforcement rule must decide which side of the band it trusts.
    AdmitUncertain,
    /// Over the window even at the point estimate.
    Refuse,
    /// No declared window, so no verdict.
    Unknown,
}

impl ShadowDecision {
    /// Stable label for logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Admit => "admit",
            Self::AdmitUncertain => "admit_uncertain",
            Self::Refuse => "refuse",
            Self::Unknown => "unknown",
        }
    }
}

/// Full shadow accounting for one request against one route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowAccounting {
    /// Estimated input tokens (shared estimator).
    pub estimated_input_tokens: u64,
    /// Output budget the caller asked for.
    pub completion_reserve: u64,
    /// Floor allowance for reasoning the provider re-prices on continuation.
    pub reasoning_allowance_tokens: u64,
    /// Upper edge of the estimator's uncertainty band, in input tokens.
    pub estimated_input_tokens_high: u64,
    /// The window admission would be checked against.
    pub effective_limit: Option<u32>,
    /// Where that window came from.
    pub limit_source: LimitSource,
    /// Hypothetical verdict.
    pub decision: ShadowDecision,
    /// How far over the window the point estimate is, when it is over.
    pub over_by_tokens: u64,
}

/// Fractional headroom added to the estimate to form the uncertainty band.
///
/// Character-based estimation under-counts structured JSON, tool schemas and
/// terse non-English text more than it over-counts. 25% is a floor chosen to make
/// the band visible; it is not a validated tokenizer margin.
const UNCERTAINTY_NUMERATOR: u64 = 125;
const UNCERTAINTY_DENOMINATOR: u64 = 100;

/// Reads the caller's requested output budget from a raw body.
///
/// Both spellings are honoured because callers use both; a missing budget means
/// the provider's default applies, which this module cannot know, so it returns 0
/// and the decision is correspondingly optimistic.
pub fn completion_reserve(body: &Value) -> u64 {
    ["max_output_tokens", "max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|key| body.get(*key).and_then(Value::as_u64))
        .unwrap_or(0)
}

/// Provider-specific overhead for reasoning that the input estimate cannot see.
///
/// The shared estimator already counts visible reasoning text, so an allowance
/// derived from that text would double-count it. What remains uncounted is:
///
/// * opaque reasoning (ciphertext, signatures, provider item ids), whose price is
///   not knowable from the body; and
/// * providers that re-price stored thinking above its visible length on
///   continuation.
///
/// Both are provider facts, not body facts, so the allowance is an explicit input
/// rather than something this module can derive. Returning a floor keeps the
/// arithmetic honest: it can only make the shadow stricter, never laxer.
pub fn provider_reasoning_overhead(opaque_reasoning_present: bool, provider_floor: u64) -> u64 {
    if opaque_reasoning_present {
        provider_floor
    } else {
        0
    }
}

/// Evaluates admission for a request against a declared window.
///
/// Pure: it reads the body and the declared limit and returns the arithmetic that
/// a future enforcement rule would act on.
pub fn evaluate(
    body: &Value,
    completion_reserve: u64,
    reasoning_allowance_tokens: u64,
    effective_limit: Option<u32>,
    limit_source: LimitSource,
    estimate: impl Fn(&Value) -> u64,
) -> ShadowAccounting {
    let estimated_input_tokens = estimate(body);
    let estimated_input_tokens_high = estimated_input_tokens
        .saturating_mul(UNCERTAINTY_NUMERATOR)
        .div_ceil(UNCERTAINTY_DENOMINATOR);

    let Some(limit) = effective_limit.filter(|limit| *limit > 0) else {
        return ShadowAccounting {
            estimated_input_tokens,
            completion_reserve,
            reasoning_allowance_tokens,
            estimated_input_tokens_high,
            effective_limit: None,
            limit_source,
            decision: ShadowDecision::Unknown,
            over_by_tokens: 0,
        };
    };
    let limit = u64::from(limit);

    let point = estimated_input_tokens
        .saturating_add(completion_reserve)
        .saturating_add(reasoning_allowance_tokens);
    let high = estimated_input_tokens_high
        .saturating_add(completion_reserve)
        .saturating_add(reasoning_allowance_tokens);

    let (decision, over_by_tokens) = if point > limit {
        (ShadowDecision::Refuse, point - limit)
    } else if high > limit {
        (ShadowDecision::AdmitUncertain, 0)
    } else {
        (ShadowDecision::Admit, 0)
    };

    ShadowAccounting {
        estimated_input_tokens,
        completion_reserve,
        reasoning_allowance_tokens,
        estimated_input_tokens_high,
        effective_limit: Some(limit as u32),
        limit_source,
        decision,
        over_by_tokens,
    }
}

/// Per-child accounting for one composite, in declared order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChildShadow {
    /// Index in the composite's declared child order.
    pub index: usize,
    /// The child's declared window, when it declares one.
    pub effective_limit: Option<u32>,
    /// What admission would decide for this child alone.
    pub decision: ShadowDecision,
}

/// Shadow accounting for a composite: every child, and the verdict that matters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositeShadow {
    /// Per-child verdicts, in declared order.
    pub children: Vec<ChildShadow>,
    /// The composite's own advertised window.
    pub advertised_limit: Option<u32>,
    /// The window of the largest child that declares one.
    pub widest_child_limit: Option<u32>,
    /// Whether at least one child could take the request. This, not the smallest
    /// child, is the admission question.
    pub any_child_admits: bool,
    /// The first child, in order, that would admit the request. `None` when no
    /// child can.
    pub first_admitting_child: Option<usize>,
    /// Children that would be skipped for size before any provider call.
    pub skipped_for_size: Vec<usize>,
}

/// Evaluates a request against a composite's children.
///
/// ADMISSION IS PER CHILD, NOT PER COMPOSITE.
///
/// A composite runs its children in order and skips one that cannot serve the
/// request: a child whose window is too small is `skipped_ineligible` in the
/// composition, before any provider call, and the next child is tried. So the
/// question admission must answer is "can ANY child take this?", never "can the
/// smallest child take this?".
///
/// Judging a composite by its smallest child refuses requests the composite would
/// have served from a larger sibling. That is exactly the false positive this
/// function exists to avoid, and it would be a regression against today's
/// behaviour, which serves those requests correctly by skipping the small child.
///
/// The advertised window is therefore reported but never used as the binding
/// limit; it is what the composite tells callers, not what it can serve.
pub fn evaluate_composite(
    child_limits: &[Option<u32>],
    body: &Value,
    completion_reserve: u64,
    reasoning_allowance_tokens: u64,
    advertised_limit: Option<u32>,
    estimate: impl Fn(&Value) -> u64 + Copy,
) -> CompositeShadow {
    let mut children = Vec::with_capacity(child_limits.len());
    let mut any_child_admits = false;
    let mut first_admitting_child = None;
    let mut skipped_for_size = Vec::new();

    for (index, limit) in child_limits.iter().copied().enumerate() {
        let accounting = evaluate(
            body,
            completion_reserve,
            reasoning_allowance_tokens,
            limit,
            if limit.is_some() {
                LimitSource::Declared
            } else {
                LimitSource::Undeclared
            },
            estimate,
        );
        let admits = matches!(
            accounting.decision,
            ShadowDecision::Admit | ShadowDecision::AdmitUncertain
        );
        if admits {
            any_child_admits = true;
            first_admitting_child.get_or_insert(index);
        } else if accounting.decision == ShadowDecision::Refuse {
            skipped_for_size.push(index);
        }
        children.push(ChildShadow {
            index,
            effective_limit: limit,
            decision: accounting.decision,
        });
    }

    let widest_child_limit = child_limits.iter().flatten().copied().max();
    CompositeShadow {
        children,
        advertised_limit,
        widest_child_limit,
        any_child_admits,
        first_admitting_child,
        skipped_for_size,
    }
}

/// Resolves the window a composite guarantees to every child.
///
/// This is the smallest declared child window, and it is the right number for one
/// purpose only: describing what the composite can serve *without* any child being
/// skipped. It is NOT the admission limit -- see [`evaluate_composite`].
pub fn guaranteed_child_window(child_limits: &[Option<u32>]) -> (Option<u32>, LimitSource) {
    if child_limits.is_empty() {
        return (None, LimitSource::Undeclared);
    }
    if child_limits.iter().any(Option::is_none) {
        return (None, LimitSource::CompositeIncomplete);
    }
    match child_limits.iter().flatten().copied().min() {
        Some(limit) => (Some(limit), LimitSource::Declared),
        None => (None, LimitSource::Undeclared),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn estimate(body: &Value) -> u64 {
        // Mirrors the shared estimator closely enough for arithmetic tests: one
        // token per four ASCII bytes, plus structural overhead.
        let text = body.to_string();
        (text.len() as u64).div_ceil(4) + 8
    }

    #[test]
    fn comfortable_request_is_admitted() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let accounting = evaluate(
            &body,
            1024,
            0,
            Some(128_000),
            LimitSource::Declared,
            estimate,
        );
        assert_eq!(accounting.decision, ShadowDecision::Admit);
        assert_eq!(accounting.over_by_tokens, 0);
    }

    // The production incident, reproduced as arithmetic: 1,077,758 input tokens
    // plus a 32,768-token reserve against a 1,048,576 window.
    #[test]
    fn observed_production_overflow_is_refused_by_the_shadow() {
        let mut body = json!({"messages": [{"role": "user", "content": ""}]});
        // Build a body whose point estimate lands near the observed size.
        body["messages"][0]["content"] = Value::String("x".repeat(4 * 1_077_758));
        let accounting = evaluate(
            &body,
            32_768,
            0,
            Some(1_048_576),
            LimitSource::Declared,
            estimate,
        );
        assert_eq!(accounting.decision, ShadowDecision::Refuse);
        assert!(accounting.over_by_tokens > 60_000, "{accounting:?}");
        // The refusal must be attributable to size, not to the reserve alone.
        assert!(accounting.estimated_input_tokens > 1_048_576);
    }

    #[test]
    fn reserve_alone_can_push_a_request_over() {
        let body = json!({"messages": [{"role": "user", "content": ""}]});
        let input = estimate(&body);
        let limit = input as u32 + 100;
        let under = evaluate(&body, 50, 0, Some(limit), LimitSource::Declared, estimate);
        let over = evaluate(&body, 500, 0, Some(limit), LimitSource::Declared, estimate);
        assert_eq!(under.decision, ShadowDecision::Admit);
        assert_eq!(over.decision, ShadowDecision::Refuse);
    }

    // A request inside the point estimate but inside the uncertainty band is
    // reported distinctly, so an enforcement decision is made deliberately.
    #[test]
    fn band_edge_is_reported_as_uncertain_not_admitted() {
        let mut body = json!({"messages": [{"role": "user", "content": ""}]});
        let target = 100_000u64;
        // 4 bytes per estimated token; choose a limit inside the 25% band.
        body["messages"][0]["content"] = Value::String("x".repeat((target * 4) as usize));
        let input = estimate(&body);
        let limit = (input + input / 10) as u32; // 10% above point, inside the 25% band
        let accounting = evaluate(&body, 0, 0, Some(limit), LimitSource::Declared, estimate);
        assert_eq!(accounting.decision, ShadowDecision::AdmitUncertain);
        assert_eq!(accounting.over_by_tokens, 0);
    }

    #[test]
    fn undeclared_window_yields_no_verdict() {
        let body = json!({"messages": []});
        let accounting = evaluate(&body, 0, 0, None, LimitSource::Undeclared, estimate);
        assert_eq!(accounting.decision, ShadowDecision::Unknown);
        assert_eq!(accounting.limit_source, LimitSource::Undeclared);
    }

    // Visible reasoning is already inside the input estimate; the allowance exists
    // only for what the estimate cannot see, so an allowance can only tighten the
    // verdict, never launder an oversized request into an admitted one.
    #[test]
    fn provider_reasoning_overhead_only_tightens_the_verdict() {
        let body = json!({
            "messages": [
                {"role": "assistant", "content": "ok", "reasoning_content": "a".repeat(4000)},
                {"role": "user", "content": "again"}
            ]
        });
        let input = estimate(&body);
        let limit = input as u32 + 500;

        let no_overhead = evaluate(&body, 0, 0, Some(limit), LimitSource::Declared, estimate);
        assert_eq!(no_overhead.decision, ShadowDecision::Admit);

        let with_overhead = evaluate(
            &body,
            0,
            provider_reasoning_overhead(true, 4096),
            Some(limit),
            LimitSource::Declared,
            estimate,
        );
        assert_eq!(with_overhead.decision, ShadowDecision::Refuse);
        assert_eq!(with_overhead.over_by_tokens, 4096 - 500);
    }

    // Opaque reasoning cannot be measured, so it contributes nothing on its own.
    // The provider floor is what covers it, and the floor applies only when opaque
    // reasoning is actually present.
    #[test]
    fn opaque_reasoning_is_covered_only_by_the_provider_floor() {
        let body = json!({
            "messages": [
                {"role": "assistant", "content": "x",
                 "reasoning_details": [{"type": "reasoning", "encrypted_content": "A".repeat(8000)}]}
            ]
        });
        assert_eq!(provider_reasoning_overhead(false, 4096), 0);
        let input = estimate(&body);
        // Above the uncertainty band, so the only thing under test is the floor.
        let limit = (input + input * 30 / 100) as u32;
        let bare = evaluate(&body, 0, 0, Some(limit), LimitSource::Declared, estimate);
        let floored = evaluate(
            &body,
            0,
            provider_reasoning_overhead(true, 4096),
            Some(limit),
            LimitSource::Declared,
            estimate,
        );
        assert_eq!(bare.decision, ShadowDecision::Admit);
        assert_eq!(floored.decision, ShadowDecision::Refuse);
    }

    #[test]
    fn guaranteed_window_is_the_smallest_declared_child() {
        assert_eq!(
            guaranteed_child_window(&[Some(1_000_000), Some(225_280), Some(262_144)]),
            (Some(225_280), LimitSource::Declared)
        );
        assert_eq!(
            guaranteed_child_window(&[Some(1_000_000), None]),
            (None, LimitSource::CompositeIncomplete)
        );
        assert_eq!(
            guaranteed_child_window(&[]),
            (None, LimitSource::Undeclared)
        );
    }

    // The request is too big for the first child and fits the second. The composite
    // skips the first and serves from the second, so admission must admit it. A
    // verdict based on the smallest child would refuse a request the deployment
    // serves correctly today.
    #[test]
    fn a_composite_admits_when_any_child_can_take_the_request() {
        let body = json!({"messages": [{"role": "user", "content": "x".repeat(4 * 300_000)}]});
        let shadow = evaluate_composite(
            &[Some(225_280), Some(1_048_576)],
            &body,
            0,
            0,
            Some(1_000_000),
            estimate,
        );
        assert!(shadow.any_child_admits, "{shadow:?}");
        assert_eq!(shadow.first_admitting_child, Some(1));
        assert_eq!(shadow.skipped_for_size, vec![0]);
        assert_eq!(shadow.children[0].decision, ShadowDecision::Refuse);
    }

    // No child can take it: only then is a composite-level refusal correct.
    #[test]
    fn a_composite_refuses_only_when_no_child_can() {
        let body = json!({"messages": [{"role": "user", "content": "x".repeat(4 * 300_000)}]});
        let shadow = evaluate_composite(
            &[Some(225_280), Some(262_144)],
            &body,
            0,
            0,
            Some(1_048_576),
            estimate,
        );
        assert!(!shadow.any_child_admits, "{shadow:?}");
        assert_eq!(shadow.first_admitting_child, None);
        assert_eq!(shadow.skipped_for_size, vec![0, 1]);
        // The advertised window is recorded but is not what admitted the request.
        assert_eq!(shadow.advertised_limit, Some(1_048_576));
        assert_eq!(shadow.widest_child_limit, Some(262_144));
    }

    // A child that declares no window cannot be judged, and must not be assumed
    // infinite: it is reported as unknown, and it never counts as an admission.
    #[test]
    fn an_undeclared_child_window_does_not_admit_by_itself() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let shadow = evaluate_composite(&[None], &body, 0, 0, Some(1_000_000), estimate);
        assert_eq!(shadow.children[0].decision, ShadowDecision::Unknown);
        assert!(!shadow.any_child_admits);
    }

    // The guaranteed window is not the admission limit. A composite advertised at
    // 1,000,000 with a 225,280 child guarantees only 225,280 to every child, yet a
    // 300,000-token request is still servable by the larger sibling -- so the
    // conservative verdict (refuse) and the composite verdict (admit) must disagree,
    // and the composite verdict is the one that governs admission.
    #[test]
    fn the_guaranteed_window_is_not_the_admission_verdict() {
        let body = json!({"messages": [{"role": "user", "content": "x".repeat(4 * 300_000)}]});
        let (limit, source) = guaranteed_child_window(&[Some(1_000_000), Some(225_280)]);
        let conservative = evaluate(&body, 0, 0, limit, source, estimate);
        assert_eq!(conservative.effective_limit, Some(225_280));
        assert_eq!(conservative.decision, ShadowDecision::Refuse);

        let shadow = evaluate_composite(
            &[Some(1_000_000), Some(225_280)],
            &body,
            0,
            0,
            Some(1_000_000),
            estimate,
        );
        assert!(shadow.any_child_admits);
        assert_eq!(shadow.first_admitting_child, Some(0));
        assert_eq!(shadow.skipped_for_size, vec![1]);
    }
}
