// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The provider's own account of why it refused a request, reduced to a closed vocabulary.
//!
//! A provider answers a refusal with prose, a status and — sometimes — a structured code. The
//! prose is not usable downstream: it is unbounded, it is occasionally wrong, and it can quote
//! the request that produced it, which is exactly what the diagnostic redactor exists to keep
//! out of logs. So the classification happens here, at the boundary where the raw body still
//! exists, and only the bounded answer travels further.
//!
//! Two rules govern every arm below:
//!
//! * A reason is only claimed from a marker the provider actually publishes. An unrecognised
//!   status or code stays [`ProviderReason::Unknown`] rather than being guessed into a quota or
//!   entitlement failure — a guess in that direction authorizes a paid leg on a false premise.
//! * Nothing here changes routing. The outcome vocabulary that gates credential blocks and
//!   terminal legs is untouched; this is classification that an operator, a metric and the
//!   decision record can group by.
//!
//! Scope is deliberately not inferred here. Whether an exhausted allowance belongs to a model
//! or to the whole account is a fact about configuration (which quota domain the target bills
//! to), not about a slug or a status code, so [`ProviderReason::ModelQuotaExhausted`] is only
//! reported where the provider itself scoped the limit to a model.

use switchyard_protocol::client::LlmClientError;

/// Why a provider refused a request, in the provider's own terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderReason {
    /// The provider stated that the account's allowance for the window is spent.
    AccountAllowanceExhausted,
    /// The provider stated that one model's quota for the window is spent.
    ///
    /// Reported only when the refusal positively names a model-scoped limit. Google's Free
    /// tier, for instance, allocates daily requests per model and says so.
    ModelQuotaExhausted,
    /// The provider is rate limiting: a window that refills, not a spent budget.
    Throttled,
    /// The provider does not support something this request requires.
    Incompatible,
    /// The provider requires a continuation this request did not carry.
    ///
    /// The measured family is a reasoning or tool-call continuation whose identifier was not
    /// replayed. It is not a malformed request: the same bytes are correct in a conversation
    /// that keeps its state.
    ContinuationRequired,
    /// The credential was rejected, or its entitlement does not cover this use.
    AuthenticationFailed,
    /// The provider's infrastructure is unavailable, in a way a later attempt may survive.
    TransientUnavailable,
    /// The request itself is invalid in the provider's canonical terms.
    InvalidRequest,
    /// The provider refused for a reason this code cannot substantiate.
    Unknown,
}

impl ProviderReason {
    /// The stable, value-free code carried into logs and decision records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AccountAllowanceExhausted => "provider_account_allowance_exhausted",
            Self::ModelQuotaExhausted => "provider_model_quota_exhausted",
            Self::Throttled => "provider_throttled",
            Self::Incompatible => "provider_incompatible",
            Self::ContinuationRequired => "provider_continuation_required",
            Self::AuthenticationFailed => "provider_auth_failed",
            Self::TransientUnavailable => "provider_unavailable",
            Self::InvalidRequest => "provider_invalid_request",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this reason states that an allowance is spent rather than momentarily short.
    pub const fn is_exhaustion(self) -> bool {
        matches!(
            self,
            Self::AccountAllowanceExhausted | Self::ModelQuotaExhausted
        )
    }
}

/// Codes providers publish for a request they consider malformed.
///
/// Closed and drawn from published documentation. A 400 carrying something else is not
/// classified here: "the provider said 400" is not evidence that the canonical request was
/// wrong, and calling it invalid is what hides a provider-specific incompatibility.
const MALFORMED_CODES: &[&str] = &[
    "invalid_request_error",
    "invalid_request",
    "invalid_parameter",
    "invalid_value",
    "missing_required_parameter",
    "bad_request",
];

/// Codes that name a continuation the provider needed and did not receive.
const CONTINUATION_CODES: &[&str] = &[
    "missing_reasoning_replay",
    "reasoning_required",
    "previous_response_not_found",
    "previous_response_id_required",
    "missing_tool_result",
    "tool_call_not_found",
];

/// Words that name a capability the provider does not have, in a 400/404/405.
///
/// Matched against the provider's own short `code`/`type` first and its message second. Kept
/// to phrases that a provider uses for an unsupported operation; a bare "unsupported" in free
/// prose is not enough, because that is how a malformed value is described too.
const INCOMPATIBLE_MARKERS: &[&str] = &[
    "unsupported_feature",
    "unsupported_capability",
    "unsupported_parameter",
    "unsupported_value",
    "unsupported_model",
    "not_supported",
    "does_not_support",
    "does not support",
    "unknown_parameter",
    "unknown_field",
    "model_not_found",
    "no_such_model",
    "unrecognized request argument",
];

/// Classify one provider refusal.
///
/// `status` is the HTTP status; `body` is the provider's response body as received. The body is
/// read here and not retained: the returned value carries no message, identifier or credential.
pub fn classify(status: u16, body: &str) -> ProviderReason {
    let document = serde_json::from_str::<serde_json::Value>(body).ok();
    let error = document.as_ref().and_then(|value| value.get("error"));
    let code = error
        .and_then(|error| {
            error
                .get("code")
                .and_then(serde_json::Value::as_str)
                .or_else(|| error.get("type").and_then(serde_json::Value::as_str))
        })
        .map(str::to_ascii_lowercase);
    let message = error
        .and_then(|error| error.get("message").and_then(serde_json::Value::as_str))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let limit_source = error
        .and_then(|error| error.get("metadata").and_then(|metadata| metadata.get("limit_source")))
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase);

    match status {
        // Rejected credential or an entitlement that does not cover this use. Both are
        // configuration facts: neither improves by being sent again.
        401 | 403 => ProviderReason::AuthenticationFailed,
        // The account's balance, which every model on the credential shares.
        402 => ProviderReason::AccountAllowanceExhausted,
        404 | 405 => {
            if names_any(&code, &message, INCOMPATIBLE_MARKERS) {
                ProviderReason::Incompatible
            } else {
                ProviderReason::Unknown
            }
        }
        408 | 425 => ProviderReason::TransientUnavailable,
        429 => {
            if names_exhausted_allowance(limit_source.as_deref(), &code, &message) {
                ProviderReason::AccountAllowanceExhausted
            } else if scopes_limit_to_one_model(body) {
                ProviderReason::ModelQuotaExhausted
            } else {
                ProviderReason::Throttled
            }
        }
        400 | 412 | 422 => {
            if names_any(&code, &message, CONTINUATION_CODES) {
                ProviderReason::ContinuationRequired
            } else if names_any(&code, &message, INCOMPATIBLE_MARKERS) {
                ProviderReason::Incompatible
            } else if code
                .as_deref()
                .is_some_and(|code| MALFORMED_CODES.contains(&code))
            {
                ProviderReason::InvalidRequest
            } else {
                // A 400 the provider did not explain in a term this code knows. Reported as
                // unknown rather than as a malformed canonical request: the request may be
                // perfectly well formed for a provider that expected something else.
                ProviderReason::Unknown
            }
        }
        status if (500..600).contains(&status) => {
            // A provider that masks a permanent condition behind a 5xx names the real cause
            // in a sibling field. That is not a transient infrastructure fault.
            let cause = crate::run::canonical_error_type(body)
                .map(str::to_ascii_lowercase)
                .or(code);
            match cause.as_deref() {
                Some("authentication" | "permission_denied") => ProviderReason::AuthenticationFailed,
                Some("payment_required" | "insufficient_quota") => {
                    ProviderReason::AccountAllowanceExhausted
                }
                _ => ProviderReason::TransientUnavailable,
            }
        }
        _ => ProviderReason::Unknown,
    }
}

/// The bounded reason behind a client error, when it has one.
///
/// Only an upstream refusal carries a provider's own account of itself. Local outcomes — a
/// spent local budget, a minute window, a translation failure — already have codes of their
/// own and must not be dressed up as provider statements.
pub fn reason_code(error: &LlmClientError) -> Option<&'static str> {
    match error {
        LlmClientError::UpstreamHttp { status, body } => {
            Some(classify(status.as_u16(), body).as_str())
        }
        _ => None,
    }
}

/// Whether the provider named an account-wide allowance as spent.
///
/// The OpenRouter marker is the measured one; the billing codes and the free-model sentence are
/// the published spellings of the same fact without it.
fn names_exhausted_allowance(limit_source: Option<&str>, code: &Option<String>, message: &str) -> bool {
    if limit_source.is_some_and(|source| source.starts_with("openrouter_free_tier")) {
        return true;
    }
    if code.as_deref().is_some_and(|code| {
        matches!(
            code,
            "credit_balance_exhausted"
                | "organization_spend_limit_exceeded"
                | "project_spend_limit_exceeded"
                | "organization_usage_limit_exceeded"
                | "insufficient_quota"
        )
    }) {
        return true;
    }
    message.contains("free-models-per-day") || message.contains("insufficient_quota")
}

/// Whether the provider scoped a spent quota to one model rather than the account.
///
/// Google's Free tier allocates daily requests per model; a daily `quota_exceeded` there is a
/// statement about that model, not about the pool. The distinction is taken from the provider's
/// structured condition, never from the model slug in the URL.
fn scopes_limit_to_one_model(body: &str) -> bool {
    crate::google::parse(Some(429), body).is_some_and(|error| {
        error.condition == crate::google::GoogleCondition::DailyQuota
            && !error.recovery().stop_candidate_list
    })
}

fn names_any(code: &Option<String>, message: &str, markers: &[&str]) -> bool {
    if let Some(code) = code {
        let code = code.to_ascii_lowercase();
        if markers.contains(&code.as_str()) {
            return true;
        }
    }
    markers.iter().any(|marker| message.contains(marker))
}

/// The error-class label a provider reason contributes to the operational record.
///
/// Deliberately finer than the local classes: `provider_incompatibility` and
/// `provider_invalid_request` answer different questions about the same 400, and an operator
/// deciding whether to change the request or the target needs the difference.
pub fn error_class(reason: ProviderReason) -> &'static str {
    reason.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify_body(status: u16, body: &str) -> ProviderReason {
        classify(status, body)
    }

    #[test]
    fn an_openrouter_free_tier_marker_is_an_account_allowance() {
        let reason = classify_body(
            429,
            r#"{"error":{"code":"rate_limit_exceeded","message":"rate limited","metadata":{"limit_source":"openrouter_free_tier_daily"}}}"#,
        );
        assert_eq!(reason, ProviderReason::AccountAllowanceExhausted);
        assert!(reason.is_exhaustion());
    }

    #[test]
    fn a_free_model_daily_sentence_without_metadata_is_an_account_allowance() {
        let reason = classify_body(
            429,
            r#"{"error":{"code":"upstream_error","type":"upstream_error","message":"Rate limit exceeded: free-models-per-day-high-balance."}}"#,
        );
        assert_eq!(reason, ProviderReason::AccountAllowanceExhausted);
    }

    #[test]
    fn a_published_billing_code_is_an_account_allowance() {
        assert_eq!(
            classify_body(429, r#"{"error":{"code":"insufficient_quota"}}"#),
            ProviderReason::AccountAllowanceExhausted
        );
    }

    #[test]
    fn a_balance_refusal_is_an_account_allowance() {
        assert_eq!(
            classify_body(402, r#"{"error":{"message":"payment required"}}"#),
            ProviderReason::AccountAllowanceExhausted
        );
    }

    #[test]
    fn a_google_daily_quota_scoped_to_one_model_is_model_scoped() {
        // Google's Free tier allocates daily requests per model; the condition names the
        // model, so this must not be reported as the account running out.
        let reason = classify_body(
            429,
            r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}"#,
        );
        assert_eq!(reason, ProviderReason::ModelQuotaExhausted);
        assert!(reason.is_exhaustion());
    }

    #[test]
    fn a_bare_throttle_stays_a_throttle() {
        assert_eq!(
            classify_body(429, r#"{"error":{"code":"rate_limit_exceeded"}}"#),
            ProviderReason::Throttled
        );
        assert_eq!(classify_body(429, ""), ProviderReason::Throttled);
    }

    #[test]
    fn rejected_credentials_and_entitlements_are_authentication() {
        assert_eq!(
            classify_body(401, r#"{"error":{"code":"invalid_api_key"}}"#),
            ProviderReason::AuthenticationFailed
        );
        assert_eq!(
            classify_body(403, r#"{"error":{"message":"forbidden"}}"#),
            ProviderReason::AuthenticationFailed
        );
    }

    #[test]
    fn a_documented_malformed_code_is_an_invalid_request() {
        assert_eq!(
            classify_body(400, r#"{"error":{"code":"invalid_request_error","message":"bad"}}"#),
            ProviderReason::InvalidRequest
        );
    }

    #[test]
    fn an_unsupported_feature_is_an_incompatibility_not_an_invalid_request() {
        assert_eq!(
            classify_body(
                400,
                r#"{"error":{"code":"unsupported_parameter","message":"unknown field: audio"}}"#
            ),
            ProviderReason::Incompatible
        );
    }

    #[test]
    fn a_missing_continuation_is_its_own_class() {
        assert_eq!(
            classify_body(
                400,
                r#"{"error":{"code":"missing_reasoning_replay","message":"reasoning item required"}}"#
            ),
            ProviderReason::ContinuationRequired
        );
    }

    #[test]
    fn an_unexplained_400_stays_unknown() {
        // The measured DeepSeek shape: a 400 that names neither a documented malformed-request
        // code nor a capability. Guessing "invalid request" here is what hides an
        // incompatibility the operator could otherwise fix.
        assert_eq!(
            classify_body(400, r#"{"error":{"message":"Bad Request"}}"#),
            ProviderReason::Unknown
        );
        assert_eq!(classify_body(400, "not json at all"), ProviderReason::Unknown);
    }

    #[test]
    fn a_five_hundred_is_transient_unless_it_masks_a_permanent_condition() {
        assert_eq!(
            classify_body(503, r#"{"error":{"message":"overloaded"}}"#),
            ProviderReason::TransientUnavailable
        );
        assert_eq!(
            classify_body(
                500,
                r#"{"error":{"code":"server_error","error_type":"payment_required","message":"masked"}}"#
            ),
            ProviderReason::AccountAllowanceExhausted
        );
        assert_eq!(
            classify_body(
                500,
                r#"{"error":{"code":"server_error","error_type":"authentication","message":"masked"}}"#
            ),
            ProviderReason::AuthenticationFailed
        );
    }

    #[test]
    fn the_classification_never_carries_the_body() {
        const LEAKED: &str = "SECRET-request-content";
        let reason = classify_body(
            400,
            &format!(r#"{{"error":{{"code":"invalid_request_error","message":"{LEAKED}"}}}}"#),
        );
        assert_eq!(reason.as_str(), "provider_invalid_request");
        for code in [
            ProviderReason::AccountAllowanceExhausted,
            ProviderReason::ModelQuotaExhausted,
            ProviderReason::Throttled,
            ProviderReason::Incompatible,
            ProviderReason::ContinuationRequired,
            ProviderReason::AuthenticationFailed,
            ProviderReason::TransientUnavailable,
            ProviderReason::InvalidRequest,
            ProviderReason::Unknown,
        ] {
            assert!(!code.as_str().contains(LEAKED));
            assert!(!code.as_str().contains("SECRET"));
        }
    }

    #[test]
    fn a_local_error_carries_no_provider_reason() {
        assert_eq!(
            reason_code(&LlmClientError::MinuteAllowanceThrottled {
                model: "vendor/model".into(),
                scope: "model".to_string(),
                retry_after_secs: 30,
            }),
            None,
            "a local throttle is not a provider statement"
        );
    }
}
