// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical client error re-export and shared context-window-overflow detection.
//!
//! The client owns overflow detection so callers receive one stable error
//! classification across providers.

use serde_json::Value;

pub use switchyard_protocol::LlmClientError;

/// Result alias for LLM client operations.
pub type Result<T> = std::result::Result<T, LlmClientError>;

/// Detects a context-overflow body using a provider-supplied structured check
/// and a substring phrase list.
///
/// Parses the body once, runs the structured check (e.g. against `error.code`),
/// then falls back to matching phrases against `error.message` or — when the
/// body is not JSON — the raw body. Centralizing the shape means each new
/// provider-wrap of the canonical error is a one-line phrase entry, not a fork
/// of the parsing logic.
pub(crate) fn is_overflow_body<F>(body: &str, structured_check: F, phrases: &[&str]) -> bool
where
    F: Fn(&Value) -> bool,
{
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if structured_check(&value) {
            return true;
        }
        if let Some(message) = value
            .get("error")
            .and_then(|err| err.get("message"))
            .and_then(Value::as_str)
            && contains_any(message, phrases)
        {
            return true;
        }
    }
    // Some upstream proxies return plain-text bodies; fall through to a string
    // match on the raw body.
    contains_any(body, phrases)
}

// Case-insensitive substring match of any phrase against the message.
fn contains_any(message: &str, phrases: &[&str]) -> bool {
    let lower = message.to_ascii_lowercase();
    phrases.iter().any(|phrase| lower.contains(phrase))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHRASES: &[&str] = &["context window", "too long"];

    fn never(_value: &Value) -> bool {
        false
    }

    #[test]
    fn structured_check_short_circuits() {
        let body = r#"{"error":{"code":"context_length_exceeded","message":"unrelated"}}"#;
        let matched = is_overflow_body(
            body,
            |value| {
                value
                    .get("error")
                    .and_then(|err| err.get("code"))
                    .and_then(Value::as_str)
                    == Some("context_length_exceeded")
            },
            &[],
        );
        assert!(matched);
    }

    #[test]
    fn falls_back_to_message_phrase_match() {
        let body = r#"{"error":{"message":"prompt too long"}}"#;
        assert!(is_overflow_body(body, never, PHRASES));
    }

    #[test]
    fn matches_plain_text_body() {
        assert!(is_overflow_body(
            "plain text mentioning context window",
            never,
            PHRASES
        ));
    }

    #[test]
    fn non_match_returns_false() {
        let body = r#"{"error":{"message":"rate limit exceeded"}}"#;
        assert!(!is_overflow_body(body, never, PHRASES));
    }

    // --- permanent-quota 429 classification: realistic provider bodies ------

    // POSITIVE: the real OpenAI insufficient-quota 429 body (structured code).
    #[test]
    fn openai_insufficient_quota_body_is_permanent() {
        let body = r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#;
        assert!(is_permanent_quota_429(body));
    }

    // POSITIVE: short plain-text aggregator bodies (the case bare "credits"
    // exists for).
    #[test]
    fn plain_text_credit_exhaustion_bodies_are_permanent() {
        for body in [
            "You've run out of credits. Please top up to continue.",
            "not enough credits",
            "Your credits have been exhausted",
            "payment required: quota exhausted",
            "you have used all your usage limit for this billing period",
        ] {
            assert!(is_permanent_quota_429(body), "body: {body}");
        }
    }

    // NEGATIVE: realistic TRANSIENT 429 bodies must keep the bounded retry.
    #[test]
    fn transient_rate_limit_bodies_are_not_permanent() {
        for body in [
            // OpenAI TPM throttle.
            r#"{"error":{"message":"Rate limit reached for gpt-5.6-luna on tokens per min (TPM): Limit 30000, Used 29999. Please try again in 20ms.","type":"tokens","param":null,"code":"rate_limit_exceeded"}}"#,
            // Per-minute request cap.
            r#"{"error":{"message":"Number of requests has exceeded your per-minute rate limit. Please slow down and try again."}}"#,
            // Retry-After-style plain text.
            "Too many requests. Please retry after 30 seconds.",
            // DeepSeek transient throttle phrasing.
            r#"{"error":{"message":"rate limit exceeded, please retry later"}}"#,
        ] {
            assert!(
                !is_permanent_quota_429(body),
                "transient body misclassified permanent: {body}"
            );
        }
    }

    // DOCUMENTED EDGE (accepted): an aggregator daily-cap 429 that MENTIONS
    // credits classifies as permanent. That is the intended semantic, not a
    // false positive: the same-candidate retry budget is futile against a
    // daily cap (resets at midnight / requires payment), so the correct fleet
    // behavior is to advance immediately to the next candidate — exactly what
    // permanent-quota classification does. No realistic provider body is known
    // where a 429 mentioning "credits" IS fixed by retrying the same
    // candidate; if soak or runtime evidence surfaces one, narrow bare
    // "credits" to credit-exhaustion phrases ("out of credits",
    // "insufficient credits", ...) as the recorded follow-up.
    #[test]
    fn credits_mentioning_daily_cap_is_classified_permanent_intentionally() {
        let body = "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day.";
        assert!(is_permanent_quota_429(body));
    }
}

/// Canonical permanent-quota markers from OpenAI and compatible providers.
///
/// Matched against the structured `error.code` (e.g. OpenAI's
/// `insufficient_quota`) or the `error.message` (both the OpenAI phrasing and
/// the provider-equivalent permanent quota conditions). Deliberately narrow so
/// a genuinely transient rate-limit 429 (`rate_limit_exceeded`, Retry-After)
/// is never misclassified.
const PERMANENT_QUOTA_CODES: &[&str] = &["insufficient_quota"];
const PERMANENT_QUOTA_MESSAGE_PHRASES: &[&str] = &[
    "insufficient_quota",
    "you exceeded your current quota",
    "exceeded your current quota",
    "your billing details",
    "billing quota",
    "exhausted your quota",
    "quota has been exhausted",
    "usage limit",
    "payment required",
    "credits",
];

/// Structured check: the canonical OpenAI `error.code` for a permanent quota
/// condition.
fn is_permanent_quota_structured(value: &serde_json::Value) -> bool {
    value
        .get("error")
        .and_then(|err| err.get("code"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|code| {
            PERMANENT_QUOTA_CODES
                .iter()
                .any(|canonical| code.contains(canonical))
        })
}

/// Detects a PERMANENT quota exhaustion 429 body (OpenAI `insufficient_quota`,
/// exhausted usage/billing quota, or provider-equivalent permanent quota).
///
/// This is distinct from a transient rate-limit 429: a permanent quota
/// condition cannot be fixed by retrying the same candidate, so the caller
/// should skip the ordinary 429 retry budget and advance to the next
/// fleet_router candidate immediately. Reuses the overflow-body shape: parses
/// JSON when the body is JSON, falls back to plain-text phrase matching.
pub(crate) fn is_permanent_quota_429(body: &str) -> bool {
    is_overflow_body(
        body,
        is_permanent_quota_structured,
        PERMANENT_QUOTA_MESSAGE_PHRASES,
    )
}
