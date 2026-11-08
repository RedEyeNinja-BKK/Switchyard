// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where a provider's own request id can be found, and what counts as one.
//!
//! WHY THIS IS SHARED
//!
//! Two layers need this answer and they must not disagree:
//!
//! * the HTTP client, which is the only place a response header still exists and
//!   which therefore has to decide whether the error body already names a request;
//! * the server, which reads a failed request's body and reports the id.
//!
//! When each carried its own list of locations, the writer checked fewer than the
//! reader: a body whose only id was in the provider's message prose was treated as
//! unnamed, so a header id was injected — and the reader, preferring the structured
//! field, then logged the injected header id instead of the provider's. One
//! definition, in the crate both depend on, removes that class of divergence.
//!
//! WHAT A PROVIDER ID IS
//!
//! Correlation only works if the value is exactly the string the provider logged.
//! A value that has been trimmed of awkward characters, truncated, or otherwise
//! "repaired" is a different string that resolves to no request, and it is worse
//! than reporting nothing because it looks like evidence. So every entry point here
//! validates and refuses; none of them repairs.

use serde_json::Value;

/// The longest identifier any provider is known to issue. A longer value is not a
/// truncated identifier, it is something else.
pub const MAX_REQUEST_ID_LEN: usize = 128;

/// Structured field names a provider may use for its request id.
const REQUEST_ID_FIELDS: &[&str] = &["request_id", "requestId", "requestID"];

/// Accepts a header value only if it is exactly a provider identifier.
///
/// Surrounding optional whitespace is trimmed because that is transport framing, not
/// identity. Every remaining byte must be an identifier character, and the value must
/// fit in [`MAX_REQUEST_ID_LEN`]. Anything else — an embedded quote, a backslash, a
/// comma-separated list, interior whitespace, control characters, an empty value, an
/// over-long value — yields `None` rather than a rewritten identifier.
pub fn validated_request_id(raw: &str) -> Option<String> {
    let trimmed = raw.trim_matches(|c: char| c == ' ' || c == '\t');
    let identifier_shaped = !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'));
    // Length is compared in bytes, which equals characters here because every byte
    // above is ASCII.
    (identifier_shaped && trimmed.len() <= MAX_REQUEST_ID_LEN).then(|| trimmed.to_string())
}

/// Whether the document already carries a provider-authored request id.
///
/// Checks every location [`provider_request_id`] reads, so the layer that decides
/// whether to supplement a body and the layer that reports from it cannot disagree
/// about what "already named" means.
pub fn names_a_request(document: &Value) -> bool {
    provider_request_id(document).is_some()
}

/// The provider's request id, from every location providers are known to use.
///
/// Priority is provider-authored evidence, most structured first:
///
/// 1. a structured field nested under `error` (the OpenAI shape);
/// 2. a structured field at the top level (the Anthropic shape);
/// 3. the provider's own message prose, e.g. DeepSeek's `(request_id: 5502af93-…)`.
///
/// A bare top-level `id` is deliberately never read: that is the response *object*
/// id (`resp_…`, `chatcmpl_…`), and reporting it would put a value beside the
/// provider's logs that does not exist there.
///
/// A document that is a JSON string is a body that was not JSON at all: the server
/// preserves such a body verbatim and hands it over as a string, so its prose is the
/// only place an id can be. That case is handled here rather than at each call site,
/// because a call site that forgot it would drop the id silently — which is exactly
/// what happened when this rule was consolidated out of the server.
pub fn provider_request_id(document: &Value) -> Option<String> {
    if let Some(text) = document.as_str() {
        return request_id_in_text(text);
    }
    if let Some(id) = document.get("error").and_then(structured_request_id) {
        return Some(id);
    }
    if let Some(id) = structured_request_id(document) {
        return Some(id);
    }
    document
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .and_then(request_id_in_text)
}

/// A validated structured field on one object, if it carries one.
///
/// Validation happens *inside* the search. Selecting the first string and validating
/// afterwards would let an empty, malformed or over-long first field mask a usable
/// second one, so the body would look unnamed, a header id would be written over a
/// real provider id, and the reader would report the injected value. A non-string
/// value is skipped for the same reason: it is not an id, so it cannot end the search.
fn structured_request_id(object: &Value) -> Option<String> {
    REQUEST_ID_FIELDS.iter().find_map(|field| {
        object
            .get(*field)
            .and_then(Value::as_str)
            .and_then(validated_request_id)
    })
}

/// The shortest prose token that can be a request id rather than a word.
///
/// Prose carries no delimiter that separates an identifier from the sentence around
/// it, so shape alone is not enough: "the request id header is missing" would yield
/// `header`. Two guards, both of which a real identifier from every provider on
/// record passes:
///
/// * at least [`MIN_PROSE_ID_LEN`] characters - an eight-character English word is
///   plausible, a shorter one is not an identifier;
/// * at least one ASCII digit - provider ids are uuid-, hex- or counter-shaped, and
///   the providers that write them into prose (DeepSeek) always include digits.
///
/// Neither guard applies to a structured field or a header, where the value's
/// position already says what it is.
const MIN_PROSE_ID_LEN: usize = 8;

/// Finds a provider request id written into prose.
///
/// Providers that report it this way give no structured field, so the message text is
/// the only place it exists. The token must look like an identifier and be bounded in
/// length, so ordinary words cannot match; the surrounding message is never returned,
/// copied, or logged.
pub fn request_id_in_text(text: &str) -> Option<String> {
    // Bound the scan: an error message is small, and a huge body is not worth it.
    const SCAN_LIMIT: usize = 64 * 1024;
    // Truncate on a CHARACTER boundary: a body may hold multi-byte UTF-8, and slicing
    // mid-character would panic on the error path.
    let limit = text.len().min(SCAN_LIMIT);
    let end = (0..=limit)
        .rev()
        .find(|offset| text.is_char_boundary(*offset))
        .unwrap_or(0);
    let text = &text[..end];
    let lower = text.to_ascii_lowercase();
    for marker in ["request_id", "request id", "request-id"] {
        let mut from = 0;
        while let Some(offset) = lower[from..].find(marker) {
            let rest = text[from + offset + marker.len()..]
                .trim_start_matches([' ', '\t', ':', '=', '"', '\'', '(', '#']);
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
                .collect();
            let looks_like_an_id =
                token.len() >= MIN_PROSE_ID_LEN && token.chars().any(|c| c.is_ascii_digit());
            if looks_like_an_id && let Some(id) = validated_request_id(&token) {
                return Some(id);
            }
            from += offset + marker.len();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_identifier_shaped_value_is_preserved_exactly() {
        for value in [
            "req_abc123",
            "5502af93-ffbb-46bc-95ec-6402a807b7e8",
            "req-1.example_v2:a",
        ] {
            assert_eq!(validated_request_id(value).as_deref(), Some(value));
        }
        assert_eq!(
            validated_request_id("  req_padded\t ").as_deref(),
            Some("req_padded")
        );
    }

    #[test]
    fn a_malformed_value_is_refused_not_repaired() {
        for value in [
            r#"req_ok","injected":"yes"#,
            r#"req_ok\injected"#,
            "req_a,req_b",
            "req ok",
            "(req_ok)",
            "",
            "   ",
        ] {
            assert_eq!(validated_request_id(value), None, "must refuse {value:?}");
        }
    }

    #[test]
    fn an_over_long_value_is_refused_not_truncated() {
        let at_limit = "a".repeat(MAX_REQUEST_ID_LEN);
        assert_eq!(
            validated_request_id(&at_limit).as_deref(),
            Some(at_limit.as_str())
        );
        for over in ["a".repeat(MAX_REQUEST_ID_LEN + 1), "a".repeat(5_000)] {
            assert_eq!(validated_request_id(&over), None);
        }
    }

    #[test]
    fn a_structured_field_wins_over_prose() {
        let document = json!({
            "error": {
                "message": "failed (request_id: req_from_prose)",
                "request_id": "req_from_field"
            }
        });
        assert_eq!(
            provider_request_id(&document).as_deref(),
            Some("req_from_field")
        );
    }

    #[test]
    fn a_top_level_field_is_read() {
        let document = json!({
            "type": "error",
            "error": {"message": "x"},
            "request_id": "req_top"
        });
        assert_eq!(provider_request_id(&document).as_deref(), Some("req_top"));
    }

    // The case that mattered: prose is the ONLY id, and it must be found so a header
    // cannot be injected over it.
    #[test]
    fn prose_is_found_when_no_structured_field_exists() {
        let document = json!({
            "error": {
                "message": "thinking mode failure (request_id: req_body_123)",
                "type": "upstream_error"
            }
        });
        assert_eq!(
            provider_request_id(&document).as_deref(),
            Some("req_body_123")
        );
        assert!(names_a_request(&document));
    }

    // A bare top-level `id` is a response object id, not a request id.
    #[test]
    fn a_top_level_object_id_is_not_a_request_id() {
        let document = json!({"id": "resp_object_id", "error": {"message": "x"}});
        assert_eq!(provider_request_id(&document), None);
        assert!(!names_a_request(&document));
    }

    // Control: prose mentioning the words without an identifier must not match.
    #[test]
    fn prose_without_an_identifier_does_not_match() {
        for message in [
            "the request id header is missing",
            "request_id was rejected",
            "no request-id supplied",
        ] {
            let document = json!({"error": {"message": message}});
            assert_eq!(
                provider_request_id(&document),
                None,
                "must not invent an id from {message:?}"
            );
        }
    }

    // The second finding in review: the first implementation selected the first
    // *string* and validated it afterwards, so a body whose first field was present
    // but unusable returned None even when a later field held a valid id. The body
    // then looked unnamed, a header id was injected over a real provider id, and the
    // reader reported the injected value.
    #[test]
    fn a_malformed_first_field_does_not_hide_a_valid_later_one() {
        for (label, first) in [
            ("empty", json!("")),
            ("whitespace", json!("   ")),
            ("malformed", json!(r#"req_ok","injected":"yes"#)),
            ("over-long", json!("a".repeat(MAX_REQUEST_ID_LEN + 1))),
            ("non-string number", json!(7)),
            ("null", Value::Null),
        ] {
            let nested = json!({"error": {"request_id": first, "requestId": "req_valid_123"}});
            assert_eq!(
                provider_request_id(&nested).as_deref(),
                Some("req_valid_123"),
                "nested: a {label} first field must not hide the valid second one"
            );
            let top_level = json!({"request_id": first, "requestID": "req_valid_456"});
            assert_eq!(
                provider_request_id(&top_level).as_deref(),
                Some("req_valid_456"),
                "top level: a {label} first field must not hide the valid second one"
            );
        }
    }

    // Control for the above: when NO field carries a usable value, the body is
    // unnamed and nothing is invented.
    #[test]
    fn no_usable_structured_field_leaves_the_body_unnamed() {
        let document = json!({"error": {"request_id": "", "requestId": "  "}});
        assert_eq!(provider_request_id(&document), None);
        assert!(!names_a_request(&document));
    }

    // A provider that answers with a non-JSON body still writes its id into the
    // prose. The server hands such a body to this rule as a JSON string - it could
    // not parse it, so it preserved the bytes - and that case must be scanned. It
    // was, before the rule was consolidated; dropping it lost ids from raw-text
    // provider rejections.
    #[test]
    fn a_plain_text_body_is_scanned_for_an_id() {
        let document = Value::String(
            "upstream rejected (request_id: 5502af93-ffbb-46bc-95ec-6402a807b7e8)".to_string(),
        );
        assert_eq!(
            provider_request_id(&document).as_deref(),
            Some("5502af93-ffbb-46bc-95ec-6402a807b7e8")
        );
        assert!(names_a_request(&document));
    }

    // Control: plain text that mentions the words without carrying an id yields
    // nothing, so no header is suppressed and no word is logged as an id.
    #[test]
    fn plain_text_without_an_identifier_does_not_match() {
        for text in [
            "upstream rejected without a correlation id",
            "the request id header is missing",
            "request_id was rejected",
        ] {
            let document = Value::String(text.to_string());
            assert_eq!(
                provider_request_id(&document),
                None,
                "must not invent an id from {text:?}"
            );
        }
    }

    #[test]
    fn a_multibyte_body_does_not_panic() {
        let padding = "漢".repeat(24_000);
        let document = json!({"error": {"message": format!("{padding} (request_id: 0123abcd-4567-89ef-0123-456789abcdef)")}});
        // The id sits past the scan limit, so it is not found — but nothing panics.
        assert_eq!(provider_request_id(&document), None);

        let document = json!({"error": {"message": format!("failed (request_id: 0123abcd-4567-89ef-0123-456789abcdef) {padding}")}});
        assert_eq!(
            provider_request_id(&document).as_deref(),
            Some("0123abcd-4567-89ef-0123-456789abcdef")
        );
    }
}
