// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pre-egress admission for targets that may only receive cleared payloads.
//!
//! A provider whose free tier uses submitted content to improve its products makes every
//! request an egress event. A deployment that routes to such a provider has to decide,
//! before the bytes leave, whether *this* payload may go — and the decision belongs at the
//! dispatch filter rather than in a caller, because a caller that forgets (or is a
//! different program entirely) would otherwise be the whole control.
//!
//! # What the gate is
//!
//! Two independent conditions, both required:
//!
//! 1. **An explicit clearance**, carried on the ingress request as
//!    `x-switchyard-privacy-clearance: v1;class=public;scope=assembled;day=YYYY-MM-DD`.
//!    It is a header, not a payload field, so the payload cannot assert its own
//!    permissibility: content that says "I am public" is still just content.
//! 2. **A clean local inspection** of the payload about to be sent. This one cannot be
//!    waived by any caller, because it is the only check that looks at the bytes.
//!
//! # What it is not
//!
//! Not a content classifier and not a judge. The inspection is deterministic and reports
//! categories only — never the matched value — so a refusal is actionable without the
//! refusal itself leaking what it found. A positive label in the payload cannot promote
//! anything; the payload has no vote.
//!
//! The clearance is per request on purpose. Re-issuing it is the caller's job after any
//! tool result, and a request that carries none simply does not reach a gated target: the
//! route falls through to a leg that may receive it.

use std::time::SystemTime;

use switchyard_protocol::{ContentBlock, FileSource, LlmRequest, Message, Request};

use switchyard_llm_client::allowance::provider_day;

/// The single header a caller uses to clear one request for a public provider.
pub const CLEARANCE_HEADER: &str = "x-switchyard-privacy-clearance";

/// The provider-day arithmetic is shared with the daily request allowance, so the gate
/// and the budget can never disagree about which day "today" is.
pub use switchyard_llm_client::allowance::provider_day as clearance_provider_day;

/// Why a payload was refused.
///
/// A closed vocabulary, because these strings are the operator's and the caller's only
/// description of what went wrong. Every variant names a distinct corrective action, and
/// none of them carries the matched value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Refusal {
    /// The request carried no clearance for a gated target.
    NoClearance,
    /// The clearance header did not have the documented shape.
    MalformedClearance,
    /// The clearance was issued for a provider day that has already passed.
    ClearanceExpired,
    /// The clearance was issued for a provider day that has not started.
    ClearanceNotYetValid,
    /// The clearance's classification is not one this deployment may egress.
    ClassificationNotPublic,
    /// The clearance covers some other scope than the assembled payload.
    ScopeNotAssembled,
    /// The payload contains something shaped like a provider API key.
    ApiKey,
    /// The payload contains something shaped like a private key block.
    PrivateKey,
    /// The payload contains an authorization token.
    BearerToken,
    /// The payload names a credential store path.
    CredentialPath,
    /// The payload contains an email address.
    EmailAddress,
    /// The payload contains a long unbroken digit run (an account or card number shape).
    LongDigitRun,
    /// The payload names a local filesystem path.
    LocalPath,
}

impl Refusal {
    /// The stable wire code, for logs, error bodies and documentation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoClearance => "no_clearance",
            Self::MalformedClearance => "malformed_clearance",
            Self::ClearanceExpired => "clearance_expired",
            Self::ClearanceNotYetValid => "clearance_not_yet_valid",
            Self::ClassificationNotPublic => "classification_not_public",
            Self::ScopeNotAssembled => "scope_not_assembled",
            Self::ApiKey => "sensitive_content_api_key",
            Self::PrivateKey => "sensitive_content_private_key",
            Self::BearerToken => "sensitive_content_bearer_token",
            Self::CredentialPath => "sensitive_content_credential_path",
            Self::EmailAddress => "sensitive_content_email_address",
            Self::LongDigitRun => "sensitive_content_long_digit_run",
            Self::LocalPath => "sensitive_content_local_path",
        }
    }

    /// Whether this refusal is about the payload rather than the clearance.
    ///
    /// The distinction is what an operator acts on: a clearance problem is fixed by the
    /// caller re-asserting, while a content finding needs the payload changed.
    pub const fn is_content_finding(self) -> bool {
        matches!(
            self,
            Self::ApiKey
                | Self::PrivateKey
                | Self::BearerToken
                | Self::CredentialPath
                | Self::EmailAddress
                | Self::LongDigitRun
                | Self::LocalPath
        )
    }
}

/// The outcome of an admission decision.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Admission {
    refusals: Vec<Refusal>,
}

impl Admission {
    /// Whether the payload may be sent to a gated target.
    pub fn allowed(&self) -> bool {
        self.refusals.is_empty()
    }

    /// Why it was refused, in a stable order, without duplicates.
    pub fn refusals(&self) -> &[Refusal] {
        &self.refusals
    }

    /// The refusal codes as they appear on the wire.
    pub fn refusal_codes(&self) -> Vec<&'static str> {
        self.refusals
            .iter()
            .map(|refusal| refusal.as_str())
            .collect()
    }
}

/// Decides whether `request` may be sent to a target that requires a cleared payload.
pub fn evaluate(request: &Request) -> Admission {
    evaluate_at(request, SystemTime::now())
}

/// [`evaluate`] against an explicit clock, for tests and replays.
pub fn evaluate_at(request: &Request, now: SystemTime) -> Admission {
    let mut refusals = Vec::new();
    let day = provider_day(now);
    match clearance_header(request) {
        None => refusals.push(Refusal::NoClearance),
        Some(value) => refusals.extend(parse_clearance(value, day).err()),
    }
    for finding in inspect(&request.llm_request) {
        refusals.push(finding);
    }
    refusals.sort_unstable();
    refusals.dedup();
    Admission { refusals }
}

/// The clearance header's value, when the request carries a readable one.
///
/// A header that is not valid text is reported as malformed rather than as absent: the
/// distinction tells the caller its header arrived and was unusable, which is the
/// difference between "you forgot" and "you got the syntax wrong".
fn clearance_header(request: &Request) -> Option<&str> {
    let headers = request.metadata.as_ref()?.http_headers.as_ref()?;
    let value = headers.get(CLEARANCE_HEADER)?;
    // `to_str` fails on non-ASCII bytes; render those as a value that cannot parse so the
    // caller still gets `malformed_clearance`.
    Some(value.to_str().unwrap_or("\u{0}"))
}

/// Parses one clearance header value against the current provider day.
///
/// The grammar is intentionally rigid: a version tag, then exactly three `key=value`
/// fields, no unknowns. A permissive parser here would be a gate that accepts what it does
/// not understand, which is the same as no gate.
fn parse_clearance(value: &str, current_day: u64) -> Result<(), Refusal> {
    let mut fields = value.split(';');
    if fields.next().map(str::trim) != Some("v1") {
        return Err(Refusal::MalformedClearance);
    }
    let (mut class, mut scope, mut day) = (None, None, None);
    for field in fields {
        let field = field.trim();
        if field.is_empty() {
            return Err(Refusal::MalformedClearance);
        }
        let Some((key, value)) = field.split_once('=') else {
            return Err(Refusal::MalformedClearance);
        };
        // A named field with no value is a syntax fault. Reporting it as a classification
        // or scope refusal would tell the caller its value was unacceptable when in fact
        // it never supplied one.
        if value.trim().is_empty() {
            return Err(Refusal::MalformedClearance);
        }
        let slot = match key.trim() {
            "class" => &mut class,
            "scope" => &mut scope,
            "day" => &mut day,
            // An unrecognised field is refused rather than ignored: a caller that meant
            // something by it would otherwise believe it had been honoured.
            _ => return Err(Refusal::MalformedClearance),
        };
        if slot.replace(value.trim()).is_some() {
            // A repeated field makes the meaning ambiguous; do not pick one.
            return Err(Refusal::MalformedClearance);
        }
    }
    let (Some(class), Some(scope), Some(day)) = (class, scope, day) else {
        return Err(Refusal::MalformedClearance);
    };
    match class {
        "public" | "synthetic" => {}
        _ => return Err(Refusal::ClassificationNotPublic),
    }
    if scope != "assembled" {
        return Err(Refusal::ScopeNotAssembled);
    }
    let day = parse_civil_day(day).ok_or(Refusal::MalformedClearance)?;
    match day.cmp(&current_day) {
        std::cmp::Ordering::Less => Err(Refusal::ClearanceExpired),
        std::cmp::Ordering::Greater => Err(Refusal::ClearanceNotYetValid),
        std::cmp::Ordering::Equal => Ok(()),
    }
}

/// Days since 1970-01-01 for a `YYYY-MM-DD` date, or `None` when it is not a date.
///
/// Howard Hinnant's civil-from-days algorithm, inverted. Written out rather than pulled
/// from a dependency so the gate has no date library of its own to disagree with the
/// allowance's day arithmetic.
fn parse_civil_day(value: &str) -> Option<u64> {
    let mut parts = value.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    u64::try_from(days).ok()
}

/// Every content finding in the payload about to be sent.
///
/// The walk covers the fields a codec can put on the wire: instruction and message text,
/// tool call arguments, tool results, file payloads, tool schemas, provider passthrough
/// blocks, and the preserved source bodies a same-format codec may replay instead of the
/// normalized form. Binary media payloads (inline image, audio and video bytes) are not
/// scanned as text: they are bytes, and scanning their encoding would generate findings
/// that say nothing about the payload. Their URLs are scanned, because a URL is text.
fn inspect(request: &LlmRequest) -> Vec<Refusal> {
    let mut findings = Vec::new();
    for block in request.instructions.iter().flat_map(|block| &block.content) {
        inspect_block(block, &mut findings);
    }
    for message in &request.messages {
        inspect_message(message, &mut findings);
    }
    for tool in &request.tools {
        if let Some(description) = &tool.description {
            inspect_text(description, &mut findings);
        }
        // Schemas routinely carry descriptions and enum values, which are text the model
        // will read.
        inspect_json(&tool.parameters, &mut findings);
    }
    for body in request.preservation.requests.values() {
        inspect_json(body, &mut findings);
    }
    findings
}

/// Whether `text` contains any category the gate refuses.
fn inspect_text(text: &str, findings: &mut Vec<Refusal>) {
    for category in CATEGORIES {
        if (category.matches)(text) {
            findings.push(category.refusal);
        }
    }
}

/// Inspects every string inside a JSON value.
fn inspect_json(value: &serde_json::Value, findings: &mut Vec<Refusal>) {
    match value {
        serde_json::Value::String(text) => inspect_text(text, findings),
        serde_json::Value::Array(items) => {
            for item in items {
                inspect_json(item, findings);
            }
        }
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                inspect_text(key, findings);
                inspect_json(value, findings);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn inspect_message(message: &Message, findings: &mut Vec<Refusal>) {
    for block in &message.content {
        inspect_block(block, findings);
    }
}

fn inspect_block(block: &ContentBlock, findings: &mut Vec<Refusal>) {
    match block {
        ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
            inspect_text(text, findings)
        }
        // Reasoning text is returned to the provider on continuation, so it leaves too.
        ContentBlock::Reasoning { text, details, .. } => {
            inspect_text(text, findings);
            for detail in details {
                inspect_json(detail, findings);
            }
        }
        ContentBlock::Image { source } => match source {
            switchyard_protocol::ImageSource::Url { url, .. } => inspect_text(url, findings),
            switchyard_protocol::ImageSource::Base64 { media_type, .. } => {
                if let Some(media_type) = media_type {
                    inspect_text(media_type, findings);
                }
            }
            switchyard_protocol::ImageSource::Raw(raw) => inspect_json(raw, findings),
        },
        ContentBlock::Audio { source } | ContentBlock::Video { source } => match source {
            switchyard_protocol::MediaSource::Url { url, media_type } => {
                inspect_text(url, findings);
                if let Some(media_type) = media_type {
                    inspect_text(media_type, findings);
                }
            }
            switchyard_protocol::MediaSource::Base64 { media_type, .. } => {
                if let Some(media_type) = media_type {
                    inspect_text(media_type, findings);
                }
            }
            switchyard_protocol::MediaSource::Raw(raw) => inspect_json(raw, findings),
        },
        ContentBlock::File { source } => match source {
            FileSource::FileId(id) => inspect_text(id, findings),
            // File data can be a pasted credential document, which is exactly the case a
            // byte-shape gate exists for, so it is scanned as text.
            FileSource::FileData { data, filename } => {
                inspect_text(data, findings);
                if let Some(filename) = filename {
                    inspect_text(filename, findings);
                }
            }
            FileSource::Raw(raw) => inspect_json(raw, findings),
        },
        ContentBlock::ToolCall(call) => {
            inspect_text(&call.name, findings);
            inspect_json(&call.arguments, findings);
        }
        ContentBlock::ToolResult(result) => {
            for block in &result.content {
                inspect_block(block, findings);
            }
        }
        ContentBlock::Unknown { raw, .. } => inspect_json(raw, findings),
    }
}

/// One deterministic content category.
struct Category {
    refusal: Refusal,
    matches: fn(&str) -> bool,
}

/// Fixed order so a refusal list is reproducible.
static CATEGORIES: &[Category] = &[
    Category {
        refusal: Refusal::PrivateKey,
        matches: has_private_key,
    },
    Category {
        refusal: Refusal::ApiKey,
        matches: has_api_key,
    },
    Category {
        refusal: Refusal::BearerToken,
        matches: has_bearer_token,
    },
    Category {
        refusal: Refusal::CredentialPath,
        matches: has_credential_path,
    },
    Category {
        refusal: Refusal::LocalPath,
        matches: has_local_path,
    },
    Category {
        refusal: Refusal::EmailAddress,
        matches: has_email_address,
    },
    Category {
        refusal: Refusal::LongDigitRun,
        matches: has_long_digit_run,
    },
];

/// `sk-…`, `AIza…`, `ghp_…`, `github_pat_…`, `xox?-…`, or `AKIA…`.
fn has_api_key(text: &str) -> bool {
    contains_prefixed_run(text, "sk-", 16, is_token_char)
        || contains_prefixed_run(text, "AIza", 30, is_token_char)
        || contains_prefixed_run(text, "ghp_", 20, is_ascii_alnum)
        || contains_prefixed_run(text, "gho_", 20, is_ascii_alnum)
        || contains_prefixed_run(text, "github_pat_", 20, is_ascii_alnum)
        || contains_prefixed_run(text, "AKIA", 16, is_ascii_alnum)
        || text
            .match_indices("xox")
            .any(|(index, _)| xox_token(&text[index..]))
}

/// `xox[baprs]-` followed by a token body.
fn xox_token(text: &str) -> bool {
    let mut characters = text.chars();
    if characters.next() != Some('x')
        || characters.next() != Some('o')
        || characters.next() != Some('x')
    {
        return false;
    }
    let kind = characters.next();
    if !matches!(kind, Some('b' | 'a' | 'p' | 'r' | 's')) || characters.next() != Some('-') {
        return false;
    }
    characters
        .take_while(|character| is_ascii_alnum(*character))
        .count()
        >= 10
}

/// A `-----BEGIN …PRIVATE KEY-----` block.
///
/// The key type sits *between* the two markers (`BEGIN RSA PRIVATE KEY`), so the check is
/// not adjacency: it is an opening marker followed, before anything that would start
/// another block, by a `PRIVATE KEY` tail.
fn has_private_key(text: &str) -> bool {
    let mut rest = text;
    while let Some(index) = rest.find("-----BEGIN ") {
        let after = &rest[index + "-----BEGIN ".len()..];
        // A label is short; searching the whole remainder would let an unrelated `BEGIN`
        // far above a stray `PRIVATE KEY` mention pair up.
        let window = after.get(..after.len().min(64)).unwrap_or(after);
        if let Some(tail) = window.find("PRIVATE KEY-----")
            && !window[..tail].contains("-----")
        {
            return true;
        }
        rest = after;
    }
    false
}

/// An `Authorization: Bearer` style token long enough to be a credential rather than prose.
fn has_bearer_token(text: &str) -> bool {
    contains_prefixed_run(text, "Bearer ", 20, is_token_char)
        || contains_prefixed_run(text, "bearer ", 20, is_token_char)
}

/// Names of paths this deployment keeps credentials in.
fn has_credential_path(text: &str) -> bool {
    const PATHS: &[&str] = &[
        "/srv/turnstone/credentials",
        "/etc/switchyard-secrets",
        "/etc/switchyard-gemini-secrets",
        "/.ssh/",
        "/root/.ssh",
    ];
    PATHS.iter().any(|path| text.contains(path))
}

/// Names of local filesystem roots, which identify the operator's own host layout.
fn has_local_path(text: &str) -> bool {
    const ROOTS: &[&str] = &[
        "/home/",
        "/Users/",
        "/mnt/turnstone-build-ssd",
        "/var/lib/turnstone",
        "/opt/turnstone-",
    ];
    ROOTS.iter().any(|root| text.contains(root))
}

/// A `local@domain.tld` address.
fn has_email_address(text: &str) -> bool {
    let bytes = text.as_bytes();
    (0..bytes.len()).any(|index| {
        if bytes[index] != b'@' {
            return false;
        }
        // Walk back over the local part, then forward over the domain.
        let local = bytes[..index]
            .iter()
            .rev()
            .take_while(|byte| is_email_local(**byte))
            .count();
        if local < 2 {
            return false;
        }
        let domain: Vec<u8> = bytes[index + 1..]
            .iter()
            .copied()
            .take_while(|byte| is_email_domain(*byte))
            .collect();
        let Some(dot) = domain.iter().position(|byte| *byte == b'.') else {
            return false;
        };
        // A label either side of the dot, and a plausible top level.
        dot > 0 && domain.len() > dot + 2 && domain[domain.len() - 1].is_ascii_alphanumeric()
    })
}

const fn is_email_local(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'%' | b'+' | b'-')
}

const fn is_email_domain(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-')
}

/// Thirteen or more consecutive digits: the shape of an account, card, or national id.
fn has_long_digit_run(text: &str) -> bool {
    let mut run = 0_usize;
    for byte in text.bytes() {
        if byte.is_ascii_digit() {
            run += 1;
            if run >= 13 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Whether `text` contains `prefix` followed by at least `minimum` `allowed` characters.
fn contains_prefixed_run(
    text: &str,
    prefix: &str,
    minimum: usize,
    allowed: fn(char) -> bool,
) -> bool {
    let mut rest = text;
    while let Some(index) = rest.find(prefix) {
        let after = &rest[index + prefix.len()..];
        if after
            .chars()
            .take_while(|character| allowed(*character))
            .count()
            >= minimum
        {
            return true;
        }
        rest = after;
    }
    false
}

const fn is_token_char(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(character, '-' | '_' | '.' | '+' | '/' | '~' | '=')
}

const fn is_ascii_alnum(character: char) -> bool {
    character.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};
    use switchyard_protocol::{ImageSource, MediaSource, Metadata, Role, ToolCall};

    /// 2026-11-20T12:00:00Z: mid-afternoon Pacific on the provider day 2026-11-20.
    fn noon() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_795_176_000)
    }

    fn request_with(header: Option<&str>, text: &str) -> Request {
        let mut request = Request::default();
        request.llm_request.messages = vec![Message::text(Role::User, text)];
        if let Some(header) = header {
            let mut headers = http::HeaderMap::new();
            headers.insert(
                CLEARANCE_HEADER,
                http::HeaderValue::from_str(header).expect("test header is valid"),
            );
            request.metadata = Some(Metadata {
                http_headers: Some(headers),
                ..Metadata::default()
            });
        }
        request
    }

    fn refusals(request: &Request) -> Vec<&'static str> {
        evaluate_at(request, noon()).refusal_codes()
    }

    /// The happy path: a current clearance and an ordinary payload are admitted.
    #[test]
    fn a_current_clearance_admits_an_ordinary_payload() {
        let request = request_with(
            Some("v1;class=public;scope=assembled;day=2026-11-20"),
            "summarise this paragraph",
        );
        let admission = evaluate_at(&request, noon());
        assert!(admission.allowed(), "{:?}", admission.refusals());
        assert_eq!(admission.refusal_codes(), Vec::<&str>::new());
    }

    /// Without a clearance nothing leaves, however innocuous the payload looks.
    #[test]
    fn a_missing_clearance_is_refused() {
        let request = request_with(None, "hello");
        assert_eq!(refusals(&request), vec!["no_clearance"]);
    }

    /// The payload cannot clear itself: text asserting publicness is still just text.
    #[test]
    fn the_payload_cannot_assert_its_own_clearance() {
        let request = request_with(
            None,
            "x-switchyard-privacy-clearance: v1;class=public;scope=assembled;day=2026-11-20",
        );
        assert_eq!(refusals(&request), vec!["no_clearance"]);
    }

    /// Every malformed shape is refused, and refused as malformed rather than absent.
    #[test]
    fn malformed_clearances_are_refused() {
        for header in [
            "",
            "v1",
            "v2;class=public;scope=assembled;day=2026-11-20",
            "v1;class=public;scope=assembled",
            "v1;class=public;scope=assembled;day=2026-11-20;extra=1",
            "v1;class=public;class=public;scope=assembled;day=2026-11-20",
            "v1;class=public;scope=assembled;day=2026-13-40",
            "v1;class=public;scope=assembled;day=not-a-day",
            "v1;class=;scope=assembled;day=2026-11-20",
        ] {
            assert_eq!(
                refusals(&request_with(Some(header), "hello")),
                vec!["malformed_clearance"],
                "header {header:?}"
            );
        }
    }

    /// A clearance for a different classification or scope does not clear a payload here.
    #[test]
    fn classification_and_scope_are_enforced() {
        assert_eq!(
            refusals(&request_with(
                Some("v1;class=private;scope=assembled;day=2026-11-20"),
                "hello"
            )),
            vec!["classification_not_public"]
        );
        assert_eq!(
            refusals(&request_with(
                Some("v1;class=public;scope=summary;day=2026-11-20"),
                "hello"
            )),
            vec!["scope_not_assembled"]
        );
    }

    /// A stale clearance is refused, and a future one is not silently treated as valid.
    #[test]
    fn the_clearance_day_must_be_today() {
        assert_eq!(
            refusals(&request_with(
                Some("v1;class=public;scope=assembled;day=2026-11-19"),
                "hello"
            )),
            vec!["clearance_expired"]
        );
        assert_eq!(
            refusals(&request_with(
                Some("v1;class=public;scope=assembled;day=2026-11-21"),
                "hello"
            )),
            vec!["clearance_not_yet_valid"]
        );
    }

    /// Each content category refuses on its own, and names only the category.
    #[test]
    fn content_categories_are_detected_individually() {
        let cases = [
            ("sensitive_content_api_key", "key sk-abcdef0123456789abcdef"),
            (
                "sensitive_content_api_key",
                "google AIzaSyA1234567890abcdefghijklmnopqrstuv",
            ),
            (
                "sensitive_content_private_key",
                "-----BEGIN RSA PRIVATE KEY-----\nMIIE",
            ),
            (
                "sensitive_content_bearer_token",
                "Authorization: Bearer abcdef0123456789abcdefghij",
            ),
            (
                "sensitive_content_credential_path",
                "read /srv/turnstone/credentials/github/classic.pat",
            ),
            (
                "sensitive_content_email_address",
                "mail vincent@example.com",
            ),
            ("sensitive_content_long_digit_run", "card 4111111111111111"),
            ("sensitive_content_local_path", "see /home/vincent/notes"),
        ];
        for (expected, text) in cases {
            assert_eq!(
                refusals(&request_with(
                    Some("v1;class=public;scope=assembled;day=2026-11-20"),
                    text
                )),
                vec![expected],
                "payload {text:?}"
            );
        }
    }

    /// Ordinary text does not trip a category: a gate that refuses everything is not a gate.
    #[test]
    fn ordinary_text_passes_inspection() {
        for text in [
            "the quick brown fox",
            "email me at the office",       // no @domain.tld
            "version 1.2.3 build 20261120", // digits, but no long run
            "task-sketch-1234",             // `sk-` but too short
            "use the Bearer scheme",        // names the scheme, no token
            "path/to/file.txt",             // relative, not a local root
            "2026-11-20T12:00:00Z",         // a timestamp is not a 13-digit run
        ] {
            assert_eq!(
                refusals(&request_with(
                    Some("v1;class=public;scope=assembled;day=2026-11-20"),
                    text
                )),
                Vec::<&str>::new(),
                "payload {text:?}"
            );
        }
    }

    /// A finding anywhere in the payload refuses the whole request, including inside a
    /// tool call's arguments and a preserved source body.
    #[test]
    fn findings_are_found_beyond_message_text() {
        let clearance = Some("v1;class=public;scope=assembled;day=2026-11-20");

        let mut in_arguments = request_with(clearance, "call the tool");
        in_arguments.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "/srv/turnstone/credentials/api/k.pat"}),
            })],
        });
        assert_eq!(
            refusals(&in_arguments),
            vec!["sensitive_content_credential_path"]
        );

        let mut in_preservation = request_with(clearance, "hello");
        in_preservation.llm_request.preservation.requests.insert(
            "openai_chat".into(),
            serde_json::json!({"messages": [{"content": "key sk-abcdef0123456789abcdef"}]}),
        );
        assert_eq!(
            refusals(&in_preservation),
            vec!["sensitive_content_api_key"]
        );

        let mut in_tool_schema = request_with(clearance, "hello");
        in_tool_schema
            .llm_request
            .tools
            .push(switchyard_protocol::ToolDefinition {
                name: "lookup".to_string(),
                description: Some("calls vincent@example.com".to_string()),
                parameters: serde_json::json!({}),
                strict: None,
            });
        assert_eq!(
            refusals(&in_tool_schema),
            vec!["sensitive_content_email_address"]
        );
    }

    /// An inline image's bytes are not scanned as text, but its remote URL is.
    #[test]
    fn media_carriers_are_scanned_by_their_text_not_their_bytes() {
        let clearance = Some("v1;class=public;scope=assembled;day=2026-11-20");
        let mut inline = request_with(clearance, "describe this");
        inline.llm_request.messages[0]
            .content
            .push(ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: Some("image/png".to_string()),
                    // A long base64 body that would trip the digit-run category if scanned.
                    data: "1111111111111111111111111111".to_string(),
                },
            });
        assert!(evaluate_at(&inline, noon()).allowed());

        let mut remote = request_with(clearance, "describe this");
        remote.llm_request.messages[0]
            .content
            .push(ContentBlock::Video {
                source: MediaSource::Url {
                    url: "https://vincent@example.com/clip.mp4".to_string(),
                    media_type: None,
                },
            });
        assert_eq!(refusals(&remote), vec!["sensitive_content_email_address"]);
    }

    /// Several independent problems are all reported, so one round trip fixes all of them.
    #[test]
    fn every_reason_is_reported_together() {
        let request = request_with(None, "card 4111111111111111");
        assert_eq!(
            refusals(&request),
            vec!["no_clearance", "sensitive_content_long_digit_run"]
        );
    }

    /// The civil-day parser agrees with the allowance's day arithmetic at a known instant.
    #[test]
    fn the_civil_day_matches_the_provider_day() {
        // 2026-11-20T12:00:00Z is Pacific 2026-11-20.
        assert_eq!(parse_civil_day("2026-11-20"), Some(provider_day(noon())));
        assert_eq!(parse_civil_day("1970-01-01"), Some(0));
        assert_eq!(parse_civil_day("2026-11-20-01"), None);
        assert_eq!(
            parse_civil_day("2026-2-3"),
            Some(parse_civil_day("2026-02-03").unwrap())
        );
    }
}
