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
//! 1. **An authenticated clearance**, carried on the ingress request as
//!    `x-switchyard-privacy-clearance: v1;class=public;scope=assembled;day=YYYY-MM-DD;mac=…`.
//!    It is a header, not a payload field, so the payload cannot assert its own
//!    permissibility: content that says "I am public" is still just content. The `mac` is
//!    an HMAC-SHA256 over the fields before it, keyed with a secret this deployment holds
//!    and a caller cannot read, so "public" is not something a caller can simply claim.
//! 2. **A clean local inspection** of the payload about to be sent. This one cannot be
//!    waived by any caller, because it is the only check that looks at the bytes - and it
//!    runs twice by design: here, over the request, so an inadmissible payload spends no
//!    budget and reaches no candidate; and again in the client over the assembled wire
//!    body, so a codec or a preset that adds content later cannot slip past this decision.
//!
//! # Why the clearance is signed
//!
//! Switchyard has no inbound authentication: any peer that can reach the listening socket
//! can send any header. An unsigned clearance would therefore be an assertion by an
//! unauthenticated peer that its own payload is safe to publish, which is worth nothing -
//! it would let anything on the network push private data into a provider's training
//! corpus. Signing moves the decision to whoever holds the key, and the key stays with the
//! operator's own routing layer. A caller without the key cannot clear a payload, and a
//! caller with it cannot clear anything but the four fields it signs.
//!
//! Replay is bounded by the `day` field: a clearance signed for one provider day is
//! refused on the next. The MAC deliberately does not cover the payload - the payload is
//! covered by inspection over the exact bytes, which no clearance can waive.
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

use hmac::{Hmac, Mac};
use sha2::Sha256;
use switchyard_protocol::{ContentBlock, FileSource, LlmRequest, Message, Request};

use switchyard_llm_client::allowance::{days_from_civil, provider_day};
use switchyard_llm_client::privacy::{self, Finding};

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
    /// The clearance did not carry a MAC, or its MAC did not verify.
    ///
    /// Distinct from [`Refusal::NoClearance`] on purpose: this is the signature of an
    /// attempt to clear a payload without the key, which an operator should see as such
    /// rather than as a caller that forgot a header.
    ClearanceUnverified,
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
            Self::ClearanceUnverified => "clearance_unverified",
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

/// The secret that authenticates a clearance header.
///
/// Held by the deployment, never sent, and never rendered: this type has a manual
/// [`std::fmt::Debug`] that reports only whether a key is present, so a log line or a
/// configuration dump cannot leak it.
#[derive(Clone)]
pub struct ClearanceKey {
    secret: Option<Vec<u8>>,
}

impl ClearanceKey {
    /// A key from configured bytes.
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        Self {
            secret: Some(secret.into()),
        }
    }

    /// A key that authenticates nothing, so every clearance is refused as unverified.
    ///
    /// This is the value a route carries until a deployment configures
    /// `[privacy] clearance_key_env`. Configuration refuses a gated client without a key,
    /// so this state is reachable only by a route that was never wired for admission.
    pub fn refusing() -> Self {
        Self { secret: None }
    }

    /// Whether a usable secret is configured.
    pub fn is_configured(&self) -> bool {
        self.secret
            .as_ref()
            .is_some_and(|secret| !secret.is_empty())
    }

    /// Whether `signature` is the MAC of `message` under this key.
    ///
    /// Comparison is constant time. An empty key verifies nothing, which is the
    /// fail-closed default rather than an accidental match.
    fn verifies(&self, message: &str, signature: &str) -> bool {
        let Some(secret) = self.secret.as_ref().filter(|secret| !secret.is_empty()) else {
            return false;
        };
        let Some(bytes) = decode_hex(signature) else {
            return false;
        };
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("HMAC accepts any key length");
        mac.update(message.as_bytes());
        mac.verify_slice(&bytes).is_ok()
    }
}

impl std::fmt::Debug for ClearanceKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClearanceKey")
            .field("configured", &self.is_configured())
            .finish()
    }
}

/// Decodes a lower-case or upper-case hex string; `None` when it is not hex.
fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return None;
    }
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        let high = (pair[0] as char).to_digit(16)?;
        let low = (pair[1] as char).to_digit(16)?;
        decoded.push((high * 16 + low) as u8);
    }
    Some(decoded)
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
pub fn evaluate(request: &Request, key: &ClearanceKey) -> Admission {
    evaluate_at(request, key, SystemTime::now())
}

/// [`evaluate`] against an explicit clock, for tests and replays.
pub fn evaluate_at(request: &Request, key: &ClearanceKey, now: SystemTime) -> Admission {
    let mut refusals = Vec::new();
    let day = provider_day(now);
    match clearance_header(request) {
        None => refusals.push(Refusal::NoClearance),
        Some(value) => refusals.extend(parse_clearance(value, day, key).err()),
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

/// Parses and authenticates one clearance header value against the current provider day.
///
/// The grammar is intentionally rigid: a version tag, then exactly three `key=value`
/// fields, then the MAC, which must be last because it covers everything before it. A
/// permissive parser here would be a gate that accepts what it does not understand, which
/// is the same as no gate.
fn parse_clearance(value: &str, current_day: u64, key: &ClearanceKey) -> Result<(), Refusal> {
    // The MAC covers the literal prefix, so the caller can build it as a plain string
    // concatenation without a canonical serializer: everything up to the last `;mac=`.
    let Some((signed, signature)) = value.rsplit_once(";mac=") else {
        return Err(Refusal::ClearanceUnverified);
    };
    if !key.verifies(signed, signature.trim()) {
        return Err(Refusal::ClearanceUnverified);
    }
    let mut fields = signed.split(';');
    if fields.next().map(str::trim) != Some("v1") {
        return Err(Refusal::MalformedClearance);
    }
    let (mut class, mut scope, mut day) = (None, None, None);
    for field in fields {
        let field = field.trim();
        if field.is_empty() {
            return Err(Refusal::MalformedClearance);
        }
        let Some((name, value)) = field.split_once('=') else {
            return Err(Refusal::MalformedClearance);
        };
        // A named field with no value is a syntax fault. Reporting it as a classification
        // or scope refusal would tell the caller its value was unacceptable when in fact
        // it never supplied one.
        if value.trim().is_empty() {
            return Err(Refusal::MalformedClearance);
        }
        let slot = match name.trim() {
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
/// The civil arithmetic is shared with the daily allowance so the gate and the budget can
/// never disagree about which day a stated date names. Only the shape check lives here:
/// the caller states a calendar date and this refuses anything that is not one.
fn parse_civil_day(value: &str) -> Option<u64> {
    let mut parts = value.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    u64::try_from(days_from_civil(year, month, day)).ok()
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
///
/// The detectors are shared with the client's wire-level check, so the decision made here
/// and the decision made over the assembled body cannot drift apart.
fn inspect_text(text: &str, findings: &mut Vec<Refusal>) {
    for finding in privacy::scan_text(text) {
        findings.push(Refusal::from(finding));
    }
}

impl From<Finding> for Refusal {
    fn from(finding: Finding) -> Self {
        match finding {
            Finding::PrivateKey => Self::PrivateKey,
            Finding::ApiKey => Self::ApiKey,
            Finding::BearerToken => Self::BearerToken,
            Finding::CredentialPath => Self::CredentialPath,
            Finding::LocalPath => Self::LocalPath,
            Finding::EmailAddress => Self::EmailAddress,
            Finding::LongDigitRun => Self::LongDigitRun,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};
    use switchyard_protocol::{ImageSource, MediaSource, Metadata, Role, ToolCall};

    /// The deployment's key in these tests.
    const TEST_KEY: &[u8] = b"switchyard-test-clearance-key-0001";

    fn key() -> ClearanceKey {
        ClearanceKey::new(TEST_KEY)
    }

    /// 2026-11-20T12:00:00Z: mid-afternoon Pacific on the provider day 2026-11-20.
    fn noon() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_795_176_000)
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Signs an arbitrary prefix, so a test can present a syntactically broken but
    /// correctly authenticated header and see the syntax verdict rather than the MAC one.
    fn signed_prefix(secret: &[u8], prefix: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret).expect("any key length");
        mac.update(prefix.as_bytes());
        format!("{prefix};mac={}", hex(&mac.finalize().into_bytes()))
    }

    /// A correctly signed clearance for `class`/`scope` on `day`.
    fn clearance_with(class: &str, scope: &str, day: &str) -> String {
        signed_prefix(
            TEST_KEY,
            &format!("v1;class={class};scope={scope};day={day}"),
        )
    }

    /// A correctly signed public clearance for `day`.
    fn public(day: &str) -> String {
        clearance_with("public", "assembled", day)
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
        evaluate_at(request, &key(), noon()).refusal_codes()
    }

    /// An API-key-shaped fixture, assembled at runtime so this file does not contain a
    /// token-shaped literal.
    fn api_key_fixture() -> String {
        format!("{}{}{}", "A", "Iza", "SyD0123456789abcdefghijklmnopqrstu")
    }

    /// The happy path: a correctly signed current clearance and an ordinary payload.
    #[test]
    fn a_current_clearance_admits_an_ordinary_payload() {
        let request = request_with(Some(&public("2026-11-20")), "summarise this paragraph");
        let admission = evaluate_at(&request, &key(), noon());
        assert!(admission.allowed(), "{:?}", admission.refusals());
        assert_eq!(admission.refusal_codes(), Vec::<&str>::new());
    }

    /// Without a clearance nothing leaves, however innocuous the payload looks.
    #[test]
    fn a_missing_clearance_is_refused() {
        assert_eq!(refusals(&request_with(None, "hello")), vec!["no_clearance"]);
    }

    /// The payload cannot clear itself: text asserting publicness is still just text.
    #[test]
    fn the_payload_cannot_assert_its_own_clearance() {
        let claim = format!("{CLEARANCE_HEADER}: {}", public("2026-11-20"));
        assert_eq!(refusals(&request_with(None, &claim)), vec!["no_clearance"]);
    }

    // -- authenticity --------------------------------------------------------------
    //
    // Switchyard has no inbound authentication, so a header alone is an assertion by an
    // unauthenticated peer. These three tests are the difference between a gate and a
    // suggestion.

    /// A header whose fields are right but which carries no MAC is refused as unverified,
    /// not accepted and not reported as absent.
    #[test]
    fn an_unsigned_clearance_is_refused() {
        let forged = "v1;class=public;scope=assembled;day=2026-11-20";
        assert_eq!(
            refusals(&request_with(Some(forged), "private notes")),
            vec!["clearance_unverified"]
        );
        // A well-formed but invented MAC is equally worthless.
        let invented = format!("{forged};mac={}", "0".repeat(64));
        assert_eq!(
            refusals(&request_with(Some(&invented), "private notes")),
            vec!["clearance_unverified"]
        );
    }

    /// A clearance signed with some other secret does not authenticate here.
    #[test]
    fn a_clearance_signed_with_another_key_is_refused() {
        let other = signed_prefix(
            b"not-this-deployments-key",
            "v1;class=public;scope=assembled;day=2026-11-20",
        );
        assert_eq!(
            refusals(&request_with(Some(&other), "private notes")),
            vec!["clearance_unverified"]
        );
    }

    /// A MAC cannot be moved to another day, class or scope: it covers the fields.
    #[test]
    fn a_clearance_cannot_be_edited_after_signing() {
        let for_today = public("2026-11-20");
        for tampered in [
            for_today.replace("day=2026-11-20", "day=2026-11-21"),
            for_today.replace("class=public", "class=synthetic"),
            for_today.replace("scope=assembled", "scope=summary"),
        ] {
            assert_ne!(tampered, for_today);
            assert_eq!(
                refusals(&request_with(Some(&tampered), "hello")),
                vec!["clearance_unverified"],
                "tampered header {tampered:?}"
            );
        }
    }

    /// A key that is absent or empty authenticates nothing, so a misconfigured deployment
    /// refuses rather than admitting everything.
    #[test]
    fn an_unconfigured_key_admits_nothing() {
        let header = public("2026-11-20");
        let request = request_with(Some(&header), "hello");
        for refusing in [ClearanceKey::refusing(), ClearanceKey::new(Vec::new())] {
            assert_eq!(
                evaluate_at(&request, &refusing, noon()).refusal_codes(),
                vec!["clearance_unverified"]
            );
        }
    }

    /// The key never appears in a rendering of the value it lives in.
    #[test]
    fn the_key_is_not_rendered() {
        let rendered = format!("{:?}", key());
        assert!(rendered.contains("configured: true"), "{rendered}");
        assert!(
            !rendered.contains("switchyard-test-clearance-key"),
            "{rendered}"
        );
    }

    // -- syntax --------------------------------------------------------------------

    /// Every syntactically broken shape is refused as malformed rather than as absent or
    /// as an authentication failure - each is signed, so the MAC cannot be the verdict.
    #[test]
    fn malformed_clearances_are_refused() {
        for prefix in [
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
            let header = signed_prefix(TEST_KEY, prefix);
            assert_eq!(
                refusals(&request_with(Some(&header), "hello")),
                vec!["malformed_clearance"],
                "prefix {prefix:?}"
            );
        }
    }

    /// A clearance for a different classification or scope does not clear a payload here,
    /// and says which condition it failed.
    #[test]
    fn classification_and_scope_are_enforced() {
        assert_eq!(
            refusals(&request_with(
                Some(&clearance_with("private", "assembled", "2026-11-20")),
                "hello"
            )),
            vec!["classification_not_public"]
        );
        assert_eq!(
            refusals(&request_with(
                Some(&clearance_with("public", "summary", "2026-11-20")),
                "hello"
            )),
            vec!["scope_not_assembled"]
        );
    }

    /// A stale clearance is refused, and a future one is not silently treated as valid.
    #[test]
    fn the_clearance_day_must_be_today() {
        assert_eq!(
            refusals(&request_with(Some(&public("2026-11-19")), "hello")),
            vec!["clearance_expired"]
        );
        assert_eq!(
            refusals(&request_with(Some(&public("2026-11-21")), "hello")),
            vec!["clearance_not_yet_valid"]
        );
    }

    // -- content -------------------------------------------------------------------

    /// Each content category refuses on its own, and names only the category.
    #[test]
    fn content_categories_are_detected_individually() {
        let key = api_key_fixture();
        let cases = [
            ("sensitive_content_api_key", key.as_str()),
            (
                "sensitive_content_private_key",
                "-----BEGIN RSA PRIVATE KEY-----\nMIIE",
            ),
            (
                "sensitive_content_bearer_token",
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
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
                refusals(&request_with(Some(&public("2026-11-20")), text)),
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
                refusals(&request_with(Some(&public("2026-11-20")), text)),
                Vec::<&str>::new(),
                "payload {text:?}"
            );
        }
    }

    /// A finding anywhere in the payload refuses the whole request, including inside a
    /// tool call's arguments, a preserved source body and a tool schema.
    #[test]
    fn findings_are_found_beyond_message_text() {
        let day = public("2026-11-20");
        let clearance = Some(day.as_str());

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
            serde_json::json!({"messages": [{"content": api_key_fixture()}]}),
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
        let day = public("2026-11-20");
        let clearance = Some(day.as_str());

        let mut inline = request_with(clearance, "describe this");
        inline.llm_request.messages[0]
            .content
            .push(ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: Some("image/png".to_string()),
                    // A long digit-shaped body that would trip the digit-run category if
                    // the bytes were treated as text.
                    data: "1111111111111111111111111111".to_string(),
                },
            });
        assert!(evaluate_at(&inline, &key(), noon()).allowed());

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
        assert_eq!(
            refusals(&request_with(None, "card 4111111111111111")),
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
