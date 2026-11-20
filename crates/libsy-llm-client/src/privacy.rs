// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic detection of content that must not leave the deployment.
//!
//! This module exists once so that the two places that need it can never disagree.
//! The runner asks it about a request before a candidate is selected; the client asks it
//! about the exact JSON body it is about to put on the wire. The second check is the one
//! that makes the guarantee structural: a codec that adds a field, a translation layer
//! that injects a system prompt, or a passthrough block replayed verbatim all pass
//! through it, because it sees the assembled body rather than the request.
//!
//! # What it is not
//!
//! Not a classifier and not a judge. Every detector is a fixed byte-shape test, reports a
//! category only, and is run over text. Nothing here calls a model, and nothing here can
//! be satisfied by content asserting its own harmlessness.
//!
//! Binary media is deliberately not treated as text: an inline image, audio clip or video
//! is bytes, and scanning its base64 would produce findings that describe the encoding
//! rather than the payload. A string that is an opaque data blob by shape is skipped for
//! the same reason, while its surrounding metadata (media type, URL) is still scanned.

use serde_json::Value;

/// A category of content that must not be sent to a provider that reuses it.
///
/// Ordered by severity so a refusal list has a stable presentation order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Finding {
    /// A `-----BEGIN … PRIVATE KEY-----` block.
    PrivateKey,
    /// A provider or platform API key in its recognisable shape.
    ApiKey,
    /// An `Authorization: Bearer` style token.
    BearerToken,
    /// A path this deployment keeps credentials in.
    CredentialPath,
    /// A local filesystem path, which identifies the operator's own layout.
    LocalPath,
    /// A `local@domain.tld` address.
    EmailAddress,
    /// Thirteen or more consecutive digits: an account, card or national id shape.
    LongDigitRun,
}

impl Finding {
    /// The stable code, for logs and error bodies.
    pub const fn code(self) -> &'static str {
        match self {
            Self::PrivateKey => "sensitive_content_private_key",
            Self::ApiKey => "sensitive_content_api_key",
            Self::BearerToken => "sensitive_content_bearer_token",
            Self::CredentialPath => "sensitive_content_credential_path",
            Self::LocalPath => "sensitive_content_local_path",
            Self::EmailAddress => "sensitive_content_email_address",
            Self::LongDigitRun => "sensitive_content_long_digit_run",
        }
    }
}

/// Every finding in `text`, in [`Finding`] order and without duplicates.
pub fn scan_text(text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for detector in DETECTORS {
        if (detector.matches)(text) {
            findings.push(detector.finding);
        }
    }
    findings
}

/// Every finding reachable in `value`, including object keys.
///
/// Keys are scanned because a provider serialises them too: a map-shaped tool argument can
/// carry a credential in a key just as easily as in a value.
pub fn scan_json(value: &Value) -> Vec<Finding> {
    let mut findings = Vec::new();
    scan_value(value, &mut findings);
    findings.sort_unstable();
    findings.dedup();
    findings
}

/// Merges `value`'s findings into `findings` without sorting; see [`scan_json`].
pub fn scan_value(value: &Value, findings: &mut Vec<Finding>) {
    match value {
        Value::String(text) => {
            if !is_opaque_blob(text) {
                findings.extend(scan_text(text));
            }
        }
        Value::Array(items) => {
            for item in items {
                scan_value(item, findings);
            }
        }
        Value::Object(fields) => {
            for (key, value) in fields {
                if !is_opaque_blob(key) {
                    findings.extend(scan_text(key));
                }
                scan_value(value, findings);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// Whether `text` is an encoded binary payload rather than something a person wrote.
///
/// Deliberately conservative: a long string with no spaces drawn from the base64 alphabet,
/// or an explicit `data:` URI. Ordinary prose — including a long pasted document — contains
/// spaces, so it is still scanned.
fn is_opaque_blob(text: &str) -> bool {
    if text.starts_with("data:") {
        return true;
    }
    if text.len() < 512 {
        return false;
    }
    text.chars().all(|character| {
        character.is_ascii_alphanumeric()
            || matches!(character, '+' | '/' | '=' | '-' | '_' | '\n' | '\r' | '\t')
    })
}

/// One deterministic detector.
struct Detector {
    finding: Finding,
    matches: fn(&str) -> bool,
}

/// Fixed order so a refusal list is reproducible.
static DETECTORS: &[Detector] = &[
    Detector {
        finding: Finding::PrivateKey,
        matches: has_private_key,
    },
    Detector {
        finding: Finding::ApiKey,
        matches: has_api_key,
    },
    Detector {
        finding: Finding::BearerToken,
        matches: has_bearer_token,
    },
    Detector {
        finding: Finding::CredentialPath,
        matches: has_credential_path,
    },
    Detector {
        finding: Finding::LocalPath,
        matches: has_local_path,
    },
    Detector {
        finding: Finding::EmailAddress,
        matches: has_email_address,
    },
    Detector {
        finding: Finding::LongDigitRun,
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
    use serde_json::json;

    fn codes(value: &Value) -> Vec<&'static str> {
        scan_json(value).into_iter().map(Finding::code).collect()
    }

    /// Ordinary text produces nothing, so the gate is not a blanket refusal.
    #[test]
    fn ordinary_prose_is_clean() {
        assert!(scan_text("summarise this paragraph about scheduling").is_empty());
        assert!(scan_text("").is_empty());
    }

    /// Each detector fires on its own shape.
    #[test]
    fn each_shape_is_detected() {
        assert_eq!(
            scan_text("-----BEGIN RSA PRIVATE KEY-----\nMIIE"),
            vec![Finding::PrivateKey]
        );
        assert_eq!(
            scan_text("AIzaSyD-1234567890abcdefghijklmnopqrstu"),
            vec![Finding::ApiKey]
        );
        assert_eq!(
            scan_text("Authorization: Bearer abcdefghijklmnopqrstuvwxyz"),
            vec![Finding::BearerToken]
        );
        assert_eq!(
            scan_text("/srv/turnstone/credentials/api/x.pat"),
            vec![Finding::CredentialPath]
        );
        assert_eq!(
            scan_text("see /home/operator/notes.txt"),
            vec![Finding::LocalPath]
        );
        assert_eq!(
            scan_text("write to me@example.com"),
            vec![Finding::EmailAddress]
        );
        assert_eq!(
            scan_text("card 4111111111111111"),
            vec![Finding::LongDigitRun]
        );
    }

    /// A credential nested anywhere in a body is found, including in a key.
    #[test]
    fn findings_are_found_deep_inside_a_body() {
        let body = json!({
            "model": "gemini-3.5-flash-lite",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_result", "content": "token sk-abcdefghijklmnopqrstuvwxyz"}
            ]}],
            "metadata": {"/home/operator/x": true}
        });
        let codes = codes(&body);
        assert!(codes.contains(&"sensitive_content_api_key"), "{codes:?}");
        assert!(codes.contains(&"sensitive_content_local_path"), "{codes:?}");
    }

    /// An encoded binary payload is not treated as text, so an ordinary image does not
    /// refuse itself.
    #[test]
    fn encoded_media_is_not_scanned_as_text() {
        let blob = "A".repeat(4096);
        assert!(codes(&json!({"image_url": {"url": blob}})).is_empty());
        assert!(
            codes(&json!({"b64": format!("data:image/png;base64,{}", "B".repeat(600))})).is_empty()
        );
    }

    /// A long pasted document is still scanned: prose contains spaces.
    #[test]
    fn a_long_text_document_is_still_scanned() {
        let document = format!("{} 4111111111111111", "word ".repeat(400));
        assert_eq!(
            codes(&json!({"text": document})),
            vec!["sensitive_content_long_digit_run"]
        );
    }
}
