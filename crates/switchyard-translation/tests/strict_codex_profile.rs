// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Row-37 strict-Codex destination-profile tests.
//!
//! A strict ChatGPT-Codex destination (derived from the backend's path
//! boundary) accepts only `output_text`/`refusal` content inside assistant
//! items, while a normal `/v1/responses` destination is standards-compliant.
//! These tests pin BOTH sides of that boundary, so a future change cannot
//! widen the strict encoding to normal endpoints or narrow it for Codex.

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::{
    PreservationPolicy, ResponsesProfile, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn engine() -> TranslationEngine {
    TranslationEngine::default()
}

fn policy_with(profile: ResponsesProfile) -> TranslationPolicy {
    TranslationPolicy {
        responses_profile: profile,
        // The destination profile governs the TRANSLATION path. A same-format
        // decode/encode round-trip is an exact-preservation replay that returns
        // the caller's body verbatim by design (see `exact_preserved_request`),
        // so the encoder is only reached with preservation off -- EXCEPT for the
        // wire-legality fixup that the preserved path still owes the strict
        // endpoint, which is what `preserved_system_roles_become_developer` covers.
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

/// Policy for the preserved (exact-replay) path: preservation is left ON, which
/// is the production default, so a same-format round-trip replays the caller's
/// body rather than re-encoding it.
fn preserved_policy(profile: ResponsesProfile) -> TranslationPolicy {
    TranslationPolicy {
        responses_profile: profile,
        ..TranslationPolicy::default()
    }
}

/// Builds a two-turn conversation whose second turn is produced by the model,
/// so the encoder has assistant history to re-encode.
fn conversation() -> Value {
    json!({
        "model": "codex-model",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "earlier answer"},
            {"type": "message", "role": "user", "content": "again"}
        ]
    })
}

fn assistant_items(output: &Value) -> Vec<Value> {
    output["input"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|item| {
            item.is_object() && item.get("role").and_then(Value::as_str) == Some("assistant")
        })
        .collect()
}

// The strict profile must carry assistant history as an output_text ARRAY: a
// bare string and an input_text block are both rejected upstream.
#[test]
fn strict_codex_encodes_assistant_history_as_output_text() -> TestResult {
    let body = conversation();
    let decoded = engine().decode_request(
        WireFormat::OpenAiResponses,
        &body,
        &policy_with(ResponsesProfile::StrictCodex),
    )?;
    let output = engine()
        .encode_request(
            WireFormat::OpenAiResponses,
            &decoded.request,
            &policy_with(ResponsesProfile::StrictCodex),
        )?
        .body;

    let items = assistant_items(&output);
    assert_eq!(
        items.len(),
        1,
        "expected exactly one assistant item: {output}"
    );
    let content = &items[0]["content"];
    assert!(
        content.is_array(),
        "strict-Codex assistant content must be an array, got {content}"
    );
    assert_eq!(content[0]["type"], json!("output_text"));
    assert_eq!(content[0]["text"], json!("earlier answer"));
    Ok(())
}

// The normal profile is the standards-compliant control and must NOT inherit
// the lossy strict encoding.
#[test]
fn normal_profile_keeps_assistant_history_standards_compliant() -> TestResult {
    let body = conversation();
    let decoded = engine().decode_request(
        WireFormat::OpenAiResponses,
        &body,
        &policy_with(ResponsesProfile::Normal),
    )?;
    let output = engine()
        .encode_request(
            WireFormat::OpenAiResponses,
            &decoded.request,
            &policy_with(ResponsesProfile::Normal),
        )?
        .body;

    let items = assistant_items(&output);
    assert_eq!(
        items.len(),
        1,
        "expected exactly one assistant item: {output}"
    );
    assert_eq!(
        items[0]["content"],
        json!("earlier answer"),
        "normal profile must keep the bare-string assistant content"
    );
    Ok(())
}

// Only assistant items are strict: a non-assistant item must not be rewritten
// into the strict output_text shape. A single text-only user turn collapses to
// the canonical bare-string form (identical in both profiles, and identical to
// production), so the invariant is that the strict profile does not change it.
#[test]
fn strict_codex_leaves_non_assistant_items_input_oriented() -> TestResult {
    // The user turn carries ARRAY content (an image), because a text-only user
    // turn collapses to a bare string and the inner per-part assertion below
    // would never execute -- a green test asserting nothing.
    let body = json!({
        "model": "codex-model",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "earlier answer"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "https://example.invalid/i.png"}
            ]}
        ]
    });
    let strict = policy_with(ResponsesProfile::StrictCodex);
    let decoded = engine().decode_request(WireFormat::OpenAiResponses, &body, &strict)?;
    let output = engine()
        .encode_request(WireFormat::OpenAiResponses, &decoded.request, &strict)?
        .body;

    let mut array_content_non_assistant = 0usize;
    for item in output["input"].as_array().cloned().unwrap_or_default() {
        if item.get("role").and_then(Value::as_str) == Some("assistant") {
            continue;
        }
        if let Some(parts) = item["content"].as_array() {
            array_content_non_assistant += 1;
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
                assert_ne!(
                    kind, "output_text",
                    "strict profile rewrote a non-assistant part as output_text: {output}"
                );
                assert!(
                    kind.starts_with("input_") || kind == "refusal",
                    "non-assistant part is not input-oriented: {kind} in {output}"
                );
            }
        }
    }
    assert!(
        array_content_non_assistant > 0,
        "no non-assistant item had array content, so the per-part assertion never ran: {output}"
    );
    Ok(())
}

// Assistant multimodal content cannot be represented on the strict wire, so
// it degrades to output_text (lossy) rather than emitting a rejected shape.
#[test]
fn strict_codex_degrades_assistant_multimodal_to_output_text() -> TestResult {
    let body = json!({
        "model": "codex-model",
        "input": [
            {"type": "message", "role": "user", "content": "describe"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "here it is"},
                {"type": "output_image", "image_url": "https://example.invalid/i.png"}
            ]},
            {"type": "message", "role": "user", "content": "more"}
        ]
    });
    let policy = policy_with(ResponsesProfile::StrictCodex);
    let decoded = engine().decode_request(WireFormat::OpenAiResponses, &body, &policy)?;

    let result = engine().encode_request(WireFormat::OpenAiResponses, &decoded.request, &policy);

    // Require the Ok arm: a test that accepts Err(_) AND only asserts the
    // ABSENCE of input_image passes even if the decoder silently DROPPED the
    // image block, in which case the degrade branch never ran. The property is
    // that the image SURVIVES as a single output_text part whose text carries
    // the image source -- i.e. lossy degradation, not silent loss.
    let encoded = result.expect("strict profile must degrade, not reject, assistant multimodal");
    let items = assistant_items(&encoded.body);
    assert_eq!(items.len(), 1, "{encoded:?}");
    let parts = items[0]["content"]
        .as_array()
        .cloned()
        .unwrap_or_else(|| panic!("strict assistant content is not an array: {encoded:?}"));
    // The text turn survives, and the image turn is degraded to an output_text
    // part whose text carries the source. The property is that nothing is
    // SILENTLY lost and no input-oriented block type reaches the strict wire.
    assert!(
        parts.len() >= 2,
        "expected the text part plus a degraded image part, got: {parts:?}"
    );
    for part in &parts {
        let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
        assert!(
            kind == "output_text" || kind == "refusal",
            "strict assistant part is not output_text/refusal: {kind} in {parts:?}"
        );
        assert_ne!(
            kind, "input_image",
            "strict-Codex assistant item leaked an input_image part"
        );
    }
    let carried = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .any(|t| t.contains("example.invalid"));
    assert!(
        carried,
        "degraded part dropped the image source instead of carrying it: {parts:?}"
    );
    assert!(
        encoded
            .diagnostics
            .iter()
            .any(|d| d.message.contains("degraded to output_text")),
        "lossy degradation must be reported as a diagnostic: {:?}",
        encoded.diagnostics
    );
    Ok(())
}

// The default policy must be Normal, so every other caller keeps
// standards-compliant behavior without opting in.
#[test]
fn responses_profile_defaults_to_normal() {
    assert_eq!(
        TranslationPolicy::default().responses_profile,
        ResponsesProfile::Normal
    );
    assert_eq!(ResponsesProfile::default(), ResponsesProfile::Normal);
    assert_ne!(ResponsesProfile::Normal, ResponsesProfile::StrictCodex);
}

// A preserved (exact-replay) body reaches a strict Codex endpoint VERBATIM,
// so the one wire-illegal shape it must still be fixed up on that path is a
// `system`-role input item: the Codex endpoint rejects it ("System messages
// are not allowed") and requires `developer`. This runs with preservation ON
// (the production default), so it covers the exact-preserved replay branch
// that the translation-path tests above deliberately bypass.
#[test]
fn preserved_system_roles_become_developer() -> TestResult {
    let body = json!({
        "model": "codex-model",
        "input": [
            {"type": "message", "role": "system", "content": "be terse"},
            {"type": "message", "role": "user", "content": "hi"}
        ]
    });
    for profile in [ResponsesProfile::StrictCodex, ResponsesProfile::Normal] {
        let policy = preserved_policy(profile);
        let decoded = engine().decode_request(WireFormat::OpenAiResponses, &body, &policy)?;
        let output = engine()
            .encode_request(WireFormat::OpenAiResponses, &decoded.request, &policy)?
            .body;
        for item in output["input"].as_array().cloned().unwrap_or_default() {
            assert_ne!(
                item.get("role").and_then(Value::as_str),
                Some("system"),
                "preserved body replayed a wire-illegal system role ({profile:?}): {output}"
            );
        }
        let input_items = output["input"].as_array().cloned().unwrap_or_default();
        let roles: Vec<_> = input_items
            .iter()
            .filter_map(|i| i.get("role").and_then(Value::as_str))
            .collect();
        assert!(
            roles.contains(&"developer"),
            "system role was dropped instead of rewritten to developer ({profile:?}): {roles:?}"
        );
        // Everything else about the preserved body must survive untouched.
        assert_eq!(input_items.len(), 2, "{output}");
    }
    Ok(())
}

// A preserved body must stay byte-identical apart from the role fixup: exact
// preservation is a passthrough guarantee, and a fixup that silently rewrote
// anything else would break callers relying on replay fidelity.
#[test]
fn preserved_body_is_otherwise_untouched() -> TestResult {
    let body = json!({
        "model": "codex-model",
        "instructions": "be terse",
        "temperature": 0.4,
        "input": [
            {"type": "message", "role": "system", "content": "be terse"},
            {"type": "message", "role": "user", "content": "hi"}
        ],
        "metadata": {"trace": "abc"}
    });
    let policy = preserved_policy(ResponsesProfile::StrictCodex);
    let decoded = engine().decode_request(WireFormat::OpenAiResponses, &body, &policy)?;
    let output = engine()
        .encode_request(WireFormat::OpenAiResponses, &decoded.request, &policy)?
        .body;

    assert_eq!(output["temperature"], json!(0.4), "{output}");
    assert_eq!(output["instructions"], json!("be terse"), "{output}");
    assert_eq!(output["metadata"], json!({"trace": "abc"}), "{output}");
    Ok(())
}
