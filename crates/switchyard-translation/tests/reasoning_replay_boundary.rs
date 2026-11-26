// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-private reasoning state is replayed only to the provider that issued it.
//!
//! `reasoning_details` and `signature` are opaque to everyone except their issuer. OpenRouter
//! returns them and expects them back unchanged; a native provider has no field for them. The
//! policy that decides this is per target, so these tests pin both directions plus the
//! boundary itself: a target that withholds details must still carry the reasoning text, and
//! must never carry the signature.

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::{
    ReasoningDetailsReplay, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn policy(replay: ReasoningDetailsReplay) -> TranslationPolicy {
    TranslationPolicy {
        reasoning_details_replay: replay,
        ..TranslationPolicy::default()
    }
}

/// An assistant turn carrying structured reasoning exactly as OpenRouter hands it back:
/// ordered details, one of them signed, plus the plaintext the caller saw.
fn body_with_signed_reasoning() -> Value {
    json!({
        "model": "route",
        "messages": [
            {"role": "user", "content": "why is the sky blue"},
            {
                "role": "assistant",
                "content": "Rayleigh scattering.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "consider wavelengths", "index": 0},
                    {"type": "reasoning.encrypted", "data": "opaque-blob-7f3a", "index": 0,
                     "signature": "sig-issuer-only"}
                ]
            },
            {"role": "user", "content": "and at sunset?"}
        ]
    })
}

// A target that issues these details must receive them back byte-for-byte and in order.
#[test]
fn verbatim_replay_preserves_details_order_and_signature() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = policy(ReasoningDetailsReplay::Verbatim);
    let request = engine
        .decode_request(
            WireFormat::OpenAiChat,
            &body_with_signed_reasoning(),
            &policy,
        )?
        .request;

    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &policy)?
        .body;

    let assistant = &output["messages"][1];
    let details = assistant["reasoning_details"]
        .as_array()
        .expect("verbatim replay keeps structured details");
    assert_eq!(details.len(), 2, "both details survive: {details:?}");
    assert_eq!(details[0]["type"], "reasoning.text");
    assert_eq!(details[1]["type"], "reasoning.encrypted");
    assert_eq!(details[1]["data"], "opaque-blob-7f3a");
    assert_eq!(details[1]["signature"], "sig-issuer-only");
    assert_eq!(assistant["content"], "Rayleigh scattering.");
    Ok(())
}

// A target that did not issue the state must not receive it — not the blob, not the signature.
#[test]
fn plaintext_only_replay_withholds_details_and_signature() -> TestResult {
    let engine = TranslationEngine::default();
    let verbatim = policy(ReasoningDetailsReplay::Verbatim);
    let request = engine
        .decode_request(
            WireFormat::OpenAiChat,
            &body_with_signed_reasoning(),
            &verbatim,
        )?
        .request;

    let narrowed = policy(ReasoningDetailsReplay::PlaintextOnly);
    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &narrowed)?
        .body;

    let assistant = &output["messages"][1];
    assert!(
        assistant.get("reasoning_details").is_none(),
        "structured details must not cross a provider boundary: {assistant}"
    );
    let encoded = serde_json::to_string(&output)?;
    assert!(
        !encoded.contains("opaque-blob-7f3a") && !encoded.contains("sig-issuer-only"),
        "neither the blob nor the signature may appear anywhere in the replayed body: {encoded}"
    );
    assert_eq!(
        assistant["content"], "Rayleigh scattering.",
        "the caller-visible answer is unaffected by the replay policy"
    );
    Ok(())
}

// Withholding details must not withhold unsiged reasoning text: a native target still gets the
// thinking it needs for continuity, in the same field it always used.
#[test]
fn plaintext_only_replay_keeps_unsigned_reasoning_text() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "route",
        "messages": [
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "step one, step two", "index": 0}
                ]
            }
        ]
    });
    let verbatim = policy(ReasoningDetailsReplay::Verbatim);
    let request = engine
        .decode_request(WireFormat::OpenAiChat, &body, &verbatim)?
        .request;

    let narrowed = policy(ReasoningDetailsReplay::PlaintextOnly);
    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &narrowed)?
        .body;

    let assistant = &output["messages"][1];
    assert!(assistant.get("reasoning_details").is_none());
    assert_eq!(assistant["reasoning"], "step one, step two");
    Ok(())
}

// The default must be the existing behaviour, so a policy built without target knowledge does
// not silently start stripping fields from live traffic.
#[test]
fn default_policy_replays_verbatim() {
    assert_eq!(
        TranslationPolicy::default().reasoning_details_replay,
        ReasoningDetailsReplay::Verbatim
    );
}
