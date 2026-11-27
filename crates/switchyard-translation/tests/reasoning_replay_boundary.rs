// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-private reasoning state is replayed only to the identity that issued it.
//!
//! `reasoning_details` and `signature` are opaque to everyone except their issuer. OpenRouter
//! returns them and expects them back unchanged; a native provider has no field for them. The
//! policy that decides this is per target, so these tests pin every direction plus the
//! boundaries that matter:
//!
//! * an explicit `Verbatim` target still receives its own details and signature unchanged;
//! * an explicit `PlaintextOnly` target never receives either, but keeps the reasoning prose;
//! * the default is `ProvenanceBound`, which sends opaque state only when the request carries
//!   proof that this target issued it;
//! * unproven state on the active continuation edge is a typed refusal, not a silent strip;
//! * unproven state in completed history is stripped while the portable prose survives;
//! * the same-format preserved-body fast path obeys the same rule, so a proven request keeps the
//!   caller's exact body and an unproven one is re-encoded.

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::provenance::{ReasoningIssuer, opaque_digest};
use switchyard_translation::{
    ReasoningAdmission, ReasoningDetailsReplay, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn policy(replay: ReasoningDetailsReplay) -> TranslationPolicy {
    TranslationPolicy {
        reasoning_details_replay: replay,
        ..TranslationPolicy::default()
    }
}

fn issuer() -> ReasoningIssuer {
    ReasoningIssuer {
        client: "openrouter-stephen".to_string(),
        model: "deepseek/deepseek-v4.1-flash-1017".to_string(),
        provider: "openrouter".to_string(),
        wire: "openai_chat".to_string(),
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

/// The digest the encoder computes for the reasoning state in [`body_with_signed_reasoning`].
fn issued_digest() -> String {
    let details = body_with_signed_reasoning()["messages"][1]["reasoning_details"]
        .as_array()
        .expect("fixture carries details")
        .clone();
    opaque_digest(&details, Some("sig-issuer-only"))
}

fn admission(admitted: Vec<String>, active_unproven: Vec<String>) -> ReasoningAdmission {
    ReasoningAdmission {
        issuer: issuer(),
        admitted,
        active_unproven,
    }
}

fn decode_request(engine: &TranslationEngine, body: &Value, policy: &TranslationPolicy) -> TestResult<switchyard_translation::LlmRequest> {
    Ok(engine
        .decode_request(WireFormat::OpenAiChat, body, policy)?
        .request)
}

// A target that issues these details must receive them back byte-for-byte and in order.
#[test]
fn verbatim_replay_preserves_details_order_and_signature() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = policy(ReasoningDetailsReplay::Verbatim);
    let request = decode_request(&engine, &body_with_signed_reasoning(), &policy)?;

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
    let request = decode_request(&engine, &body_with_signed_reasoning(), &verbatim)?;

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

// Withholding details must not withhold unsigned reasoning text: a narrowed target still gets the
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
    let request = decode_request(&engine, &body, &verbatim)?;

    let narrowed = policy(ReasoningDetailsReplay::PlaintextOnly);
    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &narrowed)?
        .body;

    let assistant = &output["messages"][1];
    assert!(assistant.get("reasoning_details").is_none());
    assert_eq!(assistant["reasoning"], "step one, step two");
    Ok(())
}

// The default must be the safe one: a policy built without target knowledge replays nothing it
// cannot prove.
#[test]
fn default_policy_is_provenance_bound() {
    assert_eq!(
        TranslationPolicy::default().reasoning_details_replay,
        ReasoningDetailsReplay::ProvenanceBound
    );
}

// Proven state is replayed to the target that issued it, details and signature intact.
#[test]
fn provenance_bound_replays_admitted_state_verbatim() -> TestResult {
    let engine = TranslationEngine::default();
    let source = policy(ReasoningDetailsReplay::Verbatim);
    let request = decode_request(&engine, &body_with_signed_reasoning(), &source)?;

    let mut bound = policy(ReasoningDetailsReplay::ProvenanceBound);
    bound.reasoning_admission = Some(admission(vec![issued_digest()], Vec::new()));
    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &bound)?
        .body;

    let details = output["messages"][1]["reasoning_details"]
        .as_array()
        .expect("admitted state is replayed");
    assert_eq!(details.len(), 2);
    assert_eq!(details[1]["data"], "opaque-blob-7f3a");
    assert_eq!(details[1]["signature"], "sig-issuer-only");
    Ok(())
}

// Unproven state with no admission at all is withheld: nothing could prove the target issued it.
#[test]
fn provenance_bound_withholds_unproven_state() -> TestResult {
    let engine = TranslationEngine::default();
    let source = policy(ReasoningDetailsReplay::Verbatim);
    let request = decode_request(&engine, &body_with_signed_reasoning(), &source)?;

    let bound = policy(ReasoningDetailsReplay::ProvenanceBound);
    assert!(
        bound.reasoning_admission.is_none(),
        "the fixture must exercise the absence of proof"
    );
    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &bound)?
        .body;

    let encoded = serde_json::to_string(&output)?;
    assert!(
        !encoded.contains("opaque-blob-7f3a") && !encoded.contains("sig-issuer-only"),
        "unproven opaque state must not be sent: {encoded}"
    );
    assert_eq!(
        output["messages"][1]["reasoning"], "consider wavelengths",
        "the portable prose still reaches the target"
    );
    Ok(())
}

// Admission is bound to the exact bytes: the same request with a different signature is not
// admitted, which is what makes a digest match evidence rather than authorization.
#[test]
fn provenance_bound_does_not_admit_state_whose_bytes_differ() -> TestResult {
    let engine = TranslationEngine::default();
    let source = policy(ReasoningDetailsReplay::Verbatim);
    let request = decode_request(&engine, &body_with_signed_reasoning(), &source)?;

    let mut bound = policy(ReasoningDetailsReplay::ProvenanceBound);
    // Admit a digest that belongs to some other signature entirely.
    let elsewhere = opaque_digest(
        body_with_signed_reasoning()["messages"][1]["reasoning_details"]
            .as_array()
            .unwrap(),
        Some("a-different-signature"),
    );
    bound.reasoning_admission = Some(admission(vec![elsewhere], Vec::new()));

    let output = engine
        .encode_request(WireFormat::OpenAiChat, &request, &bound)?
        .body;
    let encoded = serde_json::to_string(&output)?;
    assert!(
        !encoded.contains("opaque-blob-7f3a"),
        "a digest for different bytes must not admit these bytes: {encoded}"
    );
    Ok(())
}

// State the active continuation needs but cannot prove is refused outright, not silently
// dropped: the provider's own contract cannot accept a continuation missing it.
#[test]
fn provenance_bound_refuses_unproven_active_state() -> TestResult {
    let engine = TranslationEngine::default();
    let source = policy(ReasoningDetailsReplay::Verbatim);
    let request = decode_request(&engine, &body_with_signed_reasoning(), &source)?;

    let mut bound = policy(ReasoningDetailsReplay::ProvenanceBound);
    bound.reasoning_admission = Some(admission(Vec::new(), vec![issued_digest()]));

    let error = engine
        .encode_request(WireFormat::OpenAiChat, &request, &bound)
        .expect_err("unproven active continuation state must be refused");
    assert_eq!(
        error.kind(),
        "LossyConversion",
        "the refusal is typed, not a generic failure"
    );
    let message = error.to_string();
    assert!(
        message.contains("refusing rather than replaying unproven"),
        "the refusal names what it refused to do: {message}"
    );
    Ok(())
}

// The preserved-body fast path must not bypass the rule. A proven request keeps the caller's
// exact body, including fields the IR cannot represent; an unproven one is re-encoded so the
// opaque state is actually removed.
#[test]
fn the_preserved_body_path_obeys_provenance() -> TestResult {
    let engine = TranslationEngine::default();
    // An unrepresentable field the preserved-body path exists to protect.
    let mut body = body_with_signed_reasoning();
    body["messages"][1]["provider_private_note"] = json!("kept only by the preserved path");

    let source = policy(ReasoningDetailsReplay::Verbatim);
    let request = decode_request(&engine, &body, &source)?;

    // Proven: the exact body is replayed, unknown field and all.
    let mut proven = policy(ReasoningDetailsReplay::ProvenanceBound);
    proven.reasoning_admission = Some(admission(vec![issued_digest()], Vec::new()));
    let preserved = engine
        .encode_request(WireFormat::OpenAiChat, &request, &proven)?
        .body;
    assert_eq!(
        preserved["messages"][1]["provider_private_note"], "kept only by the preserved path",
        "a proven request keeps the exact caller body"
    );
    assert_eq!(
        preserved["messages"][1]["reasoning_details"][1]["signature"], "sig-issuer-only",
        "and keeps the opaque state the target issued"
    );

    // Unproven: the normalised path runs and the blob is gone.
    let unproven = policy(ReasoningDetailsReplay::ProvenanceBound);
    let reencoded = engine
        .encode_request(WireFormat::OpenAiChat, &request, &unproven)?
        .body;
    let encoded = serde_json::to_string(&reencoded)?;
    assert!(
        !encoded.contains("opaque-blob-7f3a") && !encoded.contains("sig-issuer-only"),
        "the fast path must not replay unproven opaque state: {encoded}"
    );
    assert_eq!(
        reencoded["messages"][1]["reasoning"], "consider wavelengths",
        "the portable prose survives the normalised path"
    );
    Ok(())
}
