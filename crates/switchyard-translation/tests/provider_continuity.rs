// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Red fixtures for the provider-continuity contract.
//!
//! These fixtures assert the contract implemented in
//! [`switchyard_translation::continuity`] and are EXPECTED TO FAIL against the
//! current encoder: nothing in the request path consults the contract yet, so
//! history from one provider is forwarded to another unchanged.
//!
//! They are `#[ignore]`d so the default suite stays green, and are run
//! deliberately with `--ignored` to demonstrate the defect:
//!
//! ```text
//! cargo test -p switchyard-translation --test provider_continuity -- --ignored
//! ```
//!
//! Each fixture names the exact behaviour it expects once continuity is
//! enforced. They are the acceptance tests for that work, not a description of
//! today's behaviour.

use serde_json::{Value, json};
use switchyard_translation::{
    PreservationPolicy, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A Chat request whose assistant turn carries opaque, provider-scoped reasoning
/// plus a completed tool cycle — the shape a reasoning model produces.
fn chat_with_opaque_reasoning() -> Value {
    json!({
        "model": "producer-route",
        "messages": [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "check the weather"},
            {"role": "assistant", "content": "checking",
             "reasoning_content": "plaintext chain",
             "reasoning_details": [
                 {"type": "reasoning", "id": "rs_producer_1",
                  "encrypted_content": "OPAQUE-CIPHERTEXT-BOUND-TO-PRODUCER"}
             ],
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ]
    })
}

/// A Chat request with a tool call whose result has NOT been returned yet: the
/// conversation is mid-cycle.
fn chat_mid_tool_cycle() -> Value {
    json!({
        "model": "producer-route",
        "messages": [
            {"role": "user", "content": "check the weather"},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "call_pending", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}]}
        ]
    })
}

fn preservation_enabled() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::InMemory,
        ..TranslationPolicy::default()
    }
}

fn contains(body: &Value, needle: &str) -> bool {
    body.to_string().contains(needle)
}

// CONTRACT: a provider change must not forward opaque reasoning that the target
// cannot verify. Today the preserved body is replayed verbatim, ciphertext and
// all, because `exact_preserved_request` returns the original body and only the
// model name is rewritten.
#[test]
#[ignore = "red until continuity is enforced (S2): preserved replay ignores the provider boundary"]
fn provider_change_must_not_replay_opaque_reasoning() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = preservation_enabled();
    let body = chat_with_opaque_reasoning();

    let decoded = engine.decode_request(WireFormat::OpenAiChat, &body, &policy)?;
    let encoded = engine.encode_request(WireFormat::OpenAiChat, &decoded.request, &policy)?;

    assert!(
        !contains(&encoded.body, "OPAQUE-CIPHERTEXT-BOUND-TO-PRODUCER"),
        "opaque provider-scoped reasoning reached a different provider: {}",
        encoded.body
    );
    Ok(())
}

// CONTRACT: an unfinished tool cycle is affinity-bound to the provider that
// opened it. Today the request is translated and dispatched with no check at
// all, so the pending call can be replayed to a provider that never issued it.
#[test]
#[ignore = "red until continuity is enforced (S2): no unfinished-cycle refusal exists"]
fn unfinished_tool_cycle_must_be_refused_across_providers() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = chat_mid_tool_cycle();

    let outcome = engine.translate_request(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        &body,
        &policy,
    );

    match outcome {
        Err(error) => {
            let text = error.to_string();
            assert!(
                text.contains("unfinished_tool_cycle_across_providers"),
                "refused for the wrong reason: {text}"
            );
            Ok(())
        }
        Ok(encoded) => panic!(
            "mid-cycle history crossed a provider boundary without refusal: {}",
            encoded.body
        ),
    }
}

// CONTRACT: a Responses reasoning item carries an `encrypted_content` payload
// bound to the provider that issued it. Translating to Anthropic must not
// fabricate a native `thinking` block from it, and must not forward the
// ciphertext under a different protocol.
#[test]
#[ignore = "red until continuity is enforced (S2): opaque item crosses protocol boundary"]
fn responses_encrypted_reasoning_must_not_cross_protocols() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = json!({
        "model": "producer-route",
        "input": [
            {"type": "reasoning", "id": "rs_1", "summary": [],
             "encrypted_content": "OPAQUE-RESPONSES-CIPHERTEXT"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hello"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "again"}]}
        ]
    });

    let encoded = engine.translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
        &body,
        &policy,
    )?;
    assert!(
        !contains(&encoded.body, "OPAQUE-RESPONSES-CIPHERTEXT"),
        "provider-scoped ciphertext crossed into another protocol: {}",
        encoded.body
    );
    Ok(())
}

// CONTRACT (negative control for the fixtures above): the same provider on the
// same protocol KEEPS opaque reasoning. If this ever fails, the contract has been
// over-applied and same-provider continuations are being damaged — which is the
// failure mode the review warned against.
#[test]
fn same_provider_keeps_opaque_reasoning() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = preservation_enabled();
    let body = chat_with_opaque_reasoning();

    let decoded = engine.decode_request(WireFormat::OpenAiChat, &body, &policy)?;
    let encoded = engine.encode_request(WireFormat::OpenAiChat, &decoded.request, &policy)?;

    assert!(
        contains(&encoded.body, "OPAQUE-CIPHERTEXT-BOUND-TO-PRODUCER"),
        "same-provider replay dropped reasoning it must preserve"
    );
    Ok(())
}
