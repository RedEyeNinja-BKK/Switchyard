// SPDX-License-Identifier: Apache-2.0

//! A Responses upstream that requires a list-shaped `input`.
//!
//! The Codex backend rejects the compact bare-string form with `Input must be a list`, so a
//! chat request reaching that lane used to fail on a shape every other Responses upstream
//! accepts. These cover the declared case and the neighbouring case that must not change.

use switchyard_translation::{
    TargetCapabilities, TranslationPolicy, WireFormat, decode_request, encode_request_with_policy,
};

fn policy(requires_list_input: bool) -> TranslationPolicy {
    TranslationPolicy {
        target_capabilities: TargetCapabilities {
            requires_list_input: Some(requires_list_input),
            ..TargetCapabilities::default()
        },
        ..TranslationPolicy::default()
    }
}

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 16
    })
}

/// The declared case: a lone user message must be a list when the upstream requires it.
#[test]
fn chat_single_user_text_is_a_list_when_the_upstream_requires_one() {
    let request = decode_request(WireFormat::OpenAiChat, &chat_body()).expect("decodes");
    let encoded = encode_request_with_policy(&request, WireFormat::OpenAiResponses, &policy(true))
        .expect("encodes");
    assert!(
        encoded["input"].is_array(),
        "an upstream that requires a list must not receive a bare string, got {}",
        encoded["input"]
    );
}

/// The nearby case: with the capability unset the compact form is unchanged, so every other
/// Responses upstream keeps the encoding it already accepts.
#[test]
fn chat_single_user_text_stays_a_string_without_the_capability() {
    let request = decode_request(WireFormat::OpenAiChat, &chat_body()).expect("decodes");
    let encoded = encode_request_with_policy(&request, WireFormat::OpenAiResponses, &policy(false))
        .expect("encodes");
    assert_eq!(
        encoded["input"],
        serde_json::json!("Reply OK"),
        "an upstream that does not declare the capability keeps the compact form"
    );
}

/// A native Responses request is untouched by the capability in either direction.
#[test]
fn native_responses_input_is_unchanged() {
    let native = serde_json::json!({
        "model": "gpt-6-luna",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Reply OK"}]}],
        "max_output_tokens": 16
    });
    let request = decode_request(WireFormat::OpenAiResponses, &native).expect("decodes");
    for requires in [true, false] {
        let encoded =
            encode_request_with_policy(&request, WireFormat::OpenAiResponses, &policy(requires))
                .expect("encodes");
        assert!(
            encoded["input"].is_array(),
            "a native list input must stay a list (requires_list_input={requires})"
        );
    }
}
