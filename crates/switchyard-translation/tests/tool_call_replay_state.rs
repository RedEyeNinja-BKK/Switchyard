// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Continuation state on replayed tool calls.
//!
//! Google issues a `thought_signature` beside a `functionCall` part and refuses a replayed
//! part that lacks it (HTTP 400 `INVALID_ARGUMENT`). On the OpenAI-compatible surface the
//! value travels as `tool_calls[].extra_content`, so it is opaque state that must survive a
//! Chat-to-Chat translation and must come back only to the wire that understands it.
//!
//! The documented escape for a call the wire did not issue is the literal
//! `skip_thought_signature_validator` ("history transferred from a model that does not
//! include thought signatures"). These cases pin both halves: genuine state is echoed
//! verbatim, and a call without state is replayed with the placeholder rather than bare.
//!
//! Every case is offline and spends no provider allowance.

use serde_json::{Value, json};
use switchyard_translation::{
    TargetCapabilities, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const PLACEHOLDER: &str = "skip_thought_signature_validator";
const SIGNATURE: &str = "CiQBj7Xk9pQ2mS5rV8c1tY3uE6wL0aB4nD7fH2jK5mP8sU=";

fn google_policy() -> TranslationPolicy {
    TranslationPolicy {
        target_capabilities: TargetCapabilities {
            tool_signature_placeholder: Some(PLACEHOLDER.to_string()),
            ..TargetCapabilities::default()
        },
        ..TranslationPolicy::default()
    }
}

fn translate(body: &Value, policy: &TranslationPolicy) -> TestResult<Value> {
    let out = TranslationEngine::default().translate_request(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
        body,
        policy,
    )?;
    Ok(out.body)
}

fn call(id: &str, name: &str) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": "{\"q\":\"x\"}"}
    })
}

/// An assistant turn that issued one tool call, followed by its result.
fn history_with_assistant_calls(calls: Vec<Value>) -> Value {
    json!({
        "model": "gemini-3.5-flash-lite",
        "messages": [
            {"role": "user", "content": "check it"},
            {"role": "assistant", "content": null, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {"role": "user", "content": "and now?"}
        ]
    })
}

fn outbound_call<'a>(body: &'a Value, index: usize) -> &'a Value {
    &body["messages"][1]["tool_calls"][index]
}

#[test]
fn a_genuine_signature_is_echoed_verbatim() -> TestResult {
    let mut signed = call("call-1", "lookup");
    signed["extra_content"] = json!({"google": {"thought_signature": SIGNATURE}});
    let body = history_with_assistant_calls(vec![signed]);

    let out = translate(&body, &google_policy())?;

    assert_eq!(
        outbound_call(&out, 0)["extra_content"]["google"]["thought_signature"],
        json!(SIGNATURE),
        "the provider's own state must come back byte for byte"
    );
    Ok(())
}

#[test]
fn a_foreign_call_is_replayed_with_the_documented_placeholder() -> TestResult {
    // No `extra_content`: the call was issued by a model that carries no signature.
    let body = history_with_assistant_calls(vec![call("call-1", "lookup")]);

    let out = translate(&body, &google_policy())?;

    assert_eq!(
        outbound_call(&out, 0)["extra_content"]["google"]["thought_signature"],
        json!(PLACEHOLDER),
        "a call without state would otherwise be replayed bare and refused with 400"
    );
    Ok(())
}

#[test]
fn only_the_first_call_of_a_step_needs_a_signature() -> TestResult {
    // Parallel calls: Google signs only the first part, so a second part without a signature
    // is the documented normal shape and must not be given one.
    let body =
        history_with_assistant_calls(vec![call("call-1", "lookup"), call("call-2", "other")]);

    let out = translate(&body, &google_policy())?;

    assert_eq!(
        outbound_call(&out, 0)["extra_content"]["google"]["thought_signature"],
        json!(PLACEHOLDER)
    );
    assert_eq!(
        outbound_call(&out, 1).get("extra_content"),
        None,
        "a parallel non-first part carries no signature by design"
    );
    Ok(())
}

#[test]
fn the_placeholder_never_overwrites_a_genuine_signature() -> TestResult {
    let mut signed = call("call-1", "lookup");
    signed["extra_content"] = json!({"google": {"thought_signature": SIGNATURE}});
    let body = history_with_assistant_calls(vec![signed, call("call-2", "other")]);

    let out = translate(&body, &google_policy())?;

    assert_eq!(
        outbound_call(&out, 0)["extra_content"]["google"]["thought_signature"],
        json!(SIGNATURE)
    );
    Ok(())
}

#[test]
fn each_step_of_a_multi_step_turn_is_covered() -> TestResult {
    let body = json!({
        "model": "gemini-3.5-flash-lite",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": null, "tool_calls": [call("call-1", "one")]},
            {"role": "tool", "tool_call_id": "call-1", "content": "1"},
            {"role": "assistant", "content": null, "tool_calls": [call("call-2", "two")]},
            {"role": "tool", "tool_call_id": "call-2", "content": "2"},
            {"role": "user", "content": "done?"}
        ]
    });

    let out = translate(&body, &google_policy())?;

    // Validation covers every step in the current turn, so both assistant turns are stepped.
    assert_eq!(
        out["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        json!(PLACEHOLDER)
    );
    assert_eq!(
        out["messages"][3]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        json!(PLACEHOLDER)
    );
    Ok(())
}

#[test]
fn a_cross_format_translation_drops_state_when_nothing_is_declared() -> TestResult {
    // Replay provider-private state only to a wire that asked for it. A same-format body is
    // forwarded as the caller composed it, but a translation rebuilds the call, and a target
    // that declares no placeholder has no use for another provider's signature.
    let body = json!({
        "model": "gemini-3.5-flash-lite",
        "input": [
            {"type": "message", "role": "user", "content": "check it"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{\"q\":\"x\"}",
                "extra_content": {"google": {"thought_signature": SIGNATURE}}
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
            {"type": "message", "role": "user", "content": "and now?"}
        ]
    });

    let out = TranslationEngine::default().translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &body,
        &TranslationPolicy::default(),
    )?;

    let assistant = out
        .body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message["role"] == "assistant")
        })
        .expect("the replayed call must reach the chat body");
    assert_eq!(
        assistant["tool_calls"][0].get("extra_content"),
        None,
        "no declaration means no continuation state leaves the gateway"
    );
    Ok(())
}

#[test]
fn a_cross_format_translation_carries_genuine_state_to_a_declaring_wire() -> TestResult {
    let body = json!({
        "model": "gemini-3.5-flash-lite",
        "input": [
            {"type": "message", "role": "user", "content": "check it"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{\"q\":\"x\"}",
                "extra_content": {"google": {"thought_signature": SIGNATURE}}
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
            {"type": "message", "role": "user", "content": "and now?"}
        ]
    });

    let out = TranslationEngine::default().translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &body,
        &google_policy(),
    )?;

    let assistant = out
        .body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message["role"] == "assistant")
        })
        .expect("the replayed call must reach the chat body");
    assert_eq!(
        assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        json!(SIGNATURE),
        "the provider's own signature must survive a format change unchanged"
    );
    Ok(())
}

#[test]
fn a_cross_format_foreign_call_is_given_the_placeholder() -> TestResult {
    // The Turnstone-shaped path: a Responses caller replays a call another model issued.
    let body = json!({
        "model": "gemini-3.5-flash-lite",
        "input": [
            {"type": "message", "role": "user", "content": "check it"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{\"q\":\"x\"}"
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
            {"type": "message", "role": "user", "content": "and now?"}
        ]
    });

    let out = TranslationEngine::default().translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &body,
        &google_policy(),
    )?;

    let assistant = out
        .body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message["role"] == "assistant")
        })
        .expect("the replayed call must reach the chat body");
    assert_eq!(
        assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        json!(PLACEHOLDER)
    );
    Ok(())
}

#[test]
fn a_same_format_replay_forwards_the_callers_own_bytes() -> TestResult {
    // Same-format preservation returns the caller's body unchanged, so state it already
    // carries is neither stripped nor rewritten: the gateway replays what it was given.
    let mut signed = call("call-1", "lookup");
    signed["extra_content"] = json!({"google": {"thought_signature": SIGNATURE}});
    let body = history_with_assistant_calls(vec![signed]);

    let out = translate(&body, &TranslationPolicy::default())?;

    assert_eq!(
        outbound_call(&out, 0)["extra_content"]["google"]["thought_signature"],
        json!(SIGNATURE)
    );
    Ok(())
}
