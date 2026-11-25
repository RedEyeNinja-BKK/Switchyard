// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for Chat-shaped tool traffic carried inside an OpenAI
//! Responses request.
//!
//! A client may restate a Chat transcript (an assistant message with a nested
//! `tool_calls` array, followed by `role: "tool"` results) inside a Responses
//! request, and may mix it with canonical typed items. The decoder must keep
//! every call and result — exact id, name, arguments, order, parallel grouping
//! — and a strict upstream must receive the typed form. Dropping a call while
//! keeping its `function_call_output` is what produced
//! `400 No tool call found for tool output`.
//!
//! Every case here is offline: no provider allowance is spent.

use serde_json::{Value, json};
use switchyard_translation::{
    ContentBlock, TranslationEngine, TranslationError, TranslationPolicy, WireFormat,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn engine() -> TranslationEngine {
    TranslationEngine::default()
}

fn responses_to_responses(body: &Value) -> TestResult<Value> {
    let out = engine().translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiResponses,
        body,
        &TranslationPolicy::default(),
    )?;
    Ok(out.body)
}

/// Runs the Responses->Responses translation expecting a refusal, and returns
/// the translation error itself rather than the boxed test error.
fn responses_refusal(body: &Value) -> TranslationError {
    match engine().translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiResponses,
        body,
        &TranslationPolicy::default(),
    ) {
        Ok(out) => panic!("expected a refusal, got body {}", out.body),
        Err(error) => error,
    }
}

fn responses_to_chat(body: &Value) -> TestResult<Value> {
    let out = engine().translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        body,
        &TranslationPolicy::default(),
    )?;
    Ok(out.body)
}

/// Collects `function_call` items from a Responses body, as (call_id, name, arguments).
fn responses_calls(body: &Value) -> Vec<(String, String, String)> {
    body["input"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|item| item["type"] == "function_call")
                .map(|item| {
                    (
                        item["call_id"].as_str().unwrap_or_default().to_string(),
                        item["name"].as_str().unwrap_or_default().to_string(),
                        item["arguments"]
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| item["arguments"].to_string()),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Collects `function_call_output` items from a Responses body, as (call_id, output).
fn responses_outputs(body: &Value) -> Vec<(String, Value)> {
    body["input"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|item| item["type"] == "function_call_output")
                .map(|item| {
                    (
                        item["call_id"].as_str().unwrap_or_default().to_string(),
                        item["output"].clone(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Collects assistant `tool_calls` from a Chat body, as (id, name, arguments).
fn chat_calls(body: &Value) -> Vec<(String, String, String)> {
    body["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message.get("tool_calls").and_then(Value::as_array))
                .flatten()
                .map(|call| {
                    (
                        call["id"].as_str().unwrap_or_default().to_string(),
                        call["function"]["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        call["function"]["arguments"]
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| call["function"]["arguments"].to_string()),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn chat_outputs(body: &Value) -> Vec<(String, String)> {
    body["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .filter(|message| message["role"] == "tool")
                .map(|message| {
                    (
                        message["tool_call_id"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        message["content"].as_str().unwrap_or_default().to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn chat_tool_call(id: &str, name: &str, arguments: &str) -> Value {
    json!({"id": id, "type": "function",
           "function": {"name": name, "arguments": arguments}})
}

// ---- 1. Nested Chat tool calls -------------------------------------------

#[test]
fn nested_chat_tool_calls_are_restated_as_typed_items() -> TestResult {
    let body = json!({
        "model": "direct-deepseek-deepseek-flash",
        "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
        "input": [
            {"role": "user", "content": "list /tmp"},
            {"role": "assistant", "content": "", "tool_calls": [
                chat_tool_call("call_1", "read_file", "{\"path\":\"/tmp\"}")
            ]},
            {"type": "function_call_output", "call_id": "call_1", "output": "fileA"},
            {"role": "user", "content": "thanks"}
        ]
    });

    let upstream = responses_to_responses(&body)?;
    assert_eq!(
        responses_calls(&upstream),
        vec![(
            "call_1".to_string(),
            "read_file".to_string(),
            "{\"path\":\"/tmp\"}".to_string()
        )],
        "the nested call must reach the upstream as a typed function_call"
    );
    assert_eq!(
        responses_outputs(&upstream),
        vec![("call_1".to_string(), json!("fileA"))],
        "the result must stay paired with its call"
    );
    // The chat-only field must not survive the rewrite.
    assert!(
        upstream["input"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item.get("tool_calls").is_none()),
        "no chat `tool_calls` field may reach a Responses upstream"
    );

    // And the decoder must see the call as a real tool call, not drop it.
    let decoded = engine().decode_request(
        WireFormat::OpenAiResponses,
        &body,
        &TranslationPolicy::default(),
    )?;
    let has_call = decoded.request.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolCall(_)))
    });
    let has_result = decoded.request.messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolResult(_)))
    });
    assert!(has_call, "decoded IR must carry the tool call");
    assert!(has_result, "decoded IR must carry the tool result");
    Ok(())
}

// ---- 2 & 3. Typed Responses calls and matching outputs pass through ------

#[test]
fn canonical_typed_history_is_forwarded_byte_identical() -> TestResult {
    let body = json!({
        "model": "direct-deepseek-deepseek-flash",
        "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
        "input": [
            {"role": "user", "content": "list /tmp"},
            {"type": "function_call", "call_id": "call_1", "name": "read_file",
             "arguments": "{\"path\":\"/tmp\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "fileA"},
            {"role": "user", "content": "thanks"}
        ]
    });

    let upstream = responses_to_responses(&body)?;
    assert_eq!(
        upstream, body,
        "an already-canonical body must be forwarded unchanged"
    );
    Ok(())
}

// ---- 4. Mixed representations --------------------------------------------

#[test]
fn mixed_typed_and_chat_history_preserves_order() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "first"},
            {"type": "function_call", "call_id": "typed_1", "name": "alpha",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "typed_1", "output": "ok"},
            {"role": "assistant", "content": "let me check", "tool_calls": [
                chat_tool_call("chat_1", "beta", "{\"n\":1}")
            ]},
            {"type": "function_call_output", "call_id": "chat_1", "output": "done"},
            {"role": "user", "content": "last"}
        ]
    });

    let upstream = responses_to_responses(&body)?;
    assert_eq!(
        responses_calls(&upstream),
        vec![
            ("typed_1".to_string(), "alpha".to_string(), "{}".to_string()),
            (
                "chat_1".to_string(),
                "beta".to_string(),
                "{\"n\":1}".to_string()
            ),
        ],
        "typed and restated chat calls must keep their original order"
    );
    assert_eq!(
        responses_outputs(&upstream)
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        vec!["typed_1".to_string(), "chat_1".to_string()]
    );
    // The visible assistant text of the chat turn survives.
    assert!(
        upstream["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["role"] == "assistant" && item["content"] == "let me check"),
        "assistant text must not be discarded while rewriting its calls"
    );
    Ok(())
}

// ---- 5. Parallel calls ----------------------------------------------------

#[test]
fn parallel_chat_calls_keep_ids_arguments_and_grouping() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "read three files"},
            {"role": "assistant", "content": null, "tool_calls": [
                chat_tool_call("call_a", "read_file", "{\"path\":\"/a\"}"),
                chat_tool_call("call_b", "read_file", "{\"path\":\"/b\"}"),
                chat_tool_call("call_c", "stat_file", "{\"path\":\"/c\"}")
            ]},
            {"role": "tool", "tool_call_id": "call_a", "content": "A"},
            {"role": "tool", "tool_call_id": "call_b", "content": "B"},
            {"role": "tool", "tool_call_id": "call_c", "content": "C"}
        ]
    });

    let upstream = responses_to_responses(&body)?;
    assert_eq!(
        responses_calls(&upstream),
        vec![
            (
                "call_a".to_string(),
                "read_file".to_string(),
                "{\"path\":\"/a\"}".to_string()
            ),
            (
                "call_b".to_string(),
                "read_file".to_string(),
                "{\"path\":\"/b\"}".to_string()
            ),
            (
                "call_c".to_string(),
                "stat_file".to_string(),
                "{\"path\":\"/c\"}".to_string()
            ),
        ],
        "all three parallel calls must survive with their own id, name and arguments"
    );
    let outputs = responses_outputs(&upstream);
    assert_eq!(
        outputs
            .iter()
            .map(|(id, out)| (id.as_str(), out.as_str().unwrap_or_default()))
            .collect::<Vec<_>>(),
        vec![("call_a", "A"), ("call_b", "B"), ("call_c", "C")],
        "each parallel result must stay attached to the call it answers"
    );
    Ok(())
}

// ---- 6. Streaming reconstruction -----------------------------------------

#[test]
fn replayed_streamed_turn_survives_as_typed_history() -> TestResult {
    // A streamed assistant turn is replayed on the next request as chat-shaped
    // history (call + result). The rewrite must not depend on how it arrived.
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                chat_tool_call("s_1", "search", "{\"q\":\"x\"}")
            ]},
            {"type": "function_call_output", "call_id": "s_1", "output": "hit"}
        ]
    });
    let upstream = responses_to_responses(&body)?;
    assert_eq!(responses_calls(&upstream).len(), 1);
    assert_eq!(responses_outputs(&upstream).len(), 1);
    assert_eq!(responses_calls(&upstream)[0].0, "s_1");
    Ok(())
}

// ---- 7. Compacted histories ----------------------------------------------

#[test]
fn compaction_markers_are_dropped_and_chat_calls_still_restated() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"type": "compaction_trigger"},
            {"role": "user", "content": "resume"},
            {"role": "assistant", "content": "", "tool_calls": [
                chat_tool_call("c_1", "lookup", "{}")
            ]},
            {"type": "function_call_output", "call_id": "c_1", "output": "value"}
        ]
    });
    let upstream = responses_to_responses(&body)?;
    let items = upstream["input"].as_array().unwrap();
    assert!(
        items
            .iter()
            .all(|item| item["type"] != "compaction_trigger"),
        "codex compaction markers must not reach the upstream"
    );
    assert_eq!(responses_calls(&upstream).len(), 1);
    assert_eq!(responses_outputs(&upstream).len(), 1);
    Ok(())
}

// ---- 8. Large history: ~200 messages, 122 tool calls ---------------------

#[test]
fn large_history_keeps_every_call_and_result() -> TestResult {
    // 61 tool round-trips (2 parallel calls each) = 122 calls, spread across a
    // ~200 item history, matching the shape of the production request.
    let mut input: Vec<Value> = vec![json!({"role": "user", "content": "start"})];
    let mut expected_calls: Vec<(String, String, String)> = Vec::new();
    let mut expected_outputs: Vec<String> = Vec::new();
    for round in 0..61 {
        let a = format!("r{round}_a");
        let b = format!("r{round}_b");
        input.push(json!({"role": "assistant", "content": "", "tool_calls": [
            chat_tool_call(&a, "read_file", &format!("{{\"r\":{round},\"k\":\"a\"}}")),
            chat_tool_call(&b, "read_file", &format!("{{\"r\":{round},\"k\":\"b\"}}"))
        ]}));
        input.push(json!({"role": "tool", "tool_call_id": a, "content": format!("a{round}")}));
        input.push(json!({"role": "tool", "tool_call_id": b, "content": format!("b{round}")}));
        expected_calls.push((
            a.clone(),
            "read_file".to_string(),
            format!("{{\"r\":{round},\"k\":\"a\"}}"),
        ));
        expected_calls.push((
            b.clone(),
            "read_file".to_string(),
            format!("{{\"r\":{round},\"k\":\"b\"}}"),
        ));
        expected_outputs.push(a);
        expected_outputs.push(b);
    }
    input.push(json!({"role": "user", "content": "final"}));
    // 1 + 61*3 + 1 = 185 items; add filler user turns to cross 200.
    for index in 0..20 {
        input.insert(
            1,
            json!({"role": "user", "content": format!("note {index}")}),
        );
    }
    let item_count = input.len();
    let body = json!({"model": "route", "input": input});

    let upstream = responses_to_responses(&body)?;
    let calls = responses_calls(&upstream);
    assert_eq!(
        calls.len(),
        122,
        "all 122 calls must survive (history {item_count})"
    );
    assert_eq!(
        calls, expected_calls,
        "ids, names, arguments and order must be exact"
    );
    let outputs: Vec<String> = responses_outputs(&upstream)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(outputs, expected_outputs, "every result must remain paired");
    // No id may be duplicated or invented.
    let mut unique = calls
        .iter()
        .map(|(id, _, _)| id.clone())
        .collect::<Vec<_>>();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 122, "call ids must be unique and unmodified");
    Ok(())
}

// ---- 9. Missing and unmatched tool-call ids ------------------------------

#[test]
fn nested_call_without_id_is_refused() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]}
        ]
    });
    let error = responses_refusal(&body);
    assert!(
        matches!(error, TranslationError::InvalidValue { .. }),
        "expected InvalidValue, got {error:?}"
    );
    assert!(
        error.to_string().contains("fabricate"),
        "refusal must state that no id was invented: {error}"
    );
    Ok(())
}

#[test]
fn tool_result_without_tool_call_id_is_refused() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "go"},
            {"role": "tool", "content": "orphan result"}
        ]
    });
    let error = responses_refusal(&body);
    assert!(matches!(error, TranslationError::InvalidValue { .. }));
    Ok(())
}

#[test]
fn tool_result_answering_an_absent_call_is_refused() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                chat_tool_call("known", "read_file", "{}")
            ]},
            {"role": "tool", "tool_call_id": "unknown", "content": "x"}
        ]
    });
    let error = responses_refusal(&body);
    assert!(matches!(error, TranslationError::InvalidValue { .. }));
    assert!(
        error.to_string().contains("unknown"),
        "refusal must name the unmatched id: {error}"
    );
    Ok(())
}

// ---- 10. Direct-versus-derived route parity -----------------------------

#[test]
fn chat_and_responses_derivations_carry_the_same_tool_structure() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                chat_tool_call("p_1", "read_file", "{\"path\":\"/1\"}"),
                chat_tool_call("p_2", "read_file", "{\"path\":\"/2\"}")
            ]},
            {"type": "function_call_output", "call_id": "p_1", "output": "one"},
            {"type": "function_call_output", "call_id": "p_2", "output": "two"}
        ]
    });

    // A direct Responses target keeps the typed items.
    let direct = responses_to_responses(&body)?;
    // A derived Chat target restates the same calls in Chat shape.
    let derived = responses_to_chat(&body)?;

    let direct_calls = responses_calls(&direct);
    let derived_calls = chat_calls(&derived);
    assert_eq!(
        direct_calls, derived_calls,
        "direct and derived targets must agree on call id, name, arguments and order"
    );
    assert_eq!(direct_calls.len(), 2);

    let mut direct_ids = responses_outputs(&direct)
        .into_iter()
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    let mut derived_ids = chat_outputs(&derived)
        .into_iter()
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    direct_ids.sort();
    derived_ids.sort();
    assert_eq!(
        direct_ids, derived_ids,
        "results must pair to the same ids on both derivations"
    );
    Ok(())
}

#[test]
fn empty_tool_calls_field_is_not_forwarded() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_calls": []}
        ]
    });
    let upstream = responses_to_responses(&body)?;
    assert!(
        upstream["input"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item.get("tool_calls").is_none()),
        "an empty chat call list must not be forwarded"
    );
    Ok(())
}

// ---- 11. Window-boundary orphan typed output (the production 400) --------

#[test]
fn typed_output_without_its_call_is_restated_as_readable_user_turn() -> TestResult {
    // A history window that starts mid tool block: the call is outside the
    // window, its result is inside. Forwarded verbatim this is the exact
    // `400 No tool call found for tool output`.
    let body = json!({
        "model": "direct-deepseek-deepseek-flash",
        "tools": [{"type": "function", "name": "bash", "parameters": {"type": "object"}}],
        "input": [
            {"type": "function_call_output", "call_id": "call_cut", "output": "exit 0"},
            {"role": "user", "content": "continue"}
        ]
    });

    let upstream = responses_to_responses(&body)?;
    let items = upstream["input"].as_array().unwrap();
    assert_eq!(
        responses_outputs(&upstream).len(),
        0,
        "an output with no call in the body must not be forwarded as a tool result"
    );
    assert_eq!(
        items[0],
        json!({"role": "user", "content": "Tool result call_cut: exit 0"}),
        "the stranded result becomes the same readable user turn the decoder produces"
    );
    Ok(())
}

#[test]
fn typed_output_is_preserved_when_previous_response_id_holds_the_call() -> TestResult {
    let body = json!({
        "model": "route",
        "previous_response_id": "resp_123",
        "input": [
            {"type": "function_call_output", "call_id": "call_held", "output": "ok"}
        ]
    });
    let upstream = responses_to_responses(&body)?;
    assert_eq!(
        upstream, body,
        "with stored provider state the output legitimately answers a held call"
    );
    Ok(())
}

#[test]
fn only_the_orphan_is_restated_when_calls_are_partly_present() -> TestResult {
    let body = json!({
        "model": "route",
        "input": [
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "kept", "name": "bash", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "kept", "output": "fine"},
            {"type": "function_call_output", "call_id": "orphan", "output": "stray"}
        ]
    });
    let upstream = responses_to_responses(&body)?;
    let outputs = responses_outputs(&upstream);
    assert_eq!(outputs.len(), 1, "the paired output stays a tool result");
    assert_eq!(outputs[0].0, "kept");
    assert!(
        upstream["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["role"] == "user" && item["content"] == "Tool result orphan: stray"),
        "the orphan becomes a readable user turn"
    );
    Ok(())
}
