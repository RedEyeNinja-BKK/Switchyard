// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Responses-only features a Chat-Completions target cannot honour.
//!
//! The OpenAI Responses API and the Chat Completions API are not two spellings of one
//! protocol: some Responses features are carried *out* by the provider, and a Chat target has
//! nothing to carry them. Google documents `/v1beta/openai/chat/completions` and no
//! `/openai/responses` path, so a caller's `/responses` body is exactly the case that reaches
//! a target which cannot implement these.
//!
//! The failure this prevents is silent: the decoder drops what it cannot map, the answer comes
//! back as ordinary prose, and the caller has no way to tell that the search it asked for never
//! ran. That is worse than a refusal, because it looks like success. So a request that needs a
//! feature with no proven mapping is refused here, before egress, naming the feature.
//!
//! This is deliberately a *narrow* list. Only features that change what the answer means are
//! refused. Fields that merely decorate the call - `metadata`, `user`, `service_tier`,
//! `store`, `truncation`, `prompt_cache_key` - are passed through or ignored as they are
//! today, because refusing them would break callers for no gain in fidelity.

use serde_json::Value;

/// One Responses-only feature the selected target cannot serve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedResponsesFeature {
    /// The field or tool type, exactly as the caller spelled it.
    pub name: &'static str,
    /// What this target cannot do, in the terms the caller used.
    pub reason: &'static str,
}

impl UnsupportedResponsesFeature {
    /// The diagnostic text a client receives.
    pub fn message(&self) -> String {
        format!(
            "this request needs the Responses feature `{}`, which the selected target cannot \
             serve: {}. Switchyard refuses it rather than answering without it; retry without \
             that feature, or select a target whose wire format is openai_responses",
            self.name, self.reason
        )
    }
}

/// Tool types the provider executes on its side.
///
/// A Chat-Completions target receives tools as function definitions the *caller* runs. A
/// hosted tool has no function for anyone to run, so forwarding the rest of the request is not
/// a degraded version of this feature - it is a different request that will usually answer
/// from memory and present the result as a normal reply.
const HOSTED_TOOL_TYPES: &[(&str, &str)] = &[
    (
        "web_search",
        "the search runs on the provider's side; a Chat target would answer without it",
    ),
    (
        "web_search_preview",
        "the search runs on the provider's side; a Chat target would answer without it",
    ),
    (
        "web_search_preview_2025_03_11",
        "the search runs on the provider's side; a Chat target would answer without it",
    ),
    (
        "file_search",
        "retrieval runs against the provider's vector store, which this target has no access to",
    ),
    (
        "code_interpreter",
        "code runs in the provider's sandbox; a Chat target has no sandbox to run it in",
    ),
    (
        "image_generation",
        "image generation is a provider-side tool call, not a chat completion",
    ),
    (
        "computer_use_preview",
        "the provider drives a browser on its side; a Chat target cannot",
    ),
    (
        "mcp",
        "the provider connects to the MCP server on its side, so no tool definition reaches the model here",
    ),
    (
        "local_shell",
        "a provider-side shell has no equivalent on a Chat target",
    ),
];

/// Whether this exact tool entry would be dropped entirely by the Responses decoder.
///
/// Mirrors [`super::buffered`]'s tool decoding rather than guessing: an entry contributes a
/// tool when it is a namespace container, a named `custom` tool, a `function`-shaped tool, a
/// flat `{name, parameters}` tool, or an `id`-addressed tool. Anything else has no name and no
/// id, so it cannot be dispatched to - which for a hosted tool means the model would be told
/// nothing about it.
fn is_dropped_by_the_decoder(tool: &serde_json::Map<String, Value>) -> bool {
    let kind = tool.get("type").and_then(Value::as_str);
    if kind == Some("namespace") {
        return false;
    }
    let has_name = tool
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| !name.is_empty());
    let has_function_name = tool
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .is_some_and(|name| !name.is_empty());
    let has_id = tool
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.is_empty());
    !(has_name || has_function_name || has_id)
}

/// The canonical name and the reason, when `kind` is one of the known hosted kinds.
///
/// The static name is returned rather than the caller's own spelling so the diagnostic is
/// always a value this file can name, and so the refusal cannot echo arbitrary caller text.
fn hosted_tool(kind: &str) -> Option<(&'static str, &'static str)> {
    HOSTED_TOOL_TYPES
        .iter()
        .find(|(name, _)| *name == kind)
        .copied()
}

/// Returns the first Responses feature in `body` that a non-Responses target cannot serve.
///
/// `body` is the caller's verbatim `/responses` request. Callers must pass the request they
/// actually received: a normalised IR has already lost the distinction this checks.
pub fn unsupported_feature(body: &Value) -> Option<UnsupportedResponsesFeature> {
    if body.get("background").and_then(Value::as_bool) == Some(true) {
        return Some(UnsupportedResponsesFeature {
            name: "background",
            reason: "the provider runs the response asynchronously and holds it; a Chat target \
                     returns one synchronous answer and would report it as complete",
        });
    }

    for tool in body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        if let Some(kind) = tool.get("type").and_then(Value::as_str)
            && let Some((name, reason)) = hosted_tool(kind)
        {
            return Some(UnsupportedResponsesFeature { name, reason });
        }
        if is_dropped_by_the_decoder(tool) {
            // Unnamed and unaddressable: whatever this tool was, the target would never learn
            // it existed, and the caller would never learn that either.
            return Some(UnsupportedResponsesFeature {
                name: "tools[].type",
                reason: "this tool entry carries neither a name nor an id, so it cannot be \
                         sent to a Chat target at all",
            });
        }
    }

    // Tools may also arrive inside `input` as an `additional_tools` developer item. The
    // decoder reads those into the same tool list, so the same losses apply.
    for item in body
        .get("input")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(item) = item.as_object() else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) != Some("additional_tools") {
            continue;
        }
        for tool in item
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(tool) = tool.as_object() else {
                continue;
            };
            if let Some(kind) = tool.get("type").and_then(Value::as_str)
                && let Some((name, reason)) = hosted_tool(kind)
            {
                return Some(UnsupportedResponsesFeature { name, reason });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn an_ordinary_responses_request_is_not_refused() {
        // The negative control: everything the bounded lane actually sends. A refusal here
        // would be the guard breaking real traffic, so it is asserted first.
        let body = json!({
            "model": "gemini-3.5-flash-lite",
            "instructions": "be brief",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "function_call_output", "call_id": "c1", "output": "42"}
            ],
            "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
                {"type": "custom", "name": "shell", "description": "run a command"},
                {"type": "namespace", "name": "ns", "tools": []}
            ],
            "tool_choice": "auto",
            "max_output_tokens": 256,
            "stream": false,
            "store": false,
            "metadata": {"trace": "t1"},
            "user": "u-1",
            "truncation": "auto",
            "include": ["reasoning.encrypted_content"],
            "text": {"format": {"type": "json_object"}}
        });
        assert_eq!(unsupported_feature(&body), None);
    }

    #[test]
    fn a_hosted_tool_is_named_before_egress() {
        // The dangerous one: the answer would come back without a search and read as success.
        let body = json!({
            "model": "gemini-3.5-flash-lite",
            "input": "when did the market close?",
            "tools": [{"type": "web_search"}]
        });
        let feature = unsupported_feature(&body).expect("web_search is refused");
        assert_eq!(feature.name, "web_search");
        assert!(
            feature.message().contains("web_search"),
            "the client must be told which feature: {}",
            feature.message()
        );
        assert!(feature.message().contains("openai_responses"));

        for kind in ["file_search", "code_interpreter", "image_generation", "mcp"] {
            let body = json!({"model": "m", "tools": [{"type": kind}]});
            assert_eq!(
                unsupported_feature(&body).map(|feature| feature.name),
                Some(kind),
                "{kind} must be refused by name"
            );
        }
    }

    #[test]
    fn a_tool_with_no_name_and_no_id_is_refused_rather_than_dropped() {
        // Not one of the known hosted kinds, but still unaddressable: the structural rule
        // covers types this file has never heard of, which is what keeps it honest as the
        // vendor adds them.
        let body = json!({"model": "m", "tools": [{"type": "some_future_tool"}]});
        assert_eq!(
            unsupported_feature(&body).map(|feature| feature.name),
            Some("tools[].type")
        );
    }

    #[test]
    fn hosted_tools_hidden_in_additional_tools_are_refused_too() {
        let body = json!({
            "model": "m",
            "input": [
                {"type": "additional_tools", "tools": [{"type": "web_search"}]},
                {"type": "message", "role": "user", "content": "hi"}
            ]
        });
        assert_eq!(
            unsupported_feature(&body).map(|feature| feature.name),
            Some("web_search")
        );
    }

    #[test]
    fn background_execution_is_refused_but_a_false_flag_is_not() {
        let background = json!({"model": "m", "input": "hi", "background": true});
        assert_eq!(
            unsupported_feature(&background).map(|feature| feature.name),
            Some("background")
        );
        // Callers that send the field explicitly as false are asking for the synchronous
        // behaviour this target provides, so they are served.
        let synchronous = json!({"model": "m", "input": "hi", "background": false});
        assert_eq!(unsupported_feature(&synchronous), None);
    }
}
