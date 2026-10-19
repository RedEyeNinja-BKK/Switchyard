// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request fit: whether a route's declared capabilities can serve a request.
//!
//! This lives in the runner rather than the serving crate because two very different
//! callers need one interpretation of it. HTTP ingress checks it before dispatch, and a
//! whole-route composite checks it before invoking a child, so "vision", "tools",
//! "reasoning controls" and preserved provider fields mean exactly the same thing in
//! both places. A capability mismatch found here is a routing fact, not a failure: the
//! route simply was not eligible.

use crate::ModelCapabilities;
use serde_json::Value;
use switchyard_protocol::{ContentBlock, LlmRequest};

/// Return the first explicitly disabled capability used by decoded or preserved input.
/// Forwarding can retain provider JSON fields that decoding omits.
pub fn unsupported_capability(
    capabilities: ModelCapabilities,
    request: &LlmRequest,
    body: &Value,
) -> Option<&'static str> {
    if capabilities.reasoning == Some(false)
        && (request.reasoning.effort.is_some()
            || request
                .reasoning
                .raw
                .as_ref()
                .is_some_and(|value| !value.is_null())
            || ["reasoning", "reasoning_effort", "thinking"]
                .iter()
                .any(|key| body.get(key).is_some_and(|value| !value.is_null()))
            || body
                .pointer("/output_config/effort")
                .is_some_and(|value| !value.is_null()))
    {
        return Some("reasoning");
    }
    if capabilities.tool_calling == Some(false)
        && (!request.tools.is_empty()
            || request.tool_choice.is_some()
            || ["tools", "functions"].iter().any(|key| {
                body.get(key)
                    .and_then(Value::as_array)
                    .is_some_and(|tools| !tools.is_empty())
            })
            || ["tool_choice", "parallel_tool_calls", "function_call"]
                .iter()
                .any(|key| body.get(key).is_some_and(|value| !value.is_null())))
    {
        return Some("tool_calling");
    }
    // Audio first: it has no declaration at all and is refused wherever it appears. Video
    // follows, and is admitted only where the route states it can read video. Doing this
    // before the image and tool scan also means a request carrying both reports the medium
    // nothing can serve.
    if let Some(capability) = unsupported_media(capabilities, request, body) {
        return Some(capability);
    }
    if capabilities.tool_calling != Some(false) && capabilities.vision != Some(false) {
        return None;
    }
    request
        .instructions
        .iter()
        .map(|instruction| instruction.content.as_slice())
        .chain(
            request
                .messages
                .iter()
                .map(|message| message.content.as_slice()),
        )
        .find_map(|content| unsupported_content(capabilities, content))
        .or_else(|| unsupported_preserved_content(capabilities, body))
}

/// Video and audio, refused unless the route states it can read the medium.
///
/// `vision` is a declaration about images and says nothing about video: a target that
/// accepts images need not accept video, and the providers behind the two differ. Audio
/// has no declaration at all, so no route can admit it.
fn unsupported_media(
    capabilities: ModelCapabilities,
    request: &LlmRequest,
    body: &Value,
) -> Option<&'static str> {
    if request_media(request, body, &|medium| medium == "audio").is_some() {
        return Some("audio");
    }
    if capabilities.video == Some(true) {
        return None;
    }
    request_media(request, body, &|medium| medium == "video")
}

/// Whether the request carries video.
///
/// Used where a target is uncharacterised. Vision is permitted by default there, because
/// absence of a fact must not invalidate a pre-existing deployment; video is not, because
/// no deployment carried video before video could be declared.
pub fn requires_video(request: &LlmRequest, body: &Value) -> bool {
    request_media(request, body, &|medium| medium == "video").is_some()
}

fn request_media(
    request: &LlmRequest,
    body: &Value,
    want: &dyn Fn(&str) -> bool,
) -> Option<&'static str> {
    request
        .instructions
        .iter()
        .map(|instruction| instruction.content.as_slice())
        .chain(
            request
                .messages
                .iter()
                .map(|message| message.content.as_slice()),
        )
        .find_map(|content| media_block(content, want))
        .or_else(|| preserved_media(body, want))
}

fn media_block(content: &[ContentBlock], want: &dyn Fn(&str) -> bool) -> Option<&'static str> {
    content.iter().find_map(|block| match block {
        ContentBlock::Video { .. } if want("video") => Some("video"),
        ContentBlock::Audio { .. } if want("audio") => Some("audio"),
        ContentBlock::ToolResult(result) => media_block(&result.content, want),
        _ => None,
    })
}

/// Look for media in fields that survive forwarding without becoming decoded content.
///
/// Two walks, because the evidence differs in strength. An explicit `type` discriminator is
/// unambiguous wherever it appears, so it may be followed into tool output. A bare key such
/// as `video` is only meaningful in caller-authored request shape: inside tool output it is
/// ordinary data, and refusing it would reject a text request that merely returned a field
/// named `video`. Neither walk enters tool *definitions*.
fn preserved_media(body: &Value, want: &dyn Fn(&str) -> bool) -> Option<&'static str> {
    const VIDEO: [&str; 3] = ["input_video", "video_url", "video"];
    const AUDIO: [&str; 2] = ["input_audio", "audio"];

    fn discriminant(value: &Value) -> &str {
        value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    fn roots(body: &Value) -> Vec<&Value> {
        ["messages", "input", "system"]
            .iter()
            .filter_map(|key| body.get(*key))
            .collect()
    }

    // Explicit discriminators. Tool output is followed only from a `*_call_output` part,
    // exactly as the image check does, so an unrelated nested object is never inspected.
    let mut pending = roots(body);
    while let Some(value) = pending.pop() {
        if let Some(items) = value.as_array() {
            pending.extend(items);
            continue;
        }
        let Some(object) = value.as_object() else {
            continue;
        };
        let kind = discriminant(value);
        if VIDEO.contains(&kind) && want("video") {
            return Some("video");
        }
        if AUDIO.contains(&kind) && want("audio") {
            return Some("audio");
        }
        pending.extend(object.get("content"));
        if kind.ends_with("_call_output") {
            pending.extend(object.get("output"));
        }
    }

    // Bare keys, on message and content-part objects only. Tracking the two kinds
    // separately is what keeps a media-named field in deeper data from being read as a
    // request shape: only elements of a messages/input/system array, and elements of a
    // `content` array, are inspected.
    let mut messages = roots(body);
    let mut parts: Vec<&Value> = Vec::new();
    loop {
        let object = match messages.pop() {
            Some(value) => match value.as_array() {
                Some(items) => {
                    messages.extend(items);
                    continue;
                }
                None => match value.as_object() {
                    Some(object) => object,
                    None => continue,
                },
            },
            None => match parts.pop() {
                Some(value) => match value.as_object() {
                    Some(object) => object,
                    None => continue,
                },
                None => break,
            },
        };
        if has_media_key(object, &VIDEO) && want("video") {
            return Some("video");
        }
        if has_media_key(object, &AUDIO) && want("audio") {
            return Some("audio");
        }
        if let Some(content) = object.get("content").and_then(Value::as_array) {
            parts.extend(content);
        }
    }
    None
}

fn has_media_key(object: &serde_json::Map<String, Value>, keys: &[&str]) -> bool {
    keys.iter()
        .any(|key| object.get(*key).is_some_and(|value| !value.is_null()))
}

/// Estimated input tokens for a request body.
///
/// Ported from the retired FleetRouter rather than re-derived, so an escalation decision
/// this deployment makes is comparable with the decisions the deployment it replaces made.
/// One estimator: a second, differently-tuned copy would silently move the boundary.
pub fn estimate_input_tokens(body: &Value) -> u64 {
    fn walk(value: &Value, acc: &mut u64) {
        match value {
            Value::String(text) => {
                let chars = text.chars().count() as u64;
                let ascii = text.bytes().filter(|byte| byte.is_ascii()).count() as u64;
                let non_ascii = chars.saturating_sub(ascii);
                *acc = acc
                    .saturating_add(ascii.div_ceil(4))
                    .saturating_add(non_ascii)
                    .saturating_add(1);
            }
            Value::Array(items) => items.iter().for_each(|item| walk(item, acc)),
            Value::Object(map) => map.values().for_each(|value| walk(value, acc)),
            _ => {}
        }
    }
    let mut total = 0u64;
    walk(body, &mut total);
    // Fixed structural overhead (model field, message roles, wrapper keys).
    total.saturating_add(8)
}

// Some provider fields survive forwarding without becoming decoded content blocks.
// Inspect protocol content and tool outputs, leaving tool arguments and text untouched.
fn unsupported_preserved_content(
    capabilities: ModelCapabilities,
    body: &Value,
) -> Option<&'static str> {
    let mut pending: Vec<_> = ["messages", "input", "system"]
        .iter()
        .filter_map(|key| body.get(key))
        .collect();
    while let Some(value) = pending.pop() {
        if let Some(items) = value.as_array() {
            pending.extend(items);
            continue;
        }
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if capabilities.vision == Some(false)
            && matches!(
                kind,
                "image" | "image_url" | "input_image" | "computer_screenshot"
            )
        {
            return Some("vision");
        }
        if capabilities.tool_calling == Some(false)
            && (matches!(
                kind,
                "additional_tools"
                    | "tool_use"
                    | "tool_result"
                    | "mcp_list_tools"
                    | "mcp_approval_request"
                    | "mcp_approval_response"
            ) || kind.ends_with("_call")
                || kind.ends_with("_call_output")
                || kind.ends_with("_tool_use")
                || kind.ends_with("_tool_result"))
        {
            return Some("tool_calling");
        }
        pending.extend(value.get("content"));
        if kind.ends_with("_call_output") {
            pending.extend(value.get("output"));
        }
    }
    None
}

/// Check decoded blocks, including images nested inside tool-result content.
fn unsupported_content(
    capabilities: ModelCapabilities,
    content: &[ContentBlock],
) -> Option<&'static str> {
    content.iter().find_map(|block| match block {
        ContentBlock::Image { .. } if capabilities.vision == Some(false) => Some("vision"),
        ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
            if capabilities.tool_calling == Some(false) =>
        {
            Some("tool_calling")
        }
        ContentBlock::ToolResult(result) => unsupported_content(capabilities, &result.content),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_translation::WireFormat::{AnthropicMessages, OpenAiChat, OpenAiResponses};
    use switchyard_translation::{WireFormat, decode_request, encode_request};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    // Decode each fixture so disabled, enabled, and unset cases use the same request.
    fn assert_requires_capability(
        format: WireFormat,
        body: &Value,
        disabled: ModelCapabilities,
        capability: &str,
    ) -> TestResult<LlmRequest> {
        let request = decode_request(format, body)?;
        for (capabilities, expected) in [
            (disabled, Some(capability)),
            (ModelCapabilities::default(), None),
            (
                ModelCapabilities {
                    tool_calling: Some(true),
                    reasoning: Some(true),
                    vision: Some(true),
                    ..Default::default()
                },
                None,
            ),
        ] {
            assert_eq!(
                unsupported_capability(capabilities, &request, body),
                expected,
                "{body}"
            );
        }
        Ok(request)
    }

    // Tool controls and history, including MCP approvals, can survive forwarding
    // without appearing in decoded tools.
    #[test]
    fn rejects_tool_controls_and_history_preserved_outside_normalized_tools() -> TestResult {
        let capabilities = ModelCapabilities {
            tool_calling: Some(false),
            ..Default::default()
        };
        for body in [
            json!({"input": "hello", "tools": [{"type": "web_search"}]}),
            json!({"input": "hello", "functions": [{"name": "lookup", "parameters": {"type": "object"}}]}),
            json!({"input": "hello", "tool_choice": "none"}),
            json!({"input": "hello", "tool_choice": {"type": "web_search"}}),
            json!({"input": "hello", "parallel_tool_calls": false}),
            json!({"input": [{"type": "additional_tools", "tools": []}]}),
            json!({"input": [{"type": "shell_call", "id": "shell_1", "call_id": "call_1", "action": {"commands": ["pwd"]}}]}),
            json!({"input": [{"type": "function_call_output", "call_id": "call_1", "output": "done"}]}),
            json!({"input": [{"type": "mcp_list_tools", "server_label": "server", "tools": []}]}),
            json!({"input": [{"type": "mcp_approval_request", "id": "approval_1", "name": "lookup", "arguments": "{}", "server_label": "server"}]}),
            json!({"input": [{"type": "mcp_approval_response", "approval_request_id": "approval_1", "approve": true}]}),
            json!({"input": [{"type": "mcp_approval_response", "approval_request_id": "approval_1", "approve": false}]}),
        ] {
            assert_requires_capability(OpenAiResponses, &body, capabilities, "tool_calling")?;
        }
        let body = json!({"messages": [{"role": "user", "content": "hello"}], "tools": [], "functions": []});
        let request = decode_request(OpenAiChat, &body)?;
        assert_eq!(unsupported_capability(capabilities, &request, &body), None);
        Ok(())
    }

    // Preserved reasoning controls still reach the provider when decoding omits them.
    #[test]
    fn rejects_preserved_reasoning_controls() -> TestResult {
        let capabilities = ModelCapabilities {
            reasoning: Some(false),
            ..Default::default()
        };
        for (format, body, pointer) in [
            (
                OpenAiChat,
                json!({"messages": [{"role": "user", "content": "hello"}], "reasoning": {"enabled": true}}),
                "/reasoning",
            ),
            (
                OpenAiChat,
                json!({"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": 123}),
                "/reasoning_effort",
            ),
            (
                AnthropicMessages,
                json!({"messages": [{"role": "user", "content": "hello"}], "output_config": {"effort": 123}}),
                "/output_config/effort",
            ),
        ] {
            let request = assert_requires_capability(format, &body, capabilities, "reasoning")?;
            assert_eq!(
                encode_request(&request, format)?.pointer(pointer),
                body.pointer(pointer)
            );
        }
        Ok(())
    }

    // Decoded file-ID images and preserved computer screenshots both require vision.
    #[test]
    fn rejects_file_id_images_and_computer_screenshots() -> TestResult {
        let capabilities = ModelCapabilities {
            vision: Some(false),
            ..Default::default()
        };
        for body in [
            json!({"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file_1"}]}]}),
            json!({"input": [{"type": "computer_call_output", "call_id": "call_1", "output": {"type": "computer_screenshot", "image_url": "https://example.test/image.png"}}]}),
        ] {
            let request =
                assert_requires_capability(OpenAiResponses, &body, capabilities, "vision")?;
            assert_eq!(
                encode_request(&request, OpenAiResponses)?["input"],
                body["input"]
            );
        }
        Ok(())
    }

    // Allowing tool results must not hide unsupported images nested in their content.
    #[test]
    fn rejects_images_inside_tool_results_when_tools_are_allowed() -> TestResult {
        let body = json!({"messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_1", "content": [{
                "type": "image", "source": {"type": "url", "url": "https://example.test/image.png"}
            }]
        }]}]});
        assert_requires_capability(
            AnthropicMessages,
            &body,
            ModelCapabilities {
                tool_calling: Some(true),
                vision: Some(false),
                ..Default::default()
            },
            "vision",
        )?;
        Ok(())
    }

    /// `vision` is a declaration about images. Enabling it, or leaving every capability
    /// undeclared, must not admit video or audio: video needs its own declaration and audio
    /// has none, so both are refused before any target sees them.
    #[test]
    fn media_is_refused_without_a_video_declaration() -> TestResult {
        let cases = [
            (
                OpenAiChat,
                "video",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}}
                ]}]}),
            ),
            (
                OpenAiChat,
                "audio",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
                ]}]}),
            ),
            (
                OpenAiResponses,
                "video",
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
                ]}]}),
            ),
            (
                OpenAiResponses,
                "audio",
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
                ]}]}),
            ),
            (
                AnthropicMessages,
                "video",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "video", "source": {"type": "url", "url": "https://example.test/clip.mp4"}}
                ]}]}),
            ),
            (
                AnthropicMessages,
                "audio",
                json!({"messages": [{"role": "user", "content": [
                    {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "AAAA"}}
                ]}]}),
            ),
        ];
        let permissive = ModelCapabilities {
            tool_calling: Some(true),
            reasoning: Some(true),
            vision: Some(true),
            ..Default::default()
        };
        for (format, capability, body) in cases {
            let request = decode_request(format, &body)?;
            for capabilities in [ModelCapabilities::default(), permissive] {
                assert_eq!(
                    unsupported_capability(capabilities, &request, &body),
                    Some(capability),
                    "{capability} must be refused regardless of `vision`: {body}"
                );
            }
        }
        Ok(())
    }

    /// Plain text is unaffected by the media rule.
    #[test]
    fn text_is_admitted_after_the_media_rule() -> TestResult {
        let body = json!({"messages": [{"role": "user", "content": "hello"}]});
        let request = decode_request(OpenAiChat, &body)?;
        assert_eq!(
            unsupported_capability(ModelCapabilities::default(), &request, &body),
            None
        );
        Ok(())
    }

    /// Media can survive forwarding as a key beside `content`, with no `type` to match.
    #[test]
    fn media_preserved_as_a_bare_key_is_refused() -> TestResult {
        let video = json!({"messages": [{"role": "user", "content": "hello",
            "video": {"url": "https://example.test/clip.mp4"}}]});
        let audio = json!({"input": [{"role": "user", "content": "hello",
            "audio": {"data": "AAAA", "format": "wav"}}]});
        for (format, body, capability) in [
            (OpenAiChat, video, "video"),
            (OpenAiResponses, audio, "audio"),
        ] {
            let request = decode_request(format, &body)?;
            assert_eq!(
                unsupported_capability(ModelCapabilities::default(), &request, &body),
                Some(capability),
                "{body}"
            );
        }
        Ok(())
    }

    /// A tool schema may legitimately declare a property named `video`. Refusing that would
    /// reject an ordinary text request, so tool definitions are deliberately not walked.
    #[test]
    fn a_tool_schema_naming_video_is_not_treated_as_media() -> TestResult {
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {
                "name": "probe",
                "parameters": {"type": "object", "properties": {
                    "video": {"type": "string"},
                    "audio": {"type": "string"}
                }}
            }}]
        });
        let request = decode_request(OpenAiChat, &body)?;
        assert_eq!(
            unsupported_capability(ModelCapabilities::default(), &request, &body),
            None,
            "a tool parameter named video must not reject a text request"
        );
        Ok(())
    }

    /// A field named `video` inside tool output is data the tool chose to return, not the
    /// caller sending media. Refusing it would break an ordinary text request.
    #[test]
    fn media_named_fields_inside_tool_output_are_not_request_media() -> TestResult {
        let body = json!({
            "input": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"video": "clip.mp4", "audio": "note.wav"}
            }]
        });
        let request = decode_request(OpenAiResponses, &body)?;
        assert_eq!(
            unsupported_capability(ModelCapabilities::default(), &request, &body),
            None,
            "tool output data must not be read as caller-supplied media"
        );
        Ok(())
    }

    /// Media-named data on an object that is not a message, a content part, or a
    /// `*_call_output` must not be read as request media either.
    #[test]
    fn media_named_data_in_unrelated_nesting_is_not_request_media() -> TestResult {
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"output": {"type": "audio"}}
        });
        let request = decode_request(OpenAiChat, &body)?;
        assert_eq!(
            unsupported_capability(ModelCapabilities::default(), &request, &body),
            None,
            "only request shape and *_call_output output may be inspected"
        );
        Ok(())
    }

    /// Video is admitted only where the route states it can read video. `vision` describes
    /// images and is not evidence of video support.
    #[test]
    fn video_is_admitted_only_by_an_explicit_declaration() -> TestResult {
        let bodies = [
            (
                OpenAiChat,
                json!({"messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}}
                ]}]}),
            ),
            (
                OpenAiResponses,
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
                ]}]}),
            ),
            // Survives forwarding as a bare key, with no `type` to match.
            (
                OpenAiChat,
                json!({"messages": [{"role": "user", "content": "look",
                    "video": {"url": "https://example.test/clip.mp4"}}]}),
            ),
        ];
        for (format, body) in bodies {
            let request = decode_request(format, &body)?;
            let vision_only = ModelCapabilities {
                vision: Some(true),
                ..Default::default()
            };
            assert_eq!(
                unsupported_capability(vision_only, &request, &body),
                Some("video"),
                "vision must not imply video support: {body}"
            );
            let declared = ModelCapabilities {
                vision: Some(false),
                video: Some(true),
                ..Default::default()
            };
            assert_eq!(
                unsupported_capability(declared, &request, &body),
                None,
                "an explicit video declaration admits it even without image support: {body}"
            );
        }
        Ok(())
    }

    /// Audio has no declaration at all, so a route declaring video must not widen into it.
    #[test]
    fn audio_is_refused_even_when_video_is_declared() -> TestResult {
        let body = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
        ]}]});
        let request = decode_request(OpenAiResponses, &body)?;
        assert_eq!(
            unsupported_capability(
                ModelCapabilities {
                    video: Some(true),
                    ..Default::default()
                },
                &request,
                &body
            ),
            Some("audio"),
            "a video declaration must not admit audio"
        );
        Ok(())
    }

    /// The uncharacterised-target guard uses the same reading of the request.
    #[test]
    fn requires_video_matches_the_admission_rule() -> TestResult {
        let video = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
        ]}]});
        let audio = json!({"input": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
        ]}]});
        let text = json!({"input": "hello"});
        for (body, expected) in [(video, true), (audio, false), (text, false)] {
            let request = decode_request(OpenAiResponses, &body)?;
            assert_eq!(requires_video(&request, &body), expected, "{body}");
        }
        Ok(())
    }

    /// Video survives the decode/encode round trip into the provider spelling.
    #[test]
    fn video_round_trips_to_the_responses_part() -> TestResult {
        let body = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4"},
            {"type": "input_text", "text": "describe"}
        ]}]});
        let request = decode_request(OpenAiResponses, &body)?;
        let encoded = encode_request(&request, OpenAiResponses)?;
        let parts = encoded
            .pointer("/input/0/content")
            .and_then(Value::as_array)
            .ok_or("missing content")?;
        assert_eq!(parts[0]["type"], "input_video");
        assert_eq!(parts[0]["video_url"], "https://example.test/clip.mp4");
        Ok(())
    }
}
