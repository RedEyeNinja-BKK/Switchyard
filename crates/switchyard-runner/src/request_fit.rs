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

use crate::{MediaCarrier, ModelCapabilities};
use serde_json::Value;
use switchyard_protocol::{ContentBlock, LlmRequest, MediaSource, Role};

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
    if capabilities.tool_calling == Some(false) && wants_tools(request, body) {
        return Some("tool_calling");
    }
    // Narrower than `tool_calling`, and separate because the two answers differ: a surface
    // can accept tool definitions and still be unable to continue a transaction. Google's
    // OpenAI-compatibility endpoint requires a provider-issued signature on a replayed
    // function call and refuses the request without one, unless the target also declares the
    // documented placeholder for foreign calls; a request carrying tool lineage must not be
    // sent to a target that states neither.
    if capabilities.tool_call_replay == Some(false) && replays_tool_call(request, body) {
        return Some("tool_call_replay");
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

/// Whether the request carries tools or asks for a tool call.
///
/// One predicate for two readers: the capability gate refuses a target that cannot carry
/// them, and selection costs the conversation as one that will grow. Reading the decoded
/// request and the preserved body both, because a caller may send either spelling and a
/// target must not be selected on one of them only.
pub fn wants_tools(request: &LlmRequest, body: &Value) -> bool {
    !request.tools.is_empty()
        || request.tool_choice.is_some()
        || ["tools", "functions"].iter().any(|key| {
            body.get(key)
                .and_then(Value::as_array)
                .is_some_and(|tools| !tools.is_empty())
        })
        || ["tool_choice", "parallel_tool_calls", "function_call"]
            .iter()
            .any(|key| body.get(key).is_some_and(|value| !value.is_null()))
}

/// Whether the request carries a tool call made earlier in the conversation.
///
/// This is the continuation half of a tool transaction, and it is what a signature-bound
/// wire cannot accept: the assistant's call being sent back, or the result that answers one,
/// or both. Offering tools is NOT lineage — the definitional call is a fresh request that
/// needs no provider signature, so a target excluded here stays eligible there.
///
/// Read from the decoded request and from the preserved body, like [`wants_tools`]: a caller
/// can spell the same fact in a field decoding does not normalize (`function_call`, a
/// Responses `function_call` item), and a target must not be selected on one spelling only.
pub fn replays_tool_call(request: &LlmRequest, body: &Value) -> bool {
    let decoded = request.messages.iter().any(|message| {
        message.role == Role::Tool
            || message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)))
    });
    decoded || preserved_tool_lineage(body)
}

/// Tool lineage as the raw body spells it, including the legacy chat fields.
fn preserved_tool_lineage(body: &Value) -> bool {
    let chat = body
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                let role = message
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                role == "tool"
                    || message
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|calls| !calls.is_empty())
                    || message
                        .get("function_call")
                        .is_some_and(|call| !call.is_null())
            })
        });
    if chat {
        return true;
    }
    // Responses: a call item or the output answering one. `ends_with` covers the function,
    // custom, shell, file and MCP spellings without enumerating a vocabulary that grows.
    body.get("input")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                item.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| {
                        kind.ends_with("_call")
                            || kind.ends_with("_call_output")
                            || kind.ends_with("_tool_use")
                            || kind.ends_with("_tool_result")
                    })
            })
        })
}

/// Video and audio, refused unless the route states it can read the medium.
///
/// `vision` is a declaration about images and says nothing about video: a target that
/// accepts images need not accept video, and the providers behind the two differ. Audio
/// has no declaration at all, so no route can admit it.
///
/// When video is declared, the carrier is checked as well: an endpoint that reads inline
/// media but cannot fetch a URL must not receive a `video_url` part, or the provider answers
/// the request with a 404 the caller cannot act on. A route that declares video without a
/// carrier means both, which is what a deployment written before carriers existed intends.
fn unsupported_media(
    capabilities: ModelCapabilities,
    request: &LlmRequest,
    body: &Value,
) -> Option<&'static str> {
    if request_media(request, body, &|medium| medium == "audio").is_some() {
        return Some("audio");
    }
    // Each medium's carrier is checked against that medium's own blocks. Narrowing video
    // must not disturb an image request the same endpoint serves, so the two never read
    // each other's parts: `image_url` is not a video carrier fact and `video_url` is not an
    // image one.
    if let Some(carrier) = capabilities.image_carrier
        && !carrier.admits_remote()
        && request_image_is_remote(request, body)
    {
        return Some("image_carrier");
    }
    if let Some(carrier) = capabilities.image_carrier
        && !carrier.admits_inline()
        && request_image_is_inline(request, body)
    {
        return Some("image_carrier");
    }
    if capabilities.video != Some(true) {
        return request_media(request, body, &|medium| medium == "video");
    }
    let carrier = capabilities.video_carrier.unwrap_or(MediaCarrier::Both);
    if !carrier.admits_remote() && request_video_is_remote(request, body) {
        return Some("video_carrier");
    }
    if !carrier.admits_inline() && request_video_is_inline(request, body) {
        return Some("video_carrier");
    }
    None
}

/// Whether this request's IMAGE is supplied as a URL the provider must fetch.
fn request_image_is_remote(request: &LlmRequest, body: &Value) -> bool {
    // An `image_url` part is the carrier wrapper for BOTH forms: the decoder keeps a
    // `data:` URL inside `ImageSource::Url`, so the test is the URL itself, not the
    // variant. Classifying by variant would refuse every image an inline-only endpoint
    // actually serves.
    image_blocks(request).any(|block| match block {
        ContentBlock::Image {
            source: switchyard_protocol::ImageSource::Url { url, .. },
        } => !is_data_url(url),
        ContentBlock::Image {
            source: switchyard_protocol::ImageSource::Base64 { .. },
        } => false,
        _ => false,
    }) || preserved_image_is_remote(body)
}

/// Whether this request's IMAGE is supplied inline, as base64 bytes or a `data:` URL.
fn request_image_is_inline(request: &LlmRequest, body: &Value) -> bool {
    image_blocks(request).any(|block| match block {
        ContentBlock::Image {
            source: switchyard_protocol::ImageSource::Base64 { .. },
        } => true,
        ContentBlock::Image {
            source: switchyard_protocol::ImageSource::Url { url, .. },
        } => is_data_url(url),
        _ => false,
    }) || preserved_image_is_inline(body)
}

/// Every decoded image block, including one nested in tool-result content.
fn image_blocks(request: &LlmRequest) -> impl Iterator<Item = &ContentBlock> {
    fn walk<'a>(content: &'a [ContentBlock], out: &mut Vec<&'a ContentBlock>) {
        for block in content {
            match block {
                ContentBlock::Image { .. } => out.push(block),
                ContentBlock::ToolResult(result) => walk(&result.content, out),
                _ => {}
            }
        }
    }
    let mut out: Vec<&ContentBlock> = Vec::new();
    for content in request
        .instructions
        .iter()
        .map(|instruction| instruction.content.as_slice())
        .chain(
            request
                .messages
                .iter()
                .map(|message| message.content.as_slice()),
        )
    {
        walk(content, &mut out);
    }
    out.into_iter()
}

/// The image-shaped discriminators of a preserved body.
fn preserved_image_blocks(body: &Value) -> impl Iterator<Item = &Value> {
    preserved_shaped_blocks(body, &["input_image", "image_url", "computer_screenshot"])
}

/// Whether a preserved body's image is a URL the provider must fetch.
fn preserved_image_is_remote(body: &Value) -> bool {
    preserved_image_blocks(body).any(|block| {
        let Some(object) = block.as_object() else {
            return false;
        };
        match object.get("image_url") {
            Some(Value::String(url)) => !is_data_url(url),
            Some(Value::Object(nested)) => nested
                .get("url")
                .and_then(Value::as_str)
                .is_some_and(|url| !is_data_url(url)),
            _ => false,
        }
    })
}

/// Whether a preserved body's image is inline bytes or a `data:` URL.
fn preserved_image_is_inline(body: &Value) -> bool {
    preserved_image_blocks(body).any(|block| {
        let Some(object) = block.as_object() else {
            return false;
        };
        object
            .get("image_url")
            .map(|value| match value {
                Value::String(url) => is_data_url(url),
                Value::Object(nested) => nested
                    .get("url")
                    .and_then(Value::as_str)
                    .is_some_and(is_data_url),
                _ => false,
            })
            .unwrap_or(false)
    })
}

/// Whether this request's video is supplied as a URL the provider must fetch.
///
/// Only `ContentBlock::Video` counts: the question is which carrier a *video* takes, and an
/// image part is a separate capability the endpoint may serve on either carrier.
fn request_video_is_remote(request: &LlmRequest, body: &Value) -> bool {
    video_blocks(request).any(|block| match block {
        ContentBlock::Video {
            source: MediaSource::Url { url, .. },
        } => !is_data_url(url),
        ContentBlock::Video {
            source: MediaSource::Base64 { .. },
        } => false,
        _ => false,
    }) || preserved_video_is_remote(body)
}

/// Whether this request's video is supplied inline, as base64 bytes or a `data:` URL.
fn request_video_is_inline(request: &LlmRequest, body: &Value) -> bool {
    video_blocks(request).any(|block| match block {
        ContentBlock::Video {
            source: MediaSource::Base64 { .. },
        } => true,
        ContentBlock::Video {
            source: MediaSource::Url { url, .. },
        } => is_data_url(url),
        _ => false,
    }) || preserved_video_is_inline(body)
}

/// Every decoded video block, including one nested in tool-result content.
fn video_blocks(request: &LlmRequest) -> impl Iterator<Item = &ContentBlock> {
    fn walk<'a>(content: &'a [ContentBlock], out: &mut Vec<&'a ContentBlock>) {
        for block in content {
            match block {
                ContentBlock::Video { .. } => out.push(block),
                ContentBlock::ToolResult(result) => walk(&result.content, out),
                _ => {}
            }
        }
    }
    let mut out: Vec<&ContentBlock> = Vec::new();
    for content in request
        .instructions
        .iter()
        .map(|instruction| instruction.content.as_slice())
        .chain(
            request
                .messages
                .iter()
                .map(|message| message.content.as_slice()),
        )
    {
        walk(content, &mut out);
    }
    out.into_iter()
}

/// Whether a URL is an inline `data:` payload rather than a fetch target.
fn is_data_url(url: &str) -> bool {
    let bytes = url.as_bytes();
    bytes.len() >= 5 && bytes[..5].eq_ignore_ascii_case(b"data:")
}

/// The video-shaped discriminators of a preserved body, read from the same roots the
/// admission walk uses so the two cannot disagree about what the request carries.
fn preserved_video_blocks(body: &Value) -> impl Iterator<Item = &Value> {
    preserved_shaped_blocks(body, &["input_video", "video_url", "video"])
}

/// Objects in a preserved body whose `type` is one of `kinds`, from the roots the admission
/// walk reads. One walker for both media, so a change to where media hides cannot make the
/// image check and the video check disagree.
fn preserved_shaped_blocks<'a>(body: &'a Value, kinds: &[&str]) -> impl Iterator<Item = &'a Value> {
    let mut pending: Vec<&'a Value> = ["messages", "input", "system"]
        .iter()
        .filter_map(|key| body.get(*key))
        .collect();
    let mut found: Vec<&'a Value> = Vec::new();
    while let Some(value) = pending.pop() {
        if let Some(items) = value.as_array() {
            pending.extend(items);
            continue;
        }
        let Some(object) = value.as_object() else {
            continue;
        };
        if object
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kinds.contains(&kind))
        {
            found.push(value);
        }
        if let Some(content) = object.get("content") {
            pending.push(content);
        }
    }
    found.into_iter()
}

/// Whether a preserved body's video is a URL the provider must fetch.
fn preserved_video_is_remote(body: &Value) -> bool {
    preserved_video_blocks(body).any(|block| {
        let Some(object) = block.as_object() else {
            return false;
        };
        match object.get("video_url") {
            Some(Value::String(url)) => !is_data_url(url),
            Some(Value::Object(nested)) => nested
                .get("url")
                .and_then(Value::as_str)
                .is_some_and(|url| !is_data_url(url)),
            _ => false,
        }
    })
}

/// Whether a preserved body's video is inline bytes or a `data:` URL.
fn preserved_video_is_inline(body: &Value) -> bool {
    preserved_video_blocks(body).any(|block| {
        let Some(object) = block.as_object() else {
            return false;
        };
        if let Some(url) = object.get("video_url").and_then(Value::as_str) {
            return is_data_url(url);
        }
        object
            .get("video")
            .and_then(Value::as_object)
            .is_some_and(|video| {
                video.get("data").is_some()
                    || video
                        .get("url")
                        .and_then(Value::as_str)
                        .is_some_and(is_data_url)
            })
    })
}

/// Whether the request carries video.
///
/// Used where a target is uncharacterised. Vision is permitted by default there, because
/// absence of a fact must not invalidate a pre-existing deployment; video is not, because
/// no deployment carried video before video could be declared.
pub fn requires_video(request: &LlmRequest, body: &Value) -> bool {
    request_media(request, body, &|medium| medium == "video").is_some()
}

/// A video part the OpenAI Responses wire cannot carry, described for the caller.
///
/// Measured against the provider's own `/responses` endpoint: video is accepted only as
/// `{"type":"input_video","video_url":"<https URL or data: URI>"}` with a STRING
/// `video_url`. The Chat spelling (`video_url` as an object) and inline bytes under
/// `video` are each answered `400 invalid_prompt` ("invalid_union, expected string").
///
/// This matters here because a same-format request is forwarded verbatim, by design, so a
/// caller's unsupported part reaches the provider unchanged and comes back as an opaque
/// upstream error. Detecting it before dispatch turns that into an accurate local refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoPartShape {
    /// The `type` the caller used, quoted back so the message names what was sent.
    pub part_type: String,
    /// What was wrong with that part, in the caller's vocabulary.
    pub detail: &'static str,
}

/// Rewrites video parts on the Responses surface into the one spelling the provider accepts.
///
/// The accepted spelling is `{"type":"input_video","video_url":"<https URL or data: URI>"}`
/// with a string. A caller may reasonably send the Chat spelling instead -- `video_url` as
/// the part type, or `video_url` as an object -- and Turnstone's own Responses client does
/// exactly that. Refusing those would make video unreachable from the client rather than
/// from the provider, so the shape is translated instead: only the spelling changes, and a
/// caller-supplied media type rides along rather than being dropped.
///
/// Shapes that cannot be mapped faithfully (no URL, no bytes) are left untouched for
/// [`unsupported_responses_video_part`] to refuse with a reason.
pub fn normalize_responses_video_parts(body: &mut Value) -> usize {
    fn data_uri(media_type: Option<&str>, data: &str) -> Option<String> {
        media_type.map(|media_type| format!("data:{media_type};base64,{data}"))
    }

    fn normalize_part(part: &mut Value) -> bool {
        let Some(object) = part.as_object_mut() else {
            return false;
        };
        let part_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if part_type != "input_video" && part_type != "video_url" {
            return false;
        }
        // `video_url` as an object: either a URL wrapper or the same shape Chat uses.
        if let Some(Value::Object(nested)) = object.get("video_url").cloned() {
            if let Some(url) = nested
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                object.insert("type".to_string(), Value::String("input_video".to_string()));
                object.insert("video_url".to_string(), Value::String(url));
                if let Some(media_type) = nested.get("media_type").and_then(Value::as_str)
                    && !object.contains_key("media_type")
                {
                    object.insert(
                        "media_type".to_string(),
                        Value::String(media_type.to_string()),
                    );
                }
                return true;
            }
            if let Some(uri) = nested
                .get("data")
                .and_then(Value::as_str)
                .and_then(|data| data_uri(nested.get("media_type").and_then(Value::as_str), data))
            {
                object.insert("type".to_string(), Value::String("input_video".to_string()));
                object.insert("video_url".to_string(), Value::String(uri));
                return true;
            }
        }
        // Inline bytes under `video`, which is how the Chat decoder spells them.
        if let Some(uri) = object
            .get("video")
            .and_then(Value::as_object)
            .and_then(|video| {
                video.get("data").and_then(Value::as_str).and_then(|data| {
                    data_uri(video.get("media_type").and_then(Value::as_str), data)
                })
            })
        {
            object.insert("type".to_string(), Value::String("input_video".to_string()));
            object.insert("video_url".to_string(), Value::String(uri));
            object.remove("video");
            return true;
        }
        // Already a string URL under the right type: only the type may still need fixing.
        if part_type == "video_url" && object.get("video_url").and_then(Value::as_str).is_some() {
            object.insert("type".to_string(), Value::String("input_video".to_string()));
            return true;
        }
        false
    }

    let Some(input) = body.get_mut("input") else {
        return 0;
    };
    let mut rewritten = 0;
    let items: Vec<&mut Value> = match input {
        Value::Array(items) => items.iter_mut().collect(),
        other => vec![other],
    };
    for item in items {
        if let Some(content) = item.get_mut("content").and_then(Value::as_array_mut) {
            for part in content.iter_mut() {
                rewritten += usize::from(normalize_part(part));
            }
            continue;
        }
        // A part may also appear directly as an input item.
        rewritten += usize::from(normalize_part(item));
    }
    rewritten
}

/// The first video part in a Responses body that the provider cannot parse, if any.
///
/// Only the shapes that were measured to fail are refused; a part already in the accepted
/// spelling, or one carrying a provider `file_id` this deployment cannot verify, is left
/// alone rather than guessed at.
pub fn unsupported_responses_video_part(body: &Value) -> Option<VideoPartShape> {
    preserved_shaped_blocks(body, &["input_video", "video_url", "video"]).find_map(|block| {
        let object = block.as_object()?;
        let part_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if part_type != "input_video" {
            return Some(VideoPartShape {
                part_type,
                detail: "this surface accepts video only as an `input_video` part",
            });
        }
        match object.get("video_url") {
            Some(Value::String(_)) => None,
            Some(_) => Some(VideoPartShape {
                part_type,
                detail: "`video_url` must be a string here, not an object",
            }),
            None if object.get("file_id").and_then(Value::as_str).is_some() => None,
            None if object.contains_key("video") => Some(VideoPartShape {
                part_type,
                detail: "inline bytes belong in a `data:` URI supplied as the string `video_url`",
            }),
            None => Some(VideoPartShape {
                part_type,
                detail: "the part carries no `video_url`",
            }),
        }
    })
}

/// Whether a content slice holds a media block supplied as a URL.
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

/// The part types that carry a picture. This is the one vocabulary: the capability gate
/// (`unsupported_preserved_content`) refuses exactly these types for a target without
/// vision, so the size gate and the capability gate cannot disagree about what an image is.
const IMAGE_PART_TYPES: &[&str] = &["image", "image_url", "input_image", "computer_screenshot"];

/// Tokens charged for one image part.
///
/// An image is a picture, not prose. Charging its base64 payload at the text rate priced a
/// 361 KB PNG at 120,425 tokens, which pushed a request that actually cost tens of thousands
/// of tokens past a 262,144-token window and refused it as `requires context_window`.
///
/// One figure for every vision leaf, not a per-model table: the provider is not known when
/// this gate runs, because a candidate is chosen after admission. It is set above every
/// documented per-image cost of the families these routes can select, so the estimate never
/// under-admits an image on a rule that only holds for one of them:
///
/// | family | documented rule | tokens |
/// |---|---|---|
/// | Google Gemini | 258 per 768x768 tile, four tiles after the 1,536x1,536 downscale | 1,032 |
/// | OpenAI | `detail: high`, 85 + 170 per 512x512 tile, six tiles at an extreme aspect | 1,105 |
/// | OpenAI | `detail: low` | 85 |
/// | Anthropic | (width * height) / 750 at the 1,568-long-side cap | 3,279 |
///
/// Leaves with no published per-image rule -- the free vision lanes, local engines -- are
/// charged the same figure.
///
/// That last sentence is an operational heuristic, not a proof: for an undocumented family
/// there is no published maximum and therefore no bounded error. The figure is chosen to be
/// conservative -- above every documented ceiling, and small against the windows these routes
/// declare (four images are 1/16 of a 262,144 window) -- so that being wrong refuses a
/// request that would have fitted rather than admitting one the provider then rejects. Only
/// the second failure is silent, and only it is treated as acceptable.
const IMAGE_TOKENS: u64 = 4_096;

/// Where a value sits in the request. Only a protocol content part can be a picture; the
/// rest of the body is data, however much it looks like a part.
///
/// The raw body is forwarded verbatim, so it is full of near-misses: a tool schema property
/// named `source`, a JSON-schema `"type": "image"`, a metadata object with an `image_url`
/// key. Pricing any of those as a picture charges 4,096 tokens and then *skips* a subtree
/// that may hold kilobytes of prose -- the same under-accounting class as reading a payload
/// as text, in the opposite direction.
///
/// The three slots below mirror the traversal `preserved_media` already uses for the same
/// reason: an explicit discriminator is only meaningful where the protocol puts parts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// The request body itself. Only `messages`, `input` and `system` open a message slot.
    Root,
    /// A message or input item: its `content` holds parts, and a tool-output item's `output`
    /// does too.
    Message,
    /// An element of a protocol content array: the only place a media part may appear.
    Part,
    /// Everything else -- tool definitions, tool arguments, metadata, arbitrary data.
    Data,
}

/// Whether `key` opens a content-part slot in an object at `slot`.
///
/// `content` is a message's own content and also a tool-result block's nested content, so it
/// opens parts from both. `output` is only a part slot on an item whose own type says it is a
/// call output or tool result; the same key on an unrelated object is ordinary data.
fn opens_parts(key: &str, map: &serde_json::Map<String, Value>, slot: Slot) -> bool {
    if !matches!(slot, Slot::Message | Slot::Part) {
        return false;
    }
    match key {
        "content" => true,
        "output" => map
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind.ends_with("_output") || kind.ends_with("_result")),
        _ => false,
    }
}

/// Whether this object names itself as a media part carrying a picture.
///
/// A declared type is the discriminator, and only the vocabulary the capability gate refuses
/// for a vision-less target counts. An untyped object must carry an unambiguous image
/// reference: a bare `source` does not qualify, because audio, documents and provider
/// metadata all use that key, so an untyped `source` keeps text accounting.
fn is_image_part(map: &serde_json::Map<String, Value>) -> bool {
    match map.get("type").and_then(Value::as_str) {
        Some(kind) => IMAGE_PART_TYPES.contains(&kind),
        None => has_image_url(map.get("image_url")),
    }
}

/// Whether an `image_url` value carries a usable location: a string (Responses) or an object
/// with a non-empty `url` (chat). An empty or absent one is not a picture.
fn has_image_url(value: Option<&Value>) -> bool {
    match value {
        Some(Value::String(url)) => !url.is_empty(),
        Some(Value::Object(object)) => object
            .get("url")
            .and_then(Value::as_str)
            .is_some_and(|url| !url.is_empty()),
        _ => false,
    }
}

/// Whether this field of a confirmed image part is the picture itself rather than prose.
///
/// Only the payload is skipped, and only when its shape says bytes or a location. A `source`
/// object holding a description is walked, because skipping it would hide real text. Metadata
/// fields (`detail`, `mime_type`, `media_type`, `format`) are small and are simply counted.
fn is_image_payload(key: &str, value: &Value) -> bool {
    match key {
        "image_url" => has_image_url(Some(value)),
        "source" => value.as_object().is_some_and(|source| {
            source.get("data").is_some_and(Value::is_string)
                || source.get("url").is_some_and(Value::is_string)
                || source.contains_key("file_id")
        }),
        "data" | "b64_json" | "base64" => value.is_string(),
        // A bare `url` is payload only when it carries the bytes itself; a location is
        // short and is counted as text like any other short string.
        "url" => value.as_str().is_some_and(|url| url.starts_with("data:")),
        _ => false,
    }
}

/// Estimated input tokens for a request body.
///
/// Ported from the retired FleetRouter rather than re-derived, so an escalation decision
/// this deployment makes is comparable with the decisions the deployment it replaces made.
/// One estimator: a second, differently-tuned copy would silently move the boundary.
pub fn estimate_input_tokens(body: &Value) -> u64 {
    fn walk(value: &Value, acc: &mut u64, slot: Slot) {
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
            // An array keeps its slot: the elements of `messages` are messages, and the
            // elements of a `content` array are parts. A string in a part slot is a
            // text-only content and is counted by the arm above.
            Value::Array(items) => items
                .iter()
                .for_each(|item| walk(item, acc, slot)),
            Value::Object(map) if slot == Slot::Part && is_image_part(map) => {
                *acc = acc.saturating_add(IMAGE_TOKENS);
                for (key, child) in map {
                    if !is_image_payload(key, child) {
                        walk(child, acc, Slot::Data);
                    }
                }
            }
            Value::Object(map) => {
                for (key, child) in map {
                    let next = if slot == Slot::Root {
                        match key.as_str() {
                            "messages" | "input" => Slot::Message,
                            // `system` is an array of content blocks (Anthropic), not of
                            // messages, so its elements are parts.
                            "system" => Slot::Part,
                            _ => Slot::Data,
                        }
                    } else if opens_parts(key, map, slot) {
                        Slot::Part
                    } else {
                        Slot::Data
                    };
                    walk(child, acc, next);
                }
            }
            _ => {}
        }
    }
    let mut total = 0u64;
    walk(body, &mut total, Slot::Root);
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
        // The same vocabulary the estimator prices, so the two gates cannot disagree.
        if capabilities.vision == Some(false) && IMAGE_PART_TYPES.contains(&kind) {
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
            json!({"input": [
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "done"}
            ]}),
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

    // A target that cannot carry a tool CONTINUATION is excluded only from requests that
    // replay one. Its definitional call is still served, which is what keeps Google in the
    // ladder for every request that merely offers tools.
    #[test]
    fn a_target_that_cannot_replay_a_tool_call_still_serves_its_definition() -> TestResult {
        let cannot_replay = ModelCapabilities {
            tool_call_replay: Some(false),
            ..Default::default()
        };
        let definition = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        });
        let replay = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "18C and raining"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        });
        for (label, body, expected) in [
            ("definitional call", definition, None),
            ("replayed call", replay, Some("tool_call_replay")),
        ] {
            let request = decode_request(OpenAiChat, &body)?;
            assert_eq!(
                unsupported_capability(cannot_replay, &request, &body),
                expected,
                "{label} on a target that cannot replay: {body}"
            );
            // Undeclared and explicitly-capable targets keep serving both, so the flag
            // cannot leak into a deployment that never states it.
            for permissive in [
                ModelCapabilities::default(),
                ModelCapabilities {
                    tool_call_replay: Some(true),
                    ..Default::default()
                },
            ] {
                assert_eq!(
                    unsupported_capability(permissive, &request, &body),
                    None,
                    "{label} on a permissive target: {body}"
                );
            }
        }
        Ok(())
    }

    // A tool result alone is lineage: the call it answers may sit in history this body does
    // not carry, and the provider still has to accept the replay.
    #[test]
    fn a_tool_result_alone_is_a_replay() -> TestResult {
        let cannot_replay = ModelCapabilities {
            tool_call_replay: Some(false),
            ..Default::default()
        };
        for (format, body) in [
            (
                OpenAiChat,
                json!({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                ]}),
            ),
            (
                OpenAiResponses,
                json!({"input": [
                    {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "done"},
                ]}),
            ),
            (
                AnthropicMessages,
                json!({"max_tokens": 32, "messages": [
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "done"}]},
                ]}),
            ),
        ] {
            let request = decode_request(format, &body)?;
            assert_eq!(
                unsupported_capability(cannot_replay, &request, &body),
                Some("tool_call_replay"),
                "{body}"
            );
        }
        Ok(())
    }

    // Lineage has to be read from the preserved body too: a caller can spell the replayed
    // call in a field decoding does not normalise, and it must not reach a target that
    // cannot accept it.
    #[test]
    fn lineage_is_read_from_the_preserved_body_not_only_decoded_blocks() -> TestResult {
        let cannot_replay = ModelCapabilities {
            tool_call_replay: Some(false),
            ..Default::default()
        };
        let legacy_function_call = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "function_call": {"name": "lookup", "arguments": "{}"}},
            {"role": "function", "name": "lookup", "content": "done"},
        ]});
        let request = decode_request(OpenAiChat, &legacy_function_call)?;
        assert_eq!(
            unsupported_capability(cannot_replay, &request, &legacy_function_call),
            Some("tool_call_replay"),
            "a legacy function_call is a replayed call: {legacy_function_call}"
        );
        // Control: an assistant turn with no call at all is ordinary history.
        let plain = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]});
        let request = decode_request(OpenAiChat, &plain)?;
        assert_eq!(
            unsupported_capability(cannot_replay, &request, &plain),
            None,
            "ordinary history must stay eligible: {plain}"
        );
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
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "collect", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": {"video": "clip.mp4", "audio": "note.wav"}
                }
            ]
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

    /// An endpoint that reads inline video but cannot fetch a URL must refuse the URL.
    ///
    /// This is the measured shape of the free Nemotron endpoint: 200 for a `data:` video,
    /// 404 "No endpoints found that support video URLs". A route that declared `video = true`
    /// with no carrier would have admitted both.
    #[test]
    fn an_inline_only_carrier_refuses_a_remote_video() -> TestResult {
        let inline = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video": {"data": "AAAA", "media_type": "video/mp4"}}
        ]}]});
        let remote = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
        ]}]});
        let inline_only = ModelCapabilities {
            video: Some(true),
            video_carrier: Some(MediaCarrier::Inline),
            ..Default::default()
        };
        for (body, expected) in [(inline, None), (remote, Some("video_carrier"))] {
            let request = decode_request(OpenAiResponses, &body)?;
            assert_eq!(
                unsupported_capability(inline_only, &request, &body),
                expected,
                "{body}"
            );
        }
        Ok(())
    }

    /// A remote-only carrier is the mirror case: inline bytes are refused, a URL is not.
    #[test]
    fn a_remote_only_carrier_refuses_inline_video() -> TestResult {
        let inline = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video": {"data": "AAAA", "media_type": "video/mp4"}}
        ]}]});
        let remote = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
        ]}]});
        let remote_only = ModelCapabilities {
            video: Some(true),
            video_carrier: Some(MediaCarrier::Remote),
            ..Default::default()
        };
        for (body, expected) in [(inline, Some("video_carrier")), (remote, None)] {
            let request = decode_request(OpenAiResponses, &body)?;
            assert_eq!(
                unsupported_capability(remote_only, &request, &body),
                expected,
                "{body}"
            );
        }
        Ok(())
    }

    /// Narrowing video must not touch an image request: the carriers are read per medium.
    #[test]
    fn a_narrow_video_carrier_leaves_images_alone() -> TestResult {
        let image = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.test/cat.jpg"}
        ]}]});
        let capabilities = ModelCapabilities {
            vision: Some(true),
            video: Some(true),
            video_carrier: Some(MediaCarrier::Inline),
            ..Default::default()
        };
        let request = decode_request(OpenAiResponses, &image)?;
        assert_eq!(
            unsupported_capability(capabilities, &request, &image),
            None,
            "a video carrier must not constrain images: {image}"
        );
        Ok(())
    }

    /// The image carrier is checked against images only, and the other way round.
    #[test]
    fn a_narrow_image_carrier_refuses_a_remote_image_only() -> TestResult {
        let remote_image = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.test/cat.jpg"}
        ]}]});
        let inline_image = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}]});
        let video = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
        ]}]});
        let inline_images = ModelCapabilities {
            vision: Some(true),
            image_carrier: Some(MediaCarrier::Inline),
            // Video is declared with both carriers so the last case proves the image
            // narrowing does not reach it, rather than being refused for want of video.
            video: Some(true),
            ..Default::default()
        };
        for (body, expected) in [
            (remote_image, Some("image_carrier")),
            (inline_image, None),
            (video, None),
        ] {
            let request = decode_request(OpenAiResponses, &body)?;
            assert_eq!(
                unsupported_capability(inline_images, &request, &body),
                expected,
                "{body}"
            );
        }
        Ok(())
    }

    /// The image mirror of the video case: a remote-only image carrier refuses inline bytes.
    ///
    /// This is the half the video tests do not reach, and it was a genuine gap: deleting the
    /// inline-image check left the suite green.
    #[test]
    fn a_remote_only_image_carrier_refuses_inline_images() -> TestResult {
        let inline = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}]});
        let remote = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.test/cat.jpg"}
        ]}]});
        let remote_only = ModelCapabilities {
            vision: Some(true),
            image_carrier: Some(MediaCarrier::Remote),
            ..Default::default()
        };
        for (body, expected) in [(inline, Some("image_carrier")), (remote, None)] {
            let request = decode_request(OpenAiResponses, &body)?;
            assert_eq!(
                unsupported_capability(remote_only, &request, &body),
                expected,
                "{body}"
            );
        }
        Ok(())
    }

    /// A route declaring no carrier means both, so the historical manifest is unchanged.
    #[test]
    fn an_undeclared_carrier_admits_either_form() -> TestResult {
        for body in [
            json!({"input": [{"role": "user", "content": [
                {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
            ]}]}),
            json!({"input": [{"role": "user", "content": [
                {"type": "input_video", "video": {"data": "AAAA", "media_type": "video/mp4"}}
            ]}]}),
        ] {
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
                None,
                "{body}"
            );
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

    /// Regression: an inline image is charged as a picture, not as its base64 characters.
    ///
    /// Measured on a real Discord turn: a 361,273-byte PNG became 481,700 base64 characters
    /// and the text rate priced it at 120,425 tokens, so a request that fitted a 262,144
    /// window was refused with `requires context_window` and no provider was ever called.
    #[test]
    fn an_inline_image_is_not_charged_as_text() {
        let payload = "A".repeat(481_700);
        let with_image = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{payload}")}}
        ]}]});
        let without_image = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"}
        ]}]});
        let estimate = estimate_input_tokens(&with_image);
        assert!(
            estimate < 6_000,
            "an inline image must not cost its base64 length in tokens: {estimate}"
        );
        // The picture contributes a fixed charge: ten times the payload costs the same. A
        // residual character count would move with it, which is exactly the defect.
        let ten_times = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {
                "url": format!("data:image/png;base64,{}", payload.repeat(10))
            }}
        ]}]});
        assert_eq!(
            estimate_input_tokens(&ten_times),
            estimate,
            "an image charge must not follow the payload size"
        );
        // And the picture, not its bytes, is what the estimate gained.
        assert!(
            estimate - estimate_input_tokens(&without_image) >= IMAGE_TOKENS,
            "an image must cost at least one image charge"
        );
    }

    /// The control that isolates the defect: the same byte count as text is still charged in
    /// full, so the fix narrows one claim rather than lowering the estimator everywhere.
    #[test]
    fn the_same_bytes_as_text_are_still_charged_in_full() {
        let payload = "A".repeat(481_700);
        let body = json!({"messages": [{"role": "user", "content": format!("what is this {payload}")}]});
        assert!(
            estimate_input_tokens(&body) > 120_000,
            "text must keep costing its characters: {}",
            estimate_input_tokens(&body)
        );
    }

    /// Every wire shape the deployment forwards carries the picture, not prose: chat and
    /// Responses parts, Anthropic's `source.data`, and a remote URL that cannot be weighed.
    #[test]
    fn each_image_wire_shape_is_charged_once() {
        let payload = "A".repeat(481_700);
        for body in [
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{payload}")}}
            ]}]}),
            json!({"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": format!("data:image/png;base64,{payload}")}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": payload.clone()
                }}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/photo.jpg"}}
            ]}]}),
            // A part that names no type but carries only an image reference.
            json!({"messages": [{"role": "user", "content": [
                {"image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}]}),
        ] {
            let estimate = estimate_input_tokens(&body);
            assert!(
                (IMAGE_TOKENS..6_000).contains(&estimate),
                "{estimate} for {body}"
            );
        }
    }

    /// Review correction: a base64 blob or a `data:image` URI in ordinary text, a tool
    /// argument or a tool result is text. It must keep its character cost, because replacing
    /// it with one image charge would under-admit a prose request by ~120k tokens.
    #[test]
    fn text_carrying_an_image_payload_keeps_text_accounting() {
        let payload = "A".repeat(481_700);
        for body in [
            // A user turn that merely quotes one.
            json!({"messages": [{"role": "user", "content":
                format!("here is the header: data:image/png;base64,{payload}")}]}),
            // A tool result carrying one as its whole content.
            json!({"messages": [{"role": "tool", "content":
                format!("data:image/png;base64,{payload}")}]}),
            // A tool argument naming one.
            json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "c1",
                "type": "function", "function": {"name": "inspect", "arguments":
                format!("{{\"image\":\"data:image/png;base64,{payload}\"}}")}}]}]}),
        ] {
            let estimate = estimate_input_tokens(&body);
            assert!(
                estimate > 120_000,
                "text must keep costing its characters: {estimate}"
            );
        }
    }

    /// Images accumulate: each picture is its own charge, so a turn with several images is
    /// not priced as one.
    #[test]
    fn images_accumulate_per_part() {
        let body_of = |count: usize| {
            let parts: Vec<serde_json::Value> = (0..count)
                .map(|n| json!({"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{}", "A".repeat(1_000 + n))
                }}))
                .collect();
            json!({"messages": [{"role": "user", "content": parts}]})
        };
        let one = estimate_input_tokens(&body_of(1));
        for count in 2..=4usize {
            let estimate = estimate_input_tokens(&body_of(count));
            let extra = u64::try_from(count - 1).expect("small count");
            let gained = estimate - one;
            // Each added picture must cost at least one image charge. The small excess is the
            // part's own structural keys ("image_url", the message wrapper), which are real
            // text; the upper bound is what proves the payload is not being counted, since
            // 1,000 base64 characters would cost 250 tokens each if it were.
            let per_image = gained / extra;
            assert!(
                (IMAGE_TOKENS..IMAGE_TOKENS + 64).contains(&per_image),
                "{count} images gained {gained} for {extra} extra pictures ({per_image} each)"
            );
        }
    }

    /// Resolution and detail do not change the image charge: the gate cannot weigh pixels, so
    /// it charges the ceiling whether the image declares itself small, large or `detail: low`.
    ///
    /// Pixel counts are numbers and are ignored. `detail` is a metadata string and is counted
    /// as the two tokens it is, rather than having its subtree skipped: counting it is the
    /// safer rule, and what must not happen is the charge moving with resolution or detail.
    #[test]
    fn resolution_and_detail_do_not_change_the_charge() {
        let small = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"},
             "width": 64, "height": 64}
        ]}]});
        let huge = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"},
             "width": 8192, "height": 8192}
        ]}]});
        let low_detail = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"},
             "detail": "low"}
        ]}]});
        assert_eq!(estimate_input_tokens(&small), estimate_input_tokens(&huge));
        let small_tokens = estimate_input_tokens(&small);
        let low_tokens = estimate_input_tokens(&low_detail);
        let delta = small_tokens.abs_diff(low_tokens);
        assert!(
            delta < 16,
            "`detail` is metadata, not a second picture: {small_tokens} vs {low_tokens}"
        );
        assert!(
            (IMAGE_TOKENS..6_000).contains(&low_tokens),
            "one image stays one image charge: {low_tokens}"
        );
    }

    /// Malformed and empty image parts are still charged, and never panic: a part the
    /// provider would refuse must not slip through the window gate because it has no payload.
    #[test]
    fn malformed_image_parts_are_charged_and_do_not_panic() {
        for body in [
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": null}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url"}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,"}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "image", "source": null}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": null}}
            ]}]}),
        ] {
            let estimate = estimate_input_tokens(&body);
            assert!(
                estimate >= IMAGE_TOKENS,
                "a declared image part is charged even without a payload: {estimate}"
            );
        }
    }

    /// The charge must stay above every documented per-image cost of a family these routes
    /// can select. Lowering the constant below one of them fails here, so the number cannot
    /// drift back to a Google-only figure.
    #[test]
    fn the_charge_covers_every_documented_family() {
        let documented = [
            ("google_gemini (4 tiles at 258)", 1_032u64),
            ("openai detail=high (85 + 6 tiles at 170)", 1_105),
            ("anthropic (1568x1568 / 750)", 3_279),
        ];
        for (family, tokens) in documented {
            assert!(
                IMAGE_TOKENS >= tokens,
                "{family} documents {tokens} tokens per image, above IMAGE_TOKENS {IMAGE_TOKENS}"
            );
        }
        // And it is not wild: a handful of images must still fit an ordinary window.
        assert!(10 * IMAGE_TOKENS < 262_144 / 4, "ten images must not dominate a window");
    }

    /// Mixed request: tool definitions, text and an image are each accounted for, and the
    /// tool schema is not disturbed by the image rule.
    #[test]
    fn an_image_mixed_with_tools_and_text_is_accounted_for() {
        let payload = "A".repeat(481_700);
        let body = json!({
            "model": "ST_Route",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "what is in this picture?"},
                    {"type": "image_url", "image_url": {
                        "url": format!("data:image/png;base64,{payload}")
                    }}
                ]},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"}
            ],
            "tools": [{"type": "function", "function": {"name": "lookup",
                       "description": "look something up",
                       "parameters": {"type": "object", "properties": {
                           "q": {"type": "string"}}}}}]
        });
        let estimate = estimate_input_tokens(&body);
        assert!(
            (IMAGE_TOKENS..6_000).contains(&estimate),
            "one image plus a small conversation: {estimate}"
        );
        // The same request without the image is smaller by exactly one image charge.
        let text_only = json!({"model": "ST_Route", "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "what is in this picture?"}
        ], "tools": [{"type": "function", "function": {"name": "lookup",
            "description": "look something up", "parameters": {"type": "object"}}}]});
        assert!(estimate > estimate_input_tokens(&text_only) + IMAGE_TOKENS - 200);
    }

    /// Scope check: a non-image `data:` URI keeps the old treatment, so audio and video
    /// pricing is not silently changed by this fix.
    #[test]
    fn a_non_image_data_uri_is_unchanged() {
        let payload = "A".repeat(4_000);
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {
                "data": format!("data:audio/wav;base64,{payload}"), "format": "wav"
            }}
        ]}]});
        assert!(estimate_input_tokens(&body) > 900, "{}", estimate_input_tokens(&body));
    }

    // ---- Position: only a protocol content slot can hold a media part --------------------
    //
    // The raw body is forwarded verbatim and is full of objects that merely *look* like
    // parts. A tool schema property named `source`, a JSON-schema `"type": "image"`, or any
    // metadata object carrying an `image_url` key is data, and charging it 4,096 tokens
    // while skipping its subtree under-counts prose by the same class of error as the
    // original defect. These controls hold that boundary.

    /// A JSON-schema property named `source` is a schema, not a picture. Its description is
    /// prose and must be priced as prose.
    #[test]
    fn a_tool_schema_property_named_source_keeps_text_accounting() {
        let prose = "d".repeat(40_000);
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {
                "type": "object",
                "properties": {"source": {"type": "object", "description": prose}}
            }}}]
        });
        let estimate = estimate_input_tokens(&body);
        assert!(
            estimate > 9_000,
            "a property named `source` is data, not a picture: {estimate}"
        );
    }

    /// Likewise a property named `image_url`: the name alone does not make a schema an image.
    #[test]
    fn a_tool_schema_property_named_image_url_keeps_text_accounting() {
        let prose = "d".repeat(40_000);
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {
                "type": "object",
                "properties": {"image_url": {"type": "string", "description": prose}}
            }}}]
        });
        let estimate = estimate_input_tokens(&body);
        assert!(estimate > 9_000, "a schema property is data: {estimate}");
    }

    /// `"type": "image"` is also a JSON-schema type name. Outside a content slot it is data.
    #[test]
    fn a_json_schema_type_image_outside_content_is_data() {
        let prose = "d".repeat(40_000);
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {
                "type": "object",
                "properties": {"kind": {"type": "image", "description": prose}}
            }}}]
        });
        let estimate = estimate_input_tokens(&body);
        assert!(estimate > 9_000, "a schema type is data, not a part: {estimate}");
    }

    /// A schema property *named* `content` does not open a content slot: the slot is a
    /// position in the protocol, not a key that happens to be spelled the same way.
    #[test]
    fn a_schema_property_named_content_does_not_open_a_part_slot() {
        let prose = "d".repeat(40_000);
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {
                "type": "object",
                "properties": {"content": {"image_url": {
                    "url": "data:image/png;base64,AAAA", "description": prose
                }}}
            }}}]
        });
        let estimate = estimate_input_tokens(&body);
        assert!(
            estimate > 9_000,
            "an image-shaped schema is still a schema: {estimate}"
        );
    }

    /// An untyped object carrying `source` inside a content slot is ambiguous -- audio,
    /// documents and provider metadata all use that key -- so it is priced as text.
    #[test]
    fn an_untyped_source_in_content_is_not_a_picture() {
        let payload = "A".repeat(481_700);
        let body = json!({"messages": [{"role": "user", "content": [
            {"source": {"data": payload}}
        ]}]});
        let estimate = estimate_input_tokens(&body);
        assert!(
            estimate > 120_000,
            "an untyped `source` must keep text accounting: {estimate}"
        );
    }

    /// Within a *confirmed* image part, only the payload fields are skipped. A `source`
    /// object that carries prose rather than bytes is prose.
    #[test]
    fn a_typed_image_part_with_prose_in_source_charges_the_prose() {
        let prose = "d".repeat(40_000);
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"description": prose}}
        ]}]});
        let estimate = estimate_input_tokens(&body);
        assert!(
            estimate >= IMAGE_TOKENS + 9_000,
            "an image part's prose sibling is still prose: {estimate}"
        );
    }

    /// Non-image media keeps text accounting: a document's base64 and an audio payload are
    /// not pictures, and this fix does not start pricing them as ones.
    #[test]
    fn non_image_media_keep_text_accounting() {
        let payload = "A".repeat(481_700);
        for body in [
            json!({"messages": [{"role": "user", "content": [
                {"type": "document", "source": {
                    "type": "base64", "media_type": "application/pdf", "data": payload.clone()
                }}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": payload.clone(), "format": "wav"}}
            ]}]}),
        ] {
            let estimate = estimate_input_tokens(&body);
            assert!(
                estimate > 120_000,
                "non-image media is charged by its bytes: {estimate}"
            );
        }
    }

    /// The shapes that must keep working: a genuine part in a user message, in a chat tool
    /// result, and in a Responses tool output. Each costs one charge and not its payload.
    #[test]
    fn genuine_image_positions_are_charged_once() {
        let payload = "A".repeat(481_700);
        for body in [
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{payload}")
                }}
            ]}]}),
            // A chat tool result that carries a picture.
            json!({"messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1",
                 "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": [
                    {"type": "image_url", "image_url": {
                        "url": format!("data:image/png;base64,{payload}")
                    }}
                ]}
            ]}),
            // A Responses tool output that carries a picture.
            json!({"input": [
                {"type": "function_call_output", "call_id": "c1", "output": [
                    {"type": "input_image", "image_url": format!("data:image/png;base64,{payload}")}
                ]}
            ]}),
        ] {
            let estimate = estimate_input_tokens(&body);
            assert!(
                (IMAGE_TOKENS..6_000).contains(&estimate),
                "a genuine part costs one image charge and not its payload: {estimate}"
            );
        }
    }

    /// Malformed nested metadata must not panic, and must not buy a 4,096-token discount on
    /// a subtree that actually holds text.
    #[test]
    fn malformed_nested_metadata_is_charged_and_does_not_panic() {
        let prose = "d".repeat(40_000);
        for body in [
            json!({"messages": [{"role": "user", "content": [
                {"type": "image", "source": {"nested": {"deeper": [prose.clone()]}}}
            ]}]}),
            json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": null},
                 "annotations": [{"note": prose.clone()}]}
            ]}]}),
        ] {
            let estimate = estimate_input_tokens(&body);
            assert!(
                estimate >= IMAGE_TOKENS + 9_000,
                "nested prose inside a part is charged: {estimate}"
            );
        }
    }
}

#[cfg(test)]
mod video_part_shape_tests {
    use super::unsupported_responses_video_part;
    use serde_json::json;

    /// The one accepted spelling: `input_video` with a string `video_url`, either a URL or
    /// a `data:` URI. Measured against the live provider endpoint, which rejects the
    /// object form with 400 and this one with 200.
    #[test]
    fn the_accepted_spellings_are_not_flagged() {
        for video_url in [
            "https://example.test/clip.mp4",
            "data:video/mp4;base64,AAAA",
        ] {
            let body = json!({"input": [{"role": "user", "content": [
                {"type": "input_video", "video_url": video_url},
                {"type": "input_text", "text": "describe"}
            ]}]});
            assert!(
                unsupported_responses_video_part(&body).is_none(),
                "{video_url} is the accepted shape"
            );
        }
        // A file reference is the provider's own alternative to inline bytes.
        let body = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "file_id": "file-123"}
        ]}]});
        assert!(unsupported_responses_video_part(&body).is_none());
    }

    /// The object form is what the provider answers with 400, so it must be refused here
    /// with a reason a caller can act on rather than forwarded into an opaque error.
    #[test]
    fn the_object_form_is_refused_with_its_reason() {
        for (body, expected) in [
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_video", "video_url": {"url": "https://example.test/clip.mp4"}}
                ]}]}),
                "`video_url` must be a string here, not an object",
            ),
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_video", "video": {"media_type": "video/mp4", "data": "AAAA"}}
                ]}]}),
                "inline bytes belong in a `data:` URI supplied as the string `video_url`",
            ),
            // A `video_url` part that carries a URL is normalized before this predicate
            // runs, so what is left for this branch is the part with no video payload at all.
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "video_url"}
                ]}]}),
                "this surface accepts video only as an `input_video` part",
            ),
            (
                json!({"input": [{"role": "user", "content": [
                    {"type": "input_video"}
                ]}]}),
                "the part carries no `video_url`",
            ),
        ] {
            let shape = unsupported_responses_video_part(&body)
                .unwrap_or_else(|| panic!("{body} must be refused"));
            assert_eq!(shape.detail, expected, "{body}");
        }
    }

    /// Only video is the guard's business: images, audio and files keep their own paths,
    /// and a text-only body must stay untouched.
    #[test]
    fn non_video_content_is_never_flagged() {
        let body = json!({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.test/a.png"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "input_file", "file_id": "file-1"},
            {"type": "input_text", "text": "describe"}
        ]}]});
        assert!(unsupported_responses_video_part(&body).is_none());
        assert!(unsupported_responses_video_part(&json!({"input": "hello"})).is_none());
    }

    /// The Chat spelling is what several clients send on the Responses surface. Translating
    /// the spelling is what makes video reachable; the media type must ride along.
    #[test]
    fn the_chat_spelling_is_translated_into_the_accepted_part() {
        let mut body = json!({"input": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4",
                                                "media_type": "video/mp4"}}
        ]}]});
        assert_eq!(super::normalize_responses_video_parts(&mut body), 1);
        let part = &body["input"][0]["content"][0];
        assert_eq!(part["type"], "input_video");
        assert_eq!(part["video_url"], "https://example.test/clip.mp4");
        assert_eq!(part["media_type"], "video/mp4");
        assert!(unsupported_responses_video_part(&body).is_none());
    }

    #[test]
    fn inline_bytes_become_a_data_uri_under_the_accepted_type() {
        let mut body = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video": {"media_type": "video/mp4", "data": "AAAA"}}
        ]}]});
        assert_eq!(super::normalize_responses_video_parts(&mut body), 1);
        let part = &body["input"][0]["content"][0];
        assert_eq!(part["type"], "input_video");
        assert_eq!(part["video_url"], "data:video/mp4;base64,AAAA");
        assert!(
            part.get("video").is_none(),
            "the object spelling is dropped"
        );
        assert!(unsupported_responses_video_part(&body).is_none());
    }

    /// A correct part is left byte-identical, and running the pass twice changes nothing:
    /// ingress must not rewrite a body that is already what the provider accepts.
    #[test]
    fn the_accepted_shape_is_untouched_and_the_pass_is_idempotent() {
        let original = json!({"input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"},
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "https://example.test/a.png"}
        ]}]});
        let mut body = original.clone();
        assert_eq!(super::normalize_responses_video_parts(&mut body), 0);
        assert_eq!(body, original);
        // And a body the pass did rewrite settles: a second pass finds nothing to do.
        let mut rewritten = json!({"input": [{"role": "user", "content": [
            {"type": "video_url", "video_url": "https://example.test/clip.mp4"}
        ]}]});
        assert_eq!(super::normalize_responses_video_parts(&mut rewritten), 1);
        assert_eq!(super::normalize_responses_video_parts(&mut rewritten), 0);
    }

    /// Nothing is invented for a part that carries no payload: it is left exactly as sent so
    /// the caller gets the accurate refusal rather than a silently emptied part.
    #[test]
    fn an_unmappable_part_is_left_for_the_refusal() {
        let mut body = json!({"input": [{"role": "user", "content": [
            {"type": "input_video"}
        ]}]});
        assert_eq!(super::normalize_responses_video_parts(&mut body), 0);
        assert_eq!(
            body,
            json!({"input": [{"role": "user", "content": [
                {"type": "input_video"}
            ]}]})
        );
        assert!(unsupported_responses_video_part(&body).is_some());
    }
}
