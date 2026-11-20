// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Video content parts across the Chat and Responses codecs.
//!
//! Video has one provider spelling on the Responses wire, `input_video`. These tests fix
//! the shape in both directions and confirm that a part the codec does not recognize is
//! preserved as an opaque block rather than silently converted to text.

use serde_json::{Value, json};
use switchyard_translation::{
    PreservationPolicy, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

// Translation is exercised rather than replay, so the codec does the work.
fn translating_policy() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

fn content_parts(body: &Value) -> &Vec<Value> {
    body["input"][0]["content"]
        .as_array()
        .expect("translated body should carry input content")
}

// A Chat caller spells video as an object URL; the Responses target expects `input_video`
// with a string URL. Both are the same medium and must survive the change of format.
#[test]
fn chat_video_url_becomes_a_responses_input_video() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = translating_policy();
    let chat = json!({
        "model": "smartfree",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}}
        ]}]
    });

    let responses = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &chat,
            &policy,
        )?
        .body;

    let parts = content_parts(&responses);
    let video = parts
        .iter()
        .find(|part| part["type"] == "input_video")
        .ok_or("video part was not translated")?;
    assert_eq!(video["video_url"], "https://example.test/clip.mp4");
    Ok(())
}

// Inline bytes keep their media type and payload.
#[test]
fn responses_inline_video_keeps_its_bytes() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = translating_policy();
    let body = json!({
        "input": [{"role": "user", "content": [
            {"type": "input_video", "video": {"media_type": "video/mp4", "data": "AAAA"}}
        ]}]
    });

    let responses = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    // The bytes survive, but the spelling is the one the provider accepts: an
    // `input_video` part whose `video_url` is the `data:` URI carrying those bytes. The
    // object form this test used to assert is answered with 400 invalid_union.
    let parts = content_parts(&responses);
    assert_eq!(parts[0]["type"], "input_video");
    assert_eq!(parts[0]["video_url"], "data:video/mp4;base64,AAAA");
    Ok(())
}

// A provider field the codec has no field for must not cost the caller the video. On the
// same wire format the exact body is replayed, which keeps optional knobs such as
// `processing` that the normalized request cannot carry.
#[test]
fn an_unmodelled_video_field_survives_same_format_replay() -> TestResult {
    let engine = TranslationEngine::default();
    let part = json!({
        "type": "input_video",
        "video_url": "https://example.test/clip.mp4",
        "processing": "agentic"
    });
    let body = json!({"input": [{"role": "user", "content": [part]}]});

    let responses = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(responses, body);
    Ok(())
}

// Across formats the part is rebuilt, so an unknown knob is dropped rather than invented.
// The medium itself must survive, and an absent media type must be omitted, not sent null.
#[test]
fn cross_format_translation_keeps_the_medium_and_omits_unknown_nulls() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = translating_policy();
    let body = json!({
        "input": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "https://example.test/clip.mp4",
             "processing": "agentic"}
        ]}]
    });

    let responses = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let part = &content_parts(&responses)[0];
    assert_eq!(part["type"], "input_video");
    assert_eq!(part["video_url"], "https://example.test/clip.mp4");
    assert!(
        part.get("media_type").is_none(),
        "an unknown media type must be omitted, not sent as null: {part}"
    );
    Ok(())
}

// An unrecognized video part is preserved opaque. Converting it to text would hand the
// model a JSON blob in place of a video it could otherwise have received.
#[test]
fn an_unrecognized_video_part_is_not_flattened_to_text() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = translating_policy();
    let body = json!({
        "input": [{"role": "user", "content": [
            {"type": "input_video", "video_reference": "provider-opaque-id"}
        ]}]
    });

    let responses = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let parts = content_parts(&responses);
    assert!(
        parts.iter().all(|part| part["type"] != "input_text"),
        "an opaque video part must not become text: {parts:?}"
    );
    Ok(())
}

// The provider's Responses endpoint accepts video only as `input_video` with a string
// `video_url`; measured against the live endpoint, the `{video: {data, media_type}}`
// object shape is rejected with 400 (invalid_union, expected string). Inline bytes must
// therefore cross the wire as the `data:` URI that carries the same bytes.
#[test]
fn chat_inline_video_becomes_a_responses_data_uri() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = translating_policy();
    let chat = json!({
        "model": "smartfree",
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video": {"media_type": "video/mp4", "data": "AAAA"}}
        ]}]
    });

    let responses = engine
        .translate_request(WireFormat::OpenAiChat, WireFormat::OpenAiResponses, &chat, &policy)?
        .body;

    let parts = content_parts(&responses);
    assert_eq!(parts[0]["type"], "input_video");
    assert_eq!(parts[0]["video_url"], "data:video/mp4;base64,AAAA");
    assert!(
        parts[0].get("video").is_none(),
        "the object spelling is what the provider rejects"
    );
    Ok(())
}

// Without a media type there is no faithful `data:` URI. Inventing one would mislabel the
// bytes, so the part is reported as text with a diagnostic rather than sent malformed.
#[test]
fn inline_video_without_a_media_type_is_reported_not_invented() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = translating_policy();
    let chat = json!({
        "model": "smartfree",
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video": {"data": "AAAA"}}
        ]}]
    });

    let translated = engine.translate_request(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        &chat,
        &policy,
    )?;

    let parts = content_parts(&translated.body);
    assert_eq!(parts[0]["type"], "input_text");
    assert!(
        parts[0]["text"].as_str().unwrap_or_default().contains("4 bytes"),
        "the fallback must at least say what was dropped: {}",
        parts[0]
    );
    assert!(
        translated
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.contains("lossy")),
        "dropping inline video is a lossy conversion and must be reported"
    );
    Ok(())
}
