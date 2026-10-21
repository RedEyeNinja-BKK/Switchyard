// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the provider-reported cost carried on a response's usage.
//!
//! A provider that bills per request may report what it charged. That figure is the
//! provider's own, and it must survive decoding untouched, because the economics
//! surface publishes it beside — never instead of — a Switchyard-derived estimate.
//! A provider that reports no cost must yield no cost at all, which is absent rather
//! than zero: a provider that does not bill per request has not charged zero.

use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn engine() -> TranslationEngine {
    TranslationEngine::default()
}

fn chat_body(usage: serde_json::Value) -> serde_json::Value {
    json!({
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "model": "vendor/model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Answer."},
            "finish_reason": "stop"
        }],
        "usage": usage
    })
}

fn responses_body(usage: serde_json::Value) -> serde_json::Value {
    json!({
        "id": "resp_test",
        "object": "response",
        "model": "vendor/model",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer."}]
        }],
        "usage": usage
    })
}

/// A provider that reports a cost has that cost carried on the decoded usage.
#[test]
fn a_reported_cost_survives_decoding() -> TestResult {
    let engine = engine();
    let body = chat_body(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.0025
    }));

    let decoded = engine
        .decode_response(WireFormat::OpenAiChat, &body, &TranslationPolicy::default())?
        .response;
    assert_eq!(decoded.usage.cost, Some(0.0025));
    Ok(())
}

/// A cost of zero is a measurement, not an absence. A free model still consumed a
/// provider's free-tier allowance, and the provider reported that it charged nothing.
/// Recording it as absent would lose a real observation.
#[test]
fn a_reported_zero_cost_is_kept_as_zero() -> TestResult {
    let engine = engine();
    let body = chat_body(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.0
    }));

    let decoded = engine
        .decode_response(WireFormat::OpenAiChat, &body, &TranslationPolicy::default())?
        .response;
    assert_eq!(
        decoded.usage.cost,
        Some(0.0),
        "a reported zero cost is a measurement and must not be dropped"
    );
    Ok(())
}

/// A provider that reports no cost yields no cost. Absent is not zero: a provider
/// that does not bill per request has not charged zero.
#[test]
fn a_missing_cost_is_absent_rather_than_zero() -> TestResult {
    let engine = engine();
    let body = chat_body(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    }));

    let decoded = engine
        .decode_response(WireFormat::OpenAiChat, &body, &TranslationPolicy::default())?
        .response;
    assert_eq!(decoded.usage.cost, None);
    assert_eq!(decoded.usage.cost_currency, None);
    Ok(())
}

/// The Responses shape carries the same figure, because a provider that reports a
/// cost reports it on both shapes.
#[test]
fn a_reported_cost_survives_decoding_on_the_responses_shape() -> TestResult {
    let engine = engine();
    let body = responses_body(json!({
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cost": 0.125
    }));

    let decoded = engine
        .decode_response(
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .response;
    assert_eq!(decoded.usage.cost, Some(0.125));
    Ok(())
}

/// A streamed response reports its cost on the usage chunk, and the terminal usage
/// must carry it into the decoded stream state.
#[test]
fn a_streamed_cost_reaches_the_stream_state() -> TestResult {
    let engine = engine();
    let mut state = switchyard_translation::StreamTranslationState::new(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
    );
    engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
        &json!({
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "model": "vendor/model",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.007}
        }),
    )?;
    assert_eq!(state.usage.cost, Some(0.007));
    Ok(())
}

/// A streamed response that reports no cost yields no cost.
#[test]
fn a_stream_without_a_cost_yields_none() -> TestResult {
    let engine = engine();
    let mut state = switchyard_translation::StreamTranslationState::new(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
    );
    engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
        &json!({
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "model": "vendor/model",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }),
    )?;
    assert_eq!(state.usage.cost, None);
    Ok(())
}

/// Anthropic reports no per-request cost on the Messages API, so the decoded usage
/// carries none. Inventing a cost there would attribute a charge the provider never
/// made.
#[test]
fn the_anthropic_shape_reports_no_cost() -> TestResult {
    let engine = engine();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "vendor/model",
        "content": [{"type": "text", "text": "Answer."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });

    let decoded = engine
        .decode_response(
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .response;
    assert_eq!(decoded.usage.cost, None);
    assert_eq!(decoded.usage.cost_currency, None);
    Ok(())
}
