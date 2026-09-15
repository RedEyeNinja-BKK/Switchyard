// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response usage and full-turn latency metrics for routed requests.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use opentelemetry::{KeyValue, global};
use switchyard_protocol::{LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Response, Usage};

use crate::SharedRoutingLog;
use crate::routing_log::RoutingLogContext;
use crate::stats::{StatsAccumulator, TokenUsage};

/// Observes a routed response without changing its aggregate or streaming contents.
pub(crate) fn observe(
    response: Response,
    model: &str,
    started: Instant,
    stats: StatsAccumulator,
    cache_eligible: f64,
    routing_log: Option<(SharedRoutingLog, RoutingLogContext)>,
) -> Response {
    let Response {
        llm_response,
        metadata,
    } = response;
    let model = model.to_string();

    let llm_response = match llm_response {
        LlmResponse::Agg(agg) => {
            record_terminal(&stats, &agg.usage, &model, started, cache_eligible);
            if let Some((log, context)) = routing_log {
                log.append(context, &model, None, &agg.usage);
            }
            LlmResponse::Agg(agg)
        }
        LlmResponse::Stream(mut stream) => {
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                let mut terminal_seen = false;
                let mut recorded = false;
                while let Some(item) = stream.next().await {
                    let failure = stream_failure_reason(&item);
                    let failed = failure.is_some();
                    if let Some((kind, head)) = failure {
                        // An in-band failure arrives AFTER the response headers were already
                        // sent (the client-facing status is 200), so the request-level log
                        // records a clean success and this line is the only journal record of
                        // the event. Without it the class was invisible: the counter moved
                        // while the journal showed only successes (2026-09-15 finding, the
                        // gpt-5.6-luna and Qwen3.5-9B-MTP error series).
                        // `head` is bounded and sanitised - never the raw provider text.
                        tracing::warn!(
                            model = %model,
                            failure_kind = kind,
                            reason_head = %head,
                            "stream terminated with an in-band error (counted in switchyard_errors_total)"
                        );
                    }
                    if let Ok(event) = &item {
                        for chunk in event.normalized() {
                            match chunk {
                                LlmResponseChunk::Usage(usage) => {
                                    latest_usage = Some(usage.clone());
                                }
                                LlmResponseChunk::MessageStop { .. } => {
                                    terminal_seen = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    if failed {
                        record_stream_error(&stats, &model);
                    }
                    // Responses clients may stop polling immediately after the terminal event.
                    // Commit first when that event already carries the final usage.
                    if !failed && !recorded && terminal_seen
                        && let Some(usage) = latest_usage.as_ref()
                    {
                        record_terminal(&stats, usage, &model, started, cache_eligible);
                        if let Some((log, context)) = routing_log.as_ref() {
                            log.append(context.clone(), &model, None, usage);
                        }
                        recorded = true;
                    }
                    yield item;
                    if failed {
                        return;
                    }
                }
                if !recorded {
                    let usage = latest_usage.unwrap_or_default();
                    record_terminal(&stats, &usage, &model, started, cache_eligible);
                    if let Some((log, context)) = routing_log {
                        log.append(context, &model, None, &usage);
                    }
                }
            };
            LlmResponse::Stream(Box::pin(wrapped))
        }
    };

    Response {
        llm_response,
        metadata,
    }
}

// Records a terminal stream failure after the routed call was already counted.
fn record_stream_error(stats: &StatsAccumulator, model: &str) {
    stats.record_stream_error(model);
    global::meter("switchyard")
        .u64_counter("switchyard.errors")
        .build()
        .add(1, &attributes(model));
}

/// Human-readable reason when this stream item is a failure, else `None`.
///
/// Two failure shapes reach here, both counted in `switchyard_errors_total` by
/// [`record_stream_error`]: a transport error on the item itself, and a
/// well-formed provider event that carries an in-band error (`StreamError`) or a
/// decoding failure (`DecodeError`). Neither is a HTTP-level failure - the
/// headers were already sent - which is why this record is the only place the
/// class becomes visible in the journal.
///
/// The text is provider-controlled, so only a bounded, sanitised head is kept
/// ([`bounded_reason_head`]); the classification is a fixed vocabulary word.
fn stream_failure_reason(
    item: &Result<LlmResponseStreamEvent, switchyard_protocol::LlmClientError>,
) -> Option<(&'static str, String)> {
    match item {
        Err(error) => Some(("transport", bounded_reason_head(&error.to_string()))),
        Ok(event) => event.normalized().iter().find_map(|chunk| match chunk {
            LlmResponseChunk::StreamError { message } => {
                Some(("stream_error", bounded_reason_head(message)))
            }
            LlmResponseChunk::DecodeError { message } => {
                Some(("decode_error", bounded_reason_head(message)))
            }
            _ => None,
        }),
    }
}

/// Bytes of provider-supplied error text kept for the journal (review finding
/// 2026-09-15: provider bodies are arbitrary in size and content, so the record
/// is bounded and control characters are flattened - an embedded newline could
/// otherwise forge log structure). Beyond this the text is cut, never buffered.
const FAILURE_REASON_HEAD_BYTES: usize = 200;

/// Bounded, single-line head of provider-supplied error text.
fn bounded_reason_head(text: &str) -> String {
    let mut head = String::new();
    for character in text.chars() {
        let width = character.len_utf8();
        if head.len() + width > FAILURE_REASON_HEAD_BYTES {
            head.push('\u{2026}');
            break;
        }
        head.push(if character.is_control() { ' ' } else { character });
    }
    head
}

pub(crate) fn token_usage(usage: &Usage) -> TokenUsage {
    let cached_tokens = usage.cached_input_tokens().unwrap_or(0);
    let cache_creation_tokens = usage.cache_creation_input_tokens().unwrap_or(0);
    TokenUsage {
        prompt_tokens: usage
            .input_tokens
            .unwrap_or(0)
            .saturating_add(cached_tokens)
            .saturating_add(cache_creation_tokens),
        completion_tokens: usage.output_tokens.unwrap_or(0),
        cached_tokens,
        cache_creation_tokens,
        cacheable_prompt_tokens: 0,
        reasoning_tokens: usage.reasoning_tokens.unwrap_or(0),
    }
}

/// Records final usage and latency in both OpenTelemetry metrics and JSON stats.
fn record_terminal(
    stats: &StatsAccumulator,
    usage: &Usage,
    model: &str,
    started: Instant,
    cache_eligible: f64,
) {
    let total_latency = started.elapsed();
    record_usage(usage, model);
    record_latency(model, total_latency);
    let mut token_usage = token_usage(usage);
    token_usage.cacheable_prompt_tokens =
        (token_usage.prompt_tokens as f64 * cache_eligible).round() as u64;
    stats.record_usage(model, token_usage, total_latency.as_secs_f64() * 1_000.0);
}

fn attributes(model: &str) -> [KeyValue; 1] {
    [KeyValue::new("model", model.to_string())]
}

fn record_usage(usage: &Usage, model: &str) {
    let attributes = attributes(model);
    let meter = global::meter("switchyard");
    let cached = usage.cached_input_tokens();
    let cache_creation = usage.cache_creation_input_tokens();

    if usage.input_tokens.is_some() || cached.is_some() || cache_creation.is_some() {
        let prompt =
            usage.input_tokens.unwrap_or(0) + cached.unwrap_or(0) + cache_creation.unwrap_or(0);
        meter
            .u64_counter("switchyard.prompt_tokens")
            .build()
            .add(prompt, &attributes);
    }
    for (name, value) in [
        ("switchyard.completion_tokens", usage.output_tokens),
        ("switchyard.cached_tokens", cached),
        ("switchyard.cache_creation_tokens", cache_creation),
        ("switchyard.reasoning_tokens", usage.reasoning_tokens),
    ] {
        if let Some(value) = value {
            meter.u64_counter(name).build().add(value, &attributes);
        }
    }
}

fn record_latency(model: &str, latency: Duration) {
    global::meter("switchyard")
        .f64_histogram("switchyard.total_latency_ms")
        .build()
        .record(latency.as_secs_f64() * 1000.0, &attributes(model));
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};
    use switchyard_protocol::{LlmResponseChunk, LlmResponseStreamEvent, Metadata, Response};

    use super::*;

    // ---- in-band stream failure classification -------------------------------
    // These pin the reason text that makes the class visible in the journal: the
    // counter (switchyard_errors_total) moves for in-band failures while the
    // request-level log still shows a 200, so the WARN in the stream wrapper is the
    // only record. 2026-09-15: gpt-5.6-luna (409) and Qwen3.5-9B-MTP (28) errors
    // were unattributable for exactly this reason.

    #[test]
    fn healthy_event_has_no_failure_reason() {
        let event = LlmResponseStreamEvent::new(vec![LlmResponseChunk::MessageStop { reason: None }]);
        assert_eq!(stream_failure_reason(&Ok(event)), None);
    }

    #[test]
    fn in_band_stream_error_reports_kind_and_message() {
        let event = LlmResponseStreamEvent::new(vec![LlmResponseChunk::StreamError {
            message: "unknown Responses stream error".to_string(),
        }]);
        let (kind, head) = stream_failure_reason(&Ok(event)).expect("failure classified");
        assert_eq!(kind, "stream_error");
        assert_eq!(head, "unknown Responses stream error");
    }

    #[test]
    fn decode_error_reports_kind_and_message() {
        let event = LlmResponseStreamEvent::new(vec![LlmResponseChunk::DecodeError {
            message: "upstream transport error: error decoding response body".to_string(),
        }]);
        let (kind, head) = stream_failure_reason(&Ok(event)).expect("failure classified");
        assert_eq!(kind, "decode_error");
        assert_eq!(head, "upstream transport error: error decoding response body");
    }

    #[test]
    fn transport_error_reports_its_kind_and_display() {
        let failure = Err(switchyard_protocol::LlmClientError::Transport {
            source: Box::new(std::io::Error::other("connection reset")),
        });
        let (kind, head) = stream_failure_reason(&failure).expect("transport failure classified");
        assert_eq!(kind, "transport");
        assert!(head.contains("connection reset"), "unexpected head: {head}");
    }

    /// Provider text is unbounded and may contain newlines that would forge log
    /// structure; the journal record is bounded and single-line by construction
    /// (review finding, 2026-09-15).
    #[test]
    fn reason_head_is_bounded_and_flattened() {
        let huge = format!("line one\nline two\t{}", "x".repeat(10_000));
        let head = bounded_reason_head(&huge);
        assert!(
            head.len() <= FAILURE_REASON_HEAD_BYTES + '\u{2026}'.len_utf8(),
            "head was not bounded: {} bytes",
            head.len()
        );
        assert!(!head.contains('\n') && !head.contains('\t'), "control chars survived: {head:?}");
        assert!(head.starts_with("line one line two"), "unexpected head: {head:?}");
        assert!(head.ends_with('\u{2026}'), "truncation was not marked: {head:?}");
    }

    #[test]
    fn short_reason_head_is_not_marked() {
        assert_eq!(bounded_reason_head("shorter than the bound"), "shorter than the bound");
    }

    /// Behavioural coverage for the changed path (review finding 2, 2026-09-15):
    /// a failing item must be yielded unchanged, counted exactly once, and the
    /// wrapper must stop without consuming the items after it.
    #[tokio::test]
    async fn failing_item_is_yielded_counted_once_and_ends_the_stream() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = SharedRoutingLog::new(dir.path().join("routing.jsonl")).expect("routing log");
        let stats = StatsAccumulator::default();
        let failed_event = LlmResponseStreamEvent::new(vec![LlmResponseChunk::StreamError {
            message: "unknown Responses stream error".to_string(),
        }]);
        let after = LlmResponseStreamEvent::new(vec![LlmResponseChunk::MessageStop { reason: None }]);
        let source = stream::iter([
            Ok(failed_event.clone()),
            Ok(after.clone()),
        ]);
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
        };
        let observed = observe(
            response,
            "model/worker",
            Instant::now(),
            stats.clone(),
            0.0,
            Some((
                log,
                RoutingLogContext::from_metadata(&Metadata::default()),
            )),
        );
        let LlmResponse::Stream(mut observed) = observed.llm_response else {
            panic!("expected stream");
        };

        let first = observed.next().await.expect("failing item is still delivered");
        assert_eq!(first.expect("item unchanged"), failed_event);
        assert!(
            observed.next().await.is_none(),
            "the wrapper must stop after a failed item"
        );
        drop(observed);
        assert_eq!(stats.snapshot().models["model/worker"].errors, 1);
    }

    /// An OpenAI Responses client may stop polling immediately after receiving the
    /// terminal `response.completed` event. Switchyard must record usage and routing
    /// data before returning that event because the stream wrapper will not resume
    /// after the client drops it.
    #[tokio::test]
    async fn terminal_event_is_recorded_before_the_client_drops_the_stream() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = SharedRoutingLog::new(dir.path().join("routing.jsonl")).expect("routing log");
        let context = RoutingLogContext::from_metadata(&Metadata {
            session_id: Some("streaming-session".to_string()),
            ..Metadata::default()
        });
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(3),
            ..Usage::default()
        };
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::Usage(usage),
            LlmResponseChunk::MessageStop { reason: None },
        ]))]);
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
        };
        let stats = StatsAccumulator::default();
        let observed = observe(
            response,
            "model/worker",
            Instant::now(),
            stats.clone(),
            0.0,
            Some((log.clone(), context)),
        );

        let LlmResponse::Stream(mut observed) = observed.llm_response else {
            panic!("expected stream");
        };
        assert!(observed.next().await.is_some());
        drop(observed);

        let routing = log
            .snapshot_session("streaming-session")
            .expect("read routing log")
            .expect("terminal event was recorded");
        let routing = serde_json::to_value(routing).expect("serialize routing stats");
        assert_eq!(routing["models"]["model/worker"]["calls"], 1);
        assert_eq!(routing["models"]["model/worker"]["prompt_tokens"], 10);
        assert_eq!(routing["models"]["model/worker"]["completion_tokens"], 3);

        let process = stats.snapshot();
        assert_eq!(process.models["model/worker"].prompt_tokens, 10);
        assert_eq!(process.models["model/worker"].completion_tokens, 3);
    }
}
