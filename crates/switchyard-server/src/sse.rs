// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing helpers for OpenAI, Anthropic, and Responses endpoints.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use serde_json::{Value, json};
use switchyard_runner::stream_error_summary;
use switchyard_translation::{LlmStreamError, RawEventStream, WireFormat};

use crate::redaction::Redactor;

// An HTTP SSE response body's lifetime cannot prove who initiated its closure.
// A downstream socket, intermediary, or server cancellation can drop the body.
struct SseBodyGuard {
    correlation_id: String,
    armed: bool,
}

impl SseBodyGuard {
    fn new(correlation_id: Option<String>) -> Self {
        Self {
            correlation_id: correlation_id.unwrap_or_default(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SseBodyGuard {
    fn drop(&mut self) {
        if self.armed {
            tracing::info!(
                target: "switchyard_server::request",
                correlation_id = %self.correlation_id,
                stream_boundary = "sse_http_body",
                body_outcome = "dropped_before_terminal_frame",
                "SSE body dropped before terminal frame was yielded"
            );
        }
    }
}

/// Boxed stream type accepted by Axum's SSE response wrapper.
pub(crate) type SseFrameStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// Converts translated JSON events into endpoint-specific SSE frames.
pub(crate) fn frame_stream(
    stream: RawEventStream,
    target_format: WireFormat,
    redactor: Arc<Redactor>,
    correlation_id: Option<String>,
) -> Sse<SseFrameStream> {
    // Build before the lazy generator to observe even an unpolled body.
    let mut body_guard = SseBodyGuard::new(correlation_id);
    let framed = async_stream::stream! {
        let mut stream = stream;
        let mut failed = false;
        while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
            let successful_terminal = match (&item, target_format) {
                (Ok(value), WireFormat::OpenAiResponses) => {
                    value.get("type").and_then(Value::as_str) == Some("response.completed")
                }
                (Ok(value), WireFormat::AnthropicMessages) => {
                    value.get("type").and_then(Value::as_str) == Some("message_stop")
                }
                _ => false,
            };
            let event = match item {
                Ok(value) => match frame_event(target_format, value, &redactor) {
                    Ok(event) => event,
                    Err(error) => {
                        failed = true;
                        error_event(target_format, error.to_string(), &redactor)
                    }
                },
                // Preserve the upstream error's fields, apart from credential redaction,
                // rather than replacing it with a synthesized error.
                Err(LlmStreamError::Upstream(value)) => {
                    failed = true;
                    frame_event(target_format, value.clone(), &redactor).unwrap_or_else(|error| {
                        tracing::warn!(error = %error, "in-band error event could not be framed");
                        error_event(target_format, value.to_string(), &redactor)
                    })
                }
                Err(LlmStreamError::Client(error)) => {
                    // The error text can quote request content, so the log
                    // records only the stable error class.
                    let summary = stream_error_summary(&error, None);
                    tracing::warn!(
                        error.kind = summary.kind.as_str(),
                        error.upstream_status = summary.upstream_status,
                        "stream iteration failed"
                    );
                    failed = true;
                    error_event(target_format, error.to_string(), &redactor)
                }
            };
            // Yielded into the framing stream is NOT confirmation of delivery.
            if successful_terminal || failed {
                body_guard.disarm();
            }
            yield Ok(event);
            if failed {
                break;
            }
        }

        // `[DONE]` is the OpenAI Chat success sentinel: clients stop reading there and
        // keep what they have as a finished answer, so it must not follow a failed turn.
        if !failed && target_format == WireFormat::OpenAiChat {
            body_guard.disarm();
            yield Ok(Event::default().data("[DONE]"));
        }
        // EOF is not a body cancellation; a missing upstream terminal is separate.
        body_guard.disarm();
    };

    Sse::new(Box::pin(framed) as SseFrameStream)
}

fn frame_event(
    target_format: WireFormat,
    value: Value,
    redactor: &Redactor,
) -> Result<Event, serde_json::Error> {
    let data = redactor.json(serde_json::to_string(&value)?);
    match target_format {
        WireFormat::OpenAiChat => Ok(Event::default().data(data)),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_string();
            Ok(Event::default().event(event_type).data(data))
        }
    }
}

fn error_event(target_format: WireFormat, message: String, redactor: &Redactor) -> Event {
    match target_format {
        WireFormat::OpenAiChat => Event::default().data(
            redactor.json(
                json!({
                    "error": {
                        "message": message,
                        "type": "SwitchyardError",
                    }
                })
                .to_string(),
            ),
        ),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let error_type = if target_format == WireFormat::AnthropicMessages {
                "api_error"
            } else {
                "SwitchyardError"
            };
            Event::default().event("error").data(
                redactor.json(
                    json!({
                        "type": "error",
                        "error": {
                            "message": message,
                            "type": error_type,
                        }
                    })
                    .to_string(),
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use axum::{body::to_bytes, response::IntoResponse};
    use futures_util::{StreamExt, stream};
    use switchyard_protocol::LlmClientError;
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;

    type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

    // Renders a framed body for one Chat stream.
    async fn chat_body(items: Vec<Result<Value, LlmStreamError>>) -> TestResult<String> {
        let stream: RawEventStream = Box::pin(stream::iter(items));
        let response = frame_stream(
            stream,
            WireFormat::OpenAiChat,
            Arc::new(Redactor::default()),
            None,
        )
        .into_response();
        Ok(String::from_utf8(
            to_bytes(response.into_body(), usize::MAX).await?.to_vec(),
        )?)
    }

    #[tokio::test]
    async fn stream_error_terminates_without_done_marker() -> TestResult {
        let failure = LlmClientError::General("boom".to_string());
        let body = chat_body(vec![
            Ok(json!({"id": "before"})),
            Err(LlmStreamError::Client(failure)),
            Ok(json!({"id": "after"})),
        ])
        .await?;

        // A stream error is terminal: later chunks and success markers must not be emitted.
        assert!(body.contains("before"));
        assert!(body.contains("boom"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
        Ok(())
    }

    #[tokio::test]
    async fn in_band_error_is_forwarded_verbatim_without_done_marker() -> TestResult {
        let upstream_error = json!({
            "error": {"code": "stream_failed", "message": "boom", "type": "upstream_stream_error"}
        });
        let body = chat_body(vec![
            Ok(json!({"id": "before"})),
            Err(LlmStreamError::Upstream(upstream_error)),
            Ok(json!({"id": "after"})),
        ])
        .await?;

        // The upstream owns this error, so its code and type reach the client unchanged
        // rather than being flattened into a synthesized SwitchyardError frame.
        assert!(body.contains("before"));
        assert!(body.contains("stream_failed"));
        assert!(body.contains("upstream_stream_error"));
        assert!(!body.contains("SwitchyardError"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
        Ok(())
    }

    // Collects rendered warn events so the test can assert against the final
    // log sink rather than a field mid-pipeline.
    #[derive(Clone, Default)]
    struct WarnCapture(Arc<std::sync::Mutex<Vec<String>>>);

    impl tracing_subscriber::Layer<tracing_subscriber::Registry> for WarnCapture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, tracing_subscriber::Registry>,
        ) {
            if *event.metadata().level() != tracing::Level::WARN {
                return;
            }
            let mut fields = String::new();
            event.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    fields.push_str(&format!("{}={value:?} ", field.name()));
                },
            );
            self.0.lock().unwrap().push(fields);
        }
    }

    // The client error text can quote request content, so the stream-failure
    // warn log records only the stable error class.
    #[test]
    fn stream_client_error_warn_redacts_upstream_body() -> TestResult {
        const LEAKED: &str = "SECRET-quoted-request-content";
        let capture = WarnCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let body = tracing::subscriber::with_default(subscriber, || {
            runtime.block_on(chat_body(vec![Err(LlmStreamError::Client(
                LlmClientError::UpstreamHttp {
                    status: axum::http::StatusCode::BAD_GATEWAY,
                    body: format!("upstream failed: {LEAKED}"),
                },
            ))]))
        })?;

        // The client still sees the error text in-band.
        assert!(body.contains(LEAKED), "{body}");

        let events = capture.0.lock().unwrap().clone();
        assert!(
            events
                .iter()
                .any(|event| event.contains("stream iteration failed")),
            "{events:?}"
        );
        assert!(
            !events.iter().any(|event| event.contains(LEAKED)),
            "{events:?}"
        );
        Ok(())
    }

    // Observe only INFO-level body-boundary events: the regular stream guard
    // records its own WARN separately, so this must not create duplicate WARNs.
    #[derive(Clone, Default)]
    struct BodyInfoCapture(Arc<std::sync::Mutex<Vec<String>>>);

    impl tracing_subscriber::Layer<tracing_subscriber::Registry> for BodyInfoCapture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, tracing_subscriber::Registry>,
        ) {
            if *event.metadata().level() != tracing::Level::INFO {
                return;
            }
            let mut fields = String::new();
            event.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    fields.push_str(&format!("{}={value:?} ", field.name()));
                },
            );
            self.0.lock().unwrap().push(fields);
        }
    }

    #[test]
    fn sse_body_boundary_distinguishes_abandonment_from_normal_terminal() -> TestResult {
        let capture = BodyInfoCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        tracing::subscriber::with_default(subscriber, || {
            runtime.block_on(async {
                // Stop after a data frame, before response.completed is framed.
                let early: RawEventStream = Box::pin(stream::iter(vec![
                    Ok(json!({"type":"response.output_text.delta","delta":"example"})),
                    Ok(json!({"type":"response.completed"})),
                ]));
                let body = frame_stream(
                    early,
                    WireFormat::OpenAiResponses,
                    Arc::new(Redactor::default()),
                    Some("corr-early".to_string()),
                )
                .into_response()
                .into_body();
                let mut body = body.into_data_stream();
                assert!(body.next().await.is_some());
                drop(body);

                // A completed stream must not produce a body-abandonment event.
                let finished: RawEventStream = Box::pin(stream::iter(vec![
                    Ok(json!({"type":"response.output_text.delta","delta":"example"})),
                    Ok(json!({"type":"response.completed"})),
                ]));
                let response = frame_stream(
                    finished,
                    WireFormat::OpenAiResponses,
                    Arc::new(Redactor::default()),
                    Some("corr-finished".to_string()),
                )
                .into_response();
                let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();

                // Never-polled body is also dropped before a terminal frame.
                let unpolled: RawEventStream =
                    Box::pin(stream::iter(vec![Ok(json!({"type":"response.completed"}))]));
                let response = frame_stream(
                    unpolled,
                    WireFormat::OpenAiResponses,
                    Arc::new(Redactor::default()),
                    Some("corr-unpolled".to_string()),
                )
                .into_response();
                drop(response);

                // Explicit upstream error has its own classification, not a
                // misleading second body-abandonment observation.
                let failed: RawEventStream = Box::pin(stream::iter(vec![Err(
                    LlmStreamError::Client(LlmClientError::General("error".to_string())),
                )]));
                let response = frame_stream(
                    failed,
                    WireFormat::OpenAiResponses,
                    Arc::new(Redactor::default()),
                    Some("corr-failed".to_string()),
                )
                .into_response();
                let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            });
        });

        let events = capture.0.lock().unwrap().clone();
        let dropped: Vec<_> = events
            .iter()
            .filter(|line| line.contains("dropped_before_terminal_frame"))
            .collect();
        assert_eq!(dropped.len(), 2, "{events:?}");
        assert!(
            dropped.iter().any(|line| line.contains("corr-early")),
            "{events:?}"
        );
        assert!(
            dropped.iter().any(|line| line.contains("corr-unpolled")),
            "{events:?}"
        );
        assert!(
            !events.iter().any(|line| line.contains("corr-finished")),
            "{events:?}"
        );
        assert!(
            !events.iter().any(|line| line.contains("corr-failed")),
            "{events:?}"
        );
        assert!(
            !events.iter().any(|line| line.contains("client_closed")),
            "{events:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn anthropic_stream_error_uses_api_error_type() -> TestResult {
        let failure = LlmClientError::General("boom".to_string());
        let stream: RawEventStream =
            Box::pin(stream::iter(vec![Err(LlmStreamError::Client(failure))]));
        let response = frame_stream(
            stream,
            WireFormat::AnthropicMessages,
            Arc::new(Redactor::default()),
            None,
        )
        .into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;
        let data = body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .ok_or("missing data line")?;
        assert_eq!(
            serde_json::from_str::<Value>(data)?,
            json!({"type": "error", "error": {"type": "api_error", "message": "boom"}})
        );
        Ok(())
    }
}
