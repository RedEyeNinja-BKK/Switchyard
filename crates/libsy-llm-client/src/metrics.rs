// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric labelling inherited from Python

use std::time::{Duration, Instant};

use futures::{StreamExt, stream};
use std::{
    sync::OnceLock,
    sync::atomic::{AtomicU64, Ordering},
};

use opentelemetry::metrics::ObservableGauge;
use opentelemetry::{KeyValue, global};
use switchyard_libsy::Result;
use switchyard_protocol::{LlmResponse, LlmResponseChunk, ModelId, Response};

static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERRORS: AtomicU64 = AtomicU64::new(0);
static TOTAL_GAUGES: OnceLock<(ObservableGauge<u64>, ObservableGauge<u64>)> = OnceLock::new();

/// Registers process-wide compatibility gauges with the installed global meter provider.
pub fn initialize() {
    TOTAL_GAUGES.get_or_init(|| {
        let meter = global::meter("switchyard");
        let requests = meter
            .u64_observable_gauge("switchyard.total_requests")
            .with_callback(|observer| {
                observer.observe(TOTAL_REQUESTS.load(Ordering::Relaxed), &[]);
            })
            .build();
        let errors = meter
            .u64_observable_gauge("switchyard.total_errors")
            .with_callback(|observer| {
                observer.observe(TOTAL_ERRORS.load(Ordering::Relaxed), &[]);
            })
            .build();
        (requests, errors)
    });
}

pub(crate) const fn is_retryable_http_status(status: u16) -> bool {
    status == 408 || status == 429 || (status >= 500 && status <= 599)
}

/// Maps an HTTP status code (or lack thereof) to a normalized outcome metric label:
/// - `"ok"` for HTTP 2xx success responses.
/// - `"retryable_error"` for retryable HTTP statuses (408, 429, 5xx) or non-HTTP failures (`None`).
/// - `"other_error"` for all other client or non-retryable statuses.
pub const fn http_outcome_label(status: Option<u16>) -> &'static str {
    match status {
        Some(200..=299) => "ok",
        Some(status) if is_retryable_http_status(status) => "retryable_error",
        None => "retryable_error",
        Some(_) => "other_error",
    }
}

/// Limit the cardinality of the HTTP status code.
/// This matches Python, but likely we should log the status code directly. Most of these never
/// appear.
pub const fn http_status_code_label(status: Option<u16>) -> &'static str {
    match status {
        None => "none",
        Some(200) => "200",
        Some(400) => "400",
        Some(401) => "401",
        Some(403) => "403",
        Some(404) => "404",
        Some(408) => "408",
        Some(409) => "409",
        Some(422) => "422",
        Some(429) => "429",
        Some(500) => "500",
        Some(502) => "502",
        Some(503) => "503",
        Some(504) => "504",
        Some(100..=199) => "1xx",
        Some(200..=299) => "2xx",
        Some(300..=399) => "3xx",
        Some(400..=499) => "4xx",
        Some(500..=599) => "5xx",
        Some(_) => "other",
    }
}

pub(crate) fn record_upstream_attempt(status: Option<u16>) {
    global::meter("switchyard")
        .u64_counter("switchyard.upstream_attempts")
        .build()
        .add(
            1,
            &[
                KeyValue::new("outcome", http_outcome_label(status)),
                KeyValue::new("code", http_status_code_label(status)),
            ],
        );
}

/// Records an upstream operation that succeeded after at least one retry.
pub(crate) fn record_retry_recovered() {
    global::meter("switchyard")
        .u64_counter("switchyard.router_retry_recovered")
        .build()
        .add(1, &[]);
}

/// Records the time needed to produce the routing outcome, including classifier calls,
/// target resolution, request rewrites, and decision publishing.
pub(crate) fn record_routing_overhead(algorithm: &str, overhead: Duration) {
    global::meter("switchyard")
        .f64_histogram("switchyard.routing_overhead_ms")
        .build()
        .record(
            overhead.as_secs_f64() * 1000.0,
            &[KeyValue::new("algorithm", algorithm.to_string())],
        );
}

/// Records the final outcome, retaining ownership until a streamed answer ends or is dropped.
pub(crate) fn observe_routed_request(
    algorithm: &str,
    selected_model: &ModelId,
    answer_duration: Option<Duration>,
    result: Result<Response>,
) -> Result<Response> {
    let mut call = RoutedCallMetrics {
        algorithm: algorithm.to_owned(),
        model: selected_model.clone(),
        duration: answer_duration,
        started: Instant::now(),
        // A buffered result is terminal here; a stream resets this below and is
        // classified by what the consumer drives it to. Keeping non-stream success
        // and failure as their own states is what stops an ordinary request error
        // from being reported as an abandoned stream.
        outcome: Some(if result.is_ok() {
            CallOutcome::NonStreamOk
        } else {
            CallOutcome::NonStreamError
        }),
    };
    result.map(|mut response| {
        if let LlmResponse::Stream(stream) = response.llm_response {
            call.outcome = None;
            response.llm_response = LlmResponse::Stream(Box::pin(stream::unfold(
                (stream, call),
                |(mut stream, mut call)| async move {
                    let Some(item) = stream.next().await else {
                        // A clean end of body without a terminal chunk is how some
                        // providers finish. Preserve the existing success verdict.
                        call.complete(CallOutcome::Eof);
                        return None;
                    };
                    match &item {
                        Err(_) => call.outcome = Some(CallOutcome::RawStreamError),
                        Ok(event) => {
                            for chunk in event.normalized() {
                                match chunk {
                                    LlmResponseChunk::StreamError { .. } => {
                                        call.outcome = Some(CallOutcome::StreamError)
                                    }
                                    LlmResponseChunk::DecodeError { .. } => {
                                        call.outcome = Some(CallOutcome::DecodeError)
                                    }
                                    LlmResponseChunk::MessageStop { .. } => {
                                        call.complete(CallOutcome::MessageStop);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    Some((item, (stream, call)))
                },
            )));
        }
        response
    })
}

/// How one routed call terminated.
///
/// Every distinguishable terminal condition has its own variant so the failure
/// classes stay separable in the metric. Two rules carry over from the boolean
/// this replaces, and are load-bearing:
///
/// - a clean end of stream without a terminal chunk (`Eof`) is a **success**, as is
///   an explicit terminal chunk (`MessageStop`);
/// - an explicit failure is never overwritten by a later success, so a stream that
///   fails and then closes is reported as a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallOutcome {
    /// A buffered (non-streaming) response was produced.
    NonStreamOk,
    /// The call failed before any response body existed: routing, transport,
    /// provider or translation error returned as `Err` by the algorithm.
    NonStreamError,
    /// The stream delivered its explicit terminal chunk.
    MessageStop,
    /// The stream ended without a terminal chunk: the upstream closed the body.
    Eof,
    /// The stream delivered an in-band upstream failure event.
    StreamError,
    /// The stream delivered an in-band decode-failure event.
    DecodeError,
    /// A stream item was itself an error (transport or decoding at the boundary).
    RawStreamError,
    /// The stream was dropped before reaching any terminal condition: the consumer
    /// stopped reading. This proves premature abandonment, not who caused it.
    Dropped,
}

impl CallOutcome {
    /// Whether this terminal condition counts as a successful call.
    pub(crate) const fn is_success(self) -> bool {
        matches!(self, Self::NonStreamOk | Self::MessageStop | Self::Eof)
    }

    /// Bounded `reason` label for the failed-call increment.
    ///
    /// Successes are never recorded on the errors counter, so they have none.
    pub(crate) const fn failure_reason(self) -> Option<&'static str> {
        match self {
            Self::NonStreamOk | Self::MessageStop | Self::Eof => None,
            Self::NonStreamError => Some("non_stream_error"),
            Self::StreamError => Some("stream_error"),
            Self::DecodeError => Some("decode_error"),
            Self::RawStreamError => Some("raw_stream_error"),
            Self::Dropped => Some("dropped"),
        }
    }
}

struct RoutedCallMetrics {
    algorithm: String,
    model: ModelId,
    duration: Option<Duration>,
    started: Instant,
    // `None` means no terminal condition has been observed yet. Only a live stream
    // can be in that state, and dropping it is `Dropped`.
    outcome: Option<CallOutcome>,
}

impl RoutedCallMetrics {
    /// Records a success signal unless a terminal condition is already known.
    ///
    /// This is the `get_or_insert` rule the boolean carried: a later success never
    /// overwrites an earlier failure.
    fn complete(&mut self, outcome: CallOutcome) {
        self.outcome.get_or_insert(outcome);
    }
}

impl Drop for RoutedCallMetrics {
    fn drop(&mut self) {
        let outcome = self.outcome.unwrap_or(CallOutcome::Dropped);
        let duration = self
            .duration
            .map(|duration| duration + self.started.elapsed());
        if let Some(duration) = duration {
            record_answer_call(&self.algorithm, &self.model, duration, outcome.is_success());
        }
        record_routed_request(&self.model, duration, outcome);
    }
}

/// Records one terminal model call made after routing, preserving the libsy call metric surface.
pub(crate) fn record_answer_call(
    algorithm: &str,
    selected_model: &ModelId,
    duration: Duration,
    is_success: bool,
) {
    let attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("selected_model", selected_model.to_string()),
        KeyValue::new("outcome", if is_success { "ok" } else { "error" }),
    ];
    let meter = global::meter("switchyard");
    meter
        .u64_counter("switchyard.llm_calls")
        .build()
        .add(1, &attributes);
    meter
        .f64_histogram("switchyard.llm_call_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &attributes);
}

/// Records one terminal routed request after the algorithm has produced an outcome.
///
/// `switchyard.errors` gains a bounded `reason` attribute on the failure increment.
/// The metric name, the `model` label and the success/error decision are unchanged,
/// so `sum by (model) (rate(switchyard_errors_total[5m]))` still reconciles with the
/// pre-change value. Consumers selecting `switchyard_errors_total{model="..."}`
/// without aggregation now see one series per reason instead of one per model.
pub(crate) fn record_routed_request(
    selected_model: &ModelId,
    answer_duration: Option<Duration>,
    outcome: CallOutcome,
) {
    TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    let attributes = [KeyValue::new("model", selected_model.to_string())];
    let meter = global::meter("switchyard");
    if outcome.is_success() {
        meter
            .u64_counter("switchyard.requests")
            .build()
            .add(1, &attributes);
        if let Some(duration) = answer_duration {
            meter
                .f64_histogram("switchyard.model_call_latency_ms")
                .build()
                .record(duration.as_secs_f64() * 1000.0, &attributes);
        }
    } else {
        TOTAL_ERRORS.fetch_add(1, Ordering::Relaxed);
        let attributes = [
            KeyValue::new("model", selected_model.to_string()),
            KeyValue::new("reason", outcome.failure_reason().unwrap_or("unknown")),
        ];
        meter
            .u64_counter("switchyard.errors")
            .build()
            .add(1, &attributes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_labels_match_the_retry_policy() {
        assert_eq!(http_outcome_label(Some(200)), "ok");
        for status in [408, 429, 500, 502, 599] {
            assert!(is_retryable_http_status(status));
            assert_eq!(http_outcome_label(Some(status)), "retryable_error");
        }
        for status in [400, 409, 499, 600] {
            assert!(!is_retryable_http_status(status));
            assert_eq!(http_outcome_label(Some(status)), "other_error");
        }
        assert_eq!(http_outcome_label(None), "retryable_error");
    }

    // -- CallOutcome reason instrumentation -----------------------------------

    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
    use switchyard_libsy::LibsyError;
    use switchyard_protocol::{LlmClientError, LlmResponseStreamEvent, text_response};

    /// One in-memory metric pipeline for the whole unit-test binary.
    ///
    /// The global meter provider can be installed only once per process, so each test
    /// filters by its own model name rather than installing its own pipeline.
    fn telemetry() -> &'static (InMemoryMetricExporter, SdkMeterProvider) {
        static TELEMETRY: OnceLock<(InMemoryMetricExporter, SdkMeterProvider)> = OnceLock::new();
        TELEMETRY.get_or_init(|| {
            let exporter = InMemoryMetricExporter::default();
            let reader = PeriodicReader::builder(exporter.clone()).build();
            let provider = SdkMeterProvider::builder().with_reader(reader).build();
            opentelemetry::global::set_meter_provider(provider.clone());
            (exporter, provider)
        })
    }

    /// Metrics are process-global, so tests that record them must not overlap.
    ///
    /// This also installs the pipeline before the caller records anything. An
    /// instrument created while the global provider is still the no-op default is
    /// never reported, so initialising lazily inside `flush()` would silently drop
    /// the first test's samples.
    async fn serialize() -> tokio::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
        let guard = LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        telemetry();
        guard
    }

    fn flush() -> Vec<ResourceMetrics> {
        let (exporter, provider) = telemetry();
        provider.force_flush().expect("force_flush failed");
        exporter
            .get_finished_metrics()
            .expect("get_finished_metrics failed")
    }

    fn attributes_match<'a>(
        attributes: impl Iterator<Item = &'a KeyValue>,
        wanted: &[(&str, &str)],
    ) -> bool {
        let present: Vec<(String, String)> = attributes
            .map(|kv| (kv.key.as_str().to_string(), kv.value.as_str().to_string()))
            .collect();
        wanted
            .iter()
            .all(|(key, value)| present.iter().any(|(k, v)| k == key && v == value))
    }

    /// Latest cumulative value of `name` for every wanted attribute.
    fn counter_with(snapshots: &[ResourceMetrics], name: &str, wanted: &[(&str, &str)]) -> u64 {
        snapshots
            .iter()
            .flat_map(|snapshot| snapshot.scope_metrics())
            .filter(|scope| scope.scope().name() == "switchyard")
            .flat_map(|scope| scope.metrics())
            .filter(|metric| metric.name() == name)
            .filter_map(|metric| match metric.data() {
                AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
                    .data_points()
                    .filter(|point| attributes_match(point.attributes(), wanted))
                    .map(|point| point.value())
                    .max(),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// `switchyard.errors` for one model across every reason: what
    /// `sum by (model) (rate(switchyard_errors_total[5m]))` reconciles to.
    fn errors_for_model(snapshots: &[ResourceMetrics], model: &str) -> u64 {
        snapshots
            .iter()
            .flat_map(|snapshot| snapshot.scope_metrics())
            .filter(|scope| scope.scope().name() == "switchyard")
            .flat_map(|scope| scope.metrics())
            .filter(|metric| metric.name() == "switchyard.errors")
            .filter_map(|metric| match metric.data() {
                AggregatedMetrics::U64(MetricData::Sum(sum)) => Some(
                    sum.data_points()
                        .filter(|point| attributes_match(point.attributes(), &[("model", model)]))
                        .map(|point| point.value())
                        .sum::<u64>(),
                ),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    fn errors_with_reason(snapshots: &[ResourceMetrics], model: &str, reason: &str) -> u64 {
        counter_with(
            snapshots,
            "switchyard.errors",
            &[("model", model), ("reason", reason)],
        )
    }

    fn aggregate_response() -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(Some("agg".to_string()), "hello")),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn stream_response(
        items: Vec<std::result::Result<LlmResponseStreamEvent, LlmClientError>>,
    ) -> Response {
        Response {
            llm_response: LlmResponse::Stream(Box::pin(stream::iter(items))),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn event(
        chunks: Vec<LlmResponseChunk>,
    ) -> std::result::Result<LlmResponseStreamEvent, LlmClientError> {
        Ok(LlmResponseStreamEvent::new(chunks))
    }

    fn text_delta() -> LlmResponseChunk {
        LlmResponseChunk::TextDelta {
            index: 0,
            text: "x".to_string(),
        }
    }

    fn message_stop() -> LlmResponseChunk {
        LlmResponseChunk::MessageStop {
            reason: Some("stop".to_string()),
        }
    }

    fn stream_error_chunk() -> LlmResponseChunk {
        LlmResponseChunk::StreamError {
            message: "upstream failed mid-stream".to_string(),
        }
    }

    fn decode_error_chunk() -> LlmResponseChunk {
        LlmResponseChunk::DecodeError {
            message: "malformed event".to_string(),
        }
    }

    /// Drives a stream to exhaustion, then drops it, so the terminal condition is
    /// recorded before metrics are read.
    async fn drain(response: Response) {
        let mut response = response;
        if let LlmResponse::Stream(stream) = &mut response.llm_response {
            while stream.next().await.is_some() {}
        }
        drop(response);
    }

    /// Reads one item and abandons the stream: the consumer-stopped-reading case.
    async fn read_one_then_drop(response: Response) {
        let mut response = response;
        if let LlmResponse::Stream(stream) = &mut response.llm_response {
            assert!(stream.next().await.is_some(), "expected at least one item");
        }
        drop(response);
    }

    fn non_stream_failure() -> LibsyError {
        LibsyError::AlgorithmError {
            message: "no candidate served the request".to_string(),
        }
    }

    #[test]
    fn failure_reasons_are_bounded_and_distinct() {
        let failures = [
            CallOutcome::NonStreamError,
            CallOutcome::StreamError,
            CallOutcome::DecodeError,
            CallOutcome::RawStreamError,
            CallOutcome::Dropped,
        ];
        let reasons: Vec<&str> = failures
            .iter()
            .map(|outcome| outcome.failure_reason().expect("a failure has a reason"))
            .collect();
        assert_eq!(
            reasons,
            vec![
                "non_stream_error",
                "stream_error",
                "decode_error",
                "raw_stream_error",
                "dropped"
            ]
        );
        let mut distinct = reasons.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), reasons.len(), "reasons must be distinct");

        for outcome in failures {
            assert!(!outcome.is_success(), "{outcome:?} must not be a success");
        }
        for outcome in [
            CallOutcome::NonStreamOk,
            CallOutcome::MessageStop,
            CallOutcome::Eof,
        ] {
            assert!(outcome.is_success(), "{outcome:?} must be a success");
            assert!(
                outcome.failure_reason().is_none(),
                "a success must not carry a failure reason"
            );
        }
    }

    #[tokio::test]
    async fn non_stream_ok_is_a_request_success() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-nonstream-ok");
        let result = observe_routed_request("test", &model, None, Ok(aggregate_response()));
        assert!(result.is_ok());
        drop(result);

        let snapshots = flush();
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.requests",
                &[("model", "r1-nonstream-ok")]
            ),
            1
        );
        assert_eq!(errors_for_model(&snapshots, "r1-nonstream-ok"), 0);
    }

    #[tokio::test]
    async fn non_stream_error_is_not_reported_as_dropped() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-nonstream-err");
        let result = observe_routed_request("test", &model, None, Err(non_stream_failure()));
        assert!(result.is_err());
        drop(result);

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-nonstream-err", "non_stream_error"),
            1
        );
        assert_eq!(
            errors_with_reason(&snapshots, "r1-nonstream-err", "dropped"),
            0,
            "a call that never produced a stream must not be reported as abandoned"
        );
    }

    #[tokio::test]
    async fn clean_eof_without_message_stop_is_a_success() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-eof");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![event(vec![text_delta()])])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            counter_with(&snapshots, "switchyard.requests", &[("model", "r1-eof")]),
            1
        );
        assert_eq!(
            errors_for_model(&snapshots, "r1-eof"),
            0,
            "a clean end of body is a success, as it was before"
        );
    }

    #[tokio::test]
    async fn message_stop_is_a_success() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-stop");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![
                event(vec![text_delta()]),
                event(vec![message_stop()]),
            ])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            counter_with(&snapshots, "switchyard.requests", &[("model", "r1-stop")]),
            1
        );
        assert_eq!(errors_for_model(&snapshots, "r1-stop"), 0);
    }

    #[tokio::test]
    async fn abandoning_the_stream_is_reported_as_dropped() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-dropped");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![
                event(vec![text_delta()]),
                event(vec![message_stop()]),
            ])),
        )
        .expect("stream response");
        read_one_then_drop(response).await;

        let snapshots = flush();
        assert_eq!(errors_with_reason(&snapshots, "r1-dropped", "dropped"), 1);
        assert_eq!(errors_for_model(&snapshots, "r1-dropped"), 1);
    }

    #[tokio::test]
    async fn stream_error_chunk_is_classified() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-stream-error");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![event(vec![stream_error_chunk()])])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-stream-error", "stream_error"),
            1
        );
        assert_eq!(errors_for_model(&snapshots, "r1-stream-error"), 1);
    }

    #[tokio::test]
    async fn decode_error_chunk_is_classified() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-decode-error");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![event(vec![decode_error_chunk()])])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-decode-error", "decode_error"),
            1
        );
        assert_eq!(errors_for_model(&snapshots, "r1-decode-error"), 1);
    }

    #[tokio::test]
    async fn raw_stream_error_is_classified() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-raw-stream-error");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![
                event(vec![text_delta()]),
                Err(LlmClientError::TemporarilyUnavailable),
            ])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-raw-stream-error", "raw_stream_error"),
            1
        );
        assert_eq!(errors_for_model(&snapshots, "r1-raw-stream-error"), 1);
    }

    #[tokio::test]
    async fn an_earlier_failure_survives_a_later_eof() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-error-then-eof");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![event(vec![stream_error_chunk()])])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-error-then-eof", "stream_error"),
            1
        );
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.requests",
                &[("model", "r1-error-then-eof")]
            ),
            0,
            "a failure must not be reclassified as success by a later clean close"
        );
    }

    #[tokio::test]
    async fn an_earlier_failure_survives_a_later_message_stop() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-error-then-stop");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![
                event(vec![stream_error_chunk()]),
                event(vec![message_stop()]),
            ])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-error-then-stop", "stream_error"),
            1
        );
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.requests",
                &[("model", "r1-error-then-stop")]
            ),
            0
        );
    }

    #[tokio::test]
    async fn a_late_error_overrides_an_earlier_success() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-late-error");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![
                event(vec![message_stop()]),
                event(vec![stream_error_chunk()]),
            ])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-late-error", "stream_error"),
            1
        );
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.requests",
                &[("model", "r1-late-error")]
            ),
            0,
            "an error arriving after a terminal chunk still makes the call a failure"
        );
    }

    #[tokio::test]
    async fn a_failed_call_increments_the_error_counter_exactly_once() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-single-increment");
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![
                event(vec![stream_error_chunk()]),
                event(vec![decode_error_chunk()]),
                event(vec![stream_error_chunk()]),
            ])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_for_model(&snapshots, "r1-single-increment"),
            1,
            "three failure signals in one call must produce one error, not three"
        );
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.requests",
                &[("model", "r1-single-increment")]
            ),
            0
        );
    }

    #[tokio::test]
    async fn llm_calls_stays_gated_on_a_recorded_duration() {
        let _guard = serialize().await;
        let without_duration = ModelId::from("r1-no-duration");
        let _ = observe_routed_request("test", &without_duration, None, Ok(aggregate_response()));
        let with_duration = ModelId::from("r1-with-duration");
        let _ = observe_routed_request(
            "test",
            &with_duration,
            Some(Duration::from_millis(5)),
            Ok(aggregate_response()),
        );

        let snapshots = flush();
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.llm_calls",
                &[("selected_model", "r1-no-duration")]
            ),
            0,
            "llm_calls must stay gated on a recorded duration"
        );
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.llm_calls",
                &[("selected_model", "r1-with-duration")]
            ),
            1
        );

        // A failure is recorded on switchyard.errors whether or not a duration exists.
        let failed_without_duration = ModelId::from("r1-fail-no-duration");
        let _ = observe_routed_request(
            "test",
            &failed_without_duration,
            None,
            Err(non_stream_failure()),
        );
        let snapshots = flush();
        assert_eq!(errors_for_model(&snapshots, "r1-fail-no-duration"), 1);
        assert_eq!(
            counter_with(
                &snapshots,
                "switchyard.llm_calls",
                &[("selected_model", "r1-fail-no-duration")]
            ),
            0
        );
    }

    #[tokio::test]
    async fn per_reason_series_reconcile_with_the_model_total() {
        let _guard = serialize().await;
        let model = ModelId::from("r1-reconcile");
        let _ = observe_routed_request("test", &model, None, Err(non_stream_failure()));
        let response = observe_routed_request(
            "test",
            &model,
            None,
            Ok(stream_response(vec![event(vec![stream_error_chunk()])])),
        )
        .expect("stream response");
        drain(response).await;

        let snapshots = flush();
        assert_eq!(
            errors_with_reason(&snapshots, "r1-reconcile", "non_stream_error"),
            1
        );
        assert_eq!(
            errors_with_reason(&snapshots, "r1-reconcile", "stream_error"),
            1
        );
        assert_eq!(
            errors_for_model(&snapshots, "r1-reconcile"),
            2,
            "sum by (model) must still reconcile with the per-reason series"
        );
    }
}
