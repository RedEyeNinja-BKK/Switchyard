// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response usage and full-turn latency metrics for routed requests.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use opentelemetry::{KeyValue, global};
use switchyard_protocol::{
    LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Response, Usage,
};

use crate::SharedRoutingLog;
use std::sync::Arc;

use crate::economics;
use crate::routing_log::RoutingLogContext;
use crate::stats::{StatsAccumulator, TokenUsage};

/// The observations a routed response produces.
///
/// Grouped into one value because the observation path carries several pieces of
/// request-scoped context, and a long positional list makes it easy to pass the
/// wrong one to the wrong field.
pub(crate) struct ObservationContext {
    /// The model that served the request.
    pub model: String,
    /// The instant the request arrived.
    pub started: Instant,
    /// The share of the prompt that was cache-eligible.
    pub cache_eligible: f64,
    /// The durable routing record, when the deployment records one.
    pub routing_log: Option<(SharedRoutingLog, RoutingLogContext)>,
    /// The composition facts for this request, appended to the same record as the
    /// served model. Owned so the streaming wrapper keeps them for its lifetime
    /// without a leak.
    pub composition: Option<Arc<switchyard_runner::composition::CompositionMetadata>>,
    /// The economics attribution of this request: the provider that served it, the
    /// route the caller addressed, and how it is billed. Resolved by the caller from
    /// the deployment configuration, so the attribution is a configuration fact
    /// rather than a guess made here.
    pub attribution: Option<economics::metrics::UsageAttribution>,
    /// The Bangkok calendar-day window, in which the same usage is counted so the
    /// day view and the rolling counters cannot disagree.
    pub calendar: Arc<economics::CalendarWindow>,
    /// The request correlation id, so a stream-terminal warning can be attributed to
    /// the request that produced it. The terminal request log already carries this;
    /// without it here a stream warning can only be matched by approximate timestamp,
    /// which is not attribution. Empty when the caller supplied none, matching the
    /// terminal log's own convention.
    pub correlation_id: Option<String>,
}

/// Observes a routed response without changing its aggregate or streaming contents.
pub(crate) fn observe(
    response: Response,
    stats: StatsAccumulator,
    context: ObservationContext,
) -> Response {
    let ObservationContext {
        model,
        started,
        cache_eligible,
        routing_log,
        composition,
        attribution,
        calendar,
        correlation_id,
    } = context;
    let Response {
        llm_response,
        metadata,
        upstream_headers,
    } = response;

    let llm_response = match llm_response {
        LlmResponse::Agg(agg) => {
            record_terminal(
                &stats,
                &agg.usage,
                &model,
                started,
                cache_eligible,
                attribution.as_ref(),
                &calendar,
            );
            if let Some((log, context)) = routing_log {
                match composition {
                    Some(composition) => {
                        log.append_composition(context, &model, None, &agg.usage, &composition);
                    }
                    None => log.append(context, &model, None, &agg.usage),
                }
            }
            LlmResponse::Agg(agg)
        }
        LlmResponse::Stream(mut stream) => {
            // Constructed BEFORE the generator, not inside it: `async_stream::stream!`
            // does not run its body until the first poll, so a guard built in the body
            // does not exist for a response the consumer never polls. Such a stream is
            // still abandoned, and the routed-call metric already counts it as a failure,
            // so constructing it here is what keeps the log and the counter agreeing.
            // The generator captures it by move and it is dropped with the response,
            // whether or not that response was ever polled.
            let mut drop_reporter =
                StreamDropReporter::new(&model, correlation_id.as_deref().unwrap_or(""));
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                let mut terminal_seen = false;
                let mut recorded = false;
                while let Some(item) = stream.next().await {
                    let termination = item_termination(&item);
                    let failed = termination.is_some();
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
                    if let Some(reason) = termination {
                        record_stream_error(&stats, &model);
                        // A streamed request has already answered HTTP 200, so the
                        // terminal request log line cannot report this death. Emit a
                        // distinct event with the served model, a stage marker and the
                        // category that ended the stream, so a mid-stream failure is
                        // observable instead of looking like a clean 200, and is not left
                        // as four identical warnings nobody can attribute.
                        //
                        // `upstream_status` is 0 when the failure carried no HTTP status,
                        // because 0 is not a status any provider can return. The provider's
                        // text is deliberately absent: `StreamError` and `DecodeError` carry
                        // a human-readable message that originates upstream, and a log is
                        // not a place for it.
                        tracing::event!(
                            target: "switchyard_server::request",
                            tracing::Level::WARN,
                            selected_model = %model,
                            failure_stage = "stream",
                            termination_reason = reason.as_str(),
                            upstream_status = item_termination_status(&item),
                            correlation_id = %correlation_id.as_deref().unwrap_or(""),
                            "LLM stream failed"
                        );
                        // The failure is now reported; dropping the stream afterwards is
                        // not a second event.
                        drop_reporter.disarm();
                    }
                    // A terminal event ends the stream on its own terms: a caller that
                    // then stops reading is behaving normally, not abandoning it.
                    if terminal_seen {
                        drop_reporter.disarm();
                    }
                    // Responses clients may stop polling immediately after the terminal event.
                    // Commit first when that event already carries the final usage.
                    if !failed && !recorded && terminal_seen
                        && let Some(usage) = latest_usage.as_ref()
                    {
                        record_terminal(
                            &stats,
                            usage,
                            &model,
                            started,
                            cache_eligible,
                            attribution.as_ref(),
                            &calendar,
                        );
                        // This early-commit path is the normal shape for a Responses
                        // stream whose terminal event already carries usage. It must carry
                        // the same composition facts as every other append, or a streamed
                        // request writes a record that disagrees with the counters.
                        if let Some((log, context)) = routing_log.as_ref() {
                            match composition.as_ref() {
                                Some(composition) => {
                                    log.append_composition(
                                        context.clone(),
                                        &model,
                                        None,
                                        usage,
                                        composition,
                                    );
                                }
                                None => log.append(context.clone(), &model, None, usage),
                            }
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
                    record_terminal(
                        &stats,
                        &usage,
                        &model,
                        started,
                        cache_eligible,
                        attribution.as_ref(),
                        &calendar,
                    );
                    if let Some((log, context)) = routing_log {
                        match composition {
                            Some(composition) => log.append_composition(context, &model, None, &usage, &composition),
                            None => log.append(context, &model, None, &usage),
                        }
                    }
                }
                // The stream ended on its own, so its drop is not an abandoned one.
                drop_reporter.disarm();
            };
            LlmResponse::Stream(Box::pin(wrapped))
        }
    };

    Response {
        llm_response,
        metadata,
        upstream_headers,
    }
}

// Records a terminal stream failure after the routed call was already counted.
fn record_stream_error(stats: &StatsAccumulator, model: &str) {
    stats.record_response_error(model);
}

/// Why a routed stream stopped before delivering a terminal event.
///
/// A closed set on purpose. Four warnings that all read "LLM stream failed" were reported
/// against one model with no way to tell a lost connection from an elapsed deadline, so
/// the category is now part of the event. Every variant is decided here from structure
/// only: the messages carried by [`LlmResponseChunk::StreamError`] and
/// [`LlmResponseChunk::DecodeError`] are upstream text and are never retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamTermination {
    /// The upstream delivered an error event inside an otherwise open stream.
    UpstreamStreamError,
    /// A frame arrived that this side could not decode.
    DecodeError,
    /// The connection failed or was lost mid-stream.
    Transport,
    /// A deadline elapsed while the stream was open.
    Timeout,
    /// The upstream answered with a non-success status mid-stream.
    UpstreamHttp,
    /// The upstream response could not be decoded as a whole.
    InvalidResponse,
    /// A failure this module does not classify. Kept so that adding a variant to
    /// [`LlmClientError`] cannot silently land in a category that claims something else.
    Other,
    /// The stream was dropped by its consumer before it ended.
    ConsumerDropped,
}

impl StreamTermination {
    /// Stable value used when logging.
    const fn as_str(self) -> &'static str {
        match self {
            Self::UpstreamStreamError => "upstream_stream_error",
            Self::DecodeError => "decode_error",
            Self::Transport => "transport",
            Self::Timeout => "timeout",
            Self::UpstreamHttp => "upstream_http",
            Self::InvalidResponse => "invalid_response",
            Self::Other => "other",
            Self::ConsumerDropped => "consumer_dropped",
        }
    }
}

/// One item of a routed stream, as this module receives it.
type StreamItem = Result<LlmResponseStreamEvent, LlmClientError>;

/// Classifies one stream item; `None` when the item is not a failure.
///
/// An item carrying both an upstream error and a decode error is reported as the upstream
/// error, because that is the cause and the decode failure is what it produced.
fn item_termination(item: &StreamItem) -> Option<StreamTermination> {
    match item {
        Err(error) => Some(termination_of(error)),
        Ok(event) => {
            if event
                .normalized()
                .iter()
                .any(|chunk| matches!(chunk, LlmResponseChunk::StreamError { .. }))
            {
                Some(StreamTermination::UpstreamStreamError)
            } else if event
                .normalized()
                .iter()
                .any(|chunk| matches!(chunk, LlmResponseChunk::DecodeError { .. }))
            {
                Some(StreamTermination::DecodeError)
            } else {
                None
            }
        }
    }
}

/// Classifies a transport-level stream failure without retaining its message.
fn termination_of(error: &LlmClientError) -> StreamTermination {
    match error {
        LlmClientError::Transport { .. } => StreamTermination::Transport,
        // A per-attempt expiry and a whole-request expiry are both deadlines that elapsed.
        // Which one it was matters to routing, not to the stream this event reports on.
        LlmClientError::Timeout { .. } | LlmClientError::AttemptTimeout { .. } => {
            StreamTermination::Timeout
        }
        LlmClientError::UpstreamHttp { .. } => StreamTermination::UpstreamHttp,
        LlmClientError::InvalidResponse { .. } | LlmClientError::ResponseTranslation(_) => {
            StreamTermination::InvalidResponse
        }
        _ => StreamTermination::Other,
    }
}

/// The HTTP status that ended the stream, or `0` when the failure carried none.
///
/// A status is a number, not provider text, so it is safe to report beside the category;
/// the body that came with it is not, and is never read here.
fn item_termination_status(item: &StreamItem) -> u16 {
    match item {
        Err(LlmClientError::UpstreamHttp { status, .. }) => status.as_u16(),
        _ => 0,
    }
}

/// Reports a stream that was dropped before it ended.
///
/// A buffered request abandoned by its client is observable through the request log's
/// drop guard. A streamed one is not: the response has already answered 200 and this
/// wrapper is simply dropped, so without this the only trace of an abandoned stream is
/// that nothing was written, which reads as success.
///
/// Disarmed as soon as the stream reaches a terminal event or reports its own failure,
/// because dropping a stream that already ended is normal: a Responses client may stop
/// reading immediately after the terminal event.
///
/// "Dropped" is one signal, not two. A client that disconnects and a caller that cancels
/// look identical from inside the stream, so this reports the condition it can prove
/// rather than guessing which party caused it.
struct StreamDropReporter {
    model: String,
    correlation_id: String,
    armed: bool,
}

impl StreamDropReporter {
    fn new(model: &str, correlation_id: &str) -> Self {
        Self {
            model: model.to_string(),
            correlation_id: correlation_id.to_string(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StreamDropReporter {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        tracing::event!(
            target: "switchyard_server::request",
            tracing::Level::WARN,
            selected_model = %self.model,
            failure_stage = "stream",
            termination_reason = StreamTermination::ConsumerDropped.as_str(),
            upstream_status = 0_u16,
            correlation_id = %self.correlation_id,
            "LLM stream ended before its terminal event"
        );
    }
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
    attribution: Option<&economics::metrics::UsageAttribution>,
    calendar: &economics::CalendarWindow,
) {
    let total_latency = started.elapsed();
    record_usage(usage, model);
    record_latency(model, total_latency);
    // The economics series carry the same usage, attributed to the provider, route and
    // cost class the deployment declares. They are published alongside the native
    // counters rather than instead of them, so nothing that already reads the native
    // surface changes.
    if let Some(attribution) = attribution {
        economics::metrics::record_usage(
            attribution,
            model,
            usage,
            economics::metrics::UsageOutcome::Served,
        );
        economics::metrics::record_request(
            attribution,
            model,
            economics::metrics::UsageOutcome::Served,
        );
        // The provider's own figure for what it charged, when the provider reports
        // one. It is published as its own metric family, so a dashboard can show it
        // beside a Switchyard-derived estimate without ever confusing the two.
        if let Some(cost) = usage.cost {
            economics::metrics::record_provider_cost(attribution, model, cost);
            // The same cost is accumulated into the two windows, so a "spend today"
            // or "spend in the last hour" figure is a total of what happened rather
            // than a difference between two counter samples. The currency travels
            // with the amount; a charge in CNY is never added to one in USD.
            let currency = usage.cost_currency.as_deref().unwrap_or("USD");
            economics::metrics::record_provider_cost_in_rolling(attribution, model, cost, currency);
            economics::metrics::record_provider_cost_in_calendar(
                calendar,
                attribution,
                model,
                cost,
                currency,
            );
        }
        // The same usage is counted into the Bangkok calendar-day window, with the
        // same components and the same accounting rules, so the day view and the
        // rolling counters cannot disagree about what was consumed.
        economics::metrics::record_usage_in_calendar(
            calendar,
            attribution,
            model,
            usage,
            economics::metrics::UsageOutcome::Served,
        );
        economics::metrics::record_request_in_calendar(
            calendar,
            attribution,
            model,
            economics::metrics::UsageOutcome::Served,
        );
        // And into the rolling windows, which are a third question again: the last
        // hour and the last day, ending now, rather than since the process started or
        // since Bangkok midnight.
        economics::metrics::record_usage_in_rolling(
            attribution,
            model,
            usage,
            economics::metrics::UsageOutcome::Served,
        );
        economics::metrics::record_request_in_rolling(
            attribution,
            model,
            economics::metrics::UsageOutcome::Served,
        );
    }
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
    use serde_json::Value;
    use switchyard_protocol::{
        LlmClientError, LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, Metadata,
        Response,
    };

    use super::*;

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
            upstream_headers: http::HeaderMap::new(),
        };
        let stats = StatsAccumulator::default();
        let observed = observe(
            response,
            stats.clone(),
            ObservationContext {
                model: "model/worker".to_string(),
                started: Instant::now(),
                cache_eligible: 0.0,
                routing_log: Some((log.clone(), context)),
                composition: None,
                attribution: None,
                calendar: Arc::new(economics::CalendarWindow::new()),
                correlation_id: None,
            },
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

    /// A Responses stream that commits on its terminal event must carry the same composition
    /// facts as every other append path.
    ///
    /// This is the shape that regressed: the early-commit branch wrote a record without
    /// composition, so a streamed request disagreed with the counters while an aggregate
    /// request was correct. The test supplies real composition metadata, so it fails if any
    /// append path drops it.
    #[tokio::test]
    async fn an_early_committed_stream_records_composition() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = SharedRoutingLog::new(dir.path().join("routing.jsonl")).expect("routing log");
        let context = RoutingLogContext::from_metadata(&Metadata {
            session_id: Some("composition-session".to_string()),
            ..Metadata::default()
        });
        let composition = Arc::new(switchyard_runner::composition::CompositionMetadata {
            parent_route: "tier2".to_string(),
            child_route: "smartlocal".to_string(),
            final_target: "vendor/vl".to_string(),
            children: vec![(
                "smartfree".to_string(),
                switchyard_runner::composition::ChildOutcome::Skipped {
                    capability: "vision",
                },
            )],
            crossed_child: false,
            cross_child_reason: None,
        });
        let usage = Usage {
            input_tokens: Some(7),
            output_tokens: Some(2),
            ..Usage::default()
        };
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::Usage(usage),
            LlmResponseChunk::MessageStop { reason: None },
        ]))]);
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        let stats = StatsAccumulator::default();
        let observed = observe(
            response,
            stats.clone(),
            ObservationContext {
                model: "model/worker".to_string(),
                started: Instant::now(),
                cache_eligible: 0.0,
                routing_log: Some((log.clone(), context)),
                composition: Some(composition),
                attribution: None,
                calendar: Arc::new(economics::CalendarWindow::new()),
                correlation_id: None,
            },
        );
        let LlmResponse::Stream(mut observed) = observed.llm_response else {
            panic!("expected stream");
        };
        assert!(observed.next().await.is_some());
        drop(observed);

        // The record the operator reads must name the child and the target that served.
        let written = std::fs::read_to_string(dir.path().join("routing.jsonl")).expect("read log");
        let record: Value = serde_json::from_str(written.lines().next().expect("one record"))
            .expect("record is JSON");
        assert_eq!(record["child_route"], "smartlocal", "{record}");
        assert_eq!(record["final_target"], "vendor/vl", "{record}");
    }

    // ---- stream termination observability ---------------------------------
    //
    // A streamed request answers HTTP 200 before any of this runs, so a failure here
    // cannot change a status code and leaves no other trace. These tests fix what the
    // one event it can emit is allowed to say: the category that ended the stream, and
    // never the provider text that accompanied it.

    /// Collects this module's events while it is the active subscriber.
    #[derive(Clone, Default)]
    struct CapturedEvents(Arc<parking_lot::Mutex<Vec<(tracing::Level, String)>>>);

    impl tracing_subscriber::Layer<tracing_subscriber::Registry> for CapturedEvents {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, tracing_subscriber::Registry>,
        ) {
            if event.metadata().target() != "switchyard_server::request" {
                return;
            }
            let mut message = String::new();
            event.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    message.push_str(&format!("{}={value:?} ", field.name()));
                },
            );
            self.0.lock().push((*event.metadata().level(), message));
        }
    }

    /// Observes a source stream, so a test drives the wrapper the server actually serves.
    fn observed(
        source: impl futures_util::Stream<Item = Result<LlmResponseStreamEvent, LlmClientError>>
        + Send
        + 'static,
    ) -> LlmResponseStream {
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        let observed = observe(
            response,
            StatsAccumulator::default(),
            ObservationContext {
                model: "model/terminating".to_string(),
                started: Instant::now(),
                cache_eligible: 0.0,
                routing_log: None,
                composition: None,
                attribution: None,
                calendar: Arc::new(economics::CalendarWindow::new()),
                correlation_id: Some("corr-test".to_string()),
            },
        );
        let LlmResponse::Stream(stream) = observed.llm_response else {
            panic!("expected a stream");
        };
        stream
    }

    /// Drains `take` items and then drops the stream, or drains it to the end when
    /// `take` is `None`. Returns the warnings emitted, as `field=value` text.
    async fn warnings_of(mut stream: LlmResponseStream, take: Option<usize>) -> Vec<String> {
        use tracing_subscriber::layer::SubscriberExt;

        let captured = CapturedEvents::default();
        let subscriber = tracing_subscriber::registry().with(captured.clone());
        let dispatch = tracing::Dispatch::new(subscriber);
        let guard = tracing::dispatcher::set_default(&dispatch);
        match take {
            None => while stream.next().await.is_some() {},
            Some(count) => {
                for _ in 0..count {
                    let _ = stream.next().await;
                }
                drop(stream);
            }
        }
        drop(guard);
        captured
            .0
            .lock()
            .iter()
            .filter(|(level, _)| *level == tracing::Level::WARN)
            .map(|(_, message)| message.clone())
            .collect()
    }

    fn stream_error_source(
        message: &str,
    ) -> impl futures_util::Stream<Item = Result<LlmResponseStreamEvent, LlmClientError>> + Send + 'static
    {
        stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::StreamError {
                message: message.to_string(),
            },
        ]))])
    }

    #[tokio::test]
    async fn an_upstream_stream_error_names_the_upstream_as_the_cause() {
        let warnings = warnings_of(observed(stream_error_source("upstream gave up")), None).await;

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains(r#"termination_reason="upstream_stream_error""#),
            "{}",
            warnings[0]
        );
    }

    /// An undecodable frame is our reader's problem, not the provider's, and must not be
    /// reported as an upstream failure.
    #[tokio::test]
    async fn a_decode_error_is_distinguished_from_an_upstream_failure() {
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::DecodeError {
                message: "unexpected frame".to_string(),
            },
        ]))]);

        let warnings = warnings_of(observed(source), None).await;

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains(r#"termination_reason="decode_error""#),
            "{}",
            warnings[0]
        );
        assert!(
            !warnings[0].contains("upstream_stream_error"),
            "{}",
            warnings[0]
        );
    }

    /// A lost connection and an elapsed deadline are the two causes that were
    /// indistinguishable in the field, so each must reach the log by its own name.
    #[tokio::test]
    async fn a_transport_failure_and_a_deadline_are_reported_separately() {
        let transport = stream::iter([Err(LlmClientError::Transport {
            source: Box::new(std::io::Error::other("connection reset")),
        })]);
        let timeout = stream::iter([Err(LlmClientError::Timeout {
            source: Box::new(std::io::Error::other("deadline")),
        })]);

        let transport = warnings_of(observed(transport), None).await;
        let timeout = warnings_of(observed(timeout), None).await;

        assert_eq!(transport.len(), 1, "{transport:?}");
        assert!(
            transport[0].contains(r#"termination_reason="transport""#),
            "{}",
            transport[0]
        );
        assert_eq!(timeout.len(), 1, "{timeout:?}");
        assert!(
            timeout[0].contains(r#"termination_reason="timeout""#),
            "{}",
            timeout[0]
        );
    }

    /// The status that ended the stream is a number and is safe to report; the body that
    /// came with it is not, and must not reach the log.
    #[tokio::test]
    async fn an_upstream_http_failure_reports_its_status_but_not_its_body() {
        const LEAKED: &str = "provider-body-canary";
        let source = stream::iter([Err(LlmClientError::UpstreamHttp {
            status: http::StatusCode::BAD_GATEWAY,
            body: LEAKED.to_string(),
        })]);

        let warnings = warnings_of(observed(source), None).await;

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains(r#"termination_reason="upstream_http""#),
            "{}",
            warnings[0]
        );
        assert!(
            warnings[0].contains("upstream_status=502"),
            "{}",
            warnings[0]
        );
        assert!(!warnings[0].contains(LEAKED), "{}", warnings[0]);
    }

    /// The provider's own error text is the thing this event must not carry.
    #[tokio::test]
    async fn the_failure_message_never_reaches_the_log() {
        const LEAKED: &str = "sk-or-v1-canary-in-provider-prose";

        let warnings = warnings_of(observed(stream_error_source(LEAKED)), None).await;

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(!warnings[0].contains(LEAKED), "{}", warnings[0]);
    }

    /// A stream dropped before it ends leaves no other trace, so the drop itself has to
    /// be the signal.
    #[tokio::test]
    async fn a_stream_dropped_before_its_terminal_event_reports_the_consumer() {
        let source = stream::iter([
            Ok(LlmResponseStreamEvent::new(vec![
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "partial".to_string(),
                },
            ])),
            Ok(LlmResponseStreamEvent::new(vec![
                LlmResponseChunk::MessageStop { reason: None },
            ])),
        ]);

        let warnings = warnings_of(observed(source), Some(1)).await;

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains(r#"termination_reason="consumer_dropped""#),
            "{}",
            warnings[0]
        );
    }

    /// Control: dropping a stream that already ended is normal. A Responses client may
    /// stop reading the moment it has the terminal event, and that is not a failure.
    #[tokio::test]
    async fn a_stream_dropped_after_its_terminal_event_reports_nothing() {
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::Usage(Usage {
                input_tokens: Some(1),
                output_tokens: Some(1),
                ..Usage::default()
            }),
            LlmResponseChunk::MessageStop { reason: None },
        ]))]);

        let warnings = warnings_of(observed(source), Some(1)).await;

        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// Control: a stream that ends on its own reports nothing, so the event above cannot
    /// fire on every stream and become noise.
    #[tokio::test]
    async fn a_stream_that_ends_on_its_own_reports_nothing() {
        let source = stream::iter([
            Ok(LlmResponseStreamEvent::new(vec![
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "complete".to_string(),
                },
            ])),
            Ok(LlmResponseStreamEvent::new(vec![
                LlmResponseChunk::MessageStop { reason: None },
            ])),
        ]);

        let warnings = warnings_of(observed(source), None).await;

        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// Control: a stream that ends without ever sending a terminal event, and without
    /// failing, is still a stream that finished. This module already handles that shape by
    /// committing what it saw, so its drop must stay silent; only a stream cut short by
    /// its consumer is an abandoned one.
    #[tokio::test]
    async fn a_stream_that_ends_without_a_terminal_event_reports_nothing() {
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "no stop event".to_string(),
            },
        ]))]);

        let warnings = warnings_of(observed(source), None).await;

        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A response the consumer drops without ever polling it was still abandoned.
    ///
    /// `async_stream::stream!` does not run its body until the first poll, so a guard
    /// constructed inside the body does not exist yet for an unpolled stream. That is a
    /// real shape: a caller that discards a response after inspecting headers, or gives
    /// up before reading. The routed-call metric already counts this case as a failure
    /// (an unset outcome), so a log that stays silent here would disagree with the
    /// counter it is supposed to explain.
    #[tokio::test]
    async fn a_stream_never_polled_reports_the_consumer() {
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::MessageStop { reason: None },
        ]))]);

        let warnings = warnings_of(observed(source), Some(0)).await;

        assert_eq!(
            warnings.len(),
            1,
            "an unpolled response is abandoned, not completed: {warnings:?}"
        );
        assert!(
            warnings[0].contains("consumer_dropped"),
            "the cause must name the consumer: {warnings:?}"
        );
    }

    // ---- the real Responses path -------------------------------------------------
    //
    // The tests above drive hand-built chunks, which proves the wrapper's own logic and
    // nothing about the decode in front of it. Production serves
    // `wire_format=openai_responses` through `decode_stream`, so these fixtures feed raw
    // SSE bytes through that decoder and observe the stream the server actually wraps.
    // A hand-built `MessageStop` cannot show whether `response.completed` decodes to one.

    /// One synthetic Responses stream: reasoning item, answer, then `response.completed`.
    ///
    /// Sanitised: fixed ids, no provider or prompt text.
    fn responses_sse_reasoning_plus_answer() -> String {
        [
            r#"event: response.created"#,
            r#"data: {"type":"response.created","response":{"id":"resp_fixture","model":"fixture-model","status":"in_progress","output":[]}}"#,
            "",
            r#"event: response.output_item.added"#,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_fixture","type":"reasoning","summary":[]}}"#,
            "",
            r#"event: response.reasoning_summary_text.delta"#,
            r#"data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_fixture","output_index":0,"delta":"thinking"}"#,
            "",
            r#"event: response.output_item.done"#,
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_fixture","type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}}"#,
            "",
            r#"event: response.output_text.delta"#,
            r#"data: {"type":"response.output_text.delta","item_id":"msg_fixture","output_index":1,"delta":"answer"}"#,
            "",
            r#"event: response.output_item.done"#,
            r#"data: {"type":"response.output_item.done","output_index":1,"item":{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}}"#,
            "",
            r#"event: response.completed"#,
            r#"data: {"type":"response.completed","response":{"id":"resp_fixture","model":"fixture-model","status":"completed","output":[{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#,
            "",
            r#"data: [DONE]"#,
            "",
        ]
        .join("\n")
    }

    /// The stream the server really serves for a Responses request.
    ///
    /// `decode_stream` is the same entry point the HTTP path uses, so the terminal
    /// mapping under test is the deployed one rather than a re-statement of it.
    fn responses_observed(body: String) -> LlmResponseStream {
        let bytes: Result<Vec<u8>, LlmClientError> = Ok(body.into_bytes());
        let decoded = switchyard_translation::decode_stream(
            stream::iter([bytes]),
            switchyard_translation::WireFormat::OpenAiResponses,
        )
        .expect("the fixture stream must decode");
        observed(decoded)
    }

    /// A completed reasoning-plus-answer Responses stream must not report abandonment.
    ///
    /// This is the production shape: `wire_format=openai_responses` on the Luna client,
    /// including the reasoning item the model actually emits. It asserts both halves that
    /// matter: the decoded stream still delivers the answer and a neutral terminal, and a
    /// consumer that runs it to the end is not reported as having abandoned it.
    #[tokio::test]
    async fn a_completed_responses_stream_delivers_its_terminal_and_never_reports_abandonment() {
        // What the real decoder delivers.
        let mut decoded = responses_observed(responses_sse_reasoning_plus_answer());
        let mut text = String::new();
        let mut saw_terminal = false;
        while let Some(item) = decoded.next().await {
            let event = item.expect("the fixture stream must not fail");
            for chunk in event.normalized() {
                match chunk {
                    LlmResponseChunk::MessageStop { .. } => saw_terminal = true,
                    LlmResponseChunk::TextDelta { text: delta, .. } => text.push_str(delta),
                    _ => {}
                }
            }
        }
        assert!(
            text.contains("answer"),
            "the answer must reach the consumer, got {text:?}"
        );
        assert!(
            saw_terminal,
            "response.completed must decode to a neutral terminal, or the guard can never disarm"
        );

        // And what the wrapper logs for that same stream.
        let warnings = warnings_of(
            responses_observed(responses_sse_reasoning_plus_answer()),
            None,
        )
        .await;
        assert!(
            warnings.is_empty(),
            "a stream that delivered response.completed must not report abandonment: {warnings:?}"
        );
    }

    /// The exact client behaviour under investigation: stop reading immediately after
    /// the terminal event and drop the stream.
    ///
    /// The guard must already be disarmed before that event is yielded. If it is not,
    /// every ordinary Responses call would carry a `consumer_dropped` warning, which is
    /// what the live Luna lane appeared to show.
    #[tokio::test]
    async fn stopping_immediately_after_the_terminal_event_reports_no_abandonment() {
        // Count how many items the decoded stream produces up to and including the one
        // that carries the terminal event.
        let mut probe = responses_observed(responses_sse_reasoning_plus_answer());
        let mut up_to_terminal = 0usize;
        let mut found_terminal = false;
        while let Some(item) = probe.next().await {
            up_to_terminal += 1;
            if let Ok(event) = item
                && event
                    .normalized()
                    .iter()
                    .any(|chunk| matches!(chunk, LlmResponseChunk::MessageStop { .. }))
            {
                found_terminal = true;
                break;
            }
        }
        drop(probe);
        assert!(
            found_terminal,
            "the fixture must deliver a neutral terminal, or this test proves nothing"
        );

        let warnings = warnings_of(
            responses_observed(responses_sse_reasoning_plus_answer()),
            Some(up_to_terminal),
        )
        .await;

        assert!(
            warnings.is_empty(),
            "stopping at the terminal event is normal completion, not abandonment: {warnings:?}"
        );
    }

    /// Control: stopping before the terminal event is a genuine abandonment and must
    /// still be reported exactly once. Without this the test above could pass because
    /// the warning was disabled altogether.
    #[tokio::test]
    async fn stopping_before_the_terminal_event_reports_the_consumer_once() {
        let warnings = warnings_of(
            responses_observed(responses_sse_reasoning_plus_answer()),
            Some(2),
        )
        .await;

        assert_eq!(
            warnings.len(),
            1,
            "a stream cut short before its terminal event is one abandonment: {warnings:?}"
        );
        assert!(
            warnings[0].contains(r#"termination_reason="consumer_dropped""#),
            "{}",
            warnings[0]
        );
    }

    /// The stream-terminal warning must name the request it belongs to.
    ///
    /// Without this the only way to tie a drop to a request is approximate timestamp,
    /// which is why 18 live drops could not be classified.
    #[tokio::test]
    async fn an_abandonment_warning_carries_the_request_correlation_id() {
        let warnings = warnings_of(
            responses_observed(responses_sse_reasoning_plus_answer()),
            Some(2),
        )
        .await;

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("correlation_id=corr-test")
                || warnings[0].contains("correlation_id=\"corr-test\""),
            "the warning must be attributable to a request: {}",
            warnings[0]
        );
    }

    /// The fallback id must be bounded, identifier-shaped and unique per call.
    ///
    /// It is written into a log line, so it must not be able to carry request content,
    /// and two requests must never share one.
    #[test]
    fn synthetic_request_ids_are_bounded_unique_and_opaque() {
        let first = crate::next_synthetic_request_id();
        let second = crate::next_synthetic_request_id();

        assert_ne!(first, second, "two requests must not share an id");
        assert!(
            switchyard_protocol::correlation::validated_request_id(&first).is_some(),
            "the id must be identifier-shaped: {first}"
        );
        assert!(
            first.len() <= switchyard_protocol::correlation::MAX_REQUEST_ID_LEN,
            "the id must stay within the provider identifier bound"
        );
    }
}
