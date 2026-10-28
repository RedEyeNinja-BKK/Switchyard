// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response usage and full-turn latency metrics for routed requests.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use opentelemetry::{KeyValue, global};
use switchyard_protocol::{LlmResponse, LlmResponseChunk, Response, Usage};

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
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                let mut terminal_seen = false;
                let mut recorded = false;
                while let Some(item) = stream.next().await {
                    let failed = match &item {
                        Err(_) => true,
                        Ok(event) => event.normalized().iter().any(|chunk| {
                            matches!(
                                chunk,
                                LlmResponseChunk::StreamError { .. }
                                    | LlmResponseChunk::DecodeError { .. }
                            )
                        }),
                    };
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
    use switchyard_protocol::{LlmResponseChunk, LlmResponseStreamEvent, Metadata, Response};

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
}
