// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide Prometheus export for Switchyard's OpenTelemetry metrics.

use std::sync::OnceLock;

use opentelemetry::{KeyValue, global};
use opentelemetry_sdk::metrics::{Aggregation, Instrument, SdkMeterProvider, Stream};
use prometheus::{Encoder, Registry, TextEncoder};
use switchyard_llm_client::metrics::{http_outcome_label, http_status_code_label};

pub(crate) const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Outcome label for a request the client abandoned. Kept separate from the
/// status-derived outcomes so cancelled work is never counted as a served response.
const CLIENT_DISCONNECTED_OUTCOME: &str = "client_disconnected";

/// Bucket boundaries for `switchyard.routing_overhead_ms`.
/// Need a broad range because some algos call an LLM (classifier), and some
/// do very little (passthrough).
const ROUTING_OVERHEAD_BUCKETS_MS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
];

/// Bucket boundaries for model-call and end-to-end LLM latency histograms.
/// Retains the SDK defaults through 10 seconds and extends them for long generations.
const LLM_LATENCY_BUCKETS_MS: &[f64] = &[
    0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0,
    10_000.0, 15_000.0, 30_000.0, 60_000.0, 120_000.0, 300_000.0,
];

/// Bucket boundaries for capability-executor latency (ms).
/// Embeddings and rerank run on CPU-only ComfyNinja executors, so the useful
/// range is wide: measured rerank 8 docs = 5.0 s, 16 = 8.9 s, 24 = 13.3 s,
/// 32 = 17.2 s, against a 60 s client timeout (widened from 30 s on 2026-09-12,
/// operator GO). The sub-second buckets exist so embedding calls, which are
/// much faster, are not all collapsed into the first bucket.
const CAPABILITY_DURATION_BUCKETS_MS: &[f64] = &[
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10_000.0, 20_000.0, 30_000.0,
    60_000.0, 120_000.0,
];

/// Bucket boundaries for capability batch size, in items per request.
/// Embedding `max_batch` = 64 and rerank `max_candidates` = 32, so the bounds
/// are the configured admission ceilings rather than arbitrary values.
const CAPABILITY_ITEMS_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0];

/// Bucket boundaries for the longest single document in a rerank batch (chars).
/// `max_doc_chars` = 4096 is an admission boundary, and the reason it is worth
/// instrumenting directly: one over-length document fails the ENTIRE batch with
/// an instant 400, so the distribution shows how close live calls run to it.
const CAPABILITY_DOC_CHARS_BUCKETS: &[f64] =
    &[256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0, 8192.0];

/// Default outcome for a capability call that returned without labelling itself.
/// Deliberately fail-loud: if a future edit adds a return path and forgets to set
/// an outcome, it shows up as `incomplete` rather than being silently attributed
/// to a neighbouring outcome.
pub(crate) const CAPABILITY_INCOMPLETE: &str = "incomplete";

/// Label used when the requested model id does not resolve to a capability route.
/// A FIXED sentinel, never the caller-supplied string: `model` is request-controlled
/// and using it as a label value would be an unbounded-cardinality hazard.
const CAPABILITY_UNKNOWN_ROUTE: &str = "<unknown>";

struct Metrics {
    registry: Registry,
    provider: SdkMeterProvider,
}

static METRICS: OnceLock<Result<Metrics, String>> = OnceLock::new();

/// Returns the registry shared by every server router in this process.
pub(crate) fn registry() -> Result<Registry, String> {
    match METRICS.get_or_init(initialize) {
        Ok(metrics) => Ok(metrics.registry.clone()),
        Err(error) => Err(error.clone()),
    }
}

fn initialize() -> Result<Metrics, String> {
    let registry = Registry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .map_err(|error| format!("failed to initialize Prometheus metrics: {error}"))?;
    let mut builder = SdkMeterProvider::builder()
        .with_reader(exporter)
        .with_view(routing_overhead_buckets)
        .with_view(llm_latency_buckets)
        .with_view(capability_duration_buckets)
        .with_view(capability_items_buckets)
        .with_view(capability_doc_chars_buckets)
        .with_resource(crate::observability::resource());
    if crate::observability::otlp_enabled("METRICS") {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .build()
            .map_err(|error| format!("failed to initialize OTLP metric exporter: {error}"))?;
        builder = builder.with_periodic_exporter(exporter);
    }
    let provider = builder.build();
    global::set_meter_provider(provider.clone());
    switchyard_llm_client::initialize_metrics();
    global::meter("switchyard")
        .u64_gauge("switchyard.build_info")
        .build()
        .record(1, &[KeyValue::new("version", env!("CARGO_PKG_VERSION"))]);
    seed_outcome_metrics();
    Ok(Metrics { registry, provider })
}

pub(crate) fn flush() {
    if let Some(Ok(metrics)) = METRICS.get()
        && let Err(error) = metrics.provider.force_flush()
    {
        tracing::warn!(error = %error, "failed to flush OpenTelemetry metrics");
    }
}

fn routing_overhead_buckets(instrument: &Instrument) -> Option<Stream> {
    if instrument.name() != "switchyard.routing_overhead_ms" {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: ROUTING_OVERHEAD_BUCKETS_MS.to_vec(),
            // Cumulative min/max cover the whole process, so they aren't useful.
            record_min_max: false,
        })
        .build()
        .ok()
}

fn llm_latency_buckets(instrument: &Instrument) -> Option<Stream> {
    if !matches!(
        instrument.name(),
        "switchyard.model_call_latency_ms" | "switchyard.total_latency_ms"
    ) {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: LLM_LATENCY_BUCKETS_MS.to_vec(),
            record_min_max: true,
        })
        .build()
        .ok()
}

fn capability_duration_buckets(instrument: &Instrument) -> Option<Stream> {
    if instrument.name() != "switchyard.capability_duration_ms" {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: CAPABILITY_DURATION_BUCKETS_MS.to_vec(),
            record_min_max: true,
        })
        .build()
        .ok()
}

fn capability_items_buckets(instrument: &Instrument) -> Option<Stream> {
    if instrument.name() != "switchyard.capability_items" {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: CAPABILITY_ITEMS_BUCKETS.to_vec(),
            // Lifetime min/max batch size is not a useful reading.
            record_min_max: false,
        })
        .build()
        .ok()
}

fn capability_doc_chars_buckets(instrument: &Instrument) -> Option<Stream> {
    if instrument.name() != "switchyard.capability_max_doc_chars" {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: CAPABILITY_DOC_CHARS_BUCKETS.to_vec(),
            record_min_max: false,
        })
        .build()
        .ok()
}

/// Make the metrics exist before they get a hit. Nicer for dashboards but not really necessary.
/// The HTTP status codes we seed are somewhat arbitrary.
fn seed_outcome_metrics() {
    let meter = global::meter("switchyard");
    let upstream_attempts = meter.u64_counter("switchyard.upstream_attempts").build();
    for status in [Some(200), Some(404), Some(429), Some(500), Some(504), None] {
        upstream_attempts.add(
            0,
            &[
                KeyValue::new("outcome", http_outcome_label(status)),
                KeyValue::new("code", http_status_code_label(status)),
            ],
        );
    }

    let client_responses = meter.u64_counter("switchyard.client_responses").build();
    for outcome in [
        "ok",
        "retryable_error",
        "other_error",
        CLIENT_DISCONNECTED_OUTCOME,
    ] {
        client_responses.add(0, &[KeyValue::new("outcome", outcome)]);
    }
    meter
        .u64_counter("switchyard.router_retry_recovered")
        .build()
        .add(0, &[]);

    // Seed `switchyard.escalations` so the family EXISTS before the first escalation.
    // OpenTelemetry materialises a labelled counter only once it is recorded, so
    // without this the family is absent from /metrics on any quiet day and dashboard
    // queries against it fail closed as a missing metric.
    //
    // Recorded with an EMPTY label set on purpose: source/destination/reason are
    // configuration-derived, and seeding synthetic values would invent escalation
    // rows that never happened. The unlabelled sample carries only the zero and is
    // ignored by `sum by (source, ...)` queries, so a real escalation is still the
    // first labelled series to appear.
    meter.u64_counter("switchyard.escalations").build().add(0, &[]);
}

/// Records one escalation: a request re-routed from `source` to `destination`
/// under `reason` (e.g. size-based escalation from a bounded tier to its smart parent).
pub(crate) fn record_escalation(source: &str, destination: &str, reason: &str) {
    global::meter("switchyard")
        .u64_counter("switchyard.escalations")
        .build()
        .add(
            1,
            &[
                KeyValue::new("source", source.to_string()),
                KeyValue::new("destination", destination.to_string()),
                KeyValue::new("reason", reason.to_string()),
            ],
        );
}

pub(crate) fn record_client_disconnect() {
    global::meter("switchyard")
        .u64_counter("switchyard.client_responses")
        .build()
        .add(1, &[KeyValue::new("outcome", CLIENT_DISCONNECTED_OUTCOME)]);
}

/// Records the final status returned by an LLM-serving route.
pub(crate) fn record_client_response(status: u16) {
    global::meter("switchyard")
        .u64_counter("switchyard.client_responses")
        .build()
        .add(
            1,
            &[KeyValue::new("outcome", http_outcome_label(Some(status)))],
        );
}

/// Records one embedding/rerank capability request and its admission outcome.
///
/// A guard rather than explicit bookkeeping on every return path. Both handlers
/// have a dozen early `return error_response(..)` statements for admission
/// failures, and those refusals are precisely the signal worth counting: an
/// instrument placed at the executor call would never see them. `Drop` guarantees
/// exactly one request record and one in-flight decrement per call, and the
/// outcome defaults to `incomplete`, so a return path that forgets to label
/// itself is visible rather than silently attributed elsewhere.
pub(crate) struct CapabilityCall {
    capability: &'static str,
    route: String,
    outcome: &'static str,
}

impl CapabilityCall {
    pub(crate) fn start(capability: &'static str, route: &str) -> Self {
        record_capability_in_flight(capability, 1);
        Self {
            capability,
            route: route.to_string(),
            outcome: CAPABILITY_INCOMPLETE,
        }
    }

    /// Sets the outcome recorded when this call is dropped.
    pub(crate) fn outcome(&mut self, outcome: &'static str) {
        self.outcome = outcome;
    }
}

impl Drop for CapabilityCall {
    fn drop(&mut self) {
        record_capability_in_flight(self.capability, -1);
        record_capability_request(self.capability, &self.route, self.outcome);
    }
}

/// Records a refusal that happened BEFORE a capability route was resolved, so no
/// route id exists to attribute it to. Always attributes to a fixed sentinel.
pub(crate) fn record_capability_unresolved(capability: &'static str, outcome: &'static str) {
    record_capability_request(capability, CAPABILITY_UNKNOWN_ROUTE, outcome);
}

fn record_capability_request(capability: &str, route: &str, outcome: &str) {
    global::meter("switchyard")
        .u64_counter("switchyard.capability_requests")
        .build()
        .add(
            1,
            &[
                KeyValue::new("capability", capability.to_string()),
                KeyValue::new("route", route.to_string()),
                KeyValue::new("outcome", outcome.to_string()),
            ],
        );
}

/// Records the EXECUTOR leg only, excluding admission and validation time, so the
/// histogram answers "how slow is the ComfyNinja executor" rather than "how slow
/// is the endpoint".
pub(crate) fn record_capability_duration(capability: &str, route: &str, milliseconds: f64) {
    global::meter("switchyard")
        .f64_histogram("switchyard.capability_duration_ms")
        .build()
        .record(
            milliseconds,
            &[
                KeyValue::new("capability", capability.to_string()),
                KeyValue::new("route", route.to_string()),
            ],
        );
}

/// Records the number of items in one batch (embedding inputs, rerank candidates).
pub(crate) fn record_capability_items(capability: &str, items: u64) {
    global::meter("switchyard")
        .u64_histogram("switchyard.capability_items")
        .build()
        .record(items, &[KeyValue::new("capability", capability.to_string())]);
}

/// Records the longest single document in a rerank batch, in characters.
pub(crate) fn record_capability_max_doc_chars(capability: &str, chars: u64) {
    global::meter("switchyard")
        .u64_histogram("switchyard.capability_max_doc_chars")
        .build()
        .record(
            chars,
            &[KeyValue::new("capability", capability.to_string())],
        );
}

fn record_capability_in_flight(capability: &str, delta: i64) {
    global::meter("switchyard")
        .i64_up_down_counter("switchyard.capability_in_flight")
        .build()
        .add(delta, &[KeyValue::new("capability", capability.to_string())]);
}

/// Encodes the current cumulative metric values in Prometheus text format.
pub(crate) fn encode(registry: &Registry) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    TextEncoder::new()
        .encode(&registry.gather(), &mut body)
        .map_err(|error| format!("failed to encode Prometheus metrics: {error}"))?;
    Ok(body)
}
