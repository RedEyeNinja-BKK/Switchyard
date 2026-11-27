// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`TranslatingLlmClient`] — the crate's single public entry point: encode a neutral
//! request, call the configured backend over HTTP, decode the neutral response.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use http::StatusCode;
use reqwest::RequestBuilder;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::{Map, Value, json};
use switchyard_protocol::correlation;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmRequest, LlmResponse, LlmResponseChunk, LlmResponseStream,
    LlmResponseStreamEvent, Metadata, ModelId, Request, Response, Role, RoutedLlmClient,
};
use switchyard_translation::{
    ReasoningAdmission, ReasoningDetailsReplay, TargetCapabilities, TranslationError,
    TranslationPolicy, WireFormat, decode_aggregated_response_with_policy, decode_request,
    decode_stream_with_policy, encode_aggregated_response_with_extensions,
    encode_request_with_policy, encode_stream_with_extensions,
    provenance::{ProvenanceRegistry, ReasoningIssuer, ReasoningLineage, opaque_digest},
};
use tokio::time::Instant;
use tracing::Instrument;

use crate::backend::{Backend, openai_url};
use crate::error::{LlmClientError, Result};
use crate::metrics;
use crate::raw::RawResponse;

// Caller headers safe to send when caller auth forwarding is disabled.
const ALLOWED_METADATA_HEADERS: &[&str] = &["x-request-id"];

// Headers tied to the inbound connection, destination, or body. The HTTP client
// must rebuild these for the upstream request, even when forwarding auth.
const NON_FORWARDABLE_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "connection",
    "proxy-connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-type",
    "content-encoding",
    "accept-encoding",
    "expect",
    // The pre-egress admission assertion header is gone with the gate. Nothing is stripped
    // for policy reasons any more; this list is transport hygiene only.
    "x-switchyard-privacy-clearance",
];

const INITIAL_RETRY_DELAY: Duration = Duration::from_millis(250);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(2);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// How one model is served: the `default_backend` used when the request does not
/// pin a wire format, plus any `other_backends` reachable over additional formats.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    model_name: ModelId,
    default_backend: Backend,
    other_backends: Option<Vec<Backend>>,
}

impl ModelConfig {
    /// A model named `model_name` served by `default_backend`, optionally reachable
    /// over additional wire formats via `other_backends`.
    pub fn new(
        model_name: impl Into<ModelId>,
        default_backend: Backend,
        other_backends: Option<Vec<Backend>>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            default_backend,
            other_backends,
        }
    }
}

/// A model-bearing provider operation outside the normal completion endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuxiliaryOperation {
    /// Anthropic Messages input-token counting.
    AnthropicCountTokens,
    /// OpenAI Responses input-token counting.
    ResponsesInputTokens,
    /// OpenAI Responses compaction.
    ResponsesCompact,
}

impl AuxiliaryOperation {
    const fn wire_format(self) -> WireFormat {
        match self {
            Self::AnthropicCountTokens => WireFormat::AnthropicMessages,
            Self::ResponsesInputTokens | Self::ResponsesCompact => WireFormat::OpenAiResponses,
        }
    }

    fn url(self, backend: &Backend) -> String {
        let suffix = match self {
            Self::AnthropicCountTokens => return backend.count_tokens_url(),
            Self::ResponsesInputTokens => "/responses/input_tokens",
            Self::ResponsesCompact => "/responses/compact",
        };
        let base_url = backend.url();
        openai_url(&base_url, suffix).unwrap_or_else(|error| {
            tracing::error!(%error, "Unable to build OpenAI auxiliary endpoint URL");
            base_url
        })
    }
}

/// A client that dispatches neutral-IR requests to per-model HTTP backends.
///
/// Construct it with a list of [`ModelConfig`]s — one per model, each naming a
/// default [`Backend`] and any additional per-format backends. Each call resolves
/// the model and wire format, encodes the request to that backend's wire format,
/// applies auth and forwarded headers, sends the HTTP request with a shared
/// [`reqwest::Client`], and decodes the response back to the neutral IR (buffered
/// or streamed).
pub struct TranslatingLlmClient {
    model_to_config: HashMap<ModelId, ModelConfig>,
    client: reqwest::Client,
    forward_auth_client: reqwest::Client,
    cooldown_epoch: Instant,
    // Keyed by cooldown group when the backend declares one, otherwise by model, so
    // several models behind one provider pool can share a cooldown. Milliseconds
    // since cooldown_epoch; each atomic is independent, so relaxed ordering suffices.
    unavailable_until: HashMap<String, AtomicU64>,
    // The HTTP status that armed each cooldown, so a skipped call can report the
    // real failure instead of a generic unavailability. Zero means "no status".
    cooldown_status: HashMap<String, AtomicU32>,
    // Which identity issued each piece of opaque reasoning state this process saw arrive.
    // Bounded and process-local; see `switchyard_translation::provenance`.
    provenance: Arc<ProvenanceRegistry>,
}

/// The key a model's failure cooldown is recorded under.
///
/// A declared group means "these targets share one upstream pool", so one
/// failure stands for all of them. Without a group the model is its own key,
/// which is the existing per-model behaviour.
fn cooldown_key(backend: &Backend, model: &ModelId) -> String {
    backend
        .cooldown_group()
        .map(str::to_string)
        .unwrap_or_else(|| model.as_str().to_string())
}

impl TranslatingLlmClient {
    /// Builds a client over the given [`ModelConfig`]s, with a fresh shared HTTP
    /// client and the built-in translation codecs.
    pub fn new(model_configs: &[ModelConfig]) -> Result<Self> {
        // Pre-populate so concurrent calls update only atomics and keep map access lock-free.
        let mut unavailable_until = HashMap::new();
        let mut cooldown_status = HashMap::new();
        for config in model_configs {
            for backend in std::iter::once(&config.default_backend)
                .chain(config.other_backends.iter().flatten())
            {
                backend.validate_configured_headers(&config.model_name)?;
            }
            for key in std::iter::once(&config.default_backend)
                .chain(config.other_backends.iter().flatten())
                .map(|backend| cooldown_key(backend, &config.model_name))
            {
                unavailable_until.insert(key.clone(), AtomicU64::new(0));
                cooldown_status.insert(key, AtomicU32::new(0));
            }
        }
        let build_client = |builder: reqwest::ClientBuilder| {
            builder.build().map_err(|error| LlmClientError::Transport {
                source: Box::new(error),
            })
        };
        let client = build_client(reqwest::Client::builder())?;
        // A redirect could move provider-specific headers to another origin.
        // Forwarded credentials are sent only to the configured URL.
        let forward_auth_client =
            build_client(reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()))?;
        let model_to_config = model_configs
            .iter()
            .map(|config| (config.model_name.clone(), config.clone()))
            .collect();

        Ok(Self {
            model_to_config,
            client,
            forward_auth_client,
            cooldown_epoch: Instant::now(),
            unavailable_until,
            cooldown_status,
            provenance: Arc::new(ProvenanceRegistry::default()),
        })
    }

    /// The process-local record of which target issued opaque reasoning state.
    ///
    /// Exposed for observability and tests. A restart begins empty, which degrades every
    /// unproven replay to "withhold" rather than to a wrong target.
    pub fn provenance(&self) -> &Arc<ProvenanceRegistry> {
        &self.provenance
    }

    /// The backend serving `model` over `format` — the default backend when its
    /// format matches, otherwise a matching entry in `other_backends`; `None` when
    /// the model is unknown or has no backend for `format`.
    pub fn backend_for(&self, model: &ModelId, format: WireFormat) -> Option<&Backend> {
        self.model_to_config.get(model).and_then(|config| {
            if config.default_backend.wire_format() == format {
                Some(&config.default_backend)
            } else {
                config
                    .other_backends
                    .as_ref()
                    .and_then(|backends| backends.iter().find(|b| b.wire_format() == format))
            }
        })
    }

    /// Whether `model` has a backend for `operation`.
    pub fn supports_auxiliary(&self, model: &ModelId, operation: AuxiliaryOperation) -> bool {
        self.backend_for(model, operation.wire_format()).is_some()
    }

    /// Calls a model-bearing auxiliary provider operation.
    ///
    /// Returns an error when the model has no compatible backend or the upstream
    /// request fails or returns invalid JSON.
    pub async fn call_auxiliary(
        &self,
        model: &ModelId,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value> {
        let wire_format = operation.wire_format();
        let backend =
            self.backend_for(model, wire_format)
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("model {model} has no backend for {operation:?}"),
                })?;
        let Request {
            mut llm_request,
            metadata,
            ..
        } = request;
        llm_request.model = Some(model.to_string());
        let http_response = self
            .send_encoded(
                backend,
                wire_format,
                llm_request,
                metadata.as_ref(),
                model,
                UpstreamEndpoint::Auxiliary(operation),
                None,
                None,
            )
            .await?;
        let EncodedResponse::Buffered { body, .. } = http_response else {
            return Err(LlmClientError::InvalidRequest {
                message: "auxiliary endpoints do not support streaming".to_string(),
            });
        };
        serde_json::from_slice(&body).map_err(|error| LlmClientError::InvalidResponse {
            source: Box::new(error),
        })
    }

    /// Encode `llm_request` for `wire_format`, POST it to `url` with the request's
    /// forwarded headers plus the backend's static headers and auth, and return the
    /// successful upstream response. A
    /// buffered response is fully collected within the retry boundary; a streamed
    /// response has its first decoded event checked within that boundary. The backend's
    /// deadline, when set, spans every attempt, retry delay, and subsequent stream read. A
    /// non-success status maps to a typed error — a 400 is classified as a context-window
    /// overflow via the backend's provider rules. Shared by
    /// [`call_rewrite_model`](Self::call_rewrite_model) (which POSTs to the
    /// backend's completion URL and decodes a response) and
    /// the model-bearing auxiliary operations, which return raw JSON.
    #[allow(clippy::too_many_arguments)]
    async fn send_encoded(
        &self,
        backend: &Backend,
        wire_format: WireFormat,
        llm_request: LlmRequest,
        metadata: Option<&Metadata>,
        model: &ModelId,
        endpoint: UpstreamEndpoint,
        route_reasoning_policy: Option<switchyard_protocol::ReasoningPolicy>,
        route_reasoning_effort: Option<&str>,
    ) -> Result<EncodedResponse> {
        // A caller's Responses-only feature may have no representation on this target's wire
        // format, and the decode has already discarded it. Refused before the cooldown is even
        // consulted, because the answer is about the request rather than the model's health.
        // The check reads the caller's own preserved body: the normalized IR no longer carries
        // the distinction, so this is the last point at which the loss is visible.
        if !matches!(backend, Backend::OpenAiResponses(_))
            && let Some(feature) = caller_responses_feature(&llm_request)
        {
            return Err(LlmClientError::UnsupportedFeature {
                model: model.clone(),
                feature: feature.message(),
            });
        }
        let cooldown = backend.failure_cooldown();
        let key = cooldown_key(backend, model);
        let unavailable_until =
            if matches!(endpoint, UpstreamEndpoint::Completion) && !cooldown.is_zero() {
                self.unavailable_until.get(&key)
            } else {
                None
            };
        if let Some(until) = unavailable_until
            && until.load(Ordering::Relaxed) > duration_millis(self.cooldown_epoch.elapsed())
        {
            tracing::debug!(model = %model, wire_format = %wire_format, "skipping backend during failure cooldown");
            // A cooldown the target itself armed keeps its established answer:
            // the model is merely resting, and the caller is told to come back.
            // A cooldown inherited from a sibling in the same pool is different --
            // the caller never saw that failure, so the failure is replayed rather
            // than replaced by an unavailability it would classify differently.
            if backend.cooldown_group().is_some() {
                let armed = self
                    .cooldown_status
                    .get(&key)
                    .map(|status| status.load(Ordering::Relaxed))
                    .unwrap_or(0);
                if armed != 0
                    && let Ok(code) = u16::try_from(armed)
                    && let Ok(status) = StatusCode::from_u16(code)
                {
                    return Err(LlmClientError::UpstreamHttp {
                        status,
                        body: "upstream still cooling down after a recent failure".to_string(),
                    });
                }
            }
            return Err(LlmClientError::TemporarilyUnavailable);
        }

        // A Responses upstream may declare that it requires a list-shaped `input`, and a
        // hosted upstream may embed reasoning behind a marker. Both are per-target wire
        // facts, so one helper supplies them to every translation site.
        let mut policy = translation_policy(backend);
        // Provenance is resolved here, before the request is encoded and therefore before the
        // preserved-body fast path can replay anything: the encoder decides what "proven" means,
        // but this is the only place where the registry and this request's target meet.
        apply_reasoning_admission(
            &mut policy,
            backend,
            model,
            &llm_request,
            metadata,
            &self.provenance,
        );
        let mut body = encode_request_with_policy(&llm_request, wire_format, &policy)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?;
        // `encode_request` round-trips a preserved same-format body verbatim,
        // which keeps the caller's original `model`; force the resolved model so
        // the upstream always sees the target id.
        set_json_model(&mut body, model);
        if matches!(backend, Backend::OpenAiResponses(_)) {
            sanitize_openai_responses_provider_body(&mut body);
        }
        // Strip before `merge_extra_body` so a target can reinstate either field
        // deliberately via `extra_body`.
        if matches!(backend, Backend::Anthropic(_)) {
            strip_anthropic_incompatible_fields(&mut body);
            strip_unsigned_thinking_blocks(&mut body);
        }
        omit_configured_body_fields(&mut body, backend.omit_body_fields());
        merge_extra_body(&mut body, backend.extra_body());
        // After the merge on purpose: the effort override must win over both the caller's
        // value and any `reasoning` default a target set through `extra_body`.
        apply_reasoning_effort(&mut body, backend);
        // Last, because the policy is the target's authoritative choice. `extra_body` only
        // fills keys the request omits, so a caller that set `reasoning` or
        // `chat_template_kwargs` would otherwise re-enable reasoning on a strict target.
        apply_reasoning_policy(&mut body, backend);
        // A route's graded level outranks the target's effort, so it is written after
        // `apply_reasoning_effort` and before the on/off policy fragment. Only the single key
        // the transport uses for effort is touched, so an unrelated `reasoning` field the
        // caller set survives.
        if let Some(effort) = route_reasoning_effort
            && let Value::Object(object) = &mut body
        {
            match backend {
                Backend::OpenAiResponses(_) => {
                    let reasoning = object
                        .entry("reasoning".to_string())
                        .or_insert_with(|| Value::Object(serde_json::Map::new()));
                    if !reasoning.is_object() {
                        *reasoning = Value::Object(serde_json::Map::new());
                    }
                    if let Value::Object(reasoning) = reasoning {
                        reasoning.insert("effort".to_string(), Value::String(effort.to_string()));
                    }
                }
                Backend::OpenAiChat(_) => {
                    object.insert(
                        "reasoning_effort".to_string(),
                        Value::String(effort.to_string()),
                    );
                }
                Backend::Anthropic(_) => {}
            }
        }
        // A route's explicit policy outranks the target's default and the caller's effort.
        // Applied last so the route is genuinely authoritative.
        if let Some(policy) = route_reasoning_policy
            && let Some(fragment) = policy.wire_body(backend.reasoning_dialect())
            && let (Value::Object(fragment), Value::Object(object)) = (&fragment, &mut body)
        {
            for (key, value) in fragment {
                match value {
                    Value::Object(fields) => {
                        let entry = object
                            .entry(key.clone())
                            .or_insert_with(|| Value::Object(Default::default()));
                        if !entry.is_object() {
                            *entry = Value::Object(Default::default());
                        }
                        if let Value::Object(entry) = entry {
                            entry.extend(fields.clone());
                        }
                    }
                    _ => {
                        object.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        // The provider's refusal list has the final word: after every override, a value the
        // provider rejects is removed rather than forwarded. Removing it leaves the provider's
        // own default, which is the only other outcome that wire allows.
        drop_unsupported_reasoning_effort(&mut body, backend.unsupported_reasoning_effort());
        if matches!(backend, Backend::Anthropic(_)) {
            enable_anthropic_prompt_caching(&mut body);
        }
        if matches!(backend, Backend::OpenAiChat(_)) {
            ensure_openai_stream_usage(&mut body);
        }
        let streaming = endpoint.allows_streaming()
            && body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let url = endpoint.url(backend);
        record_gen_ai_request(&url, model, streaming);

        let result = self
            .send_with_retries(&url, backend, &body, metadata, model, streaming)
            .await;
        // Forwarded credentials can hit a user's quota while the backend remains healthy.
        if let Some(until) = unavailable_until
            && let Err(error) = &result
            && rests_the_model(error)
            && !(backend.is_forwarding_auth()
                && matches!(error, LlmClientError::UpstreamHttp { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS))
        {
            let deadline = duration_millis(self.cooldown_epoch.elapsed().saturating_add(cooldown));
            if let Some(status) = self.cooldown_status.get(&key) {
                let observed = match error {
                    LlmClientError::UpstreamHttp { status, .. } => status.as_u16(),
                    _ => 0,
                };
                status.store(u32::from(observed), Ordering::Relaxed);
            }
            until.fetch_max(deadline, Ordering::Relaxed);
        }
        // Let cooldown expire naturally; a successful in-flight call may overlap a newer failure.
        result
    }

    // Sends the encoded body, retrying retryable failures within the backend's retry budget.
    async fn send_with_retries(
        &self,
        url: &str,
        backend: &Backend,
        body: &Value,
        metadata: Option<&Metadata>,
        model: &ModelId,
        streaming: bool,
    ) -> Result<EncodedResponse> {
        let max_retries = u64::from(backend.max_retries());
        let max_attempts = max_retries + 1;
        // A target that declares a daily request allowance charges that budget per
        // attempt, so its retry budget is separate from a paid lane's (see
        // AttemptFailure::retry_budget_allows). Resolved once: the backend cannot gain
        // or lose an allowance during a call.
        let quota_limited = backend.allowance().is_some();
        let expires_at = backend
            .timeout()
            .map(|timeout| (tokio::time::sleep(timeout).deadline(), timeout));
        let deadline = async {
            match expires_at {
                Some((at, timeout)) => {
                    tokio::time::sleep_until(at).await;
                    timeout
                }
                None => std::future::pending().await,
            }
        };
        tokio::pin!(deadline);
        let mut attempt = 0_u64;
        loop {
            if let Some((at, timeout)) = expires_at
                && tokio::time::Instant::now() >= at
            {
                return Err(deadline_error(timeout));
            }
            let span = tracing::debug_span!(
                target: "libsy",
                "libsy.upstream_attempt",
                model = %model,
                wire_format = %backend.wire_format(),
                attempt = attempt + 1,
                max_attempts,
                retry = attempt > 0,
                openinference.span.kind = "CHAIN",
                outcome = tracing::field::Empty,
                status_code = tracing::field::Empty,
                will_retry = tracing::field::Empty,
                retry_delay_ms = tracing::field::Empty,
            );
            let mut attempt_started = false;
            // Per-attempt bound: send, response headers and the first decoded event.
            // It stops applying once this attempt returns, so it can never cut a
            // healthy stream short. Expiry here happens before anything reaches the
            // caller, which is what makes it safe to try another candidate.
            let attempt_deadline = backend.attempt_timeout().map(|bound| {
                let at = tokio::time::Instant::now() + bound;
                match expires_at {
                    Some((overall, _)) => (at.min(overall), bound),
                    None => (at, bound),
                }
            });
            let attempt_expired = async {
                match attempt_deadline {
                    Some((at, bound)) => {
                        tokio::time::sleep_until(at).await;
                        bound
                    }
                    None => std::future::pending().await,
                }
            };
            tokio::pin!(attempt_expired);
            let result = tokio::select! {
                biased;
                timeout = &mut deadline => {
                    if attempt_started {
                        metrics::record_upstream_attempt(None);
                    }
                    span.record("outcome", "error");
                    span.record("will_retry", false);
                    return Err(deadline_error(timeout));
                }
                bound = &mut attempt_expired => {
                    if attempt_started {
                        metrics::record_upstream_attempt(None);
                    }
                    span.record("outcome", "error");
                    span.record("will_retry", false);
                    Err(AttemptFailure {
                        error: attempt_timeout_error(bound),
                        status: None,
                        retry_after: None,
                    })
                }
                result = async {
                    attempt_started = true;
                    self.send_once(url, backend, body, metadata, model, streaming).await
                }.instrument(span.clone()) => result,
            };
            // The retained handle updates this same attempt span with its outcome.
            match result {
                Ok(response) => {
                    span.record("outcome", "ok");
                    span.record("status_code", response.status());
                    span.record("will_retry", false);
                    if attempt > 0 {
                        metrics::record_retry_recovered();
                    }
                    return Ok(match response {
                        EncodedResponse::Streaming {
                            status,
                            chunks,
                            upstream_headers,
                        } => EncodedResponse::Streaming {
                            status,
                            // Post-commit bounds: the caller holds this stream, so an
                            // expiry is terminal rather than a fallback opportunity.
                            chunks: stream_with_idle_deadline(
                                stream_with_deadline(chunks, expires_at),
                                backend.stream_idle_timeout(),
                            ),
                            upstream_headers,
                        },
                        buffered => buffered,
                    });
                }
                Err(failure) => {
                    let will_retry = failure.may_retry(attempt, max_retries, quota_limited);
                    span.record("outcome", "error");
                    if let Some(status) = failure.status {
                        span.record("status_code", status.as_u16());
                    }
                    span.record("will_retry", will_retry);
                    if !will_retry {
                        return Err(failure.error);
                    }

                    let delay = retry_delay(attempt, failure.retry_after);
                    span.record("retry_delay_ms", duration_millis(delay));
                    // Close the attempt span before sleeping so backoff is not attempt latency.
                    drop(span);
                    tokio::select! {
                        biased;
                        timeout = &mut deadline => return Err(deadline_error(timeout)),
                        _ = tokio::time::sleep(delay) => {}
                    }
                    attempt += 1;
                }
            }
        }
    }

    // Performs one HTTP attempt and retains the retry metadata alongside any error.
    async fn send_once(
        &self,
        url: &str,
        backend: &Backend,
        body: &Value,
        metadata: Option<&Metadata>,
        model: &ModelId,
        streaming: bool,
    ) -> std::result::Result<EncodedResponse, AttemptFailure> {
        // The payload is sent as assembled. The egress gate that used to inspect this body
        // here is removed: nothing about a request's content stops it reaching a provider.
        // Paced before the daily budget, because the two refusals mean different things and
        // only one of them is reversible. A minute window has room again shortly, so a
        // request refused here should not spend a unit of the day's budget; the reservation
        // is released again below if the daily allowance then refuses, since nothing reached
        // the provider. Ordering it this way also means a route under minute pressure pays in
        // the resource that refills rather than the one that does not.
        let mut pacing = None;
        if let Some(quota) = backend.pacing() {
            match quota.reserve(crate::allowance::estimate_input_tokens(body)) {
                Ok(ticket) => pacing = ticket,
                Err(crate::allowance::PacingError::Throttled(throttled)) => {
                    return Err(AttemptFailure {
                        error: LlmClientError::MinuteAllowanceThrottled {
                            model: model.clone(),
                            scope: throttled.scope.as_str().to_string(),
                            retry_after_secs: throttled.retry_after.as_secs().max(1),
                        },
                        // No request left this process, so there is no upstream status to
                        // report and nothing was counted by the provider.
                        status: None,
                        retry_after: Some(throttled.retry_after),
                    });
                }
                Err(crate::allowance::PacingError::ExceedsWindow {
                    needed,
                    limit,
                    margin_permille,
                }) => {
                    // Its own classification, not a capability defect: this model could read
                    // the input, the deployment's own minute budget cannot admit it at the
                    // margin it insists on. A sibling with a wider window may serve the
                    // request now, so the route advances instead of ending.
                    return Err(AttemptFailure {
                        error: LlmClientError::InputExceedsMinuteWindow {
                            model: model.clone(),
                            needed,
                            limit,
                            margin_permille,
                        },
                        status: None,
                        retry_after: None,
                    });
                }
            }
        }
        // Spent before the request leaves, at the one point every attempt passes
        // through. A reservation is never refunded: an attempt that fails below may
        // still have reached the provider, and refunding it would let a retrying
        // client exceed the daily cap it is trying to respect.
        if let Some(allowance) = backend.allowance() {
            match allowance.try_reserve() {
                Ok(()) => {}
                Err(crate::allowance::ReserveError::Exhausted) => {
                    if let (Some(quota), Some(ticket)) = (backend.pacing(), pacing.as_ref()) {
                        quota.release(ticket);
                    }
                    return Err(AttemptFailure {
                        error: LlmClientError::DailyRequestAllowanceExhausted {
                            model: model.clone(),
                            limit: allowance.limit(),
                        },
                        status: None,
                        retry_after: None,
                    });
                }
                Err(crate::allowance::ReserveError::ProviderExhausted { limit }) => {
                    if let (Some(quota), Some(ticket)) = (backend.pacing(), pacing.as_ref()) {
                        quota.release(ticket);
                    }
                    return Err(AttemptFailure {
                        error: LlmClientError::ProviderAllowanceExhausted {
                            model: model.clone(),
                            limit,
                        },
                        status: None,
                        retry_after: None,
                    });
                }
                Err(crate::allowance::ReserveError::Unusable(reason)) => {
                    if let (Some(quota), Some(ticket)) = (backend.pacing(), pacing.as_ref()) {
                        quota.release(ticket);
                    }
                    return Err(AttemptFailure {
                        error: LlmClientError::DailyRequestAllowanceUnavailable {
                            model: model.clone(),
                            reason,
                        },
                        status: None,
                        retry_after: None,
                    });
                }
                // A reservation path never orders an observation, so this cannot be
                // produced here. It is handled as a closed refusal rather than a panic: if
                // the invariant ever breaks, the request must not be sent unaccounted.
                Err(crate::allowance::ReserveError::ObservationRejected { reason }) => {
                    if let (Some(quota), Some(ticket)) = (backend.pacing(), pacing.as_ref()) {
                        quota.release(ticket);
                    }
                    return Err(AttemptFailure {
                        error: LlmClientError::DailyRequestAllowanceUnavailable {
                            model: model.clone(),
                            reason: reason.to_string(),
                        },
                        status: None,
                        retry_after: None,
                    });
                }
            }
        }
        let client = if backend.is_forwarding_auth() {
            &self.forward_auth_client
        } else {
            &self.client
        };
        let builder = client.post(url).json(body);
        let builder = forward_metadata_headers(builder, metadata, backend);
        let builder = backend.apply_forwarded_auth(builder, metadata);
        let builder = apply_extra_headers(builder, backend);
        let builder = backend.apply_auth(builder);

        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                metrics::record_upstream_attempt(None);
                return Err(AttemptFailure {
                    error: convert_reqwest_error(error),
                    status: None,
                    retry_after: None,
                });
            }
        };
        let status = response.status();
        if status.is_success() {
            if streaming {
                let upstream_headers = response.headers().clone();
                let chunks = match prepare_response_stream(response, backend, model).await {
                    Ok(chunks) => chunks,
                    Err(error) => {
                        metrics::record_upstream_attempt(None);
                        return Err(AttemptFailure {
                            error,
                            status: Some(status),
                            retry_after: None,
                        });
                    }
                };
                metrics::record_upstream_attempt(Some(status.as_u16()));
                return Ok(EncodedResponse::Streaming {
                    status: status.as_u16(),
                    chunks,
                    upstream_headers,
                });
            }
            let upstream_headers = response.headers().clone();
            let body = match response.bytes().await {
                Ok(body) => body,
                Err(error) => {
                    metrics::record_upstream_attempt(None);
                    return Err(AttemptFailure {
                        error: convert_reqwest_error(error),
                        status: Some(status),
                        retry_after: None,
                    });
                }
            };
            // The provider's own input count, fed straight back into the window that
            // admitted this request. This is the only place the two are both in scope: the
            // reservation lives here, and the response body is about to leave for the
            // decoder. Doing it before returning matters — the very next request on this
            // model reads this ledger, and an under-count left uncorrected is exactly how a
            // minute window silently overruns its published budget.
            //
            // Parsed only for a paced target, so an unpaced one pays nothing. A body this
            // cannot read leaves the conservative reservation in place: "no count" is not
            // "no tokens".
            if let (Some(quota), Some(ticket)) = (backend.pacing(), pacing.as_ref())
                && let Some(reported) = serde_json::from_slice::<Value>(&body)
                    .ok()
                    .as_ref()
                    .and_then(crate::allowance::reported_input_tokens)
            {
                quota.reconcile(ticket, reported);
            }
            metrics::record_upstream_attempt(Some(status.as_u16()));
            return Ok(EncodedResponse::Buffered {
                status: status.as_u16(),
                body: body.to_vec(),
                upstream_headers,
            });
        }

        // A provider that issues a correlation id puts it in a response header, and
        // this is the last point where that header exists: the body read consumes the
        // response, and every caller above sees only the error.
        let request_id = upstream_request_id(response.headers());
        let retry_after = retry_after_delay(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => {
                metrics::record_upstream_attempt(None);
                return Err(AttemptFailure {
                    error: convert_reqwest_error(error),
                    status: Some(status),
                    retry_after,
                });
            }
        };
        let body = redact_forwarded_headers(body, metadata, backend.is_forwarding_auth());
        // Carried in the body so one extractor serves both transports: a provider
        // that reports the id as a field and one that reports it in a header end up
        // in the same place.
        let body = with_request_id(body, request_id.as_deref());
        metrics::record_upstream_attempt(Some(status.as_u16()));
        let error =
            if status == reqwest::StatusCode::BAD_REQUEST && backend.is_context_overflow(&body) {
                LlmClientError::ContextWindowExceeded {
                    model: model.clone(),
                    message: body,
                }
            } else {
                LlmClientError::UpstreamHttp { status, body }
            };
        Err(AttemptFailure {
            error,
            status: Some(status),
            retry_after,
        })
    }

    /// Calls the backend for `model_name` (or the request's own model), over the
    /// wire format the request pins in its metadata (else the model's default
    /// backend), and returns the neutral response.
    ///
    /// Resolution: `model_name` wins over `request.llm_request.model`; the
    /// resolved name is both the outer map key and the model id written into the
    /// request before translation. Missing models are invalid requests; unknown
    /// models or wire formats are configuration errors.
    pub async fn call_rewrite_model(
        &self,
        request: Request,
        model_name: Option<&ModelId>,
    ) -> Result<Response> {
        let Request {
            mut llm_request,
            metadata,
            route_reasoning_policy,
            route_reasoning_effort,
            ..
        } = request;

        let model_id = model_name
            .cloned()
            .or_else(|| llm_request.model.map(ModelId::from))
            .ok_or_else(|| LlmClientError::InvalidRequest {
                message: "no model given".to_string(),
            })?;
        llm_request.model = Some(model_id.to_string());

        // A caller that asked for zero output gets an empty answer by request, not by
        // failure. Captured here because the request is moved into the send below.
        let wants_no_output = llm_request.output.max_output_tokens == Some(0);

        let orig_format = metadata.as_ref().and_then(|m| m.wire_format);
        let wire_format = orig_format.unwrap_or(
            self.model_to_config
                .get(&model_id)
                .map(|config| config.default_backend.wire_format())
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("no backend configured for model {model_id:?}"),
                })?,
        );
        let backend = self.backend_for(&model_id, wire_format).ok_or_else(|| {
            LlmClientError::Configuration {
                message: format!("model {model_id:?} has no backend for format {wire_format}"),
            }
        })?;

        let http_response = self
            .send_encoded(
                backend,
                wire_format,
                llm_request,
                metadata.as_ref(),
                &model_id,
                UpstreamEndpoint::Completion,
                route_reasoning_policy,
                route_reasoning_effort.as_deref(),
            )
            .await?;

        let (llm_response, upstream_headers) = match http_response {
            EncodedResponse::Streaming {
                chunks,
                upstream_headers,
                ..
            } => (
                LlmResponse::Stream(
                    // Streamed state is recorded from the assembled response when the stream
                    // ends: the same accumulator the consumer uses, so the recorded details are
                    // the ones the caller will actually see.
                    tap_streamed_provenance(
                        chunks,
                        Arc::clone(&self.provenance),
                        reasoning_issuer(backend, &model_id),
                        request_lineage(metadata.as_ref()),
                    ),
                ),
                upstream_headers,
            ),
            EncodedResponse::Buffered {
                body,
                upstream_headers,
                ..
            } => {
                let body = serde_json::from_slice::<Value>(&body).map_err(|error| {
                    LlmClientError::ResponseTranslation(format!("invalid upstream JSON: {error}"))
                })?;
                // Map a provider's failed generation to 502, even under HTTP 200.
                // Redact forwarded credentials before returning the provider error.
                let policy = translation_policy(backend);
                let agg = decode_aggregated_response_with_policy(&body, wire_format, &policy)
                    .map_err(|error| match error {
                        TranslationError::UpstreamFailure { error } => {
                            LlmClientError::UpstreamHttp {
                                status: StatusCode::BAD_GATEWAY,
                                body: redact_forwarded_headers(
                                    json!({ "error": error }).to_string(),
                                    metadata.as_ref(),
                                    backend.is_forwarding_auth(),
                                ),
                            }
                        }
                        error => LlmClientError::ResponseTranslation(error.to_string()),
                    })?;
                // Retain which target issued this response's opaque reasoning state, so a later
                // request can be proven a continuation rather than assumed one.
                record_reasoning_provenance(
                    &agg,
                    backend,
                    &model_id,
                    metadata.as_ref(),
                    &self.provenance,
                );
                (LlmResponse::Agg(agg), upstream_headers)
            }
        };

        // A delivered transport with an unusable result is not a successful call. Recorded
        // here, where the decoded response exists and the wire status no longer does, so
        // that"the model answered" and "the caller got something" stay separate facts.
        //
        // The route then advances rather than returning the non-answer. A candidate that
        // answered with nothing has not served the request, and this layer knows it while
        // the caller does not; handing back an empty 200 is how a leg that never works keeps
        // being selected. Measured 2026-11-21: measured selection kept choosing a free Gemma
        // leg that returned `completion_tokens: 0` because an empty reply was recorded as a
        // success. The one caller that legitimately wants no output asks for zero tokens, and
        // that request is left alone.
        // The policy is read for one fact: whether this target declares that it writes
        // reasoning inline. That is what makes a token total above input-plus-output mean
        // thinking rather than a provider accounting quirk.
        let hosted_thinking = translation_policy(backend).hosted_reasoning.is_some();
        if let Some(reason) = llm_response
            .as_agg()
            .and_then(|agg| unusable_answer(agg, hosted_thinking))
        {
            metrics::record_incomplete_answer(reason);
            // Two shapes are refused: a reported output count of zero, and a reply whose
            // whole output was hidden reasoning on a target declared to charge it against
            // the caller's reservation. A provider that reports no usage has told us
            // nothing about its output, and a reply with output tokens but no visible text
            // is a reasoning- or format-level question; convicting either would turn a
            // diagnostic into a false accusation. The caller that legitimately wants no
            // output asks for zero tokens, and that request is left alone.
            let unanswered = matches!(
                reason,
                "empty_answer_zero_completion" | "empty_answer_thinking_only"
            );
            if unanswered && !wants_no_output {
                return Err(LlmClientError::EmptyAnswer {
                    model: model_id,
                    reason,
                });
            }
        }
        Ok(Response {
            llm_response,
            metadata,
            upstream_headers,
        })
    }

    /// The whole decode → call → encode path a wire endpoint needs, in one call.
    ///
    /// Decodes `raw_http_request` from `wire_format` to the neutral IR, serves it via
    /// [`call_rewrite_model`](Self::call_rewrite_model) — the *upstream* wire format is
    /// resolved there from the model's backend, independently of `wire_format` — then
    /// encodes the neutral response back into `wire_format`. The result is a buffered
    /// [`RawResponse::Buffered`] JSON body or a streamed [`RawResponse::Stream`] of
    /// wire events (the caller frames the stream as SSE). The response's `model` is
    /// restamped with the model that actually served the call, so the body names the
    /// model that answered rather than the route the caller addressed.
    ///
    /// `http_headers` are carried through as the request's
    /// [`Metadata::http_headers`]. Backends with `forward_auth` disabled forward only
    /// allowed metadata headers; `forward_auth` backends forward all application
    /// headers. All backends reachable through a forwarding route must use the same
    /// provider. Transport headers are always rebuilt. Pass `None` to forward nothing.
    pub async fn call_rewrite_model_raw(
        &self,
        raw_http_request: Value,
        http_headers: Option<http::HeaderMap>,
        model: Option<&ModelId>,
        wire_format: WireFormat,
    ) -> Result<RawResponse> {
        let llm_request = decode_request(wire_format, &raw_http_request)
            .map_err(|error| LlmClientError::RequestTranslation(error.to_string()))?;
        let request_extensions = llm_request.extensions.clone();
        // The model that serves the call — the rewrite target when the caller pinned
        // one, else the request's own model. Mirrors `call_rewrite_model`'s own
        // resolution so the response names whoever answered.
        let served_model = model
            .map(ModelId::to_string)
            .or_else(|| llm_request.model.clone());

        let request = Request {
            llm_request,
            raw_request: None,
            route_reasoning_policy: None,
            route_reasoning_effort: None,
            metadata: Some(Metadata {
                session_id: None,
                agent_id: None,
                task_id: None,
                correlation_id: None,
                extra_metadata: None,
                http_headers,
                wire_format: None,
                ..Default::default()
            }),
        };
        let response = self.call_rewrite_model(request, model).await?;

        match response.llm_response {
            LlmResponse::Agg(agg) => {
                let body = encode_aggregated_response_with_extensions(
                    &agg,
                    wire_format,
                    served_model.as_deref(),
                    &request_extensions,
                )
                .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
                Ok(RawResponse::Buffered(body))
            }
            LlmResponse::Stream(chunks) => {
                let events = encode_stream_with_extensions(
                    chunks,
                    wire_format,
                    served_model,
                    &request_extensions,
                )?;
                Ok(RawResponse::Stream(events))
            }
        }
    }
}

#[async_trait]
impl RoutedLlmClient for TranslatingLlmClient {
    async fn call(&self, request: Request) -> Result<Response> {
        self.call_rewrite_model(request, None).await
    }
}

#[derive(Clone, Copy)]
enum UpstreamEndpoint {
    Completion,
    Auxiliary(AuxiliaryOperation),
}

impl UpstreamEndpoint {
    fn url(self, backend: &Backend) -> String {
        match self {
            UpstreamEndpoint::Completion => backend.url(),
            UpstreamEndpoint::Auxiliary(operation) => operation.url(backend),
        }
    }

    fn allows_streaming(self) -> bool {
        matches!(self, UpstreamEndpoint::Completion)
    }
}

enum EncodedResponse {
    Buffered {
        status: u16,
        body: Vec<u8>,
        upstream_headers: HeaderMap,
    },
    Streaming {
        status: u16,
        chunks: LlmResponseStream,
        upstream_headers: HeaderMap,
    },
}

impl EncodedResponse {
    fn status(&self) -> u16 {
        match self {
            EncodedResponse::Buffered { status, .. }
            | EncodedResponse::Streaming { status, .. } => *status,
        }
    }
}

// The typed error decides retry eligibility; status and Retry-After feed
// attempt telemetry and delay selection.
struct AttemptFailure {
    error: LlmClientError,
    status: Option<StatusCode>,
    retry_after: Option<Duration>,
}

fn deadline_error(timeout: Duration) -> LlmClientError {
    LlmClientError::Timeout {
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!(
                "response did not finish within {} ms, retries included",
                duration_millis(timeout)
            ),
        )),
    }
}

// Pre-commit expiry: no response was returned for this attempt, so routing may
// still choose another candidate. Kept distinct from `deadline_error` so the
// fallback decision never has to guess which deadline fired.
fn attempt_timeout_error(bound: Duration) -> LlmClientError {
    LlmClientError::AttemptTimeout {
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!(
                "attempt produced no usable output within {} ms",
                duration_millis(bound)
            ),
        )),
    }
}

// Post-commit inactivity: the caller holds the stream and already received at
// least one event, so this is a terminal `Timeout`, never a fallback trigger.
fn idle_timeout_error(idle: Duration) -> LlmClientError {
    LlmClientError::Timeout {
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!(
                "stream produced no event within {} ms",
                duration_millis(idle)
            ),
        )),
    }
}

impl AttemptFailure {
    fn is_retryable(&self) -> bool {
        is_transient_failure(&self.error)
    }

    /// Whether the retry loop may dispatch once more to this same target.
    ///
    /// One decision in one place, and it fails closed: the configured budget, whether the
    /// failure class can clear at all, and - on a target that declares a daily request
    /// allowance - the narrower budget that allowance implies. Keeping the conjunction
    /// here means a caller cannot accidentally retry a 401 by checking only the budget.
    fn may_retry(&self, attempt: u64, max_retries: u64, quota_limited: bool) -> bool {
        if attempt >= max_retries || !self.is_retryable() {
            return false;
        }
        self.retry_budget_allows(attempt, quota_limited)
    }

    /// Whether a quota-limited leaf may spend one more attempt on this same target.
    ///
    /// [`Self::is_retryable`] answers whether the condition is the kind that can clear;
    /// this answers whether spending another request on the *same* target is worth it.
    /// They are different questions on a target that declares a daily request
    /// allowance, because every attempt reserves against that finite budget before the
    /// provider is contacted and the reservation is never refunded on an HTTP failure.
    /// A retry there spends the same scarce resource the retry is trying to obtain: one
    /// 429 currently charges four reservations while the routing evidence records one
    /// attempt.
    ///
    /// The budget for a quota-limited leaf is:
    ///
    /// * HTTP 429, any flavour: no further attempt on this target. A daily cap cannot
    ///   clear on a backoff, and sleeping into a minute window spends the room the next
    ///   request needs. The route decides whether an eligible sibling has headroom.
    /// * any other retryable failure (transport, timeout, 5xx): at most ONE further
    ///   attempt, so a bad second of upstream health costs two reservations, not four.
    ///
    /// A target with no allowance keeps the configured budget unchanged. Paid lanes are
    /// not quota-limited per request and this rule must not move them.
    fn retry_budget_allows(&self, attempt: u64, quota_limited: bool) -> bool {
        if !quota_limited {
            return true;
        }
        if self.status.is_some_and(|status| status.as_u16() == 429) {
            return false;
        }
        attempt == 0
    }
}

/// Whether a failure says the credential's account-wide allowance is exhausted.
///
/// OpenRouter reports this on a 429 as `error.metadata.limit_source =
/// "openrouter_free_tier_daily"`, an allowance shared by every free model on the
/// credential. Two decisions need this answer and must agree on it:
///
/// - the retry loop, because retrying a daily cap immediately cannot succeed and each
///   attempt is another request against an allowance already at zero;
/// - the route's candidate fallback, because every candidate draws on the same allowance.
///
/// One definition, so the two layers cannot drift apart.
///
/// The same account-shaped fact also arrives on a 402 (`payment_required`) and on a
/// 403 carrying a budget `limit_source`. Those statuses are NOT treated as account
/// exhaustion here in general: a 402 is the caller's balance, and a bare 403 is a
/// guardrail or policy refusal that must stay terminal. Only a status with an explicit
/// budget `limit_source` qualifies, so a content-policy 403 still refuses.
/// The provider's own `error.code` values for a 429 that retrying cannot clear.
///
/// OpenAI documents these as billing, spend, or quota conditions rather than throttles,
/// and states plainly that "retrying billing, spend, or quota errors won't restore API
/// access". They carry no OpenRouter `metadata.limit_source`, so recognizing the marker
/// alone leaves them classified as ordinary rate limits.
///
/// The list is deliberately closed and drawn from published provider documentation. An
/// unknown code stays a rate limit, because guessing that an unrecognized 429 is
/// permanent would strand a request that another attempt could have served.
const BILLING_LIMIT_CODES: &[&str] = &[
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "insufficient_quota",
];

/// The free-model daily allowance, spelled out in a message instead of the marker.
///
/// OpenRouter normally names this allowance in `error.metadata.limit_source`, but not
/// always: measured 2026-11-19 the refusal arrived as
/// `{"error":{"code":"upstream_error","type":"upstream_error","message":"Rate limit
/// exceeded: free-models-per-day-high-balance."}}` — a bare 429 with no metadata, on a
/// paid-tier key with credit remaining. The sentence can only name the per-credential
/// free-model allowance, so it belongs with the marker rather than in the retry loop.
///
/// Matched on the whole phrase and not on "rate limit exceeded": the per-minute
/// spelling of the same sentence is a genuine throttle that must keep its retries.
const FREE_MODEL_DAILY_MESSAGE: &str = "free-models-per-day";

fn names_free_model_daily_cap(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value["error"]["message"].as_str().map(str::to_string))
        .is_some_and(|message| message.contains(FREE_MODEL_DAILY_MESSAGE))
}

fn is_documented_billing_limit(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value["error"]["code"]
                .as_str()
                .or_else(|| value["error"]["type"].as_str())
                .map(str::to_string)
        })
        .is_some_and(|code| BILLING_LIMIT_CODES.contains(&code.as_str()))
}

pub(crate) fn is_account_wide_quota(source: &LlmClientError) -> bool {
    let LlmClientError::UpstreamHttp { status, body } = source else {
        return false;
    };
    let limit_source = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value["error"]["metadata"]["limit_source"]
                .as_str()
                .map(str::to_string)
        });
    match *status {
        StatusCode::TOO_MANY_REQUESTS => {
            limit_source.is_some_and(|s| s == "openrouter_free_tier_daily")
                || is_documented_billing_limit(body)
                || names_free_model_daily_cap(body)
                // Google returns two different 429s. `quota_exceeded` is the daily one;
                // `rate_limit_exceeded` is a window that clears and is retried instead. The
                // daily one is only ACCOUNT-wide when the body positively scoped it that way:
                // this Free tier allocates RPD per model, so one Model's spent quota must not
                // take the rest of the pool down with it.
                || crate::google::parse(Some(status.as_u16()), body).is_some_and(|error| {
                    error.condition == crate::google::GoogleCondition::DailyQuota
                        && error.recovery().stop_candidate_list
                })
        }
        // Insufficient credits: the account balance is the shared resource, so no
        // candidate on this credential can serve the request.
        StatusCode::PAYMENT_REQUIRED => true,
        // A 403 with a budget limit_source is an account cap. A 403 without one is a
        // guardrail or policy refusal and is deliberately NOT included here.
        StatusCode::FORBIDDEN => limit_source.is_some(),
        // OpenRouter masks a billing failure as `server_error` and states the real cause
        // only in the sibling `error_type`. The status hides an account-level condition
        // rather than describing a transient infrastructure fault.
        status if status.is_server_error() => {
            crate::run::canonical_error_type(body) == Some("payment_required")
        }
        _ => false,
    }
}

/// A permanent provider condition reported behind a 5xx.
///
/// A 5xx normally means the provider's infrastructure failed and another attempt may
/// succeed. OpenRouter also uses `server_error` to mask an authentication, authorization
/// or billing failure, naming the real cause only in the sibling `error_type`. Retrying
/// such a response cannot succeed, so the retry loop must not treat it as transient — the
/// same marker the fallback classifier reads, applied here as well.
fn is_masked_permanent_failure(body: &str) -> bool {
    matches!(
        crate::run::canonical_error_type(body),
        Some("authentication" | "permission_denied" | "payment_required")
    )
}

/// Whether a failure should rest the model it came from, independent of whether the same
/// request may be sent to that model again.
///
/// These are two questions and Google separates them. A 503 `UNAVAILABLE` means the model is
/// overloaded: replaying the same call would fail the same way, but the model genuinely needs
/// to be left alone for a while. A 400 policy block means the request was refused without
/// saying anything about the model's health, so it must not rest it. Arming the cooldown on
/// `is_transient_failure` alone conflated the two, with the result that a target configured
/// with a long capacity cooldown never armed it on the very condition it was configured for:
/// the 31B's `failure_cooldown_ms = 300000` was unreachable because its own 503 correctly
/// forbids a same-model replay, and forbidding the replay was read as "nothing to cool down".
///
/// The retry decision is not loosened here. `is_transient_failure` still decides what may be
/// replayed; this only decides what rests.
fn rests_the_model(error: &LlmClientError) -> bool {
    if is_transient_failure(error) {
        return true;
    }
    match error {
        LlmClientError::UpstreamHttp { status, body } => {
            crate::google::parse(Some(status.as_u16()), body)
                .is_some_and(|error| error.recovery().cool_down_model)
        }
        _ => false,
    }
}

/// Why an answered response carries nothing a caller can use, if that is the case.
///
/// `HTTP 200` says the provider accepted the request and answered it. It does not say the
/// caller received an answer. The measured case is a Gemma reply whose entire output budget
/// went to inline `<thought>` reasoning: `completion_tokens: 0`, empty content, no tool
/// call, status 200. Treating that as a success is how a model looks healthy in transport
/// telemetry while every caller gets nothing, and it is how a selector learns the wrong
/// lesson about which leg serves a task.
///
/// Only the unambiguous shape is flagged: no text, no tool call, no refusal, and no output
/// the caller can use. A response with output tokens but no visible text is left alone —
/// that is a reasoning- or format-level question this function cannot settle, and guessing
/// would turn a diagnostic into a false accusation.
///
/// The one provider-specific certainty is `hosted_thinking`: a target that declares it writes
/// reasoning inline *and* charges that reasoning against the caller's output reservation. Its
/// reply can therefore contain thinking only, and its own `completion_tokens` reads `0`
/// because the provider does not count reasoning as output. Measured 2026-11-21 on Gemma 4
/// 26B: `finish_reason: length`, `completion_tokens: 0`, `total_tokens` 499 against a
/// 438-token prompt — 61 tokens of thinking and no answer. Classifying that as
/// "the provider produced nothing" is wrong twice over: it blames the leg for a budget that
/// the caller set, and it hides the only fact that would let an operator act on it. The
/// hidden count is taken from the usage block rather than assumed, so a provider that reports
/// no totals keeps the older, quieter classification.
fn unusable_answer(
    agg: &switchyard_protocol::llm::AggLlmResponse,
    hosted_thinking: bool,
) -> Option<&'static str> {
    use switchyard_protocol::llm::ContentBlock;

    let has_content = agg.outputs.iter().any(|output| {
        output.content.iter().any(|block| match block {
            ContentBlock::Text { text } => !text.trim().is_empty(),
            // A tool call is a usable answer: the caller acts on it. A refusal is an answer
            // too - the model declined, which is information, not silence.
            ContentBlock::ToolCall(_) | ContentBlock::Refusal { .. } => true,
            _ => false,
        })
    });
    if has_content {
        return None;
    }
    if hosted_thinking && hidden_output_tokens(&agg.usage) > 0 {
        return Some("empty_answer_thinking_only");
    }
    match agg.usage.output_tokens {
        Some(0) => Some("empty_answer_zero_completion"),
        None => Some("empty_answer_no_usage"),
        // Output was produced but none of it is visible: the model spent its budget on
        // reasoning, which is a real and separate condition, not an empty answer.
        Some(_) => Some("empty_answer_reasoning_only"),
    }
}

/// Tokens the provider produced that it did not count as output and did not show.
///
/// For a target that hides its reasoning this is the thinking the reply contains. Taken from
/// the provider's own totals, so an upstream that reports nothing reports zero here rather
/// than a guess: `total - input - output`, saturating, with the cache details subtracted from
/// the input side because the codec normalises input to its non-cached component.
fn hidden_output_tokens(usage: &switchyard_protocol::llm::Usage) -> u64 {
    let Some(total) = usage.total_tokens else {
        return 0;
    };
    let cache = usage.cache.as_deref();
    let input = usage.input_tokens.unwrap_or(0)
        + cache
            .and_then(|cache| cache.cached_input_tokens)
            .unwrap_or(0)
        + cache
            .and_then(|cache| cache.cache_creation_input_tokens)
            .unwrap_or(0);
    total
        .saturating_sub(input)
        .saturating_sub(usage.output_tokens.unwrap_or(0))
}

fn is_transient_failure(error: &LlmClientError) -> bool {
    // Checked before the status table: 429 is retryable in general, but a daily
    // account cap is not a throttle that clears in a moment.
    if is_account_wide_quota(error) {
        return false;
    }
    match error {
        LlmClientError::Transport { .. }
        | LlmClientError::Timeout { .. }
        | LlmClientError::AttemptTimeout { .. } => true,
        // A daily allowance resets on the provider's clock, not on a backoff: retrying
        // now would burn the next day's budget against an answer already known to be no.
        LlmClientError::DailyRequestAllowanceExhausted { .. }
        | LlmClientError::DailyRequestAllowanceUnavailable { .. } => false,
        // A minute window refills shortly, but not within a backoff: the same legs' next
        // attempt would spend the room the next request needs. The route decides whether a
        // sibling has headroom; this call does not sleep into its own throttle.
        LlmClientError::MinuteAllowanceThrottled { .. } => false,
        // No wait on this leg makes the request smaller, and resending it would spend the
        // window on a request this code already knows it must refuse.
        LlmClientError::InputExceedsMinuteWindow { .. } => false,
        // The model answered with nothing. The same budget on the same model produces the
        // same non-answer, so this call does not retry in place; the route advances.
        LlmClientError::EmptyAnswer { .. } => false,
        // The feature is missing from the target, not from the attempt: sending it again,
        // or to a sibling on the same wire format, cannot add it.
        LlmClientError::UnsupportedFeature { .. } => false,
        LlmClientError::UpstreamHttp { status, body } => {
            if status.is_server_error() && is_masked_permanent_failure(body) {
                return false;
            }
            // A provider that names its condition decides whether the same bytes go out
            // again. Google distinguishes a window throttle from a spent daily quota on the
            // same status, and a policy block from an outage on the same 400.
            if crate::google::parse(Some(status.as_u16()), body)
                .is_some_and(|error| !error.recovery().retry_same_model)
            {
                return false;
            }
            metrics::is_retryable_http_status(status.as_u16())
        }
        _ => false,
    }
}

fn stream_with_deadline(
    chunks: LlmResponseStream,
    deadline: Option<(tokio::time::Instant, Duration)>,
) -> LlmResponseStream {
    let Some((at, timeout)) = deadline else {
        return chunks;
    };
    stream::unfold(Some(chunks), move |chunks| async move {
        let mut chunks = chunks?;
        let chunk = tokio::select! {
            biased;
            _ = tokio::time::sleep_until(at) => return Some((Err(deadline_error(timeout)), None)),
            chunk = chunks.next() => chunk?,
        };
        let next = chunk.is_ok().then_some(chunks);
        Some((chunk, next))
    })
    .boxed()
}

// Bounds the gap between events on a stream the caller already holds. This is a
// post-commit bound: a slow but progressing stream is left alone, and an expiry
// surfaces as an in-band error rather than a new candidate.
fn stream_with_idle_deadline(
    chunks: LlmResponseStream,
    idle: Option<Duration>,
) -> LlmResponseStream {
    let Some(idle) = idle else {
        return chunks;
    };
    stream::unfold(Some(chunks), move |chunks| async move {
        let mut chunks = chunks?;
        let chunk = tokio::select! {
            biased;
            _ = tokio::time::sleep(idle) => return Some((Err(idle_timeout_error(idle)), None)),
            chunk = chunks.next() => chunk?,
        };
        let next = chunk.is_ok().then_some(chunks);
        Some((chunk, next))
    })
    .boxed()
}

async fn prepare_response_stream(
    response: reqwest::Response,
    backend: &Backend,
    model: &ModelId,
) -> Result<LlmResponseStream> {
    let bytes = response.bytes_stream().map(|chunk| {
        chunk
            .map(|bytes| bytes.to_vec())
            .map_err(convert_reqwest_error)
    });
    let mut chunks =
        decode_stream_with_policy(bytes, backend.wire_format(), &translation_policy(backend))?;
    // Hold non-output events until something usable arrives. A provider announces
    // the response (`response.created`, a role-only delta) and may then stall; if
    // the attempt returned at that announcement, the attempt bound would stop
    // applying with no output delivered and no bound left but the overall one.
    // Staying inside this future keeps the wait covered by the attempt deadline,
    // which also keeps the failure pre-commit and therefore routable.
    let mut held: Vec<LlmResponseStreamEvent> = Vec::new();
    loop {
        match chunks.next().await {
            None => break,
            Some(Err(error)) => {
                // Nothing usable was produced, so a transport failure here can
                // still be retried and another candidate may serve the request.
                return Err(error);
            }
            Some(Ok(event)) => {
                // An in-band context overflow skips retries and advances to another candidate.
                if let Some(message) = first_event_overflow(&Ok(event.clone()), backend) {
                    return Err(LlmClientError::ContextWindowExceeded {
                        model: model.clone(),
                        message,
                    });
                }
                let usable = event.normalized().iter().any(is_usable_output);
                held.push(event);
                if usable {
                    break;
                }
            }
        }
    }
    Ok(stream::iter(held.into_iter().map(Ok)).chain(chunks).boxed())
}

// Output a caller can act on. Identity and framing events (MessageStart, empty
// normalized frames) are deliberately not usable: they say a response exists,
// not that it produced anything.
fn is_usable_output(chunk: &LlmResponseChunk) -> bool {
    matches!(
        chunk,
        LlmResponseChunk::TextDelta { .. }
            | LlmResponseChunk::ReasoningDelta { .. }
            | LlmResponseChunk::ReasoningDetailsDelta { .. }
            | LlmResponseChunk::ToolCallDelta { .. }
            | LlmResponseChunk::MessageStop { .. }
            | LlmResponseChunk::Usage(_)
    )
}

// The overflow message when a stream's first event is an in-band provider rejection
// of the whole request, rather than the start of a response.
fn first_event_overflow(
    first: &Result<LlmResponseStreamEvent>,
    backend: &Backend,
) -> Option<String> {
    first
        .as_ref()
        .ok()?
        .normalized()
        .iter()
        .find_map(|chunk| match chunk {
            LlmResponseChunk::StreamError { message } if backend.is_context_overflow(message) => {
                Some(message.clone())
            }
            _ => None,
        })
}

// Uses Retry-After when supplied, capped so an upstream cannot stall a request indefinitely.
fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let delay = if let Ok(seconds) = value.parse::<u64>() {
        Duration::from_secs(seconds)
    } else {
        let retry_at = httpdate::parse_http_date(value).ok()?;
        retry_at
            .duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO)
    };
    Some(delay.min(MAX_RETRY_AFTER))
}

fn retry_delay(retry_number: u64, retry_after: Option<Duration>) -> Duration {
    // Retry-After wins; otherwise double 250 ms up to the two-second cap.
    retry_after.unwrap_or_else(|| {
        let multiplier = 1_u32 << retry_number.min(3);
        INITIAL_RETRY_DELAY
            .saturating_mul(multiplier)
            .min(MAX_RETRY_BACKOFF)
    })
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn record_gen_ai_request(url: &str, model: &str, streaming: bool) {
    let span = tracing::Span::current();
    span.record("gen_ai.request.model", model);
    if streaming {
        span.record("gen_ai.request.stream", true);
    }
    if let Ok(url) = reqwest::Url::parse(url) {
        if let Some(host) = url.host_str() {
            span.record("server.address", host);
        }
        if let Some(port) = url.port_or_known_default() {
            span.record("server.port", i64::from(port));
        }
    }
}

fn convert_reqwest_error(error: reqwest::Error) -> LlmClientError {
    // Reqwest labels truncated or otherwise unreadable response bodies as decode
    // errors, so distinguish them from serde JSON failures at the call site.
    let error = error.without_url();
    if error.is_timeout() {
        LlmClientError::Timeout {
            source: Box::new(error),
        }
    } else if error.is_builder() {
        LlmClientError::Configuration {
            message: format!("failed to build upstream request: {error}"),
        }
    } else {
        LlmClientError::Transport {
            source: Box::new(error),
        }
    }
}

// Forwards safe metadata headers, or all application headers when forwarding auth.
fn forward_metadata_headers(
    builder: RequestBuilder,
    metadata: Option<&Metadata>,
    backend: &Backend,
) -> RequestBuilder {
    let Some(headers) = metadata.and_then(|metadata| metadata.http_headers.as_ref()) else {
        return builder;
    };
    let mut forwarded = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let is_allowed = if backend.is_forwarding_auth() {
            !is_non_forwardable_header(name.as_str(), headers)
                && !backend.is_provider_owned_header(name.as_str())
        } else {
            is_allowed_metadata_header(name.as_str())
        };
        if !is_allowed {
            continue;
        }
        forwarded.append(name, value.clone());
    }
    builder.headers(forwarded)
}

// Adds the backend's custom per-call headers.
fn apply_extra_headers(mut builder: RequestBuilder, backend: &Backend) -> RequestBuilder {
    for (name, value) in backend.extra_headers() {
        builder = builder.header(name, value);
    }
    builder
}

// The translation policy one backend declares.
//
// Both facts are per-target wire properties: whether the Responses `input` must be a list,
// and which marker the upstream uses for reasoning it writes inline in the message. They are
// assembled here rather than at each encode/decode site because a decode path that forgets
// the marker silently returns the provider's monologue to the caller.
//
// The third fact is where opaque reasoning state may be replayed. `reasoning_details` and a
// `signature` belong to the provider that issued them: OpenRouter hands them back and expects
// them returned intact, while a native provider has no field for them and must not be given
// another provider's private blob. Tying this to the dialect means one target cannot be told
// to replay and another told to withhold by accident.
fn translation_policy(backend: &Backend) -> TranslationPolicy {
    // An OpenRouter-shaped target can *accept* opaque details, but acceptance is not proof: the
    // same wire reached on another account is another issuer. Such a target is provenance-bound,
    // and a target with no structured reasoning wire gets the plaintext form only.
    let reasoning_details_replay = match backend.reasoning_dialect() {
        switchyard_protocol::ReasoningDialect::OpenRouterEnabled => {
            ReasoningDetailsReplay::ProvenanceBound
        }
        _ => ReasoningDetailsReplay::PlaintextOnly,
    };
    TranslationPolicy {
        target_capabilities: TargetCapabilities {
            requires_list_input: Some(backend.requires_list_input()),
            ..TargetCapabilities::default()
        },
        hosted_reasoning: backend.hosted_reasoning(),
        reasoning_details_replay,
        ..TranslationPolicy::default()
    }
}

// The `HttpBackendConfig` behind any backend variant.
fn backend_config(backend: &Backend) -> &crate::backend::HttpBackendConfig {
    match backend {
        Backend::OpenAiChat(config)
        | Backend::OpenAiResponses(config)
        | Backend::Anthropic(config) => config,
    }
}

// The host part of a base URL, used as the human-readable provider name.
fn url_host(base_url: &str) -> String {
    let without_scheme = base_url.split_once("://").map_or(base_url, |(_, rest)| rest);
    without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(without_scheme)
        .to_string()
}

// A stable, non-secret identifier for one credential domain.
//
// This is what "the same account" means for provenance: two targets with the same base URL but
// different keys are different domains, and a forwarding target is a different domain again
// because the credential is the caller's, not ours. The key itself never leaves this function.
fn credential_domain(config: &crate::backend::HttpBackendConfig) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"switchyard.credential-domain.v1\0");
    hasher.update(config.base_url.as_bytes());
    hasher.update([0u8]);
    hasher.update(config.api_key.as_deref().unwrap_or(""));
    if config.forward_auth {
        hasher.update(b"\0forwarded");
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(16);
    for byte in &digest[..8] {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

// The identity that would issue opaque reasoning state on this target.
fn reasoning_issuer(backend: &Backend, model: &ModelId) -> ReasoningIssuer {
    let config = backend_config(backend);
    let provider = url_host(&config.base_url);
    ReasoningIssuer {
        client: format!("{provider}#{}", credential_domain(config)),
        model: model.as_str().to_string(),
        provider,
        wire: backend.wire_format().to_string(),
    }
}

// The lineage facts a request carries about itself.
//
// Switchyard has no inbound authentication: `agent_id` and `session_id` are caller-supplied
// headers and can be forged, so they are recorded as advisory only and never authorise replay.
// Until a verified caller boundary exists, every production request builds an untrusted lineage,
// and admission rests on the issuer (credential domain, model, wire) alone.
fn request_lineage(metadata: Option<&Metadata>) -> ReasoningLineage {
    ReasoningLineage::untrusted(
        metadata.and_then(|metadata| metadata.agent_id.clone()),
        metadata.and_then(|metadata| metadata.session_id.clone()),
    )
}

// Message indexes forming the active continuation edge.
//
// The most recent assistant turn is what the next call continues, provided nothing but tool
// results follows it. Its opaque state cannot be dropped without changing the request the
// provider is being asked to continue; older assistant turns are completed history.
fn active_message_indexes(request: &LlmRequest) -> BTreeSet<usize> {
    let mut active = BTreeSet::new();
    let Some(last_assistant) = request
        .messages
        .iter()
        .rposition(|message| message.role == Role::Assistant)
    else {
        return active;
    };
    if request.messages[last_assistant + 1..]
        .iter()
        .all(|message| message.role == Role::Tool)
    {
        active.insert(last_assistant);
    }
    active
}

// Resolves, before dispatch, which opaque state in this request this target may replay.
//
// This is the one place the registry and a request meet. The encoder is then a pure function of
// the request and the policy, so the same decision is enforced on the normalised path and on the
// preserved-body fast path alike.
fn apply_reasoning_admission(
    policy: &mut TranslationPolicy,
    backend: &Backend,
    model: &ModelId,
    request: &LlmRequest,
    metadata: Option<&Metadata>,
    registry: &ProvenanceRegistry,
) {
    // `Verbatim` replays everything by construction, so there is nothing to resolve. Every
    // other mode needs the active continuation edge identified, because that edge may not be
    // quietly narrowed: `PlaintextOnly` never admits opaque state, but it still has to refuse
    // active state rather than strip it.
    let mode = policy.reasoning_details_replay;
    if mode == ReasoningDetailsReplay::Verbatim {
        return;
    }
    let issuer = reasoning_issuer(backend, model);
    let lineage = request_lineage(metadata);
    let active = active_message_indexes(request);
    let mut admitted = Vec::new();
    let mut active_unproven = Vec::new();
    for (index, message) in request.messages.iter().enumerate() {
        for block in &message.content {
            let ContentBlock::Reasoning {
                details, signature, ..
            } = block
            else {
                continue;
            };
            if details.is_empty() && signature.is_none() {
                continue;
            }
            let digest = opaque_digest(details, signature.as_deref());
            // Only `ProvenanceBound` can admit; `PlaintextOnly` has no notion of proof and
            // therefore leaves every digest unproven.
            let proven = mode == ReasoningDetailsReplay::ProvenanceBound
                && registry
                    .resolve(&digest)
                    .is_some_and(|(recorded, recorded_lineage)| {
                        recorded.permits_replay_to(&issuer)
                            && recorded_lineage.permits_replay_in(&lineage)
                    });
            if proven {
                admitted.push(digest);
            } else if active.contains(&index) {
                active_unproven.push(digest);
            }
        }
    }
    policy.reasoning_admission = Some(ReasoningAdmission {
        issuer,
        admitted,
        active_unproven,
    });
}

// Records, for every opaque state block in a decoded response, which target issued it.
fn record_reasoning_provenance(
    response: &AggLlmResponse,
    backend: &Backend,
    model: &ModelId,
    metadata: Option<&Metadata>,
    registry: &ProvenanceRegistry,
) {
    let issuer = reasoning_issuer(backend, model);
    let mut lineage = request_lineage(metadata);
    lineage.response = response.id.clone();
    for output in &response.outputs {
        for block in &output.content {
            let ContentBlock::Reasoning {
                details, signature, ..
            } = block
            else {
                continue;
            };
            if details.is_empty() && signature.is_none() {
                continue;
            }
            let digest = opaque_digest(details, signature.as_deref());
            registry.record(digest, issuer.clone(), lineage.clone());
        }
    }
}

// Records the provenance of a streamed response when the stream ends.
//
// The tap folds the same `ResponseAccumulator` the consumer uses, so the details it records are
// the assembled ones and cannot diverge from what the caller receives.
fn tap_streamed_provenance(
    chunks: LlmResponseStream,
    registry: Arc<ProvenanceRegistry>,
    issuer: ReasoningIssuer,
    lineage: ReasoningLineage,
) -> LlmResponseStream {
    use switchyard_protocol::stream::ResponseAccumulator;

    let state = (chunks, ResponseAccumulator::new(), registry, issuer, lineage);
    Box::pin(stream::unfold(state, |state| async move {
        let (mut chunks, mut accumulator, registry, issuer, mut lineage) = state;
        match chunks.next().await {
            Some(Ok(event)) => {
                for chunk in event.normalized() {
                    accumulator.push(chunk.clone());
                }
                Some((Ok(event), (chunks, accumulator, registry, issuer, lineage)))
            }
            Some(Err(error)) => Some((Err(error), (chunks, accumulator, registry, issuer, lineage))),
            None => {
                let response = accumulator.finish();
                lineage.response = response.id.clone();
                for output in &response.outputs {
                    for block in &output.content {
                        let ContentBlock::Reasoning {
                            details, signature, ..
                        } = block
                        else {
                            continue;
                        };
                        if details.is_empty() && signature.is_none() {
                            continue;
                        }
                        registry.record(
                            opaque_digest(details, signature.as_deref()),
                            issuer.clone(),
                            lineage.clone(),
                        );
                    }
                }
                None
            }
        }
    }))
}

// The caller's own Responses body, when this request arrived in that format.
//
// The decoder records the verbatim request for same-format replay, which is the only place
// the fields a Chat target cannot carry survive. Without preservation there is nothing to
// inspect and the guard cannot fire; that is the pre-existing behaviour, and it is why the
// guard reads what was captured rather than reconstructing a body it never saw.
fn caller_responses_feature(
    request: &switchyard_protocol::LlmRequest,
) -> Option<switchyard_translation::UnsupportedResponsesFeature> {
    let format: switchyard_translation::FormatId =
        switchyard_protocol::WireFormat::OpenAiResponses.into();
    let body = request.preservation.requests.get(&format)?;
    switchyard_translation::unsupported_feature(body)
}

// Overwrites the outbound body's `model` field with the resolved model id.
fn set_json_model(body: &mut Value, model: &str) {
    if let Value::Object(object) = body {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
}

const CODEX_NAMESPACE_SEPARATOR: &str = "__";

// Codex extends Responses with namespace containers and namespaced function
// calls. OpenAI-compatible providers expect a flat Responses tool namespace.
fn sanitize_openai_responses_provider_body(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    sanitize_openai_responses_input_for_provider(object.get_mut("input"));
    sanitize_openai_responses_tools_for_provider(object.get_mut("tools"));
    sanitize_openai_responses_tool_choice_for_provider(object.get_mut("tool_choice"));
}

fn sanitize_openai_responses_input_for_provider(input: Option<&mut Value>) {
    let Some(Value::Array(items)) = input else {
        return;
    };
    for item in items {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        if object.get("type").and_then(Value::as_str) == Some("function_call") {
            qualify_responses_function_name(object);
        }
    }
}

fn sanitize_openai_responses_tools_for_provider(tools: Option<&mut Value>) {
    let Some(Value::Array(tools)) = tools else {
        return;
    };
    let mut flat_tools = Vec::with_capacity(tools.len());
    for tool in std::mem::take(tools) {
        push_sanitized_openai_responses_tool(&mut flat_tools, tool);
    }
    *tools = flat_tools;
}

fn push_sanitized_openai_responses_tool(out: &mut Vec<Value>, tool: Value) {
    let Value::Object(mut object) = tool else {
        out.push(tool);
        return;
    };
    if object.get("type").and_then(Value::as_str) == Some("namespace") {
        let namespace = object
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let Some(Value::Array(children)) = object.remove("tools") else {
            out.push(Value::Object(object));
            return;
        };
        for child in children {
            push_sanitized_namespaced_tool(out, &namespace, child);
        }
        return;
    }
    ensure_responses_function_tool_description(&mut object);
    out.push(Value::Object(object));
}

fn push_sanitized_namespaced_tool(out: &mut Vec<Value>, namespace: &str, tool: Value) {
    let Value::Object(mut object) = tool else {
        out.push(tool);
        return;
    };
    if object.get("type").and_then(Value::as_str) == Some("function") {
        qualify_responses_function_name_with_namespace(&mut object, namespace);
        ensure_responses_function_tool_description(&mut object);
    }
    out.push(Value::Object(object));
}

fn sanitize_openai_responses_tool_choice_for_provider(tool_choice: Option<&mut Value>) {
    let Some(Value::Object(object)) = tool_choice else {
        return;
    };
    if object.get("type").and_then(Value::as_str) == Some("function") {
        qualify_responses_function_name(object);
    }
}

fn qualify_responses_function_name(object: &mut Map<String, Value>) {
    let namespace = object
        .remove("namespace")
        .and_then(|value| value.as_str().map(ToOwned::to_owned));
    let Some(namespace) = namespace.as_deref() else {
        return;
    };
    qualify_responses_function_name_with_namespace(object, namespace);
}

fn qualify_responses_function_name_with_namespace(
    object: &mut Map<String, Value>,
    namespace: &str,
) {
    if namespace.is_empty() {
        return;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return;
    };
    let prefix = format!("{namespace}{CODEX_NAMESPACE_SEPARATOR}");
    if name.starts_with(&prefix) {
        return;
    }
    object.insert("name".to_string(), Value::String(format!("{prefix}{name}")));
}

fn ensure_responses_function_tool_description(object: &mut Map<String, Value>) {
    if object.get("type").and_then(Value::as_str) == Some("function")
        && !matches!(object.get("description"), Some(Value::String(_)))
    {
        object.insert("description".to_string(), Value::String(String::new()));
    }
}

// Drops fields accepted by OpenAI-like APIs but rejected by Anthropic Messages.
//
// A router can serve earlier turns of a session from an OpenAI-format target and
// later turns from an Anthropic one. Clients such as Claude Code send
// `context_management` on every turn, so the Anthropic leg must strip it or the
// upstream rejects the request (for example `clear_thinking_20251015` strategy
// requires `thinking` to be enabled or adaptive).
fn strip_anthropic_incompatible_fields(body: &mut Value) {
    if let Value::Object(object) = body {
        object.remove("reasoning_effort");
        object.remove("context_management");
    }
}

// Removes replayed `thinking` blocks that carry no signature.
//
// Anthropic requires signed thinking blocks on replay. A router can serve earlier
// turns of a session from an OpenAI-format target whose thinking blocks are
// unsigned, so the Anthropic leg must drop them or the upstream rejects the
// request. Bedrock enforces this (surfacing as a SigV4 signature mismatch) where
// Azure-hosted Anthropic currently does not.
fn strip_unsigned_thinking_blocks(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    let Some(Value::Array(messages)) = object.get_mut("messages") else {
        return;
    };
    for message in messages {
        strip_unsigned_thinking_from_message(message);
    }
}

// Drops unsigned thinking blocks from one message, collapsing content that ends
// up empty to an empty string so the message stays valid.
fn strip_unsigned_thinking_from_message(message: &mut Value) {
    let Value::Object(message) = message else {
        return;
    };
    let Some(Value::Array(blocks)) = message.get("content") else {
        return;
    };
    if !blocks.iter().any(is_unsigned_thinking_block) {
        return;
    }
    let Some(Value::Array(blocks)) = message.get_mut("content") else {
        return;
    };
    blocks.retain(|block| !is_unsigned_thinking_block(block));
    if blocks.is_empty() {
        message.insert("content".to_string(), Value::String(String::new()));
    }
}

// A thinking block is unsigned when `signature` is absent or empty.
fn is_unsigned_thinking_block(block: &Value) -> bool {
    if block.get("type").and_then(Value::as_str) != Some("thinking") {
        return false;
    }
    !matches!(
        block.get("signature").and_then(Value::as_str),
        Some(signature) if !signature.is_empty()
    )
}

// Forces the target's configured reasoning effort onto the outbound body, replacing the
// caller's value. Unlike `extra_body`, this is an override: a route that sends one model at a
// higher effort than the client asked for is the point of the setting.
fn apply_reasoning_effort(body: &mut Value, backend: &Backend) {
    let Some(effort) = backend.reasoning_effort() else {
        return;
    };
    let Value::Object(object) = body else {
        return;
    };
    match backend {
        Backend::OpenAiResponses(_) => {
            // Responses nests effort under `reasoning` next to fields the caller may have set
            // (`summary`, for example), so only the `effort` key is replaced. A `reasoning`
            // value that is not an object is malformed and is replaced whole.
            let reasoning = object
                .entry("reasoning".to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            if !reasoning.is_object() {
                *reasoning = Value::Object(serde_json::Map::new());
            }
            if let Value::Object(reasoning) = reasoning {
                reasoning.insert("effort".to_string(), Value::String(effort.to_string()));
            }
        }
        Backend::OpenAiChat(_) => {
            // Chat Completions takes effort as a top-level field.
            object.insert(
                "reasoning_effort".to_string(),
                Value::String(effort.to_string()),
            );
        }
        // Anthropic has no effort field (thinking is a token budget); the runner rejects the
        // setting on Anthropic clients at load time, so this arm is unreachable in practice.
        Backend::Anthropic(_) => {}
    }
}

/// Forces the backend's authoritative reasoning policy onto the outbound body.
///
/// Unlike `extra_body`, this is an override: a strict non-reasoning target must not be
/// re-enabled by a caller that sent its own `reasoning` field. Provider-shaped keys are
/// merged one level deep so unrelated caller fields (`reasoning.summary`, say) survive.
fn apply_reasoning_policy(body: &mut Value, backend: &Backend) {
    let Some(policy) = backend.reasoning_policy() else {
        return;
    };
    if !policy.is_specified() {
        return;
    }
    let Some(fragment) = policy.wire_body(backend.reasoning_dialect()) else {
        return;
    };
    let (Value::Object(fragment), Value::Object(object)) = (&fragment, body) else {
        return;
    };
    for (key, value) in fragment {
        match value {
            Value::Object(fields) => {
                let existing = object
                    .entry(key.clone())
                    .or_insert_with(|| Value::Object(Default::default()));
                if !existing.is_object() {
                    *existing = Value::Object(Default::default());
                }
                if let Value::Object(existing) = existing {
                    for (name, field) in fields {
                        existing.insert(name.clone(), field.clone());
                    }
                }
            }
            _ => {
                object.insert(key.clone(), value.clone());
            }
        }
    }
}

// Removes a reasoning-effort value the provider is known to refuse.
//
// The effort rides under two spellings depending on the wire, so both are inspected: a Chat
// body carries a flat `reasoning_effort`, a Responses body nests it under `reasoning.effort`.
// Only the effort key is touched, so an unrelated sibling (`reasoning.summary`, for example)
// survives; an emptied `reasoning` object is removed rather than left as `{}`.
fn drop_unsupported_reasoning_effort(body: &mut Value, unsupported: &BTreeSet<String>) {
    if unsupported.is_empty() {
        return;
    }
    let Value::Object(object) = body else {
        return;
    };
    if let Some(Value::String(effort)) = object.get("reasoning_effort")
        && unsupported.contains(effort)
    {
        object.remove("reasoning_effort");
    }
    let emptied = if let Some(Value::Object(reasoning)) = object.get_mut("reasoning") {
        if let Some(Value::String(effort)) = reasoning.get("effort")
            && unsupported.contains(effort)
        {
            reasoning.remove("effort");
        }
        reasoning.is_empty()
    } else {
        false
    };
    if emptied {
        object.remove("reasoning");
    }
}

// Applies target defaults without overriding fields supplied by the caller.
fn merge_extra_body(body: &mut Value, extra_body: &BTreeMap<String, Value>) {
    let Value::Object(object) = body else {
        return;
    };
    for (key, value) in extra_body {
        object.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

fn omit_configured_body_fields(body: &mut Value, omit_body_fields: &BTreeSet<String>) {
    let Value::Object(object) = body else {
        return;
    };
    for key in omit_body_fields {
        object.remove(key);
    }
}

/// The provider's correlation id, if it sent one in a response header.
///
/// Reads the two spellings providers actually use: `x-request-id`, and Anthropic's
/// unprefixed `request-id`. A header value is attacker-adjacent input - it arrives
/// from the network and is about to be written into a JSON document - so it is
/// filtered to identifier characters and length-capped. A value that does not look
/// like an identifier after filtering is dropped rather than repaired.
fn upstream_request_id(headers: &HeaderMap) -> Option<String> {
    ["x-request-id", "request-id"]
        .iter()
        .find_map(|name| correlation::validated_request_id(headers.get(*name)?.to_str().ok()?))
}

/// Folds a header-borne request id into an error body, where the field-borne one lives.
///
/// WHY THE BODY AND NOT A NEW ERROR FIELD
///
/// Only the HTTP client has the response headers; every other constructor of an
/// upstream error has no header to offer, so a new field on the error variant would
/// be filled with `None` in twenty-odd places to serve one. The body already has a
/// defined location for this value - `error.request_id` for the OpenAI shape, a
/// top-level `request_id` for Anthropic's - and one extractor already reads both. So
/// the id is moved to where the schema puts it rather than given a parallel home.
///
/// WHAT THIS DOES NOT DO
///
/// * It never invents an id: without a header the body is returned unchanged, byte
///   for byte.
/// * It never overrides the provider: a body that already carries an id keeps it.
/// * It never restructures: a non-JSON body is left alone rather than wrapped, and a
///   body whose `error` is not an object is left alone.
fn with_request_id(body: String, request_id: Option<&str>) -> String {
    let Some(request_id) = request_id else {
        return body;
    };
    let Ok(mut parsed) = serde_json::from_str::<Value>(&body) else {
        return body;
    };
    // A provider-authored id is authoritative wherever it already sits, including at
    // the top level (Anthropic's shape). Injecting beside it would put two different
    // ids in one document, and the extractor prefers the nested one - so the header
    // would silently override what the provider actually said.
    if provider_named_a_request(&parsed) {
        return body;
    }
    let Some(error) = parsed.get_mut("error").and_then(Value::as_object_mut) else {
        return body;
    };
    error.insert(
        "request_id".to_string(),
        Value::String(request_id.to_string()),
    );
    parsed.to_string()
}

/// Whether the document already carries a provider-authored request id.
///
/// Delegates to [`correlation::names_a_request`], the same function the server uses
/// to report the id. Two copies of "where a provider id can live" would drift, and
/// did: the writer once checked only structured fields, so a body whose only id was
/// in the provider's message prose was treated as unnamed and had a header id
/// injected over it. The reader then preferred the structured field and reported the
/// injected value instead of the provider's.
fn provider_named_a_request(document: &Value) -> bool {
    correlation::names_a_request(document)
}

// Anthropic and Bedrock both cap a request at four blocks carrying
// `cache_control`, counting tools, system blocks and message blocks together.
const MAX_CACHE_CONTROL_BLOCKS: usize = 4;

// Counts the blocks upstream will see carrying a `cache_control` marker. The
// marker's own value holds no such key, so it is never counted twice.
fn count_cache_control_blocks(value: &Value) -> usize {
    match value {
        Value::Object(map) => {
            usize::from(map.contains_key("cache_control"))
                + map.values().map(count_cache_control_blocks).sum::<usize>()
        }
        Value::Array(items) => items.iter().map(count_cache_control_blocks).sum(),
        _ => 0,
    }
}

// Marks the final message content block as the Anthropic prompt-cache breakpoint.
fn enable_anthropic_prompt_caching(body: &mut Value) {
    // Top-level cache control manages the final breakpoint and its TTL.
    if body.get("cache_control").is_some() {
        return;
    }
    // Abstain once the caller has spent the budget itself. Adding a fifth marker
    // turns a request that was valid on arrival into an upstream HTTP 400, and a
    // caller that placed four breakpoints deliberately needs them more than we
    // need a fifth. When the final block is already marked this returns early
    // and changes nothing, which is what the insert below would have done.
    if count_cache_control_blocks(body) >= MAX_CACHE_CONTROL_BLOCKS {
        return;
    }
    let Some(content) = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| messages.last_mut())
        .and_then(|message| message.get_mut("content"))
    else {
        return;
    };
    match content {
        Value::String(text) => {
            *content = serde_json::json!([{
                "type": "text",
                "text": std::mem::take(text),
                "cache_control": {"type": "ephemeral"}
            }]);
        }
        Value::Array(blocks) => {
            if let Some(block) = blocks.last_mut().and_then(Value::as_object_mut) {
                block
                    .entry("cache_control".to_string())
                    .or_insert_with(|| serde_json::json!({"type": "ephemeral"}));
            }
        }
        _ => {}
    }
}

// Requests streamed Chat usage by default while preserving an explicit caller choice.
fn ensure_openai_stream_usage(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    if object.get("stream").and_then(Value::as_bool) != Some(true) {
        return;
    }

    match object.get_mut("stream_options") {
        Some(Value::Object(options)) => {
            options
                .entry("include_usage".to_string())
                .or_insert(Value::Bool(true));
        }
        _ => {
            let mut options = Map::new();
            options.insert("include_usage".to_string(), Value::Bool(true));
            object.insert("stream_options".to_string(), Value::Object(options));
        }
    }
}

fn is_allowed_metadata_header(name: &str) -> bool {
    ALLOWED_METADATA_HEADERS
        .iter()
        .any(|allowed| name.eq_ignore_ascii_case(allowed))
}

fn is_non_forwardable_header(name: &str, headers: &HeaderMap) -> bool {
    NON_FORWARDABLE_HEADERS
        .iter()
        .any(|blocked| name.eq_ignore_ascii_case(blocked))
        || headers.get_all("connection").iter().any(|value| {
            value
                .as_bytes()
                .split(|byte| *byte == b',')
                .any(|option| option.trim_ascii().eq_ignore_ascii_case(name.as_bytes()))
        })
}

// Unknown application headers can carry credentials when auth forwarding is enabled.
fn redact_forwarded_headers(
    mut body: String,
    metadata: Option<&Metadata>,
    is_forwarding_auth: bool,
) -> String {
    if !is_forwarding_auth {
        return body;
    }
    let Some(headers) = metadata.and_then(|metadata| metadata.http_headers.as_ref()) else {
        return body;
    };
    for (name, value) in headers {
        if is_non_forwardable_header(name.as_str(), headers) {
            continue;
        }
        let Ok(value) = value.to_str() else {
            continue;
        };
        if !value.is_empty() {
            body = body.replace(value, "[REDACTED]");
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, LlmResponseChunk, Message, PreservationMetadata, ProviderExtensions,
        ResponseOutput, Usage, completion_text, text_request,
    };
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::backend::HttpBackendConfig;
    use switchyard_protocol::{ReasoningDialect, ReasoningPolicy};

    /// Google's two 429s must not be collapsed: one clears, the other waits for the day.
    #[test]
    fn a_google_daily_quota_is_not_retried_but_its_window_throttle_is() {
        let error = |status: u16, body: &str| LlmClientError::UpstreamHttp {
            status: StatusCode::from_u16(status).expect("status"),
            body: body.to_string(),
        };

        let throttle = error(
            429,
            r#"{"error":{"code":"rate_limit_exceeded","message":"slow"}}"#,
        );
        assert!(is_transient_failure(&throttle));
        assert!(!is_account_wide_quota(&throttle));

        // An unscoped daily quota retires the model that reported it. It is NOT account
        // wide: this project allocates RPD per model, so its siblings still have budget.
        let quota = error(
            429,
            r#"{"error":{"code":"quota_exceeded","message":"daily"}}"#,
        );
        assert!(!is_transient_failure(&quota));
        assert!(!is_account_wide_quota(&quota));

        // Only a scope the body positively identifies as project-wide stops the pool.
        let project_wide = error(
            429,
            r#"{"error":{"code":"quota_exceeded","message":"daily","details":[
                {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[
                  {"quotaId":"GenerateRequestsPerDayPerProject-FreeTier"}]}]}}"#,
        );
        assert!(!is_transient_failure(&project_wide));
        assert!(is_account_wide_quota(&project_wide));

        // A daily quota is still a daily quota when it arrives behind a 5xx: the code, not
        // the status, says the allowance is spent, so no retry may follow it.
        let quota_behind_5xx = error(
            503,
            r#"{"error":{"code":"quota_exceeded","message":"daily quota"}}"#,
        );
        assert!(!is_transient_failure(&quota_behind_5xx));
        assert!(!is_account_wide_quota(&quota_behind_5xx));

        // A content block is a 400 that must not be resent, and it is not a quota.
        let blocked = error(400, r#"{"error":{"code":"safety","message":"blocked"}}"#);
        assert!(!is_transient_failure(&blocked));
        assert!(!is_account_wide_quota(&blocked));

        // The other providers' contracts are untouched by the Google decoder.
        let openrouter = error(
            429,
            r#"{"error":{"code":402,"message":"credits","metadata":{"limit_source":"openrouter_free_tier_daily"}}}"#,
        );
        assert!(!is_transient_failure(&openrouter));
        assert!(is_account_wide_quota(&openrouter));
    }

    fn config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: Some("secret".to_string()),
            forward_auth: false,
            extra_headers: BTreeMap::new(),
            extra_body: BTreeMap::new(),
            omit_body_fields: BTreeSet::new(),
            unsupported_reasoning_effort: BTreeSet::new(),
            reasoning_effort: None,
            reasoning_policy: None,
            reasoning_dialect: ReasoningDialect::default(),
            max_retries: 0,
            failure_cooldown: Duration::ZERO,
            cooldown_group: None,
            timeout: None,
            attempt_timeout: None,
            stream_idle_timeout: None,
            requires_list_input: false,
            allowance: None,
            pacing: None,
            hosted_reasoning: None,
        }
    }

    #[test]
    fn provider_endpoint_urls_preserve_query_and_fragment() {
        let invalid_url = "not a URL/";
        let backend = Backend::OpenAiResponses(config(invalid_url));
        assert!(openai_url(invalid_url, "/responses").is_err());
        assert_eq!(backend.url(), invalid_url);
        assert_eq!(
            AuxiliaryOperation::ResponsesCompact.url(&backend),
            invalid_url
        );

        let anthropic = Backend::Anthropic(config(invalid_url));
        assert_eq!(anthropic.url(), invalid_url);
        assert_eq!(anthropic.count_tokens_url(), invalid_url);

        for path in ["", "/v1/", "/v1/messages"] {
            for tail in [
                "",
                "?api-version=2026-09-01&value=%2F/",
                "#fragment?value=/",
            ] {
                let base_url = format!("https://example.com{path}{tail}");
                let backend = Backend::Anthropic(config(&base_url));
                assert_eq!(
                    backend.url(),
                    format!("https://example.com/v1/messages{tail}")
                );
                assert_eq!(
                    AuxiliaryOperation::AnthropicCountTokens.url(&backend),
                    format!("https://example.com/v1/messages/count_tokens{tail}")
                );
            }
        }

        for path in ["/v1", "/v1/responses/", "/v1/chat/completions"] {
            for tail in [
                "",
                "?api-version=2026-09-01&value=%2F/",
                "#fragment?value=/",
            ] {
                let base_url = format!("https://example.com{path}{tail}");
                let responses = Backend::OpenAiResponses(config(&base_url));
                let chat = Backend::OpenAiChat(config(&base_url));
                for (actual, endpoint) in [
                    (responses.url(), "/responses"),
                    (chat.url(), "/chat/completions"),
                    (
                        AuxiliaryOperation::ResponsesCompact.url(&responses),
                        "/responses/compact",
                    ),
                    (
                        AuxiliaryOperation::ResponsesInputTokens.url(&responses),
                        "/responses/input_tokens",
                    ),
                ] {
                    assert_eq!(actual, format!("https://example.com/v1{endpoint}{tail}"));
                }
            }
        }
    }

    fn config_with_retries(base_url: &str, max_retries: u32) -> HttpBackendConfig {
        HttpBackendConfig {
            max_retries,
            ..config(base_url)
        }
    }

    fn forwarding_config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            api_key: None,
            forward_auth: true,
            ..config(base_url)
        }
    }

    // A one-model config list: "gpt" served over OpenAI Chat at base_url.
    fn chat_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(config(base_url)),
            None,
        )]
    }

    fn responses_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(config(base_url)),
            None,
        )]
    }

    // A chat backend whose upstream writes its reasoning inline behind a marker.
    fn chat_map_with_hosted_reasoning(base_url: &str) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.hosted_reasoning = Some(switchyard_translation::HostedReasoningMarker::ThoughtTag);
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn chat_map_with_extra_body(
        base_url: &str,
        extra_body: BTreeMap<String, Value>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.extra_body = extra_body;
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn chat_map_with_omit_body_fields(
        base_url: &str,
        omit_body_fields: BTreeSet<String>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.omit_body_fields = omit_body_fields;
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn chat_map_with_unsupported_effort(
        base_url: &str,
        unsupported: BTreeSet<String>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.unsupported_reasoning_effort = unsupported;
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn responses_map_with_unsupported_effort(
        base_url: &str,
        unsupported: BTreeSet<String>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.unsupported_reasoning_effort = unsupported;
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(backend),
            None,
        )]
    }

    // Opaque reasoning state follows the dialect, not the format: only the OpenRouter wire may
    // be given back `reasoning_details`, and even then only state this exact target was proven
    // to have issued. A native target is narrowed so it can never receive another provider's
    // signed blob, and the default (ProvenanceBound) is never what a real target resolves to by
    // accident.
    #[test]
    fn replay_policy_follows_the_target_dialect() {
        let mut base = config("http://stub");
        base.reasoning_dialect = ReasoningDialect::OpenRouterEnabled;
        assert_eq!(
            translation_policy(&Backend::OpenAiChat(base.clone())).reasoning_details_replay,
            switchyard_translation::ReasoningDetailsReplay::ProvenanceBound,
            "the issuing wire may replay only state it can prove it issued"
        );
        for dialect in [
            ReasoningDialect::OpenAiEffort,
            ReasoningDialect::LlamaCppEnableThinking,
            ReasoningDialect::Unspecified,
        ] {
            let mut narrowed = base.clone();
            narrowed.reasoning_dialect = dialect;
            assert_eq!(
                translation_policy(&Backend::OpenAiChat(narrowed)).reasoning_details_replay,
                switchyard_translation::ReasoningDetailsReplay::PlaintextOnly,
                "{dialect:?} must not replay opaque provider state"
            );
        }
    }

    fn chat_map_with_effort(base_url: &str, effort: &str) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.reasoning_effort = Some(effort.to_string());
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn responses_map_with_effort(base_url: &str, effort: &str) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.reasoning_effort = Some(effort.to_string());
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(backend),
            None,
        )]
    }

    /// A Responses target on the llama.cpp dialect, which gates reasoning through
    /// `chat_template_kwargs.enable_thinking` rather than an effort axis.
    fn responses_map_with_local_policy(
        base_url: &str,
        policy: ReasoningPolicy,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.reasoning_dialect = ReasoningDialect::LlamaCppEnableThinking;
        backend.reasoning_policy = Some(policy);
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(backend),
            None,
        )]
    }

    fn responses_map_with_policy(base_url: &str, policy: ReasoningPolicy) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        // An `extra_body` default that the caller will try to override.
        backend.extra_body = BTreeMap::from([("reasoning".to_string(), json!({"enabled": true}))]);
        backend.reasoning_dialect = ReasoningDialect::OpenRouterEnabled;
        backend.reasoning_policy = Some(policy);
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(backend),
            None,
        )]
    }

    fn anthropic_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "claude",
            Backend::Anthropic(config(base_url)),
            None,
        )]
    }

    fn chat_map_with_retries(base_url: &str, max_retries: u32) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(config_with_retries(base_url, max_retries)),
            None,
        )]
    }

    fn chat_success_response() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "model": "gpt",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop"
            }],
            "usage": {}
        }))
    }

    fn truncated_response_server(
        content_type: &str,
        body: &str,
    ) -> std::io::Result<(String, JoinHandle<std::io::Result<()>>)> {
        response_sequence_server(vec![format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len() + 100
        )])
    }

    fn response_sequence_server(
        responses: Vec<String>,
    ) -> std::io::Result<(String, JoinHandle<std::io::Result<()>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let handle = std::thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 1024];
                if stream.read(&mut request)? == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "client closed before sending a request",
                    ));
                }
                stream.write_all(response.as_bytes())?;
            }
            Ok(())
        });
        Ok((format!("http://{address}/v1"), handle))
    }

    fn raw_chat_success_response() -> String {
        let body = r#"{"id":"chatcmpl-1","model":"gpt","choices":[{"index":0,"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}],"usage":{}}"#;
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn request_for(model: Option<&str>, stream: bool) -> Request {
        let mut llm_request = text_request(model.map(str::to_string), "hi");
        llm_request.stream = stream;
        Request {
            llm_request,
            raw_request: None,
            metadata: None,
            route_reasoning_policy: None,
            route_reasoning_effort: None,
        }
    }

    fn request_with_headers(model: &str, headers: HeaderMap) -> Request {
        let mut request = request_for(Some(model), false);
        request.metadata = Some(Metadata {
            http_headers: Some(headers),
            ..Default::default()
        });
        request
    }

    #[tokio::test]
    async fn transport_errors_drop_the_upstream_url() {
        let error = reqwest::Client::new()
            .post("http://127.0.0.1:1/v1?key=CANARY")
            .send()
            .await
            .expect_err("closed port");

        assert!(!convert_reqwest_error(error).to_string().contains("CANARY"));
    }

    #[test]
    fn anthropic_prompt_caching_marks_final_message() {
        let caller_managed = json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "messages": [{"role": "user", "content": "hello"}]
        });
        let mut body = caller_managed.clone();
        enable_anthropic_prompt_caching(&mut body);
        assert_eq!(body, caller_managed);

        let mut body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });

        enable_anthropic_prompt_caching(&mut body);

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    // Counts the blocks upstream will see carrying a `cache_control` marker.
    fn cache_control_blocks(value: &Value) -> usize {
        count_cache_control_blocks(value)
    }

    // A body whose four breakpoints are all spent elsewhere: the caller manages
    // its own caching and left the final block unmarked on purpose.
    fn body_at_the_cache_control_limit() -> Value {
        json!({
            "system": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [
                {"name": "t", "input_schema": {"type": "object"},
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "old", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": "new"}]}
            ]
        })
    }

    #[test]
    fn anthropic_prompt_caching_abstains_at_the_cache_control_limit() {
        let mut body = body_at_the_cache_control_limit();
        assert_eq!(cache_control_blocks(&body), 4);

        enable_anthropic_prompt_caching(&mut body);

        // Anthropic and Bedrock both reject a fifth marker with HTTP 400, which
        // would fail a request that was valid before it reached us.
        assert_eq!(cache_control_blocks(&body), 4);
        assert!(
            body["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn anthropic_prompt_caching_still_marks_one_below_the_limit() {
        let mut body = body_at_the_cache_control_limit();
        // Free one breakpoint, so there is room for ours.
        body["system"].as_array_mut().unwrap().pop();
        assert_eq!(cache_control_blocks(&body), 3);

        enable_anthropic_prompt_caching(&mut body);

        assert_eq!(cache_control_blocks(&body), 4);
        assert_eq!(
            body["messages"][2]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    // A request that pins `format` in its metadata, so the client resolves that
    // wire format instead of the model's default backend.
    fn request_with_wire_format(model: &str, format: WireFormat) -> Request {
        let mut request = request_for(Some(model), false);
        request.metadata = Some(Metadata {
            session_id: None,
            agent_id: None,
            task_id: None,
            correlation_id: None,
            extra_metadata: None,
            http_headers: None,
            wire_format: Some(format),
            ..Default::default()
        });
        request
    }

    #[tokio::test]
    async fn missing_model_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        let Err(error) = client
            .call_rewrite_model(request_for(None, false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::InvalidRequest { message } if message == "no model given"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn unknown_model_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::Configuration { message }
                if message.contains("gpt")
        ));
        Ok(())
    }

    #[tokio::test]
    async fn unknown_model_format_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // "gpt" exists but only over OpenAI Chat; the request pins Anthropic.
        let client = TranslatingLlmClient::new(&chat_map("https://example.test/v1"))?;
        let Err(error) = client
            .call_rewrite_model(
                request_with_wire_format("gpt", WireFormat::AnthropicMessages),
                None,
            )
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::Configuration { message }
                if message.contains("gpt")
                    && message.contains(&WireFormat::AnthropicMessages.to_string())
        ));
        Ok(())
    }

    #[test]
    fn backend_for_resolves_configured_format()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&chat_map("https://example.test/v1"))?;
        // "gpt" is served over OpenAI Chat only; other formats and models miss.
        assert!(
            client
                .backend_for(&ModelId::from("gpt"), WireFormat::OpenAiChat)
                .is_some()
        );
        assert!(
            client
                .backend_for(&ModelId::from("gpt"), WireFormat::AnthropicMessages)
                .is_none()
        );
        assert!(
            client
                .backend_for(&ModelId::from("missing"), WireFormat::OpenAiChat)
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn model_name_arg_wins_over_request_model()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        // Arg "b" is looked up (and reported), not the request's "a".
        let Err(error) = client
            .call_rewrite_model(request_for(Some("a"), false), Some(&ModelId::from("b")))
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::Configuration { message }
                if message.contains("\"b\"")
        ));
        Ok(())
    }

    #[tokio::test]
    async fn buffered_openai_chat_round_trips()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let response = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hi there");

        Ok(())
    }

    #[tokio::test]
    async fn a_responses_only_feature_is_refused_before_egress()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // A caller posts /responses with a hosted search tool at a Chat target. The decoder
        // drops the tool, so forwarding the rest would answer from memory and present that as
        // the requested search. Nothing may reach the provider instead.
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                observed.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_json(json!({
                    "id": "answer",
                    "model": "gemini",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "no search happened"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }))
            })
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gemini",
            Backend::OpenAiChat(config(&format!("{}/v1", server.uri()))),
            None,
        )])?;
        let body = json!({
            "model": "gemini",
            "input": "what closed higher yesterday?",
            "tools": [{"type": "web_search"}]
        });

        let refused = client
            .call_rewrite_model_raw(
                body,
                None,
                Some(&ModelId::from("gemini".to_string())),
                WireFormat::OpenAiResponses,
            )
            .await;
        let Err(error) = refused else {
            panic!("a hosted tool at a Chat target must not be served");
        };
        let message = error.to_string();
        assert!(
            message.contains("web_search"),
            "the refusal must name the feature: {message}"
        );
        assert!(
            message.contains("openai_responses"),
            "and the way to serve it: {message}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the refusal must happen before any upstream request"
        );

        // Control: the same request without the hosted tool is served, so the guard refuses
        // the feature rather than the endpoint.
        let served = client
            .call_rewrite_model_raw(
                json!({"model": "gemini", "input": "hello"}),
                None,
                Some(&ModelId::from("gemini".to_string())),
                WireFormat::OpenAiResponses,
            )
            .await?;
        assert!(matches!(served, RawResponse::Buffered(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        Ok(())
    }

    #[tokio::test]
    async fn an_overloaded_google_model_rests_without_being_replayed()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // The measured shape Google sent for the 31B while it was starting: a list-wrapped
        // 503 whose own status word is UNAVAILABLE. It must be neither replayed nor left
        // available for the next caller, and its sibling must be untouched by either fact.
        let server = MockServer::start().await;
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let observed = Arc::clone(&calls);
        const UNAVAILABLE: &str = r#"[{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}]"#;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                let model = body["model"].as_str().unwrap_or_default().to_string();
                observed.lock().expect("call log").push(model.clone());
                if model == "large" {
                    ResponseTemplate::new(503).set_body_string(UNAVAILABLE)
                } else {
                    ResponseTemplate::new(200).set_body_json(json!({
                        "id": "answer",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    }))
                }
            })
            .mount(&server)
            .await;

        // The 31B's production configuration: a five-minute cooldown, no group, and a retry
        // budget that would be spent if the condition did not forbid a replay.
        let large = {
            let mut backend = config(&format!("{}/v1", server.uri()));
            backend.failure_cooldown = Duration::from_secs(300);
            backend.max_retries = 2;
            Backend::OpenAiChat(backend)
        };
        let small = {
            let mut backend = config(&format!("{}/v1", server.uri()));
            backend.failure_cooldown = Duration::from_secs(5);
            backend.max_retries = 2;
            Backend::OpenAiChat(backend)
        };
        let client = TranslatingLlmClient::new(&[
            ModelConfig::new("large", large, None),
            ModelConfig::new("small", small, None),
        ])?;
        let name = |model: &str| ModelId::from(model.to_string());

        // First call: the overloaded model answers 503 once and is not replayed, even though
        // three attempts were authorised.
        let first = client
            .call_rewrite_model(request_for(Some("large"), false), Some(&name("large")))
            .await;
        assert!(
            matches!(
                first,
                Err(LlmClientError::UpstreamHttp {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    ..
                })
            ),
            "the caller sees the provider's own 503"
        );
        assert_eq!(
            std::mem::take(&mut *calls.lock().expect("call log")),
            ["large"],
            "an UNAVAILABLE model must not be asked again inside the call"
        );

        // Second call: the cooldown is armed even though the failure was not retryable, so no
        // request reaches the provider at all.
        let second = client
            .call_rewrite_model(request_for(Some("large"), false), Some(&name("large")))
            .await;
        assert!(
            matches!(second, Err(LlmClientError::TemporarilyUnavailable)),
            "a resting model answers without a provider call"
        );
        assert!(
            calls.lock().expect("call log").is_empty(),
            "the cooldown must arm on an unretryable capacity failure"
        );

        // The sibling is unaffected: the capacity fact was the large model's, and it was not
        // declared as a shared pool.
        let third = client
            .call_rewrite_model(request_for(Some("small"), false), Some(&name("small")))
            .await?;
        assert_eq!(
            third
                .llm_response
                .into_agg()
                .await
                .map(|agg| completion_text(&agg))?,
            "ok"
        );
        assert_eq!(
            std::mem::take(&mut *calls.lock().expect("call log")),
            ["small"]
        );

        Ok(())
    }

    #[test]
    fn only_a_model_that_needs_resting_is_rested() {
        let call = |status: u16, body: &str| LlmClientError::UpstreamHttp {
            status: StatusCode::from_u16(status).expect("status"),
            body: body.to_string(),
        };
        // Capacity: rest, do not replay.
        let overloaded = call(
            503,
            r#"[{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}]"#,
        );
        assert!(rests_the_model(&overloaded));
        assert!(!is_transient_failure(&overloaded));
        // A request fault says nothing about the model's health.
        let blocked = call(400, r#"{"error":{"code":"safety","message":"blocked"}}"#);
        assert!(!rests_the_model(&blocked));
        // A credential fault is the operator's, and resting the model would hide it.
        let denied = call(
            403,
            r#"{"error":{"code":"permission_denied","message":"no"}}"#,
        );
        assert!(!rests_the_model(&denied));
        // An ordinary transport failure is retryable and therefore rests, as before.
        let transport = LlmClientError::Transport {
            source: Box::new(std::io::Error::other("reset")),
        };
        assert!(rests_the_model(&transport));
    }

    #[tokio::test]
    async fn buffered_hosted_reasoning_never_reaches_the_caller()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // The upstream writes its chain of thought into `content`. A backend that declares
        // the marker must hand the caller only the answer.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gemma",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "<thought>plan</thought>\n\nready"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 71}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_hosted_reasoning(&format!(
            "{}/v1",
            server.uri()
        )))?;
        let response = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;

        assert_eq!(completion_text(&agg), "ready");
        // Not merely out of the answer: the monologue must not survive in ANY block, because a
        // reasoning block is re-encoded as `reasoning_content` and would reach the caller
        // through that second field.
        let output = agg.first_output().expect("one output item");
        assert!(
            !output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Reasoning { .. })),
            "a declared hosted marker must not leave a reasoning block: {:?}",
            output.content
        );
        assert!(
            !format!("{:?}", output.content).contains("plan"),
            "the thought must not survive in any block: {:?}",
            output.content
        );
        Ok(())
    }

    #[tokio::test]
    async fn streamed_hosted_reasoning_never_reaches_the_caller()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"<thou\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"ght>plan</thought>\\n\\nready\"}}]}\n\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_raw(sse, "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_hosted_reasoning(&format!(
            "{}/v1",
            server.uri()
        )))?;
        let response = client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;

        assert_eq!(completion_text(&agg), "ready");
        assert!(
            !format!("{:?}", agg.first_output().map(|o| &o.content)).contains("plan"),
            "the streamed thought must not survive in any block"
        );
        Ok(())
    }

    #[tokio::test]
    async fn invalid_json_is_a_response_translation_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_raw("not json", "application/json")
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 2))?;
        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected invalid JSON to fail");
        };

        assert!(matches!(
            error,
            LlmClientError::ResponseTranslation(message)
                if message.contains("invalid upstream JSON")
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn response_body_io_failure_is_a_transport_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let (base_url, server) = truncated_response_server("application/json", "{}")?;
        let client = TranslatingLlmClient::new(&chat_map(&base_url))?;
        let result = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await;
        server
            .join()
            .map_err(|_| std::io::Error::other("response server thread panicked"))??;

        let Err(error) = result else {
            panic!("expected the truncated response body to fail");
        };
        let LlmClientError::Transport { source } = error else {
            panic!("expected a transport error");
        };
        let Some(source) = source.downcast_ref::<reqwest::Error>() else {
            panic!("expected the reqwest transport source");
        };
        assert!(source.is_decode());
        assert!(
            !std::error::Error::source(&source)
                .is_some_and(|source| source.is::<serde_json::Error>())
        );
        Ok(())
    }

    #[tokio::test]
    async fn response_body_transport_failures_are_retried()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let truncated_responses = [
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: 102\r\nConnection: close\r\n\r\n{}",
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/plain\r\n\
             Content-Length: 100\r\nConnection: close\r\n\r\nbad",
        ];

        for truncated in truncated_responses {
            let (base_url, server) =
                response_sequence_server(vec![truncated.to_string(), raw_chat_success_response()])?;
            let client = TranslatingLlmClient::new(&chat_map_with_retries(&base_url, 1))?;
            let response = client
                .call_rewrite_model(request_for(Some("gpt"), false), None)
                .await?;
            server
                .join()
                .map_err(|_| std::io::Error::other("response server thread panicked"))??;

            assert_eq!(
                completion_text(&response.llm_response.into_agg().await?),
                "recovered"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn pre_first_event_transport_failure_uses_retry_budget()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let truncated = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                         Content-Length: 100\r\nConnection: close\r\n\r\n";
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"recovered\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let success = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );

        for second in [success, truncated.to_string()] {
            let is_exhausted = second == truncated;
            let (base_url, server) = response_sequence_server(vec![truncated.to_string(), second])?;
            let client = TranslatingLlmClient::new(&chat_map_with_retries(&base_url, 1))?;
            let result = client
                .call_rewrite_model(request_for(Some("gpt"), true), None)
                .await;
            if is_exhausted {
                assert!(matches!(result, Err(LlmClientError::Transport { .. })));
            } else {
                assert_eq!(
                    completion_text(&result?.llm_response.into_agg().await?),
                    "recovered"
                );
            }
            server
                .join()
                .map_err(|_| std::io::Error::other("response server thread panicked"))??;
        }
        Ok(())
    }

    #[tokio::test]
    async fn streaming_body_io_failure_preserves_transport_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let (base_url, server) = truncated_response_server("text/event-stream", body)?;
        let client = TranslatingLlmClient::new(&chat_map_with_retries(&base_url, 2))?;
        let response = client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await?;
        let result = response.llm_response.into_agg().await;
        server
            .join()
            .map_err(|_| std::io::Error::other("response server thread panicked"))??;

        let Err(error) = result else {
            panic!("expected the truncated stream body to fail");
        };

        assert!(matches!(error, LlmClientError::Transport { .. }));
        Ok(())
    }

    #[tokio::test]
    async fn rewrites_model_to_resolved_upstream_id()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // Inbound body says "switchyard"; the upstream must receive "gpt".
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({"model": "gpt"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        // Inbound model differs from the map key / resolved model.
        client
            .call_rewrite_model(
                request_for(Some("switchyard"), false),
                Some(&ModelId::from("gpt")),
            )
            .await?;
        // The body_partial_json matcher asserts the upstream saw model "gpt".
        Ok(())
    }

    /// Measured 2026-11-21 against hosted Gemma 4 26B through `smart-nt`: an output budget
    /// went entirely to inline `<thought>` reasoning, so the reply was `completion_tokens: 0`
    /// with no content and `finish_reason: length` while the wire said 200. The route returned
    /// that empty body to the caller even though a sibling candidate could answer. An empty
    /// answer is not a served request: it must advance.
    ///
    /// Replayed against a target that does *not* declare inline reasoning. That is the older
    /// classification and it must not move: without a declaration, a token total above input
    /// plus output is a provider accounting quirk, not evidence of hidden thinking. The
    /// declared-reasoning case is the test that follows.
    #[tokio::test]
    async fn a_reply_that_spent_its_budget_on_reasoning_advances_rather_than_returning_empty()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                "usage": {"prompt_tokens": 225, "completion_tokens": 0, "total_tokens": 1123}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let mut request = request_for(Some("gpt"), false);
        request.llm_request.output.max_output_tokens = Some(900);

        let error = client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await
            .err()
            .expect("an empty answer must not be reported as a successful call");

        let LlmClientError::EmptyAnswer { reason, .. } = &error else {
            panic!("expected EmptyAnswer, got {error:?}");
        };
        assert_eq!(*reason, "empty_answer_zero_completion");
        // The route may advance, but this call must not resend the same budget: the same
        // model produces the same non-answer, so a retry only spends the window again.
        assert!(
            !is_transient_failure(&error),
            "an empty answer must advance, not retry in place"
        );
        Ok(())
    }

    /// The measured Gemma shape, replayed against a target that *does* declare inline
    /// reasoning: `completion_tokens: 0`, no content, and a total that exceeds prompt plus
    /// completion. The difference is a fact the deployment did not have before — the model
    /// produced tokens, they were thinking, and the caller's reservation was too small to
    /// hold both. `empty_answer_zero_completion` would blame the leg; this says what happened.
    #[tokio::test]
    async fn a_thought_only_reply_on_a_hosted_reasoning_target_is_named_rather_than_blamed()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                // 438 prompt + 0 counted completion + 61 of thinking = 499, the measured
                // 64-token-reservation row.
                "usage": {"prompt_tokens": 438, "completion_tokens": 0, "total_tokens": 499}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_hosted_reasoning(&format!(
            "{}/v1",
            server.uri()
        )))?;
        let mut request = request_for(Some("gpt"), false);
        request.llm_request.output.max_output_tokens = Some(64);

        let error = client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await
            .err()
            .expect("a reply with no visible answer must not be reported as a success");

        let LlmClientError::EmptyAnswer { reason, .. } = &error else {
            panic!("expected EmptyAnswer, got {error:?}");
        };
        assert_eq!(*reason, "empty_answer_thinking_only");
        // Same advance-not-retry rule as any empty answer: nothing about the next attempt on
        // this budget changes the fact that the thinking alone filled it.
        assert!(!is_transient_failure(&error));
        Ok(())
    }

    /// A hosted-reasoning target that reports no totals has told us nothing about what it
    /// produced. The invisible-token count must come from the provider's own numbers, so this
    /// keeps the quieter classification rather than inventing thinking out of a missing field.
    #[tokio::test]
    async fn a_hosted_reasoning_target_without_totals_keeps_the_quieter_classification()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_hosted_reasoning(&format!(
            "{}/v1",
            server.uri()
        )))?;
        let mut request = request_for(Some("gpt"), false);
        request.llm_request.output.max_output_tokens = Some(64);

        let error = client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await
            .err()
            .expect("a reply with no visible answer must not be reported as a success");

        let LlmClientError::EmptyAnswer { reason, .. } = &error else {
            panic!("expected EmptyAnswer, got {error:?}");
        };
        assert_eq!(*reason, "empty_answer_zero_completion");
        Ok(())
    }

    /// The declaration alone must not be enough: a target that hides reasoning still has to
    /// show spare tokens in its own totals. Here prompt plus completion accounts for the
    /// whole total, so nothing was produced off the books and the reply is simply empty.
    #[tokio::test]
    async fn a_hosted_reasoning_target_whose_totals_account_for_every_token_is_not_thinking()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_hosted_reasoning(&format!(
            "{}/v1",
            server.uri()
        )))?;
        let mut request = request_for(Some("gpt"), false);
        request.llm_request.output.max_output_tokens = Some(64);

        let error = client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await
            .err()
            .expect("a reply with no visible answer must not be reported as a success");

        let LlmClientError::EmptyAnswer { reason, .. } = &error else {
            panic!("expected EmptyAnswer, got {error:?}");
        };
        assert_eq!(*reason, "empty_answer_zero_completion");
        Ok(())
    }

    /// A hosted-reasoning target that does answer is a success, thinking or not. The new
    /// classification must not start refusing replies that carry visible text.
    #[tokio::test]
    async fn a_hosted_reasoning_target_that_answers_is_still_a_success()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<thought>two plus two</thought>four"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 438, "completion_tokens": 32, "total_tokens": 675}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_hosted_reasoning(&format!(
            "{}/v1",
            server.uri()
        )))?;
        let mut request = request_for(Some("gpt"), false);
        request.llm_request.output.max_output_tokens = Some(512);

        let response = client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await?;
        let agg = response.llm_response.into_agg().await?;
        assert!(
            completion_text(&agg).contains("four"),
            "the visible answer must survive: {:?}",
            agg.outputs
        );
        Ok(())
    }

    /// The one caller that legitimately wants no output asks for zero tokens. Refusing that
    /// as a failure would turn a deliberate probe into an error.
    #[tokio::test]
    async fn a_caller_that_asks_for_zero_output_is_not_an_empty_answer()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let mut request = request_for(Some("gpt"), false);
        request.llm_request.output.max_output_tokens = Some(0);

        client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await?;
        Ok(())
    }

    /// A route's explicit reasoning policy outranks the target default AND the caller's
    /// effort, on the real request path rather than a synthetic body.
    ///
    /// This is the guarantee that lets one physical model serve a thinking route and a
    /// strict-NT route at the same time: the target declares thinking enabled, the caller
    /// A route that asks for `High` must reach the wire as a graded effort, not as the
    /// target's weaker default. `openai_effort` has no fragment meaning "think", so the
    /// policy alone would be inert and the request would silently keep the target's level.
    #[tokio::test]
    async fn a_route_effort_overrides_the_target_default_on_the_wire() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(wiremock::matchers::body_partial_json(json!({
                "reasoning": {"effort": "high"}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "model": "gpt",
                "status": "completed",
                "output": [{
                    "type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;
        // The target defaults to medium; the route must still win with high.
        let client = TranslatingLlmClient::new(&responses_map_with_effort(
            &format!("{}/v1", server.uri()),
            "medium",
        ))
        .expect("client");

        let request = Request {
            llm_request: switchyard_protocol::text_request(Some("gpt".into()), "hi"),
            raw_request: None,
            metadata: None,
            route_reasoning_policy: Some(switchyard_protocol::ReasoningPolicy::Enabled),
            route_reasoning_effort: Some("high".to_string()),
        };
        client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await
            .expect("call completes");
    }

    /// A route's explicit reasoning policy outranks the target default AND the caller's
    /// effort, on the real request path rather than a synthetic body.
    #[tokio::test]
    async fn a_route_policy_overrides_both_the_target_default_and_the_caller() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(wiremock::matchers::body_partial_json(json!({
                "chat_template_kwargs": {"enable_thinking": false}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "model": "gpt",
                "status": "completed",
                "output": [{
                    "type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&responses_map_with_local_policy(
            &format!("{}/v1", server.uri()),
            switchyard_protocol::ReasoningPolicy::Enabled,
        ))
        .expect("client");

        // Built through the typed Request so the route policy is carried the way the
        // runner carries it, not by a hand-written body field.
        let request = Request {
            llm_request: switchyard_protocol::text_request(Some("gpt".into()), "hi"),
            raw_request: None,
            metadata: None,
            route_reasoning_policy: Some(switchyard_protocol::ReasoningPolicy::Disabled),
            route_reasoning_effort: None,
        };
        client
            .call_rewrite_model(request, Some(&ModelId::from("gpt")))
            .await
            .expect("call completes");
    }

    /// A configured reasoning effort replaces the caller's value on both OpenAI wire formats,
    /// which `extra_body` (defaults only) cannot do.
    #[tokio::test]
    async fn reasoning_effort_override_replaces_the_callers_effort()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "gpt",
                "reasoning_effort": "max"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })))
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&chat_map_with_effort(
            &format!("{}/v1", server.uri()),
            "max",
        ))?;
        client
            .call_rewrite_model_raw(
                json!({
                    "model": "client-facing",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "high"
                }),
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "gpt",
                "reasoning": {"effort": "max", "summary": "auto"}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "model": "gpt",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&responses_map_with_effort(
            &format!("{}/v1", server.uri()),
            "max",
        ))?;
        client
            .call_rewrite_model_raw(
                json!({
                    "model": "client-facing",
                    "input": [{"role": "user", "content": "hi"}],
                    "reasoning": {"effort": "high", "summary": "auto"}
                }),
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?;
        Ok(())
    }

    /// A target reasoning policy is authoritative: it replaces whatever the caller sent
    /// *and* any conflicting `extra_body` default. `extra_body` alone cannot express this,
    /// because it only fills keys the request omits (`merge_extra_body`), so a caller that
    /// sets `reasoning` would silently re-enable reasoning on a strict-NT target.
    #[tokio::test]
    async fn reasoning_policy_overrides_caller_and_extra_body()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "gpt",
                "reasoning": {"enabled": false}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "model": "gpt",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&responses_map_with_policy(
            &format!("{}/v1", server.uri()),
            ReasoningPolicy::Disabled,
        ))?;
        client
            .call_rewrite_model_raw(
                json!({
                    "model": "client-facing",
                    "input": [{"role": "user", "content": "hi"}],
                    "reasoning": {"enabled": true, "effort": "high"}
                }),
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn extra_body_adds_defaults_without_overriding_the_request()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "gpt",
                "max_tokens": 7,
                "service_tier": "priority"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let extra_body = BTreeMap::from([
            ("max_tokens".to_string(), json!(999)),
            ("service_tier".to_string(), json!("priority")),
        ]);
        let client = TranslatingLlmClient::new(&chat_map_with_extra_body(
            &format!("{}/v1", server.uri()),
            extra_body,
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn omit_body_fields_strips_a_field_extra_body_cannot_override()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                body.get("max_tokens").is_none()
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_omit_body_fields(
            &format!("{}/v1", server.uri()),
            BTreeSet::from(["max_tokens".to_string()]),
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }

    // Google's OpenAI-compatibility surface answers HTTP 400 to `reasoning_effort: "none"`
    // while accepting the graded values, and a no-thinking caller sends exactly `none`.
    // The declared refusal is what keeps that caller's request from becoming a 400.
    #[tokio::test]
    async fn a_refused_effort_value_is_never_sent_on_the_chat_wire()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                body.get("reasoning_effort").is_none()
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_unsupported_effort(
            &format!("{}/v1", server.uri()),
            BTreeSet::from(["none".to_string()]),
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none"
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }

    // The refusal is a value list, not a switch: a graded effort the provider accepts still
    // reaches the wire, so declaring `none` unsupported does not silently disable control.
    #[tokio::test]
    async fn an_accepted_effort_value_still_reaches_the_chat_wire()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "reasoning_effort": "high"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map_with_unsupported_effort(
            &format!("{}/v1", server.uri()),
            BTreeSet::from(["none".to_string()]),
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }

    // A Responses body nests the effort under `reasoning`, beside fields the caller may have
    // set. Only the refused `effort` key is removed; an unrelated sibling such as `summary`
    // survives, and `reasoning` is not left behind as an empty object.
    #[tokio::test]
    async fn a_refused_effort_value_is_dropped_without_disturbing_its_siblings()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                let reasoning = body.get("reasoning");
                reasoning.and_then(|value| value.get("effort")).is_none()
                    && reasoning.and_then(|value| value.get("summary")) == Some(&json!("auto"))
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&responses_map_with_unsupported_effort(
            &format!("{}/v1", server.uri()),
            BTreeSet::from(["none".to_string()]),
        ))?;
        let raw = json!({
            "model": "client-facing",
            "input": "hi",
            "reasoning": {"effort": "none", "summary": "auto"}
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?;
        Ok(())
    }

    // A weak OpenAI-format tier emits thinking blocks with no signature. Replaying
    // them to Anthropic is rejected (Bedrock reports it as a SigV4 mismatch), so
    // the Anthropic leg must drop them while keeping signed ones.
    #[tokio::test]
    async fn anthropic_requests_drop_unsigned_thinking_blocks()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                let messages = body.get("messages").and_then(Value::as_array).cloned();
                let Some(messages) = messages else {
                    return false;
                };
                // The unsigned block is gone, the signed one survives, and the
                // message whose only block was unsigned is not left with an empty
                // content array.
                let blocks: Vec<&Value> = messages
                    .iter()
                    .filter_map(|message| message.get("content"))
                    .filter_map(Value::as_array)
                    .flatten()
                    .collect();
                let thinking: Vec<&&Value> = blocks
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("thinking"))
                    .collect();
                thinking.len() == 1
                    && thinking[0].get("signature").and_then(Value::as_str) == Some("sig-abc")
                    && messages
                        .iter()
                        .all(|message| message.get("content") != Some(&json!([])))
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&anthropic_map(&server.uri()))?;
        let raw = json!({
            "model": "client-facing",
            "max_tokens": 7,
            "messages": [
                {"role": "user", "content": "fix the build"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "weak tier reasoning", "signature": ""}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "signed reasoning", "signature": "sig-abc"},
                    {"type": "text", "text": "here goes"}
                ]},
                {"role": "user", "content": "continue"}
            ]
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("claude")),
                WireFormat::AnthropicMessages,
            )
            .await?;
        Ok(())
    }

    // A router can serve earlier turns from an OpenAI target and later turns from
    // an Anthropic one, so the Anthropic leg must drop OpenAI-only fields the
    // caller keeps sending or the upstream rejects the whole request.
    #[tokio::test]
    async fn anthropic_requests_drop_openai_only_fields()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                body.get("context_management").is_none() && body.get("reasoning_effort").is_none()
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&anthropic_map(&server.uri()))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "reasoning_effort": "high",
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015"}]
            }
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("claude")),
                WireFormat::AnthropicMessages,
            )
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn streaming_openai_chat_aggregates()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n\
             data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let response = client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await?;
        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hello world");
        assert_eq!(agg.usage.input_tokens, Some(1));
        assert_eq!(agg.usage.output_tokens, Some(2));
        assert_eq!(agg.usage.total_tokens, Some(3));
        Ok(())
    }

    #[tokio::test]
    async fn streaming_openai_chat_preserves_usage_opt_out()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "stream": true,
                "stream_options": {"include_usage": false}
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": false}
        });

        let response = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        assert!(matches!(response, RawResponse::Stream(_)));
        Ok(())
    }

    #[tokio::test]
    async fn upstream_500_is_upstream_http()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                ..
            }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn retryable_http_failure_recovers_within_budget()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                if observed_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(503)
                        .insert_header("retry-after", "0")
                        .set_body_string("temporarily unavailable")
                } else {
                    chat_success_response()
                }
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 1))?;
        let response = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;

        assert_eq!(completion_text(&agg), "recovered");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn deterministic_http_failure_is_not_retried()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(401).set_body_string("invalid key")
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 2))?;
        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an upstream error");
        };

        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp {
                status: StatusCode::UNAUTHORIZED,
                body
            } if body == "invalid key"
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn rate_limit_cooldown_depends_on_shared_credentials()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        for forward_auth in [true, false] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
                .expect(if forward_auth { 6 } else { 3 })
                .mount(&server)
                .await;
            let backend = HttpBackendConfig {
                forward_auth,
                failure_cooldown: Duration::from_secs(60),
                cooldown_group: None,
                ..config_with_retries(&format!("{}/v1", server.uri()), 2)
            };
            let client = TranslatingLlmClient::new(&[ModelConfig::new(
                "gpt",
                Backend::OpenAiChat(backend),
                None,
            )])?;
            for credential in ["Bearer first-user", "Bearer second-user"] {
                let mut headers = http::HeaderMap::new();
                headers.insert("authorization", http::HeaderValue::from_static(credential));
                let result = client
                    .call_rewrite_model(request_with_headers("gpt", headers), None)
                    .await;
                if !forward_auth && credential == "Bearer second-user" {
                    assert!(matches!(
                        result,
                        Err(LlmClientError::TemporarilyUnavailable)
                    ));
                } else {
                    assert!(matches!(
                        result,
                        Err(LlmClientError::UpstreamHttp {
                            status: StatusCode::TOO_MANY_REQUESTS,
                            ..
                        })
                    ));
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn retry_exhaustion_returns_the_final_upstream_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                let attempt = observed_calls.fetch_add(1, Ordering::SeqCst) + 1;
                ResponseTemplate::new(500)
                    .insert_header("retry-after", "0")
                    .set_body_string(format!("attempt {attempt}"))
            })
            .mount(&server)
            .await;

        let client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 2))?;
        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected retry exhaustion");
        };

        assert!(matches!(
            error,
            LlmClientError::UpstreamHttp {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body
            } if body == "attempt 3"
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[tokio::test]
    async fn deadline_expires_before_send_or_stream_poll()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"id\":\"test\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;
        for timeout in [Duration::ZERO, Duration::from_millis(200)] {
            let mut backend = config(&server.uri());
            backend.timeout = Some(timeout);
            let client = TranslatingLlmClient::new(&[ModelConfig::new(
                "gpt",
                Backend::OpenAiChat(backend),
                None,
            )])?;
            let result = client
                .call_rewrite_model(request_for(Some("gpt"), true), None)
                .await;
            if timeout.is_zero() {
                assert!(matches!(result, Err(LlmClientError::Timeout { .. })));
                assert!(
                    server
                        .received_requests()
                        .await
                        .expect("recorded requests")
                        .is_empty()
                );
            } else {
                let LlmResponse::Stream(mut chunks) = result?.llm_response else {
                    return Err("expected a streaming response".into());
                };
                tokio::time::sleep(Duration::from_millis(300)).await;
                assert!(matches!(
                    chunks.next().await,
                    Some(Err(LlmClientError::Timeout { .. }))
                ));
                assert!(chunks.next().await.is_none());
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn timeout_is_retried_before_a_response_is_returned()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                if observed_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(200).set_delay(Duration::from_millis(500))
                } else {
                    chat_success_response()
                }
            })
            .mount(&server)
            .await;

        let mut client =
            TranslatingLlmClient::new(&chat_map_with_retries(&format!("{}/v1", server.uri()), 1))?;
        client.client = reqwest::Client::builder()
            .timeout(Duration::from_millis(100))
            .build()?;
        let response = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await?;

        assert_eq!(
            completion_text(&response.llm_response.into_agg().await?),
            "recovered"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    // Serves a scripted SSE response, writing each event after its own delay.
    // Used for the cases wiremock cannot express: headers now, body later, and a
    // stream that starts promptly then stops producing.
    fn paced_stream_server(
        events: Vec<(u64, String)>,
    ) -> std::io::Result<(String, JoinHandle<std::io::Result<()>>, Arc<AtomicUsize>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let connections = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&connections);
        let handle = std::thread::spawn(move || {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept()?;
                observed.fetch_add(1, Ordering::SeqCst);
                let mut request = [0_u8; 1024];
                if stream.read(&mut request)? == 0 {
                    continue;
                }
                stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                      Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )?;
                for (delay_ms, payload) in &events {
                    std::thread::sleep(Duration::from_millis(*delay_ms));
                    write!(stream, "{:X}\r\n{payload}\r\n", payload.len())?;
                    stream.flush()?;
                }
                stream.write_all(b"0\r\n\r\n")?;
            }
            Ok(())
        });
        Ok((format!("http://{address}/v1"), handle, connections))
    }

    fn chat_delta(text: &str) -> String {
        format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n")
    }

    #[tokio::test]
    async fn attempt_timeout_before_headers_is_pre_commit()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(600)))
            .mount(&server)
            .await;

        let mut backend = config_with_retries(&format!("{}/v1", server.uri()), 0);
        backend.attempt_timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let error = match client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await
        {
            Ok(_) => return Err("attempt bound must fire before the response".into()),
            Err(error) => error,
        };
        assert!(
            matches!(error, LlmClientError::AttemptTimeout { .. }),
            "{error:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn attempt_timeout_after_headers_before_first_event_is_pre_commit()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // Headers arrive immediately; the first event does not. Without an attempt
        // bound this looked like a success until the overall deadline fired.
        let (base_url, _handle, connections) =
            paced_stream_server(vec![(600, chat_delta("late"))])?;
        let mut backend = config_with_retries(&base_url, 0);
        backend.attempt_timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let error = match client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await
        {
            Ok(_) => return Err("the first event must not arrive".into()),
            Err(error) => error,
        };
        assert!(
            matches!(error, LlmClientError::AttemptTimeout { .. }),
            "{error:?}"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        Ok(())
    }

    fn chat_role_only() -> String {
        // Announces the response without producing anything the caller can use.
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".to_string()
    }

    #[tokio::test]
    async fn a_non_output_event_does_not_end_the_attempt_boundary()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // The provider announces the response (role-only delta) and then stalls.
        // If the attempt returned at the announcement, the bound would stop
        // applying with zero output delivered and nothing left but the overall
        // deadline, which is unset on this lane.
        let (base_url, _handle, connections) =
            paced_stream_server(vec![(0, chat_role_only()), (600, chat_delta("late"))])?;
        let mut backend = config_with_retries(&base_url, 0);
        backend.attempt_timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let error = match client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await
        {
            Ok(_) => return Err("a non-output event must not satisfy the attempt bound".into()),
            Err(error) => error,
        };
        assert!(
            matches!(error, LlmClientError::AttemptTimeout { .. }),
            "{error:?}"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_non_output_event_followed_by_output_still_streams_normally()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // The control for the test above: holding the announcement must not cost
        // anything when output does follow.
        let (base_url, _handle, _connections) = paced_stream_server(vec![
            (0, chat_role_only()),
            (10, chat_delta("Hello")),
            (10, "data: [DONE]\n\n".to_string()),
        ])?;
        let mut backend = config_with_retries(&base_url, 0);
        backend.attempt_timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let response = client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hello");
        Ok(())
    }

    #[tokio::test]
    async fn the_overall_deadline_stays_terminal_and_is_not_an_attempt_timeout()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(600)))
            .mount(&server)
            .await;

        let mut backend = config_with_retries(&format!("{}/v1", server.uri()), 0);
        backend.timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let error = match client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await
        {
            Ok(_) => return Err("overall deadline must fire".into()),
            Err(error) => error,
        };
        assert!(matches!(error, LlmClientError::Timeout { .. }), "{error:?}");
        Ok(())
    }

    #[tokio::test]
    async fn a_slow_but_progressing_stream_is_not_cut_by_the_attempt_bound()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // Two events, 150 ms apart: the whole response outlives a 100 ms attempt
        // bound, but the first event arrives inside it. The attempt bound must stop
        // applying once the attempt returns, or valid streams get truncated.
        let (base_url, _handle, _connections) = paced_stream_server(vec![
            (10, chat_delta("Hello")),
            (150, chat_delta(" world")),
            (10, "data: [DONE]\n\n".to_string()),
        ])?;
        let mut backend = config_with_retries(&base_url, 0);
        backend.attempt_timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let response = client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await?;
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "Hello world");
        Ok(())
    }

    #[tokio::test]
    async fn idle_timeout_after_first_event_is_terminal_and_never_reaches_a_second_attempt()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        // The first event is delivered, then silence. This is post-commit: the
        // caller holds output, so the only correct outcome is a timeout on that
        // same stream, with no new upstream call behind its back.
        let (base_url, _handle, connections) = paced_stream_server(vec![
            (10, chat_delta("partial")),
            (600, chat_delta("never")),
        ])?;
        let mut backend = config_with_retries(&base_url, 2);
        backend.stream_idle_timeout = Some(Duration::from_millis(100));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;

        let response = client
            .call_rewrite_model(request_for(Some("gpt"), true), None)
            .await?;
        let LlmResponse::Stream(mut chunks) = response.llm_response else {
            return Err("expected a stream".into());
        };
        let mut saw_partial = false;
        let mut timeout = None;
        while let Some(item) = chunks.next().await {
            match item {
                Ok(event) => {
                    saw_partial |= event
                        .normalized()
                        .iter()
                        .any(|chunk| matches!(chunk, LlmResponseChunk::TextDelta { .. }));
                }
                Err(error) => {
                    timeout = Some(error);
                    break;
                }
            }
        }
        assert!(saw_partial, "the first event must reach the caller");
        assert!(
            matches!(timeout, Some(LlmClientError::Timeout { .. })),
            "{timeout:?}"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn retryable_error_classes_are_explicit() {
        let transport = AttemptFailure {
            error: LlmClientError::Transport {
                source: std::io::Error::other("disconnected").into(),
            },
            status: None,
            retry_after: None,
        };
        assert!(transport.is_retryable());

        for status in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::from_u16(599)
                .expect("599 is a syntactically valid HTTP code in the http crate"),
        ] {
            let failure = AttemptFailure {
                error: LlmClientError::UpstreamHttp {
                    status,
                    body: String::new(),
                },
                status: Some(status),
                retry_after: None,
            };
            assert!(failure.is_retryable(), "HTTP {status} should retry");
        }
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::CONFLICT,
            StatusCode::from_u16(600)
                .expect("600 is a syntactically valid HTTP code in the http crate"),
        ] {
            let failure = AttemptFailure {
                error: LlmClientError::UpstreamHttp {
                    status,
                    body: String::new(),
                },
                status: Some(status),
                retry_after: None,
            };
            assert!(!failure.is_retryable(), "HTTP {status} should fail fast");
        }

        let configuration = AttemptFailure {
            error: LlmClientError::Configuration {
                message: "invalid header".to_string(),
            },
            status: None,
            retry_after: None,
        };
        assert!(!configuration.is_retryable());

        let context_window = AttemptFailure {
            error: LlmClientError::ContextWindowExceeded {
                model: ModelId::from("gpt"),
                message: "too long".to_string(),
            },
            status: Some(StatusCode::BAD_REQUEST),
            retry_after: None,
        };
        assert!(!context_window.is_retryable());
    }

    /// A quota-limited leaf may not turn one bad response into four reservations.
    ///
    /// The two axes are deliberately separate. `is_retryable` still says a 429 or a 503
    /// is the kind of failure that can clear, because the route uses that answer to
    /// decide whether an eligible sibling is worth trying. `retry_budget_allows` says
    /// whether *this* target may spend another reservation, because on a target with a
    /// daily allowance each attempt charges the budget before the provider is called and
    /// nothing refunds it.
    ///
    /// The first half of each case pins the class answer, so a change that quietly makes
    /// 429s non-retryable everywhere fails here rather than passing as a tightened budget.
    #[test]
    fn a_quota_limited_leaf_retries_a_throttle_zero_times_and_a_5xx_once() {
        let with_status = |status: StatusCode, body: &str| AttemptFailure {
            error: LlmClientError::UpstreamHttp {
                status,
                body: body.to_string(),
            },
            status: Some(status),
            retry_after: None,
        };
        let transport = AttemptFailure {
            error: LlmClientError::Transport {
                source: std::io::Error::other("disconnected").into(),
            },
            status: None,
            retry_after: None,
        };

        // Unmarked 429 and provider-pool 429 are both retryable classes, and neither may
        // spend a second reservation on the same free target.
        for body in [
            "{}",
            r#"{"error":{"code":429,"metadata":{"limit_source":"upstream_provider_shared_pool"}}}"#,
        ] {
            let throttle = with_status(StatusCode::TOO_MANY_REQUESTS, body);
            assert!(throttle.is_retryable(), "{body} is still a throttle class");
            for attempt in 0..4 {
                assert!(
                    !throttle.may_retry(attempt, 2, true),
                    "429 at attempt {attempt} must not reserve again on a free leaf"
                );
            }
        }

        // Transient infrastructure: exactly one further attempt, whatever the configured
        // budget says, so the worst case is two reservations rather than four.
        for transient in [
            with_status(StatusCode::INTERNAL_SERVER_ERROR, ""),
            with_status(StatusCode::SERVICE_UNAVAILABLE, ""),
            with_status(StatusCode::BAD_GATEWAY, ""),
            transport,
        ] {
            assert!(transient.is_retryable());
            assert!(
                transient.may_retry(0, 2, true),
                "the first attempt may still try once more"
            );
            for attempt in 1..4 {
                assert!(
                    !transient.may_retry(attempt, 2, true),
                    "attempt {attempt} must be terminal on a free leaf"
                );
            }
        }

        // A non-retryable class stays non-retryable; the budget must not resurrect it.
        let unauthorized = with_status(StatusCode::UNAUTHORIZED, "");
        assert!(!unauthorized.may_retry(0, 2, true));
        assert!(!unauthorized.may_retry(0, 2, false));

        // Control: a lane with no daily allowance keeps today's configured behaviour
        // exactly, so a paid request that 429s still gets both retries.
        let throttle = with_status(StatusCode::TOO_MANY_REQUESTS, "{}");
        assert!(throttle.may_retry(0, 2, false));
        assert!(throttle.may_retry(1, 2, false));
        assert!(
            !throttle.may_retry(2, 2, false),
            "the configured budget still bounds an unmetered lane"
        );
        // A lane configured with zero retries retries nothing, metered or not.
        assert!(!throttle.may_retry(0, 0, false));
        assert!(!throttle.may_retry(0, 0, true));
    }

    /// A 5xx that masks a permanent auth or billing failure is not infrastructure.
    ///
    /// OpenRouter reports `authentication`, `permission_denied` and `payment_required`
    /// behind `server_error`, naming the real cause only in the sibling `error_type`.
    /// Retrying cannot restore either condition, so the retry loop must read the same
    /// marker the fallback classifier reads.
    ///
    /// The controls keep the general 5xx retry intact: a bare 5xx, an empty body, and a
    /// 5xx naming an unrelated type must all still retry, or this rule would pass while
    /// quietly removing recovery that works today.
    #[test]
    fn a_masked_permanent_failure_is_not_retried_but_plain_5xxs_are() {
        let failure = |body: &str| AttemptFailure {
            error: LlmClientError::UpstreamHttp {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: body.to_string(),
            },
            status: Some(StatusCode::INTERNAL_SERVER_ERROR),
            retry_after: None,
        };

        for masked in ["authentication", "permission_denied", "payment_required"] {
            let body = format!(r#"{{"error":{{"code":"server_error"}},"error_type":"{masked}"}}"#);
            assert!(
                !failure(&body).is_retryable(),
                "a masked {masked} must fail fast"
            );
        }

        assert!(failure(r#"{"error":{"code":"server_error"}}"#).is_retryable());
        assert!(failure("").is_retryable());
        assert!(
            failure(r#"{"error":{"code":"server_error"},"error_type":"rate_limit_exceeded"}"#)
                .is_retryable()
        );
    }

    // End to end through the real HTTP client: a provider that reports its
    // correlation id only in a response header must still hand that id to its
    // caller, because the header is gone by the time the error reaches them.
    #[tokio::test]
    async fn a_header_only_request_id_reaches_the_returned_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400)
                    .insert_header("x-request-id", "req_header_only_42")
                    .set_body_json(json!({
                        "error": {"message": "provider returned error", "code": 400}
                    })),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        // Response is not Debug (it may own a live stream), so match rather than
        // unwrap: the error is the expected outcome, not an inconvenience.
        let error = match client
            .call_rewrite_model(request_for(None, false), Some(&ModelId::from("gpt")))
            .await
        {
            Ok(_) => panic!("a 400 must not succeed"),
            Err(error) => error,
        };

        let LlmClientError::UpstreamHttp { status, body } = &error else {
            panic!("expected an upstream HTTP error, got {error:?}");
        };
        assert_eq!(*status, StatusCode::BAD_REQUEST);
        let parsed: Value = serde_json::from_str(body).expect("body is still JSON");
        assert_eq!(
            parsed["error"]["request_id"],
            Value::String("req_header_only_42".into()),
            "the header id must survive into the error body: {body}"
        );
        // The provider's own message is untouched.
        assert_eq!(
            parsed["error"]["message"],
            Value::String("provider returned error".into())
        );
        Ok(())
    }

    // Integration: when the provider names a request in its body AND sends a
    // different header id, the provider's own value must survive the whole path
    // through the real client. The header is a supplement, never an override.
    #[tokio::test]
    async fn a_provider_authored_id_survives_a_differing_header()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400)
                    .insert_header("x-request-id", "req_header_999")
                    .set_body_json(json!({
                        "error": {
                            "message": "thinking mode failure (request_id: req_body_123)",
                            "type": "upstream_error",
                            "request_id": "req_body_123"
                        }
                    })),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let error = match client
            .call_rewrite_model(request_for(None, false), Some(&ModelId::from("gpt")))
            .await
        {
            Ok(_) => panic!("a 400 must not succeed"),
            Err(error) => error,
        };

        let LlmClientError::UpstreamHttp { body, .. } = &error else {
            panic!("expected an upstream HTTP error, got {error:?}");
        };
        assert!(
            !body.contains("req_header_999"),
            "the header id must not be injected over the provider's: {body}"
        );
        let parsed: Value = serde_json::from_str(body).expect("body is still JSON");
        assert_eq!(
            switchyard_protocol::correlation::provider_request_id(&parsed).as_deref(),
            Some("req_body_123")
        );
        Ok(())
    }

    // Control: a provider that sends no id header must produce a byte-identical
    // body, so the propagation cannot be mistaken for rewriting provider data.
    #[tokio::test]
    async fn a_failure_without_an_id_header_is_passed_through_untouched()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"message": "provider returned error", "code": 400}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let error = match client
            .call_rewrite_model(request_for(None, false), Some(&ModelId::from("gpt")))
            .await
        {
            Ok(_) => panic!("a 400 must not succeed"),
            Err(error) => error,
        };

        let LlmClientError::UpstreamHttp { body, .. } = &error else {
            panic!("expected an upstream HTTP error, got {error:?}");
        };
        assert!(
            !body.contains("request_id"),
            "no id must not be invented: {body}"
        );
        Ok(())
    }

    // ---- upstream request id propagation -------------------------------------
    //
    // The id must survive the only place it exists: a response header on a failure.

    fn header_map(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).expect("valid name"),
                reqwest::header::HeaderValue::from_str(value).expect("valid value"),
            );
        }
        headers
    }

    // Attaching an id must not change how the failure is CLASSIFIED.
    //
    // The error body is read by the routing classifier as well as by the retry
    // predicate, so a body rewritten in a way those read differently would trade an
    // observability gain for a routing regression. This pins the class, the status and
    // the provider's own message across the change.
    #[test]
    fn attaching_an_id_leaves_the_failure_classification_unchanged() {
        use switchyard_protocol::LlmClientError;

        let base = r#"{"error":{"message":"Provider returned error","code":"upstream_error"}}"#;
        let with_id = with_request_id(base.to_string(), Some("req_folded"));

        let classify = |body: &str| {
            let error = switchyard_libsy::LibsyError::client_call(
                "gpt",
                LlmClientError::UpstreamHttp {
                    status: StatusCode::BAD_REQUEST,
                    body: body.to_string(),
                },
            );
            crate::run::failure_class(&error)
        };

        assert_eq!(
            classify(base),
            classify(&with_id),
            "classification must not move when an id is attached"
        );
        // And it is still a request failure, not something another candidate recovers from.
        assert_eq!(classify(&with_id), crate::run::FailureClass::Request);

        let parsed: Value = serde_json::from_str(&with_id).expect("valid json");
        assert_eq!(
            parsed["error"]["message"],
            Value::String("Provider returned error".into()),
            "the provider's message must be untouched"
        );
        assert_eq!(
            parsed["error"]["code"],
            Value::String("upstream_error".into()),
            "the provider's code must be untouched"
        );
    }

    #[test]
    fn an_identifier_shaped_header_value_is_preserved_exactly() {
        // Exact preservation is the whole point: this string must be searchable in
        // the provider's logs, so it may not be altered in any way.
        for value in [
            "req_abc123",
            "5502af93-ffbb-46bc-95ec-6402a807b7e8",
            "req-1.example_v2:a",
        ] {
            assert_eq!(
                upstream_request_id(&header_map(&[("x-request-id", value)])).as_deref(),
                Some(value),
                "an identifier must survive byte for byte"
            );
        }
        // Anthropic's spelling, without the `x-` prefix.
        assert_eq!(
            upstream_request_id(&header_map(&[("request-id", "req_anthropic")])).as_deref(),
            Some("req_anthropic")
        );
    }

    #[test]
    fn surrounding_whitespace_is_transport_framing_not_identity() {
        // Optional whitespace around a value is framing, so trimming it still yields
        // the provider's identifier exactly.
        assert_eq!(
            upstream_request_id(&header_map(&[("x-request-id", "  req_padded\t ")])).as_deref(),
            Some("req_padded")
        );
    }

    // A value that is not exactly an identifier is REFUSED, never repaired.
    //
    // Repairing produces a string that looks like evidence and matches nothing in the
    // provider's logs. Each of these previously yielded a synthetic id.
    #[test]
    fn a_malformed_header_value_yields_no_id() {
        for value in [
            // Embedded quote plus injected fields: the old filter produced
            // "req_okinjectedyes", which matches no request anywhere.
            r#"req_ok","injected":"yes"#,
            r#"req_ok\injected"#,
            // A comma-separated list is several ids, not one.
            "req_a,req_b",
            // Interior whitespace is not part of an identifier.
            "req ok",
            "(req_ok)",
            "",
            "   ",
        ] {
            assert_eq!(
                upstream_request_id(&header_map(&[("x-request-id", value)])),
                None,
                "must refuse {value:?} rather than rewrite it"
            );
        }
    }

    // An over-long value is a different string, not the provider's id truncated.
    #[test]
    fn an_over_long_header_value_yields_no_id() {
        let at_limit = "a".repeat(switchyard_protocol::correlation::MAX_REQUEST_ID_LEN);
        assert_eq!(
            upstream_request_id(&header_map(&[("x-request-id", &at_limit)])).as_deref(),
            Some(at_limit.as_str()),
            "exactly at the limit is still an identifier"
        );
        for over in [
            "a".repeat(switchyard_protocol::correlation::MAX_REQUEST_ID_LEN + 1),
            "a".repeat(5_000),
        ] {
            assert_eq!(
                upstream_request_id(&header_map(&[("x-request-id", &over)])),
                None,
                "truncating would invent an id that matches nothing"
            );
        }
    }

    #[test]
    fn the_id_is_folded_into_the_body_it_was_missing_from() {
        let body = with_request_id(
            r#"{"error":{"message":"bad request","code":"invalid_request"}}"#.to_string(),
            Some("req_folded"),
        );
        let parsed: Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(
            parsed["error"]["request_id"],
            Value::String("req_folded".into())
        );
        // Everything the provider said is preserved.
        assert_eq!(
            parsed["error"]["message"],
            Value::String("bad request".into())
        );
        assert_eq!(
            parsed["error"]["code"],
            Value::String("invalid_request".into())
        );
    }

    // The edge case that outlived the first round: prose is the ONLY id in the body,
    // and a header must not be injected over it.
    //
    // Before the shared rule, `provider_named_a_request` checked only structured
    // fields, so this body was treated as unnamed: the header id was written as
    // `error.request_id`, and the server - preferring the structured field - then
    // reported the header id instead of the provider's own. The body must be left
    // byte-identical.
    #[test]
    fn a_prose_only_provider_id_is_not_overridden_by_a_header() {
        let body = r#"{"error":{"message":"thinking mode failure (request_id: req_body_123)","type":"upstream_error"}}"#;
        let with_header = with_request_id(body.to_string(), Some("req_header_999"));

        assert_eq!(
            with_header, body,
            "a prose-only provider id must leave the body byte-identical"
        );
        assert!(
            !with_header.contains("req_header_999"),
            "the header id must not be injected: {with_header}"
        );

        // And the reader reports the provider's prose id, not a structured field that
        // was never there.
        let document: Value = serde_json::from_str(&with_header).expect("valid json");
        assert_eq!(
            switchyard_protocol::correlation::provider_request_id(&document).as_deref(),
            Some("req_body_123")
        );
    }

    // A header still supplements a body that has no id at all - the supplement is not
    // withdrawn, only subordinated.
    #[test]
    fn a_header_supplements_a_body_with_no_id() {
        let body = r#"{"error":{"message":"provider returned error","code":400}}"#;
        let with_header = with_request_id(body.to_string(), Some("req_header_only"));
        assert_ne!(with_header, body, "an id-free body should be supplemented");
        let document: Value = serde_json::from_str(&with_header).expect("valid json");
        assert_eq!(
            switchyard_protocol::correlation::provider_request_id(&document).as_deref(),
            Some("req_header_only")
        );
    }

    // Control: prose that mentions a request id without carrying one must NOT count as
    // a provider-authored id, or a header would be needlessly suppressed.
    #[test]
    fn prose_without_an_identifier_does_not_suppress_the_header() {
        let body = r#"{"error":{"message":"the request id header is missing","code":400}}"#;
        let with_header = with_request_id(body.to_string(), Some("req_header_only"));
        let document: Value = serde_json::from_str(&with_header).expect("valid json");
        assert_eq!(
            switchyard_protocol::correlation::provider_request_id(&document).as_deref(),
            Some("req_header_only"),
            "a header must still be attached when the prose carries no id"
        );
    }

    // A provider-authored id is authoritative in EVERY location the extractor reads,
    // including Anthropic's top level. Injecting beside it would put two ids in one
    // document and, because the extractor prefers the nested one, would silently
    // override what the provider actually said.
    #[test]
    fn a_provider_authored_id_is_never_overridden() {
        for body in [
            r#"{"error":{"message":"x","code":"bad","request_id":"req_provider"}}"#,
            r#"{"error":{"message":"x","code":"bad","requestId":"req_provider"}}"#,
            // Top level, the Anthropic shape.
            r#"{"type":"error","error":{"message":"x"},"request_id":"req_provider"}"#,
        ] {
            let with_header = with_request_id(body.to_string(), Some("req_from_header"));
            assert_eq!(
                with_header, body,
                "a provider id must leave the body byte-identical: {body}"
            );
            assert!(
                !with_header.contains("req_from_header"),
                "the header id must not appear: {with_header}"
            );
        }
    }

    // Controls: nothing is added when there is nothing to add, and nothing is
    // restructured when the body is not a shape this understands.
    #[test]
    fn the_body_is_untouched_when_it_cannot_be_added_to() {
        let untouched = r#"{"error":{"message":"x","code":"bad"}}"#.to_string();
        assert_eq!(with_request_id(untouched.clone(), None), untouched);

        let plain = "upstream said no".to_string();
        assert_eq!(with_request_id(plain.clone(), Some("req_x")), plain);

        let odd = r#"{"error":"a string"}"#.to_string();
        assert_eq!(with_request_id(odd.clone(), Some("req_x")), odd);
    }

    #[test]
    fn an_account_wide_quota_429_is_not_retried_but_pool_429s_are() {
        let failure = |body: &str| AttemptFailure {
            error: LlmClientError::UpstreamHttp {
                status: StatusCode::TOO_MANY_REQUESTS,
                body: body.to_string(),
            },
            status: Some(StatusCode::TOO_MANY_REQUESTS),
            retry_after: None,
        };

        let account = failure(
            r#"{"error":{"code":429,"metadata":{"limit_source":"openrouter_free_tier_daily","remaining":0}}}"#,
        );
        assert!(
            !account.is_retryable(),
            "a daily account cap cannot clear on retry"
        );
        assert!(is_account_wide_quota(&account.error));

        // A provider-pool 429 is model-scoped and short-lived.
        let pool = failure(
            r#"{"error":{"code":429,"metadata":{"limit_source":"upstream_provider_shared_pool"}}}"#,
        );
        assert!(pool.is_retryable(), "a provider pool 429 keeps its retries");
        assert!(!is_account_wide_quota(&pool.error));

        // An unmarked 429 keeps today's behaviour.
        assert!(failure("{}").is_retryable());

        // The marker only counts on a 429.
        let mislabelled = AttemptFailure {
            error: LlmClientError::UpstreamHttp {
                status: StatusCode::SERVICE_UNAVAILABLE,
                body: r#"{"error":{"metadata":{"limit_source":"openrouter_free_tier_daily"}}}"#
                    .to_string(),
            },
            status: Some(StatusCode::SERVICE_UNAVAILABLE),
            retry_after: None,
        };
        assert!(
            mislabelled.is_retryable(),
            "a 503 is transient whatever its body says"
        );
    }

    /// OpenAI documents several distinct 429 causes that retrying cannot fix.
    ///
    /// A billing or spend cap is not a throttle. The provider's own documentation is
    /// explicit: "Retrying billing, spend, or quota errors won't restore API access."
    /// These arrive without OpenRouter's `metadata.limit_source` marker, so a check that
    /// only understands that marker classifies them as ordinary rate limits and spends
    /// requests against an allowance known to be empty.
    ///
    /// The control matters as much as the rule: a genuine rate limit on the same status
    /// must keep its retries, or this change removes recovery that works today.
    #[test]
    fn documented_billing_429_codes_are_account_wide_but_rate_limits_still_retry() {
        let failure = |body: &str| AttemptFailure {
            error: LlmClientError::UpstreamHttp {
                status: StatusCode::TOO_MANY_REQUESTS,
                body: body.to_string(),
            },
            status: Some(StatusCode::TOO_MANY_REQUESTS),
            retry_after: None,
        };

        for code in [
            "credit_balance_exhausted",
            "organization_spend_limit_exceeded",
            "project_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
            "insufficient_quota",
        ] {
            let body = format!(r#"{{"error":{{"type":"insufficient_quota","code":"{code}"}}}}"#);
            let billing = failure(&body);
            assert!(
                is_account_wide_quota(&billing.error),
                "{code} is an account-wide condition"
            );
            assert!(
                !billing.is_retryable(),
                "{code} cannot be fixed by retrying the same credential"
            );
        }

        // A genuine rate limit keeps retrying. The provider sends `slow_down` and a bare
        // "rate limit reached" for these; neither is a billing condition.
        for body in [
            r#"{"error":{"type":"rate_limit_error","code":"slow_down"}}"#,
            r#"{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded"}}"#,
            r#"{"error":{"message":"Rate limit reached for requests"}}"#,
        ] {
            assert!(
                failure(body).is_retryable(),
                "a genuine rate limit must keep its retries: {body}"
            );
        }
    }

    /// The free-model daily allowance, when OpenRouter omits its usual marker.
    ///
    /// Measured 2026-11-19 against this deployment's own credential: a request for
    /// `openrouter/free` answered
    /// `{"error":{"code":"upstream_error","type":"upstream_error","message":"Rate limit
    /// exceeded: free-models-per-day-high-balance. "}}` — a plain 429 carrying no
    /// `metadata` at all. The key was paid tier with credit remaining, so the only
    /// resource that sentence can name is the per-credential free-model allowance every
    /// free model on it shares. Without recognizing it the request walks the rest of the
    /// ladder, spending one refused call per rung against an allowance already at zero.
    ///
    /// The control is the per-minute spelling of the same sentence: that is a genuine
    /// throttle and must keep its retries, so the match cannot be "rate limit exceeded".
    #[test]
    fn an_unmarked_free_model_daily_cap_is_account_wide_but_a_per_minute_throttle_retries() {
        let failure = |body: &str| AttemptFailure {
            error: LlmClientError::UpstreamHttp {
                status: StatusCode::TOO_MANY_REQUESTS,
                body: body.to_string(),
            },
            status: Some(StatusCode::TOO_MANY_REQUESTS),
            retry_after: None,
        };

        // The recorded body, trimmed only of the provider's trailing space.
        let measured = failure(
            r#"{"error":{"code":"upstream_error","type":"upstream_error","message":"Rate limit exceeded: free-models-per-day-high-balance."}}"#,
        );
        assert!(
            is_account_wide_quota(&measured.error),
            "the free-model daily allowance is shared by every candidate on the credential"
        );
        assert!(
            !measured.is_retryable(),
            "a daily allowance cannot clear on the next attempt"
        );

        // A throttle on the same status keeps recovering.
        let per_minute = failure(
            r#"{"error":{"code":"upstream_error","type":"upstream_error","message":"Rate limit exceeded: free-models-per-min."}}"#,
        );
        assert!(
            !is_account_wide_quota(&per_minute.error),
            "a per-minute throttle is not the daily allowance"
        );
        assert!(
            per_minute.is_retryable(),
            "a per-minute throttle must keep its retries"
        );
    }

    #[test]
    fn retry_after_supports_seconds_and_http_dates() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, reqwest::header::HeaderValue::from_static("3"));
        assert_eq!(retry_after_delay(&headers), Some(Duration::from_secs(3)));

        let retry_at = SystemTime::now() + Duration::from_secs(2);
        let value = httpdate::fmt_http_date(retry_at);
        let Ok(value) = reqwest::header::HeaderValue::from_str(&value) else {
            panic!("formatted HTTP date should be a valid header");
        };
        headers.insert(RETRY_AFTER, value);
        let Some(delay) = retry_after_delay(&headers) else {
            panic!("HTTP date should produce a retry delay");
        };
        assert!(delay <= Duration::from_secs(2));
    }

    #[tokio::test]
    async fn routed_llm_client_exposes_timeout_variant()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_delay(std::time::Duration::from_millis(100)),
            )
            .mount(&server)
            .await;

        let mut client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        client.client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(10))
            .build()?;

        let Err(error) = client.call(request_for(Some("gpt"), false)).await else {
            panic!("expected a timeout");
        };
        let LlmClientError::Timeout { source } = error else {
            panic!("expected the protocol timeout variant");
        };
        let Some(source) = source.downcast_ref::<reqwest::Error>() else {
            panic!("expected the reqwest timeout source");
        };
        assert!(source.is_timeout());
        Ok(())
    }

    #[tokio::test]
    async fn context_overflow_400_is_mapped()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": {"code": "context_length_exceeded", "message": "too big"}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected an error");
        };
        assert!(matches!(
            error,
            LlmClientError::ContextWindowExceeded { model, .. } if model == "gpt"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn content_policy_400_preserves_upstream_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let body = json!({
            "error": {
                "code": "content_policy_violation",
                "message": "request blocked by content policy",
                "type": "invalid_request_error"
            },
            "metadata": {"documentation_section": "context window"}
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(&body))
            .expect(1)
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let Err(error) = client
            .call_rewrite_model(request_for(Some("gpt"), false), None)
            .await
        else {
            panic!("expected a policy denial");
        };
        let LlmClientError::UpstreamHttp {
            status,
            body: actual,
        } = error
        else {
            panic!("expected the upstream HTTP error, got {error:?}");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(serde_json::from_str::<serde_json::Value>(&actual)?, body);
        Ok(())
    }

    #[tokio::test]
    async fn forwards_only_allowlisted_metadata_headers()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::header("x-request-id", "abc"))
            // A forwarded Authorization must NOT override the backend's bearer key.
            .and(wiremock::matchers::header("authorization", "Bearer secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static("abc"));
        headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-key"),
        );
        headers.insert(
            "x-goog-api-key",
            http::HeaderValue::from_static("client-google-key"),
        );
        headers.insert(
            "x-custom-internal-secret",
            http::HeaderValue::from_static("client-custom-key"),
        );
        let request = request_with_headers("gpt", headers);

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        // The allowed request id survives and caller authorization cannot replace
        // the backend's credential.
        client.call_rewrite_model(request, None).await?;
        let received = server
            .received_requests()
            .await
            .ok_or("request recording should be enabled")?;
        let received = received.first().ok_or("expected one upstream request")?;
        assert!(!received.headers.contains_key("x-goog-api-key"));
        assert!(!received.headers.contains_key("x-custom-internal-secret"));
        Ok(())
    }

    #[tokio::test]
    async fn forwards_application_headers_with_forward_auth()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::header(
                "authorization",
                "Bearer client-key",
            ))
            .and(wiremock::matchers::header(
                "x-goog-api-key",
                "client-google-key",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let mut headers = http::HeaderMap::new();
        headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-key"),
        );
        headers.insert(
            "x-goog-api-key",
            http::HeaderValue::from_static("client-google-key"),
        );
        headers.insert("host", http::HeaderValue::from_static("client.example"));
        headers.insert("content-type", http::HeaderValue::from_static("text/plain"));
        headers.insert("connection", http::HeaderValue::from_static("x-hop-by-hop"));
        headers.insert(
            "x-hop-by-hop",
            http::HeaderValue::from_static("client-only"),
        );
        let request = request_with_headers("gpt", headers);
        let backend_config = forwarding_config(&format!("{}/v1", server.uri()));
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend_config),
            None,
        )])?;

        client.call_rewrite_model(request, None).await?;
        let received = server
            .received_requests()
            .await
            .ok_or("request recording should be enabled")?;
        let received = received.first().ok_or("expected one upstream request")?;
        assert_ne!(received.headers["host"], "client.example");
        assert_eq!(received.headers["content-type"], "application/json");
        assert!(!received.headers.contains_key("connection"));
        assert!(!received.headers.contains_key("x-hop-by-hop"));
        Ok(())
    }

    #[tokio::test]
    async fn forward_auth_preserves_anthropic_headers_and_redacts_errors()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(|request: &wiremock::Request| {
                request
                    .headers
                    .get("anthropic-beta")
                    .and_then(|value| value.to_str().ok())
                    == Some("oauth-2025-04-20")
                    && request.headers.get_all("anthropic-version").iter().count() == 1
                    && request
                        .headers
                        .get("anthropic-version")
                        .and_then(|value| value.to_str().ok())
                        == Some("2023-06-01")
            })
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {"message": "rejected client-google-key"}
            })))
            .mount(&server)
            .await;

        let mut headers = http::HeaderMap::new();
        headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-key"),
        );
        headers.insert(
            "x-goog-api-key",
            http::HeaderValue::from_static("client-google-key"),
        );
        headers.insert(
            "anthropic-beta",
            http::HeaderValue::from_static("oauth-2025-04-20,prompt-caching-2024-07-31"),
        );
        headers.insert(
            "anthropic-version",
            http::HeaderValue::from_static("caller-version"),
        );
        let backend_config = forwarding_config(&server.uri());
        let client = TranslatingLlmClient::new(&[ModelConfig::new(
            "claude",
            Backend::Anthropic(backend_config),
            None,
        )])?;
        let raw = json!({
            "model": "claude",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hello"}]
        });

        let Err(LlmClientError::UpstreamHttp { body, .. }) = client
            .call_rewrite_model_raw(
                raw,
                Some(headers),
                Some(&ModelId::from("claude")),
                WireFormat::AnthropicMessages,
            )
            .await
        else {
            panic!("expected an upstream HTTP error");
        };
        assert_eq!(body, r#"{"error":{"message":"rejected [REDACTED]"}}"#);
        Ok(())
    }

    // Exercises the `RoutedLlmClient` impl: `call` uses the model already materialized in the
    // request and round-trips a buffered response.
    #[tokio::test]
    async fn routed_llm_client_serves_the_request_model()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "routed hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let response = client.call(request_for(Some("gpt"), false)).await?;
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "routed hi");
        Ok(())
    }

    #[tokio::test]
    async fn invalid_raw_request_is_a_request_translation_error()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let client = TranslatingLlmClient::new(&[])?;
        let Err(error) = client
            .call_rewrite_model_raw(
                json!("invalid"),
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await
        else {
            panic!("expected request translation to fail");
        };

        assert!(matches!(
            error,
            LlmClientError::RequestTranslation(message) if !message.is_empty()
        ));
        Ok(())
    }

    // Raw path, buffered: decode an OpenAI Chat body -> call -> encode back to OpenAI
    // Chat JSON, with the served `model` restamped over the id the caller addressed.
    #[tokio::test]
    async fn call_rewrite_model_raw_round_trips_buffered_json()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let RawResponse::Buffered(body) = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?
        else {
            panic!("expected a buffered response");
        };

        assert_eq!(body["choices"][0]["message"]["content"], "Hi there");
        // The client sees the model that answered, not the "client-facing" route id.
        assert_eq!(body["model"], "gpt");
        Ok(())
    }

    // A Codex namespace is folded into the upstream tool name, then split back
    // into name and namespace on the Responses call that returns to Codex.
    #[tokio::test]
    async fn call_rewrite_model_raw_restores_codex_mcp_namespace()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "tools": [{
                    "type": "function",
                    "function": {"name": "mcp__open_websearch__search"}
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-1",
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "mcp__open_websearch__search",
                                "arguments": "{\"q\":\"rust\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "input": "Search for Rust.",
            "tools": [{
                "type": "namespace",
                "name": "mcp__open_websearch",
                "tools": [{
                    "type": "function",
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
                }]
            }]
        });

        let RawResponse::Buffered(body) = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?
        else {
            panic!("expected a buffered response");
        };

        assert_eq!(body["output"][0]["type"], "function_call");
        assert_eq!(body["output"][0]["name"], "search");
        assert_eq!(body["output"][0]["namespace"], "mcp__open_websearch");
        // Arguments are parsed and re-serialized, so the spacing is normalized.
        assert_eq!(body["output"][0]["arguments"], "{\"q\": \"rust\"}");
        Ok(())
    }

    #[tokio::test]
    async fn openai_responses_backend_flattens_codex_namespaces_before_upstream()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "model": "gpt",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "mcp__open_websearch__search",
                    "arguments": "{\"q\":\"rust\"}"
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&responses_map(&format!("{}/v1", server.uri())))?;
        let parameters = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let raw = json!({
            "model": "client-facing",
            "input": [{
                "type": "function_call",
                "call_id": "call_0",
                "name": "search",
                "namespace": "mcp__open_websearch",
                "arguments": "{}"
            }],
            "tool_choice": {
                "type": "function",
                "name": "search",
                "namespace": "mcp__open_websearch"
            },
            "tools": [{
                "type": "namespace",
                "name": "mcp__open_websearch",
                "tools": [{
                    "type": "function",
                    "name": "search",
                    "parameters": parameters.clone()
                }]
            }]
        });

        let RawResponse::Buffered(body) = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?
        else {
            panic!("expected a buffered response");
        };

        let received = server
            .received_requests()
            .await
            .ok_or("request recording should be enabled")?;
        let request_body: Value = serde_json::from_slice(&received[0].body)?;
        assert_eq!(request_body["model"], "gpt");
        assert_eq!(
            request_body["tools"],
            json!([{
                "type": "function",
                "name": "mcp__open_websearch__search",
                "description": "",
                "parameters": parameters
            }])
        );
        assert_eq!(
            request_body["tool_choice"],
            json!({"type": "function", "name": "mcp__open_websearch__search"})
        );
        assert_eq!(
            request_body["input"][0]["name"],
            "mcp__open_websearch__search"
        );
        assert!(request_body["input"][0].get("namespace").is_none());

        assert_eq!(body["output"][0]["type"], "function_call");
        assert_eq!(body["output"][0]["name"], "search");
        assert_eq!(body["output"][0]["namespace"], "mcp__open_websearch");
        Ok(())
    }

    // Raw path, streaming: an inbound `stream: true` request yields an unframed stream
    // of OpenAI Chat chunk objects whose deltas reassemble the completion.
    #[tokio::test]
    async fn call_rewrite_model_raw_streams_wire_events()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use futures::TryStreamExt;

        let server = MockServer::start().await;
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let RawResponse::Stream(stream) = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?
        else {
            panic!("expected a streamed response");
        };

        let events: Vec<Value> = stream.try_collect().await?;
        assert!(!events.is_empty(), "expected at least one wire event");
        let content: String = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "Hello world");
        // The mock frames carry no `model`, so every chunk's model comes from the
        // served id rather than the "unknown" fallback or the caller's route id.
        assert!(events.iter().all(|event| event["model"] == "gpt"));
        Ok(())
    }

    // Raw path forwards allowed caller headers to the upstream.
    #[tokio::test]
    async fn call_rewrite_model_raw_forwards_headers()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::header("x-request-id", "abc"))
            // A forwarded authorization must NOT override the backend's bearer key.
            .and(wiremock::matchers::header("authorization", "Bearer secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "1", "model": "gpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {}
            })))
            .mount(&server)
            .await;

        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static("abc"));
        headers.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-key"),
        );

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]});
        // The allowed request id survives and caller authorization cannot replace
        // the backend's credential.
        client
            .call_rewrite_model_raw(
                raw,
                Some(headers),
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        Ok(())
    }

    // -------------------------------------------------------------------------------------
    // Provenance at the dispatch layer.
    //
    // The encoder tests prove what happens once a policy carries an admission. These prove the
    // runner produces that admission from the registry, the request's own lineage and the
    // target's canonical identity -- and that the active continuation edge is what decides
    // between "strip" and "refuse".
    // -------------------------------------------------------------------------------------

    fn reasoning_details() -> Value {
        json!([
            {"type": "reasoning.text", "text": "consider wavelengths", "index": 0},
            {"type": "reasoning.encrypted", "data": "opaque-blob-7f3a", "index": 0,
             "signature": "sig-issuer-only"}
        ])
    }

    fn reasoning_digest() -> String {
        let details = reasoning_details();
        opaque_digest(
            details.as_array().expect("array"),
            Some("sig-issuer-only"),
        )
    }

    fn reasoning_message() -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Reasoning {
                text: "consider wavelengths".to_string(),
                signature: Some("sig-issuer-only".to_string()),
                details: reasoning_details().as_array().expect("array").clone(),
            }],
        }
    }

    fn metadatum(agent: Option<&str>, session: Option<&str>) -> Metadata {
        Metadata {
            agent_id: agent.map(ToOwned::to_owned),
            session_id: session.map(ToOwned::to_owned),
            ..Metadata::default()
        }
    }

    fn backend_with_key(base_url: &str, api_key: &str) -> Backend {
        let mut config = config(base_url);
        config.api_key = Some(api_key.to_string());
        Backend::OpenAiChat(config)
    }

    fn openrouter_backend() -> Backend {
        let mut backend = backend_with_key("https://openrouter.ai/api/v1", "key-a");
        if let Backend::OpenAiChat(config) = &mut backend {
            config.reasoning_dialect = ReasoningDialect::OpenRouterEnabled;
        }
        backend
    }

    fn model() -> ModelId {
        ModelId::from("deepseek/deepseek-v4.1-flash-1017")
    }

    // Two accounts on one wire are two issuers: the credential is part of the identity even
    // though the base URL is not. The key itself must never appear in the identifier.
    #[test]
    fn credential_domain_separates_accounts_on_the_same_wire() {
        let first = backend_with_key("https://openrouter.ai/api/v1", "key-a");
        let second = backend_with_key("https://openrouter.ai/api/v1", "key-b");
        assert_ne!(
            credential_domain(backend_config(&first)),
            credential_domain(backend_config(&second)),
            "two keys on one base URL are two credential domains"
        );

        let repeat = backend_with_key("https://openrouter.ai/api/v1", "key-a");
        assert_eq!(
            credential_domain(backend_config(&first)),
            credential_domain(backend_config(&repeat)),
            "the same key and URL are the same domain"
        );

        // A different endpoint on the same provider is also a different domain.
        let other_host = backend_with_key("https://api.deepseek.com/v1", "key-a");
        assert_ne!(
            credential_domain(backend_config(&first)),
            credential_domain(backend_config(&other_host))
        );

        let issuer = reasoning_issuer(&first, &model());
        assert!(
            !issuer.client.contains("key-a") && !issuer.client.contains("key-b"),
            "the identifier must not carry the credential: {}",
            issuer.client
        );
        assert!(
            issuer.client.starts_with("openrouter.ai#"),
            "the identifier names the endpoint then the domain: {}",
            issuer.client
        );
    }

    // A blob recorded from this exact target is admitted when the request is dispatched back to
    // that same issuer. The issuer (credential domain, model, wire) is the whole of the proof;
    // no caller-supplied header participates.
    #[test]
    fn admission_admits_state_this_issuer_recorded() {
        let backend = openrouter_backend();
        let registry = ProvenanceRegistry::default();
        let metadata = metadatum(Some("agent-7"), Some("conv-1"));

        // Record by decoding a response that carries the state.
        let response = AggLlmResponse {
            id: Some("resp-9".to_string()),
            model: None,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Reasoning {
                    text: "consider wavelengths".to_string(),
                    signature: Some("sig-issuer-only".to_string()),
                    details: reasoning_details().as_array().expect("array").clone(),
                }],
                url_citations: Vec::new(),
                stop_reason: None,
            }],
            usage: Usage::default(),
            extensions: ProviderExtensions::default(),
            preservation: PreservationMetadata::default(),
        };
        record_reasoning_provenance(&response, &backend, &model(), Some(&metadata), &registry);

        let request = LlmRequest {
            messages: vec![Message::text(Role::User, "hi"), reasoning_message()],
            ..LlmRequest::default()
        };
        let mut policy = translation_policy(&backend);
        apply_reasoning_admission(
            &mut policy,
            &backend,
            &model(),
            &request,
            Some(&metadata),
            &registry,
        );
        let admission = policy.reasoning_admission.expect("provenance-bound sets one");
        // The recorded lineage is untrusted (no verified caller boundary), and the assistant
        // message is the active continuation edge, so admission fails closed: unproven active
        // state is refused rather than replayed.
        assert!(
            !admission.admits(&reasoning_digest()),
            "untrusted active state must not be replayable"
        );
        assert!(
            admission.active_unproven.contains(&reasoning_digest()),
            "the active digest is recorded as unproven"
        );
    }

    // Ingress-to-egress trust boundary. Switchyard has no inbound authentication, so `agent_id`
    // and `session_id` are advisory headers a caller can forge. Without a verified caller
    // boundary the recorded lineage is untrusted, and untrusted active state fails closed: a
    // spoofed, absent, or mismatched caller identity must NOT authorise replay of the active
    // continuation edge. This is the regression that fails if header matching is (re)introduced
    // as if it were auth, or if the untrusted path is made permissive again.
    #[test]
    fn advisory_headers_do_not_change_admission() {
        let backend = openrouter_backend();
        let registry = ProvenanceRegistry::default();
        let recorded = metadatum(Some("agent-7"), Some("conv-1"));
        let response = AggLlmResponse {
            id: Some("resp-9".to_string()),
            model: None,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Reasoning {
                    text: "consider wavelengths".to_string(),
                    signature: Some("sig-issuer-only".to_string()),
                    details: reasoning_details().as_array().expect("array").clone(),
                }],
                url_citations: Vec::new(),
                stop_reason: None,
            }],
            usage: Usage::default(),
            extensions: ProviderExtensions::default(),
            preservation: PreservationMetadata::default(),
        };
        record_reasoning_provenance(&response, &backend, &model(), Some(&recorded), &registry);

        let request = LlmRequest {
            messages: vec![Message::text(Role::User, "hi"), reasoning_message()],
            ..LlmRequest::default()
        };

        // Same bytes, same issuer, but every caller-supplied identity is different or absent.
        for (label, metadata) in [
            ("a spoofed consumer", metadatum(Some("attacker"), Some("conv-1"))),
            ("another conversation", metadatum(Some("agent-7"), Some("conv-2"))),
            ("absent ids", metadatum(None, None)),
        ] {
            let mut policy = translation_policy(&backend);
            apply_reasoning_admission(
                &mut policy,
                &backend,
                &model(),
                &request,
                Some(&metadata),
                &registry,
            );
            let admission = policy.reasoning_admission.expect("admission");
            assert!(
                !admission.admits(&reasoning_digest()),
                "{label} must not authorise replay: untrusted active state fails closed"
            );
            assert!(
                admission.active_unproven.contains(&reasoning_digest()),
                "{label}: the active digest is unproven"
            );
        }
    }

    // The same opaque bytes on a different issuer (another account or another native model) are
    // refused even though the caller-supplied conversation is identical. This is the
    // "same digest, different scope" case: identity comes from the credential and the model, not
    // from anything the caller sends.
    #[test]
    fn same_digest_on_another_issuer_is_refused() {
        let registry = ProvenanceRegistry::default();
        let metadata = metadatum(Some("agent-7"), Some("conv-1"));
        let issuer = openrouter_backend();
        let response = AggLlmResponse {
            id: Some("resp-9".to_string()),
            model: None,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Reasoning {
                    text: "consider wavelengths".to_string(),
                    signature: Some("sig-issuer-only".to_string()),
                    details: reasoning_details().as_array().expect("array").clone(),
                }],
                url_citations: Vec::new(),
                stop_reason: None,
            }],
            usage: Usage::default(),
            extensions: ProviderExtensions::default(),
            preservation: PreservationMetadata::default(),
        };
        record_reasoning_provenance(&response, &issuer, &model(), Some(&metadata), &registry);

        let request = LlmRequest {
            messages: vec![Message::text(Role::User, "hi"), reasoning_message()],
            ..LlmRequest::default()
        };

        let other_account = {
            let mut backend = backend_with_key("https://openrouter.ai/api/v1", "key-b");
            if let Backend::OpenAiChat(config) = &mut backend {
                config.reasoning_dialect = ReasoningDialect::OpenRouterEnabled;
            }
            backend
        };
        let other_model = ModelId::from("deepseek/deepseek-v4-pro");

        for (label, backend, target) in [
            ("another account", &other_account, model()),
            ("another model", &issuer, other_model),
        ] {
            let mut policy = translation_policy(backend);
            apply_reasoning_admission(
                &mut policy,
                backend,
                &target,
                &request,
                Some(&metadata),
                &registry,
            );
            let admission = policy.reasoning_admission.expect("admission");
            assert!(
                !admission.admits(&reasoning_digest()),
                "{label} must not replay another issuer's state"
            );
            assert!(
                admission.refuses(&reasoning_digest()),
                "{label} lands on the active edge as required-and-unproven"
            );
        }
    }

    // Lineage for the response id is recorded too, so the fact is available rather than merely
    // implied: the recorded entry carries the provider response id it arrived with.
    #[test]
    fn recorded_lineage_carries_the_provider_response_id() {
        let backend = openrouter_backend();
        let registry = ProvenanceRegistry::default();
        let metadata = metadatum(Some("agent-7"), Some("conv-1"));
        let response = AggLlmResponse {
            id: Some("resp-9".to_string()),
            model: None,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Reasoning {
                    text: "t".to_string(),
                    signature: Some("sig-issuer-only".to_string()),
                    details: reasoning_details().as_array().expect("array").clone(),
                }],
                url_citations: Vec::new(),
                stop_reason: None,
            }],
            usage: Usage::default(),
            extensions: ProviderExtensions::default(),
            preservation: PreservationMetadata::default(),
        };
        record_reasoning_provenance(&response, &backend, &model(), Some(&metadata), &registry);

        let (recorded, lineage) = registry
            .resolve(&reasoning_digest())
            .expect("the decoded state is recorded");
        assert_eq!(lineage.advisory_consumer(), Some("agent-7"));
        assert_eq!(lineage.advisory_conversation(), Some("conv-1"));
        assert_eq!(lineage.response.as_deref(), Some("resp-9"));
        assert_eq!(recorded, reasoning_issuer(&backend, &model()));
    }

    // Tool lineage decides the edge: the assistant turn that issued the state is active while
    // only tool results follow it, and becomes completed history once a user turn follows.
    #[test]
    fn active_edge_follows_tool_lineage() {
        let tool_result = Message {
            role: Role::Tool,
            content: vec![ContentBlock::Text {
                text: "port = 8080".to_string(),
            }],
        };
        let user_turn = Message::text(Role::User, "and now?");

        let in_flight = LlmRequest {
            messages: vec![
                Message::text(Role::User, "read the config"),
                reasoning_message(),
                tool_result.clone(),
            ],
            ..LlmRequest::default()
        };
        assert_eq!(
            active_message_indexes(&in_flight),
            BTreeSet::from([1]),
            "an assistant turn followed only by tool results is the continuation"
        );

        let completed = LlmRequest {
            messages: vec![
                Message::text(Role::User, "read the config"),
                reasoning_message(),
                tool_result,
                user_turn,
            ],
            ..LlmRequest::default()
        };
        assert!(
            active_message_indexes(&completed).is_empty(),
            "once a user turn follows, the assistant turn is completed history"
        );
    }

    // The native path is marked too: `PlaintextOnly` may never replay, but required active state
    // is pinned as unproven so the encoder refuses instead of stripping it.
    #[test]
    fn plaintext_only_runner_marks_active_state_as_required() {
        let backend = backend_with_key("https://api.deepseek.com/v1", "key-a");
        let registry = ProvenanceRegistry::default();
        let request = LlmRequest {
            messages: vec![
                Message::text(Role::User, "read the config"),
                reasoning_message(),
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::Text {
                        text: "port = 8080".to_string(),
                    }],
                },
            ],
            ..LlmRequest::default()
        };

        let mut policy = translation_policy(&backend);
        assert_eq!(
            policy.reasoning_details_replay,
            ReasoningDetailsReplay::PlaintextOnly,
            "a native dialect has no structured reasoning wire"
        );
        apply_reasoning_admission(&mut policy, &backend, &model(), &request, None, &registry);
        let admission = policy.reasoning_admission.expect("native paths are marked too");
        assert!(
            admission.refuses(&reasoning_digest()),
            "required active state is refused rather than silently stripped"
        );
        assert!(
            admission.admitted.is_empty(),
            "a native wire can never admit opaque state"
        );
    }

    // Streaming: the tap folds the same accumulator the consumer receives, so what is recorded
    // is the assembled state, not a partial frame.
    #[tokio::test]
    async fn streamed_reasoning_is_recorded_from_the_assembled_response() {
        let registry = Arc::new(ProvenanceRegistry::default());
        let backend = openrouter_backend();
        let metadata = metadatum(Some("agent-7"), Some("conv-1"));

        // The details arrive as a stream delta, exactly as a provider sends them: the
        // accumulator is what turns them into the assembled block the caller receives.
        let details = reasoning_details().as_array().expect("array").clone();
        let events: Vec<std::result::Result<LlmResponseStreamEvent, LlmClientError>> = vec![
            Ok(LlmResponseStreamEvent::from(
                LlmResponseChunk::MessageStart {
                    id: Some("resp-9".to_string()),
                    model: None,
                },
            )),
            Ok(LlmResponseStreamEvent::from(
                LlmResponseChunk::ReasoningDetailsDelta {
                    index: 0,
                    details,
                    text: "consider wavelengths".to_string(),
                },
            )),
        ];
        let stream: LlmResponseStream = Box::pin(stream::iter(events));

        let tapped = tap_streamed_provenance(
            stream,
            Arc::clone(&registry),
            reasoning_issuer(&backend, &model()),
            request_lineage(Some(&metadata)),
        );
        let drained: Vec<_> = tapped.collect().await;
        assert!(
            drained.iter().all(|event| event.is_ok()),
            "the tap must not alter the stream"
        );

        let (recorded, lineage) = registry
            .resolve(&reasoning_digest())
            .expect("streamed state is recorded once the stream completes");
        assert_eq!(recorded, reasoning_issuer(&backend, &model()));
        assert_eq!(lineage.advisory_consumer(), Some("agent-7"));
    }

}