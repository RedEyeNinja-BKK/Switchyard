// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`TranslatingLlmClient`] — the crate's single public entry point: encode a neutral
//! request, call the configured backend over HTTP, decode the neutral response.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use http::StatusCode;
use reqwest::RequestBuilder;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::{Map, Value, json};
use switchyard_protocol::{
    LlmRequest, LlmResponse, LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, Metadata,
    ModelId, Request, Response, RoutedLlmClient,
};
use switchyard_translation::{
    TargetCapabilities, TranslationError, TranslationPolicy, WireFormat,
    decode_aggregated_response, decode_request, decode_stream,
    encode_aggregated_response_with_extensions, encode_request_with_policy,
    encode_stream_with_extensions,
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
        })
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

        // A Responses upstream may declare that it requires a list-shaped `input`; the
        // default policy carries no target capabilities, so it is supplied here.
        let policy = TranslationPolicy {
            target_capabilities: TargetCapabilities {
                requires_list_input: Some(backend.requires_list_input()),
                ..TargetCapabilities::default()
            },
            ..TranslationPolicy::default()
        };
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
            && is_transient_failure(error)
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
                    let will_retry = attempt < max_retries && failure.is_retryable();
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
            metrics::record_upstream_attempt(Some(status.as_u16()));
            return Ok(EncodedResponse::Buffered {
                status: status.as_u16(),
                body: body.to_vec(),
                upstream_headers,
            });
        }

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
            } => (LlmResponse::Stream(chunks), upstream_headers),
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
                let agg =
                    decode_aggregated_response(&body, wire_format).map_err(
                        |error| match error {
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
                        },
                    )?;
                (LlmResponse::Agg(agg), upstream_headers)
            }
        };

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
pub(crate) fn is_account_wide_quota(source: &LlmClientError) -> bool {
    let LlmClientError::UpstreamHttp { status, body } = source else {
        return false;
    };
    let limit_source = serde_json::from_str::<serde_json::Value>(body).ok().and_then(|value| {
        value["error"]["metadata"]["limit_source"]
            .as_str()
            .map(str::to_string)
    });
    match *status {
        StatusCode::TOO_MANY_REQUESTS => {
            limit_source.is_some_and(|s| s == "openrouter_free_tier_daily")
        }
        // Insufficient credits: the account balance is the shared resource, so no
        // candidate on this credential can serve the request.
        StatusCode::PAYMENT_REQUIRED => true,
        // A 403 with a budget limit_source is an account cap. A 403 without one is a
        // guardrail or policy refusal and is deliberately NOT included here.
        StatusCode::FORBIDDEN => limit_source.is_some(),
        _ => false,
    }
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
        LlmClientError::UpstreamHttp { status, .. } => {
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
    let mut chunks = decode_stream(bytes, backend.wire_format())?;
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::JoinHandle;

    use serde_json::json;
    use switchyard_protocol::{completion_text, text_request};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::backend::HttpBackendConfig;
    use switchyard_protocol::{ReasoningDialect, ReasoningPolicy};

    fn config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: Some("secret".to_string()),
            forward_auth: false,
            extra_headers: BTreeMap::new(),
            extra_body: BTreeMap::new(),
            omit_body_fields: BTreeSet::new(),
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

    /// A daily account cap is not a throttle: retrying it immediately cannot succeed.
    ///
    /// The control matters as much as the rule. A 429 from a provider's shared pool
    /// clears in seconds and must keep its retries, and so must a 429 carrying no
    /// marker at all, or this test would pass while quietly removing recovery that
    /// works today.
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
}
