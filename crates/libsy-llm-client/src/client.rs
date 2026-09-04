// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`TranslatingLlmClient`] — the crate's single public entry point: encode a neutral
//! request, call the configured backend over HTTP, decode the neutral response.

use std::collections::{BTreeMap, HashMap};
use std::future::ready;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use http::StatusCode;
use reqwest::RequestBuilder;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::{Map, Value};
use switchyard_protocol::{
    LlmRequest, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Metadata, ModelId, Request,
    Response, RoutedLlmClient,
};
use switchyard_translation::{
    WireFormat, decode_aggregated_response, decode_request, decode_stream,
    encode_aggregated_response_with_extensions, encode_request_with_profile,
    encode_stream_with_extensions,
};
use tracing::Instrument;

use crate::backend::Backend;
use crate::error::is_permanent_quota_429;
use crate::error::{LlmClientError, Result};
use crate::metrics;
use crate::raw::RawResponse;

// Headers this client owns or that are hop-by-hop. Backends apply an explicitly
// enabled caller credential after generic metadata forwarding skips these.
// Azure/OpenAI credentials and tenant selectors are also backend-owned.
const RESERVED_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "connection",
    "authorization",
    "proxy-authorization",
    "proxy-authenticate",
    "cookie",
    "set-cookie",
    "x-api-key",
    "chatgpt-account-id",
    "x-openai-fedramp",
    "api-key",
    "openai-organization",
    "openai-project",
    "anthropic-beta",
    "anthropic-version",
    "content-type",
    "accept-encoding",
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
    /// Exact input-token count against an OpenAI-chat backend's
    /// `/chat/completions/input_tokens` endpoint (llama.cpp-style).
    OpenAiChatInputTokens,
    /// OpenAI Responses compaction.
    ResponsesCompact,
}

impl AuxiliaryOperation {
    const fn wire_format(self) -> WireFormat {
        match self {
            Self::AnthropicCountTokens => WireFormat::AnthropicMessages,
            Self::ResponsesInputTokens | Self::ResponsesCompact => WireFormat::OpenAiResponses,
            Self::OpenAiChatInputTokens => WireFormat::OpenAiChat,
        }
    }

    fn url(self, backend: &Backend) -> String {
        match self {
            Self::AnthropicCountTokens => backend.count_tokens_url(),
            Self::ResponsesInputTokens => format!("{}/input_tokens", backend.url()),
            Self::OpenAiChatInputTokens => backend.input_tokens_url(),
            Self::ResponsesCompact => format!("{}/compact", backend.url()),
        }
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
}

impl TranslatingLlmClient {
    /// Builds a client over the given [`ModelConfig`]s, with a fresh shared HTTP
    /// client and the built-in translation codecs.
    pub fn new(model_configs: &[ModelConfig]) -> Result<Self> {
        for config in model_configs {
            config
                .default_backend
                .validate_extra_headers(&config.model_name)?;
            for backend in config.other_backends.iter().flatten() {
                backend.validate_extra_headers(&config.model_name)?;
            }
        }
        // Provider-identity invariant: `provider_key()` derives the fallback
        // driver's provider identity from ONE configured backend's base URL.
        // That is exact only while every backend in a client shares the same
        // canonical base URL (true by schema today: one scalar `base_url` per
        // `[llm_clients.*]`). Enforce it at construction so a future
        // multi-backend representation cannot silently undermine
        // provider-aware fallback skip decisions. Hermes qualification
        // finding N2 (2026-09-04).
        for config in model_configs {
            let mut backends: Vec<&Backend> = std::iter::once(&config.default_backend)
                .chain(config.other_backends.iter().flatten())
                .collect();
            backends.dedup_by(|a, b| a.provider_identity() == b.provider_identity());
            if backends.len() > 1 {
                let urls: Vec<String> = backends
                    .iter()
                    .map(|backend| backend.provider_identity())
                    .collect();
                return Err(LlmClientError::RequestEncoding(format!(
                    "client for model {} spans multiple provider identities \
                     ({urls:?}); provider-aware fallback requires one provider \
                     per [llm_clients.*] client",
                    config.model_name
                )));
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

    /// Exact input-token count against an OpenAI-chat backend's
    /// `/chat/completions/input_tokens` endpoint.
    ///
    /// The count reuses the shared [`send_encoded`](Self::send_encoded) path, so
    /// the counted body is the same token-relevant final target representation
    /// that generation would POST: resolved target model stamping, same-format
    /// preservation, target `extra_body`, metadata/backend headers, auth, and
    /// retry behavior all stay identical. Only a valid `{"input_tokens": N}`
    /// result is accepted; `N` must be a non-negative integer representable in
    /// [`u64`].
    ///
    /// Returns an error when the model has no OpenAI-chat backend, the upstream
    /// request fails, or the response is not a valid `{"input_tokens": N}`.
    pub async fn count_input_tokens(&self, model: &ModelId, request: &Request) -> Result<u64> {
        let value = self
            .call_auxiliary(
                model,
                request.clone(),
                AuxiliaryOperation::OpenAiChatInputTokens,
            )
            .await?;
        let input_tokens =
            value
                .get("input_tokens")
                .ok_or_else(|| LlmClientError::InvalidResponse {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "input_tokens response missing `input_tokens` field",
                    )),
                })?;
        input_tokens
            .as_u64()
            .ok_or_else(|| LlmClientError::InvalidResponse {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "input_tokens response field is not a non-negative integer",
                )),
            })
    }

    /// Encode `llm_request` for `wire_format`, POST it to `url` with the request's
    /// forwarded headers plus the backend's static headers and auth, and return the
    /// successful upstream response. A
    /// buffered response is fully collected within the retry boundary; a streamed
    /// response is returned as soon as its successful headers arrive. A non-success
    /// status maps to a typed error — a 400 is classified as a context-window
    /// overflow via the backend's provider rules. Shared by
    /// [`call_rewrite_model`](Self::call_rewrite_model) (which POSTs to the
    /// backend's completion URL and decodes a response) and
    /// the model-bearing auxiliary operations, which return raw JSON.
    async fn send_encoded(
        &self,
        backend: &Backend,
        wire_format: WireFormat,
        llm_request: LlmRequest,
        metadata: Option<&Metadata>,
        model: &ModelId,
        endpoint: UpstreamEndpoint,
    ) -> Result<EncodedResponse> {
        // The destination profile is a property of the backend class (the same
        // path-boundary rule that classifies strict Codex endpoints), not of
        // individual routes: a strict Codex backend encodes assistant history
        // as output_text; normal Responses endpoints stay standards-compliant.
        let responses_profile =
            if matches!(wire_format, WireFormat::OpenAiResponses) && backend.is_codex() {
                switchyard_translation::ResponsesProfile::StrictCodex
            } else {
                switchyard_translation::ResponsesProfile::Normal
            };
        let mut body = encode_request_with_profile(&llm_request, wire_format, responses_profile)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?;
        // `encode_request` round-trips a preserved same-format body verbatim,
        // which keeps the caller's original `model`; force the resolved model so
        // the upstream always sees the target id.
        set_json_model(&mut body, model);
        // Strip before `merge_extra_body` so a target can reinstate either field
        // deliberately via `extra_body`.
        if matches!(backend, Backend::Anthropic(_)) {
            strip_anthropic_incompatible_fields(&mut body);
            strip_unsigned_thinking_blocks(&mut body);
        }
        merge_extra_body(&mut body, backend.extra_body());
        // The ChatGPT Codex backend rejects `max_output_tokens` outright (2026-08
        // API), on every inbound path - chat (`max_tokens`), Responses passthrough,
        // and preserved same-format bodies alike. Strip AFTER `merge_extra_body`
        // so no target `extra_body` can reinstate it. Normal OpenAI
        // `/v1/responses` backends keep the parameter.
        if backend.is_codex() {
            strip_codex_incompatible_fields(&mut body);
        }
        if matches!(backend, Backend::Anthropic(_)) {
            enable_anthropic_prompt_caching(&mut body);
        }
        // Target-pinned reasoning policy is canonical on both OpenAI leg formats:
        // when the target pins a nested `reasoning` policy (e.g. an NT lane's
        // `reasoning = { effort = "none" }` or `enabled = false` pin via
        // extra_body), it overrides any caller-supplied reasoning representation.
        // merge_extra_body only fills absent keys (or_insert), so the target pin
        // must be re-asserted here; without it, a caller's own `reasoning` object
        // would silently displace the target's policy.
        if matches!(backend, Backend::OpenAiChat(_)) {
            canonicalize_reasoning_effort(&mut body, backend.extra_body().get("reasoning"));
            ensure_openai_stream_usage(&mut body);
        } else if matches!(backend, Backend::OpenAiResponses(_)) {
            canonicalize_reasoning_effort(&mut body, backend.extra_body().get("reasoning"));
            // Stream-mandatory Responses backends (chatgpt.com Codex: "Stream
            // must be set to true") - force upstream stream=true regardless of
            // the caller's stream value; the server layer aggregates back to
            // buffered JSON for callers that asked for stream=false. The chat-
            // only usage decoration above never leaks stream_options onto this
            // Responses path (stream_options is a chat-completions parameter).
            if backend.is_codex()
                && body.get("stream").and_then(Value::as_bool) != Some(true)
                && let Value::Object(object) = &mut body
            {
                object.insert("stream".to_string(), Value::Bool(true));
            }
        }
        let streaming = endpoint.allows_streaming()
            && body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let url = endpoint.url(backend);
        record_gen_ai_request(&url, model, streaming);

        let max_retries = u64::from(backend.max_retries());
        let max_attempts = max_retries + 1;
        let mut attempt = 0_u64;
        loop {
            let span = tracing::debug_span!(
                target: "libsy",
                "libsy.upstream_attempt",
                model = %model,
                wire_format = %wire_format,
                attempt = attempt + 1,
                max_attempts,
                retry = attempt > 0,
                openinference.span.kind = "CHAIN",
                outcome = tracing::field::Empty,
                status_code = tracing::field::Empty,
                will_retry = tracing::field::Empty,
                retry_delay_ms = tracing::field::Empty,
            );
            let result = self
                .send_once(&url, backend, &body, metadata, model, streaming)
                .instrument(span.clone())
                .await;
            // The retained handle updates this same attempt span with its outcome.
            match result {
                Ok(response) => {
                    span.record("outcome", "ok");
                    span.record("status_code", response.status());
                    span.record("will_retry", false);
                    if attempt > 0 {
                        metrics::record_retry_recovered();
                    }
                    return Ok(response);
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
                    tokio::time::sleep(delay).await;
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

        // One shared per-attempt deadline covering connect/send AND the first
        // body byte: a hung upstream must fail over to the next candidate
        // instead of stalling the downstream consumer until its idle timeout.
        let deadline = tokio::time::Instant::now() + UPSTREAM_FIRST_BYTE_TIMEOUT;
        let response = match tokio::time::timeout_at(deadline, builder.send()).await {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                metrics::record_upstream_attempt(None);
                return Err(AttemptFailure {
                    error: convert_reqwest_error(error),
                    status: None,
                    retry_after: None,
                    deadline_elapsed: false,
                });
            }
            Err(_) => {
                metrics::record_upstream_attempt(None);
                return Err(AttemptFailure {
                    error: LlmClientError::Timeout {
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!(
                                "upstream did not deliver response headers within {}s",
                                UPSTREAM_FIRST_BYTE_TIMEOUT.as_secs()
                            ),
                        )),
                    },
                    status: None,
                    retry_after: None,
                    deadline_elapsed: true,
                });
            }
        };
        let status = response.status();
        if status.is_success() {
            if streaming {
                // Do not commit this candidate until the upstream proves it will
                // actually deliver body bytes: a hung upstream (200 headers, body
                // never starts) must fall through to the next candidate rather
                // than stall the downstream consumer until its idle timeout.
                let mut stream: UpstreamByteStream = Box::pin(response.bytes_stream());
                let first_chunk = match tokio::time::timeout_at(deadline, stream.as_mut().next())
                    .await
                {
                    Ok(Some(Ok(bytes))) => Some(bytes.to_vec()),
                    Ok(Some(Err(error))) => {
                        metrics::record_upstream_attempt(None);
                        return Err(AttemptFailure {
                            error: convert_reqwest_error(error),
                            status: Some(status),
                            retry_after: None,
                            deadline_elapsed: false,
                        });
                    }
                    Ok(None) => {
                        metrics::record_upstream_attempt(None);
                        return Err(AttemptFailure {
                            error: LlmClientError::Transport {
                                source: Box::new(std::io::Error::new(
                                    std::io::ErrorKind::UnexpectedEof,
                                    "upstream closed the stream before sending any body bytes",
                                )),
                            },
                            status: Some(status),
                            retry_after: None,
                            deadline_elapsed: false,
                        });
                    }
                    Err(_) => {
                        metrics::record_upstream_attempt(None);
                        return Err(AttemptFailure {
                            error: LlmClientError::Timeout {
                                source: Box::new(std::io::Error::new(
                                    std::io::ErrorKind::TimedOut,
                                    format!(
                                        "upstream sent no body bytes within {}s of the attempt deadline",
                                        UPSTREAM_FIRST_BYTE_TIMEOUT.as_secs()
                                    ),
                                )),
                            },
                            status: Some(status),
                            retry_after: None,
                            deadline_elapsed: true,
                        });
                    }
                };
                metrics::record_upstream_attempt(Some(status.as_u16()));
                return Ok(EncodedResponse::Streaming {
                    status: status.as_u16(),
                    first_chunk,
                    stream,
                });
            }
            let body = match response.bytes().await {
                Ok(body) => body,
                Err(error) => {
                    metrics::record_upstream_attempt(None);
                    return Err(AttemptFailure {
                        error: convert_reqwest_error(error),
                        status: Some(status),
                        retry_after: None,
                        deadline_elapsed: false,
                    });
                }
            };
            metrics::record_upstream_attempt(Some(status.as_u16()));
            return Ok(EncodedResponse::Buffered {
                status: status.as_u16(),
                body: body.to_vec(),
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
                    deadline_elapsed: false,
                });
            }
        };
        let body = backend.redact_forwarded_auth(body, metadata);
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
            deadline_elapsed: false,
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
            )
            .await?;

        let llm_response = match http_response {
            EncodedResponse::Streaming {
                first_chunk,
                stream,
                ..
            } => {
                // Adapt the reqwest body stream to plain bytes; the SSE-decode itself is
                // transport-agnostic and lives in `switchyard-translation`.
                // Replay the first chunk awaited at the candidate boundary ahead of
                // the live stream.
                let prefix =
                    futures_util::stream::iter(first_chunk.map(|chunk| {
                        Ok::<bytes::Bytes, reqwest::Error>(bytes::Bytes::from(chunk))
                    }));
                let bytes = prefix.chain(stream).map(|chunk| {
                    chunk.map(|bytes| bytes.to_vec()).map_err(|error| {
                        if error.is_timeout() {
                            LlmClientError::Timeout {
                                source: Box::new(error),
                            }
                        } else {
                            LlmClientError::Transport {
                                source: Box::new(error),
                            }
                        }
                    })
                });
                let mut chunks = decode_stream(bytes, wire_format)?;
                // Providers reject an over-ceiling streaming request with an in-band
                // error event on an HTTP 200. Classify the first event before returning
                // the stream: nothing has reached the caller yet, so an overflow can
                // still fail the call and let routing try the next candidate.
                match chunks.next().await {
                    None => LlmResponse::Stream(stream::empty().boxed()),
                    Some(first) => {
                        if let Some(message) = first_event_overflow(&first, backend) {
                            return Err(LlmClientError::ContextWindowExceeded {
                                model: model_id.clone(),
                                message,
                            });
                        }
                        LlmResponse::Stream(stream::once(ready(first)).chain(chunks).boxed())
                    }
                }
            }
            EncodedResponse::Buffered { body, .. } => {
                let body = serde_json::from_slice::<Value>(&body).map_err(|error| {
                    LlmClientError::ResponseTranslation(format!("invalid upstream JSON: {error}"))
                })?;
                let agg = decode_aggregated_response(&body, wire_format)
                    .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
                LlmResponse::Agg(agg)
            }
        };

        Ok(Response {
            llm_response,
            metadata,
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
    /// [`Metadata::http_headers`] and forwarded to the upstream (minus the reserved
    /// set); pass `None` to forward nothing.
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
            candidate_input_tokens: Default::default(),
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

    fn provider_key(&self) -> Option<&str> {
        // One client is built per `[llm_clients.*]` section, so every model it
        // serves shares the same upstream base URL. The base URL is the provider
        // identity used by the fallback driver to skip whole providers that are
        // proven unavailable (drained balance / auth): same-provider clients use
        // the identical base_url string (config convention).
        //
        // IDENTITY IS EXACT, NOT HEURISTIC, for any configuration the schema
        // can express: `LlmClientConfig` carries a single scalar `base_url`
        // (crates/switchyard-runner/src/config.rs), so one client cannot span
        // two upstream URLs and all `model_to_config` backends within a client
        // share the same provider key by construction. Live-topology proof
        // (GO-D qualification 2026-09-04): 20 clients -> 6 distinct URLs, and
        // every URL-sharing group is one physical upstream service
        // (deepseek x3, chatgpt-codex x3, htpc x3, openrouter x5,
        // reninja llama.cpp profiles x5, thaillm x1).
        self.model_to_config
            .values()
            .next()
            .map(|config| match &config.default_backend {
                Backend::OpenAiChat(backend)
                | Backend::OpenAiResponses(backend)
                | Backend::Anthropic(backend) => backend.base_url.trim_end_matches('/'),
            })
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
    },
    Streaming {
        status: u16,
        /// First body byte already awaited at the candidate boundary (see
        /// [`UPSTREAM_FIRST_BYTE_TIMEOUT`]); replayed ahead of the live stream.
        first_chunk: Option<Vec<u8>>,
        stream: UpstreamByteStream,
    },
}

/// Type-erased upstream SSE byte stream.
type UpstreamByteStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// Maximum time for one upstream attempt to prove liveness: response headers
/// received AND the first body byte delivered. Past the deadline the candidate
/// fails and the next candidate runs.
///
/// A hung upstream (connect stall, or 200 headers with a body that never
/// starts) must fail over instead of stalling the downstream consumer until its
/// own idle timeout. Generous enough for slow first tokens on large prompts;
/// short enough that one failover attempt still completes inside a typical
/// downstream ~300s stream-idle budget.
const UPSTREAM_FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(45);

impl EncodedResponse {
    fn status(&self) -> u16 {
        match self {
            EncodedResponse::Buffered { status, .. } => *status,
            EncodedResponse::Streaming { status, .. } => *status,
        }
    }
}

// The typed error decides retry eligibility; status and Retry-After feed
// attempt telemetry and delay selection.
struct AttemptFailure {
    error: LlmClientError,
    status: Option<StatusCode>,
    retry_after: Option<Duration>,
    /// Set when the per-attempt liveness deadline elapsed (connect/send stall or
    /// no first body byte). The upstream already consumed its whole budget, so a
    /// same-candidate retry cannot help — fail over to the next candidate.
    deadline_elapsed: bool,
}

impl AttemptFailure {
    fn is_retryable(&self) -> bool {
        if self.deadline_elapsed {
            return false;
        }
        match &self.error {
            LlmClientError::Transport { .. } | LlmClientError::Timeout { .. } => true,
            LlmClientError::UpstreamHttp { status, body } => {
                // A 429 that is a PERMANENT quota exhaustion (OpenAI
                // `insufficient_quota` / exhausted usage-billing quota) cannot
                // be fixed by retrying the same candidate: skip the ordinary
                // 429 retry budget so the request advances immediately to the
                // next fleet_router candidate. Genuinely transient 429s
                // (`rate_limit_exceeded`, Retry-After) keep the bounded retry.
                if *status == StatusCode::TOO_MANY_REQUESTS && is_permanent_quota_429(body) {
                    return false;
                }
                metrics::is_retryable_http_status(status.as_u16())
            }
            _ => false,
        }
    }
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

// Forwards caller-supplied metadata headers except credentials, client-owned
// headers, and headers the backend overrides via `extra_headers`.
//
// A backend that configures a canonical header (e.g. ThaiLLM's `User-Agent` to a
// fixed value the upstream WAF requires) must win exactly once: suppressing any
// inbound header whose name matches an `extra_headers` key here means the backend
// value applied later by `apply_extra_headers` is not duplicated or shadowed by a
// caller-supplied one. Comparison is case-insensitive; auth/credential headers are
// already reserved and never forwarded.
fn forward_metadata_headers(
    mut builder: RequestBuilder,
    metadata: Option<&Metadata>,
    backend: &Backend,
) -> RequestBuilder {
    let Some(headers) = metadata.and_then(|metadata| metadata.http_headers.as_ref()) else {
        return builder;
    };
    for (name, value) in headers {
        if is_reserved_header(name.as_str()) {
            continue;
        }
        // The backend overrides this header via extra_headers; do not forward the
        // inbound value so the backend's configured value is applied exactly once.
        if backend
            .extra_headers()
            .keys()
            .any(|k| k.eq_ignore_ascii_case(name.as_str()))
        {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder
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

// Applies target defaults without overriding fields supplied by the caller.
// Removes fields the ChatGPT Codex Responses backend rejects outright.
fn strip_codex_incompatible_fields(body: &mut Value) {
    if let Value::Object(object) = body {
        object.remove("max_output_tokens");
    }
}

fn merge_extra_body(body: &mut Value, extra_body: &BTreeMap<String, Value>) {
    let Value::Object(object) = body else {
        return;
    };
    for (key, value) in extra_body {
        object.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

// Enforces exactly ONE canonical reasoning representation per upstream request
// on OpenAI-family backends (OpenAI Chat and OpenAI Responses; Anthropic has a
// separate thinking contract), and makes the TARGET's reasoning pin
// (route-author policy via extra_body) authoritative over caller-supplied
// reasoning.
//
// Two rules, applied AFTER `merge_extra_body` (or_insert - caller values win -
// so the target pin must be re-asserted here):
//
// 1. Target reasoning pin (e.g. `reasoning = { effort = "none" }` on an NT
//    lane, or `reasoning = { enabled = false }`): the target pin REPLACES the
//    caller's nested `reasoning` object (if any) and the flat `reasoning_effort`
//    is dropped. Without this, a caller-supplied `reasoning` object displaces
//    the target's policy through or_insert merge precedence, and an NT lane
//    would run reasoning its route author disabled.
// 2. No target pin: OpenRouter rejects the dual representation when the nested
//    `reasoning` object AND the caller's flat `reasoning_effort` ride along
//    together ("reasoning_effort and reasoning.effort are both provided with
//    conflicting values"). ANY object-valued `reasoning` is canonical - not
//    only one carrying a string `effort` (e.g. `{"enabled": false}` must also
//    suppress the flat field, or the conflicting dual representation ships).
fn canonicalize_reasoning_effort(body: &mut Value, reasoning_pin: Option<&Value>) {
    let Value::Object(object) = body else {
        return;
    };
    if let Some(pin) = reasoning_pin {
        // Rule 1: target policy pin is canonical - replaces any caller reasoning.
        object.insert("reasoning".to_string(), pin.clone());
        object.remove("reasoning_effort");
        return;
    }
    // Rule 2: no target pin - avoid the conflicting dual representation.
    // Any object-valued `reasoning` is canonical; the flat field is dropped
    // regardless of whether the object carries a string `effort`.
    if object.get("reasoning").and_then(Value::as_object).is_some() {
        object.remove("reasoning_effort");
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

// Case-insensitive membership test against RESERVED_HEADERS.
fn is_reserved_header(name: &str) -> bool {
    RESERVED_HEADERS
        .iter()
        .any(|reserved| name.eq_ignore_ascii_case(reserved))
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
    use switchyard_protocol::{LlmRequest, completion_text, text_request};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::backend::HttpBackendConfig;

    fn config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: Some("secret".to_string()),
            forward_auth: false,
            extra_headers: BTreeMap::new(),
            extra_body: BTreeMap::new(),
            max_retries: 0,
        }
    }

    fn config_with_retries(base_url: &str, max_retries: u32) -> HttpBackendConfig {
        HttpBackendConfig {
            max_retries,
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

    fn chat_map_with_extra_body(
        base_url: &str,
        extra_body: BTreeMap<String, Value>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.extra_body = extra_body;
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    // A one-model config list whose backend carries fixed `extra_headers`.
    fn chat_map_with_extra_headers(
        base_url: &str,
        extra_headers: BTreeMap<String, String>,
    ) -> Vec<ModelConfig> {
        let mut backend = config(base_url);
        backend.extra_headers = extra_headers;
        vec![ModelConfig::new("gpt", Backend::OpenAiChat(backend), None)]
    }

    fn anthropic_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "claude",
            Backend::Anthropic(config(base_url)),
            None,
        )]
    }

    // A one-model config list: "gpt" served over OpenAI Responses at base_url.
    fn responses_map(base_url: &str) -> Vec<ModelConfig> {
        vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(config(base_url)),
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
            candidate_input_tokens: Default::default(),
        }
    }

    #[test]
    fn anthropic_prompt_caching_marks_final_message() {
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

    #[test]
    fn retryable_error_classes_are_explicit() {
        let transport = AttemptFailure {
            error: LlmClientError::Transport {
                source: std::io::Error::other("disconnected").into(),
            },
            status: None,
            retry_after: None,
            deadline_elapsed: false,
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
                deadline_elapsed: false,
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
                deadline_elapsed: false,
            };
            assert!(!failure.is_retryable(), "HTTP {status} should fail fast");
        }

        let configuration = AttemptFailure {
            error: LlmClientError::Configuration {
                message: "invalid header".to_string(),
            },
            status: None,
            retry_after: None,
            deadline_elapsed: false,
        };
        assert!(!configuration.is_retryable());

        let context_window = AttemptFailure {
            error: LlmClientError::ContextWindowExceeded {
                model: ModelId::from("gpt"),
                message: "too long".to_string(),
            },
            status: Some(StatusCode::BAD_REQUEST),
            retry_after: None,
            deadline_elapsed: false,
        };
        assert!(!context_window.is_retryable());
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
    async fn forwards_metadata_headers_except_reserved()
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
            "accept-encoding",
            http::HeaderValue::from_static("gzip, br"),
        );
        headers.insert(
            "api-key",
            http::HeaderValue::from_static("client-azure-key"),
        );
        headers.insert(
            "openai-organization",
            http::HeaderValue::from_static("org-client"),
        );
        headers.insert(
            "openai-project",
            http::HeaderValue::from_static("proj-client"),
        );
        let request = Request {
            llm_request: LlmRequest {
                model: Some("gpt".to_string()),
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                session_id: None,
                agent_id: None,
                task_id: None,
                correlation_id: None,
                extra_metadata: None,
                http_headers: Some(headers),
                wire_format: None,
                ..Default::default()
            }),
            candidate_input_tokens: Default::default(),
        };

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;

        // Matchers assert forwarded x-request-id survives and reserved
        // authorization is the backend's, not the client's.
        client.call_rewrite_model(request, None).await?;
        let received = server
            .received_requests()
            .await
            .ok_or("request recording should be enabled")?;
        let received = received.first().ok_or("expected one upstream request")?;
        assert!(!received.headers.contains_key("accept-encoding"));
        assert!(!received.headers.contains_key("api-key"));
        assert!(!received.headers.contains_key("openai-organization"));
        assert!(!received.headers.contains_key("openai-project"));
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

    // Raw path forwards caller headers (minus the reserved set) to the upstream.
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
        // Matchers assert the forwarded x-request-id survives and reserved
        // authorization is the backend's, not the client's.
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
    // A target that pins its thinking policy as a nested `reasoning` object must
    // not be displaced by a caller-supplied nested `reasoning` object: the merge
    // only fills absent keys, so the target pin is re-asserted after it. An NT
    // lane must reliably remain non-thinking regardless of caller parameters.
    #[tokio::test]
    async fn target_reasoning_pin_overrides_caller_nested_reasoning()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use std::sync::Mutex as StdMutex;
        let server = MockServer::start().await;
        let seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let seen_for_assert = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_assert.lock().unwrap() = body.clone();
                body.pointer("/reasoning/effort").and_then(Value::as_str) == Some("none")
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

        let extra_body = BTreeMap::from([("reasoning".to_string(), json!({"effort": "none"}))]);
        let client = TranslatingLlmClient::new(&chat_map_with_extra_body(
            &format!("{}/v1", server.uri()),
            extra_body,
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"},
            "reasoning_effort": "high",
            "max_tokens": 16
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;

        let seen_body = seen.lock().unwrap().clone();
        assert_eq!(
            seen_body
                .pointer("/reasoning/effort")
                .and_then(Value::as_str),
            Some("none"),
            "target NT pin must replace the caller's nested reasoning object"
        );
        assert!(
            seen_body.get("reasoning_effort").is_none(),
            "flat reasoning_effort must be dropped when the target pins its policy"
        );
        Ok(())
    }

    // A target pinning `reasoning.enabled = false` must likewise override a
    // caller-supplied nested reasoning object.
    #[tokio::test]
    async fn target_enabled_false_pin_overrides_caller_reasoning()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use std::sync::Mutex as StdMutex;
        let server = MockServer::start().await;
        let seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let seen_for_assert = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_assert.lock().unwrap() = body.clone();
                body.pointer("/reasoning/enabled") == Some(&Value::Bool(false))
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

        let extra_body = BTreeMap::from([("reasoning".to_string(), json!({"enabled": false}))]);
        let client = TranslatingLlmClient::new(&chat_map_with_extra_body(
            &format!("{}/v1", server.uri()),
            extra_body,
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "medium"},
            "max_tokens": 16
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;

        let seen_body = seen.lock().unwrap().clone();
        assert_eq!(
            seen_body.pointer("/reasoning/enabled"),
            Some(&Value::Bool(false)),
            "target enabled=false pin must replace the caller's reasoning object"
        );
        assert!(
            seen_body.pointer("/reasoning/effort").is_none(),
            "caller effort must not ride alongside the target's enabled=false pin"
        );
        Ok(())
    }

    // A target pinning nested `reasoning.effort` plus a caller flat
    // `reasoning_effort` with a conflicting value is rejected by OpenRouter
    // (dual representation). The target's nested object is canonical; the flat
    // field is dropped.
    #[tokio::test]
    async fn body_nested_reasoning_effort_suppresses_conflicting_flat_reasoning_effort()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use std::sync::Mutex as StdMutex;
        let server = MockServer::start().await;
        let seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let seen_for_assert = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_assert.lock().unwrap() = body.clone();
                body.pointer("/reasoning/effort").and_then(Value::as_str) == Some("low")
                    && body.get("reasoning_effort").is_none()
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

        let extra_body = BTreeMap::from([(
            "reasoning".to_string(),
            json!({"enabled": true, "effort": "low"}),
        )]);
        let client = TranslatingLlmClient::new(&chat_map_with_extra_body(
            &format!("{}/v1", server.uri()),
            extra_body,
        ))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "max_tokens": 16
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;

        let seen_body = seen.lock().unwrap().clone();
        assert!(
            seen_body.get("reasoning_effort").is_none(),
            "flat reasoning_effort must be suppressed when the body pins nested reasoning.effort"
        );
        assert_eq!(
            seen_body
                .pointer("/reasoning/effort")
                .and_then(Value::as_str),
            Some("low"),
            "nested reasoning.effort must reach the upstream"
        );
        Ok(())
    }

    // No target pin: a caller body carrying BOTH a nested `reasoning.effort`
    // and a flat `reasoning_effort` with a conflicting value must not ship the
    // dual representation to the upstream (OpenRouter rejects it). The nested
    // object wins; the flat field is dropped.
    #[tokio::test]
    async fn no_pin_nested_reasoning_effort_suppresses_conflicting_flat_reasoning_effort()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use std::sync::Mutex as StdMutex;
        let server = MockServer::start().await;
        let seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let seen_for_assert = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_assert.lock().unwrap() = body.clone();
                body.pointer("/reasoning/effort").and_then(Value::as_str) == Some("medium")
                    && body.get("reasoning_effort").is_none()
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

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "medium"},
            "reasoning_effort": "high",
            "max_tokens": 16
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;

        let seen_body = seen.lock().unwrap().clone();
        assert_eq!(
            seen_body
                .pointer("/reasoning/effort")
                .and_then(Value::as_str),
            Some("medium"),
            "nested reasoning.effort must reach the upstream"
        );
        assert!(
            seen_body.get("reasoning_effort").is_none(),
            "conflicting flat reasoning_effort must be dropped without a target pin"
        );
        Ok(())
    }

    // No target pin: a nested `reasoning` object WITHOUT a string `effort`
    // (e.g. `{"enabled": false}`) is still canonical - the flat
    // `reasoning_effort` must be dropped so the upstream never sees the
    // conflicting dual representation.
    #[tokio::test]
    async fn no_pin_object_reasoning_without_effort_suppresses_flat_reasoning_effort()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use std::sync::Mutex as StdMutex;
        let server = MockServer::start().await;
        let seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let seen_for_assert = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_assert.lock().unwrap() = body.clone();
                body.pointer("/reasoning/enabled") == Some(&Value::Bool(false))
                    && body.get("reasoning_effort").is_none()
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

        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"enabled": false},
            "reasoning_effort": "high",
            "max_tokens": 16
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;

        let seen_body = seen.lock().unwrap().clone();
        assert_eq!(
            seen_body.pointer("/reasoning/enabled"),
            Some(&Value::Bool(false)),
            "object-valued reasoning without effort must reach the upstream"
        );
        assert!(
            seen_body.get("reasoning_effort").is_none(),
            "flat reasoning_effort must be dropped when any object-valued reasoning is present"
        );
        Ok(())
    }

    // OpenAI Responses backend: a target `reasoning = { effort = "none" }` pin
    // must replace the caller's nested reasoning object on the Responses wire.
    #[tokio::test]
    async fn responses_target_reasoning_pin_overrides_caller_nested_reasoning()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        use std::sync::Mutex as StdMutex;
        let server = MockServer::start().await;
        let seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let seen_for_assert = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_assert.lock().unwrap() = body.clone();
                body.pointer("/reasoning/effort").and_then(Value::as_str) == Some("none")
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "model": "gpt",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;
        let extra_body = BTreeMap::from([("reasoning".to_string(), json!({"effort": "none"}))]);
        // Responses backend with a reasoning pin; the caller's nested reasoning
        // object arrives via the raw Responses body.
        let mut backend = config(&format!("{}/v1", server.uri()));
        backend.extra_body = extra_body;
        let client = TranslatingLlmClient::new(&vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiResponses(backend),
            None,
        )])?;
        let raw = json!({
            "model": "client-facing",
            "reasoning": {"effort": "high"},
            "input": "hi"
        });

        client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?;

        let seen_body = seen.lock().unwrap().clone();
        assert_eq!(
            seen_body
                .pointer("/reasoning/effort")
                .and_then(Value::as_str),
            Some("none"),
            "Responses-leg target NT pin must replace the caller's nested reasoning"
        );
        Ok(())
    }
    // The ChatGPT Codex Responses backend rejects `max_output_tokens` on every
    // inbound path; a normal OpenAI `/v1/responses` endpoint keeps it. The
    // strip must be Codex-specific, not a blanket Responses-format behavior.
    #[tokio::test]
    async fn codex_responses_requests_drop_max_output_tokens()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen_for_mock = seen.clone();
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                seen_for_mock.store(body.get("max_output_tokens").is_some(), Ordering::SeqCst);
                true
            })
            // Codex legs are stream-mandatory; the client forces stream=true, so
            // the mock must speak SSE (first-byte/stream-force work).
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&responses_map(&format!(
            "{}/backend-api/codex",
            server.uri()
        )))?;
        let raw = json!({
            "model": "client-facing",
            "max_output_tokens": 4096,
            "input": "hi"
        });
        let response = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?;
        assert!(matches!(response, RawResponse::Stream(_)));
        // The parameter never reached the Codex-shaped upstream.
        assert!(
            !seen.load(Ordering::SeqCst),
            "max_output_tokens leaked upstream"
        );
        Ok(())
    }

    // Chat requests routed to the Codex backend lose the translated
    // `max_tokens -> max_output_tokens` field too.
    #[tokio::test]
    async fn codex_chat_requests_drop_translated_max_output_tokens()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen_for_mock = seen.clone();
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                seen_for_mock.store(body.get("max_output_tokens").is_some(), Ordering::SeqCst);
                true
            })
            // Codex legs are stream-mandatory; the client forces stream=true,
            // so the mock must speak SSE.
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&responses_map(&format!(
            "{}/backend-api/codex",
            server.uri()
        )))?;
        let raw = json!({
            "model": "client-facing",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hi"}]
        });
        // Inbound chat -> internal -> outbound Responses (forced stream).
        let response = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiChat,
            )
            .await?;
        assert!(matches!(response, RawResponse::Stream(_)));
        assert!(
            !seen.load(Ordering::SeqCst),
            "translated max_output_tokens leaked upstream"
        );
        Ok(())
    }

    // Streaming chat requests routed to the Codex backend are forced to
    // upstream `stream = true` (stream-mandatory API), and the chat-only
    // `stream_options` usage decoration never leaks onto the Responses leg.
    #[tokio::test]
    async fn codex_responses_stream_force_without_stream_options()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let seen = Arc::new(std::sync::Mutex::new(None::<Value>));
        let seen_for_mock = seen.clone();
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_mock.lock().unwrap() = Some(body);
                true
            })
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&responses_map(&format!(
            "{}/backend-api/codex",
            server.uri()
        )))?;
        // Inbound chat with a buffered intent: the client forces upstream
        // stream=true, while the server layer aggregates SSE back to buffered
        // JSON for callers that asked for stream=false.
        let raw = json!({
            "model": "client-facing",
            "messages": [{"role": "user", "content": "hi"}]
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
        let captured = seen.lock().unwrap().clone().expect("request captured");
        assert_eq!(captured["stream"], json!(true));
        assert!(
            captured.get("stream_options").is_none(),
            "chat-only stream_options leaked onto a Codex Responses request: {captured}"
        );
        Ok(())
    }

    // Non-Codex OpenAI Responses backends keep `max_output_tokens`.
    #[tokio::test]
    async fn openai_responses_requests_keep_max_output_tokens()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen_for_mock = seen.clone();
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                seen_for_mock.store(
                    body.get("max_output_tokens") == Some(&json!(4096)),
                    Ordering::SeqCst,
                );
                true
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "model": "gpt",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&responses_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "max_output_tokens": 4096,
            "input": "hi"
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
        assert_eq!(body["status"], "completed");
        assert!(
            seen.load(Ordering::SeqCst),
            "non-codex Responses backends keep max_output_tokens"
        );
        Ok(())
    }

    // Normal OpenAI Responses legs are untouched by the Codex stream-force
    // path: a caller-supplied `stream_options` on a same-format Responses
    // request survives verbatim (no strip, and no chat-only decoration).
    #[tokio::test]
    async fn openai_responses_preserves_caller_stream_options()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let server = MockServer::start().await;
        let seen = Arc::new(std::sync::Mutex::new(None::<Value>));
        let seen_for_mock = seen.clone();
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(move |request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                *seen_for_mock.lock().unwrap() = Some(body);
                true
            })
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let client = TranslatingLlmClient::new(&responses_map(&format!("{}/v1", server.uri())))?;
        let raw = json!({
            "model": "client-facing",
            "stream": true,
            "stream_options": {"include_usage": true},
            "input": "hi"
        });
        let RawResponse::Stream(_) = client
            .call_rewrite_model_raw(
                raw,
                None,
                Some(&ModelId::from("gpt")),
                WireFormat::OpenAiResponses,
            )
            .await?
        else {
            panic!("expected a streamed response");
        };
        let captured = seen.lock().unwrap().clone().expect("request captured");
        assert_eq!(
            captured["stream_options"],
            json!({"include_usage": true}),
            "caller-supplied stream_options must round-trip verbatim on normal Responses legs"
        );
        Ok(())
    }

    // Negative control: a URL that merely CONTAINS a codex-like path segment is
    // not a Codex backend; the parameter must survive.
    #[tokio::test]
    async fn codex_near_match_urls_keep_max_output_tokens()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        for near_path in ["/backend-api/codex-compat", "/not-backend-api/codex"] {
            let server = MockServer::start().await;
            let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let seen_for_mock = seen.clone();
            let path_suffix = if near_path == "/backend-api/codex-compat" {
                "/backend-api/codex-compat/responses"
            } else {
                "/not-backend-api/codex/responses"
            };
            Mock::given(method("POST"))
                .and(path(path_suffix))
                .and(move |request: &wiremock::Request| {
                    let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
                    seen_for_mock.store(
                        body.get("max_output_tokens") == Some(&json!(4096)),
                        Ordering::SeqCst,
                    );
                    true
                })
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt",
                    "output": [{"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                })))
                .mount(&server)
                .await;

            let base = format!("{}{}", server.uri(), near_path);
            let client = TranslatingLlmClient::new(&responses_map(&base))?;
            let raw = json!({
                "model": "client-facing",
                "max_output_tokens": 4096,
                "input": "hi"
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
            assert_eq!(body["status"], "completed");
            assert!(
                seen.load(Ordering::SeqCst),
                "near-match URL must not be treated as Codex"
            );
        }
        Ok(())
    }
    // --- S2-E.2: exact OpenAI-chat input-token counting -------------------------

    #[tokio::test]
    async fn count_input_tokens_returns_exact_count() -> Result<()> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions/input_tokens"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!(
                {"input_tokens": 42, "object": "response.input_tokens"}
            )))
            .mount(&server)
            .await;
        let client = TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri())))?;
        let model = ModelId::from("gpt");
        let n = client
            .count_input_tokens(&model, &request_for(Some("gpt"), false))
            .await?;
        assert_eq!(n, 42);
        Ok(())
    }

    #[tokio::test]
    async fn count_input_tokens_rejects_invalid_payloads() {
        // Rejects every non-`{ "input_tokens": N }`-with-non-negative-integer
        // result. (A numeric literal larger than u64 cannot be held by
        // serde_json's default Number, so the u64-overflow path is covered by
        // the strict `as_u64` validation in `count_input_tokens`, not a
        // constructible JSON fixture.)
        let cases: &[serde_json::Value] = &[
            json!({}),                     // missing field
            json!({"input_tokens": null}), // null
            json!({"input_tokens": -1}),   // negative
            json!({"input_tokens": 3.5}),  // float
            json!({"input_tokens": "12"}), // string
        ];
        for payload in cases {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(payload.clone()))
                .mount(&server)
                .await;
            let client =
                TranslatingLlmClient::new(&chat_map(&format!("{}/v1", server.uri()))).unwrap();
            let model = ModelId::from("gpt");
            let result = client
                .count_input_tokens(&model, &request_for(Some("gpt"), false))
                .await;
            assert!(result.is_err(), "payload {payload} must be rejected");
        }
    }

    #[tokio::test]
    async fn input_tokens_count_carries_target_extra_body_and_model_stamp() -> Result<()> {
        // The counted body is the same token-relevant final representation
        // generation would POST: target model stamp + target extra_body.
        let server = MockServer::start().await;
        let seen = Arc::new(std::sync::Mutex::new(None::<serde_json::Value>));
        let seen_for_mock = Arc::clone(&seen);
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions/input_tokens"))
            .respond_with(move |request: &wiremock::Request| {
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).unwrap_or(serde_json::Value::Null);
                *seen_for_mock.lock().unwrap() = Some(body);
                wiremock::ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 7}))
            })
            .mount(&server)
            .await;
        let mut backend = config(&format!("{}/v1", server.uri()));
        backend.extra_body = BTreeMap::from([(
            "chat_template_kwargs".to_string(),
            json!({"enable_thinking": false}),
        )]);
        let client = TranslatingLlmClient::new(&vec![ModelConfig::new(
            "gpt",
            Backend::OpenAiChat(backend),
            None,
        )])?;
        let model = ModelId::from("gpt");
        let n = client
            .count_input_tokens(&model, &request_for(Some("wrong-model"), false))
            .await?;
        assert_eq!(n, 7);
        let body = seen.lock().unwrap().clone().expect("request captured");
        assert_eq!(body["model"], json!("gpt"), "target model stamp");
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(false),
            "target extra_body applies to the counted representation"
        );
        Ok(())
    }
    // A backend that configures a canonical header via `extra_headers` must win
    // exactly once: an inbound caller header whose name matches an `extra_headers`
    // key is suppressed so the backend value applies once (not duplicated or
    // shadowed by a caller-supplied value). ThaiLLM's fixed User-Agent is the
    // production case (the upstream WAF requires the canonical UA).
    #[tokio::test]
    async fn extra_headers_override_inbound_headers_exactly_once()
    -> std::result::Result<(), Box<dyn Error + Sync + Send + 'static>> {
        let mut extra_headers = BTreeMap::new();
        extra_headers.insert(
            "User-Agent".to_string(),
            "Switchyard/backend-canonical-ua".to_string(),
        );
        let client = TranslatingLlmClient::new(&chat_map_with_extra_headers(
            "http://127.0.0.1:9/v1",
            extra_headers,
        ))?;
        let backend = client
            .backend_for(&ModelId::from("gpt"), WireFormat::OpenAiChat)
            .unwrap();

        // Caller metadata carries its own User-Agent plus a passthrough header we
        // expect to be forwarded (the backend does not override it).
        let mut caller_headers = http::HeaderMap::new();
        caller_headers.insert("user-agent", http::HeaderValue::from_static("caller-ua"));
        caller_headers.insert("x-resource-ref", http::HeaderValue::from_static("ref-1"));

        let builder = reqwest::Client::new().post("http://127.0.0.1:9/v1/chat/completions");
        let builder = forward_metadata_headers(
            builder,
            Some(&Metadata {
                session_id: None,
                agent_id: None,
                task_id: None,
                correlation_id: None,
                extra_metadata: None,
                http_headers: Some(caller_headers),
                wire_format: None,
                ..Default::default()
            }),
            backend,
        );
        let builder = apply_extra_headers(builder, backend);
        let request = builder.build().expect("request builds");

        // Reserved/suppressed inbound override: the caller's User-Agent is NOT
        // forwarded; the backend's canonical value is set exactly once.
        let ua_values: Vec<&str> = request
            .headers()
            .get_all("user-agent")
            .iter()
            .map(|v| v.to_str().unwrap())
            .collect();
        assert_eq!(ua_values, vec!["Switchyard/backend-canonical-ua"]);
        // Non-overridden inbound header still forwards.
        assert_eq!(
            request
                .headers()
                .get("x-resource-ref")
                .map(|v| v.to_str().unwrap()),
            Some("ref-1"),
        );
        Ok(())
    }
}
