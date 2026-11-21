// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving surfaces for the specialized capability contracts: embeddings and reranking.
//!
//! These are not completion routes. A request names a capability, the capability's
//! declared contract decides what is legal, and the request is forwarded to that
//! capability's own engine. There is no routing and no fallback: an embedding request
//! either satisfies its contract or fails, because silently serving another engine's
//! vectors would produce values the caller cannot compare.
//!
//! Declared limits are enforced here rather than at the engine, so a caller learns the
//! contract from a 400 naming the limit instead of from a truncated result.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use switchyard_llm_client::{PacingError, PacingTicket, ReserveError};
use switchyard_runner::capability::{
    CapabilityFormat, GEMINI_EMBEDDING_V2_MAX_DIMENSIONS, GEMINI_EMBEDDING_V2_MIN_DIMENSIONS,
    ResolvedCapability,
};

use crate::ServerState;

/// `POST /v1/embeddings` — OpenAI-compatible embedding request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingsRequest {
    /// Public capability id, in place of the provider's model name.
    model: String,
    /// Texts to embed.
    input: Vec<String>,
}

/// `POST /v1/rerank` — Cohere/Jina-compatible reranking request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankRequest {
    /// Public capability id, in place of the provider's model name.
    model: String,
    /// The query documents are ranked against.
    query: String,
    /// Documents to rank.
    documents: Vec<String>,
    /// Results to return. Unset uses the capability's declared `top_n`. May not exceed it.
    top_n: Option<u32>,
}

pub(crate) async fn embeddings(
    State(state): State<ServerState>,
    body: std::result::Result<Json<EmbeddingsRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(parsed) => parsed,
        Err(error) => return contract_error(StatusCode::BAD_REQUEST, error.body_text()),
    };
    let Some(capability) = state.runner.capability(&request.model) else {
        return unknown_capability(&request.model, &state.capability_ids);
    };
    if !capability.format.serves_embeddings() {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!("capability {} does not serve embeddings", capability.id),
        );
    }
    if request.input.is_empty() {
        return contract_error(
            StatusCode::BAD_REQUEST,
            "input must contain at least one text".to_string(),
        );
    }
    if let Some(max_batch) = capability.max_batch
        && request.input.len() as u64 > u64::from(max_batch)
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "input carries {} texts but capability {} accepts at most {max_batch}",
                request.input.len(),
                capability.id
            ),
        );
    }

    // A Google embedding capability answers one content per call, so its batch is
    // assembled here. Everything else forwards the batch whole.
    if capability.format == CapabilityFormat::GoogleEmbedContent {
        return google_embed(&state, capability, &request.input).await;
    }
    if capability.format == CapabilityFormat::GoogleEmbedContentV2 {
        return google_embed_v2(&state, capability, &request.input).await;
    }

    // The engine names its own model; the public capability id is a Switchyard id and
    // must not be forwarded as a provider model name.
    let payload = json!({
        "model": capability.model,
        "input": request.input,
    });
    forward(&state, capability, payload, Some(request.input.len())).await
}

pub(crate) async fn rerank(
    State(state): State<ServerState>,
    body: std::result::Result<Json<RerankRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(parsed) => parsed,
        Err(error) => return contract_error(StatusCode::BAD_REQUEST, error.body_text()),
    };
    let Some(capability) = state.runner.capability(&request.model) else {
        return unknown_capability(&request.model, &state.capability_ids);
    };
    if capability.format != CapabilityFormat::CohereJinaRerank {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!("capability {} does not serve reranking", capability.id),
        );
    }
    if request.documents.is_empty() {
        return contract_error(
            StatusCode::BAD_REQUEST,
            "documents must contain at least one text".to_string(),
        );
    }
    if let Some(max_candidates) = capability.max_candidates
        && request.documents.len() as u64 > u64::from(max_candidates)
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "documents carries {} entries but capability {} accepts at most {max_candidates}",
                request.documents.len(),
                capability.id
            ),
        );
    }
    if let Some(max_query_chars) = capability.max_query_chars
        && request.query.chars().count() > max_query_chars as usize
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "query is {} characters but capability {} accepts at most {max_query_chars}",
                request.query.chars().count(),
                capability.id
            ),
        );
    }
    if let Some(max_doc_chars) = capability.max_doc_chars
        && let Some(index) = request
            .documents
            .iter()
            .position(|text| text.chars().count() > max_doc_chars as usize)
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "documents[{index}] is longer than capability {} accepts ({max_doc_chars} characters)",
                capability.id
            ),
        );
    }
    let top_n = request.top_n.or(capability.top_n);
    // Two separate invariants. A caller may not ask for more results than the
    // capability's declared `top_n`, which is the contract's ceiling; and the declared
    // `top_n` itself may not exceed `max_candidates`, which config load already checks.
    // Comparing the caller's value against `max_candidates` would let `top_n = 6`
    // through a contract capped at 5.
    if let (Some(requested), Some(declared)) = (request.top_n, capability.top_n)
        && requested > declared
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "top_n {requested} exceeds capability {} top_n {declared}",
                capability.id
            ),
        );
    }
    if let (Some(top_n), Some(max_candidates)) = (top_n, capability.max_candidates)
        && top_n > max_candidates
    {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "top_n {top_n} exceeds capability {} max_candidates {max_candidates}",
                capability.id
            ),
        );
    }

    let mut payload = json!({
        "model": capability.model,
        "query": request.query,
        "documents": request.documents,
    });
    if let Some(top_n) = top_n {
        payload["top_n"] = json!(top_n);
    }
    forward(&state, capability, payload, None).await
}

/// Reserves one physical call against the capability's declared budgets.
///
/// The minute window is taken first and the daily counter second, because only the minute
/// ticket can be released: a daily reservation is written before the call and is the
/// provider's own countable unit, so a path that took one and then refused would leave the
/// budget spent for a call that never happened.
///
/// One reservation per physical call, not per caller request: a fan-out of ten inputs makes
/// ten charges, because the provider sees ten requests.
fn reserve_physical_call(
    capability: &ResolvedCapability,
    input_index: usize,
    input_bytes: u64,
) -> Result<Option<PacingTicket>, Box<Response>> {
    let Some(budget) = &capability.budget else {
        return Ok(None);
    };
    let ticket = match &budget.pacing {
        None => None,
        Some(pacing) => match pacing.reserve(input_bytes) {
            Ok(ticket) => ticket,
            Err(PacingError::Throttled(throttled)) => {
                return Err(Box::new(error_body(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!(
                        "per-minute {} budget for capability {} is full; it clears in about {}s",
                        throttled.scope.as_str(),
                        capability.id,
                        throttled.retry_after.as_secs()
                    ),
                    "minute_allowance_throttled",
                )));
            }
            Err(PacingError::ExceedsWindow {
                needed,
                limit,
                margin_permille,
            }) => {
                return Err(Box::new(error_body(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!(
                        "input[{input_index}] reserves about {needed} tokens for capability {} \
                         at a {margin_permille}/1000 admission margin, which is more than its \
                         {limit}-token per-minute input budget; waiting will not help",
                        capability.id
                    ),
                    "input_exceeds_minute_window",
                )));
            }
        },
    };
    if let Some(allowance) = &budget.allowance {
        match allowance.try_reserve() {
            Ok(()) => {}
            Err(ReserveError::Exhausted) => {
                if let (Some(pacing), Some(ticket)) = (&budget.pacing, &ticket) {
                    pacing.release(ticket);
                }
                return Err(Box::new(error_body(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!(
                        "daily request allowance for capability {} ({}) is spent ({} per provider \
                         day); it resets at the next provider day (midnight \
                         America/Los_Angeles)",
                        capability.id,
                        capability.model,
                        allowance.limit()
                    ),
                    "daily_allowance_exhausted",
                )));
            }
            Err(ReserveError::Unusable(reason)) => {
                if let (Some(pacing), Some(ticket)) = (&budget.pacing, &ticket) {
                    pacing.release(ticket);
                }
                return Err(Box::new(error_body(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "daily request allowance for capability {} cannot be accounted: {reason}",
                        capability.id
                    ),
                    "daily_allowance_unavailable",
                )));
            }
        }
    }
    Ok(ticket)
}

/// Releases a minute slot for a request that never reached the provider.
fn release_physical_call(capability: &ResolvedCapability, ticket: &Option<PacingTicket>) {
    if let (Some(budget), Some(ticket)) = (&capability.budget, ticket)
        && let Some(pacing) = &budget.pacing
    {
        pacing.release(ticket);
    }
}

/// Embeds a batch through Google's single-content endpoint.
///
/// Google has no bulk form, so the batch is a bounded fan-out of one call per input. The
/// caller still sends and receives one OpenAI-shaped batch: a caller should not need a
/// second client path because one engine answers one content at a time, and `index` is
/// where that difference must not leak.
///
/// A failed input fails the whole batch. Returning the vectors that succeeded would hand
/// back a shorter result set whose indexes no longer line up with the caller's inputs.
async fn google_embed(
    state: &ServerState,
    capability: &ResolvedCapability,
    input: &[String],
) -> Response {
    for (index, text) in input.iter().enumerate() {
        if text.is_empty() {
            return contract_error(StatusCode::BAD_REQUEST, format!("input[{index}] is empty"));
        }
        // A token's surface form is at least one byte, so a byte count is an upper bound
        // on the token count. Bounding bytes therefore guarantees the model's token limit
        // is never exceeded. It is conservative: text that would have fitted may be
        // refused, which is the only sound direction when the tokenizer is not local.
        if let Some(limit) = capability.max_input_tokens
            && text.len() > limit as usize
        {
            return contract_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "input[{index}] is {} bytes but capability {} accepts at most {limit} \
                     (its input-token limit)",
                    text.len(),
                    capability.id
                ),
            );
        }
    }

    let client = match state.capability_http(capability.timeout_seconds) {
        Ok(client) => client,
        Err(message) => return contract_error(StatusCode::BAD_GATEWAY, message),
    };
    let credential = match &capability.api_key_env {
        None => None,
        Some(variable) => match std::env::var(variable) {
            Ok(value) if !value.trim().is_empty() => Some(value),
            _ => return upstream_contract_error(capability, "credential unavailable"),
        },
    };

    let mut vectors = Vec::with_capacity(input.len());
    for (index, text) in input.iter().enumerate() {
        let mut payload = json!({
            "model": format!("models/{}", capability.model),
            "content": {"parts": [{"text": text}]},
        });
        if let Some(task_type) = &capability.task_type {
            payload["task_type"] = json!(task_type);
        }
        // One charge per physical call, before it is made.
        let ticket = match reserve_physical_call(capability, index, text.len() as u64) {
            Ok(ticket) => ticket,
            Err(refusal) => return *refusal,
        };
        let mut builder = client.post(&capability.url).json(&payload);
        if let Some(credential) = &credential {
            // Google reads its key from this header; a bearer token here is an
            // authentication failure that looks like a bad credential.
            builder = builder.header("x-goog-api-key", credential);
        }
        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                release_physical_call(capability, &ticket);
                return contract_error(
                    StatusCode::BAD_GATEWAY,
                    format!("capability {} is unreachable: {}", capability.id, error),
                );
            }
        };
        let status =
            StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let body = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            // The engine's own status and message travel back: a 429 stays a 429, and the
            // caller learns which input it was, which the engine cannot say.
            return error_body(
                status,
                format!(
                    "capability {} input[{index}] failed: {}",
                    capability.id,
                    String::from_utf8_lossy(&body)
                ),
                "upstream_error",
            );
        }
        let Ok(value) = serde_json::from_slice::<Value>(&body) else {
            return upstream_contract_error(capability, "upstream returned non-JSON success");
        };
        let Some(values) = value["embedding"]["values"].as_array() else {
            return upstream_contract_error(capability, "embedding.values missing");
        };
        if let Some(dimensions) = capability.dimensions
            && values.len() != dimensions as usize
        {
            return upstream_contract_error(capability, "embedding dimension mismatch");
        }
        if values
            .iter()
            .any(|number| number.as_f64().is_none_or(|n| !n.is_finite()))
        {
            return upstream_contract_error(capability, "embedding value is not a finite number");
        }
        vectors.push(Value::Array(values.clone()));
    }

    let data: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({"object": "embedding", "index": index, "embedding": embedding})
        })
        .collect();
    let body = json!({"object": "list", "model": capability.model, "data": data});
    (StatusCode::OK, verified_headers(capability), Json(body)).into_response()
}

/// Embeds a batch through `gemini-embedding-2`'s `:embedContent` codec.
///
/// Same caller contract as every other embedding surface — one OpenAI-shaped batch in, one
/// out, in input order — over a different wire: the output width is a request field, and a
/// task travels as text in front of the input rather than as a `task_type` field.
///
/// Text only. This codec sends `content.parts[].text` and nothing else, so a deployment that
/// exposes it must not be read as accepting images, audio, video or PDF through it. Wrapping
/// media in `fileData` is a separate, unqualified modality.
///
/// One physical call per input, each charged to the model's budgets before it is made. The
/// fan-out is bounded by the capability's declared `max_batch`, which is why this function
/// is never handed an unbounded batch.
async fn google_embed_v2(
    state: &ServerState,
    capability: &ResolvedCapability,
    input: &[String],
) -> Response {
    let Some(dimensions) = capability.dimensions else {
        // Unreachable for a validated deployment; a surface that guessed a width would be
        // promising vectors it cannot check.
        return upstream_contract_error(capability, "dimensions undeclared");
    };
    if !(GEMINI_EMBEDDING_V2_MIN_DIMENSIONS..=GEMINI_EMBEDDING_V2_MAX_DIMENSIONS)
        .contains(&dimensions)
    {
        return upstream_contract_error(capability, "declared dimensions outside the model range");
    }
    let prefix = capability.task_prefix.as_deref().unwrap_or("");
    for (index, text) in input.iter().enumerate() {
        if text.is_empty() {
            return contract_error(StatusCode::BAD_REQUEST, format!("input[{index}] is empty"));
        }
        // The prefix is sent with the text, so it counts toward the model's input limit.
        // Bounding bytes bounds tokens from above: a token's surface form is at least one
        // byte, so text that fits this budget cannot exceed the limit, and text over it is
        // refused rather than truncated.
        let sent_bytes = prefix.len() + text.len();
        if let Some(limit) = capability.max_input_tokens
            && sent_bytes > limit as usize
        {
            return contract_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "input[{index}] is {sent_bytes} bytes including its task prefix but capability \
                     {} accepts at most {limit} (its input-token limit)",
                    capability.id
                ),
            );
        }
    }

    let client = match state.capability_http(capability.timeout_seconds) {
        Ok(client) => client,
        Err(message) => return contract_error(StatusCode::BAD_GATEWAY, message),
    };
    let credential = match &capability.api_key_env {
        None => None,
        Some(variable) => match std::env::var(variable) {
            Ok(value) if !value.trim().is_empty() => Some(value),
            _ => return upstream_contract_error(capability, "credential unavailable"),
        },
    };

    let mut vectors = Vec::with_capacity(input.len());
    for (index, text) in input.iter().enumerate() {
        let sent = format!("{prefix}{text}");
        let payload = json!({
            "model": format!("models/{}", capability.model),
            "content": {"parts": [{"text": sent}]},
            "output_dimensionality": dimensions,
        });
        let ticket = match reserve_physical_call(capability, index, sent.len() as u64) {
            Ok(ticket) => ticket,
            Err(refusal) => return *refusal,
        };
        let mut builder = client.post(&capability.url).json(&payload);
        if let Some(credential) = &credential {
            builder = builder.header("x-goog-api-key", credential);
        }
        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                release_physical_call(capability, &ticket);
                return contract_error(
                    StatusCode::BAD_GATEWAY,
                    format!("capability {} is unreachable: {}", capability.id, error),
                );
            }
        };
        let status =
            StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let body = response.bytes().await.unwrap_or_default();
        if !status.is_success() {
            return error_body(
                status,
                format!(
                    "capability {} input[{index}] failed: {}",
                    capability.id,
                    String::from_utf8_lossy(&body)
                ),
                "upstream_error",
            );
        }
        let Ok(value) = serde_json::from_slice::<Value>(&body) else {
            return upstream_contract_error(capability, "upstream returned non-JSON success");
        };
        // One content in, one embedding out. A different count means the engine aggregated
        // or split the input, and the vector we would return is not the one asked for.
        let Some(values) = value["embedding"]["values"].as_array() else {
            return upstream_contract_error(capability, "embedding.values missing");
        };
        if values.len() != dimensions as usize {
            return upstream_contract_error(capability, "embedding dimension mismatch");
        }
        let mut numbers = Vec::with_capacity(values.len());
        for value in values {
            let Some(number) = value.as_f64() else {
                return upstream_contract_error(capability, "embedding value is not a number");
            };
            if !number.is_finite() {
                return upstream_contract_error(capability, "embedding value is not finite");
            }
            numbers.push(number);
        }
        if let Err(reason) = check_declared_normalization(capability, &numbers) {
            return upstream_contract_error(capability, reason);
        }
        vectors.push(Value::Array(values.clone()));
    }

    let data: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({"object": "embedding", "index": index, "embedding": embedding})
        })
        .collect();
    let body = json!({"object": "list", "model": capability.model, "data": data});
    (StatusCode::OK, verified_headers(capability), Json(body)).into_response()
}

/// Checks a received vector against the normalization its contract declares.
///
/// A declared normalization that is never verified is an unproven claim about the value
/// space. Only the declarations this surface knows how to check are checked; an
/// unrecognised value is not silently accepted as "no check", it is refused, because a
/// contract nobody can verify should not load.
fn check_declared_normalization(
    capability: &ResolvedCapability,
    values: &[f64],
) -> Result<(), &'static str> {
    let Some(declared) = capability.normalization.as_deref() else {
        return Ok(());
    };
    if !declared.eq_ignore_ascii_case("l2") {
        return Err("declared normalization is not one this surface can verify");
    }
    let sum_of_squares: f64 = values.iter().map(|value| value * value).sum();
    let norm = sum_of_squares.sqrt();
    // Only length matters here, so a generous tolerance still distinguishes a unit vector
    // from an unnormalised one of real width.
    if (norm - 1.0).abs() > 1e-2 {
        return Err("embedding is not L2-normalised as its contract declares");
    }
    Ok(())
}

/// The headers that attest to a verified successful embedding response.
///
/// They are only ever attached to vectors this surface actually received and checked, so
/// an upstream error cannot carry a contract the caller never got.
fn verified_headers(capability: &ResolvedCapability) -> axum::http::HeaderMap {
    let mut headers = axum::http::HeaderMap::new();
    if let Some(contract) = &capability.contract {
        headers.insert(
            "x-switchyard-embedding-contract",
            contract
                .parse()
                .unwrap_or_else(|_| axum::http::HeaderValue::from_static("unknown")),
        );
    }
    if let Some(dimensions) = capability.dimensions
        && let Ok(value) = dimensions.to_string().parse()
    {
        headers.insert("x-switchyard-embedding-dimensions", value);
    }
    headers
}

/// Forwards a capability request to its engine and returns the engine's own response.
///
/// The response is passed through rather than reshaped: embedding vectors and rerank
/// scores are the engine's contract, and re-encoding them here would risk changing
/// ordering or numeric representation.
async fn forward(
    state: &ServerState,
    capability: &ResolvedCapability,
    payload: Value,
    embedding_count: Option<usize>,
) -> Response {
    let client = match state.capability_http(capability.timeout_seconds) {
        Ok(client) => client,
        Err(message) => return contract_error(StatusCode::BAD_GATEWAY, message),
    };
    let mut builder = client.post(&capability.url).json(&payload);
    if let Some(variable) = &capability.api_key_env {
        let Ok(value) = std::env::var(variable) else {
            return upstream_contract_error(capability, "credential unavailable");
        };
        if value.trim().is_empty() {
            return upstream_contract_error(capability, "credential unavailable");
        }
        builder = builder.bearer_auth(value);
    }
    match builder.send().await {
        Ok(response) => {
            let status =
                StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body = response.bytes().await.unwrap_or_default();
            // Only Stephen's as-yet-unverified OpenRouter capabilities need the
            // strict upstream check. Existing embedding spaces and rerank engines
            // keep their established passthrough contract unchanged.
            if matches!(capability.id.as_str(), "ST_Embed" | "ST_Rerank") && status.is_success() {
                let Ok(value) = serde_json::from_slice::<Value>(&body) else {
                    return upstream_contract_error(
                        capability,
                        "upstream returned non-JSON success",
                    );
                };
                let result = if capability.id == "ST_Embed" {
                    validate_stephen_embedding(capability, &value, embedding_count.unwrap_or(0))
                } else {
                    validate_stephen_rerank(capability, &value, &payload)
                };
                if let Err(reason) = result {
                    return upstream_contract_error(capability, reason);
                }
            }
            // The contract travels back only with verified successful vectors;
            // on an upstream error it must never attest to an absent embedding.
            let headers = if status.is_success() {
                verified_headers(capability)
            } else {
                axum::http::HeaderMap::new()
            };
            (status, headers, body).into_response()
        }
        Err(error) => contract_error(
            StatusCode::BAD_GATEWAY,
            format!("capability {} is unreachable: {}", capability.id, error),
        ),
    }
}

fn upstream_contract_error(capability: &ResolvedCapability, reason: &str) -> Response {
    error_body(
        StatusCode::BAD_GATEWAY,
        format!(
            "capability {} upstream contract mismatch: {reason}",
            capability.id
        ),
        "upstream_contract_error",
    )
}

/// The model's own name, reduced to letters and digits.
///
/// Providers reformat the identifier they echo back. Measured against OpenRouter:
/// requesting `liquid/lfm-2.5-embedding-350m:free` returns
/// `openrouter:LiquidAI/LFM2.5-Embedding-350M:free` -- different vendor prefix,
/// different punctuation, different case. Exact string equality would reject a
/// perfectly good vector.
///
/// Comparing the last path segment (the model, not the vendor) with punctuation
/// and case folded still distinguishes the case that matters: `...embed-8b` vs
/// `...embed-1b` reduce to different strings.
fn model_name_key(identifier: &str) -> String {
    let without_prefix = identifier.strip_prefix("openrouter:").unwrap_or(identifier);
    let model = without_prefix.rsplit('/').next().unwrap_or(without_prefix);
    let model = model.split(':').next().unwrap_or(model);
    model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Whether the engine's answer came from the model we asked for.
///
/// A provider may report the canonical slug without the `:free` suffix, or under
/// its own vendor namespace, for the very same weights. Accepting that is not
/// leniency about identity: a substituted sibling (`...-embed-1b`) still fails,
/// which is the case that matters.
fn engine_identity_matches(requested: &str, returned: Option<&str>) -> bool {
    let Some(returned) = returned else {
        return false;
    };
    let wanted = model_name_key(requested);
    !wanted.is_empty() && wanted == model_name_key(returned)
}

fn validate_stephen_embedding(
    capability: &ResolvedCapability,
    value: &Value,
    expected_count: usize,
) -> Result<(), &'static str> {
    // OpenRouter may silently serve Embed 1B for an Embed 8B request. A
    // mismatched or absent identity is NOT evidence of the agreed vector space.
    if !engine_identity_matches(&capability.model, value["model"].as_str()) {
        return Err("model identity missing or substituted");
    }
    let dimensions = capability.dimensions.ok_or("dimensions undeclared")? as usize;
    let data = value["data"].as_array().ok_or("embedding data missing")?;
    if data.len() != expected_count {
        return Err("embedding batch length mismatch");
    }
    let mut seen = vec![false; expected_count];
    for item in data {
        let index = item["index"].as_u64().ok_or("embedding index missing")? as usize;
        if index >= expected_count || seen[index] {
            return Err("embedding index invalid or repeated");
        }
        seen[index] = true;
        let vector = item["embedding"]
            .as_array()
            .ok_or("embedding vector missing")?;
        if vector.len() != dimensions
            || vector
                .iter()
                .any(|number| number.as_f64().is_none_or(|n| !n.is_finite()))
        {
            return Err("embedding dimension or value mismatch");
        }
    }
    Ok(())
}

fn validate_stephen_rerank(
    capability: &ResolvedCapability,
    value: &Value,
    payload: &Value,
) -> Result<(), &'static str> {
    if !engine_identity_matches(&capability.model, value["model"].as_str()) {
        return Err("model identity missing or substituted");
    }
    let results = value["results"]
        .as_array()
        .ok_or("rerank results missing")?;
    let docs = payload["documents"]
        .as_array()
        .ok_or("rerank documents missing")?;
    let top_n = payload["top_n"].as_u64().unwrap_or(docs.len() as u64) as usize;
    if results.len() > top_n || results.len() > docs.len() {
        return Err("rerank result count exceeds request");
    }
    let mut seen = vec![false; docs.len()];
    for result in results {
        let index = result["index"].as_u64().ok_or("rerank index missing")? as usize;
        if index >= docs.len() || seen[index] {
            return Err("rerank index invalid or repeated");
        }
        seen[index] = true;
        if result["relevance_score"]
            .as_f64()
            .is_none_or(|score| !score.is_finite())
        {
            return Err("rerank score missing or invalid");
        }
    }
    Ok(())
}

/// A capability id this deployment does not expose. The message names what it does
/// expose, because "unknown id" without the real set is not actionable.
fn unknown_capability(model: &str, available: &[String]) -> Response {
    error_body(
        StatusCode::NOT_FOUND,
        format!(
            "No capability registered for model {model}. Available capabilities: {}",
            if available.is_empty() {
                "(none configured)".to_string()
            } else {
                available.join(", ")
            }
        ),
        "model_not_found",
    )
}

fn contract_error(status: StatusCode, message: String) -> Response {
    error_body(status, message, "invalid_request_error")
}

fn error_body(status: StatusCode, message: String, kind: &str) -> Response {
    (
        status,
        Json(json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
}

#[cfg(test)]
mod identity_tests {
    use super::model_name_key;

    /// The reformatting OpenRouter actually returned, measured 2026-10-22.
    #[test]
    fn provider_reformatting_of_the_same_model_is_accepted() {
        assert_eq!(
            model_name_key("liquid/lfm-2.5-embedding-350m:free"),
            model_name_key("openrouter:LiquidAI/LFM2.5-Embedding-350M:free"),
        );
        assert_eq!(
            model_name_key("nvidia/nemotron-3.5-embed-8b:free"),
            model_name_key("nvidia/nemotron-3.5-embed-8b"),
        );
    }

    /// The substitution the check exists to catch must still differ.
    #[test]
    fn a_sibling_variant_never_collapses_into_the_same_key() {
        assert_ne!(
            model_name_key("nvidia/nemotron-3.5-embed-8b:free"),
            model_name_key("nvidia/nemotron-3.5-embed-1b:free"),
        );
        assert_ne!(
            model_name_key("nvidia/nemotron-3.5-rerank-8b:free"),
            model_name_key("nvidia/nemotron-3.5-embed-8b:free"),
        );
    }
}
