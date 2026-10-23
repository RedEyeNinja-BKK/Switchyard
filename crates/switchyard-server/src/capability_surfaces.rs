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

use switchyard_runner::capability::{CapabilityFormat, ResolvedCapability};

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
    if capability.format != CapabilityFormat::OpenaiEmbeddings {
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
            let mut headers = axum::http::HeaderMap::new();
            if status.is_success() {
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
            }
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
