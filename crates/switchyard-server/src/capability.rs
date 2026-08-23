// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Capability-specific routing: embeddings and reranking.
//!
//! These are NOT LLM chat capabilities. They are typed utility endpoints
//! (OpenAI-compatible `POST /v1/embeddings` and Cohere/Jina-compatible
//! `POST /v1/rerank`) that Switchyard routes to a single canonical executor
//! each. Switchyard does not compute embeddings, execute reranking, or persist
//! vectors — it selects the executor target and proxies the typed contract,
//! with contract/bounds admission enforced at the route boundary.
//!
//! The executor is a deployment fact (PRIMARY today: ComfyNinja CPU ingress
//! `reninja.tailc8ef7b.ts.net:8448/8449`; fallback: HTPC :8080/:8082).
//! Consumers only know the Switchyard route identity (e.g. `localclaw/embed`,
//! `localclaw/rerank`), never the physical host.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::ModelId;

use crate::config::HttpBaseUrl;

/// Formats a capability client speaks with its executor.
#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) enum CapabilityClientFormat {
    #[serde(rename = "openai_embeddings")]
    OpenAiEmbeddings,
    #[serde(rename = "cohere_jina_rerank")]
    CohereJinaRerank,
}

/// Declares one capability executor target (`[capability_clients.<name>]`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilityClientConfig {
    pub(crate) format: CapabilityClientFormat,
    pub(crate) base_url: HttpBaseUrl,
    pub(crate) model: String,
    api_key_env: Option<String>,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_capability_timeout_seconds")]
    pub(crate) timeout_seconds: u64,
}

const fn default_capability_timeout_seconds() -> u64 {
    30
}

/// A failure from a capability executor call or admission check.
#[derive(Debug)]
pub(crate) struct CapabilityError {
    message: String,
}

impl CapabilityError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for CapabilityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CapabilityError {}

/// A typed client bound to one capability executor.
pub(crate) struct CapabilityClient {
    format: CapabilityClientFormat,
    base_url: String,
    model: String,
    api_key: Option<String>,
    extra_headers: BTreeMap<String, String>,
    http: reqwest::Client,
}

impl CapabilityClient {
    pub(crate) fn new(config: &CapabilityClientConfig) -> Result<Self, CapabilityError> {
        let api_key = match &config.api_key_env {
            Some(name) => std::env::var(name).ok().filter(|value| !value.is_empty()),
            None => None,
        };
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|error| {
                CapabilityError::new(format!("failed to build HTTP client: {error}"))
            })?;
        Ok(Self {
            format: config.format,
            base_url: config.base_url.as_str().trim_end_matches('/').to_string(),
            model: config.model.clone(),
            api_key,
            extra_headers: config.extra_headers.clone(),
            http,
        })
    }

    /// The executor-side model identity (distinct from the Switchyard route id).
    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    fn endpoint(&self) -> String {
        match self.format {
            CapabilityClientFormat::OpenAiEmbeddings => format!("{}/v1/embeddings", self.base_url),
            CapabilityClientFormat::CohereJinaRerank => format!("{}/v1/rerank", self.base_url),
        }
    }

    async fn post(&self, body: Value) -> Result<Value, CapabilityError> {
        let url = self.endpoint();
        let mut request = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        for (name, value) in &self.extra_headers {
            request = request.header(name, value);
        }
        let response = request.send().await.map_err(|error| {
            CapabilityError::new(format!("capability executor {url} unreachable: {error}"))
        })?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|error| {
            CapabilityError::new(format!("capability executor {url} read failed: {error}"))
        })?;
        if !status.is_success() {
            return Err(CapabilityError::new(format!(
                "capability executor {url} returned HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }
        serde_json::from_slice(&bytes).map_err(|error| {
            CapabilityError::new(format!(
                "capability executor {url} returned malformed JSON: {error}"
            ))
        })
    }

    /// Proxy one OpenAI-compatible embedding request to the executor.
    pub(crate) async fn embeddings(&self, body: Value) -> Result<Value, CapabilityError> {
        self.post(body).await
    }

    /// Proxy one Cohere/Jina rerank request to the executor.
    pub(crate) async fn rerank(&self, body: Value) -> Result<Value, CapabilityError> {
        self.post(body).await
    }
}

/// Capability-specific route semantics (non-LLM).
///
/// Discrimination happens in [`CapabilityRouteConfig::deserialize`] by
/// required-field presence (Embedding requires `contract`+`dimensions`;
/// Rerank requires none of the embedding fields).
#[derive(Clone, Debug)]
pub(crate) enum CapabilityKind {
    Embedding {
        /// Logical embedding-space contract identity (e.g. `localclaw-embedding-space:v1`).
        contract: String,
        /// Expected vector dimension; responses that differ fail closed.
        dimensions: usize,
        /// Expected normalization (informational admission field).
        normalization: String,
        /// Maximum inputs per request.
        max_batch: usize,
    },
    Rerank {
        max_candidates: usize,
        top_n: usize,
        max_doc_chars: usize,
        max_query_chars: usize,
    },
}

fn default_normalization() -> String {
    "L2".to_string()
}

/// One routed capability: the executor client plus its admission envelope.
#[derive(Clone)]
pub(crate) struct CapabilityRoute {
    pub(crate) kind: CapabilityKind,
    pub(crate) client: Arc<CapabilityClient>,
}

/// Capability-specific route declaration (`[capabilities.<name>]`).
///
/// Custom deserializer: `[capabilities.embed]` carries `id`/`target` plus
/// either the Embedding fields (`contract`, `dimensions`, …) or the Rerank
/// fields (`max_candidates`, …). Presence of `contract` selects Embedding;
/// otherwise Rerank. Unknown fields fail closed.
#[derive(Clone, Debug)]
pub(crate) struct CapabilityRouteConfig {
    /// Route id served to callers (e.g. `localclaw/embed`, `localclaw/rerank`).
    pub(crate) id: String,
    /// Name of the `[capability_clients.*]` executor this route targets.
    pub(crate) target: String,
    /// Capability-specific admission contract.
    pub(crate) kind: CapabilityKind,
}

impl<'de> Deserialize<'de> for CapabilityRouteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            id: String,
            target: String,
            contract: Option<String>,
            dimensions: Option<usize>,
            normalization: Option<String>,
            max_batch: Option<usize>,
            max_candidates: Option<usize>,
            top_n: Option<usize>,
            max_doc_chars: Option<usize>,
            max_query_chars: Option<usize>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let kind = if raw.contract.is_some() || raw.dimensions.is_some() {
            let contract = raw.contract.ok_or_else(|| {
                serde::de::Error::custom("embedding capability requires `contract`")
            })?;
            let dimensions = raw.dimensions.ok_or_else(|| {
                serde::de::Error::custom("embedding capability requires `dimensions`")
            })?;
            CapabilityKind::Embedding {
                contract,
                dimensions,
                normalization: raw.normalization.unwrap_or_else(default_normalization),
                max_batch: raw.max_batch.unwrap_or_else(default_max_batch),
            }
        } else {
            CapabilityKind::Rerank {
                max_candidates: raw.max_candidates.unwrap_or_else(default_max_candidates),
                top_n: raw.top_n.unwrap_or_else(default_top_n),
                max_doc_chars: raw.max_doc_chars.unwrap_or_else(default_max_doc_chars),
                max_query_chars: raw.max_query_chars.unwrap_or_else(default_max_query_chars),
            }
        };
        Ok(Self {
            id: raw.id,
            target: raw.target,
            kind,
        })
    }
}

const fn default_max_batch() -> usize {
    64
}

const fn default_max_candidates() -> usize {
    32
}

const fn default_top_n() -> usize {
    5
}

const fn default_max_doc_chars() -> usize {
    4096
}

const fn default_max_query_chars() -> usize {
    2048
}

/// Validates an OpenAI-compatible embedding response shape against the route's
/// declared contract (dimension admission). Malformed or dimension-mismatched
/// responses fail closed — never silently proxied.
pub(crate) fn validate_embedding_response(
    value: &Value,
    expected_dimensions: usize,
) -> Result<(), String> {
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "embedding executor response is missing a data array".to_string())?;
    for item in data {
        let vector = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| "embedding executor response item is missing its embedding vector")?;
        if vector.len() != expected_dimensions {
            return Err(format!(
                "embedding executor returned {} dimensions, expected {}",
                vector.len(),
                expected_dimensions
            ));
        }
    }
    Ok(())
}

/// Validates a Cohere/Jina rerank response shape.
pub(crate) fn validate_rerank_response(value: &Value) -> Result<(), String> {
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| "rerank executor response is missing a results array".to_string())?;
    for result in results {
        if !result.get("index").and_then(Value::as_u64).is_some() {
            return Err("rerank executor result is missing a numeric index".to_string());
        }
        if !result
            .get("relevance_score")
            .and_then(Value::as_f64)
            .is_some()
        {
            return Err("rerank executor result is missing a numeric relevance_score".to_string());
        }
    }
    Ok(())
}

/// Builds the executor-side rerank body from a stack-facing request.
///
/// Candidates may be plain strings or objects carrying `id`/`text`/`metadata`.
/// The executor accepts strings only, so objects are reduced to their text for
/// the call; result `index` positions refer to the original candidate order, so
/// the caller's `index -> candidate` join preserves id/metadata losslessly.
pub(crate) fn rerank_executor_body(
    route_model: &ModelId,
    executor_model: &str,
    query: &str,
    documents: &[Value],
    effective_top_n: usize,
) -> Result<Value, String> {
    let mut texts = Vec::with_capacity(documents.len());
    for document in documents {
        match document {
            Value::String(text) => texts.push(text.clone()),
            Value::Object(object) => {
                let text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "rerank candidate object is missing a text field".to_string())?;
                texts.push(text.to_string());
            }
            _ => {
                return Err(
                    "rerank documents must be strings or objects with a text field".to_string()
                )
            }
        }
    }
    Ok(json!({
        "model": executor_model,
        "query": query,
        "documents": texts,
        "top_n": effective_top_n,
        // Provenance note: route identity is echoed back for correlation.
        "route": route_model,
    }))
}
