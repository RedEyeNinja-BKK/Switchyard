// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Capability-specific serving: embeddings and reranking.
//!
//! Switchyard does not compute embeddings, execute reranking, or persist
//! vectors — it selects the executor target declared by the parsed capability
//! configuration and proxies the typed contract, with contract/bounds
//! admission enforced at the route boundary. Consumers only know the
//! Switchyard route identity (e.g. `localclaw/embed`, `localclaw/rerank`),
//! never the physical host.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use switchyard_protocol::ModelId;
use switchyard_runner::{CapabilityClientConfig, CapabilityKind};

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
    model: String,
    api_key: Option<String>,
    extra_headers: BTreeMap<String, String>,
    http: reqwest::Client,
    endpoint: String,
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
        let endpoint = match config.format {
            switchyard_runner::CapabilityClientFormat::OpenAiEmbeddings => {
                format!(
                    "{}/v1/embeddings",
                    config.base_url.as_str().trim_end_matches('/')
                )
            }
            switchyard_runner::CapabilityClientFormat::CohereJinaRerank => {
                format!(
                    "{}/v1/rerank",
                    config.base_url.as_str().trim_end_matches('/')
                )
            }
        };
        Ok(Self {
            model: config.model.clone(),
            api_key,
            extra_headers: config.extra_headers.clone(),
            http,
            endpoint,
        })
    }

    /// The executor-side model identity (distinct from the Switchyard route id).
    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    async fn post(&self, body: Value) -> Result<Value, CapabilityError> {
        let url = &self.endpoint;
        let mut request = self.http.post(url).json(&body);
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

    /// Proxy one typed capability request to the executor.
    pub(crate) async fn call(&self, body: Value) -> Result<Value, CapabilityError> {
        self.post(body).await
    }
}

/// One routed capability: the executor client plus its admission envelope.
#[derive(Clone)]
pub(crate) struct CapabilityRoute {
    pub(crate) kind: CapabilityKind,
    pub(crate) client: Arc<CapabilityClient>,
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
        .ok_or("embedding executor response is missing a data array")?;
    for item in data {
        let vector = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or("embedding executor response item is missing its embedding vector")?;
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
        .ok_or("rerank executor response is missing a results array")?;
    for result in results {
        if result.get("index").and_then(Value::as_u64).is_none() {
            return Err("rerank executor result is missing a numeric index".into());
        }
        if result
            .get("relevance_score")
            .and_then(Value::as_f64)
            .is_none()
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
                    "rerank documents must be strings or objects with a text field".to_string(),
                );
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
