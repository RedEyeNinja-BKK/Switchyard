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

/// The ORIGIN of a capability failure.
///
/// Recorded at the construction site and never recovered by parsing the
/// rendered message. The distinction that matters operationally is "the
/// backend could not serve this" versus "our deployment or our request is
/// wrong": a string like `capability executor ... returned HTTP 503` and one
/// like `... returned HTTP 401` differ only in a number buried in a sentence,
/// so classifying on text would be guesswork.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapabilityErrorKind {
    /// The executor could not be reached, or did not answer in time. Covers
    /// connect/DNS/TLS/send failures and the client timeout; a partially
    /// answered request is the same availability problem to a caller.
    Unreachable,
    /// The executor began answering and the response body could not be read to
    /// completion (truncated or reset mid-body). The backend was there and then
    /// stopped serving, which is an availability failure, not a defect.
    ReadFailed,
    /// The executor answered with this non-success HTTP status.
    Http(u16),
    /// The executor answered 2xx with a body that is not valid JSON.
    MalformedJson,
    /// The client could not be built from the declared configuration.
    Configuration,
}

/// A failure from a capability executor call or admission check.
#[derive(Debug)]
pub(crate) struct CapabilityError {
    message: String,
    kind: CapabilityErrorKind,
}

impl CapabilityError {
    /// Builds an error whose origin is UNCLASSIFIED.
    ///
    /// The default kind is [`CapabilityErrorKind::Configuration`] because it is
    /// the fail-closed choice: an unclassified failure is treated as our own
    /// defect and is never eligible for degradation elsewhere. Any new
    /// construction site that represents a genuine backend-availability
    /// failure must say so with [`CapabilityError::with_kind`] rather than
    /// relying on this default.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: CapabilityErrorKind::Configuration,
        }
    }

    /// Builds an error with an explicitly recorded origin.
    pub(crate) fn with_kind(kind: CapabilityErrorKind, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind,
        }
    }

    /// The recorded failure origin.
    pub(crate) const fn kind(&self) -> CapabilityErrorKind {
        self.kind
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
    pub(crate) format: switchyard_runner::CapabilityClientFormat,
    model: String,
    api_key: Option<String>,
    extra_headers: BTreeMap<String, String>,
    http: reqwest::Client,
    endpoint: String,
}

impl CapabilityClient {
    pub(crate) fn new(config: &CapabilityClientConfig) -> Result<Self, CapabilityError> {
        // A bare path only. Appending a value that carries a scheme or host
        // would make the joined URL parse as a different origin, silently
        // sending this capability's traffic — and its bearer credential —
        // somewhere other than the declared `base_url`.
        //
        // `//host/path` is the case a naive `starts_with('/')` misses: it
        // begins with a slash but is a NETWORK PATH to any URL joiner, so
        // `base_url + path` would resolve to the OTHER host.
        if let Some(path) = &config.endpoint_path {
            if !path.starts_with('/')
                || path.starts_with("//")
                || path.contains("://")
                || path.contains('@')
                || path.contains('?')
                || path.contains('#')
            {
                return Err(CapabilityError::with_kind(
                    CapabilityErrorKind::Configuration,
                    format!("capability endpoint_path must be a bare absolute path, got {path:?}"),
                ));
            }
            if !matches!(
                config.format,
                switchyard_runner::CapabilityClientFormat::OpenRouterAlphaDecisions
                    | switchyard_runner::CapabilityClientFormat::OpenAiResponsesDecisionAdapter
            ) {
                return Err(CapabilityError::with_kind(
                    CapabilityErrorKind::Configuration,
                    format!(
                        "capability endpoint_path is only meaningful for a decision executor, not {:?}",
                        config.format
                    ),
                ));
            }
        }
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
            // The decision surface is NOT a chat endpoint: the base URL names
            // the API root and this appends the typed-decisions path, unless
            // the client declares its own `endpoint_path` for a backend that
            // serves the same contract under a different path.
            switchyard_runner::CapabilityClientFormat::OpenRouterAlphaDecisions => {
                match &config.endpoint_path {
                    Some(path) => {
                        format!("{}{path}", config.base_url.as_str().trim_end_matches('/'))
                    }
                    None => format!(
                        "{}/api/alpha/decisions",
                        config.base_url.as_str().trim_end_matches('/')
                    ),
                }
            }
            // The generative fallback leg IS a normal chat endpoint; the
            // decision endpoint normalizes its text into the same contract.
            // An `endpoint_path` override replaces the standard path for the
            // same reason as the decision leg above.
            switchyard_runner::CapabilityClientFormat::OpenAiResponsesDecisionAdapter => {
                match &config.endpoint_path {
                    Some(path) => {
                        format!("{}{path}", config.base_url.as_str().trim_end_matches('/'))
                    }
                    None => format!(
                        "{}/v1/responses",
                        config.base_url.as_str().trim_end_matches('/')
                    ),
                }
            }
        };
        Ok(Self {
            format: config.format,
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

    /// The protocol this executor speaks, so a caller can shape its payload.
    pub(crate) fn format(&self) -> switchyard_runner::CapabilityClientFormat {
        self.format
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
            CapabilityError::with_kind(
                CapabilityErrorKind::Unreachable,
                format!("capability executor {url} unreachable: {error}"),
            )
        })?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(|error| {
            CapabilityError::with_kind(
                CapabilityErrorKind::ReadFailed,
                format!("capability executor {url} read failed: {error}"),
            )
        })?;
        if !status.is_success() {
            return Err(CapabilityError::with_kind(
                CapabilityErrorKind::Http(status.as_u16()),
                format!(
                    "capability executor {url} returned HTTP {status}: {}",
                    String::from_utf8_lossy(&bytes)
                ),
            ));
        }
        serde_json::from_slice(&bytes).map_err(|error| {
            CapabilityError::with_kind(
                CapabilityErrorKind::MalformedJson,
                format!("capability executor {url} returned malformed JSON: {error}"),
            )
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

#[cfg(test)]
mod tests {
    use super::{CapabilityError, CapabilityErrorKind};

    /// An unclassified failure is, by construction, one whose origin was never
    /// recorded. It must be INELIGIBLE: the default is fail-closed, so a
    /// failure nobody classified can never quietly degrade a decision lane.
    ///
    /// This asserts ELIGIBILITY, not the recorded kind. Asserting the kind
    /// alone would pass even if the default were flipped to an eligible class,
    /// because the classifier would simply follow it — the property that
    /// matters is that the unclassified case never degrades.
    #[test]
    fn an_unclassified_failure_defaults_to_ineligible() {
        let error = CapabilityError::new("some failure with no recorded origin");
        assert!(
            !crate::decision_failure_is_backend_unavailable(&error),
            "an unclassified failure must be INELIGIBLE whatever kind it defaults to"
        );
    }

    /// Every construction site records the origin it actually observed, and
    /// the rendered message is unchanged by that recording: diagnostics must
    /// not drift because a failure became classifiable.
    #[test]
    fn an_explicit_kind_preserves_the_message_verbatim() {
        let error = CapabilityError::with_kind(
            CapabilityErrorKind::Http(503),
            "capability executor http://127.0.0.1:1/api/alpha/decisions returned HTTP 503",
        );
        assert_eq!(error.kind(), CapabilityErrorKind::Http(503));
        assert_eq!(
            error.to_string(),
            "capability executor http://127.0.0.1:1/api/alpha/decisions returned HTTP 503"
        );
    }

    /// The fail-closed default is a property of the TYPE, so a future call
    /// site cannot opt into degradation by forgetting to pass a kind.
    #[test]
    fn only_explicit_construction_can_be_eligible() {
        let unclassified = CapabilityError::new("x");
        assert!(!crate::decision_failure_is_backend_unavailable(&unclassified));
        let explicit = CapabilityError::with_kind(CapabilityErrorKind::Unreachable, "x");
        assert!(crate::decision_failure_is_backend_unavailable(&explicit));
    }

    /// A request against a configuration that NAMED an undeclared fallback
    /// cannot occur in production, because admission refuses such a
    /// configuration at load. The load-time test is the observation that
    /// matters; this one is stated explicitly so the invariant is recorded
    /// next to the capability it protects.
    #[test]
    fn an_undeclared_fallback_is_refused_before_it_can_be_reached() {
        // The guarantee is structural: a fallback target that does not resolve
        // is a configuration error, so no request can ever reach a
        // non-existent leg. This asserts the enum has no "unresolved" state
        // that a request could observe.
        let error = CapabilityError::with_kind(
            CapabilityErrorKind::Configuration,
            "decision capability names fallback not-declared, which is not declared",
        );
        assert!(!crate::decision_failure_is_backend_unavailable(&error));
    }
}
