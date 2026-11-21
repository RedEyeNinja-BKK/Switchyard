// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for specialized capability surfaces: embeddings and reranking.
//!
//! These are not completion routes. They expose a contract (an embedding space, a
//! reranking budget) over an HTTP endpoint, and they deliberately carry no routing,
//! eligibility or fallback machinery: an embedding request either satisfies its
//! declared contract or fails.
//!
//! Two embedding clients that agree on `dimensions` are still not interchangeable.
//! Vectors are only comparable within the space that produced them, so the contract
//! id travels with the client and is reported to callers rather than inferred.

use std::collections::BTreeMap;

use serde::Deserialize;

/// One specialized capability endpoint.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityClientConfig {
    /// Wire contract this endpoint speaks.
    pub format: CapabilityFormat,
    /// Base URL of the capability service.
    pub base_url: String,
    /// Provider-side model or engine identifier.
    pub model: String,
    /// Path appended to `base_url` when the contract is not at its default path.
    #[serde(default)]
    pub endpoint_path: Option<String>,
    /// Environment variable holding this client's credential.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Deadline for one call, in seconds. Unset is the client's own default.
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    /// Task type this lane embeds for, sent as the provider's own field.
    ///
    /// Only `google_embed_content` has this field. It changes the vectors a query is
    /// mapped to, so it is part of the capability's vector space and a contract id must
    /// change with it.
    #[serde(default)]
    pub task_type: Option<String>,
}

impl CapabilityClientConfig {
    /// Full request URL for this client.
    pub fn url(&self, default_path: &str) -> String {
        let path = self
            .endpoint_path
            .clone()
            .unwrap_or_else(|| default_path.to_string());
        format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            if path.starts_with('/') {
                path
            } else {
                format!("/{path}")
            }
        )
    }
}

/// Wire contract a specialized endpoint speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityFormat {
    /// OpenAI-compatible `POST /embeddings`.
    OpenaiEmbeddings,
    /// Cohere/Jina-compatible reranking over `POST /rerank`.
    CohereJinaRerank,
    /// Google `POST /v1beta/models/{model}:embedContent`.
    ///
    /// One content per request: the endpoint has no bulk form, so a batch is a bounded
    /// fan-out of single-input calls rather than one wide request.
    GoogleEmbedContent,
}

/// The header a capability credential is presented under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityAuth {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `x-goog-api-key: <key>`.
    GoogleApiKey,
}

impl CapabilityFormat {
    /// Whether this format produces embedding vectors.
    pub fn serves_embeddings(self) -> bool {
        matches!(self, Self::OpenaiEmbeddings | Self::GoogleEmbedContent)
    }

    /// Whether this format produces rerank scores.
    pub fn serves_rerank(self) -> bool {
        matches!(self, Self::CohereJinaRerank)
    }

    /// How this format's credential is presented.
    ///
    /// The header is a property of the wire, not a preference: Google reads its key from
    /// `x-goog-api-key` and a bearer token there is an authentication failure that looks
    /// like a bad credential.
    pub fn auth(self) -> CapabilityAuth {
        match self {
            Self::GoogleEmbedContent => CapabilityAuth::GoogleApiKey,
            Self::OpenaiEmbeddings | Self::CohereJinaRerank => CapabilityAuth::Bearer,
        }
    }

    /// Default request path, when the client declares none.
    ///
    /// Google names the model inside the path, so this is the one format whose default
    /// path depends on the configured model.
    pub fn default_path(self, model: &str) -> String {
        match self {
            Self::OpenaiEmbeddings => "/v1/embeddings".to_string(),
            Self::CohereJinaRerank => "/v1/rerank".to_string(),
            Self::GoogleEmbedContent => format!("/v1beta/models/{model}:embedContent"),
        }
    }
}

/// A caller-visible capability: which endpoint serves it, under which contract.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityConfig {
    /// Capability identifier callers request.
    pub id: String,
    /// Name of the `capability_clients` entry that serves it.
    pub target: String,
    /// Identifier of the value space this capability produces.
    ///
    /// Vectors from different spaces are not comparable, so a change here invalidates
    /// every previously stored vector. Two capabilities may share `dimensions` and still
    /// need distinct contract ids.
    #[serde(default)]
    pub contract: Option<String>,
    /// Vector width this capability produces.
    #[serde(default)]
    pub dimensions: Option<u32>,
    /// Vector normalization applied by the engine.
    #[serde(default)]
    pub normalization: Option<String>,
    /// Largest batch one request may carry.
    #[serde(default)]
    pub max_batch: Option<u32>,
    /// Largest candidate set one rerank request may carry.
    #[serde(default)]
    pub max_candidates: Option<u32>,
    /// Results returned per rerank request.
    #[serde(default)]
    pub top_n: Option<u32>,
    /// Longest document text one rerank request may carry, in characters.
    #[serde(default)]
    pub max_doc_chars: Option<u32>,
    /// Longest query text one rerank request may carry, in characters.
    #[serde(default)]
    pub max_query_chars: Option<u32>,
    /// Input-token limit of one embedding input, as the provider publishes it.
    ///
    /// Required for `google_embed_content`. The surface enforces a byte budget derived
    /// from it, because a provider token limit can only be bounded locally through a
    /// quantity that is provably at least as large as the token count.
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
}

/// Configured capability endpoints and the capabilities that name them.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CapabilityDeployment {
    /// Endpoints, by configuration name.
    #[serde(default)]
    pub clients: BTreeMap<String, CapabilityClientConfig>,
    /// Caller-visible capabilities, by configuration name.
    #[serde(default)]
    pub capabilities: BTreeMap<String, CapabilityConfig>,
}

impl CapabilityDeployment {
    /// Checks that every capability resolves, declares the fields its format requires, and
    /// that distinct clients keep distinct contracts.
    pub fn validate(&self) -> Result<(), String> {
        for (name, capability) in &self.capabilities {
            let client_name = &capability.target;
            let Some(client) = self.clients.get(client_name) else {
                return Err(format!(
                    "capability {name} references unknown capability client {client_name}"
                ));
            };
            match client.format {
                CapabilityFormat::OpenaiEmbeddings => {
                    if capability.dimensions.is_none() {
                        return Err(format!(
                            "capability {name} serves an embedding endpoint and must declare dimensions"
                        ));
                    }
                    if capability.contract.is_none() {
                        return Err(format!(
                            "capability {name} serves an embedding endpoint and must declare contract, \
                             because vectors from different engines are not comparable"
                        ));
                    }
                }
                CapabilityFormat::GoogleEmbedContent => {
                    if capability.dimensions.is_none() {
                        return Err(format!(
                            "capability {name} serves an embedding endpoint and must declare dimensions"
                        ));
                    }
                    if capability.contract.is_none() {
                        return Err(format!(
                            "capability {name} serves an embedding endpoint and must declare contract, \
                             because vectors from different engines are not comparable"
                        ));
                    }
                    // Google publishes a 2,048-token input limit on this model. Without the
                    // number, the surface could only truncate silently or send and hope,
                    // and both are worse than refusing the configuration.
                    if capability.max_input_tokens.is_none() {
                        return Err(format!(
                            "capability {name} serves a google embedding endpoint and must declare \
                             max_input_tokens, because an input longer than the model's limit must be \
                             refused rather than truncated"
                        ));
                    }
                }
                CapabilityFormat::CohereJinaRerank => {
                    if capability.max_candidates.is_none() {
                        return Err(format!(
                            "capability {name} serves a rerank endpoint and must declare max_candidates"
                        ));
                    }
                    if let (Some(top_n), Some(max_candidates)) =
                        (capability.top_n, capability.max_candidates)
                        && top_n > max_candidates
                    {
                        return Err(format!(
                            "capability {name} top_n {top_n} exceeds max_candidates {max_candidates}"
                        ));
                    }
                }
            }

            // A task type is a Google wire field. Declaring it on a client that has no
            // such field would read as a configured behaviour the deployment does not
            // have, so it is refused instead of ignored.
            if client.task_type.is_some() && client.format != CapabilityFormat::GoogleEmbedContent {
                return Err(format!(
                    "capability client {client_name} declares task_type, which only \
                     google_embed_content sends"
                ));
            }
        }

        // One engine serving two capabilities may be deliberate, but two engines sharing
        // one contract id would make stored vectors ambiguous, so the id must be unique
        // per engine rather than per capability name.
        let mut contract_engines: BTreeMap<&str, &str> = BTreeMap::new();
        for capability in self.capabilities.values() {
            let Some(contract) = capability.contract.as_deref() else {
                continue;
            };
            let engine = &capability.target;
            if let Some(previous) = contract_engines.insert(contract, engine)
                && previous != engine
            {
                return Err(format!(
                    "capabilities share embedding contract {contract} but name different clients \
                     {previous} and {engine}; vectors from different engines are not comparable, \
                     so each contract belongs to exactly one client"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deployment(source: &str) -> CapabilityDeployment {
        toml::from_str(source).expect("capability fixture parses")
    }

    const CLIENTS: &str = r#"
[clients.engine_a]
format = "openai_embeddings"
base_url = "https://engine-a.test"
model = "a"
"#;

    #[test]
    fn an_embedding_capability_must_declare_dimensions_and_a_contract() {
        let missing_contract = deployment(&format!(
            "{CLIENTS}\n[capabilities.space]\nid = \"space\"\ntarget = \"engine_a\"\ndimensions = 1024\n"
        ));
        assert!(
            missing_contract
                .validate()
                .expect_err("a contract is required")
                .contains("must declare contract")
        );

        let missing_dimensions = deployment(&format!(
            "{CLIENTS}\n[capabilities.space]\nid = \"space\"\ntarget = \"engine_a\"\ncontract = \"space:v1\"\n"
        ));
        assert!(
            missing_dimensions
                .validate()
                .expect_err("dimensions are required")
                .contains("must declare dimensions")
        );

        let complete = deployment(&format!(
            "{CLIENTS}\n[capabilities.space]\nid = \"space\"\ntarget = \"engine_a\"\ncontract = \"space:v1\"\ndimensions = 1024\n"
        ));
        complete
            .validate()
            .expect("a complete embedding capability is valid");
    }

    #[test]
    fn two_spaces_of_equal_width_keep_separate_contracts() {
        // The real case: both engines emit 1024-dimension vectors, which is exactly why
        // width cannot stand in for identity.
        let source = format!(
            "{CLIENTS}\n[clients.engine_b]\nformat = \"openai_embeddings\"\nbase_url = \"https://engine-b.test\"\nmodel = \"b\"\n\
             \n[capabilities.space_a]\nid = \"space_a\"\ntarget = \"engine_a\"\ncontract = \"space:a:v1\"\ndimensions = 1024\n\
             \n[capabilities.space_b]\nid = \"space_b\"\ntarget = \"engine_b\"\ncontract = \"space:b:v1\"\ndimensions = 1024\n"
        );
        deployment(&source)
            .validate()
            .expect("distinct contracts are valid");

        let shared = format!(
            "{CLIENTS}\n[clients.engine_b]\nformat = \"openai_embeddings\"\nbase_url = \"https://engine-b.test\"\nmodel = \"b\"\n\
             \n[capabilities.space_a]\nid = \"space_a\"\ntarget = \"engine_a\"\ncontract = \"space:v1\"\ndimensions = 1024\n\
             \n[capabilities.space_b]\nid = \"space_b\"\ntarget = \"engine_b\"\ncontract = \"space:v1\"\ndimensions = 1024\n"
        );
        assert!(
            deployment(&shared)
                .validate()
                .expect_err("one contract cannot name two engines")
                .contains("not comparable")
        );
    }

    #[test]
    fn rerank_limits_are_carried_and_bounded() {
        let source = r#"
[clients.reranker]
format = "cohere_jina_rerank"
base_url = "https://reranker.test"
model = "r"

[capabilities.rerank]
id = "rerank"
target = "reranker"
max_candidates = 64
top_n = 5
max_doc_chars = 1024
max_query_chars = 512
"#;
        let loaded = deployment(source);
        loaded
            .validate()
            .expect("a complete rerank capability is valid");
        let rerank = &loaded.capabilities["rerank"];
        assert_eq!(rerank.max_candidates, Some(64));
        assert_eq!(rerank.top_n, Some(5));
        assert_eq!(rerank.max_doc_chars, Some(1024));
        assert_eq!(rerank.max_query_chars, Some(512));

        let inverted = r#"
[clients.reranker]
format = "cohere_jina_rerank"
base_url = "https://reranker.test"
model = "r"

[capabilities.rerank]
id = "rerank"
target = "reranker"
max_candidates = 2
top_n = 5
"#;
        assert!(
            deployment(inverted)
                .validate()
                .expect_err("top_n cannot exceed max_candidates")
                .contains("exceeds max_candidates")
        );
    }

    #[test]
    fn an_unknown_target_fails_rather_than_serving_nothing() {
        let dangling = r#"
[clients.engine_a]
format = "openai_embeddings"
base_url = "https://engine-a.test"
model = "a"

[capabilities.space]
id = "space"
target = "missing"
contract = "space:v1"
dimensions = 1024
"#;
        assert!(
            deployment(dangling)
                .validate()
                .expect_err("a dangling target must not load")
                .contains("unknown capability client")
        );
    }

    #[test]
    fn an_explicit_path_overrides_the_format_default() {
        let client = CapabilityClientConfig {
            format: CapabilityFormat::CohereJinaRerank,
            base_url: "https://service.test/".to_string(),
            model: "m".to_string(),
            endpoint_path: Some("/v1/rerank".to_string()),
            api_key_env: None,
            timeout_seconds: None,
            task_type: None,
        };
        assert_eq!(client.url("/rerank"), "https://service.test/v1/rerank");

        let default_path = CapabilityClientConfig {
            endpoint_path: None,
            ..client
        };
        assert_eq!(default_path.url("/rerank"), "https://service.test/rerank");
    }
}
/// A capability resolved for serving: the endpoint, its contract, and its limits.
///
/// This is the runtime form. `Deployment::validate` checks configuration; this is what
/// a serving surface dispatches against, so a contract that validated cannot be lost
/// between load and dispatch.
#[derive(Clone, Debug)]
pub struct ResolvedCapability {
    /// Caller-visible capability id.
    pub id: String,
    /// Configuration name of the serving client.
    pub client: String,
    /// Wire contract the client speaks.
    pub format: CapabilityFormat,
    /// Full request URL.
    pub url: String,
    /// Provider-side model or engine id.
    pub model: String,
    /// Environment variable holding the credential, when the client needs one.
    pub api_key_env: Option<String>,
    /// Deadline for one call to this engine. Unset uses the surface default.
    pub timeout_seconds: Option<u64>,
    /// Caller-visible contract identity, for embeddings.
    pub contract: Option<String>,
    /// Vector width, for embeddings.
    pub dimensions: Option<u32>,
    /// Vector normalization, for embeddings.
    pub normalization: Option<String>,
    /// Largest batch one request may carry, for embeddings.
    pub max_batch: Option<u32>,
    /// Largest candidate set, for reranking.
    pub max_candidates: Option<u32>,
    /// Results returned per rerank request.
    pub top_n: Option<u32>,
    /// Longest document text, in characters, for reranking.
    pub max_doc_chars: Option<u32>,
    /// Longest query text, in characters, for reranking.
    pub max_query_chars: Option<u32>,
    /// Task type sent as the provider's own field, for Google embedding.
    pub task_type: Option<String>,
    /// Provider-published input-token limit of one embedding input.
    pub max_input_tokens: Option<u32>,
}

/// Resolves configuration into the serving form, so dispatch never re-reads config.
pub fn resolve(deployment: &CapabilityDeployment) -> BTreeMap<String, ResolvedCapability> {
    deployment
        .capabilities
        .iter()
        .filter_map(|(name, capability)| {
            let client = deployment.clients.get(&capability.target)?;
            let default_path = client.format.default_path(&client.model);
            Some((
                capability.id.clone(),
                ResolvedCapability {
                    id: capability.id.clone(),
                    client: name.clone(),
                    format: client.format,
                    url: client.url(&default_path),
                    model: client.model.clone(),
                    api_key_env: client.api_key_env.clone(),
                    timeout_seconds: client.timeout_seconds,
                    contract: capability.contract.clone(),
                    dimensions: capability.dimensions,
                    normalization: capability.normalization.clone(),
                    max_batch: capability.max_batch,
                    max_candidates: capability.max_candidates,
                    top_n: capability.top_n,
                    max_doc_chars: capability.max_doc_chars,
                    max_query_chars: capability.max_query_chars,
                    task_type: client.task_type.clone(),
                    max_input_tokens: capability.max_input_tokens,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    const SOURCE: &str = r#"
[clients.engine_a]
format = "openai_embeddings"
base_url = "https://a.test"
model = "Qwen3-Embedding-0.6B"

[clients.engine_b]
format = "openai_embeddings"
base_url = "https://b.test"
model = "lfm"

[clients.reranker]
format = "cohere_jina_rerank"
base_url = "https://r.test"
model = "qwen3-reranker"
timeout_seconds = 120

[capabilities.space_a]
id = "switchyard-embedding-local"
target = "engine_a"
contract = "localclaw-embedding-space:v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.space_b]
id = "switchyard-embedding-free"
target = "engine_b"
contract = "localclaw-embedding-space:lfm2.5-350m-v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.rerank]
id = "switchyard-rerank-local"
target = "reranker"
max_candidates = 64
top_n = 5
max_doc_chars = 1024
max_query_chars = 512
"#;

    fn resolved() -> BTreeMap<String, ResolvedCapability> {
        let deployment: CapabilityDeployment = toml::from_str(SOURCE).expect("fixture parses");
        deployment.validate().expect("fixture is valid");
        resolve(&deployment)
    }

    /// The point of keeping two embedding capabilities is that dispatch must be able to
    /// tell them apart. Equal width is not identity, so the contract must survive.
    #[test]
    fn dispatch_distinguishes_two_embedding_spaces_of_equal_width() {
        let resolved = resolved();
        let local = &resolved["switchyard-embedding-local"];
        let free = &resolved["switchyard-embedding-free"];
        assert_eq!(local.dimensions, free.dimensions);
        assert_ne!(local.contract, free.contract);
        assert_eq!(
            local.contract.as_deref(),
            Some("localclaw-embedding-space:v1")
        );
        assert_eq!(
            free.contract.as_deref(),
            Some("localclaw-embedding-space:lfm2.5-350m-v1")
        );
        assert_ne!(local.url, free.url, "each space reaches its own engine");
    }

    #[test]
    fn dispatch_carries_the_rerank_contract_and_limits() {
        let resolved = resolved();
        let rerank = &resolved["switchyard-rerank-local"];
        assert_eq!(rerank.max_candidates, Some(64));
        assert_eq!(rerank.top_n, Some(5));
        assert_eq!(rerank.max_doc_chars, Some(1024));
        assert_eq!(rerank.max_query_chars, Some(512));
        assert_eq!(rerank.format, CapabilityFormat::CohereJinaRerank);
        assert_eq!(rerank.url, "https://r.test/v1/rerank");
    }

    /// A capability naming a missing client must not appear at all: a surface that
    /// resolved to nothing would otherwise report an empty capability rather than fail.
    #[test]
    fn an_unresolvable_capability_is_absent_rather_than_empty() {
        let source = r#"
[clients.engine_a]
format = "openai_embeddings"
base_url = "https://a.test"
model = "a"

[capabilities.space]
id = "space"
target = "engine_a"
contract = "space:v1"
dimensions = 8

[capabilities.dangling]
id = "dangling"
target = "missing"
contract = "other:v1"
dimensions = 8
"#;
        let deployment: CapabilityDeployment = toml::from_str(source).expect("parses");
        assert!(
            deployment.validate().is_err(),
            "load-time validation must catch it"
        );
        let resolved = resolve(&deployment);
        assert!(resolved.contains_key("space"));
        assert!(!resolved.contains_key("dangling"));
    }
}

#[cfg(test)]
mod google_embedding_tests {
    use super::*;

    const SOURCE: &str = r#"
[clients.google]
format = "google_embed_content"
base_url = "https://generativelanguage.googleapis.com/"
model = "gemini-embedding-001"
task_type = "RETRIEVAL_DOCUMENT"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"

[capabilities.gem]
id = "switchyard-embedding-google"
target = "google"
contract = "google-embedding-001:v1"
dimensions = 3072
normalization = "L2"
max_batch = 8
max_input_tokens = 2048
"#;

    fn deployment() -> CapabilityDeployment {
        let deployment: CapabilityDeployment = toml::from_str(SOURCE).expect("fixture parses");
        deployment.validate().expect("fixture is valid");
        deployment
    }

    /// Google names the model in the path, so the default path is a function of the
    /// configured model rather than a constant. A bare `/v1beta/embedContent` would 404.
    #[test]
    fn the_default_path_names_the_model_and_the_method() {
        let resolved = resolve(&deployment());
        assert_eq!(
            resolved["switchyard-embedding-google"].url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        );
    }

    /// The task type changes the vectors a text is mapped to, so it is part of the vector
    /// space and must survive into dispatch; dropping it would silently serve a different
    /// space than the contract id claims.
    #[test]
    fn dispatch_carries_the_task_type_and_the_input_limit() {
        let resolved = resolve(&deployment());
        let capability = &resolved["switchyard-embedding-google"];
        assert_eq!(capability.task_type.as_deref(), Some("RETRIEVAL_DOCUMENT"));
        assert_eq!(capability.max_input_tokens, Some(2048));
        assert_eq!(capability.format.auth(), CapabilityAuth::GoogleApiKey);
    }

    /// Without the published input limit the surface could only truncate silently or
    /// send and hope, so the configuration is refused rather than guessed.
    #[test]
    fn a_google_capability_must_declare_its_input_token_limit() {
        let source = SOURCE.replace("max_input_tokens = 2048\n", "");
        let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("an undeclared input limit must not load")
                .contains("max_input_tokens")
        );
    }

    /// `task_type` is a Google wire field. Accepting it on another client would describe
    /// behaviour the deployment does not have.
    #[test]
    fn a_task_type_on_another_wire_is_refused_rather_than_ignored() {
        let source = SOURCE.replace(
            "format = \"google_embed_content\"",
            "format = \"openai_embeddings\"",
        );
        let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("an inert field must not load")
                .contains("task_type")
        );
    }

    /// The transport identity is a property of the wire: Google rejects a bearer token.
    #[test]
    fn only_the_google_wire_presents_its_key_as_a_google_key() {
        assert_eq!(
            CapabilityFormat::GoogleEmbedContent.auth(),
            CapabilityAuth::GoogleApiKey
        );
        assert_eq!(
            CapabilityFormat::OpenaiEmbeddings.auth(),
            CapabilityAuth::Bearer
        );
        assert_eq!(
            CapabilityFormat::CohereJinaRerank.auth(),
            CapabilityAuth::Bearer
        );
    }

    /// The two serving surfaces must stay disjoint, or a rerank call could reach an
    /// embedding engine.
    #[test]
    fn google_serves_embeddings_and_not_rerank() {
        assert!(CapabilityFormat::GoogleEmbedContent.serves_embeddings());
        assert!(!CapabilityFormat::GoogleEmbedContent.serves_rerank());
        assert!(!CapabilityFormat::CohereJinaRerank.serves_embeddings());
    }
}
