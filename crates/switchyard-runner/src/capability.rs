// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Capability-specific route configuration schema (`[capability_clients.*]`
//! and `[capabilities.*]` root tables).
//!
//! These are NOT LLM chat capabilities. They declare typed utility endpoints
//! (OpenAI-compatible `POST /v1/embeddings` and Cohere/Jina-compatible
//! `POST /v1/rerank`) that the server proxies to a single canonical executor
//! each, with contract/bounds admission enforced at the route boundary. The
//! parsed schema lives here so the deployment file is the single config
//! surface; the executor client construction and request validation live in
//! the serving host.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::config::HttpBaseUrl;

/// Formats a capability client speaks with its executor.
#[derive(Clone, Copy, Debug, Deserialize)]
pub enum CapabilityClientFormat {
    #[serde(rename = "openai_embeddings")]
    OpenAiEmbeddings,
    #[serde(rename = "cohere_jina_rerank")]
    CohereJinaRerank,
}

/// Declares one capability executor target (`[capability_clients.<name>]`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityClientConfig {
    pub format: CapabilityClientFormat,
    pub base_url: HttpBaseUrl,
    pub model: String,
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_capability_timeout_seconds")]
    pub timeout_seconds: u64,
}

const fn default_capability_timeout_seconds() -> u64 {
    30
}

/// Capability-specific route semantics (non-LLM).
///
/// Discrimination happens in [`CapabilityRouteConfig::deserialize`] by
/// required-field presence (Embedding requires `contract`+`dimensions`;
/// Rerank requires none of the embedding fields).
#[derive(Clone, Debug)]
pub enum CapabilityKind {
    Embedding {
        /// Logical embedding-space contract identity (e.g.
        /// `localclaw-embedding-space:v1`).
        contract: String,
        /// Expected vector dimension; responses that differ fail closed.
        dimensions: usize,
        /// Expected normalization (informational admission field).
        normalization: String,
        /// Maximum inputs per request.
        max_batch: usize,
    },
    Rerank {
        /// Maximum documents per request.
        max_candidates: usize,
        /// Default `top_n` when the caller omits it.
        top_n: usize,
        /// Maximum characters per document.
        max_doc_chars: usize,
        /// Maximum characters for the query.
        max_query_chars: usize,
    },
}

/// Capability-specific route declaration (`[capabilities.<name>]`).
///
/// Custom deserializer: `[capabilities.embed]` carries `id`/`target` plus
/// either the Embedding fields (`contract`, `dimensions`, …) or the Rerank
/// fields (`max_candidates`, …). Presence of `contract` selects Embedding;
/// otherwise Rerank. Unknown fields fail closed.
#[derive(Clone, Debug)]
pub struct CapabilityRouteConfig {
    /// Route id served to callers (e.g. `localclaw/embed`, `localclaw/rerank`).
    pub id: String,
    /// Name of the `[capability_clients.*]` executor this route targets.
    pub target: String,
    /// Capability-specific admission contract.
    pub kind: CapabilityKind,
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

fn default_normalization() -> String {
    "L2".to_string()
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
