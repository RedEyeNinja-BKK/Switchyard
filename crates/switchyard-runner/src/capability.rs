// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Capability-specific route configuration schema (`[capability_clients.*]`
//! and `[capabilities.*]` root tables).
//!
//! These are NOT LLM chat capabilities. They declare typed utility endpoints
//! (OpenAI-compatible `POST /v1/embeddings` and Cohere/Jina-compatible
//! `POST /v1/rerank`) that the host proxies to a single canonical executor
//! each, with contract/bounds admission enforced at the route boundary. The
//! parsed schema lives here so the deployment file is the single config
//! surface; the executor client construction and request validation live in
//! the serving host.
//!
//! A capability route's `target` names a `[capability_clients.*]` key, not a
//! `[targets.*]` key. The two key spaces are disjoint: no live capability
//! target resolves in `[targets.*]`, and no `[targets.*]` entry is an
//! embedding or rerank executor. Nothing in this module participates in
//! `FleetRouter` candidate selection.

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
    /// A typed decision surface (`POST /api/alpha/decisions`): the provider
    /// answers with structured `noul` / `choice` / `score` objects rather than
    /// generated assistant text. Deliberately NOT a chat/Responses model — a
    /// decision model must never be declared as one.
    #[serde(rename = "openrouter_alpha_decisions")]
    OpenRouterAlphaDecisions,
    /// The generative half of a typed decision lane: a normal chat/Responses
    /// model whose generated text is normalized into the SAME decision contract
    /// by the decision endpoint. It is a real generative model, so it is
    /// declared as one, but it is only ever reached through that contract.
    #[serde(rename = "openai_responses_decision_adapter")]
    OpenAiResponsesDecisionAdapter,
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
    /// Overrides the path the executor is called on: a single absolute path
    /// APPENDED to `base_url` (e.g. `/v1/systemone`).
    ///
    /// This exists so the decision surface is not welded to one provider's
    /// path shape: `openrouter_alpha_decisions` otherwise hardcodes
    /// `/api/alpha/decisions`, so a second decision backend serving the same
    /// typed contract under its own path could not be declared without a
    /// provider-specific client. A bare path only — an absolute URL, or one
    /// carrying a scheme or host, is rejected at load, so an override can
    /// never silently redirect the capability to a different host.
    ///
    /// Honored by the decision formats only. The embedding and rerank paths
    /// are part of those contracts, not a transport detail.
    pub endpoint_path: Option<String>,
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
    /// A typed decision lane: the provider answers with structured
    /// `noul` / `choice` / `score` objects over a caller-supplied `state`.
    Decisions {
        /// Typed decision-contract identity shared by the primary decision model
        /// and its generative fallback, e.g. `switchyard-decision:v1`.
        contract: String,
        /// Maximum questions admitted in one decision request.
        max_questions: usize,
        /// Maximum characters admitted for the `state` under assessment.
        max_state_chars: usize,
        /// Optional ONE-HOP fallback capability target, used only when the
        /// primary target fails in a way classified as backend unavailability
        /// (see `decision_failure_is_backend_unavailable`). `None` preserves
        /// the pre-fallback behaviour exactly: one target, one invocation,
        /// error returned.
        ///
        /// One hop means one hop: a fallback capability must not itself declare
        /// a `fallback_target`, and a capability may not name itself. Both are
        /// rejected at configuration-admission time rather than detected
        /// recursively at request time.
        fallback_target: Option<String>,
        /// Optional per-question partition of the ONE-HOP fallback.
        ///
        /// Absent (the default) preserves the historical behaviour exactly: the
        /// fallback leg receives the caller's COMPLETE question object. When
        /// present, every question is still answered by the fallback, but each is
        /// routed to one of the declared partition executors instead, so a single
        /// fallback event may fan out across those executors.
        ///
        /// Partitioning is applied ONLY inside the eligible-fallback branch, so the
        /// healthy primary path is untouched: one primary call, no partition work,
        /// no additional backend call.
        ///
        /// The bounded contract this preserves (see `DecisionPartitionTarget`):
        /// ROUTING STAGES per request stay at two (primary, then exactly one
        /// assigned fallback executor per question), while the number of PHYSICAL
        /// backend calls may exceed two because one stage may fan out across the
        /// declared partition targets. There is never a third stage, no recursive
        /// fallback, and no target chaining inside a partition.
        question_partition: Option<Vec<DecisionPartitionTarget>>,
    },
}

/// One declared executor group of a decision fallback partition.
///
/// `question_ids` is matched on the caller's question KEY (stable identity), never
/// on prose. A question matching no group falls to the capability's
/// `fallback_target`, which is therefore the DEFAULT executor: an unmapped or newly
/// added question keeps historical behaviour rather than being dropped.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionPartitionTarget {
    /// Fallback capability id that answers the listed questions.
    pub target: String,
    /// Caller question keys this executor answers.
    pub question_ids: Vec<String>,
}

/// Capability-specific route declaration (`[capabilities.<name>]`).
///
/// Custom deserializer: `[capabilities.embed]` carries `id`/`target` plus
/// either the Embedding fields (`contract`, `dimensions`, …) or the Rerank
/// fields (`max_candidates`, …). Presence of `contract` or `dimensions`
/// selects Embedding; otherwise Rerank. Unknown fields fail closed.
#[derive(Clone, Debug)]
pub struct CapabilityRouteConfig {
    /// Route id served to callers (e.g. `switchyard-smartlocal-embedding`).
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
            decision_contract: Option<String>,
            max_questions: Option<usize>,
            max_state_chars: Option<usize>,
            fallback_target: Option<String>,
            question_partition: Option<Vec<DecisionPartitionTarget>>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let kind = if raw.decision_contract.is_some() {
            let contract = raw.decision_contract.ok_or_else(|| {
                serde::de::Error::custom("decisions capability requires `decision_contract`")
            })?;
            CapabilityKind::Decisions {
                contract,
                max_questions: raw.max_questions.unwrap_or_else(default_max_questions),
                max_state_chars: raw.max_state_chars.unwrap_or_else(default_max_state_chars),
                fallback_target: raw.fallback_target,
                question_partition: raw.question_partition,
            }
        } else if raw.contract.is_some() || raw.dimensions.is_some() {
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

const fn default_max_questions() -> usize {
    16
}

const fn default_max_state_chars() -> usize {
    65536
}

const fn default_max_doc_chars() -> usize {
    4096
}

const fn default_max_query_chars() -> usize {
    2048
}
