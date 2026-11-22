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
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use switchyard_llm_client::{
    DailyRequestAllowance, GovernorKey, PacedQuota, QuotaGovernor, WindowLimits, allowance_path,
};

/// Widest vector `gemini-embedding-2` can produce.
pub const GEMINI_EMBEDDING_V2_MAX_DIMENSIONS: u32 = 3072;

/// Narrowest vector `gemini-embedding-2` can produce.
pub const GEMINI_EMBEDDING_V2_MIN_DIMENSIONS: u32 = 128;

/// Input-token limit `gemini-embedding-2` publishes.
pub const GEMINI_EMBEDDING_V2_INPUT_TOKENS: u32 = 8192;

/// Maximum thinking budget Google's Live API accepts, in tokens.
pub const GEMINI_LIVE_MAX_THINKING_BUDGET: u32 = 32768;

/// Whether a string is shaped like a prebuilt voice name.
///
/// The catalogue belongs to the provider, so this checks shape only: a non-empty
/// letters/digits/hyphen/underscore token. An unknown but well-formed name travels to the
/// engine and comes back as the engine's own error, which is more truthful than a locally
/// invented list that would refuse a voice the provider has added.
pub fn is_voice_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

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
    /// Provider-side project whose counters this client's calls are charged to.
    ///
    /// The provider caps requests per project and physical model, so two clients that name
    /// one project and one model are one budget. Unset means this client is its own group,
    /// which keeps a single-client deployment unchanged.
    #[serde(default)]
    pub quota_group: Option<String>,
    /// Requests permitted per provider day for this client's physical model.
    ///
    /// Enforced as one counter per (quota group, model), persisted under `[quota]
    /// state_dir`, so every capability and alias of the model draws on one budget. A
    /// bounded fan-out spends one unit per physical call, not one per caller request.
    #[serde(default)]
    pub daily_request_limit: Option<u64>,
    /// Physical calls permitted per rolling minute for this client's model.
    #[serde(default)]
    pub requests_per_minute: Option<u64>,
    /// Input tokens permitted per rolling minute for this client's model.
    ///
    /// An embedding call reserves its input's byte count, which is an upper bound on its
    /// token count, so the window is never over-spent by an estimate that undercounts.
    #[serde(default)]
    pub input_tokens_per_minute: Option<u64>,
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
    /// This adapter sends one content per request. The synchronous
    /// `:batchEmbedContents` form also exists and takes a `requests` array, but its
    /// availability on a Free-tier project and its response shape are not qualified here,
    /// so a caller's batch is a bounded fan-out of single-content calls rather than one
    /// wide request. "No bulk form" is not a statement about Google's API.
    GoogleEmbedContent,
    /// Google `:embedContent` as `gemini-embedding-2` speaks it.
    ///
    /// A different codec, not a version bump: output width is a request field
    /// (`output_dimensionality`, 128..=3072) rather than a fixed property of the model,
    /// `task_type` is unsupported so a task has to travel as a text prefix, the input limit
    /// is 8,192 tokens, and vectors below full width are auto-normalised by the model.
    /// Vectors from this dialect are not comparable with `google_embed_content` output.
    GoogleEmbedContentV2,
    /// Google `:generateContent` asked for an AUDIO modality: speech synthesis.
    ///
    /// Not a chat lane with a different output type. The request carries the text and a
    /// voice, the response carries one `inlineData` blob of raw PCM, and a TTS model has its
    /// own tiny request budget (three per minute, ten per day on this Free project). Serving
    /// it as chat text — or falling back to a chat model when it is unavailable — would spend
    /// the wrong quota and hand back something the caller did not ask for.
    GoogleSpeech,
    /// Google `:generateContent` that reads inline audio and answers text.
    ///
    /// The audio is a typed payload (container + bytes), not a chat attachment, so the
    /// container is part of the contract: the surface parses it to bound duration and token
    /// cost, and refuses a container it cannot measure rather than forwarding an unbounded
    /// request.
    GoogleTranscribe,
    /// Google's stateful bidirectional Live API (`BidiGenerateContent`) over WebSocket.
    ///
    /// Not a stateless HTTP generateContent call: the first message over the connection must
    /// be a `BidiGenerateContentSetup`, and both sides stream JSON frames until one side
    /// closes. The credential travels in the WebSocket URL as a `key=` query parameter, so it
    /// is never rendered into client-visible request bodies or logs.
    ///
    /// Model-specific fields (response modalities, thinking config, system instruction) are
    /// declared per capability so the setup message is built locally and the caller cannot
    /// override them.
    GoogleLive,
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
        matches!(
            self,
            Self::OpenaiEmbeddings | Self::GoogleEmbedContent | Self::GoogleEmbedContentV2
        )
    }

    /// Whether this format synthesises speech from text.
    pub fn serves_speech(self) -> bool {
        matches!(self, Self::GoogleSpeech)
    }

    /// Whether this format transcribes audio into text.
    pub fn serves_transcription(self) -> bool {
        matches!(self, Self::GoogleTranscribe)
    }

    /// Whether this format produces rerank scores.
    pub fn serves_rerank(self) -> bool {
        matches!(self, Self::CohereJinaRerank)
    }

    /// How this format's credential is presented.
    ///
    /// The credential is a property of the wire, not a preference: Google reads its Live key
    /// from the WebSocket URL's `key=` query parameter, and its HTTP surfaces from
    /// `x-goog-api-key`; a bearer token on either wire is an authentication failure that
    /// looks like a bad credential.
    pub fn auth(self) -> CapabilityAuth {
        match self {
            Self::GoogleEmbedContent
            | Self::GoogleEmbedContentV2
            | Self::GoogleSpeech
            | Self::GoogleTranscribe
            | Self::GoogleLive => CapabilityAuth::GoogleApiKey,
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
            Self::GoogleEmbedContentV2 => format!("/v1beta/models/{model}:embedContent"),
            Self::GoogleSpeech | Self::GoogleTranscribe => {
                format!("/v1beta/models/{model}:generateContent")
            }
            Self::GoogleLive => {
                // The Live WebSocket endpoint is not an HTTP path: it names the generative
                // service method and carries the credential as a query parameter. The surface
                // builds the full wss:// URL from base_url, this path and the live key.
                format!(
                    "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
                )
            }
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
    ///
    /// For a Google embedding capability this is a physical-call budget, not a payload
    /// limit: one caller batch becomes this many upstream requests at most.
    #[serde(default)]
    pub max_batch: Option<u32>,
    /// Instruction prefix prepended to every input, for a task-specific space.
    ///
    /// `gemini-embedding-2` has no `task_type` field, so a retrieval task travels as text
    /// in front of the input. Because the prefix changes the vectors, it is part of the
    /// value space and must be declared here — beside the contract it belongs to — rather
    /// than chosen per request. The value is used verbatim: no separator is invented,
    /// so the operator writes the exact prefix the model's documentation prescribes.
    #[serde(default)]
    pub task_prefix: Option<String>,
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
    /// Voice a speech capability speaks in when the caller names none.
    ///
    /// A prebuilt Google voice name, passed through verbatim. The surface validates its
    /// shape, not the catalogue: the provider is the authority on which names exist, and an
    /// unknown name comes back as the provider's own error rather than a guessed list.
    #[serde(default)]
    pub voice: Option<String>,
    /// Output audio container a speech capability serves: `wav` or `pcm`.
    ///
    /// The engine returns raw PCM; `wav` is that same PCM with a RIFF header this surface
    /// writes. Compressed containers are refused because producing them would require an
    /// encoder that is not part of this deployment, and mislabelling PCM as `mp3` would be
    /// worse than refusing.
    #[serde(default)]
    pub audio_format: Option<String>,
    /// Sample rate of the engine's audio output, in hertz.
    ///
    /// Declared so the surface can check the rate the provider reports in its
    /// `inlineData` MIME type against what the caller is promised. A WAV header written at
    /// the wrong rate plays back at the wrong speed, so this is verified rather than assumed.
    #[serde(default)]
    pub audio_sample_rate: Option<u32>,
    /// Longest audio one transcription request may carry, in seconds.
    ///
    /// Audio tokens are proportional to duration, so duration is what the per-minute token
    /// budget is spent in. It is parsed from the container rather than estimated from size:
    /// a size bound would admit hours of low-bitrate audio.
    #[serde(default)]
    pub max_audio_seconds: Option<u32>,
    /// Largest audio payload one transcription request may carry, in raw bytes.
    ///
    /// Separate from the duration bound: this is the provider's inline-request ceiling, and
    /// an over-size body is refused before it is base64-encoded and sent.
    #[serde(default)]
    pub max_audio_bytes: Option<u64>,
    /// Evidence that keeps this capability out of dispatch, quoted from the provider.
    ///
    /// Present means held: the declaration, its contract and its limits are kept and
    /// validated, but the capability is not resolved, so no caller can name it and no
    /// request can reach the engine. A held capability is inventory, not a fallback — the
    /// same rule a held target follows, for the same reason: quietly serving something else
    /// would hand back vectors from a space nobody asked for.
    #[serde(default)]
    pub held: Option<String>,
    /// The response modalities this Live capability produces.
    ///
    /// Declared once per capability so the setup message is built locally and the caller
    /// cannot ask for a modality this deployment does not serve.
    #[serde(default)]
    pub response_modalities: Option<Vec<String>>,
    /// System instruction prepended to every session, part of the capability's contract.
    ///
    /// Travels verbatim in the setup message; no separator is invented.
    #[serde(default)]
    pub system_instruction: Option<String>,
    /// Thinking budget for Live variants that expose it (Extended Thinking).
    ///
    /// `None` means the model's native default; `Some(0)` means disabled. A budget above the
    /// model's published maximum is refused at load rather than sent to fail on every call.
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    /// Thinking level for Live variants that expose a level rather than a token budget.
    ///
    /// Gemini 3.x Live models reject `thinkingBudget` outright (the session closes with
    /// code 1007 "Thinking budget is not supported for this model") and require
    /// `thinkingLevel` instead. The two controls are mutually exclusive: declaring both is
    /// refused, because only one of them would take effect and the other would be a lie
    /// about the session's shape.
    #[serde(default)]
    pub thinking_level: Option<String>,
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
            if capability.held.as_deref().is_some_and(str::is_empty) {
                return Err(format!(
                    "capability {name} declares an empty held reason; state the evidence that \
                     keeps it out of dispatch, or omit the field to serve it"
                ));
            }
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
                CapabilityFormat::GoogleEmbedContentV2 => {
                    if capability.contract.is_none() {
                        return Err(format!(
                            "capability {name} serves an embedding endpoint and must declare contract, \
                             because vectors from different engines are not comparable"
                        ));
                    }
                    // This model's width is a request field, so the declaration is the
                    // output_dimensionality actually sent. Outside the model's range the
                    // engine would refuse every call, and a width the caller was promised
                    // but never received is worse than a load failure.
                    let Some(dimensions) = capability.dimensions else {
                        return Err(format!(
                            "capability {name} serves a gemini-embedding-2 endpoint and must declare \
                             dimensions, which is the output_dimensionality sent with every call"
                        ));
                    };
                    if !(GEMINI_EMBEDDING_V2_MIN_DIMENSIONS..=GEMINI_EMBEDDING_V2_MAX_DIMENSIONS)
                        .contains(&dimensions)
                    {
                        return Err(format!(
                            "capability {name} declares dimensions {dimensions}, outside the \
                             {GEMINI_EMBEDDING_V2_MIN_DIMENSIONS}..={GEMINI_EMBEDDING_V2_MAX_DIMENSIONS} \
                             range gemini-embedding-2 accepts"
                        ));
                    }
                    // Vectors below full width are normalised by the model. Declaring it
                    // is how the surface knows to check the norm it actually receives;
                    // leaving it undeclared would make the value space a claim nobody
                    // verified.
                    if capability.normalization.is_none() {
                        return Err(format!(
                            "capability {name} serves a gemini-embedding-2 endpoint and must declare \
                             normalization, so the received vectors can be checked against it"
                        ));
                    }
                    let Some(max_input_tokens) = capability.max_input_tokens else {
                        return Err(format!(
                            "capability {name} serves a gemini-embedding-2 endpoint and must declare \
                             max_input_tokens, because an input past the model's limit must be refused \
                             rather than truncated"
                        ));
                    };
                    if max_input_tokens > GEMINI_EMBEDDING_V2_INPUT_TOKENS {
                        return Err(format!(
                            "capability {name} declares max_input_tokens {max_input_tokens}, above the \
                             {GEMINI_EMBEDDING_V2_INPUT_TOKENS}-token limit of gemini-embedding-2; a \
                             declared limit above the model's own promises input the engine will refuse"
                        ));
                    }
                }
                CapabilityFormat::GoogleSpeech => {
                    let Some(max_input_tokens) = capability.max_input_tokens else {
                        return Err(format!(
                            "capability {name} serves a speech endpoint and must declare \
                             max_input_tokens, because text past the model's input limit has to be \
                             refused rather than truncated mid-sentence"
                        ));
                    };
                    if max_input_tokens == 0 {
                        return Err(format!(
                            "capability {name} declares max_input_tokens = 0, which would refuse \
                             every request"
                        ));
                    }
                    let Some(voice) = capability.voice.as_deref() else {
                        return Err(format!(
                            "capability {name} serves a speech endpoint and must declare a default \
                             voice, because a request that names none would otherwise have one \
                             invented for it"
                        ));
                    };
                    if !is_voice_name(voice) {
                        return Err(format!(
                            "capability {name} declares voice {voice:?}, which is not a voice name \
                             (letters, digits, hyphen or underscore)"
                        ));
                    }
                    match capability.audio_format.as_deref() {
                        Some("wav") | Some("pcm") => {}
                        Some(other) => {
                            return Err(format!(
                                "capability {name} declares audio_format {other:?}; this surface \
                                 produces wav and pcm only, because any other container would need \
                                 an encoder it does not ship"
                            ));
                        }
                        None => {
                            return Err(format!(
                                "capability {name} serves a speech endpoint and must declare \
                                 audio_format (wav or pcm)"
                            ));
                        }
                    }
                    match capability.audio_sample_rate {
                        Some(0) => {
                            return Err(format!(
                                "capability {name} declares audio_sample_rate = 0, which would \
                                 produce a WAV container with no timing"
                            ));
                        }
                        Some(_) => {}
                        None => {
                            return Err(format!(
                                "capability {name} serves a speech endpoint and must declare \
                                 audio_sample_rate, so the rate the provider reports can be checked \
                                 against the one the caller is promised"
                            ));
                        }
                    }
                }
                CapabilityFormat::GoogleTranscribe => {
                    // The instruction is part of the contract: without it the model is asked
                    // to "do something" with audio, and different callers would silently get
                    // different behaviour from the same capability id.
                    if capability.task_prefix.as_deref().is_none_or(str::is_empty) {
                        return Err(format!(
                            "capability {name} serves a transcription endpoint and must declare \
                             task_prefix, the instruction sent with every recording"
                        ));
                    }
                    if capability.max_audio_seconds.is_none() {
                        return Err(format!(
                            "capability {name} serves a transcription endpoint and must declare \
                             max_audio_seconds, because audio tokens are proportional to duration \
                             and an unbounded input would spend a budget nobody can plan for"
                        ));
                    }
                    if capability.max_audio_bytes.is_none() {
                        return Err(format!(
                            "capability {name} serves a transcription endpoint and must declare \
                             max_audio_bytes, the provider's inline request ceiling"
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
                CapabilityFormat::GoogleLive => {
                    // A Live session is stateful and typed: the setup message is built from
                    // the capability declaration, not per-request, so every field that shapes
                    // the session must be declared here and never overridden by a caller.
                    let Some(modalities) = &capability.response_modalities else {
                        return Err(format!(
                            "capability {name} serves a Live endpoint and must declare \
                             response_modalities, the output types this session produces"
                        ));
                    };
                    if modalities.is_empty() {
                        return Err(format!(
                            "capability {name} declares an empty response_modalities list; \
                             name at least one output modality"
                        ));
                    }
                    for modality in modalities {
                        if !matches!(modality.as_str(), "AUDIO" | "TEXT") {
                            return Err(format!(
                                "capability {name} declares response_modality {modality:?}, \
                                 but a Live session only serves AUDIO or TEXT"
                            ));
                        }
                    }
                    if let Some(budget) = capability.thinking_budget {
                        if budget > GEMINI_LIVE_MAX_THINKING_BUDGET {
                            return Err(format!(
                                "capability {name} declares thinking budget {budget}, above the \
                                 {GEMINI_LIVE_MAX_THINKING_BUDGET}-token maximum; a budget the \
                                 model rejects would fail every session"
                            ));
                        }
                    }
                    if let Some(level) = &capability.thinking_level {
                        if !matches!(level.as_str(), "MINIMAL" | "LOW" | "MEDIUM" | "HIGH") {
                            return Err(format!(
                                "capability {name} declares thinking level {level:?}, but a Live \
                                 session only serves MINIMAL, LOW, MEDIUM or HIGH"
                            ));
                        }
                        if capability.thinking_budget.is_some() {
                            return Err(format!(
                                "capability {name} declares both thinking_budget and thinking_level; \
                                 a Live session takes one thinking control, not two"
                            ));
                        }
                    }
                    if capability
                        .system_instruction
                        .as_deref()
                        .is_some_and(str::is_empty)
                    {
                        return Err(format!(
                            "capability {name} declares an empty system_instruction; state the \
                             exact instruction text, or omit the field"
                        ));
                    }
                }
            }

            // A task prefix is text the model reads, so it is configuration for one wire
            // only. On any other capability it would silently change nothing while
            // appearing to define the space.
            if capability.task_prefix.is_some()
                && !matches!(
                    client.format,
                    CapabilityFormat::GoogleEmbedContentV2 | CapabilityFormat::GoogleTranscribe
                )
            {
                return Err(format!(
                    "capability {name} declares task_prefix, which only google_embed_content_v2 \
                     and google_transcribe consume; this client's wire has no slot for it, so the \
                     value would be inert"
                ));
            }
            if capability.task_prefix.as_deref().is_some_and(str::is_empty) {
                return Err(format!(
                    "capability {name} declares an empty task_prefix; state the exact instruction \
                     text, or omit the field for the model's generic space"
                ));
            }

            // Speech and transcription configuration on a lane that cannot speak or listen
            // would read as a capability the deployment does not have, so each field is
            // refused on a wire with no slot for it rather than silently ignored.
            if client.format != CapabilityFormat::GoogleSpeech {
                for (field, present) in [
                    ("voice", capability.voice.is_some()),
                    ("audio_format", capability.audio_format.is_some()),
                    ("audio_sample_rate", capability.audio_sample_rate.is_some()),
                ] {
                    if present {
                        return Err(format!(
                            "capability {name} declares {field}, which only google_speech \
                             consumes; this client's wire does not synthesise audio, so the value \
                             would be inert"
                        ));
                    }
                }
            }
            if client.format != CapabilityFormat::GoogleTranscribe {
                for (field, present) in [
                    ("max_audio_seconds", capability.max_audio_seconds.is_some()),
                    ("max_audio_bytes", capability.max_audio_bytes.is_some()),
                ] {
                    if present {
                        return Err(format!(
                            "capability {name} declares {field}, which only google_transcribe \
                             consumes; this client's wire does not accept audio, so the value \
                             would be inert"
                        ));
                    }
                }
            }
            // An audio lane is not a vector space. A width or normalization here would
            // describe an embedding contract the lane never produces.
            if matches!(
                client.format,
                CapabilityFormat::GoogleSpeech | CapabilityFormat::GoogleTranscribe
            ) && (capability.dimensions.is_some()
                || capability.normalization.is_some()
                || capability.max_batch.is_some())
            {
                return Err(format!(
                    "capability {name} serves an audio endpoint and declares embedding fields \
                     (dimensions, normalization or max_batch), which describe a vector contract \
                     this lane does not produce"
                ));
            }

            // A task type is a Google wire field. Declaring it on a client that has no
            // such field would read as a configured behaviour the deployment does not
            // have, so it is refused instead of ignored.
            // The embedding-2 wire has no task_type: the task travels in the text. A
            // declaration here would be sent nowhere and would read as a configured
            // behaviour the deployment does not have.
            if client.task_type.is_some() && client.format == CapabilityFormat::GoogleEmbedContentV2
            {
                return Err(format!(
                    "capability client {client_name} declares task_type, which gemini-embedding-2 \
                     does not accept; declare the task as task_prefix on the capability instead"
                ));
            }
            if client.task_type.is_some() && client.format != CapabilityFormat::GoogleEmbedContent {
                return Err(format!(
                    "capability client {client_name} declares task_type, which only \
                     google_embed_content sends"
                ));
            }

            // A zero budget is a refusal of every call, which is never what an operator
            // meant to write. Stated once, at load, rather than discovered as a permanent
            // 429 on the first request.
            for (field, value) in [
                ("daily_request_limit", client.daily_request_limit),
                ("requests_per_minute", client.requests_per_minute),
                ("input_tokens_per_minute", client.input_tokens_per_minute),
            ] {
                if value == Some(0) {
                    return Err(format!(
                        "capability client {client_name} declares {field} = 0, which would refuse \
                         every call"
                    ));
                }
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
            quota_group: None,
            daily_request_limit: None,
            requests_per_minute: None,
            input_tokens_per_minute: None,
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
    /// Instruction text prepended to every input, for a prefix-defined space.
    pub task_prefix: Option<String>,
    /// Provider-published input-token limit of one embedding input.
    pub max_input_tokens: Option<u32>,
    /// Default voice a speech capability speaks in.
    pub voice: Option<String>,
    /// Output audio container a speech capability serves (`wav` or `pcm`).
    pub audio_format: Option<String>,
    /// Sample rate of the engine's audio output, in hertz.
    pub audio_sample_rate: Option<u32>,
    /// Longest audio one transcription request may carry, in seconds.
    pub max_audio_seconds: Option<u32>,
    /// Largest audio payload one transcription request may carry, in raw bytes.
    pub max_audio_bytes: Option<u64>,
    /// The budgets this capability's physical calls are charged to.
    ///
    /// `None` when the client declares none, which is the unchanged behaviour of a
    /// deployment that does not budget its capability calls.
    pub budget: Option<Arc<CapabilityBudget>>,
    /// The output modalities this Live session serves (`AUDIO`, `TEXT`, or both).
    ///
    /// Declared per capability so the setup message is built locally and the caller cannot
    /// request a modality this deployment does not serve.
    pub response_modalities: Option<Vec<String>>,
    /// System instruction that shapes every session, part of the capability's contract.
    pub system_instruction: Option<String>,
    /// Thinking budget for Extended Thinking variants, in tokens.
    pub thinking_budget: Option<u32>,
    /// Thinking level for variants that take a level instead of a budget (`MINIMAL`..`HIGH`).
    pub thinking_level: Option<String>,
}

/// The budgets one capability's physical calls draw on.
///
/// Both are keyed by the provider's own identity — quota group and physical model — not by
/// the caller-visible capability id, so two capabilities over one model share one counter.
/// The feature axis keeps an embedding call from spending the minute window a chat
/// generation on the same model would use: the provider counts them separately, and a
/// fan-out of N physical calls must not silently consume N chat slots.
#[derive(Debug)]
pub struct CapabilityBudget {
    /// Persistent per-provider-day request counter, when a daily limit is declared.
    pub allowance: Option<Arc<DailyRequestAllowance>>,
    /// Rolling per-minute request and input-token window, when it declares either.
    pub pacing: Option<Arc<PacedQuota>>,
}

/// Resolves configuration into the serving form, so dispatch never re-reads config.
///
/// Unbudgeted resolution: the caller declares no limits, so dispatch is not paced. Used by
/// tests and by any embedding of the runner that does not build a deployment.
pub fn resolve(deployment: &CapabilityDeployment) -> BTreeMap<String, ResolvedCapability> {
    resolve_with_budget(deployment, None, None)
        .expect("a deployment that declares no capability budget cannot fail to resolve")
}

/// Resolves configuration into the serving form and binds each capability's declared
/// budget.
///
/// `governor` is the deployment's one minute-window governor, shared with the completion
/// routes so an embedding call and a generation on the same project draw on the same
/// ledger. `quota_state_dir` is where a daily counter is persisted; a client that declares
/// `daily_request_limit` without it is refused rather than left uncounted, because a
/// budget that is never written cannot be enforced.
pub fn resolve_with_budget(
    deployment: &CapabilityDeployment,
    governor: Option<&Arc<QuotaGovernor>>,
    quota_state_dir: Option<&Path>,
) -> Result<BTreeMap<String, ResolvedCapability>, String> {
    let mut resolved = BTreeMap::new();
    for (name, capability) in &deployment.capabilities {
        // A held capability keeps its declaration but leaves dispatch entirely, so its id
        // cannot be named by a caller and cannot appear in discovery.
        if capability.held.is_some() {
            continue;
        }
        let Some(client) = deployment.clients.get(&capability.target) else {
            // An unresolvable capability is absent rather than empty: a surface that
            // reported an empty capability would answer for a space it cannot serve.
            continue;
        };
        let default_path = client.format.default_path(&client.model);
        let budget = capability_budget(name, client, governor, quota_state_dir)?;
        resolved.insert(
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
                task_prefix: capability.task_prefix.clone(),
                max_input_tokens: capability.max_input_tokens,
                voice: capability.voice.clone(),
                audio_format: capability.audio_format.clone(),
                audio_sample_rate: capability.audio_sample_rate,
                max_audio_seconds: capability.max_audio_seconds,
                max_audio_bytes: capability.max_audio_bytes,
                budget,
                response_modalities: capability.response_modalities.clone(),
                system_instruction: capability.system_instruction.clone(),
                thinking_budget: capability.thinking_budget,
                thinking_level: capability.thinking_level.clone(),
            },
        );
    }
    Ok(resolved)
}

/// The budget one capability draws on, from its client's declarations.
fn capability_budget(
    capability_name: &str,
    client: &CapabilityClientConfig,
    governor: Option<&Arc<QuotaGovernor>>,
    quota_state_dir: Option<&Path>,
) -> Result<Option<Arc<CapabilityBudget>>, String> {
    let group = client
        .quota_group
        .clone()
        .unwrap_or_else(|| capability_name.to_string());
    let allowance = match client.daily_request_limit {
        None => None,
        Some(limit) => {
            let state_dir = quota_state_dir.ok_or_else(|| {
                format!(
                    "capability {capability_name} declares daily_request_limit but the deployment \
                     configures no [quota] state_dir; a daily budget has to be counted to be \
                     enforced"
                )
            })?;
            Some(Arc::new(DailyRequestAllowance::new(
                group.clone(),
                switchyard_protocol::ModelId::new(client.model.clone()),
                limit,
                allowance_path(state_dir, &group, &client.model),
            )))
        }
    };
    let limits = WindowLimits {
        requests_per_minute: client.requests_per_minute,
        input_tokens_per_minute: client.input_tokens_per_minute,
    };
    let pacing = if limits.is_unlimited() {
        None
    } else {
        let governor = governor.ok_or_else(|| {
            format!(
                "capability {capability_name} declares a per-minute budget, which needs the \
                 deployment's quota governor"
            )
        })?;
        Some(Arc::new(PacedQuota::new(
            Arc::clone(governor),
            GovernorKey::new(
                group,
                switchyard_protocol::ModelId::new(client.model.clone()),
                "embed",
            ),
            limits,
            switchyard_llm_client::AdmissionMargin::NONE,
        )))
    };
    if allowance.is_none() && pacing.is_none() {
        return Ok(None);
    }
    Ok(Some(Arc::new(CapabilityBudget { allowance, pacing })))
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

#[cfg(test)]
mod gemini_embedding_2_tests {
    use super::*;

    const SOURCE: &str = r#"
[clients.google2]
format = "google_embed_content_v2"
base_url = "https://generativelanguage.googleapis.com"
model = "gemini-embedding-2"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"

[capabilities.gem2]
id = "switchyard-embedding-google2"
target = "google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = 768
normalization = "L2"
max_batch = 8
max_input_tokens = 8192
task_prefix = "task: search result | query: "
"#;

    fn validated(source: &str) -> CapabilityDeployment {
        let deployment: CapabilityDeployment = toml::from_str(source).expect("fixture parses");
        deployment.validate().expect("fixture is valid");
        deployment
    }

    /// Width is a request field on this model, so a declaration outside the model's range
    /// would be sent and refused on every call.
    #[test]
    fn a_width_outside_the_model_range_is_refused() {
        for width in ["4", "4096"] {
            let source = SOURCE.replace("dimensions = 768", &format!("dimensions = {width}"));
            let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
            assert!(
                deployment
                    .validate()
                    .expect_err("an out-of-range width must not load")
                    .contains("outside")
            );
        }
    }

    /// The normalization is checked against the vectors actually received, so it has to be
    /// declared; and the input limit cannot exceed the model's own.
    #[test]
    fn the_normalization_and_input_limit_are_required_and_bounded() {
        let no_normalization = SOURCE.replace("normalization = \"L2\"\n", "");
        let deployment: CapabilityDeployment = toml::from_str(&no_normalization).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("undeclared normalization must not load")
                .contains("normalization")
        );

        let above_model = SOURCE.replace("max_input_tokens = 8192", "max_input_tokens = 9000");
        let deployment: CapabilityDeployment = toml::from_str(&above_model).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("a limit above the model's own must not load")
                .contains("8192")
        );
    }

    /// This model has no `task_type`, so a task travels as a prefix. Declaring the wrong one
    /// of the two would describe a wire the deployment does not speak.
    #[test]
    fn the_task_travels_as_a_prefix_and_never_as_a_task_type() {
        let with_task_type = SOURCE.replace(
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"",
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"\ntask_type = \"RETRIEVAL_DOCUMENT\"",
        );
        let deployment: CapabilityDeployment = toml::from_str(&with_task_type).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("task_type must be refused on this wire")
                .contains("task_prefix")
        );
    }

    /// A prefix on a wire with no prefix slot would change nothing while appearing to define
    /// the space.
    #[test]
    fn a_prefix_is_refused_on_a_wire_that_has_no_prefix_slot() {
        let source = SOURCE.replace("google_embed_content_v2", "google_embed_content");
        let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("an inert prefix must not load")
                .contains("task_prefix")
        );
    }

    /// An empty prefix is a declaration with no content: either the generic space (omit it)
    /// or the exact instruction text.
    #[test]
    fn an_empty_prefix_is_refused() {
        let source = SOURCE.replace(
            "task_prefix = \"task: search result | query: \"",
            "task_prefix = \"\"",
        );
        let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("an empty prefix must not load")
                .contains("empty task_prefix")
        );
    }

    /// A zero budget refuses every call, which is never an intended configuration.
    #[test]
    fn a_zero_budget_is_refused() {
        let source = SOURCE.replace(
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"",
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"\ndaily_request_limit = 0",
        );
        let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("a zero budget must not load")
                .contains("refuse every call")
        );
    }

    /// Dispatch must carry the prefix, the width as declared, and the model's own path.
    #[test]
    fn dispatch_carries_the_prefix_and_names_the_model_in_the_path() {
        let resolved = resolve(&validated(SOURCE));
        let capability = &resolved["switchyard-embedding-google2"];
        assert_eq!(
            capability.task_prefix.as_deref(),
            Some("task: search result | query: ")
        );
        assert_eq!(capability.dimensions, Some(768));
        assert_eq!(capability.task_type, None);
        assert_eq!(
            capability.url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
        );
        assert!(capability.budget.is_none(), "no limit was declared");
    }

    /// The counter is the provider's unit: project plus physical model. Two capabilities over
    /// one model must therefore land on one file, not two.
    #[test]
    fn a_budget_is_keyed_by_quota_group_and_physical_model() {
        let source = format!(
            "{SOURCE}\n[capabilities.gem2b]\nid = \"switchyard-embedding-google2-b\"\ntarget = \
             \"google2\"\ncontract = \"google-embedding-space:gemini-embedding-2-v1-b\"\n\
             dimensions = 768\nnormalization = \"L2\"\nmax_input_tokens = 8192\n"
        );

        let limited = source.replace(
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"",
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"\nquota_group = \"google-free\"\n\
             daily_request_limit = 20\nrequests_per_minute = 6",
        );
        let deployment = validated(&limited);
        let governor = Arc::new(QuotaGovernor::new());
        let state_dir = std::env::temp_dir().join(format!(
            "switchyard-capability-budget-{}",
            std::process::id()
        ));
        let resolved = resolve_with_budget(&deployment, Some(&governor), Some(state_dir.as_path()))
            .expect("budgeted resolution");

        let a = resolved["switchyard-embedding-google2"]
            .budget
            .as_ref()
            .expect("first capability is budgeted");
        let b = resolved["switchyard-embedding-google2-b"]
            .budget
            .as_ref()
            .expect("second capability is budgeted");
        let allowance_a = a.allowance.as_ref().expect("daily limit declared");
        let allowance_b = b.allowance.as_ref().expect("daily limit declared");
        assert_eq!(allowance_a.group(), "google-free");
        assert_eq!(allowance_a.model().as_str(), "gemini-embedding-2");
        assert_eq!(
            allowance_a.limit(),
            allowance_b.limit(),
            "one model, one budget"
        );
        assert_eq!(
            allowance_a.remaining(),
            allowance_b.remaining(),
            "both read the same counter"
        );
        let pacing = a.pacing.as_ref().expect("minute window declared");
        assert_eq!(pacing.key().feature(), "embed");
        assert_eq!(pacing.key().group(), "google-free");
    }

    /// A daily budget with nowhere to persist it would be uncountable, and uncounted caps are
    /// not caps.
    #[test]
    fn a_daily_limit_without_a_state_dir_is_refused() {
        let source = SOURCE.replace(
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"",
            "api_key_env = \"SWITCHYARD_GEMINI_API_KEY\"\ndaily_request_limit = 20",
        );
        let deployment = validated(&source);
        let governor = Arc::new(QuotaGovernor::new());
        assert!(
            resolve_with_budget(&deployment, Some(&governor), None)
                .expect_err("uncountable budget must be refused")
                .contains("state_dir")
        );
    }
}

#[cfg(test)]
mod held_capability_tests {
    use super::*;

    const SOURCE: &str = r#"
[clients.google2]
format = "google_embed_content_v2"
base_url = "https://generativelanguage.googleapis.com"
model = "gemini-embedding-2"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"

[capabilities.gem2]
id = "gemini-embedding-2"
target = "google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = 3072
normalization = "L2"
max_batch = 8
max_input_tokens = 8192
held = "response format not yet qualified against the project key"
"#;

    /// Held keeps the declaration and its validation, and takes the capability out of the
    /// serving form entirely: a surface that resolved it would answer for a space nobody
    /// qualified.
    #[test]
    fn a_held_capability_is_validated_but_never_resolved() {
        let deployment: CapabilityDeployment = toml::from_str(SOURCE).expect("fixture parses");
        deployment
            .validate()
            .expect("a held capability is still checked");
        assert!(resolve(&deployment).is_empty());
    }

    /// Omitting the field is the whole difference between held and serving, so an empty
    /// value - which reads as held while saying nothing - is refused.
    #[test]
    fn an_empty_held_reason_is_refused() {
        let source = SOURCE.replace(
            "held = \"response format not yet qualified against the project key\"",
            "held = \"\"",
        );
        let deployment: CapabilityDeployment = toml::from_str(&source).expect("parses");
        assert!(
            deployment
                .validate()
                .expect_err("an empty hold reason must not load")
                .contains("empty held reason")
        );
    }
}

#[cfg(test)]
mod audio_tests {
    use super::*;

    const SPEECH_CLIENT: &str = r#"
[clients.tts]
format = "google_speech"
base_url = "https://generativelanguage.test"
model = "gemini-2.5-flash-preview-tts"
"#;

    const TRANSCRIBE_CLIENT: &str = r#"
[clients.asr]
format = "google_transcribe"
base_url = "https://generativelanguage.test"
model = "gemini-3.5-transcribe"
"#;

    const COMPLETE_SPEECH: &str = r#"
[capabilities.speak]
id = "speak"
target = "tts"
voice = "Kore"
audio_format = "wav"
audio_sample_rate = 24000
max_input_tokens = 8192
"#;

    const COMPLETE_TRANSCRIBE: &str = r#"
[capabilities.listen]
id = "listen"
target = "asr"
task_prefix = "Transcribe this recording verbatim."
max_input_tokens = 98304
max_audio_seconds = 300
max_audio_bytes = 15728640
"#;

    fn deployment(source: &str) -> CapabilityDeployment {
        toml::from_str(source).expect("capability fixture parses")
    }

    fn refusal(source: &str) -> String {
        deployment(source)
            .validate()
            .expect_err("fixture must be refused")
    }

    #[test]
    fn a_complete_speech_capability_validates() {
        let source = format!("{SPEECH_CLIENT}{COMPLETE_SPEECH}");
        let parsed = deployment(&source);
        parsed.validate().expect("complete speech lane is valid");
        let resolved = resolve(&parsed);
        let capability = resolved.get("speak").expect("resolved");
        assert_eq!(capability.voice.as_deref(), Some("Kore"));
        assert_eq!(capability.audio_format.as_deref(), Some("wav"));
        assert_eq!(capability.audio_sample_rate, Some(24_000));
        assert!(capability.format.serves_speech());
        assert_eq!(
            capability.url,
            "https://generativelanguage.test/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
        );
    }

    #[test]
    fn a_speech_lane_must_declare_its_voice_container_rate_and_input_limit() {
        for (drop_line, expected) in [
            ("voice = \"Kore\"\n", "must declare a default voice"),
            ("audio_format = \"wav\"\n", "must declare audio_format"),
            (
                "audio_sample_rate = 24000\n",
                "must declare audio_sample_rate",
            ),
            ("max_input_tokens = 8192\n", "must declare max_input_tokens"),
        ] {
            let capability = COMPLETE_SPEECH.replace(drop_line, "");
            let refused = refusal(&format!("{SPEECH_CLIENT}{capability}"));
            assert!(refused.contains(expected), "{refused}");
        }
    }

    /// The engine returns PCM. A container this surface would have to encode is refused at
    /// load rather than mislabelled at response time.
    #[test]
    fn a_speech_lane_refuses_a_container_it_cannot_produce() {
        let capability =
            COMPLETE_SPEECH.replace("audio_format = \"wav\"", "audio_format = \"mp3\"");
        let refused = refusal(&format!("{SPEECH_CLIENT}{capability}"));
        assert!(refused.contains("produces wav and pcm only"), "{refused}");
    }

    /// A declared rate of zero is present but meaningless: the container it would produce has
    /// no timing, so it is refused by name rather than written.
    #[test]
    fn a_speech_lane_refuses_a_zero_sample_rate() {
        let capability =
            COMPLETE_SPEECH.replace("audio_sample_rate = 24000", "audio_sample_rate = 0");
        let refused = refusal(&format!("{SPEECH_CLIENT}{capability}"));
        assert!(refused.contains("no timing"), "{refused}");
    }

    #[test]
    fn a_complete_transcription_capability_validates() {
        let source = format!("{TRANSCRIBE_CLIENT}{COMPLETE_TRANSCRIBE}");
        let parsed = deployment(&source);
        parsed
            .validate()
            .expect("complete transcription lane is valid");
        let resolved = resolve(&parsed);
        let capability = resolved.get("listen").expect("resolved");
        assert_eq!(capability.max_audio_seconds, Some(300));
        assert_eq!(capability.max_audio_bytes, Some(15_728_640));
        assert!(capability.format.serves_transcription());
        assert_eq!(
            capability.url,
            "https://generativelanguage.test/v1beta/models/gemini-3.5-transcribe:generateContent"
        );
    }

    #[test]
    fn a_transcription_lane_must_declare_its_instruction_and_bounds() {
        for (drop_line, expected) in [
            (
                "task_prefix = \"Transcribe this recording verbatim.\"\n",
                "must declare task_prefix",
            ),
            (
                "max_audio_seconds = 300\n",
                "must declare max_audio_seconds",
            ),
            (
                "max_audio_bytes = 15728640\n",
                "must declare max_audio_bytes",
            ),
        ] {
            let capability = COMPLETE_TRANSCRIBE.replace(drop_line, "");
            let refused = refusal(&format!("{TRANSCRIBE_CLIENT}{capability}"));
            assert!(refused.contains(expected), "{refused}");
        }
    }

    /// Each new field is refused on a wire that has no slot for it, so a deployment cannot
    /// read as a capability it does not have.
    #[test]
    fn audio_fields_are_refused_on_a_lane_that_cannot_use_them() {
        let embedding_with_voice = "[clients.embed]\nformat = \"openai_embeddings\"\nbase_url = \"https://engine.test\"\nmodel = \"e\"\n\
             \n[capabilities.space]\nid = \"space\"\ntarget = \"embed\"\ncontract = \"space:v1\"\ndimensions = 1024\nvoice = \"Kore\"\n";
        assert!(
            refusal(embedding_with_voice).contains("only google_speech consumes"),
            "{}",
            refusal(embedding_with_voice)
        );

        let embedding_with_audio_bound = "[clients.embed]\nformat = \"openai_embeddings\"\nbase_url = \"https://engine.test\"\nmodel = \"e\"\n\
             \n[capabilities.space]\nid = \"space\"\ntarget = \"embed\"\ncontract = \"space:v1\"\ndimensions = 1024\nmax_audio_seconds = 60\n";
        assert!(
            refusal(embedding_with_audio_bound).contains("only google_transcribe consumes"),
            "{}",
            refusal(embedding_with_audio_bound)
        );

        let speech_with_dimensions = COMPLETE_SPEECH.replace(
            "voice = \"Kore\"\n",
            "voice = \"Kore\"\ndimensions = 1024\n",
        );
        assert!(
            refusal(&format!("{SPEECH_CLIENT}{speech_with_dimensions}"))
                .contains("describe a vector contract this lane does not produce"),
            "{}",
            refusal(&format!("{SPEECH_CLIENT}{speech_with_dimensions}"))
        );
    }

    #[test]
    fn a_voice_name_is_a_token_not_a_sentence() {
        assert!(is_voice_name("Kore"));
        assert!(is_voice_name("en-US-Chirp3-HD-Achernar"));
        assert!(!is_voice_name(""));
        assert!(!is_voice_name("Kore please"));
        assert!(!is_voice_name("Kore;drop"));
    }

    #[test]
    fn audio_lanes_do_not_serve_embeddings_and_vice_versa() {
        assert!(!CapabilityFormat::GoogleSpeech.serves_embeddings());
        assert!(!CapabilityFormat::GoogleTranscribe.serves_embeddings());
        assert!(!CapabilityFormat::GoogleSpeech.serves_transcription());
        assert!(!CapabilityFormat::GoogleTranscribe.serves_speech());
        assert!(CapabilityFormat::GoogleSpeech.serves_speech());
        assert!(CapabilityFormat::GoogleTranscribe.serves_transcription());
        assert_eq!(
            CapabilityFormat::GoogleSpeech.auth(),
            CapabilityAuth::GoogleApiKey
        );
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    const LIVE_CLIENT: &str = r#"
[clients.live]
format = "google_live"
base_url = "https://generativelanguage.googleapis.com"
model = "gemini-3.8-live"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"
"#;

    const COMPLETE_LIVE: &str = r#"
[capabilities.live]
id = "live"
target = "live"
response_modalities = ["AUDIO", "TEXT"]
system_instruction = "You are a helpful assistant."
thinking_budget = 1024
"#;

    fn deployment(source: &str) -> CapabilityDeployment {
        toml::from_str(source).expect("capability fixture parses")
    }

    #[test]
    fn a_complete_live_capability_validates() {
        let source = format!("{LIVE_CLIENT}{COMPLETE_LIVE}");
        let parsed = deployment(&source);
        parsed.validate().expect("complete Live lane is valid");
        let resolved = resolve(&parsed);
        let capability = resolved.get("live").expect("resolved");
        assert_eq!(
            capability.response_modalities.as_deref(),
            Some(&["AUDIO".to_string(), "TEXT".to_string()][..])
        );
        assert_eq!(
            capability.system_instruction.as_deref(),
            Some("You are a helpful assistant.")
        );
        assert_eq!(capability.thinking_budget, Some(1024));
        assert!(capability.format.serves_embeddings() == false);
        assert_eq!(
            capability.url,
            "https://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        );
    }

    #[test]
    fn a_live_capability_must_declare_response_modalities() {
        let without = COMPLETE_LIVE.replace("response_modalities = [\"AUDIO\", \"TEXT\"]\n", "");
        let refused = refusal(&format!("{LIVE_CLIENT}{without}"));
        assert!(
            refused.contains("must declare response_modalities"),
            "{refused}"
        );
    }

    #[test]
    fn an_empty_modality_list_is_refused() {
        let without = COMPLETE_LIVE.replace(
            "response_modalities = [\"AUDIO\", \"TEXT\"]",
            "response_modalities = []",
        );
        let refused = refusal(&format!("{LIVE_CLIENT}{without}"));
        assert!(
            refused.contains("empty response_modalities list"),
            "{refused}"
        );
    }

    #[test]
    fn an_unknown_modality_is_refused() {
        let without = COMPLETE_LIVE.replace(
            "response_modalities = [\"AUDIO\", \"TEXT\"]",
            "response_modalities = [\"VIDEO\"]",
        );
        let refused = refusal(&format!("{LIVE_CLIENT}{without}"));
        assert!(refused.contains("only serves AUDIO or TEXT"), "{refused}");
    }

    #[test]
    fn a_thinking_budget_above_the_maximum_is_refused() {
        let too_big = COMPLETE_LIVE.replace("thinking_budget = 1024", "thinking_budget = 65536");
        let refused = refusal(&format!("{LIVE_CLIENT}{too_big}"));
        assert!(refused.contains("above the"), "{refused}");
    }

    #[test]
    fn a_thinking_level_is_accepted_and_resolved() {
        let level =
            COMPLETE_LIVE.replace("thinking_budget = 1024\n", "thinking_level = \"HIGH\"\n");
        let source = format!("{LIVE_CLIENT}{level}");
        deployment(&source).validate().expect("level lane is valid");
        let resolved = resolve(&deployment(&source));
        let capability = resolved.get("live").expect("resolved");
        assert_eq!(capability.thinking_level.as_deref(), Some("HIGH"));
        assert_eq!(capability.thinking_budget, None);
    }

    #[test]
    fn an_unknown_thinking_level_is_refused() {
        // Gemini publishes MINIMAL/LOW/MEDIUM/HIGH; MINIMAL is itself refused by the
        // 3.8 extended-thinking model, so the manifest must not silently accept anything
        // outside the serviceable set.
        let level =
            COMPLETE_LIVE.replace("thinking_budget = 1024\n", "thinking_level = \"TURBO\"\n");
        let refused = refusal(&format!("{LIVE_CLIENT}{level}"));
        assert!(
            refused.contains("only serves MINIMAL, LOW, MEDIUM or HIGH"),
            "{refused}"
        );
    }

    #[test]
    fn both_thinking_controls_are_refused() {
        let both = COMPLETE_LIVE.replace(
            "thinking_budget = 1024\n",
            "thinking_budget = 1024\nthinking_level = \"HIGH\"\n",
        );
        let refused = refusal(&format!("{LIVE_CLIENT}{both}"));
        assert!(
            refused.contains("one thinking control, not two"),
            "{refused}"
        );
    }

    #[test]
    fn an_empty_system_instruction_is_refused() {
        let without = COMPLETE_LIVE.replace(
            "system_instruction = \"You are a helpful assistant.\"\n",
            "system_instruction = \"\"\n",
        );
        let refused = refusal(&format!("{LIVE_CLIENT}{without}"));
        assert!(refused.contains("empty system_instruction"), "{refused}");
    }

    fn refusal(source: &str) -> String {
        deployment(source)
            .validate()
            .expect_err("fixture must be refused")
    }
}
