// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy knobs that control translation strictness and target capabilities.

use serde::{Deserialize, Serialize};

use crate::format::FormatId;

/// Controls how unknown provider fields are handled.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum UnknownFieldPolicy {
    Preserve,
    DropWithWarning,
    Reject,
}

/// Controls whether known lossy conversions are allowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LossyConversionPolicy {
    AllowWithDiagnostics,
    Reject,
}

/// Controls how missing provider IDs are generated.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DeterministicIdPolicy {
    Preserve,
    GenerateStable { prefix: String },
}

/// Controls whether exact source payloads are preserved for round trips.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreservationPolicy {
    /// Keep exact source bodies in the in-memory IR only.
    InMemory,
    /// Also embed a Switchyard metadata envelope in translated wire bodies.
    ///
    /// This enables multi-hop round trips through wire formats, but should be
    /// used deliberately because provider APIs may reject or limit metadata.
    Embed,
    /// Do not store preservation metadata.
    Disabled,
}

/// Capability constraints for the target model or provider profile.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetCapabilities {
    pub supports_tools: Option<bool>,
    pub supports_images: Option<bool>,
    pub supports_audio: Option<bool>,
    pub supports_video: Option<bool>,
    pub supports_files: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_json_schema_response_format: Option<bool>,
    pub supports_parallel_tool_calls: Option<bool>,
    pub supports_code_execution: Option<bool>,
    pub supports_safety_settings: Option<bool>,
    pub openai_compatible: Option<bool>,
    /// The upstream requires `input` to be a JSON array and rejects the bare-string form.
    ///
    /// The Codex backend does: it answers a lone text prompt encoded as a string with
    /// `Input must be a list`. When set, a single user message is encoded in list form.
    /// Absent or false the compact string form is kept, which is what every other Responses
    /// upstream expects.
    pub requires_list_input: Option<bool>,
}

/// Provider-level profile that can be used to configure target capabilities.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderProfile {
    pub format: Option<FormatId>,
    pub provider: Option<String>,
    pub capabilities: TargetCapabilities,
}

/// Controls whether provider-private reasoning state is replayed to an upstream.
///
/// `reasoning_details` and a `signature` are opaque state belonging to the provider that issued
/// them. OpenRouter hands them back and expects them returned verbatim; a native provider
/// (DeepSeek, Google, a local llama.cpp) either does not understand them or would be receiving
/// another provider's private state. This is a per-target wire fact, not a global preference, so
/// the runner derives it from the target's `reasoning_dialect`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReasoningDetailsReplay {
    /// Replay structured details and signatures exactly as received.
    #[default]
    Verbatim,
    /// Drop structured details and signatures; carry only the reasoning text.
    PlaintextOnly,
}

/// End-to-end policy applied by request, response, and stream translation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranslationPolicy {
    pub unknown_field_policy: UnknownFieldPolicy,
    pub lossy_conversion_policy: LossyConversionPolicy,
    pub deterministic_ids: DeterministicIdPolicy,
    pub preservation: PreservationPolicy,
    pub target_capabilities: TargetCapabilities,
    /// The upstream embeds its reasoning in the message text behind this marker.
    ///
    /// Set per target, because it is a wire property of one provider family rather than a
    /// global preference: applying it everywhere would mangle a model that legitimately
    /// writes those tokens as text. `None` leaves response text untouched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted_reasoning: Option<crate::codecs::openai_chat::HostedReasoningMarker>,
    /// Whether opaque provider reasoning state may be replayed to this target.
    ///
    /// Defaults to `Verbatim`, the pre-existing behaviour, so a policy built without target
    /// knowledge never silently changes what is sent. The runner narrows it per target.
    #[serde(default)]
    pub reasoning_details_replay: ReasoningDetailsReplay,
}

impl Default for TranslationPolicy {
    fn default() -> Self {
        Self {
            unknown_field_policy: UnknownFieldPolicy::Preserve,
            lossy_conversion_policy: LossyConversionPolicy::AllowWithDiagnostics,
            deterministic_ids: DeterministicIdPolicy::GenerateStable {
                prefix: "sw".to_string(),
            },
            preservation: PreservationPolicy::InMemory,
            target_capabilities: TargetCapabilities::default(),
            hosted_reasoning: None,
            reasoning_details_replay: ReasoningDetailsReplay::default(),
        }
    }
}
