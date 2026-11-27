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
    /// The signature value this wire accepts for a replayed call it did not issue.
    ///
    /// Presence is the declaration that the wire validates a provider-issued signature on
    /// the first `functionCall` part of every step and cannot be sent a foreign call bare.
    /// Google documents the case — history transferred from a model that carries no
    /// signatures — and accepts the literal `skip_thought_signature_validator` in place of
    /// the real value, warning that it costs context fidelity. When this is set, a call that
    /// arrives with [`switchyard_protocol::ToolCall::provider_state`] echoes it verbatim and
    /// a call that arrives without one is replayed with this placeholder; when it is absent,
    /// no continuation state is emitted at all and the wire must not be sent replayed lineage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_signature_placeholder: Option<String>,
}

/// Provider-level profile that can be used to configure target capabilities.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderProfile {
    pub format: Option<FormatId>,
    pub provider: Option<String>,
    pub capabilities: TargetCapabilities,
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
        }
    }
}
