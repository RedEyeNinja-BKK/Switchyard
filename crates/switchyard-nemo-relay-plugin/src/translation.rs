// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::LlmRequest as RelayRequest;
use serde_json::Value as Json;
use switchyard_protocol::{AggLlmResponse, LlmRequest, ProviderExtensions, WireFormat};
use switchyard_translation::{
    DeterministicIdPolicy, DiagnosticSeverity, LossyConversionPolicy, PreservationPolicy,
    TargetCapabilities, TranslationDiagnostic, TranslationEngine, TranslationPolicy,
    UnknownFieldPolicy,
};

pub(crate) fn decode_request(
    engine: &TranslationEngine,
    protocol: WireFormat,
    request: &RelayRequest,
) -> Result<LlmRequest, String> {
    let output = engine
        .decode_request(protocol, &request.content, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.request)
}

pub(crate) fn encode_response(
    engine: &TranslationEngine,
    protocol: WireFormat,
    response: &AggLlmResponse,
    request_extensions: &ProviderExtensions,
) -> Result<Json, String> {
    let output = engine
        .encode_response_with_extensions(protocol, response, request_extensions, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.body)
}

fn policy() -> TranslationPolicy {
    TranslationPolicy {
        unknown_field_policy: UnknownFieldPolicy::Preserve,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        deterministic_ids: DeterministicIdPolicy::GenerateStable {
            prefix: "relay".into(),
        },
        preservation: PreservationPolicy::InMemory,
        target_capabilities: TargetCapabilities::default(),
        // The relay carries no provider-specific response shaping: a marker belongs to the
        // upstream that emits it, and this plugin is not that upstream.
        hosted_reasoning: None,
        // The relay encodes responses only and never decides a replay, so the safe default is
        // the one that sends nothing it cannot prove.
        reasoning_details_replay: switchyard_translation::ReasoningDetailsReplay::ProvenanceBound,
        reasoning_admission: None,
    }
}

fn safe(diagnostics: &[TranslationDiagnostic]) -> Result<(), String> {
    let unsafe_diagnostics = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity != DiagnosticSeverity::Info)
        .collect::<Vec<_>>();
    if unsafe_diagnostics.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Switchyard translation was not lossless: {unsafe_diagnostics:?}"
        ))
    }
}

fn error(error: switchyard_translation::TranslationError) -> String {
    format!("Switchyard translation failed: {error}")
}
