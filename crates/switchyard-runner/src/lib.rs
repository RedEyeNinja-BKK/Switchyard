// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.

mod algorithm;
pub mod capability;
mod config;
pub mod facts_config;
mod failure;
mod provider_key_redactor;
mod route;
mod runner;

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, CategoryModelConfig, ClassifierMode,
    ClassifierPolicyConfig, FleetCandidateConfig, LlmClassifierRouteConfig, StageClassifierConfig,
    SubagentRouteConfig, build_fleet_router,
};
pub use capability::{
    CapabilityClientConfig, CapabilityClientFormat, CapabilityKind, CapabilityRouteConfig,
};
pub use facts_config::{
    ComfyFactConfig, ComfyGovernedConfig, FleetReadinessConfig, HtpcFactConfig, ResourceFactConfig,
};
pub use failure::{RouteErrorKind, RouteErrorPhase, RouteErrorSummary, stream_error_summary};
// Re-exported because `Route::new` takes it, so a host wiring routes does not need a libsy dep.
pub use libsy::RuntimeModels;
pub use provider_key_redactor::ProviderKeyRedactor;
pub use route::{
    AuxiliaryTarget, CallerAuthKind, ModelCapabilities, Route, RunOutput, RunnerError,
};
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
