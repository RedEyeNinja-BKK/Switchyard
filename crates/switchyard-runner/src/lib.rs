// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.

mod algorithm;
pub mod capability;
mod config;
mod failure;
mod route;
mod runner;

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, ClassifierMode,
    ClassifierPolicyConfig, FleetBuildContext, FleetCandidateConfig,
    FleetCandidateContextPolicyConfig, FleetWorkShapeSourceName, InputTokenSourceConfig,
    LlmClassifierRouteConfig, StageClassifierConfig, SubagentRouteConfig,
};
pub use capability::{
    CapabilityClientConfig, CapabilityClientFormat, CapabilityKind, CapabilityRouteConfig,
};
pub use config::{
    ComfyFactConfig, FleetReadinessConfig, HtpcFactConfig, HttpBaseUrl, ResourceFactConfig,
};
pub use failure::{RouteErrorKind, RouteErrorPhase, RouteErrorSummary, stream_error_summary};
pub use route::{
    AuxiliaryTarget, CallerAuthKind, ModelCapabilities, Route, RunOutput, RunnerError,
};
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
