// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One configured algorithm and the clients that serve its targets.

use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, LibsyError, RoutingOutcome, RuntimeModels};
use serde_json::Value;
use switchyard_llm_client::{
    AuxiliaryOperation, ClientRouter, RunObservation, RunObserver, TranslatingLlmClient,
};
use switchyard_protocol::{LlmClientError, ModelId, Request, Response, WireFormat};
use thiserror::Error;

use crate::DecisionTarget;
use crate::config::RouteReasoningPolicy;

/// Capabilities declared for one route.
///
/// `GET /v1/models` includes `context_window`, `tool_calling`, and `vision` in each
/// standard `data` entry, using `null` for unset values. It omits `reasoning`.
/// The server rejects tool inputs, reasoning controls, or images when the
/// corresponding declaration is explicitly `false` for the selected route.
#[derive(Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub context_window: Option<u32>,
    pub tool_calling: Option<bool>,
    /// Whether the routed model accepts reasoning controls, as declared in config.
    pub reasoning: Option<bool>,
    /// Whether the routed model accepts image input, as declared in config.
    pub vision: Option<bool>,
}

/// Caller credential family required by a forwarded-auth route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerAuthKind {
    Anthropic,
    OpenAi,
}

impl CallerAuthKind {
    /// Stable provider name used by the server compatibility API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }

    const fn accepts(self, wire_format: WireFormat) -> bool {
        matches!(
            (self, wire_format),
            (Self::Anthropic, WireFormat::AnthropicMessages)
                | (
                    Self::OpenAi,
                    WireFormat::OpenAiChat | WireFormat::OpenAiResponses
                )
        )
    }
}

/// Exact upstream model and client used for an auxiliary provider operation.
#[derive(Clone)]
pub struct AuxiliaryTarget {
    pub model: ModelId,
    pub client: Arc<TranslatingLlmClient>,
}

/// Error returned while loading or executing configured routes.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("{message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    #[error("unknown route model {0:?}")]
    UnknownRouteModel(String),
    /// Two children of one composite claim the same provider state for a request.
    ///
    /// Picking by child order would make the outcome depend on configuration order, so the
    /// conflict is reported and no provider is contacted.
    #[error("conflicting provider state: {0}")]
    StateConflict(String),
    /// A request is valid, but no eligible route can satisfy a capability it requires.
    ///
    /// This is the caller's request, not a deployment fault, so it is kept distinct from
    /// [`Self::Configuration`] and reported as a 4xx rather than a server failure.
    #[error("no eligible route for model {model:?}: requires {capability}")]
    UnsupportedCapability {
        model: String,
        capability: &'static str,
    },
    #[error("caller format is incompatible with {} credentials", .0.as_str())]
    IncompatibleCallerFormat(CallerAuthKind),
    #[error("route has no compatible target for the auxiliary operation")]
    AuxiliaryUnsupported,
    #[error(transparent)]
    Algorithm(#[from] LibsyError),
    #[error(transparent)]
    Client(#[from] LlmClientError),
}

impl RunnerError {
    pub(crate) fn unsupported_capability(
        model: impl Into<String>,
        capability: &'static str,
    ) -> Self {
        Self::UnsupportedCapability {
            model: model.into(),
            capability,
        }
    }

    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn configuration_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

/// A configured algorithm and the per-target clients its calls resolve through.
pub struct Route {
    algorithm: Arc<dyn Algorithm>,
    // Resolves each offloaded call to the client configured for the target the algorithm
    // selected. A route is a synthetic model with no upstream of its own, so this is a
    // per-target lookup, never one client serving the whole route.
    clients: ClientRouter,
    /// Reasoning policy imposed by the route, overriding the target's factual policy.
    route_reasoning_policy: crate::config::RouteReasoningPolicy,
    /// Context-pressure threshold for preferring a wider-context leg.
    escalation_max_input_tokens: Option<u64>,
    /// Target promoted ahead of the declared order under context pressure, resolved
    /// to its model id at load.
    escalation_target: Option<ModelId>,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
    anthropic_auxiliary_target: Option<AuxiliaryTarget>,
    responses_auxiliary_target: Option<AuxiliaryTarget>,
    decision_targets: Vec<DecisionTarget>,
    models: Arc<RuntimeModels>,
}

/// The selected model and untouched response produced by a route execution.
pub struct RunOutput {
    pub selected_model: ModelId,
    pub response: Response,
}

impl Route {
    /// Creates a fully configured execution route.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        decision_targets: Vec<DecisionTarget>,
        models: RuntimeModels,
        route_reasoning_policy: crate::config::RouteReasoningPolicy,
            escalation_max_input_tokens: Option<u64>,
        escalation_target: Option<ModelId>,
    ) -> Self {
        Self {
            algorithm,
            clients,
            route_reasoning_policy,
            escalation_max_input_tokens,
            escalation_target,
            caller_auth,
            capabilities,
            anthropic_auxiliary_target,
            responses_auxiliary_target,
            decision_targets,
            models: Arc::new(models),
        }
    }

    /// Returns the configured libsy algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }

    /// The reasoning policy this route imposes, overriding the target's factual policy.
    pub fn route_reasoning_policy(&self) -> crate::config::RouteReasoningPolicy {
        self.route_reasoning_policy
    }

    /// Estimated input tokens above which this route should prefer a wider-context leg.
    pub fn escalation_max_input_tokens(&self) -> Option<u64> {
        self.escalation_max_input_tokens
    }

    /// Whether this request is at or below the configured context-pressure threshold.
    ///
    /// The comparator is strictly greater-than, reproducing the historical
    /// `escalation_max_input_tokens` behaviour rather than approximating it. A request
    /// exactly at the threshold stays on the primary leg.
    pub fn under_context_pressure(&self, estimated_input_tokens: u64) -> bool {
        self.escalation_max_input_tokens
            .is_some_and(|threshold| estimated_input_tokens > threshold)
    }

    /// Returns model-list capability metadata.
    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    /// Returns the forwarded caller credential family.
    pub fn caller_auth(&self) -> Option<CallerAuthKind> {
        self.caller_auth
    }

    /// Resolves a selected model to this route's non-secret target metadata.
    pub(crate) fn decision_target(&self, model: &ModelId) -> Option<DecisionTarget> {
        self.decision_targets
            .iter()
            .find(|target| target.model == *model)
            .cloned()
    }

    /// Returns the models grouped for one algorithm execution.
    pub fn models(&self) -> &RuntimeModels {
        &self.models
    }

    /// Rejects a caller format incompatible with forwarded credentials.
    pub fn check_caller_format(&self, input_format: WireFormat) -> Result<(), RunnerError> {
        if let Some(kind) = self.caller_auth
            && !kind.accepts(input_format)
        {
            return Err(RunnerError::IncompatibleCallerFormat(kind));
        }
        Ok(())
    }

    /// Executes the configured route without consuming or proxying streamed responses.
    /// The model this route's client router holds provider state for, when a request
    /// continues a conversation this route previously served.
    pub fn continuation_owner(
        &self,
        request: &switchyard_protocol::Request,
    ) -> Option<switchyard_protocol::ModelId> {
        self.clients.continuation_owner(request)
    }

    pub async fn execute(
        &self,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<RunOutput, RunnerError> {
        // Ordered candidates are filtered by what each target can actually serve, before any
        // provider is contacted. Declared order is preserved among the survivors, and the
        // native execution, retry, cooldown and fallback machinery still decides among them,
        // so this adds no second fallback engine and issues no request to a target that
        // cannot answer it.
        let facts = self.clients.target_capabilities().clone();
        let raw = request.raw_request.clone().unwrap_or(serde_json::Value::Null);
        let estimated_input_tokens = crate::request_fit::estimate_input_tokens(&raw);
        let eligibility = |model: &switchyard_protocol::ModelId| {
            match facts.get(model) {
                Some(facts) => {
                    // An explicit `false` is a refusal; `None` would mean "undeclared",
                    // which the predicate treats as permitted, so the flag must be carried
                    // as a concrete boolean.
                    let capabilities = crate::ModelCapabilities {
                        vision: Some(!facts.no_vision),
                        tool_calling: Some(!facts.no_tool_calling),
                        ..crate::ModelCapabilities::default()
                    };
                    crate::request_fit::unsupported_capability(
                        capabilities,
                        &request.llm_request,
                        &raw,
                    )
                    .is_none()
                }
                None => true,
            }
        };
        let models = self.models.filtered(&eligibility);
        // Under context pressure the declared escalation leg is offered first, before any
        // provider is contacted. This reorders one list; it does not add a second
        // escalation engine, and the native retry, cooldown and fallback machinery still
        // decides among the survivors.
        //
        // The promotion is only real if the leg survived capability filtering *and* was
        // already in the list, so that is what decides whether to reorder and what to
        // report. Reporting on intent alone would over-count: an ineligible or unlisted leg
        // leaves the order untouched while the counter still claimed a wider leg was
        // preferred.
        let promotion = self.escalation_target.as_ref().filter(|escalation_target| {
            self.under_context_pressure(estimated_input_tokens)
                && models.contains(escalation_target)
        });
        let models = match promotion {
            Some(escalation_target) => {
                if let Some(observer) = observer.as_ref() {
                    observer(RunObservation::ContextPressurePromotion {
                        promoted: escalation_target.clone(),
                        estimated_input_tokens,
                    });
                }
                models.promoting(std::slice::from_ref(escalation_target))
            }
            None => models,
        };
        let request = match self.route_reasoning_policy {
            RouteReasoningPolicy::Inherit => request,
            RouteReasoningPolicy::Medium => request.with_route_reasoning_policy(
                switchyard_protocol::ReasoningPolicy::Enabled,
            ),
            // `High` needs the graded level as well as the on/off intent: on a dialect that
            // only grades effort the policy alone has no wire form, so the request also
            // carries the effort for the client to write.
            RouteReasoningPolicy::High => request
                .with_route_reasoning_policy(switchyard_protocol::ReasoningPolicy::Enabled)
                .with_route_reasoning_effort(
                    crate::config::route_policy_effort(RouteReasoningPolicy::High)
                        .unwrap_or_default(),
                ),
            RouteReasoningPolicy::None => {
                request.with_route_reasoning_policy(switchyard_protocol::ReasoningPolicy::Disabled)
            }
        };
        if models.is_empty() {
            // Nothing in this route can serve the request, which is the caller's request
            // rather than a deployment fault. The skipped targets get no provider call.
            // Name what the request needs, not what the route happens to lack: the route's
            // own declaration may be silent while a specific target was the blocker.
            let capability = self
                .clients
                .target_capabilities()
                .iter()
                .find_map(|(_model, facts)| {
                    let capabilities = crate::ModelCapabilities {
                        vision: Some(!facts.no_vision),
                        tool_calling: Some(!facts.no_tool_calling),
                        ..crate::ModelCapabilities::default()
                    };
                    crate::request_fit::unsupported_capability(
                        capabilities,
                        &request.llm_request,
                        &raw,
                    )
                    .or_else(|| None)
                })
                .or_else(|| {
                    crate::request_fit::unsupported_capability(
                        self.capabilities,
                        &request.llm_request,
                        &raw,
                    )
                })
                .unwrap_or("an eligible route");
            return Err(RunnerError::unsupported_capability(
                self.algorithm.name(),
                capability,
            ));
        }
        let (selected_model, response) = switchyard_llm_client::run(
            Arc::clone(&self.algorithm),
            self.clients.clone(),
            request,
            Arc::new(models),
            observer,
        )
        .await?;
        Ok(RunOutput {
            selected_model,
            response,
        })
    }

    /// Completes routing-time calls without serving a post-routing completion.
    pub async fn decide(&self, request: Request) -> Result<RoutingOutcome, RunnerError> {
        switchyard_llm_client::decide(
            Arc::clone(&self.algorithm),
            self.clients.clone(),
            request,
            Arc::clone(&self.models),
        )
        .await
        .map_err(Into::into)
    }

    /// Executes a model-bearing provider operation through a compatible target.
    pub async fn call_auxiliary(
        &self,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value, RunnerError> {
        let target = match operation {
            AuxiliaryOperation::AnthropicCountTokens => &self.anthropic_auxiliary_target,
            AuxiliaryOperation::ResponsesInputTokens | AuxiliaryOperation::ResponsesCompact => {
                &self.responses_auxiliary_target
            }
        }
        .as_ref()
        .ok_or(RunnerError::AuxiliaryUnsupported)?;
        target
            .client
            .call_auxiliary(&target.model, request, operation)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod context_policy_tests {
    use super::*;
    use crate::RouteReasoningPolicy;

    /// Only the boundary predicate and the stored policy are under test here; both are
    /// plain value logic, so this builds a route with the cheapest algorithm the runner
    /// already uses in its own tests.
    fn build(threshold: Option<u64>, policy: RouteReasoningPolicy) -> Route {
        Route::new(
            Arc::new(libsy::Passthrough),
            ClientRouter::new(std::collections::HashMap::new()),
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::default(),
            policy,
            threshold,
            None,
        )
    }

    /// The recovered historical comparator was strictly greater-than: a request exactly
    /// at the threshold stays on the primary leg and only 61,441 escalates.
    #[test]
    fn the_context_boundary_is_strictly_greater_than() {
        let route = build(Some(61_440), RouteReasoningPolicy::Inherit);
        assert!(
            !route.under_context_pressure(61_440),
            "exactly at the threshold stays on the primary leg"
        );
        assert!(
            route.under_context_pressure(61_441),
            "one token over the threshold escalates"
        );
        assert!(
            !build(None, RouteReasoningPolicy::Inherit).under_context_pressure(1_000_000),
            "a route without a threshold never escalates on context pressure"
        );
    }

    /// The route's policy is what it imposes on a shared physical target.
    #[test]
    fn the_route_policy_is_what_the_route_imposes() {
        for policy in [
            RouteReasoningPolicy::Inherit,
            RouteReasoningPolicy::Medium,
            RouteReasoningPolicy::None,
        ] {
            assert_eq!(
                build(Some(61_440), policy).route_reasoning_policy(),
                policy,
                "the route keeps the policy it was configured with"
            );
        }
    }
}
