// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-1 TOML deployment loading for the shared runner.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use libsy::RuntimeModels;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_llm_client::{
    AuxiliaryOperation, Backend, ClientRouter, DEFAULT_MAX_RETRIES, HttpBackendConfig, ModelConfig,
    TranslatingLlmClient,
};
use switchyard_protocol::{Category, ModelId, RoutedLlmClient, WireFormat};

use crate::{
    AlgorithmSpec, AuxiliaryTarget, CallerAuthKind, DecisionTarget, FleetBuildContext,
    ModelCapabilities, Route, Runner, RunnerError,
};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
const MAX_CONFIGURED_RETRIES: u32 = 10;

type RunnerResult<T> = Result<T, RunnerError>;

pub(crate) fn load_runner(path: impl AsRef<Path>) -> RunnerResult<Runner> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|error| {
        RunnerError::configuration_source(
            format!("failed to read server config {}: {error}", path.display()),
            error,
        )
    })?;
    runner_from_toml(&source).map_err(|error| {
        RunnerError::configuration_source(
            format!("invalid server config {}: {error}", path.display()),
            error,
        )
    })
}

pub(crate) fn runner_from_toml(source: &str) -> RunnerResult<Runner> {
    let config: DeploymentConfig = toml::from_str(source).map_err(|error| {
        RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
    })?;
    config.build()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeploymentConfig {
    schema_version: u32,
    fallback_client: Option<String>,
    #[serde(default)]
    llm_clients: BTreeMap<String, LlmClientConfig>,
    targets: BTreeMap<String, TargetConfig>,
    routes: BTreeMap<String, RouteConfig>,
    /// LocalClaw extension: the host-owned fleet-readiness monitor config.
    #[serde(default)]
    fleet_readiness: Option<crate::facts_config::FleetReadinessConfig>,
    /// LocalClaw extension: capability executor routes (embeddings/rerank).
    /// Parsed by the capability layer; accepted here so the live deployment
    /// file parses as a whole.
    #[serde(default)]
    capability_clients: BTreeMap<String, crate::capability::CapabilityClientConfig>,
    #[serde(default)]
    capabilities: BTreeMap<String, crate::capability::CapabilityRouteConfig>,
}

#[derive(Debug)]
struct RouteConfig {
    id: ModelId,
    context_window: Option<u32>,
    tool_calling: Option<bool>,
    reasoning: Option<bool>,
    vision: Option<bool>,
    /// Route-authoritative reasoning policy (`reasoning_policy = "none" | "low" |
    /// "medium" | "high" | "max"`). Intent, not provider syntax: every target the
    /// route can reach must be able to honor it, checked at load time.
    reasoning_policy: Option<switchyard_protocol::ReasoningPolicy>,
    algorithm: AlgorithmSpec,
}

struct TargetPromptPolicy {
    prompts: HashMap<ModelId, String>,
    routing_answer_target: Option<ModelId>,
}

impl<'de> Deserialize<'de> for RouteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut table = toml::Table::deserialize(deserializer)?;
        let id = take_required(&mut table, "id")?;
        let context_window = take_optional(&mut table, "context_window")?;
        let tool_calling = take_optional(&mut table, "tool_calling")?;
        let reasoning = take_optional(&mut table, "reasoning")?;
        let vision =
            take_optional(&mut table, "vision")?.or(take_optional(&mut table, "supports_vision")?);
        let reasoning_policy = take_optional::<String, _>(&mut table, "reasoning_policy")?
            .map(|value| {
                switchyard_protocol::ReasoningPolicy::parse(&value).ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "unknown reasoning_policy {value:?}; expected one of none, low, medium, high, max"
                    ))
                })
            })
            .transpose()?;
        let algorithm = AlgorithmSpec::deserialize(toml::Value::Table(table))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            id,
            context_window,
            tool_calling,
            reasoning,
            vision,
            reasoning_policy,
            algorithm,
        })
    }
}

fn take_required<T, E>(table: &mut toml::Table, name: &'static str) -> Result<T, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    let value = table.remove(name).ok_or_else(|| E::missing_field(name))?;
    T::deserialize(value).map_err(E::custom)
}

fn take_optional<T, E>(table: &mut toml::Table, name: &'static str) -> Result<Option<T>, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    table
        .remove(name)
        .map(|value| T::deserialize(value).map_err(E::custom))
        .transpose()
}

impl RouteConfig {
    /// A `fleet_router` route's advertised capability envelope must be
    /// satisfiable by its candidate set (truthful model advertisement).
    fn validate_fleet_capabilities(&self, route_name: &str) -> RunnerResult<()> {
        let AlgorithmSpec::FleetRouter { candidates, .. } = &self.algorithm else {
            return Ok(());
        };
        let route_tool_calling = self.tool_calling;
        let route_reasoning = self.reasoning;
        let any_tool = candidates.iter().any(|candidate| candidate.tool_calling);
        let any_reasoning = candidates.iter().any(|candidate| candidate.reasoning);

        if route_tool_calling == Some(true) && !any_tool {
            return Err(RunnerError::configuration(format!(
                "fleet_router route {route_name} advertises tool_calling=true but no \
                 candidate profile supports tool calling"
            )));
        }
        if route_reasoning == Some(true) && !any_reasoning {
            return Err(RunnerError::configuration(format!(
                "fleet_router route {route_name} advertises reasoning=true but no \
                 candidate profile supports reasoning"
            )));
        }
        Ok(())
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            context_window: self.context_window,
            tool_calling: self.tool_calling,
            reasoning: self.reasoning,
            supports_vision: self.vision,
        }
    }

    fn routing_target_names(&self) -> Vec<&str> {
        self.algorithm.routing_target_names()
    }

    fn callable_target_names(&self) -> Vec<&str> {
        self.algorithm.callable_target_names()
    }
}

impl DeploymentConfig {
    /// Validates every declared escalation edge before any route is built.
    ///
    /// Invariants (kept deliberately simple so escalation stays bounded):
    /// * `escalation_max_input_tokens` without an `escalation` destination is a
    ///   configuration error;
    /// * the destination must be a registered route id and not the route itself;
    /// * the destination must not itself declare an escalation (no chains, no
    ///   cycles).
    fn validate_escalations(&self) -> RunnerResult<()> {
        let route_ids = self
            .routes
            .values()
            .map(|config| &config.id)
            .collect::<std::collections::HashSet<&ModelId>>();
        for (route_name, config) in &self.routes {
            let Some(destination) = config.algorithm.escalation_destination() else {
                if config.algorithm.escalation_threshold().is_some() {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} declares escalation_max_input_tokens without an \
                         escalation destination"
                    )));
                }
                continue;
            };
            if destination == &config.id {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} cannot escalate to itself"
                )));
            }
            if !route_ids.contains(destination) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} escalation destination {destination:?} is not a \
                     registered route id"
                )));
            }
            let destination_config = self
                .routes
                .values()
                .find(|candidate| candidate.id == *destination)
                .expect("destination verified above");
            if destination_config
                .algorithm
                .escalation_destination()
                .is_some()
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} escalation destination {destination:?} must not itself \
                     declare an escalation (no escalation chains/cycles)"
                )));
            }
        }
        Ok(())
    }

    /// Validates a route's authoritative reasoning policy against every target
    /// the route can reach (candidates, routing targets, escalation excluded —
    /// the destination route re-validates under its own policy at its own
    /// entry).
    ///
    /// Fail-closed rules, evaluated per reachable target:
    /// * the target's llm client MUST declare a `reasoning_dialect` — a
    ///   policy-bearing route reaching a dialect-less client is a configuration
    ///   error, never a runtime downgrade;
    /// * the dialect must honor the policy (every current dialect expresses
    ///   both modes; unknown future dialects fail here);
    /// * a hard reasoning pin on the TARGET (`reasoning_effort` or a reasoning
    ///   key in `extra_body`) CONTRADICTS the route policy — two authoritative
    ///   sources must not fight; the neutral-target end state is no target pin.
    fn validate_route_reasoning_policy(
        &self,
        route_name: &str,
        route: &RouteConfig,
        policy: &switchyard_protocol::ReasoningPolicy,
        llm_clients: &BTreeMap<String, LlmClientConfig>,
    ) -> RunnerResult<()> {
        let mut checked = std::collections::BTreeSet::new();
        for target_name in route.callable_target_names() {
            let target = self.targets.get(target_name).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "route {route_name} references unknown target {target_name}"
                ))
            })?;
            if !checked.insert(target_name) {
                continue;
            }
            let client_config = llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            if target.reasoning_effort.is_some()
                || target_extra_body_pins_reasoning(&target.extra_body)
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} declares reasoning_policy = {policy} but target {target_name} \
                     also carries a hard reasoning pin (reasoning_effort or extra_body reasoning key); \
                     route policy and target hard pins must not fight — give the target no reasoning \
                     pin (that is what makes it neutral) and let the route decide"
                )));
            }
            match &client_config.reasoning_dialect {
                None => {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} declares reasoning_policy = {policy} but target {target_name}'s \
                         llm client {} declares no reasoning_dialect; add reasoning_dialect to the \
                         [llm_clients.{}] section or drop the route policy",
                        target.llm_client, target.llm_client
                    )));
                }
                Some(dialect) => {
                    // Generalized expressibility invariant (steering §1.3/§1.4):
                    // `honors` must be true only when the dialect can faithfully
                    // express the EXACT policy. Effort-bearing dialects
                    // (openai_effort, chat_template_reasoning_effort) pass the
                    // policy spelling verbatim and upstream vocabularies differ,
                    // so they require the operator-declared `reasoning_efforts`
                    // list; boolean dialects reject every exact effort outright
                    // (an exact effort must never collapse to a bare switch).
                    if !dialect
                        .honors_with_vocabulary(*policy, client_config.reasoning_efforts.as_deref())
                    {
                        let effort_bearing = matches!(
                            dialect,
                            switchyard_protocol::ReasoningDialect::OpenAiEffort
                                | switchyard_protocol::ReasoningDialect::ChatTemplateReasoningEffort
                        );
                        return Err(RunnerError::configuration(
                            // Effort-bearing dialect with NO declared vocabulary:
                            // the operator must declare what the upstream accepts.
                            if effort_bearing && client_config.reasoning_efforts.is_none() {
                                format!(
                                    "route {route_name} declares reasoning_policy = {policy} but target {target_name}'s \
                                     llm client {} uses effort-bearing dialect {dialect:?} without declaring reasoning_efforts; \
                                     declare the upstream's accepted effort vocabulary (e.g. reasoning_efforts = \
                                     [\"none\", \"low\", \"high\", \"max\"]) so unsupported values fail at load time",
                                    target.llm_client
                                )
                            }
                            // Vocabulary declared (or boolean dialect): the policy
                            // simply is not faithfully expressible on this target.
                            else {
                                format!(
                                    "route {route_name} declares reasoning_policy = {policy} but target {target_name}'s \
                                     llm client {} (dialect {dialect:?}, declared vocabulary {:?}) cannot faithfully \
                                     express it; widen the declared vocabulary, switch dialect, or drop the policy",
                                    target.llm_client, client_config.reasoning_efforts
                                )
                            },
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn decision_target(&self, name: &str) -> Option<DecisionTarget> {
        let target = self.targets.get(name)?;
        let client = self.llm_clients.get(&target.llm_client)?;
        Some(DecisionTarget {
            target: name.to_string(),
            model: target.id.clone(),
            format: client.format.wire_format(),
            base_url: client.base_url.as_str().to_string(),
            extra_body: target.extra_body.clone(),
        })
    }

    fn build(self) -> RunnerResult<Runner> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(RunnerError::configuration(format!(
                "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
                self.schema_version
            )));
        }

        let mut route_names_by_id = HashMap::new();
        for (route_name, config) in &self.routes {
            validate_value("route name", route_name)?;
            validate_value(&format!("route {route_name} id"), &config.id)?;
            if let Some(first_route_name) =
                route_names_by_id.insert(config.id.as_str(), route_name.as_str())
            {
                return Err(RunnerError::configuration(format!(
                    "routes {first_route_name} and {route_name} both use id {}; route ids must be unique",
                    config.id
                )));
            }
        }

        if let Some(readiness) = &self.fleet_readiness {
            readiness.validate()?;
        }
        self.validate_escalations()?;
        let fleet_state = if self
            .routes
            .values()
            .any(|route| matches!(route.algorithm, AlgorithmSpec::FleetRouter { .. }))
        {
            Some(std::sync::Arc::new(libsy::SharedFleetState::new(
                libsy::FleetSnapshot::default(),
            )))
        } else {
            None
        };

        // The LLM client keeps one backend per model id, so two targets naming the same model on
        // the same client share it. That is harmless when their request settings agree (an alias
        // for a different system prompt, say) and silently wrong when they do not: the second
        // target's reasoning_effort or extra_body would never reach the wire.
        let mut seen_client_model_ids: HashMap<(&str, &str), (&String, &TargetConfig)> =
            HashMap::new();
        for (target_name, target) in &self.targets {
            validate_value("target name", target_name)?;
            validate_value(&format!("target {target_name} id"), &target.id)?;
            match seen_client_model_ids.entry((target.llm_client.as_str(), target.id.as_str())) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert((target_name, target));
                }
                std::collections::hash_map::Entry::Occupied(slot) => {
                    let (first_name, first) = slot.get();
                    if first.reasoning_effort != target.reasoning_effort
                        || first.extra_body != target.extra_body
                    {
                        return Err(RunnerError::configuration(format!(
                            "targets {first_name} and {target_name} both name model {} on llm client {} but with different reasoning_effort or extra_body; one target per model id is kept, so give each its own model id or llm client",
                            target.id, target.llm_client
                        )));
                    }
                    tracing::warn!(
                        "target {target_name} reuses model id {} on llm client {}; only one target per id is kept and the other is dropped. Give each target a unique model id, or point both routes at one target.",
                        target.id,
                        target.llm_client
                    );
                }
            }
        }

        let mut provider_api_keys = Vec::new();
        let clients = self.build_clients(&mut provider_api_keys)?;
        let targets = self.build_targets();
        let fallback_base_url = self.fallback_base_url()?;
        let mut routes = Vec::with_capacity(self.routes.len());
        for (route_name, config) in &self.routes {
            for target_name in config.callable_target_names() {
                self.targets.get(target_name).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "route references unknown target {target_name}"
                    ))
                })?;
            }
            config.validate_fleet_capabilities(route_name)?;
            if let Some(policy) = &config.reasoning_policy {
                self.validate_route_reasoning_policy(
                    route_name,
                    config,
                    policy,
                    &self.llm_clients,
                )?;
            }
            let mut capabilities = config.capabilities();
            if let AlgorithmSpec::FleetRouter { candidates, .. } = &config.algorithm {
                let any_vision = candidates.iter().any(|candidate| candidate.supports_vision);
                capabilities.supports_vision = capabilities.supports_vision.or(Some(any_vision));
            }
            // A declared reasoning policy IS the route's reasoning capability:
            // advertisement derives from the effective policy (the legacy
            // `reasoning` boolean remains an independent declaration and is
            // only overridden where the policy actually governs).
            if let Some(policy) = &config.reasoning_policy {
                capabilities.reasoning = Some(policy.is_thinking());
            }
            if capabilities.context_window == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} context_window must be greater than zero"
                )));
            }
            let fleet_context = match fleet_state.as_ref() {
                Some(state) => FleetBuildContext {
                    fleet_state: Some(std::sync::Arc::clone(state)),
                },
                None => FleetBuildContext::default(),
            };
            let algorithm = config
                .algorithm
                .build_with_fleet(route_name, &targets, &fleet_context)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let (route_clients, caller_auth) =
                self.build_route_clients(route_name, config, &clients)?;
            let anthropic_auxiliary_target =
                self.build_anthropic_auxiliary_target(config, &clients);
            let responses_auxiliary_target =
                self.build_responses_auxiliary_target(config, &clients);
            let decision_targets = config
                .routing_target_names()
                .into_iter()
                .filter_map(|name| self.decision_target(name))
                .collect();
            let names = config
                .algorithm
                .runtime_model_names(route_name)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let mut models = RuntimeModels::new(resolve_category_models(names.parent, &targets)?);
            if let Some(subagent) = names.subagent {
                models = models.with_subagent(resolve_category_models(subagent, &targets)?);
            }
            let route = Route::new(
                algorithm,
                route_clients,
                caller_auth,
                capabilities,
                anthropic_auxiliary_target,
                responses_auxiliary_target,
                decision_targets,
                models,
            );
            let route = route.with_escalation(
                config.algorithm.escalation_destination().cloned(),
                config.algorithm.escalation_threshold(),
            );
            let route = route.with_reasoning_policy(config.reasoning_policy);
            routes.push((config.id.clone(), route));
        }
        let runner = Runner::new(routes)
            .with_fallback_url(fallback_base_url)
            .with_provider_api_keys(provider_api_keys)
            .with_fleet_state(fleet_state)
            .with_fleet_readiness(self.fleet_readiness)
            .with_capabilities(self.capability_clients, self.capabilities);
        Ok(runner)
    }

    fn build_clients(
        &self,
        provider_api_keys: &mut Vec<String>,
    ) -> RunnerResult<BTreeMap<String, Arc<TranslatingLlmClient>>> {
        let mut models_by_client = self
            .llm_clients
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<BTreeMap<String, Vec<ModelConfig>>>();

        for (name, client_config) in &self.llm_clients {
            validate_value("llm client name", name)?;
            let backend = build_backend(name, client_config, &BTreeMap::new(), None, false)?;
            let (Backend::OpenAiChat(config)
            | Backend::OpenAiResponses(config)
            | Backend::Anthropic(config)) = backend;
            if let Some(key) = config.api_key {
                provider_api_keys.push(key);
            }
        }
        for (target_name, target) in &self.targets {
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            let model_configs = models_by_client
                .get_mut(&target.llm_client)
                .ok_or_else(|| {
                    RunnerError::configuration("validated llm client was not initialized")
                })?;
            if let Some(effort) = &target.reasoning_effort {
                if effort.trim().is_empty() {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort must not be empty"
                    )));
                }
                if matches!(client_config.format, ClientFormat::AnthropicMessages) {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort is only supported on openai_chat and openai_responses clients"
                    )));
                }
            }
            model_configs.push(ModelConfig::new(
                target.id.clone(),
                build_backend(
                    &target.llm_client,
                    client_config,
                    &target.extra_body,
                    target.reasoning_effort.clone(),
                    target.strip_reasoning_content,
                )?,
                None,
            ));
        }

        let mut clients = BTreeMap::new();
        for (name, model_configs) in models_by_client {
            let client = Arc::new(
                TranslatingLlmClient::new(&model_configs)
                    .map_err(|error| RunnerError::configuration(error.to_string()))?,
            );
            clients.insert(name, client);
        }
        Ok(clients)
    }

    fn build_targets(&self) -> BTreeMap<String, ModelId> {
        self.targets
            .iter()
            .map(|(name, config)| (name.clone(), config.id.clone()))
            .collect()
    }

    fn build_route_clients(
        &self,
        route_name: &str,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> RunnerResult<(ClientRouter, Option<CallerAuthKind>)> {
        let mut by_model = HashMap::new();
        let mut caller_auth = None;
        for name in route.callable_target_names() {
            let target = self.targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            let client = clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!("target {name} has no constructed llm client"))
            })?;
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            if client_config.forward_auth {
                let target_auth = client_config.format.caller_auth_kind();
                if caller_auth.is_some_and(|kind| kind != target_auth) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} cannot forward both Anthropic and OpenAI caller credentials"
                    )));
                }
                caller_auth = Some(target_auth);
            }
            let client: Arc<dyn RoutedLlmClient> = client.clone();
            by_model.insert(target.id.clone(), client);
        }
        let TargetPromptPolicy {
            prompts,
            routing_answer_target,
        } = self.build_route_target_prompts(route_name, route)?;
        let router =
            ClientRouter::new_with_target_prompts(by_model, prompts, routing_answer_target);
        Ok((router, caller_auth))
    }

    /// Builds the effective prompt policy for this route's completion targets.
    fn build_route_target_prompts(
        &self,
        route_name: &str,
        route: &RouteConfig,
    ) -> RunnerResult<TargetPromptPolicy> {
        let mut prompts = HashMap::new();
        let mut aliases = HashMap::<&ModelId, Option<&str>>::new();
        for name in route.algorithm.routing_target_names() {
            let target = self.targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            let prompt = target.system_prompt.as_deref();
            if aliases
                .insert(&target.id, prompt)
                .is_some_and(|configured| configured != prompt)
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} maps completion target aliases to model {} with different system_prompt values",
                    target.id
                )));
            }
            if let Some(prompt) = prompt {
                prompts.insert(target.id.clone(), prompt.to_string());
            }
        }
        let mut policy = TargetPromptPolicy {
            prompts,
            routing_answer_target: None,
        };
        let Some((response_name, dependency_name)) =
            route.algorithm.routing_response_and_dependency()
        else {
            return Ok(policy);
        };
        let response = self.targets.get(response_name).ok_or_else(|| {
            RunnerError::configuration(format!("route references unknown target {response_name}"))
        })?;
        if !policy.prompts.contains_key(&response.id) {
            return Ok(policy);
        }
        let dependency = self.targets.get(dependency_name).ok_or_else(|| {
            RunnerError::configuration(format!("route references unknown target {dependency_name}"))
        })?;
        if response.id == dependency.id {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot apply system_prompt to target {response_name}: model {} is also used by routing-only target {dependency_name}",
                response.id,
            )));
        }
        policy.routing_answer_target = Some(response.id.clone());
        Ok(policy)
    }

    fn fallback_base_url(&self) -> RunnerResult<Option<String>> {
        let Some(name) = &self.fallback_client else {
            return Ok(None);
        };
        let config = self.llm_clients.get(name).ok_or_else(|| {
            RunnerError::configuration(format!(
                "fallback_client references unknown llm client {name}"
            ))
        })?;
        Ok(Some(config.base_url.as_str().to_string()))
    }

    fn build_anthropic_auxiliary_target(
        &self,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<AuxiliaryTarget> {
        route
            .routing_target_names()
            .into_iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let target = self.build_auxiliary_target(
                    name,
                    clients,
                    AuxiliaryOperation::AnthropicCountTokens,
                )?;
                Some((count_tokens_priority(name, &target.model), index, target))
            })
            .min_by_key(|(priority, index, _)| (*priority, *index))
            .map(|(_, _, target)| target)
    }

    fn build_responses_auxiliary_target(
        &self,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<AuxiliaryTarget> {
        route.routing_target_names().into_iter().find_map(|name| {
            self.build_auxiliary_target(name, clients, AuxiliaryOperation::ResponsesInputTokens)
        })
    }

    fn build_auxiliary_target(
        &self,
        name: &str,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        operation: AuxiliaryOperation,
    ) -> Option<AuxiliaryTarget> {
        let target = self.targets.get(name)?;
        let client = clients.get(&target.llm_client)?;
        client
            .supports_auxiliary(&target.id, operation)
            .then(|| AuxiliaryTarget {
                model: target.id.clone(),
                client: client.clone(),
            })
    }
}

fn count_tokens_priority(target_name: &str, model_id: &ModelId) -> usize {
    let target_name = target_name.to_ascii_lowercase();
    let model_id = model_id.to_ascii_lowercase();
    ["opus", "sonnet", "haiku"]
        .iter()
        .position(|hint| target_name.contains(hint) || model_id.contains(hint))
        .unwrap_or(3)
}

/// A client endpoint, parsed when the config loads rather than checked afterwards.
///
/// Holding a `HttpBaseUrl` is proof the value is an absolute HTTP(S) URL, so no
/// later stage has to re-check it or can forget to.
#[derive(Clone, Debug)]
pub struct HttpBaseUrl(pub(crate) reqwest::Url);

impl HttpBaseUrl {
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl<'de> Deserialize<'de> for HttpBaseUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let url = reqwest::Url::parse(raw.trim()).map_err(|error| {
            serde::de::Error::custom(format!("base_url must be an absolute HTTP(S) URL: {error}"))
        })?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(serde::de::Error::custom(
                "base_url must be an absolute HTTP(S) URL",
            ));
        }
        Ok(Self(url))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClientConfig {
    format: ClientFormat,
    base_url: HttpBaseUrl,
    api_key_env: Option<String>,
    #[serde(default)]
    forward_auth: bool,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
    /// Reasoning-control dialect of this client's upstream: how an abstract
    /// route [`ReasoningPolicy`] is expressed on the wire. `Responses is the
    /// common transport, not the reasoning dialect` — two clients can share
    /// `format = "openai_responses"` yet speak different reasoning controls.
    /// Required whenever some route policy must be enforced through this
    /// client; a route that declares a `reasoning_policy` and reaches a
    /// client without one fails configuration.
    ///
    /// * `openai_effort` — `reasoning.effort = "<policy>"` with effort `none`
    ///   disabling reasoning (DeepSeek, OpenAI, and OpenAI-compatible
    ///   Responses endpoints); preserves effort gradations verbatim.
    /// * `openrouter_enabled` — `reasoning = { enabled = <bool> }` (OpenRouter);
    ///   a thinking policy collapses to "enabled" (the endpoint cannot
    ///   express amounts).
    /// * `llama_cpp_enable_thinking` —
    ///   `chat_template_kwargs.enable_thinking = <bool>` (llama.cpp-derived
    ///   Qwen template endpoints); same boolean-collapse semantics (HTPC
    ///   contract).
    /// * `chat_template_reasoning_effort` — none =
    ///   `chat_template_kwargs.enable_thinking = false`, any thinking policy =
    ///   `chat_template_kwargs.reasoning_effort = "<policy>"` (the ComfyNinja
    ///   Q3/Q4 contract: effort-bearing thinking side, boolean NT side).
    ///
    /// Effort vocabulary: boolean dialects collapse any thinking policy to
    /// their native "on" semantic (part of the dialect contract, tested). An
    /// `openai_effort` client passes the effort string through VERBATIM, and
    /// providers differ on what they accept (e.g. DeepSeek documents
    /// none|low|high|max with no `medium`; OpenRouter and OpenAI vocabularies
    /// differ again), so an `openai_effort` client serving a policy-bearing
    /// route MUST also declare `reasoning_efforts` — the operator-validated
    /// vocabulary for that upstream — and every policy-bearing route reaching
    /// it is checked against that list at load time. Missing list or a policy
    /// outside it is a configuration error, never a silent approximation.
    reasoning_dialect: Option<switchyard_protocol::ReasoningDialect>,
    /// The authoritative effort vocabulary of this client's upstream, required
    /// for `openai_effort` clients reachable from policy-bearing routes.
    /// Values use the dialect's wire spelling (e.g. "none", "medium").
    reasoning_efforts: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    id: ModelId,
    llm_client: String,
    #[serde(default)]
    extra_body: BTreeMap<String, Value>,
    system_prompt: Option<String>,
    /// Reasoning effort forced on every request to this target, replacing the caller's value.
    /// Only meaningful on `openai_chat` and `openai_responses` clients.
    reasoning_effort: Option<String>,
    /// Drop replayed reasoning payloads from outbound messages for this target.
    /// Set on OpenRouter-hosted lanes so replayed chain-of-thought is never
    /// forwarded to a provider that bills for it without requiring it.
    #[serde(default)]
    strip_reasoning_content: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum ClientFormat {
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

impl ClientFormat {
    const fn wire_format(self) -> WireFormat {
        match self {
            Self::OpenAiChat => WireFormat::OpenAiChat,
            Self::OpenAiResponses => WireFormat::OpenAiResponses,
            Self::AnthropicMessages => WireFormat::AnthropicMessages,
        }
    }

    const fn caller_auth_kind(self) -> CallerAuthKind {
        match self {
            Self::AnthropicMessages => CallerAuthKind::Anthropic,
            Self::OpenAiChat | Self::OpenAiResponses => CallerAuthKind::OpenAi,
        }
    }
}

/// Resolves one scope's configured target names to the models the driver serves.
fn resolve_category_models(
    names: HashMap<Category, Vec<String>>,
    targets: &BTreeMap<String, ModelId>,
) -> RunnerResult<HashMap<Category, Vec<ModelId>>> {
    names
        .into_iter()
        .map(|(category, names)| {
            let models = names
                .into_iter()
                .map(|name| {
                    targets.get(&name).cloned().ok_or_else(|| {
                        RunnerError::configuration(format!(
                            "route references unknown target {name}"
                        ))
                    })
                })
                .collect::<RunnerResult<Vec<_>>>()?;
            Ok((category, models))
        })
        .collect()
}

/// Whether a target's `extra_body` carries a hard reasoning-control pin
/// (`reasoning`, `reasoning_effort`, or `chat_template_kwargs.enable_thinking`
/// at the top level of the target's defaults).
fn target_extra_body_pins_reasoning(extra_body: &BTreeMap<String, Value>) -> bool {
    extra_body.contains_key("reasoning")
        || extra_body.contains_key("reasoning_effort")
        || extra_body
            .get("chat_template_kwargs")
            .and_then(Value::as_object)
            .is_some_and(|kwargs| kwargs.contains_key("enable_thinking"))
}

fn build_backend(
    client_name: &str,
    config: &LlmClientConfig,
    extra_body: &BTreeMap<String, Value>,
    reasoning_effort: Option<String>,
    strip_reasoning_content: bool,
) -> RunnerResult<Backend> {
    if config.max_retries > MAX_CONFIGURED_RETRIES {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} max_retries must be at most {MAX_CONFIGURED_RETRIES}"
        )));
    }
    if config.forward_auth && config.api_key_env.is_some() {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} cannot set both forward_auth and api_key_env"
        )));
    }
    let api_key = config
        .api_key_env
        .as_deref()
        .map(|variable| {
            if variable.trim().is_empty() {
                return Err(RunnerError::configuration(format!(
                    "llm client {client_name} api_key_env must not be empty"
                )));
            }
            let api_key = std::env::var(variable).map_err(|error| {
                RunnerError::configuration(format!(
                    "llm client {client_name} could not read api_key_env {variable}: {error}"
                ))
            })?;
            if api_key.trim().is_empty() {
                return Err(RunnerError::configuration(format!(
                    "llm client {client_name} api_key_env {variable} is empty"
                )));
            }
            Ok(api_key)
        })
        .transpose()?;
    let http = HttpBackendConfig {
        base_url: config.base_url.as_str().to_string(),
        api_key,
        forward_auth: config.forward_auth,
        extra_headers: config.extra_headers.clone(),
        extra_body: extra_body.clone(),
        reasoning_effort,
        strip_reasoning_content,
        reasoning_dialect: config.reasoning_dialect,
        reasoning_efforts: config.reasoning_efforts.clone(),
        max_retries: config.max_retries,
    };
    let backend = match config.format {
        ClientFormat::OpenAiChat => Backend::OpenAiChat(http),
        ClientFormat::OpenAiResponses => Backend::OpenAiResponses(http),
        ClientFormat::AnthropicMessages => Backend::Anthropic(http),
    };
    Ok(backend)
}

// A function so that serde default can use it.
const fn default_max_retries() -> u32 {
    DEFAULT_MAX_RETRIES
}

fn validate_value(label: &str, value: &str) -> RunnerResult<()> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(RunnerError::configuration(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_presentation_fields_are_split_from_the_algorithm() {
        let route: RouteConfig = toml::from_str(
            r#"
type = "random"
id = "switchyard/random"
context_window = 128000
tool_calling = true
reasoning = false
targets = ["fast", "strong"]
weights = [1.0, 2.0]
seed = 7
"#,
        )
        .expect("route should deserialize");

        assert_eq!(route.id, "switchyard/random");
        assert_eq!(route.context_window, Some(128_000));
        assert_eq!(route.tool_calling, Some(true));
        assert_eq!(route.reasoning, Some(false));
        assert_eq!(route.algorithm.routing_target_names(), ["fast", "strong"]);
    }

    #[test]
    fn route_algorithm_unknown_fields_are_rejected() {
        let error = toml::from_str::<RouteConfig>(
            r#"
type = "passthrough"
id = "switchyard/fast"
target = "fast"
bogus = true
"#,
        )
        .expect_err("unknown route keys must be rejected");

        assert!(error.to_string().contains("bogus"), "{error}");
    }

    #[test]
    fn deployment_errors_keep_one_path_context() {
        let path = Path::new("/definitely/missing/switchyard-routes.toml");
        let error = match load_runner(path) {
            Ok(_) => panic!("missing config should fail"),
            Err(error) => error,
        };
        let message = error.to_string();

        assert_eq!(message.matches(&path.display().to_string()).count(), 1);
        assert!(message.starts_with("failed to read server config"));
    }
}
#[cfg(test)]
mod deployment_tests {
    use super::*;
    use serde_json::json;

    const VALID_CONFIG: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.classifier]
id = "classifier/model"
llm_client = "primary"

[targets.strong]
id = "strong/model"
llm_client = "responses"

[targets.weak]
id = "weak/model"
llm_client = "anthropic"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["strong", "weak"]

[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "weak"
"#;

    #[test]
    fn public_runner_from_toml_builds_a_deployment() -> RunnerResult<()> {
        let runner = Runner::from_toml(VALID_CONFIG)?;

        let classifier = runner
            .route("switchyard/classifier")
            .expect("classifier route should exist");
        let models = classifier.models();
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("classifier/model")]
        );
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("weak/model")]
        );
        assert_eq!(
            models.models_for(&Category::Capable),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("weak/model"), ModelId::from("strong/model")]
        );
        assert!(runner.route("switchyard/passthrough").is_some());
        Ok(())
    }

    #[test]
    fn duplicate_route_ids_are_rejected() {
        let config = format!(
            r#"{VALID_CONFIG}

[routes.duplicate]
id = "switchyard/passthrough"
type = "passthrough"
target = "strong"
"#
        );

        let error = error_message(&config);

        assert!(
            error.contains(
                "routes duplicate and passthrough both use id switchyard/passthrough; route ids must be unique"
            ),
            "{error}"
        );
    }

    fn error_message(toml: &str) -> String {
        match runner_from_toml(toml) {
            Ok(_) => "configuration unexpectedly succeeded".to_string(),
            Err(error) => error.to_string(),
        }
    }

    fn with_subagent_llm_classifier(config: &str, route: &str, extra: &str) -> String {
        let mut configured = config.to_string();
        configured.push_str(&format!("\n[routes.{route}.subagents]\n"));
        configured.push_str(
            r#"type = "llm_classifier"
mode = "custom"
models = { judge = ["classifier"], capable = ["strong"], efficient = ["weak"], any = ["strong", "weak"] }
default_target = "efficient"
prompt = "Select a target for this delegated task."
response_schema = '{"type":"object","properties":{"target":{"type":"string","enum":["capable","efficient"]}},"required":["target"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/target" }
classify_trigger = "new_session""#,
        );
        configured.push_str(extra);
        configured
    }

    #[test]
    fn subagent_models_stay_separate_from_the_parent_tiers() -> RunnerResult<()> {
        // The sub-agent target is also the parent's capable tier. Merged into one group it
        // would be indistinguishable from that tier, and delegated work would follow the
        // parent's ordering instead of its own configured target.
        let runner = runner_from_toml(&with_subagent_passthrough(&stage_config(), "stage"))?;
        let models = runner
            .route("switchyard/stage")
            .expect("stage route should exist")
            .models();

        assert_eq!(
            models.subagent_models_for(&Category::Any),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("strong/model"), ModelId::from("weak/model")]
        );
        Ok(())
    }

    fn with_subagent_passthrough(config: &str, route: &str) -> String {
        format!("{config}\n[routes.{route}.subagents]\ntype = \"passthrough\"\ntarget = \"strong\"")
    }

    fn stage_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.stage_judge]
id = "stage-judge/model"
llm_client = "primary"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0
capable_hold_turns = 2

[routes.stage.tool_semantics]
observe = ["lookup_customer"]
mutate = ["send_payment"]
plan = ["create_workflow"]
new = ["send_message"]

[routes.stage.classifier]
target = "stage_judge"
base_threshold = 0.5
"#
        )
    }

    fn composite_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.tier_judge]
id = "tier-judge/model"
llm_client = "primary"

[routes.composed]
id = "switchyard/hier"
type = "composite"

[routes.composed.classifier]
target = "tier_judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.composed.stage]
capable_target = "strong"
efficient_target = "weak"
confidence_threshold = 0.5
capable_hold_turns = 2

[routes.composed.stage.tool_semantics]
new = ["send_message"]
"#
        )
    }

    #[test]
    fn composite_route_builds_and_claims_both_tiers_and_its_judge() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_config())?;
        assert!(
            runner
                .models()
                .any(|model| model.id.as_str() == "switchyard/hier")
        );
        Ok(())
    }

    #[test]
    fn builds_all_supported_algorithm_types() -> RunnerResult<()> {
        let state = runner_from_toml(VALID_CONFIG)?;
        // The model id array is sorted alphabetically
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            [
                "switchyard/classifier",
                "switchyard/noop",
                "switchyard/passthrough",
                "switchyard/random",
            ]
        );
        Ok(())
    }

    #[test]
    fn stage_rejects_ambiguous_or_non_additive_tool_semantics() {
        for (configured, expected) in [
            (
                stage_config().replace(
                    "mutate = [\"send_payment\"]",
                    "mutate = [\"LOOKUP_CUSTOMER\"]",
                ),
                "appears in both tool_semantics.observe and tool_semantics.mutate",
            ),
            (
                stage_config().replace("new = [\"send_message\"]", "new = [\"Read\"]"),
                "already has built-in semantics and cannot be reclassified",
            ),
            (
                stage_config().replace("observe = [\"lookup_customer\"]", "observe = [\" \"]"),
                "contains an empty tool name",
            ),
        ] {
            let message = error_message(&configured);
            assert!(
                message.contains(expected),
                "expected {expected:?} in error, got: {message}"
            );
        }
    }

    #[test]
    fn passthrough_and_stage_accept_subagent_routing() -> RunnerResult<()> {
        let stage = stage_config();
        let stage_with_classifier = with_subagent_llm_classifier(&stage, "stage", "");
        let parsed: DeploymentConfig = toml::from_str(&stage_with_classifier).map_err(|error| {
            RunnerError::configuration(format!("failed to parse stage config: {error}"))
        })?;
        let Some(stage_route) = parsed.routes.get("stage") else {
            return Err(RunnerError::configuration("stage route is missing"));
        };
        let callable_targets = stage_route.callable_target_names();
        for expected in ["strong", "weak", "stage_judge", "classifier"] {
            assert!(callable_targets.contains(&expected));
        }

        for configured in [
            with_subagent_llm_classifier(VALID_CONFIG, "passthrough", ""),
            with_subagent_passthrough(VALID_CONFIG, "passthrough"),
            stage_with_classifier,
            with_subagent_passthrough(&stage, "stage"),
        ] {
            runner_from_toml(&configured)?;
        }
        Ok(())
    }

    #[test]
    fn aliased_completion_targets_reject_prompt_conflicts() {
        let configured = stage_config()
            .replace(
                "id = \"strong/model\"\nllm_client = \"responses\"",
                "id = \"strong/model\"\nllm_client = \"responses\"\nsystem_prompt = \"capable\"",
            )
            .replace(
                "[routes.stage]",
                "[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\"\n\n[routes.stage]",
            )
            .replace("efficient_target = \"weak\"", "efficient_target = \"strong_alias\"");
        let message = error_message(&configured);
        assert!(
            message.contains("completion target aliases to model strong/model with different system_prompt values"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn prompted_routing_response_cannot_share_a_model_with_a_dependency() {
        let configured = VALID_CONFIG
            .replace(
                "id = \"classifier/model\"\nllm_client = \"primary\"",
                "id = \"weak/model\"\nllm_client = \"primary\"",
            )
            .replace(
                "id = \"weak/model\"\nllm_client = \"anthropic\"",
                "id = \"weak/model\"\nllm_client = \"anthropic\"\nsystem_prompt = \"answer prompt\"",
            )
            .replace(
                "base_threshold = 0.5",
                "base_threshold = 0.5\nescalation = { confirmations = 1 }",
            );

        let message = error_message(&configured);

        assert!(
            message.contains("cannot apply system_prompt to target weak: model weak/model is also used by routing-only target classifier"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn rejects_invalid_unreferenced_llm_client() {
        let invalid = format!(
            "{VALID_CONFIG}\n\
             [llm_clients.unused]\n\
             format = \"openai_chat\"\n\
             base_url = \"not a url\"\n"
        );
        let message = error_message(&invalid);
        assert!(
            message.contains("base_url must be an absolute HTTP(S) URL"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn an_escalation_table_switches_the_classifier_route_to_escalation() -> RunnerResult<()> {
        // Present: the classifier target judges the weak tier's reply each turn instead of
        // picking a tier ahead of it. The route builds either way, so the assertion is that
        // the knob parses and its settings reach the algorithm's validation.
        let escalating = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalating)?;

        // A setting that would starve the judge is rejected here rather than on the first
        // request, the same as any other unusable route configuration.
        let starved = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 0 }",
        );
        assert!(error_message(&starved).contains("confirmations must be at least 1"));
        Ok(())
    }

    #[test]
    fn a_target_reasoning_effort_parses_and_is_rejected_where_unsupported() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let weak = "[targets.weak]\nid = \"weak/model\"\nllm_client = \"anthropic\"";
        assert!(VALID_CONFIG.contains(strong) && VALID_CONFIG.contains(weak));

        let forced = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \"max\""));
        runner_from_toml(&forced)?;

        let blank = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \" \""));
        assert!(error_message(&blank).contains("reasoning_effort must not be empty"));

        let anthropic = VALID_CONFIG.replace(weak, &format!("{weak}\nreasoning_effort = \"high\""));
        assert!(
            error_message(&anthropic)
                .contains("only supported on openai_chat and openai_responses")
        );
        Ok(())
    }

    #[test]
    fn duplicate_targets_with_conflicting_settings_are_rejected() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        assert!(VALID_CONFIG.contains(strong));
        // Same model, same client, different effort: the second target could never take effect.
        let conflicting = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\n\n[targets.strong_max]\nid = \"strong/model\"\nllm_client = \"responses\"\nreasoning_effort = \"max\""
            ),
        );
        assert!(
            error_message(&conflicting).contains("different reasoning_effort or extra_body"),
            "{}",
            error_message(&conflicting)
        );
        // An alias with identical settings is still allowed (it only warns).
        let alias = VALID_CONFIG.replace(
            strong,
            &format!("{strong}\n\n[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\""),
        );
        runner_from_toml(&alias)?;
        Ok(())
    }

    #[test]
    fn classifier_judge_completion_caps_are_configurable() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 512",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 256\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;
        Ok(())
    }

    #[test]
    fn classifier_prompts_are_configurable_in_both_modes() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom capability rubric\"",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom trajectory rubric\"\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;

        let empty = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"   \"",
        );
        assert!(error_message(&empty).contains("classifier prompt must not be empty"));

        let schema_placeholder = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"{{RESPONSE_SCHEMA}}\"",
        );
        assert!(
            error_message(&schema_placeholder)
                .contains("Switchyard supplies the schema automatically")
        );
        Ok(())
    }

    #[test]
    fn mode_custom_rejects_capability_fields() {
        let mixed = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "mode = \"custom\"\nbase_threshold = 0.5",
        );

        assert!(
            error_message(&mixed)
                .contains("mode custom cannot use capability or escalation fields")
        );
    }

    #[test]
    fn stage_router_rejects_an_unknown_field() {
        let config = stage_config().replace(
            "picker = \"efficient_first\"",
            "picker = \"efficient_first\"\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn auto_route_builds_a_stage_router_with_no_extra_fields() -> RunnerResult<()> {
        let config = format!(
            r#"{VALID_CONFIG}
[routes.auto]
id = "switchyard/auto"
type = "auto"
capable_target = "strong"
efficient_target = "weak"
"#
        );
        let runner = runner_from_toml(&config)?;
        assert!(runner.route("switchyard/auto").is_some());
        Ok(())
    }

    #[test]
    fn composite_stage_block_rejects_an_unknown_field() {
        let config = composite_config().replace(
            "confidence_threshold = 0.5",
            "confidence_threshold = 0.5\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn rejects_unknown_fields_and_algorithm_types() {
        let unknown_field =
            VALID_CONFIG.replace("schema_version = 1", "schema_version = 1\nmagic = true");
        assert!(error_message(&unknown_field).contains("unknown field"));

        let nested_completion_cap = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { max_output_tokens = 256 }",
        );
        assert!(error_message(&nested_completion_cap).contains("unknown field"));

        let unknown_classifier_field = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nclassifier_magic = true",
        );
        assert!(error_message(&unknown_classifier_field).contains("unknown field"));

        let target_capability = VALID_CONFIG.replace(
            "llm_client = \"responses\"",
            "llm_client = \"responses\"\ncontext_window = 1000000",
        );
        assert!(error_message(&target_capability).contains("unknown field `context_window`"));

        let unknown_algorithm = VALID_CONFIG.replace("type = \"noop\"", "type = \"imaginary\"");
        assert!(error_message(&unknown_algorithm).contains("unknown variant"));
    }

    #[test]
    fn rejects_unknown_stage_classifier_fields() {
        // Nested classifier typos must fail instead of silently using a default.
        let config = format!(
            r#"{VALID_CONFIG}

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
classifier_magic = true
"#
        );

        let error = error_message(&config);
        assert!(
            error.contains("unknown field `classifier_magic`"),
            "{error}"
        );
    }

    #[test]
    fn rejects_invalid_references_and_parameters() {
        let cases = [
            (
                VALID_CONFIG.replace("llm_client = \"primary\"", "llm_client = \"missing\""),
                "unknown llm client missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"missing\"]",
                ),
                "unknown target missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [0, 0]",
                ),
                "at least one weight must be positive",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"strong\"]",
                ),
                "targets must be unique, strong is repeated",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [1]",
                ),
                "expected 2 weights, got 1",
            ),
            (
                VALID_CONFIG.replace("base_threshold = 0.5", "base_threshold = 1.5"),
                "base_threshold must be between 0 and 1",
            ),
            (
                VALID_CONFIG.replace("classifier_target = \"classifier\"\n", ""),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"\"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"   \"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nthreshold_step = -0.1",
                ),
                "threshold_step must be finite and greater than or equal to 0",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.8\nthreshold_step = 0.11",
                ),
                "base_threshold + 2 * threshold_step must be at most 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmax_output_tokens = 0\nescalation = { confirmations = 2 }",
                ),
                "max_output_tokens must be at least 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmessage_hash_fallback = true",
                ),
                "message_hash_fallback requires classify_trigger = new_session",
            ),
            (
                with_subagent_llm_classifier(
                    VALID_CONFIG,
                    "passthrough",
                    "\nmessage_hash_fallback = true",
                ),
                "cannot use message_hash_fallback",
            ),
            (
                with_subagent_llm_classifier(VALID_CONFIG, "passthrough", "")
                    .replace("mode = \"custom\"", "mode = \"capability\""),
                "mode capability cannot use custom classifier fields",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "escalation = { confirmations = 2 }\nclassify_trigger = \"user_turn\"",
                ),
                "mode escalation cannot use classify_trigger",
            ),
            (
                VALID_CONFIG.replace("schema_version = 1", "schema_version = 2"),
                "unsupported schema_version 2",
            ),
            (
                VALID_CONFIG.replace("[targets.strong]", "[targets.\" strong \"]"),
                "target name must be non-empty and have no surrounding whitespace",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\ncontext_window = 0",
                ),
                "route random context_window must be greater than zero",
            ),
        ];

        for (toml, expected) in cases {
            let error = error_message(&toml);
            assert!(
                error.contains(expected),
                "expected error containing {expected}, got {error}"
            );
        }
    }

    #[test]
    fn accepts_duplicate_target_model_ids_on_one_client() -> RunnerResult<()> {
        // Two targets share one model id on one client. The client keeps one and drops the
        // other, so the build warns and still succeeds, and both routes resolve. Serving one
        // model under two route names this way is allowed; pointing both routes at one target
        // is the tidier form.
        const SAME_MODEL_TWO_ROUTES: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.fast]
id = "gpt-4o"
llm_client = "primary"

[targets.smart]
id = "gpt-4o"
llm_client = "primary"

[routes.fast]
id = "switchyard/fast"
type = "passthrough"
target = "fast"

[routes.smart]
id = "switchyard/smart"
type = "passthrough"
target = "smart"
"#;
        let state = runner_from_toml(SAME_MODEL_TWO_ROUTES)?;
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/fast", "switchyard/smart"]
        );
        Ok(())
    }

    #[test]
    fn accepts_same_model_id_on_different_llm_clients() -> RunnerResult<()> {
        // The same model id served by two llm clients never collides (each client keys its own
        // models), so cross-provider A/B builds with no warning; only a repeat within one client
        // warns.
        const CROSS_PROVIDER: &str = r#"
schema_version = 1

[llm_clients.openai]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.azure]
format = "openai_chat"
base_url = "https://azure.test/v1"

[targets.openai]
id = "gpt-4o"
llm_client = "openai"

[targets.azure]
id = "gpt-4o"
llm_client = "azure"

[routes.openai]
id = "switchyard/openai-gpt4o"
type = "passthrough"
target = "openai"

[routes.azure]
id = "switchyard/azure-gpt4o"
type = "passthrough"
target = "azure"
"#;
        runner_from_toml(CROSS_PROVIDER)?;
        Ok(())
    }

    #[test]
    fn accepts_relative_weights_and_seed() -> RunnerResult<()> {
        let weighted = VALID_CONFIG.replace(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nweights = [1, 3]\nseed = 42",
        );
        runner_from_toml(&weighted)?;
        Ok(())
    }

    #[test]
    fn accepts_new_session_trigger_with_message_hash_fallback() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.25\nthreshold_step = 0.1\nclassify_trigger = \"new_session\"\nmessage_hash_fallback = true",
        );
        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn target_extra_body_is_parsed_and_applied_to_its_backend() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             extra_body = { service_tier = \"priority\", \
             chat_template_kwargs = { enable_thinking = false } }",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(RunnerError::configuration("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        let backend = build_backend("primary", client, &target.extra_body, None, false)?;

        assert_eq!(
            backend.extra_body().get("service_tier"),
            Some(&json!("priority"))
        );
        assert_eq!(
            backend
                .extra_body()
                .get("chat_template_kwargs")
                .and_then(|value| value.get("enable_thinking")),
            Some(&json!(false))
        );
        Ok(())
    }

    #[test]
    fn retry_budget_defaults_and_accepts_an_override() -> RunnerResult<()> {
        let default: DeploymentConfig = toml::from_str(VALID_CONFIG).map_err(|error| {
            RunnerError::configuration(format!("failed to parse default config: {error}"))
        })?;
        let Some(primary) = default.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, DEFAULT_MAX_RETRIES);

        let explicit = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 0",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&explicit).map_err(|error| {
            RunnerError::configuration(format!("failed to parse explicit retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, 0);

        let maximum = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!(
                "base_url = \"https://example.test/v1\"\nmax_retries = {MAX_CONFIGURED_RETRIES}"
            ),
            1,
        );
        let config: DeploymentConfig = toml::from_str(&maximum).map_err(|error| {
            RunnerError::configuration(format!("failed to parse maximum retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, MAX_CONFIGURED_RETRIES);
        Ok(())
    }

    #[test]
    fn rejects_headers_that_switchyard_sets() {
        let cases = [
            (
                "base_url = \"https://example.test/v1\"",
                "base_url = \"https://example.test/v1\"\n\
                 extra_headers = { AUTHORIZATION = \"Bearer custom-key\" }",
                "AUTHORIZATION",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"X-Api-Key\" = \"custom-key\" }",
                "X-Api-Key",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"ANTHROPIC-VERSION\" = \"custom-version\" }",
                "ANTHROPIC-VERSION",
            ),
        ];

        for (original, replacement, header) in cases {
            let configured = VALID_CONFIG.replacen(original, replacement, 1);
            let error = error_message(&configured);
            assert!(
                error.contains(&format!("extra_headers cannot set {header:?}")),
                "expected {header} to be rejected, got: {error}"
            );
        }
    }

    #[test]
    fn accepts_additional_headers() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { X-Inference-Priority = \"batch\" }",
            1,
        );

        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn retry_budget_rejects_negative_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = -1",
            1,
        );
        assert!(error_message(&invalid).contains("max_retries"));
    }

    #[test]
    fn retry_budget_rejects_excessive_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 11",
            1,
        );
        assert!(
            error_message(&invalid).contains("llm client primary max_retries must be at most 10")
        );
    }

    #[test]
    fn api_key_environment_reference_is_validated() {
        let missing = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\napi_key_env = \"SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET\"",
            1,
        );
        assert!(error_message(&missing).contains("SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET"));

        const EMPTY_KEY_ENV: &str = "SWITCHYARD_CONFIG_TEST_EMPTY_KEY";
        unsafe {
            // "unsafe" is for concurrent reads and writes, very rare
            std::env::set_var(EMPTY_KEY_ENV, "");
        }
        let empty = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{EMPTY_KEY_ENV}\""),
            1,
        );
        let message = error_message(&empty);
        unsafe {
            std::env::remove_var(EMPTY_KEY_ENV);
        }
        assert!(message.contains("is empty"));
    }

    #[test]
    fn forward_auth_rejects_conflicting_credentials() {
        let competing_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
                 forward_auth = true\n\
                 api_key_env = \"UNUSED_TEST_KEY\"",
            1,
        );
        assert!(
            error_message(&competing_auth).contains("cannot set both forward_auth and api_key_env")
        );

        let static_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test\"",
            "base_url = \"https://example.test\"\n\
                 forward_auth = true\n\
                 extra_headers = { Authorization = \"static-value\" }",
            1,
        );
        assert!(error_message(&static_auth).contains("extra_headers cannot set \"Authorization\""));

        let static_beta = static_auth.replace("Authorization", "anthropic-beta");
        assert!(
            error_message(&static_beta).contains("extra_headers cannot set \"anthropic-beta\"")
        );

        for header in ["chatgpt-account-id", "x-openai-fedramp"] {
            let static_context = VALID_CONFIG.replacen(
                "base_url = \"https://example.test/v1\"",
                &format!(
                    "base_url = \"https://example.test/v1\"\n\
                     forward_auth = true\n\
                     extra_headers = {{ \"{header}\" = \"static-value\" }}"
                ),
                1,
            );
            assert!(
                error_message(&static_context)
                    .contains(&format!("extra_headers cannot set \"{header}\""))
            );
        }
    }

    const ADVISOR_CONFIG: &str = r#"
schema_version = 1

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.executor]
id = "executor/model"
llm_client = "anthropic"

[targets.advisor]
id = "advisor/model"
llm_client = "anthropic"

[routes.gated]
id = "switchyard/advisor"
type = "advisor"
executor_target = "executor"
advisor_target = "advisor"
"#;

    #[test]
    fn advisor_route_parses_with_defaults_and_builds() -> RunnerResult<()> {
        let state = runner_from_toml(ADVISOR_CONFIG)?;
        let route = state
            .route("switchyard/advisor")
            .expect("advisor route should exist");
        let models = route.models();
        // The gate calls the executor through `efficient`; `any` keeps it in the
        // route's last-resort pool.
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("executor/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("executor/model")]
        );
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("advisor/model")]
        );
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/advisor"]
        );
        Ok(())
    }

    #[test]
    fn advisor_route_accepts_every_gate_knob() -> RunnerResult<()> {
        let tuned = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            concat!(
                "advisor_target = \"advisor\"\n",
                "reviewer_system_prompt = \"review it\"\n",
                "redo_feedback_prefix = \"REVIEWER SAYS: \"\n",
                "gate_trigger = \"pattern\"\n",
                "gate_trigger_pattern = 'task_complete[\"\\s>:]*true'\n",
                "max_reviews = 2\n",
                "gate_stall_turns = 40\n",
                "gate_min_tool_results = 1\n",
                "advisor_max_tokens = 1024\n",
                "advisor_temperature = 0.0\n",
                "transcript_max_chars = 100000\n",
                "fail_open = false\n",
                "context_window = 200000\n",
                "tool_calling = true\n",
                "reasoning = true",
            ),
        );
        runner_from_toml(&tuned)?;
        Ok(())
    }

    #[test]
    fn advisor_route_rejects_unknown_keys() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nbogus_field = 1",
        );
        assert!(error_message(&invalid).contains("bogus_field"));
    }

    #[test]
    fn advisor_route_requires_both_targets() {
        let missing = ADVISOR_CONFIG.replace("advisor_target = \"advisor\"\n", "");
        assert!(error_message(&missing).contains("advisor_target"));
    }

    #[test]
    fn advisor_route_rejects_unknown_target() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"missing\"",
        );
        assert!(error_message(&invalid).contains("missing"));
    }

    #[test]
    fn advisor_route_rejects_invalid_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"\ngate_trigger_pattern = \"(unclosed\"",
        );
        assert!(error_message(&invalid).contains("not a valid regex"));
    }

    #[test]
    fn advisor_route_rejects_pattern_without_pattern_trigger() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger_pattern = \"done\"",
        );
        assert!(
            error_message(&invalid)
                .contains("gate_trigger_pattern requires gate_trigger = \"pattern\"")
        );
    }

    #[test]
    fn advisor_route_pattern_trigger_requires_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"",
        );
        assert!(error_message(&invalid).contains("non-empty gate_trigger_pattern"));
    }

    #[test]
    fn advisor_route_rejects_zero_max_reviews() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nmax_reviews = 0",
        );
        assert!(error_message(&invalid).contains("max_reviews must be at least 1"));
    }

    // ---- Route-authoritative reasoning policy (Gate 1 P-battery) ------------

    const POLICY_TOML_NONE: &str = "reasoning_policy = \"none\"";

    fn policy_config(route_block: &str, extra: &str) -> String {
        // A minimal fleet-free route set: one passthrough route on the primary
        // (chat) client plus one on the responses client, each with its own
        // target and a declared reasoning dialect.
        format!(
            r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"
reasoning_dialect = "llama_cpp_enable_thinking"

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"
reasoning_dialect = "openai_effort"

[targets.neutral_chat]
id = "neutral-chat/model"
llm_client = "primary"

[targets.neutral_responses]
id = "neutral-responses/model"
llm_client = "responses"

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "neutral_chat"
{route_block}

[routes.passthrough_responses]
id = "switchyard/passthrough-responses"
type = "passthrough"
target = "neutral_responses"
{extra}
"#
        )
    }

    #[test]
    fn route_policy_none_with_dialect_parses_and_builds() -> RunnerResult<()> {
        let configured = policy_config(POLICY_TOML_NONE, "");
        let runner = runner_from_toml(&configured)?;
        // Build succeeded: route policy accepted with a declared dialect.
        assert!(runner.route("switchyard/passthrough").is_some());
        assert_eq!(
            runner
                .route("switchyard/passthrough")
                .and_then(|route| route.reasoning_policy()),
            Some(switchyard_protocol::ReasoningPolicy::None)
        );
        Ok(())
    }

    #[test]
    fn route_policy_requires_dialect_on_every_reachable_client() {
        // Strip the chat client's dialect, then give its route a policy —
        // the policy-bearing route reaches a dialect-less client and must fail.
        let configured = policy_config(POLICY_TOML_NONE, "")
            .replace("reasoning_dialect = \"llama_cpp_enable_thinking\"\n", "");
        let actual = error_message(&configured);
        assert!(
            actual.contains("declares no reasoning_dialect"),
            "unexpected error: {actual}"
        );
    }

    #[test]
    fn route_policy_rejects_target_hard_reasoning_pin() {
        // A target carrying a reasoning pin + a route policy = contradictory
        // authoritative sources -> configuration error.
        let configured = policy_config(POLICY_TOML_NONE, "").replace(
            "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"",
            "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"\n\
             reasoning_effort = \"medium\"",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("must not fight"),
            "unexpected error: {actual}"
        );
    }

    #[test]
    fn route_policy_rejects_target_extra_body_reasoning_pin() {
        for pin in [
            "extra_body = { reasoning = { effort = \"none\" } }",
            "extra_body = { reasoning_effort = \"low\" }",
            "extra_body = { chat_template_kwargs = { enable_thinking = true } }",
        ] {
            let configured = policy_config(POLICY_TOML_NONE, "").replace(
                "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"",
                &format!(
                    "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"\n{pin}"
                ),
            );
            let actual = error_message(&configured);
            assert!(
                actual.contains("must not fight"),
                "unexpected error for pin {pin}: {actual}"
            );
        }
    }

    #[test]
    fn route_policy_unknown_value_fails_parse() {
        let configured = policy_config("reasoning_policy = \"sometimes\"", "");
        let actual = error_message(&configured);
        assert!(
            actual.contains("unknown reasoning_policy"),
            "unexpected error: {actual}"
        );
    }

    #[test]
    fn openai_effort_policy_requires_declared_vocabulary() {
        // Review finding: effort passthrough must not assume a universal
        // five-value set. Missing list = config error...
        let configured = policy_config(POLICY_TOML_NONE, "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"openai_effort\"",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("without declaring reasoning_efforts"),
            "unexpected error: {actual}"
        );
        // ...policy outside the declared list (no `none` in it) = config error;
        // the vocabulary is quoted so the operator can widen it deliberately...
        let configured = policy_config(POLICY_TOML_NONE, "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"openai_effort\"\nreasoning_efforts = [\"low\", \"medium\"]",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("cannot faithfully express it")
                && actual.contains("[\"low\", \"medium\"]"),
            "unexpected error: {actual}"
        );
        // ...and a policy inside the declared list builds.
        let configured = policy_config(POLICY_TOML_NONE, "")
            .replace(
                "reasoning_dialect = \"llama_cpp_enable_thinking\"",
                "reasoning_dialect = \"openai_effort\"\nreasoning_efforts = [\"none\", \"low\", \"high\", \"max\"]",
            );
        if let Err(error) = runner_from_toml(&configured) {
            panic!("policy inside the declared vocabulary must build: {error}");
        }
    }

    #[test]
    fn s03_row10_11_openai_effort_vocabulary_gating() {
        // openai_effort accepts ONLY values listed in reasoning_efforts...
        let configured = policy_config(POLICY_TOML_NONE, "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"openai_effort\"\nreasoning_efforts = [\"none\", \"low\", \"high\", \"max\"]",
        );
        runner_from_toml(&configured).expect("declared policy must build");
        // ...and rejects unsupported values at load time.
        for policy in ["medium", "xhigh"] {
            let configured = policy_config(&format!("reasoning_policy = \"{policy}\""), "").replace(
                "reasoning_dialect = \"llama_cpp_enable_thinking\"",
                "reasoning_dialect = \"openai_effort\"\nreasoning_efforts = [\"none\", \"low\", \"high\", \"max\"]",
            );
            let actual = error_message(&configured);
            assert!(
                actual.contains("cannot faithfully express it") && actual.contains(policy),
                "unexpected error for {policy}: {actual}"
            );
        }
    }

    #[test]
    fn s03_row12_chat_template_dialect_requires_vocabulary() {
        // chat_template_reasoning_effort: `none` has a fixed observed NT shape
        // and needs no vocabulary (steering §1.3), but a CONCRETE EFFORT is
        // accepted only from the operator-declared vocabulary: without one it
        // fails at load time...
        let configured = policy_config("reasoning_policy = \"medium\"", "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"chat_template_reasoning_effort\"",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("without declaring reasoning_efforts"),
            "unexpected error: {actual}"
        );
        // ...and accepts only declared effort values (DeepSeek-style: no medium).
        let configured = policy_config(POLICY_TOML_NONE, "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"chat_template_reasoning_effort\"\n             reasoning_efforts = [\"none\", \"low\", \"high\", \"max\"]",
        );
        runner_from_toml(&configured).expect("declared none must build");
        let configured = policy_config("reasoning_policy = \"medium\"", "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"chat_template_reasoning_effort\"\n             reasoning_efforts = [\"none\", \"low\", \"high\", \"max\"]",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("cannot faithfully express it"),
            "unexpected error: {actual}"
        );
    }

    #[test]
    fn s03_row18_route_policy_plus_target_pin_still_rejected() {
        // The dialect rework must not have weakened the hard-pin contradiction rule.
        let configured = policy_config(POLICY_TOML_NONE, "").replace(
            "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"",
            "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"\n\
             reasoning_effort = \"medium\"",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("must not fight"),
            "unexpected error: {actual}"
        );
    }

    #[test]
    fn s03_row20_unsupported_candidate_fails_at_load_time() {
        // A policy-bearing route whose reachable target's client cannot express
        // the policy (boolean dialect + exact effort) must fail at LOAD time,
        // not at request time.
        let configured = policy_config("reasoning_policy = \"high\"", "").replace(
            "reasoning_dialect = \"llama_cpp_enable_thinking\"",
            "reasoning_dialect = \"openrouter_enabled\"",
        );
        let actual = error_message(&configured);
        assert!(
            actual.contains("cannot faithfully express it"),
            "unexpected error: {actual}"
        );
    }

    #[test]
    fn target_pins_without_route_policy_unchanged() -> RunnerResult<()> {
        // The fail-closed rule is scoped to policy-bearing routes: a plain
        // target pin with NO route policy keeps today's behavior exactly.
        let configured = policy_config("", "").replace(
            "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"",
            "[targets.neutral_chat]\nid = \"neutral-chat/model\"\nllm_client = \"primary\"\n\
             reasoning_effort = \"medium\"",
        );
        runner_from_toml(&configured)?;
        Ok(())
    }
}
