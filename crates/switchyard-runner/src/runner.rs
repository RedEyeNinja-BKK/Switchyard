// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named route table and server-facing route metadata.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use libsy::{FleetSnapshot, RoutingOutcome, SharedFleetState};
use serde_json::Value;
use switchyard_protocol::{ModelId, WireFormat};

use crate::capability::{CapabilityClientConfig, CapabilityKind, CapabilityRouteConfig};
use crate::config;
use crate::facts_config::FleetReadinessConfig;
use crate::{ModelCapabilities, Route, RunnerError};

/// Immutable named route table.
pub struct Runner {
    routes: Vec<(ModelId, Route)>,
    fallback_base_url: Option<String>,
    provider_api_keys: Vec<String>,
    capability_clients: BTreeMap<String, CapabilityClientConfig>,
    capabilities: BTreeMap<String, CapabilityRouteConfig>,
    fleet_readiness: Option<FleetReadinessConfig>,
    /// The deployment-wide fleet readiness handle, when any fleet route exists.
    ///
    /// This is the seam a host-owned readiness monitor publishes through: it
    /// replaces the whole snapshot atomically, and every fleet route reads
    /// exactly one immutable generation per decision.
    fleet_state: Option<Arc<SharedFleetState>>,
}

/// Borrowed model metadata returned while listing routes.
pub struct ModelInfo<'a> {
    pub id: &'a ModelId,
    pub algorithm: &'a str,
    pub capabilities: ModelCapabilities,
}

/// Fully resolved routing decision.
pub struct DecisionDescription {
    pub selected: DecisionTarget,
    pub fallbacks: Vec<DecisionTarget>,
}

/// Non-secret configured target details returned by the decision endpoint.
#[derive(Clone)]
pub struct DecisionTarget {
    pub target: String,
    pub model: ModelId,
    pub format: WireFormat,
    pub base_url: String,
    pub extra_body: BTreeMap<String, Value>,
}

impl Runner {
    /// Loads and validates a version-1 deployment TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RunnerError> {
        config::load_runner(path)
    }

    /// Loads and validates a version-1 deployment TOML document.
    ///
    /// Use [`Self::load`] when the deployment is stored in a file.
    pub fn from_toml(source: &str) -> Result<Self, RunnerError> {
        config::runner_from_toml(source)
    }

    /// Builds a runner from named routes in caller-provided order.
    /// Pre-condition: There must be at least one route.
    pub fn new(routes: Vec<(ModelId, Route)>) -> Self {
        Self {
            routes,
            fallback_base_url: None,
            provider_api_keys: Vec::new(),
            capability_clients: BTreeMap::new(),
            capabilities: BTreeMap::new(),
            fleet_readiness: None,
            fleet_state: None,
        }
    }

    /// Registers deployment-owned API keys for serving-surface output redaction.
    /// TOML loading registers these automatically; programmatic hosts must supply them.
    pub fn with_provider_api_keys(mut self, keys: Vec<String>) -> Self {
        self.provider_api_keys = keys;
        self
    }

    /// Returns deployment-owned secrets for serving-surface output redactors.
    pub fn provider_api_keys(&self) -> &[String] {
        &self.provider_api_keys
    }

    pub(crate) fn with_fallback_url(mut self, fallback_base_url: Option<String>) -> Self {
        self.fallback_base_url = fallback_base_url;
        self
    }

    /// Registers the parsed fleet-readiness monitor declaration.
    ///
    /// This is the *declaration* only. A host that constructs the monitor
    /// publishes readiness observations through its own producer; until then
    /// every governed candidate stays fail-closed.
    pub fn with_fleet_readiness(mut self, fleet_readiness: Option<FleetReadinessConfig>) -> Self {
        self.fleet_readiness = fleet_readiness;
        self
    }

    /// The parsed `[fleet_readiness]` monitor declaration, when present.
    pub fn fleet_readiness(&self) -> Option<&FleetReadinessConfig> {
        self.fleet_readiness.as_ref()
    }

    /// Attaches the deployment-wide fleet readiness handle.
    pub fn with_fleet_state(mut self, fleet_state: Option<Arc<SharedFleetState>>) -> Self {
        self.fleet_state = fleet_state;
        self
    }

    /// The deployment-wide fleet readiness handle, when any fleet route exists.
    ///
    /// A host readiness monitor calls [`SharedFleetState::set`] on this to
    /// publish a new generation. A deployment with no `fleet_router` route has
    /// no handle, and therefore nothing that would read the result.
    pub fn fleet_state(&self) -> Option<&Arc<SharedFleetState>> {
        self.fleet_state.as_ref()
    }

    /// Replaces the deployment-wide fleet readiness snapshot.
    ///
    /// Refuses a deployment with no fleet route rather than inventing a
    /// receiver, because nothing would read the result.
    pub fn set_fleet_snapshot(&self, snapshot: FleetSnapshot) -> Result<(), String> {
        match self.fleet_state.as_ref() {
            Some(state) => {
                state.set(snapshot);
                Ok(())
            }
            None => Err("deployment declares no fleet_router route".to_string()),
        }
    }

    /// Registers the parsed capability executor and route declarations.
    ///
    /// These describe typed non-LLM utility endpoints. They are served by the
    /// host, not by the named route table, and take no part in model routing.
    ///
    /// Decision fallback relationships are validated HERE, at admission, rather
    /// than discovered at request time: a self-reference, an unresolvable
    /// target, a target of the wrong capability kind, a contract mismatch, or a
    /// nested (fallback-of-fallback) relationship are all operator defects that
    /// must fail the load instead of degrading silently at runtime.
    pub fn with_capabilities(
        mut self,
        capability_clients: BTreeMap<String, CapabilityClientConfig>,
        capabilities: BTreeMap<String, CapabilityRouteConfig>,
    ) -> Result<Self, RunnerError> {
        validate_capability_fallbacks(&capabilities)?;
        self.capability_clients = capability_clients;
        self.capabilities = capabilities;
        Ok(self)
    }

    /// The parsed `[capability_clients.*]` executor declarations.
    pub fn capability_clients(&self) -> &BTreeMap<String, CapabilityClientConfig> {
        &self.capability_clients
    }

    /// The parsed `[capabilities.*]` route declarations.
    pub fn capabilities(&self) -> &BTreeMap<String, CapabilityRouteConfig> {
        &self.capabilities
    }

    /// Returns the route registered for a model.
    pub fn route(&self, model: &str) -> Option<&Route> {
        self.routes
            .iter()
            .find(|(id, _)| id.as_str() == model)
            .map(|(_, route)| route)
    }

    /// Iterates over configured routes in caller-provided order.
    pub fn models(&self) -> impl Iterator<Item = ModelInfo<'_>> {
        self.routes.iter().map(|(id, route)| ModelInfo {
            id,
            algorithm: route.algorithm_name(),
            capabilities: route.capabilities(),
        })
    }

    /// Returns the validated API root used for unmatched HTTP requests.
    pub fn fallback_base_url(&self) -> Option<&str> {
        self.fallback_base_url.as_deref()
    }

    /// Resolves an outcome to configured target names and non-secret client settings.
    pub fn describe_decision(
        &self,
        model: &ModelId,
        outcome: &RoutingOutcome,
    ) -> Option<DecisionDescription> {
        let route = self.route(model.as_str())?;
        let resolve = |selected: &ModelId| {
            let mut target = route.decision_target(selected)?;
            let mut url = reqwest::Url::parse(&target.base_url).ok()?;
            let query: Vec<_> = url
                .query_pairs()
                .filter(|(name, _)| !matches!(name.as_ref(), "key" | "api_key"))
                .map(|(name, value)| (name.into_owned(), value.into_owned()))
                .collect();
            if query.len() != url.query_pairs().count() {
                url.set_query(None);
                if !query.is_empty() {
                    url.query_pairs_mut().extend_pairs(query);
                }
                // Only the returned metadata changes; inference still needs its credentials.
                target.base_url = url.into();
            }
            Some(target)
        };
        let mut model_ids = outcome.selected_model_ids.iter();
        Some(DecisionDescription {
            selected: resolve(model_ids.next()?)?,
            fallbacks: model_ids.map(resolve).collect::<Option<Vec<_>>>()?,
        })
    }
}

/// Validates every declared decision-capability fallback relationship.
///
/// One hop means one hop, and each of these is a CONFIGURATION error that must
/// surface at load time rather than become a runtime degradation:
///
/// * a capability naming itself (an infinite loop if honoured),
/// * a `fallback_target` that is not a declared capability (unresolvable),
/// * a `fallback_target` that is not a decisions capability (wrong kind),
/// * a fallback whose decision contract differs from the primary's (the caller
///   would receive two different contracts under one route id),
/// * a fallback that itself declares a `fallback_target` (fallback-of-fallback).
fn validate_capability_fallbacks(
    capabilities: &BTreeMap<String, CapabilityRouteConfig>,
) -> Result<(), RunnerError> {
    for (id, route) in capabilities {
        let CapabilityKind::Decisions {
            contract,
            fallback_target: Some(fallback_id),
            ..
        } = &route.kind
        else {
            continue;
        };
        if fallback_id == id {
            return Err(RunnerError::configuration(format!(
                "decision capability {id} declares itself as its own fallback"
            )));
        }
        let Some(fallback) = capabilities.get(fallback_id) else {
            return Err(RunnerError::configuration(format!(
                "decision capability {id} names fallback {fallback_id}, which is not declared"
            )));
        };
        let CapabilityKind::Decisions {
            contract: fallback_contract,
            fallback_target: nested,
            ..
        } = &fallback.kind
        else {
            return Err(RunnerError::configuration(format!(
                "decision capability {id} names fallback {fallback_id}, which is not a decisions capability"
            )));
        };
        if nested.is_some() {
            return Err(RunnerError::configuration(format!(
                "decision capability {id} names fallback {fallback_id}, which itself declares a fallback; \
                 only one hop is supported"
            )));
        }
        if fallback_contract != contract {
            return Err(RunnerError::configuration(format!(
                "decision capability {id} serves contract {contract} but its fallback {fallback_id} serves {fallback_contract}"
            )));
        }
    }
    Ok(())
}
