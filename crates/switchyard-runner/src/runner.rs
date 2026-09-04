// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named route table and server-facing route metadata.

use std::collections::BTreeMap;
use std::path::Path;

use libsy::RoutingOutcome;
use serde_json::Value;
use switchyard_protocol::{ModelId, WireFormat};

use crate::config;
use crate::{ModelCapabilities, Route, RunnerError};

/// Immutable named route table.
pub struct Runner {
    routes: Vec<(ModelId, Route)>,
    fallback_base_url: Option<String>,
    /// Deployment-wide fleet-state handle shared by every `fleet_router`
    /// route; the host-owned readiness monitor replaces its snapshots. `None`
    /// when the deployment configures no `fleet_router` route.
    fleet_state: Option<std::sync::Arc<libsy::SharedFleetState>>,
    /// The parsed `[fleet_readiness]` monitor configuration, when declared.
    fleet_readiness: Option<crate::config::FleetReadinessConfig>,
    /// Parsed `[capability_clients.*]` executor declarations.
    capability_clients: BTreeMap<String, crate::capability::CapabilityClientConfig>,
    /// Parsed `[capabilities.*]` route declarations.
    capabilities: BTreeMap<String, crate::capability::CapabilityRouteConfig>,
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
            fleet_state: None,
            fleet_readiness: None,
            capability_clients: BTreeMap::new(),
            capabilities: BTreeMap::new(),
        }
    }

    /// Declares the deployment-wide fleet-state handle for the host's
    /// readiness monitor.
    pub fn with_fleet_state(
        mut self,
        fleet_state: Option<std::sync::Arc<libsy::SharedFleetState>>,
    ) -> Self {
        self.fleet_state = fleet_state;
        self
    }

    /// The deployment-wide fleet-state handle, when any `fleet_router` route is
    /// configured. The host monitor owns snapshot replacement through it.
    pub fn fleet_state(&self) -> Option<&std::sync::Arc<libsy::SharedFleetState>> {
        self.fleet_state.as_ref()
    }

    /// Declares the parsed `[fleet_readiness]` monitor configuration.
    pub fn with_fleet_readiness(
        mut self,
        fleet_readiness: Option<crate::config::FleetReadinessConfig>,
    ) -> Self {
        self.fleet_readiness = fleet_readiness;
        self
    }

    /// The parsed `[fleet_readiness]` monitor configuration, when declared.
    pub fn fleet_readiness(&self) -> Option<&crate::config::FleetReadinessConfig> {
        self.fleet_readiness.as_ref()
    }

    /// Declares the parsed capability executor + route tables.
    pub fn with_capabilities(
        mut self,
        capability_clients: BTreeMap<String, crate::capability::CapabilityClientConfig>,
        capabilities: BTreeMap<String, crate::capability::CapabilityRouteConfig>,
    ) -> Self {
        self.capability_clients = capability_clients;
        self.capabilities = capabilities;
        self
    }

    /// The parsed `[capability_clients.*]` executor declarations.
    pub fn capability_clients(
        &self,
    ) -> &BTreeMap<String, crate::capability::CapabilityClientConfig> {
        &self.capability_clients
    }

    /// The parsed `[capabilities.*]` route declarations.
    pub fn capabilities(&self) -> &BTreeMap<String, crate::capability::CapabilityRouteConfig> {
        &self.capabilities
    }

    pub(crate) fn with_fallback_url(mut self, fallback_base_url: Option<String>) -> Self {
        self.fallback_base_url = fallback_base_url;
        self
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
        let resolve = |selected: &ModelId| route.decision_target(selected);
        let mut model_ids = outcome.selected_model_ids.iter();
        Some(DecisionDescription {
            selected: resolve(model_ids.next()?)?,
            fallbacks: model_ids.map(resolve).collect::<Option<Vec<_>>>()?,
        })
    }
}
