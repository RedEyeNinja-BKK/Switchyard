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
    provider_api_keys: Vec<String>,
    /// Specialized capability surfaces, keyed by the public id callers request.
    capabilities: BTreeMap<String, crate::capability::ResolvedCapability>,
    /// Configured logical decision lanes, by name. More than one when candidate order
    /// differs per question type.
    decision_lanes: BTreeMap<String, Lane>,
    /// Name of the lane that serves every question kind, when one is configured.
    default_decision_lane: Option<String>,
    /// Ordered composites over complete child routes.
    composites: BTreeMap<String, crate::composition::CompositeSpec>,
    /// Declared capabilities of composite routes, which are policy promises rather than
    /// derived from any one child.
    composite_capabilities: BTreeMap<String, ModelCapabilities>,
}

/// One configured lane: its resolver and the kinds it serves.
#[derive(Clone)]
pub struct Lane {
    /// The lane's ordered candidates.
    pub resolver: std::sync::Arc<crate::decision_executor::DecisionResolver>,
    /// The question kinds this lane serves. Empty means every kind.
    pub types: Vec<crate::decision_transport::QuestionKind>,
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
    /// Wire family this decision backend speaks, when it serves typed decisions.
    pub decision_transport: Option<crate::decision_transport::DecisionTransport>,
    /// Which question kinds this backend currently answers.
    pub supported_types: Option<crate::decision_transport::SupportedTypes>,
    /// State shapes this backend accepts. A target accepts every shape by default.
    pub state_forms: crate::decision_transport::StateForms,
    /// Path appended to `base_url` for a typed decision call.
    pub decision_path: Option<String>,
    /// Base URL for a typed decision call, when it is not the completion endpoint.
    pub decision_base_url: Option<String>,
    /// Environment variable holding this backend's credential.
    pub decision_api_key_env: Option<String>,
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
            capabilities: BTreeMap::new(),
            decision_lanes: BTreeMap::new(),
            default_decision_lane: None,
            composites: BTreeMap::new(),
            composite_capabilities: BTreeMap::new(),
        }
    }

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

    /// Returns the route registered for a model.
    pub fn route(&self, model: &str) -> Option<&Route> {
        self.routes
            .iter()
            .find(|(id, _)| id.as_str() == model)
            .map(|(_, route)| route)
    }

    /// Returns the registered routes, for advertising and capability lookups.
    pub fn models(&self) -> impl Iterator<Item = ModelInfo<'_>> {
        self.routes.iter().map(|(id, route)| ModelInfo {
            id,
            algorithm: route.algorithm_name(),
            capabilities: route.capabilities(),
        })
    }

    /// The URL unmatched requests may fall through to, when configured.
    pub fn fallback_base_url(&self) -> Option<&str> {
        self.fallback_base_url.as_deref()
    }

    /// Registers the URL unmatched requests may fall through to.
    pub(crate) fn with_fallback_url(mut self, fallback_base_url: Option<String>) -> Self {
        self.fallback_base_url = fallback_base_url;
        self
    }

    /// Registers deployment-owned API keys for serving-surface output redaction.
    pub fn with_provider_api_keys(mut self, keys: Vec<String>) -> Self {
        self.provider_api_keys = keys;
        self
    }

    /// Returns deployment-owned secrets for serving-surface output redactors.
    pub fn provider_api_keys(&self) -> &[String] {
        &self.provider_api_keys
    }

    /// Registers the specialized capability surfaces a deployment exposes.
    pub fn with_capabilities(
        mut self,
        capabilities: BTreeMap<String, crate::capability::ResolvedCapability>,
    ) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Resolves a caller-requested capability by its public id.
    pub fn capability(&self, id: &str) -> Option<&crate::capability::ResolvedCapability> {
        self.capabilities.get(id)
    }

    /// Every exposed capability, for advertising on `/v1/models`.
    pub fn capabilities(&self) -> &BTreeMap<String, crate::capability::ResolvedCapability> {
        &self.capabilities
    }

    /// Registers the default logical decision lane, used by the compatibility surface.
    pub fn with_decision_lane(
        mut self,
        name: impl Into<String>,
        resolver: std::sync::Arc<crate::decision_executor::DecisionResolver>,
        types: Vec<crate::decision_transport::QuestionKind>,
    ) -> Self {
        let name = name.into();
        if types.is_empty() {
            self.default_decision_lane = Some(name.clone());
        }
        self.decision_lanes.insert(
            name,
            Lane {
                resolver,
                types,
            },
        );
        self
    }

    /// Registers an ordered composite over complete child routes.
    pub fn with_composite(
        mut self,
        route_id: impl Into<String>,
        children: Vec<String>,
    ) -> Self {
        self.composites
            .insert(route_id.into(), crate::composition::CompositeSpec { children });
        self
    }

    /// Declares the capabilities a composite route guarantees.
    pub fn with_composite_capabilities(
        mut self,
        route_id: impl Into<String>,
        capabilities: ModelCapabilities,
    ) -> Self {
        self.composite_capabilities.insert(route_id.into(), capabilities);
        self
    }

    /// The capabilities any servable route advertises.
    ///
    /// A composite is servable and declares its own guarantee, so ingress resolves it the same
    /// way it resolves a leaf. Returns `None` only when nothing is registered under the name.
    pub fn has_route(&self, route_id: &str) -> bool {
        self.route_capabilities(route_id).is_some()
    }

    /// The capabilities a route advertises: a composite's declared promise, or a leaf's.
    pub fn route_capabilities(&self, route_id: &str) -> Option<ModelCapabilities> {
        if let Some(capabilities) = self.composite_capabilities.get(route_id) {
            return Some(*capabilities);
        }
        self.route(route_id).map(|route| route.capabilities())
    }

    /// Every route this runner serves, composite or leaf.
    pub fn route_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.routes.iter().map(|(id, _)| id.to_string()).collect();
        ids.extend(self.composites.keys().cloned());
        ids
    }

    /// The composite spec for a route, when it is one.
    pub fn composite_spec(&self, route_id: &str) -> Option<&crate::composition::CompositeSpec> {
        self.composites.get(route_id)
    }

    /// Every registered composite, for diagnostics and validation.
    pub fn composites(&self) -> &BTreeMap<String, crate::composition::CompositeSpec> {
        &self.composites
    }

    /// The configured name of the lane that serves every question kind.
    pub fn decision_lane_name(&self) -> Option<&str> {
        self.default_decision_lane.as_deref()
    }

    /// Selects the lane that serves a request, by its question kinds.
    ///
    /// A lane scoped to exactly this request's kinds wins over a broader one, so a
    /// type-specific order is never shadowed by the general lane. A request no lane
    /// covers is reported as such rather than falling through to whichever lane happens
    /// to be registered first.
    pub fn lane_for_request(
        &self,
        request: &switchyard_protocol::DecisionRequest,
    ) -> Option<&Lane> {
        // Scoped lanes are considered before the unscoped one, so a type-specific order
        // is never shadowed by the general lane. Registering lanes in a sorted map means
        // iteration order alone must not decide this.
        let mut scoped: Option<&Lane> = None;
        for lane in self.decision_lanes.values() {
            if lane.types.is_empty() {
                continue;
            }
            if !crate::decision_transport::lane_serves(&lane.types, request) {
                continue;
            }
            let best_fit = scoped.map_or(usize::MAX, |current| current.types.len());
            if lane.types.len() < best_fit {
                scoped = Some(lane);
            } else if lane.types.len() == best_fit {
                // Two lanes claim the same kinds; that is ambiguous, so neither is used.
                return None;
            }
        }
        if scoped.is_some() {
            return scoped;
        }
        self.decision_lanes
            .values()
            .find(|lane| crate::decision_transport::lane_serves(&lane.types, request))
    }
}

