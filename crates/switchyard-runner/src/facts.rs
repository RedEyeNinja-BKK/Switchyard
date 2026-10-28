// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The deployment facts an economics surface needs, extracted from configuration.
//!
//! These are the facts that cannot be inferred from a model name: which provider
//! serves a target, what a provider's base URL is, whether its credential is a
//! metered API key or an allowance-bearing one, and which routes are composites.
//! The loader produces them once, and the server's economics layer consumes them.

use std::collections::BTreeMap;

/// The configuration facts of one client.
#[derive(Clone, Debug)]
pub struct ClientFacts {
    /// The client's name, which is the provider identity.
    pub name: String,
    /// The base URL the client sends requests to.
    pub base_url: String,
    /// The environment variable holding the client's credential, when it has one.
    pub api_key_env: Option<String>,
    /// How the client's usage is billed, when the deployment states it.
    pub billing: Option<crate::config::BillingKind>,
}

/// The configuration facts of one target.
#[derive(Clone, Debug)]
pub struct TargetFacts {
    /// The target's name.
    pub name: String,
    /// The model id the provider reports for this target.
    pub model: String,
    /// The client that serves this target.
    pub llm_client: String,
}

/// The configuration facts of one route.
#[derive(Clone, Debug)]
pub struct RouteFacts {
    /// The route's id.
    pub id: String,
    /// The route's primary target, when it has one.
    pub target: Option<String>,
    /// Whether the route declares an ordered candidate list.
    pub has_candidates: bool,
}

/// Every fact the economics layer needs about a deployment.
#[derive(Clone, Debug, Default)]
pub struct DeploymentFacts {
    /// The clients the deployment defines, by name.
    pub clients: BTreeMap<String, ClientFacts>,
    /// The targets the deployment defines, by name.
    pub targets: BTreeMap<String, TargetFacts>,
    /// The routes the deployment defines, by id.
    pub routes: BTreeMap<String, RouteFacts>,
    /// The composites the deployment defines, by route id, with their children.
    pub composites: BTreeMap<String, Vec<String>>,
}

impl DeploymentFacts {
    /// Returns the client that serves a target.
    pub fn client_for_target(&self, target: &str) -> Option<&ClientFacts> {
        let target = self.targets.get(target)?;
        self.clients.get(&target.llm_client)
    }

    /// Returns the model id a target declares.
    pub fn model_for_target(&self, target: &str) -> Option<&str> {
        self.targets.get(target).map(|facts| facts.model.as_str())
    }
}
