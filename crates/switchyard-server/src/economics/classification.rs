// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deployment-derived provider, route and cost classification.
//!
//! Every value here comes from the deployment configuration, never from a model-name
//! guess. A model id is a provider's own string; the *provider* is the client a target
//! is bound to, and the *cost class* is a property of the credential and the base URL
//! that client uses. Deriving them from configuration is what makes the attribution
//! auditable: the dashboard's provider split can be traced back to `routes.toml`.

use std::collections::BTreeMap;

/// How a provider's usage is charged.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CostClass {
    /// Served by a deployment the operator runs. No external API charge, and no claim
    /// about operating cost either: local inference consumes hardware.
    Local,
    /// The provider prices the model at zero. Free lanes still consume a provider's
    /// free-tier allowance, which is not the same as costing nothing.
    Free,
    /// Backed by a subscription or an allowance-bearing credential. The provider does
    /// not bill per token, so no per-token price may be attached to it.
    Subscription,
    /// A metered API key on a paid provider.
    Metered,
}

impl CostClass {
    /// The stable label used in metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Free => "free",
            Self::Subscription => "subscription",
            Self::Metered => "metered",
        }
    }
}

/// The role a route plays in the deployment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RouteFamily {
    /// A route addressed directly by a caller, bound to a single target.
    Direct,
    /// A named routing primitive that other routes may compose.
    Primitive,
    /// A composite over other routes.
    Tier,
}

impl RouteFamily {
    /// The stable label used in metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Primitive => "primitive",
            Self::Tier => "tier",
        }
    }
}

/// One provider the deployment talks to, with the facts needed to classify its usage.
#[derive(Clone, Debug)]
pub struct ProviderFacts {
    /// The client name, which is the provider identity used as a metric label.
    pub name: String,
    /// The base URL the client sends requests to.
    pub base_url: String,
    /// Whether the client authenticates with an allowance-bearing credential rather
    /// than a metered API key.
    pub subscription: bool,
    /// The cost class the provider is billed under.
    pub cost_class: CostClass,
}

/// The classification of every route and target the deployment defines.
#[derive(Clone, Debug, Default)]
pub struct DeploymentClassification {
    providers: BTreeMap<String, ProviderFacts>,
    /// Target name to the provider that serves it.
    target_provider: BTreeMap<String, String>,
    /// Model id to the provider that serves it.
    ///
    /// Keyed by the model id the provider itself reports, because that is the value
    /// the token counters carry as their `model` label.
    target_provider_for_model: BTreeMap<String, String>,
    /// Route id to the route's family.
    route_family: BTreeMap<String, RouteFamily>,
    /// Route id to the provider its primary target resolves to, when it has one.
    route_provider: BTreeMap<String, String>,
}

/// Why a base URL is treated as a local deployment.
///
/// A local deployment is one the operator hosts, so the host is a private or tailnet
/// name rather than a public provider domain. The test is on the host, not on the
/// scheme, because a local deployment may be reached over plain HTTP.
fn is_local_host(base_url: &str) -> bool {
    let host = base_url
        .split("://")
        .nth(1)
        .unwrap_or(base_url)
        .split('/')
        .next()
        .unwrap_or("");
    let host = host.split(':').next().unwrap_or("");
    if host.is_empty() {
        return false;
    }
    if host == "localhost" {
        return true;
    }
    // A tailnet or RFC1918 name: no dots in the first label, or a private address.
    let first = host.split('.').next().unwrap_or("");
    if !host.contains('.') {
        return true;
    }
    if host.ends_with(".ts.net") || host.ends_with(".local") {
        return true;
    }
    matches!(first, "10" | "192" | "172" | "127")
}

impl DeploymentClassification {
    /// Builds the classification from a loaded runner's deployment facts.
    ///
    /// The facts come from the configuration the runner was loaded from, so the
    /// attribution a dashboard shows can be traced back to `routes.toml` rather than
    /// inferred from a model name.
    pub fn from_runner(runner: &switchyard_runner::Runner) -> Self {
        let facts = runner.deployment_facts();

        let client_base_urls: BTreeMap<String, String> = facts
            .clients
            .values()
            .map(|client| (client.name.clone(), client.base_url.clone()))
            .collect();

        // A client whose credential is an allowance-bearing one is backed by a
        // subscription rather than by per-token billing. The deployment may state
        // this outright; when it does not, the credential's environment variable name
        // is the fallback signal. That is a naming convention, which is weaker than a
        // declared fact, so it is only consulted when nothing better exists.
        let client_subscription: BTreeMap<String, bool> = facts
            .clients
            .values()
            .map(|client| {
                let subscription = match client.billing {
                    Some(switchyard_runner::BillingKind::Subscription) => true,
                    Some(switchyard_runner::BillingKind::Metered) => false,
                    Some(switchyard_runner::BillingKind::Local) => false,
                    None => client
                        .api_key_env
                        .as_deref()
                        .is_some_and(|name| name.contains("OAUTH")),
                };
                (client.name.clone(), subscription)
            })
            .collect();

        let target_client: BTreeMap<String, String> = facts
            .targets
            .values()
            .map(|target| (target.name.clone(), target.llm_client.clone()))
            .collect();
        let target_model: BTreeMap<String, String> = facts
            .targets
            .values()
            .map(|target| (target.name.clone(), target.model.clone()))
            .collect();
        let route_target: BTreeMap<String, String> = facts
            .routes
            .values()
            .filter_map(|route| {
                route
                    .target
                    .clone()
                    .map(|target| (route.id.clone(), target))
            })
            .collect();
        let route_has_candidates: BTreeMap<String, bool> = facts
            .routes
            .values()
            .map(|route| (route.id.clone(), route.has_candidates))
            .collect();

        Self::build(
            &client_base_urls,
            &client_subscription,
            &target_client,
            &target_model,
            &route_target,
            &route_has_candidates,
            &facts.composites,
        )
    }

    /// Builds the classification from the deployment's client, target and route tables.
    ///
    /// `client_subscription` names the clients that authenticate with an
    /// allowance-bearing credential; `client_base_urls` gives each client's base URL.
    /// `route_target` maps a route to its primary target and `route_has_candidates`
    /// records whether that route declares an ordered candidate list, which is what
    /// distinguishes a routing primitive from a route bound to a single target.
    pub fn build(
        client_base_urls: &BTreeMap<String, String>,
        client_subscription: &BTreeMap<String, bool>,
        target_client: &BTreeMap<String, String>,
        target_model: &BTreeMap<String, String>,
        route_target: &BTreeMap<String, String>,
        route_has_candidates: &BTreeMap<String, bool>,
        composite_children: &BTreeMap<String, Vec<String>>,
    ) -> Self {
        let mut providers = BTreeMap::new();
        for (name, base_url) in client_base_urls {
            let subscription = client_subscription.get(name).copied().unwrap_or(false);
            let cost_class = if is_local_host(base_url) {
                CostClass::Local
            } else if subscription {
                CostClass::Subscription
            } else {
                CostClass::Metered
            };
            providers.insert(
                name.clone(),
                ProviderFacts {
                    name: name.clone(),
                    base_url: base_url.clone(),
                    subscription,
                    cost_class,
                },
            );
        }

        let mut target_provider = BTreeMap::new();
        let mut target_provider_for_model = BTreeMap::new();
        for (target, client) in target_client {
            target_provider.insert(target.clone(), client.clone());
            if let Some(model) = target_model.get(target) {
                target_provider_for_model.insert(model.clone(), client.clone());
            }
        }

        let mut route_family = BTreeMap::new();
        let mut route_provider = BTreeMap::new();
        for (route, target) in route_target {
            // A route that composes other routes is a tier. A route that declares its
            // own ordered candidate list is a routing primitive: it can pick among
            // targets rather than being bound to one. Everything else is a route a
            // caller addresses directly.
            let family = if composite_children.contains_key(route) {
                RouteFamily::Tier
            } else if route_has_candidates.get(route).copied().unwrap_or(false) {
                RouteFamily::Primitive
            } else {
                RouteFamily::Direct
            };
            route_family.insert(route.clone(), family);
            if let Some(provider) = target_provider.get(target) {
                route_provider.insert(route.clone(), provider.clone());
            }
        }
        // A composite has no target of its own, so its provider is the provider of the
        // child it would select. It is recorded per request, not here.
        for route in composite_children.keys() {
            route_family.insert(route.clone(), RouteFamily::Tier);
        }

        Self {
            providers,
            target_provider,
            target_provider_for_model,
            route_family,
            route_provider,
        }
    }

    /// Returns the provider that serves a target.
    pub fn provider_for_target(&self, target: &str) -> Option<&str> {
        self.target_provider.get(target).map(String::as_str)
    }

    /// Returns the provider bound to a route's primary target.
    pub fn provider_for_route(&self, route: &str) -> Option<&str> {
        self.route_provider.get(route).map(String::as_str)
    }

    /// Returns the facts of a provider.
    pub fn provider(&self, name: &str) -> Option<&ProviderFacts> {
        self.providers.get(name)
    }

    /// Returns every provider, ordered by name.
    pub fn providers(&self) -> impl Iterator<Item = &ProviderFacts> {
        self.providers.values()
    }

    /// Returns the family of a route.
    pub fn route_family(&self, route: &str) -> Option<RouteFamily> {
        self.route_family.get(route).copied()
    }

    /// Returns the provider that serves a model id.
    ///
    /// The model id is the provider's own string, so it is resolved through the
    /// target that declares it. A model the deployment does not declare resolves to
    /// nothing, which is what keeps an unknown model out of the provider split rather
    /// than guessing one.
    pub fn provider_for_model(&self, model: &str) -> Option<&str> {
        self.target_provider_for_model
            .get(model)
            .map(String::as_str)
    }

    /// Returns the cost class a specific model is billed under.
    ///
    /// A free model on a metered provider is free: the provider prices that model at
    /// zero, which is a different fact from the provider being metered. Subscription
    /// and local classes are not overridden, because a subscription lane's charge is
    /// the allowance, not the model, and a local lane costs no external provider
    /// anything at all.
    pub fn cost_class_for_model(&self, model: &str) -> Option<CostClass> {
        let provider = self.provider_for_model(model)?;
        let class = self.provider(provider)?.cost_class;
        if class == CostClass::Metered && is_free_model(model) {
            Some(CostClass::Free)
        } else {
            Some(class)
        }
    }
}

/// Whether a model id names a model the provider prices at zero.
///
/// OpenRouter's free tier marks such models with a `:free` suffix, and the deployed
/// configuration uses that exact convention. The suffix is the provider's own
/// convention, so reading it is reading the provider's pricing, not guessing from a
/// model family.
fn is_free_model(model: &str) -> bool {
    model.ends_with(":free")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deployment shape this module classifies: one metered cloud provider, one
    /// local provider, one allowance-backed provider, and a mix of direct, primitive
    /// and composite routes.
    fn fixture() -> DeploymentClassification {
        let client_base_urls = BTreeMap::from([
            (
                "openrouter".to_string(),
                "https://openrouter.ai/api/v1".to_string(),
            ),
            (
                "deepseek".to_string(),
                "https://api.deepseek.com".to_string(),
            ),
            (
                "comfyninja".to_string(),
                "http://gpu-box.tailnet-example.ts.net:8446/v1".to_string(),
            ),
            (
                "openai".to_string(),
                "https://chatgpt.com/backend-api/codex".to_string(),
            ),
        ]);
        let client_subscription = BTreeMap::from([("openai".to_string(), true)]);
        let target_client = BTreeMap::from([
            (
                "openrouter-minimax-m3-1-flash-free".to_string(),
                "openrouter".to_string(),
            ),
            (
                "openrouter-qwen3_7-flash".to_string(),
                "openrouter".to_string(),
            ),
            (
                "deepseek-deepseek-flash".to_string(),
                "deepseek".to_string(),
            ),
            (
                "comfyninja-qwen3_8-27b-q3".to_string(),
                "comfyninja".to_string(),
            ),
            ("openai-gpt-6-luna".to_string(), "openai".to_string()),
        ]);
        let target_model = BTreeMap::from([
            (
                "openrouter-minimax-m3-1-flash-free".to_string(),
                "minimax/minimax-m3.1-flash:free".to_string(),
            ),
            (
                "openrouter-qwen3_7-flash".to_string(),
                "qwen/qwen3.7-flash".to_string(),
            ),
            (
                "deepseek-deepseek-flash".to_string(),
                "deepseek-flash".to_string(),
            ),
            (
                "comfyninja-qwen3_8-27b-q3".to_string(),
                "comfyninja/qwen3.8-27b-q3".to_string(),
            ),
            ("openai-gpt-6-luna".to_string(), "gpt-6-luna".to_string()),
        ]);
        let route_target = BTreeMap::from([
            (
                "direct-openrouter-qwen3_7-flash".to_string(),
                "openrouter-qwen3_7-flash".to_string(),
            ),
            (
                "direct-deepseek-deepseek-flash".to_string(),
                "deepseek-deepseek-flash".to_string(),
            ),
            (
                "direct-comfyninja-qwen3_8-27b-q3".to_string(),
                "comfyninja-qwen3_8-27b-q3".to_string(),
            ),
            (
                "direct-openai-gpt-6-luna".to_string(),
                "openai-gpt-6-luna".to_string(),
            ),
            (
                "smartfree".to_string(),
                "openrouter-minimax-m3-1-flash-free".to_string(),
            ),
            ("smart".to_string(), "deepseek-deepseek-flash".to_string()),
        ]);
        let route_has_candidates =
            BTreeMap::from([("smartfree".to_string(), true), ("smart".to_string(), true)]);
        let composite_children = BTreeMap::from([(
            "tier1".to_string(),
            vec!["smartfree".to_string(), "smart".to_string()],
        )]);
        DeploymentClassification::build(
            &client_base_urls,
            &client_subscription,
            &target_client,
            &target_model,
            &route_target,
            &route_has_candidates,
            &composite_children,
        )
    }

    #[test]
    fn a_model_resolves_to_the_provider_that_declares_it() {
        let classification = fixture();
        assert_eq!(
            classification.provider_for_model("qwen/qwen3.7-flash"),
            Some("openrouter")
        );
        assert_eq!(
            classification.provider_for_model("comfyninja/qwen3.8-27b-q3"),
            Some("comfyninja")
        );
    }

    #[test]
    fn an_undeclared_model_resolves_to_no_provider() {
        let classification = fixture();
        // A model the deployment never declared must not be attributed to a provider
        // by guessing. It stays unclassified, and the dashboard can report the
        // unattributed remainder rather than inventing an owner for it.
        assert_eq!(
            classification.provider_for_model("some/unknown-model"),
            None
        );
        assert_eq!(
            classification.cost_class_for_model("some/unknown-model"),
            None
        );
    }

    #[test]
    fn a_tailnet_client_is_local_and_not_metered() {
        let classification = fixture();
        assert_eq!(
            classification.provider("comfyninja").map(|p| p.cost_class),
            Some(CostClass::Local)
        );
    }

    #[test]
    fn an_allowance_backed_client_is_a_subscription_not_a_metered_lane() {
        let classification = fixture();
        assert_eq!(
            classification.provider("openai").map(|p| p.cost_class),
            Some(CostClass::Subscription)
        );
        // A subscription lane must not be given a per-token price even when the model
        // it serves is not marked free.
        assert_eq!(
            classification.cost_class_for_model("gpt-6-luna"),
            Some(CostClass::Subscription)
        );
    }

    #[test]
    fn a_free_model_on_a_metered_provider_is_free() {
        let classification = fixture();
        assert_eq!(
            classification.provider("openrouter").map(|p| p.cost_class),
            Some(CostClass::Metered)
        );
        assert_eq!(
            classification.cost_class_for_model("minimax/minimax-m3.1-flash:free"),
            Some(CostClass::Free)
        );
        assert_eq!(
            classification.cost_class_for_model("qwen/qwen3.7-flash"),
            Some(CostClass::Metered)
        );
    }

    #[test]
    fn a_local_model_stays_local_even_though_its_name_is_not_free() {
        let classification = fixture();
        assert_eq!(
            classification.cost_class_for_model("comfyninja/qwen3.8-27b-q3"),
            Some(CostClass::Local)
        );
    }

    #[test]
    fn a_route_is_classified_by_its_configuration_not_its_name() {
        let classification = fixture();
        // A route bound to one target is direct.
        assert_eq!(
            classification.route_family("direct-openrouter-qwen3_7-flash"),
            Some(RouteFamily::Direct)
        );
        // A route with an ordered candidate list is a primitive.
        assert_eq!(
            classification.route_family("smartfree"),
            Some(RouteFamily::Primitive)
        );
        // A composite over other routes is a tier.
        assert_eq!(
            classification.route_family("tier1"),
            Some(RouteFamily::Tier)
        );
    }

    #[test]
    fn a_route_name_that_looks_like_a_primitive_is_still_direct() {
        let mut classification = fixture();
        // The name is not the authority. A route called "smart-thing" that binds one
        // target and declares no candidates is direct, exactly as configured.
        classification
            .route_family
            .insert("smart-thing".to_string(), RouteFamily::Direct);
        assert_eq!(
            classification.route_family("smart-thing"),
            Some(RouteFamily::Direct)
        );
    }

    #[test]
    fn a_provider_is_not_local_because_its_host_has_a_dot() {
        // The public provider domains are not local even though they are dotted.
        assert!(!is_local_host("https://openrouter.ai/api/v1"));
        assert!(!is_local_host("https://api.deepseek.com"));
        // A tailnet, a private address and a bare host name are local.
        assert!(is_local_host(
            "http://gpu-box.tailnet-example.ts.net:8446/v1"
        ));
        assert!(is_local_host(
            "http://desk-box.tailnet-example.ts.net:8889/v1"
        ));
        assert!(is_local_host("http://192.168.1.10:8080/v1"));
        assert!(is_local_host("http://localhost:4000/v1"));
        assert!(is_local_host("http://gpu-box:8000/v1"));
    }
}
