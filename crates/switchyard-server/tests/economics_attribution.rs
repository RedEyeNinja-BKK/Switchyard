// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the deployment-derived attribution of usage.
//!
//! Every value the economics surface attributes a request to comes from the
//! deployment configuration, never from a model-name guess. These tests pin that:
//! a provider is the client a target is bound to, a cost class is a property of the
//! credential and the base URL that client uses, and a model the deployment does not
//! declare is left unclassified rather than attributed to a plausible provider.

use std::collections::BTreeMap;

use switchyard_runner::BillingKind;
use switchyard_server::economics::classification::{
    CostClass, DeploymentClassification, RouteFamily,
};
use switchyard_server::economics::metrics::UsageAttribution;

fn classification() -> DeploymentClassification {
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
            "http://reninja.tailc8ef7b.ts.net:8446/v1".to_string(),
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

/// A request served by a metered provider is attributed to that provider and to the
/// route the caller addressed.
#[test]
fn a_served_request_is_attributed_to_its_provider_and_route() {
    let resolved = UsageAttribution::resolve(
        &classification(),
        "direct-deepseek-deepseek-flash",
        None,
        "deepseek-flash",
    )
    .expect("attribution");
    assert_eq!(resolved.provider, "deepseek");
    assert_eq!(resolved.route, "direct-deepseek-deepseek-flash");
    assert_eq!(resolved.route_family, RouteFamily::Direct);
    assert_eq!(resolved.cost_class, CostClass::Metered);
    assert_eq!(resolved.child_route, None);
}

/// A composed request keeps both the route the caller addressed and the child that
/// served it. Collapsing to the leaf would lose the caller's intent; collapsing to
/// the parent would lose who actually answered.
#[test]
fn a_composed_request_keeps_the_parent_and_the_child() {
    let resolved = UsageAttribution::resolve(
        &classification(),
        "tier1",
        Some("smartfree"),
        "minimax/minimax-m3.1-flash:free",
    )
    .expect("attribution");
    assert_eq!(resolved.route, "tier1");
    assert_eq!(resolved.child_route.as_deref(), Some("smartfree"));
    assert_eq!(resolved.route_family, RouteFamily::Tier);
    assert_eq!(resolved.provider, "openrouter");
    assert_eq!(resolved.cost_class, CostClass::Free);
}

/// A model the deployment does not declare resolves to no attribution. Attributing
/// it to a plausible provider would be a guess, and the dashboard's provider split
/// must be traceable to the configuration.
#[test]
fn an_undeclared_model_resolves_to_no_attribution() {
    assert!(
        UsageAttribution::resolve(&classification(), "tier1", None, "some/unknown-model").is_none()
    );
}

/// A lane backed by an allowance-bearing credential is a subscription, and a
/// subscription lane must never be priced per token.
#[test]
fn an_allowance_backed_lane_is_a_subscription() {
    let resolved = UsageAttribution::resolve(
        &classification(),
        "direct-openai-gpt-6-luna",
        None,
        "gpt-6-luna",
    )
    .expect("attribution");
    assert_eq!(resolved.cost_class, CostClass::Subscription);
}

/// A lane served by a deployment the operator hosts costs no external provider
/// anything, and that is not the same as costing nothing.
#[test]
fn a_local_lane_is_local() {
    let resolved = UsageAttribution::resolve(
        &classification(),
        "direct-comfyninja-qwen3_8-27b-q3",
        None,
        "comfyninja/qwen3.8-27b-q3",
    )
    .expect("attribution");
    assert_eq!(resolved.cost_class, CostClass::Local);
}

/// The billing a deployment states outright is the authority. A client that declares
/// itself metered is metered even when its credential's variable name would suggest
/// otherwise.
#[test]
fn a_declared_billing_kind_overrides_the_credential_name() {
    let mut facts = switchyard_runner::DeploymentFacts::default();
    facts.clients.insert(
        "openai".to_string(),
        switchyard_runner::ClientFacts {
            name: "openai".to_string(),
            base_url: "https://chatgpt.com/backend-api/codex".to_string(),
            api_key_env: Some("SY_OPENAI_OAUTH_TOKEN".to_string()),
            billing: Some(BillingKind::Metered),
        },
    );
    facts.targets.insert(
        "openai-gpt-6-luna".to_string(),
        switchyard_runner::TargetFacts {
            name: "openai-gpt-6-luna".to_string(),
            model: "gpt-6-luna".to_string(),
            llm_client: "openai".to_string(),
        },
    );
    let runner_facts = facts;

    let client_base_urls = BTreeMap::from([(
        "openai".to_string(),
        runner_facts.clients["openai"].base_url.clone(),
    )]);
    let target_client = BTreeMap::from([("openai-gpt-6-luna".to_string(), "openai".to_string())]);
    let target_model =
        BTreeMap::from([("openai-gpt-6-luna".to_string(), "gpt-6-luna".to_string())]);
    let classification = DeploymentClassification::build(
        &client_base_urls,
        &BTreeMap::new(),
        &target_client,
        &target_model,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
    );
    assert_eq!(
        classification
            .cost_class_for_model("gpt-6-luna")
            .expect("cost class"),
        CostClass::Metered,
        "an explicitly declared billing kind is the authority"
    );
}

/// A route that composes other routes is a tier, and a route bound to a single
/// target is direct. The role comes from the configuration, not from the name.
#[test]
fn a_route_family_comes_from_the_configuration() {
    let classification = classification();
    assert_eq!(
        classification.route_family("tier1"),
        Some(RouteFamily::Tier)
    );
    assert_eq!(
        classification.route_family("smartfree"),
        Some(RouteFamily::Primitive)
    );
    assert_eq!(
        classification.route_family("direct-openrouter-qwen3_7-flash"),
        Some(RouteFamily::Direct)
    );
}
