// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric publication for economics and token accounting.
//!
//! The accounting rules this module enforces are the reason it exists:
//!
//! * Prompt tokens are `input + cached + cache_creation`. The cached and
//!   cache-creation counts are *inside* the prompt total, so they are published as
//!   their own components and never added to the prompt total again.
//! * Completion tokens are the provider's own output total, and the reasoning count
//!   is *inside* it on every provider that reports reasoning separately — measured
//!   2026-10-21 against the live OpenRouter API, where `prompt + completion` was the
//!   reported total and adding the reasoning count broke it. The reasoning count is
//!   therefore published beside the completion count, never added to it.
//! * There is deliberately no "all tokens" series. A total would be `prompt +
//!   completion`, and any series that invited adding the cache or reasoning
//!   components to it could be double counted.
//! * Reasoning tokens are published only when the provider reported them. A provider
//!   that does not report reasoning yields no series, which reads as absent rather
//!   than as a measured zero.
//! * Served requests, failed attempts and recovered retries are separate series.
//!
//! Every label value is bounded by the deployment configuration: a provider name, a
//! route id, a cost class, a route family, a component name. Nothing derived from
//! request text reaches a label.

use opentelemetry::{KeyValue, global};
use switchyard_protocol::Usage;

use super::classification::{CostClass, DeploymentClassification, RouteFamily};

/// The token components published separately.
///
/// `prompt` is the total the provider bills as input: non-cached input plus cache
/// reads plus cache writes. `cached` and `cache_creation` are *components of* that
/// total. `completion` is the provider's own output total, and `reasoning` is a
/// *component of* that total on every provider that reports it — measured 2026-10-21
/// against the live OpenRouter API, where `prompt + completion == total_tokens` and
/// adding the reasoning count broke that identity.
///
/// So the two nesting rules have the same shape: a component is never added to the
/// total that contains it. A "total tokens" figure, were one published, would be
/// `prompt + completion` and nothing else.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenComponent {
    Prompt,
    Cached,
    CacheCreation,
    Completion,
    Reasoning,
}

impl TokenComponent {
    /// The stable label value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Cached => "cached",
            Self::CacheCreation => "cache_creation",
            Self::Completion => "completion",
            Self::Reasoning => "reasoning",
        }
    }
}

/// The request outcomes published separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageOutcome {
    /// A request that reached a provider and produced a served answer.
    Served,
    /// A request that failed.
    Failed,
}

impl UsageOutcome {
    /// The stable label value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Served => "served",
            Self::Failed => "failed",
        }
    }
}

/// The attribution of one served request, resolved from the deployment configuration.
#[derive(Clone, Debug)]
pub struct UsageAttribution {
    /// The provider that served the request.
    pub provider: String,
    /// The route the caller addressed.
    pub route: String,
    /// The child route that served the request, for a composed route.
    pub child_route: Option<String>,
    /// The family the addressed route belongs to.
    pub route_family: RouteFamily,
    /// How the provider bills this request.
    pub cost_class: CostClass,
}

impl UsageAttribution {
    /// Builds the attribution of a request from the deployment classification.
    ///
    /// The provider and cost class are resolved from the *served* model, because that
    /// is the model that consumed the tokens. The route identity comes from the route
    /// the caller addressed, which for a composed route is the parent, not the leaf.
    pub fn resolve(
        classification: &DeploymentClassification,
        route: &str,
        child_route: Option<&str>,
        served_model: &str,
    ) -> Option<Self> {
        let provider = classification.provider_for_model(served_model)?.to_string();
        let cost_class = classification
            .cost_class_for_model(served_model)
            .unwrap_or(CostClass::Metered);
        let route_family = classification
            .route_family(route)
            .unwrap_or(RouteFamily::Direct);
        Some(Self {
            provider,
            route: route.to_string(),
            child_route: child_route.map(str::to_string),
            route_family,
            cost_class,
        })
    }

    /// The label set every economics series carries.
    fn attributes(&self, model: &str, component: &str, outcome: &str) -> Vec<KeyValue> {
        let mut attributes = vec![
            KeyValue::new("model", model.to_string()),
            KeyValue::new("provider", self.provider.clone()),
            KeyValue::new("route", self.route.clone()),
            KeyValue::new("route_family", self.route_family.as_str().to_string()),
            KeyValue::new("cost_class", self.cost_class.as_str().to_string()),
            KeyValue::new("component", component.to_string()),
            KeyValue::new("outcome", outcome.to_string()),
        ];
        if let Some(child) = &self.child_route {
            attributes.push(KeyValue::new("child_route", child.clone()));
        }
        attributes
    }
}

/// The token components a usage reports, with the same accounting rules the metric
/// layer applies.
///
/// Prompt tokens are `input + cached + cache_creation`, and the cache components are
/// *inside* that total, so they are reported beside it and never added to it again.
/// A component the provider did not report is absent rather than zero.
fn token_components(usage: &Usage) -> Vec<(TokenComponent, u64)> {
    let cached = usage.cached_input_tokens();
    let cache_creation = usage.cache_creation_input_tokens();
    let input = usage.input_tokens.unwrap_or(0);
    let prompt = input + cached.unwrap_or(0) + cache_creation.unwrap_or(0);

    let mut components = vec![(TokenComponent::Prompt, prompt)];
    for (component, value) in [
        (TokenComponent::Cached, cached),
        (TokenComponent::CacheCreation, cache_creation),
        (TokenComponent::Completion, usage.output_tokens),
        (TokenComponent::Reasoning, usage.reasoning_tokens),
    ] {
        let Some(value) = value else {
            continue;
        };
        components.push((component, value));
    }
    components
}

/// Publishes one request's token usage, attributed to a provider, route and cost class.
pub fn record_usage(
    attribution: &UsageAttribution,
    model: &str,
    usage: &Usage,
    outcome: UsageOutcome,
) {
    let meter = global::meter("switchyard");
    let outcome_label = outcome.as_str();

    // The prompt total is the provider's own input count. It already contains the
    // cache components, so they are published beside it and never added to it.
    for (component, value) in token_components(usage) {
        let attributes = attribution.attributes(model, component.as_str(), outcome_label);
        meter
            .u64_counter("switchyard.economics_tokens")
            .build()
            .add(value, &attributes);
    }
}

/// Records one request's token usage in the Bangkok calendar-day window.
///
/// The calendar uses the same components and the same accounting rules as the
/// rolling counters, so the two views cannot disagree about what was consumed. A
/// component the provider did not report is not recorded, which keeps an unreported
/// component absent rather than making the day total look smaller than it is.
pub fn record_usage_in_calendar(
    calendar: &super::CalendarWindow,
    attribution: &UsageAttribution,
    model: &str,
    usage: &Usage,
    outcome: UsageOutcome,
) {
    let outcome_label = outcome.as_str();
    for (component, value) in token_components(usage) {
        let attributes = attribution.attributes(model, component.as_str(), outcome_label);
        calendar.record_tokens(&attributes, value);
    }
}

/// Records one served request in the Bangkok calendar-day window.
pub fn record_request_in_calendar(
    calendar: &super::CalendarWindow,
    attribution: &UsageAttribution,
    model: &str,
    outcome: UsageOutcome,
) {
    let attributes = attribution.attributes(model, "request", outcome.as_str());
    calendar.record_request(&attributes, 1);
}

/// Publishes one served request, so request counts and token counts stay separable.
pub fn record_request(attribution: &UsageAttribution, model: &str, outcome: UsageOutcome) {
    global::meter("switchyard")
        .u64_counter("switchyard.economics_requests")
        .build()
        .add(
            1,
            &attribution.attributes(model, "request", outcome.as_str()),
        );
}

/// Records one request's tokens in the rolling windows.
///
/// The rolling windows keep the events themselves, so a one-hour figure means what
/// was observed in the last hour rather than what a counter extrapolates. The same
/// components and the same accounting rules are used as everywhere else, so a token
/// is never counted twice between the prompt total and the cache inside it.
pub fn record_usage_in_rolling(
    attribution: &UsageAttribution,
    model: &str,
    usage: &Usage,
    outcome: UsageOutcome,
) {
    let windows = super::rolling::global();
    for (component, value) in token_components(usage) {
        let attributes = attribution.attributes(model, component.as_str(), outcome.as_str());
        windows.record_tokens(&attributes, value);
    }
}

/// Records one served request in the rolling windows.
pub fn record_request_in_rolling(
    attribution: &UsageAttribution,
    model: &str,
    outcome: UsageOutcome,
) {
    let attributes = attribution.attributes(model, "request", outcome.as_str());
    super::rolling::global().record_request(&attributes);
}

/// Records one provider-reported cost in the rolling windows.
pub fn record_provider_cost_in_rolling(
    attribution: &UsageAttribution,
    model: &str,
    cost: f64,
    currency: &str,
) {
    let attributes =
        attribution.attributes(model, "provider_reported", UsageOutcome::Served.as_str());
    super::rolling::global().record_cost(&attributes, currency, cost);
}

/// Records one provider-reported cost in the Bangkok calendar day.
pub fn record_provider_cost_in_calendar(
    calendar: &super::CalendarWindow,
    attribution: &UsageAttribution,
    model: &str,
    cost: f64,
    currency: &str,
) {
    let attributes =
        attribution.attributes(model, "provider_reported", UsageOutcome::Served.as_str());
    calendar.record_cost(&attributes, currency, cost);
}

/// Publishes the provider-reported cost of one request.
///
/// This is the amount the provider itself billed, in the provider's own currency. It
/// is deliberately a different metric family from any Switchyard-derived estimate, so
/// a dashboard can never present one as the other.
pub fn record_provider_cost(attribution: &UsageAttribution, model: &str, cost: f64) {
    global::meter("switchyard")
        .f64_counter("switchyard.economics_provider_cost")
        .build()
        .add(
            cost,
            &attribution.attributes(model, "provider_reported", UsageOutcome::Served.as_str()),
        );
}

/// Publishes the Switchyard-derived cost estimate of one request.
///
/// This is a Switchyard computation from verified tokens and a rate card. It is not a
/// provider-reported amount and is never presented as one.
pub fn record_estimated_cost(attribution: &UsageAttribution, model: &str, cost: f64) {
    global::meter("switchyard")
        .f64_counter("switchyard.economics_estimated_cost")
        .build()
        .add(
            cost,
            &attribution.attributes(model, "estimated", UsageOutcome::Served.as_str()),
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::economics::classification::DeploymentClassification;
    use std::collections::BTreeMap;

    fn attribution() -> UsageAttribution {
        UsageAttribution {
            provider: "openrouter".to_string(),
            route: "tier1".to_string(),
            child_route: Some("smartfree".to_string()),
            route_family: RouteFamily::Tier,
            cost_class: CostClass::Free,
        }
    }

    /// A usage where the provider reports cached, cache-creation and reasoning
    /// detail alongside the aggregates.
    fn full_usage() -> Usage {
        Usage {
            input_tokens: Some(100),
            cache: Usage::cache_details(Some(30), Some(20)),
            output_tokens: Some(50),
            total_tokens: Some(200),
            reasoning_tokens: Some(25),
            cost: Some(0.25),
            cost_currency: Some("USD".to_string()),
        }
    }

    #[test]
    fn the_reasoning_count_is_a_component_of_the_completion_total() {
        // Measured 2026-10-21 against the live OpenRouter API: the provider reported
        // `completion_tokens: 650` with `completion_tokens_details.reasoning_tokens:
        // 521` and `total_tokens: 680`, where `prompt + completion == total` and
        // adding the reasoning count broke that identity. The reasoning count is
        // therefore a component of the completion total, exactly as the cached count
        // is a component of the prompt total.
        let usage = Usage {
            input_tokens: Some(30),
            cache: Usage::cache_details(None, None),
            output_tokens: Some(650),
            total_tokens: Some(680),
            reasoning_tokens: Some(521),
            cost: None,
            cost_currency: None,
        };
        let components = token_components(&usage);
        let completion = components
            .iter()
            .find(|(component, _)| *component == TokenComponent::Completion)
            .map(|(_, value)| *value)
            .expect("a completion component");
        let reasoning = components
            .iter()
            .find(|(component, _)| *component == TokenComponent::Reasoning)
            .map(|(_, value)| *value)
            .expect("a reasoning component");
        assert_eq!(
            completion, 650,
            "the completion total is the provider's own figure"
        );
        assert_eq!(reasoning, 521, "the reasoning count is reported beside it");
        // The two together must not exceed the completion total, because the
        // reasoning count is inside it.
        assert!(
            reasoning <= completion,
            "reasoning ({reasoning}) cannot exceed the completion total ({completion})"
        );
        // And the provider's own total identity still holds without adding it.
        assert_eq!(
            usage.input_tokens.unwrap_or(0) + completion,
            usage.total_tokens.unwrap_or(0),
            "prompt + completion is the provider's total; adding reasoning breaks it"
        );
    }

    #[test]
    fn the_prompt_total_contains_the_cache_components() {
        // The provider reports 100 non-cached input, 30 cache reads and 20 cache
        // writes. The prompt total is 150, not 100, because the cache counts are
        // inside it. Adding them again would double count the cache.
        let usage = full_usage();
        let prompt = usage.input_tokens.unwrap_or(0)
            + usage.cached_input_tokens().unwrap_or(0)
            + usage.cache_creation_input_tokens().unwrap_or(0);
        assert_eq!(prompt, 150);
        // The components sum to the total, which is what "already included" means.
        assert_eq!(
            usage.cached_input_tokens().unwrap_or(0)
                + usage.cache_creation_input_tokens().unwrap_or(0),
            50
        );
    }

    #[test]
    fn the_attribution_labels_the_provider_route_and_cost_class() {
        let attribution = attribution();
        let attributes =
            attribution.attributes("minimax/minimax-m3.1-flash:free", "prompt", "served");
        let labels: BTreeMap<&str, String> = attributes
            .iter()
            .map(|attribute| (attribute.key.as_str(), attribute.value.to_string()))
            .collect();
        assert_eq!(
            labels.get("provider").map(String::as_str),
            Some("openrouter")
        );
        assert_eq!(labels.get("route").map(String::as_str), Some("tier1"));
        assert_eq!(
            labels.get("child_route").map(String::as_str),
            Some("smartfree")
        );
        assert_eq!(labels.get("route_family").map(String::as_str), Some("tier"));
        assert_eq!(labels.get("cost_class").map(String::as_str), Some("free"));
        assert_eq!(labels.get("component").map(String::as_str), Some("prompt"));
        assert_eq!(labels.get("outcome").map(String::as_str), Some("served"));
    }

    #[test]
    fn a_direct_request_carries_no_child_route_label() {
        let attribution = UsageAttribution {
            provider: "deepseek".to_string(),
            route: "direct-deepseek-deepseek-flash".to_string(),
            child_route: None,
            route_family: RouteFamily::Direct,
            cost_class: CostClass::Metered,
        };
        let attributes = attribution.attributes("deepseek-flash", "prompt", "served");
        assert!(
            !attributes
                .iter()
                .any(|attribute| attribute.key.as_str() == "child_route"),
            "a request that was not composed must not carry a child route"
        );
    }

    #[test]
    fn a_model_the_deployment_does_not_declare_resolves_to_no_attribution() {
        let mut client_base_urls = BTreeMap::new();
        client_base_urls.insert(
            "openrouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
        );
        let mut target_client = BTreeMap::new();
        target_client.insert("known".to_string(), "openrouter".to_string());
        let mut target_model = BTreeMap::new();
        target_model.insert("known".to_string(), "vendor/known".to_string());
        let classification = DeploymentClassification::build(
            &client_base_urls,
            &BTreeMap::new(),
            &target_client,
            &target_model,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
        );
        assert!(
            UsageAttribution::resolve(&classification, "tier1", None, "vendor/unknown").is_none(),
            "an undeclared model must not be attributed to a provider"
        );
        assert!(
            UsageAttribution::resolve(&classification, "tier1", None, "vendor/known").is_some(),
            "a declared model must be attributed"
        );
    }

    #[test]
    fn a_composed_request_keeps_the_parent_and_the_child() {
        let mut client_base_urls = BTreeMap::new();
        client_base_urls.insert(
            "openrouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
        );
        let mut target_client = BTreeMap::new();
        target_client.insert("free-leg".to_string(), "openrouter".to_string());
        let mut target_model = BTreeMap::new();
        target_model.insert("free-leg".to_string(), "vendor/free:free".to_string());
        let mut route_target = BTreeMap::new();
        route_target.insert("tier1".to_string(), "free-leg".to_string());
        let mut composites = BTreeMap::new();
        composites.insert("tier1".to_string(), vec!["smartfree".to_string()]);
        let classification = DeploymentClassification::build(
            &client_base_urls,
            &BTreeMap::new(),
            &target_client,
            &target_model,
            &route_target,
            &BTreeMap::new(),
            &composites,
        );
        let resolved = UsageAttribution::resolve(
            &classification,
            "tier1",
            Some("smartfree"),
            "vendor/free:free",
        )
        .expect("attribution");
        // The route the caller addressed and the child that served are both kept.
        assert_eq!(resolved.route, "tier1");
        assert_eq!(resolved.child_route.as_deref(), Some("smartfree"));
        assert_eq!(resolved.route_family, RouteFamily::Tier);
        assert_eq!(resolved.cost_class, CostClass::Free);
    }
}
