// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the labels the provider account figures are published under.
//!
//! The economics dashboards query these series by label. A label that carries more
//! than one fact at once — a currency together with an accounting scope, or a
//! currency together with a reporting period — makes two different amounts
//! indistinguishable and invites a query to sum series that were never meant to be
//! summed. These tests pin the published label set, so the schema is checked at the
//! point a reader depends on it rather than only at the point it is built.
//!
//! The metric registry is process-wide, and these tests run in parallel, so each
//! one publishes under its own provider and reads only that provider's series.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use opentelemetry_sdk::metrics::SdkMeterProvider;
use prometheus::proto::MetricFamily;
use prometheus::{Encoder, Registry};
use switchyard_server::economics::providers::{
    AccountAmount, AccountObservation, AccountPublisher, AccountScope, ReportingPeriod,
};

/// The registry the publisher writes into, installed once as the global provider.
fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
            .unwrap_or_else(|error| panic!("failed to build metrics exporter: {error}"));
        let provider = SdkMeterProvider::builder().with_reader(exporter).build();
        opentelemetry::global::set_meter_provider(provider);
        registry
    })
}

/// The labels the exporter adds on its own, which are not part of the schema.
fn is_instrument_label(name: &str) -> bool {
    name == "otel_scope_name" || name == "otel_scope_version"
}

/// The published series of one metric family and one provider, keyed by label set.
///
/// The key is the label set rendered as sorted `name=value` pairs, so two series
/// that differ in any label are two different keys.
fn published_series(
    families: &[MetricFamily],
    family: &str,
    provider: &str,
) -> BTreeMap<String, f64> {
    let mut found = BTreeMap::new();
    for metric_family in families {
        if metric_family.name() != family {
            continue;
        }
        for metric in metric_family.get_metric() {
            let labels = metric.get_label();
            let is_provider = labels
                .iter()
                .any(|label| label.name() == "provider" && label.value() == provider);
            if !is_provider {
                continue;
            }
            let mut rendered: Vec<String> = labels
                .iter()
                .filter(|label| !is_instrument_label(label.name()))
                .map(|label| format!("{}={}", label.name(), label.value()))
                .collect();
            rendered.sort();
            let value = metric
                .get_gauge()
                .0
                .as_ref()
                .and_then(|gauge| gauge.value)
                .unwrap_or_default();
            found.insert(rendered.join(","), value);
        }
    }
    found
}

/// The label names the series of a family and provider carry.
fn label_names(families: &[MetricFamily], family: &str, provider: &str) -> Vec<Vec<String>> {
    families
        .iter()
        .filter(|metric_family| metric_family.name() == family)
        .flat_map(|metric_family| metric_family.get_metric())
        .filter(|metric| {
            metric
                .get_label()
                .iter()
                .any(|label| label.name() == "provider" && label.value() == provider)
        })
        .map(|metric| {
            let mut labels: Vec<String> = metric
                .get_label()
                .iter()
                .filter(|label| !is_instrument_label(label.name()))
                .map(|label| label.name().to_string())
                .collect();
            labels.sort();
            labels
        })
        .collect()
}

/// The published provider account series carry a currency, a scope and a period.
///
/// These are three independent labels. The scope says whether the figure covers the
/// account or the single key Switchyard holds, and the period says which window it
/// covers. A reader selects on them without parsing any label value.
#[test]
fn the_published_account_figures_carry_currency_scope_and_period() {
    let registry = registry();
    let provider = "schema-labels";

    let mut publisher = AccountPublisher::new();
    publisher.record_success(
        provider,
        &AccountObservation {
            balances: vec![AccountAmount::new(
                "USD",
                AccountScope::Account,
                ReportingPeriod::Total,
                0.75,
            )],
            credits: vec![AccountAmount::new(
                "USD",
                AccountScope::Account,
                ReportingPeriod::Total,
                125.0,
            )],
            usage: vec![
                AccountAmount::new("USD", AccountScope::Key, ReportingPeriod::Total, 12.5),
                AccountAmount::new("USD", AccountScope::Key, ReportingPeriod::Monthly, 40.0),
                AccountAmount::new("USD", AccountScope::Account, ReportingPeriod::Total, 124.0),
            ],
            ..AccountObservation::default()
        },
        Duration::from_millis(120),
    );

    let families = registry.gather();
    let usage = published_series(&families, "switchyard_economics_provider_usage", provider);
    assert_eq!(usage.len(), 3, "three usage figures, got {usage:?}");

    let expected = BTreeMap::from([
        (
            "currency=USD,period=monthly,provider=schema-labels,scope=key".to_string(),
            40.0,
        ),
        (
            "currency=USD,period=total,provider=schema-labels,scope=account".to_string(),
            124.0,
        ),
        (
            "currency=USD,period=total,provider=schema-labels,scope=key".to_string(),
            12.5,
        ),
    ]);
    assert_eq!(usage, expected);

    // Every family carries the same three facts, so one query shape reads all of
    // them.
    for family in [
        "switchyard_economics_provider_usage",
        "switchyard_economics_provider_credit",
        "switchyard_economics_provider_balance",
    ] {
        let names = label_names(&families, family, provider);
        assert!(
            names
                .iter()
                .all(|labels| { labels == &["currency", "period", "provider", "scope"] }),
            "{family} must carry currency, scope and period, got {names:?}"
        );
    }
}

/// The purchased credit and the remaining balance are different series.
///
/// A provider sells credit and reports how much of it was consumed. What is left is
/// a third figure. Publishing the credit and the balance under the same labels but
/// different names keeps an operator from reading the purchased credit as the money
/// that is still available.
#[test]
fn the_credit_and_the_remaining_balance_are_different_families() {
    let registry = registry();
    let provider = "credit-balance";

    let mut publisher = AccountPublisher::new();
    publisher.record_success(
        provider,
        &AccountObservation {
            balances: vec![AccountAmount::new(
                "USD",
                AccountScope::Account,
                ReportingPeriod::Total,
                0.75,
            )],
            credits: vec![AccountAmount::new(
                "USD",
                AccountScope::Account,
                ReportingPeriod::Total,
                125.0,
            )],
            usage: vec![AccountAmount::new(
                "USD",
                AccountScope::Account,
                ReportingPeriod::Total,
                124.25,
            )],
            ..AccountObservation::default()
        },
        Duration::from_millis(10),
    );

    let families = registry.gather();
    let credit = published_series(&families, "switchyard_economics_provider_credit", provider);
    let balance = published_series(&families, "switchyard_economics_provider_balance", provider);
    let usage = published_series(&families, "switchyard_economics_provider_usage", provider);

    let key = "currency=USD,period=total,provider=credit-balance,scope=account";
    let credit_value = credit.get(key).copied().expect("a purchased credit series");
    let balance_value = balance
        .get(key)
        .copied()
        .expect("a remaining balance series");
    let consumed = usage.get(key).copied().expect("a consumed credit series");

    assert_eq!(credit_value, 125.0, "the credit sold");
    assert_eq!(consumed, 124.25, "the credit consumed");
    assert_eq!(balance_value, 0.75, "the credit remaining");
    assert_ne!(
        credit_value, balance_value,
        "the purchased credit is not the remaining balance"
    );
}

/// The series are published as gauges, not counters.
///
/// A counter would grow every collection round, so a dashboard would show the
/// balance climbing with each read. The balance is a level, so it is a gauge.
#[test]
fn the_account_figures_are_published_as_gauges() {
    let registry = registry();
    let provider = "gauge-check";

    let mut publisher = AccountPublisher::new();
    publisher.record_success(
        provider,
        &AccountObservation {
            balances: vec![AccountAmount::new(
                "CNY",
                AccountScope::Account,
                ReportingPeriod::Total,
                12.34,
            )],
            ..AccountObservation::default()
        },
        Duration::from_millis(5),
    );

    let families = registry.gather();
    let family = families
        .iter()
        .find(|family| family.name() == "switchyard_economics_provider_balance")
        .expect("the balance family");
    assert_eq!(
        family.type_,
        Some(prometheus::proto::MetricType::GAUGE.into()),
        "a balance is a level, so it must not be a counter"
    );

    // The encoded text carries no counter suffix, which is what a Prometheus query
    // on a gauge expects.
    let mut encoded = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&families, &mut encoded)
        .expect("encode the metrics");
    let text = String::from_utf8(encoded).expect("the metrics are text");
    assert!(
        text.contains("switchyard_economics_provider_balance{"),
        "the balance must be published"
    );
    assert!(
        !text.contains("switchyard_economics_provider_balance_total"),
        "a balance must not be published as a counter"
    );
}
