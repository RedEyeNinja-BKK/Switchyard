// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider account observations, collected with Switchyard's own credentials.
//!
//! This is the piece that makes the economics surface Switchyard-owned rather than a
//! second accounting system. Switchyard already holds the provider credentials it
//! uses to serve inference, so it is the component with the authority to read the
//! matching account APIs. It does so with those credentials, and publishes the result
//! through its own metric surface. Nothing downstream of Switchyard contacts a
//! provider.
//!
//! The collection is deliberately narrow:
//!
//! * It is read-only. No account, key or balance is ever written.
//! * It is bounded. Each provider contributes a fixed number of series, and every
//!   label value comes from a fixed set.
//! * It is asynchronous. A provider that is slow, down or unauthorized must not
//!   delay an inference request, so the poller runs on its own task and records a
//!   failure rather than propagating one.
//! * It never invents a measurement. A provider that exposes no balance produces no
//!   balance series, which is absent rather than zero.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use opentelemetry::{KeyValue, global};

/// The accounting scope a provider figure covers.
///
/// A provider reports some figures for the whole account and some for the single key
/// Switchyard holds. The two are different amounts of money, so they are published
/// under different scopes rather than added together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccountScope {
    /// The whole provider account.
    Account,
    /// The single API key Switchyard holds.
    Key,
}

impl AccountScope {
    /// The label value for this scope.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Account => "account",
            Self::Key => "key",
        }
    }
}

/// The reporting period a provider figure covers.
///
/// A provider reports usage over several windows at once. Each window is a different
/// amount, so each is published under its own period.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReportingPeriod {
    /// The whole life of the account or key.
    Total,
    /// The current day.
    Daily,
    /// The current week.
    Weekly,
    /// The current month.
    Monthly,
}

impl ReportingPeriod {
    /// The label value for this period.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Total => "total",
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
        }
    }
}

/// One amount a provider reported, with the three facts that describe it kept apart.
///
/// The currency, the accounting scope and the reporting period are separate facts.
/// A provider that reports a monthly figure and an account-wide figure must produce
/// two series that differ in one label each, so a query can select one of them and
/// so a reader can tell which figure is which.
#[derive(Clone, Debug, PartialEq)]
pub struct AccountAmount {
    /// The currency the amount is denominated in.
    pub currency: String,
    /// Whether the amount covers the account or the key.
    pub scope: AccountScope,
    /// The window the amount covers.
    pub period: ReportingPeriod,
    /// The amount itself.
    pub value: f64,
}

impl AccountAmount {
    /// Builds one reported amount.
    pub fn new(
        currency: impl Into<String>,
        scope: AccountScope,
        period: ReportingPeriod,
        value: f64,
    ) -> Self {
        Self {
            currency: currency.into(),
            scope,
            period,
            value,
        }
    }
}

/// What a provider reported about its account.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AccountObservation {
    /// The balance the provider reported.
    pub balances: Vec<AccountAmount>,
    /// The purchased credit the provider reported.
    pub credits: Vec<AccountAmount>,
    /// The usage the provider reported.
    pub usage: Vec<AccountAmount>,
    /// The remaining allowance of a provider-reported window, by pool name, as a
    /// percentage.
    pub allowance_remaining_percent: BTreeMap<String, f64>,
    /// The reset time of a provider-reported window, by pool name, as a duration
    /// from the observation.
    pub allowance_reset_seconds: BTreeMap<String, f64>,
}

impl AccountObservation {
    /// Returns whether the observation carries any measurement at all.
    pub fn is_empty(&self) -> bool {
        self.balances.is_empty()
            && self.credits.is_empty()
            && self.usage.is_empty()
            && self.allowance_remaining_percent.is_empty()
            && self.allowance_reset_seconds.is_empty()
    }
}

/// The outcome of one collection attempt.
#[derive(Clone, Debug, PartialEq)]
pub enum FetchOutcome {
    /// The provider answered and the answer was understood.
    Ok(AccountObservation),
    /// The provider rejected the credential.
    Unauthorized,
    /// The provider could not be reached, or answered something unusable.
    Failed(String),
}

impl FetchOutcome {
    /// Whether the attempt reached the provider successfully.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }
}

/// The read-only account endpoints a provider exposes.
#[async_trait::async_trait]
pub trait AccountApi: Send + Sync {
    /// The provider's name, used as the metric label.
    fn provider(&self) -> &str;

    /// Reads the provider's account state.
    ///
    /// This is asynchronous because it performs network reads. The collector drives
    /// it on a runtime of its own, so a provider that is slow cannot delay an
    /// inference request.
    async fn fetch(&self) -> FetchOutcome;
}

/// A provider whose account API Switchyard does not read.
///
/// A provider with no account surface produces no account series at all. It is not
/// recorded as a zero balance, because a provider that exposes nothing has not
/// reported a balance of zero.
pub struct NoAccountApi {
    provider: String,
}

impl NoAccountApi {
    /// Names a provider that exposes no account surface.
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
        }
    }
}

#[async_trait::async_trait]
impl AccountApi for NoAccountApi {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn fetch(&self) -> FetchOutcome {
        FetchOutcome::Failed("this provider exposes no account API".to_string())
    }
}

/// Publishes the account observations and the health of the collection itself.
pub struct AccountPublisher {
    /// When each provider last answered successfully, by provider.
    last_success: BTreeMap<String, Instant>,
}

impl Default for AccountPublisher {
    fn default() -> Self {
        Self::new()
    }
}

impl AccountPublisher {
    /// Builds a publisher with no observations yet.
    pub fn new() -> Self {
        Self {
            last_success: BTreeMap::new(),
        }
    }

    /// Publishes a successful observation for a provider.
    pub fn record_success(
        &mut self,
        provider: &str,
        observation: &AccountObservation,
        fetch_duration: Duration,
    ) {
        // Freshness is the age of the newest successful reading, which is what tells
        // an operator whether the account state on the dashboard is current. The
        // fetch duration is a different fact and is published under its own name.
        let now = Instant::now();
        let previous = self.last_success.insert(provider.to_string(), now);
        let freshness = previous.map_or(0.0, |last| last.elapsed().as_secs_f64());

        let meter = global::meter("switchyard");
        meter
            .u64_counter("switchyard.economics_account_fetch")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("outcome", "ok"),
                    KeyValue::new("provider", provider.to_string()),
                ],
            );
        meter
            .f64_gauge("switchyard.economics_account_fetch_duration_seconds")
            .build()
            .record(
                fetch_duration.as_secs_f64(),
                &[KeyValue::new("provider", provider.to_string())],
            );
        meter
            .f64_gauge("switchyard.economics_account_freshness_seconds")
            .build()
            .record(
                freshness,
                &[KeyValue::new("provider", provider.to_string())],
            );

        publish_amounts(
            &meter,
            "switchyard.economics_provider_balance",
            &observation.balances,
            provider,
        );
        publish_amounts(
            &meter,
            "switchyard.economics_provider_credit",
            &observation.credits,
            provider,
        );
        publish_amounts(
            &meter,
            "switchyard.economics_provider_usage",
            &observation.usage,
            provider,
        );
        for (pool, value) in &observation.allowance_remaining_percent {
            meter
                .f64_gauge("switchyard.economics_allowance_remaining_percent")
                .build()
                .record(
                    *value,
                    &[
                        KeyValue::new("provider", provider.to_string()),
                        KeyValue::new("pool", pool.clone()),
                    ],
                );
        }
        for (pool, value) in &observation.allowance_reset_seconds {
            meter
                .f64_gauge("switchyard.economics_allowance_reset_seconds")
                .build()
                .record(
                    *value,
                    &[
                        KeyValue::new("provider", provider.to_string()),
                        KeyValue::new("pool", pool.clone()),
                    ],
                );
        }
    }

    /// Records an authentication failure for a provider.
    pub fn record_unauthorized(&mut self, provider: &str) {
        let meter = global::meter("switchyard");
        meter
            .u64_counter("switchyard.economics_account_fetch")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("outcome", "unauthorized"),
                    KeyValue::new("provider", provider.to_string()),
                ],
            );
        meter
            .u64_counter("switchyard.economics_account_auth_failures")
            .build()
            .add(1, &[KeyValue::new("provider", provider.to_string())]);
    }

    /// Records a collection failure for a provider.
    pub fn record_failure(&mut self, provider: &str) {
        global::meter("switchyard")
            .u64_counter("switchyard.economics_account_fetch")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("outcome", "failed"),
                    KeyValue::new("provider", provider.to_string()),
                ],
            );
    }
}

/// Publishes a set of reported amounts under one metric family.
///
/// Each amount carries its own currency, scope and period, so the three facts stay
/// separable in the metric namespace and a query can select one of them.
fn publish_amounts(
    meter: &opentelemetry::metrics::Meter,
    name: &'static str,
    amounts: &[AccountAmount],
    provider: &str,
) {
    for amount in amounts {
        meter.f64_gauge(name).build().record(
            amount.value,
            &[
                KeyValue::new("provider", provider.to_string()),
                KeyValue::new("currency", amount.currency.clone()),
                KeyValue::new("scope", amount.scope.as_str()),
                KeyValue::new("period", amount.period.as_str()),
            ],
        );
    }
}
