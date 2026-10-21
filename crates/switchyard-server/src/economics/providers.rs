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
//!   label value is a provider name, a currency or a pool name.
//! * It is asynchronous. A provider that is slow, down or unauthorized must not
//!   delay an inference request, so the poller runs on its own task and records a
//!   failure rather than propagating one.
//! * It never invents a measurement. A provider that exposes no balance produces no
//!   balance series, which is absent rather than zero.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use opentelemetry::{KeyValue, global};

/// What a provider reported about its account.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AccountObservation {
    /// The provider's reported balance, by currency.
    pub balances: BTreeMap<String, f64>,
    /// The provider's reported purchased credit, by currency.
    pub credits: BTreeMap<String, f64>,
    /// The provider's reported cumulative usage, by currency.
    pub usage: BTreeMap<String, f64>,
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

        for (currency, value) in &observation.balances {
            meter
                .f64_gauge("switchyard.economics_provider_balance")
                .build()
                .record(
                    *value,
                    &[
                        KeyValue::new("provider", provider.to_string()),
                        KeyValue::new("currency", currency.clone()),
                    ],
                );
        }
        for (currency, value) in &observation.credits {
            meter
                .f64_gauge("switchyard.economics_provider_credit")
                .build()
                .record(
                    *value,
                    &[
                        KeyValue::new("provider", provider.to_string()),
                        KeyValue::new("currency", currency.clone()),
                    ],
                );
        }
        for (currency, value) in &observation.usage {
            meter
                .f64_gauge("switchyard.economics_provider_usage")
                .build()
                .record(
                    *value,
                    &[
                        KeyValue::new("provider", provider.to_string()),
                        KeyValue::new("currency", currency.clone()),
                    ],
                );
        }
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
