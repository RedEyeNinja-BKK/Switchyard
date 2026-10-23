// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rolling time windows over served usage.
//!
//! A counter's lifetime value is not a rolling-window value, and a PromQL
//! `increase()` over a counter is not the same figure either: it extrapolates from
//! the samples that happen to exist, so a window that starts before the process
//! started, or that contains a restart, is reported as something the window never
//! held. This module keeps the events themselves, so a one-hour figure means the
//! usage observed in the last hour by this process, and nothing else.
//!
//! Three windows are published, each under its own `window` label:
//!
//! * `1h`  — the last hour.
//! * `24h` — the last day, rolling. This is not a calendar day; the Bangkok calendar
//!   day is a separate view with its own series.
//!
//! Events are aggregated into one-minute buckets. A window is exact to the minute: a
//! bucket is included when its start is inside the window, so a rolling figure can
//! differ from the true instant-by-instant value by at most the last partial minute.
//! That is stated here rather than hidden, because a figure presented as exact that
//! is actually minute-granular is the kind of quiet approximation this module exists
//! to avoid.
//!
//! The events are bounded: buckets older than the longest window are dropped on every
//! publish, so memory follows the last day of activity rather than the life of the
//! process.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use opentelemetry::{KeyValue, global};

/// The bucket width. One minute bounds both memory and the window's resolution.
const BUCKET_SECONDS: u64 = 60;

/// The longest window; older buckets are dropped.
const RETENTION_SECONDS: u64 = 24 * 60 * 60;

/// The windows published, as (label, length in seconds).
const WINDOWS: [(&str, u64); 2] = [("1h", 60 * 60), ("24h", 24 * 60 * 60)];

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

fn bucket_of(seconds: u64) -> u64 {
    seconds - (seconds % BUCKET_SECONDS)
}

/// One measured quantity recorded against one attribution in one minute.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EventKey {
    bucket: u64,
    /// The attribution labels, already flattened, so two events compare equal exactly
    /// when every label is equal.
    attributes: Vec<(String, String)>,
    /// Whether this is a token, a request or a cost amount.
    kind: MetricKind,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MetricKind {
    Tokens,
    Requests,
    /// A provider-reported cost, in the currency the event carries.
    Cost,
}

/// Rolling-window totals over served usage.
pub struct RollingWindows {
    totals: Mutex<BTreeMap<EventKey, f64>>,
}

/// The process-wide rolling windows.
///
/// The windows are process state, like the metric registry they publish into, so the
/// code that records usage does not have to carry a handle through every call path to
/// reach them. A test that needs isolation builds its own `RollingWindows` instead of
/// using this one.
pub fn global() -> &'static RollingWindows {
    static WINDOWS: std::sync::OnceLock<RollingWindows> = std::sync::OnceLock::new();
    WINDOWS.get_or_init(RollingWindows::new)
}

/// Publishes the process-wide windows.
pub fn publish_global() {
    global().publish();
}

impl Default for RollingWindows {
    fn default() -> Self {
        Self::new()
    }
}

impl RollingWindows {
    /// Builds empty windows.
    pub fn new() -> Self {
        Self {
            totals: Mutex::new(BTreeMap::new()),
        }
    }

    /// Records one amount against an attribution, now.
    fn record(&self, kind: MetricKind, attributes: &[KeyValue], amount: f64) {
        self.record_at(kind, attributes, amount, now_seconds());
    }

    /// Records one amount at a given instant, so windows can be exercised without
    /// waiting for real time to pass.
    fn record_at(&self, kind: MetricKind, attributes: &[KeyValue], amount: f64, at: u64) {
        let flattened: Vec<(String, String)> = attributes
            .iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();
        let key = EventKey {
            bucket: bucket_of(at),
            attributes: flattened,
            kind,
        };
        let mut totals = self.totals.lock().expect("rolling windows poisoned");
        *totals.entry(key).or_insert(0.0) += amount;
    }

    /// Records one served request's token components.
    pub fn record_tokens(&self, attributes: &[KeyValue], amount: u64) {
        self.record(MetricKind::Tokens, attributes, amount as f64);
    }

    /// Records one served request.
    pub fn record_request(&self, attributes: &[KeyValue]) {
        self.record(MetricKind::Requests, attributes, 1.0);
    }

    /// Records one provider-reported cost, with the currency it was reported in.
    pub fn record_cost(&self, attributes: &[KeyValue], currency: &str, amount: f64) {
        let mut with_currency = attributes.to_vec();
        with_currency.push(KeyValue::new("currency", currency.to_string()));
        self.record(MetricKind::Cost, &with_currency, amount);
    }

    /// Drops the buckets that no window can still reach.
    fn prune(&self, totals: &mut BTreeMap<EventKey, f64>, now: u64) {
        let oldest = bucket_of(now.saturating_sub(RETENTION_SECONDS));
        totals.retain(|key, _| key.bucket >= oldest);
    }

    /// The total of one kind over one window, ending now.
    #[cfg(test)]
    fn total(&self, kind: MetricKind, window_seconds: u64, at: u64) -> f64 {
        let totals = self.totals.lock().expect("rolling windows poisoned");
        let cutoff = at.saturating_sub(window_seconds);
        totals
            .iter()
            .filter(|(key, _)| key.kind == kind && key.bucket >= bucket_of(cutoff))
            .map(|(_, value)| *value)
            .sum()
    }

    /// The per-attribution totals of one kind over one window.
    fn by_attribution(
        &self,
        kind: MetricKind,
        window_seconds: u64,
        at: u64,
    ) -> BTreeMap<Vec<(String, String)>, f64> {
        let totals = self.totals.lock().expect("rolling windows poisoned");
        let cutoff = bucket_of(at.saturating_sub(window_seconds));
        let mut out: BTreeMap<Vec<(String, String)>, f64> = BTreeMap::new();
        for (key, value) in totals.iter() {
            if key.kind != kind || key.bucket < cutoff {
                continue;
            }
            *out.entry(key.attributes.clone()).or_insert(0.0) += *value;
        }
        out
    }

    /// Publishes every window as gauges.
    ///
    /// These are gauges rather than counters because a rolling total is a level that
    /// rises and falls: as a counter it would only ever grow, and would report the
    /// usage of the last day as though it were the lifetime of the process.
    pub fn publish(&self) {
        let now = now_seconds();
        {
            let mut totals = self.totals.lock().expect("rolling windows poisoned");
            self.prune(&mut totals, now);
        }
        let meter = global::meter("switchyard");
        for (window_label, window_seconds) in WINDOWS {
            for (attributes, total) in self.by_attribution(MetricKind::Tokens, window_seconds, now)
            {
                let mut labels: Vec<KeyValue> = attributes
                    .into_iter()
                    .map(|(key, value)| KeyValue::new(key, value))
                    .collect();
                labels.push(KeyValue::new("window", window_label.to_string()));
                meter
                    .f64_gauge("switchyard.economics_rolling_tokens")
                    .build()
                    .record(total, &labels);
            }
            for (attributes, total) in
                self.by_attribution(MetricKind::Requests, window_seconds, now)
            {
                let mut labels: Vec<KeyValue> = attributes
                    .into_iter()
                    .map(|(key, value)| KeyValue::new(key, value))
                    .collect();
                labels.push(KeyValue::new("window", window_label.to_string()));
                meter
                    .f64_gauge("switchyard.economics_rolling_requests")
                    .build()
                    .record(total, &labels);
            }
            for (attributes, total) in self.by_attribution(MetricKind::Cost, window_seconds, now) {
                let mut labels: Vec<KeyValue> = attributes
                    .into_iter()
                    .map(|(key, value)| KeyValue::new(key, value))
                    .collect();
                labels.push(KeyValue::new("window", window_label.to_string()));
                meter
                    .f64_gauge("switchyard.economics_rolling_provider_cost")
                    .build()
                    .record(total, &labels);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attribution() -> Vec<KeyValue> {
        vec![
            KeyValue::new("provider", "openrouter"),
            KeyValue::new("component", "prompt"),
        ]
    }

    /// A window must not report usage that has aged out of it. This is the whole
    /// reason the module exists: `increase()` cannot tell the two apart.
    #[test]
    fn usage_ages_out_of_the_window() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        windows.record_at(MetricKind::Tokens, &attribution(), 100.0, now - 7200);
        windows.record_at(MetricKind::Tokens, &attribution(), 7.0, now - 600);

        assert_eq!(windows.total(MetricKind::Tokens, 3600, now), 7.0);
        assert_eq!(windows.total(MetricKind::Tokens, 86400, now), 107.0);
    }

    /// A window that starts before the process did must not claim to cover the time
    /// before it. The figure is what was observed, which is what the labels promise.
    #[test]
    fn a_window_never_reports_more_than_was_observed() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        // The process started ten minutes ago and saw 5 tokens.
        windows.record_at(MetricKind::Tokens, &attribution(), 5.0, now - 600);
        assert_eq!(windows.total(MetricKind::Tokens, 86400, now), 5.0);
    }

    /// An idle period contributes nothing, and does not create a zero series for an
    /// attribution that produced no usage at all.
    #[test]
    fn an_idle_window_is_zero_and_absent_rather_than_invented() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        windows.record_at(MetricKind::Tokens, &attribution(), 3.0, now - 10);
        assert_eq!(windows.total(MetricKind::Tokens, 3600, now), 3.0);
        // An attribution that never appeared is not in the per-attribution map.
        let other = [KeyValue::new("provider", "deepseek")];
        let by_attr = windows.by_attribution(MetricKind::Tokens, 3600, now);
        let flattened: Vec<(String, String)> = other
            .iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();
        assert!(!by_attr.contains_key(&flattened));
    }

    /// Requests and tokens are separate quantities and are never added together.
    #[test]
    fn requests_and_tokens_are_counted_separately() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        windows.record_at(MetricKind::Requests, &attribution(), 1.0, now - 30);
        windows.record_at(MetricKind::Tokens, &attribution(), 500.0, now - 30);
        assert_eq!(windows.total(MetricKind::Requests, 3600, now), 1.0);
        assert_eq!(windows.total(MetricKind::Tokens, 3600, now), 500.0);
    }

    /// A cost recorded in one currency must not be folded into a total in another.
    #[test]
    fn costs_keep_their_currency() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        let base = vec![KeyValue::new("provider", "openrouter")];
        windows.record_at(
            MetricKind::Cost,
            &{
                let mut a = base.clone();
                a.push(KeyValue::new("currency", "USD"));
                a
            },
            2.0,
            now - 60,
        );
        windows.record_at(
            MetricKind::Cost,
            &{
                let mut a = base.clone();
                a.push(KeyValue::new("currency", "CNY"));
                a
            },
            10.0,
            now - 60,
        );
        let by_attr = windows.by_attribution(MetricKind::Cost, 3600, now);
        let usd: Vec<(String, String)> = vec![
            ("provider".to_string(), "openrouter".to_string()),
            ("currency".to_string(), "USD".to_string()),
        ];
        let cny: Vec<(String, String)> = vec![
            ("provider".to_string(), "openrouter".to_string()),
            ("currency".to_string(), "CNY".to_string()),
        ];
        assert_eq!(by_attr.get(&usd), Some(&2.0));
        assert_eq!(by_attr.get(&cny), Some(&10.0));
    }

    /// Retention is bounded, so a long-running process does not accumulate every event
    /// it has ever seen.
    #[test]
    fn buckets_older_than_retention_are_dropped() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        windows.record_at(MetricKind::Tokens, &attribution(), 1.0, now - 90_000);
        windows.record_at(MetricKind::Tokens, &attribution(), 2.0, now - 60);
        {
            let mut totals = windows.totals.lock().unwrap();
            windows.prune(&mut totals, now);
            assert_eq!(totals.len(), 1, "only the bucket inside retention survives");
        }
    }

    /// Publishing is idempotent: a rolling total is a level, so publishing it twice
    /// must not change it.
    #[test]
    fn publishing_does_not_change_the_totals() {
        let windows = RollingWindows::new();
        let now = 1_800_000_000;
        windows.record_at(MetricKind::Tokens, &attribution(), 9.0, now - 30);
        let before = windows.total(MetricKind::Tokens, 3600, now);
        windows.publish();
        windows.publish();
        assert_eq!(windows.total(MetricKind::Tokens, 3600, now), before);
    }
}
