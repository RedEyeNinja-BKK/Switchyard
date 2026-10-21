// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Bangkok calendar-day window.
//!
//! A rolling 24-hour window and a calendar day are different questions, and a
//! dashboard that shows one while labelling it as the other is wrong. The calendar
//! window is a Bangkok day: the day boundary is midnight in `UTC+07:00`, not midnight
//! in UTC and not a rolling 24 hours.
//!
//! Prometheus cannot express this: the `@` modifier needs a literal timestamp, and
//! a computed one is rejected by the API. So Switchyard publishes the calendar-day
//! counters itself, reset at the Bangkok midnight.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use opentelemetry::{KeyValue, global};

/// The offset of the deployment's calendar from UTC, in seconds.
pub const BANGKOK_OFFSET_SECONDS: u64 = 7 * 60 * 60;

/// The Unix timestamp of the start of the current Bangkok calendar day.
///
/// A day boundary is midnight in `UTC+07:00`. The local wall clock is `now + 7h`,
/// so the local midnight is that value floored to a day, and the instant back on the
/// UTC scale is that minus the offset.
pub fn bangkok_day_start(now_seconds: u64) -> u64 {
    let local = now_seconds + BANGKOK_OFFSET_SECONDS;
    (local - (local % 86_400)) - BANGKOK_OFFSET_SECONDS
}

/// The Unix timestamp of the next Bangkok midnight.
pub fn bangkok_day_end(now_seconds: u64) -> u64 {
    bangkok_day_start(now_seconds) + 86_400
}

/// The number of seconds since the Unix epoch, or zero if the clock is before it.
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// The counters of one calendar day, reset when the day rolls over.
#[derive(Default)]
struct DayCounters {
    /// The day the counters describe.
    day_start: u64,
    /// The requests served in the day, by attribution.
    requests: Vec<(Vec<KeyValue>, u64)>,
    /// The tokens consumed in the day, by attribution and component.
    tokens: Vec<(Vec<KeyValue>, u64)>,
}

/// Publishes calendar-day counters alongside the rolling and cumulative ones.
///
/// The rolling windows and the process-lifetime counters stay where they are. This
/// adds a third, clearly separate view: the current Bangkok day.
pub struct CalendarWindow {
    counters: Mutex<DayCounters>,
}

impl Default for CalendarWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl CalendarWindow {
    /// Builds an empty calendar window.
    pub fn new() -> Self {
        Self {
            counters: Mutex::new(DayCounters {
                day_start: bangkok_day_start(now_seconds()),
                ..DayCounters::default()
            }),
        }
    }

    /// Records a request in the current calendar day.
    pub fn record_request(&self, attributes: &[KeyValue], count: u64) {
        let mut counters = self.counters.lock().expect("calendar window poisoned");
        self.roll_over(&mut counters);
        let key = attributes.to_vec();
        match counters
            .requests
            .iter_mut()
            .find(|(existing, _)| *existing == key)
        {
            Some((_, total)) => *total += count,
            None => counters.requests.push((key, count)),
        }
    }

    /// Records tokens in the current calendar day.
    pub fn record_tokens(&self, attributes: &[KeyValue], count: u64) {
        let mut counters = self.counters.lock().expect("calendar window poisoned");
        self.roll_over(&mut counters);
        let key = attributes.to_vec();
        match counters
            .tokens
            .iter_mut()
            .find(|(existing, _)| *existing == key)
        {
            Some((_, total)) => *total += count,
            None => counters.tokens.push((key, count)),
        }
    }

    /// Resets the counters when the Bangkok day has rolled over.
    fn roll_over(&self, counters: &mut DayCounters) {
        let today = bangkok_day_start(now_seconds());
        if counters.day_start != today {
            counters.day_start = today;
            counters.requests.clear();
            counters.tokens.clear();
        }
    }

    /// The tokens recorded for one attribution in the current day.
    #[cfg(test)]
    fn tokens_for(&self, attributes: &[KeyValue]) -> u64 {
        let counters = self.counters.lock().expect("calendar window poisoned");
        counters
            .tokens
            .iter()
            .find(|(existing, _)| existing == attributes)
            .map(|(_, total)| *total)
            .unwrap_or(0)
    }

    /// The requests recorded for one attribution in the current day.
    #[cfg(test)]
    fn requests_for(&self, attributes: &[KeyValue]) -> u64 {
        let counters = self.counters.lock().expect("calendar window poisoned");
        counters
            .requests
            .iter()
            .find(|(existing, _)| existing == attributes)
            .map(|(_, total)| *total)
            .unwrap_or(0)
    }

    /// Moves the window's day to another day, so the roll-over can be exercised
    /// without waiting for a real midnight.
    #[cfg(test)]
    fn set_day_start_for_test(&self, day_start: u64) {
        let mut counters = self.counters.lock().expect("calendar window poisoned");
        counters.day_start = day_start;
    }

    /// Publishes the current day's counters.
    ///
    /// These are published as gauges, not counters. A counter accumulates, and this
    /// value is republished every publish interval: as a counter it would grow by the
    /// day's total on every publish and report a figure many times larger than the
    /// truth. A gauge reports the day's total as it stands, which is what a
    /// calendar-day view asks for.
    pub fn publish(&self) {
        let counters = self.counters.lock().expect("calendar window poisoned");
        let day_start = counters.day_start;
        let day_end = bangkok_day_end(now_seconds());
        let meter = global::meter("switchyard");
        meter
            .f64_gauge("switchyard.economics_calendar_day_start")
            .build()
            .record(day_start as f64, &[]);
        meter
            .f64_gauge("switchyard.economics_calendar_day_end")
            .build()
            .record(day_end as f64, &[]);
        meter
            .f64_gauge("switchyard.economics_calendar_day_elapsed_seconds")
            .build()
            .record((now_seconds().saturating_sub(day_start)) as f64, &[]);
        for (attributes, total) in &counters.requests {
            meter
                .f64_gauge("switchyard.economics_calendar_requests")
                .build()
                .record(*total as f64, attributes);
        }
        for (attributes, total) in &counters.tokens {
            meter
                .f64_gauge("switchyard.economics_calendar_tokens")
                .build()
                .record(*total as f64, attributes);
        }
    }
}

/// The instant the calendar window was last published, so a stalled publisher is
/// visible rather than silent.
static LAST_PUBLISH: AtomicU64 = AtomicU64::new(0);

/// Publishes the calendar window and records when it happened.
pub fn publish_calendar(window: &CalendarWindow) {
    window.publish();
    LAST_PUBLISH.store(now_seconds(), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bangkok_day_starts_at_bangkok_midnight() {
        // 2026-11-13T07:50:00Z is 14:50 in Bangkok, so the Bangkok day started at
        // 2026-11-12T17:00:00Z, which is midnight in UTC+07:00.
        let now = 1_794_556_200;
        assert_eq!(bangkok_day_start(now), 1_794_502_800);
        assert_eq!(bangkok_day_end(now), 1_794_502_800 + 86_400);
    }

    #[test]
    fn the_bangkok_day_boundary_is_not_the_utc_day_boundary() {
        // The Bangkok day starts seven hours before the UTC day does.
        let now = 1_794_556_200;
        let utc_day_start = now - (now % 86_400);
        assert_eq!(utc_day_start - bangkok_day_start(now), 7 * 3600);
    }

    #[test]
    fn the_day_start_is_stable_within_a_day() {
        let start = bangkok_day_start(1_794_502_800);
        // A second later, an hour later and the last second of the day all belong to
        // the same Bangkok day.
        assert_eq!(bangkok_day_start(start + 1), start);
        assert_eq!(bangkok_day_start(start + 3600), start);
        assert_eq!(bangkok_day_start(start + 86_399), start);
        // The next instant is the next day.
        assert_eq!(bangkok_day_start(start + 86_400), start + 86_400);
    }

    /// A calendar-day counter that keeps counting after the day rolls over is wrong:
    /// the new day would start with the previous day's total instead of zero.
    #[test]
    fn a_new_day_starts_from_zero() {
        let window = CalendarWindow::new();
        let attributes = [KeyValue::new("provider", "openrouter")];
        window.record_tokens(&attributes, 1_000);
        window.record_request(&attributes, 1);
        assert_eq!(window.tokens_for(&attributes), 1_000);
        assert_eq!(window.requests_for(&attributes), 1);

        // Move the window to the previous day and record again. The roll-over must
        // clear the old day's counters rather than adding to them.
        window.set_day_start_for_test(bangkok_day_start(now_seconds()) - 86_400);
        window.record_tokens(&attributes, 5);
        assert_eq!(
            window.tokens_for(&attributes),
            5,
            "the new day must start from zero, not carry the previous day"
        );
        assert_eq!(window.requests_for(&attributes), 0);
    }

    /// The same usage recorded twice is counted twice, and a different attribution is
    /// counted separately. This is what keeps the day view from collapsing distinct
    /// providers into one number.
    #[test]
    fn attributions_are_counted_separately() {
        let window = CalendarWindow::new();
        let openrouter = [KeyValue::new("provider", "openrouter")];
        let deepseek = [KeyValue::new("provider", "deepseek")];
        window.record_tokens(&openrouter, 10);
        window.record_tokens(&openrouter, 5);
        window.record_tokens(&deepseek, 7);
        assert_eq!(window.tokens_for(&openrouter), 15);
        assert_eq!(window.tokens_for(&deepseek), 7);
    }

    /// A day total is published repeatedly, so it must be a gauge. A counter would
    /// add the day's total again on every publish and report a figure that grows
    /// with the number of publishes rather than with the usage.
    #[test]
    fn the_day_total_is_published_as_a_gauge_and_not_as_a_counter() {
        let window = CalendarWindow::new();
        let attributes = [KeyValue::new("provider", "openrouter")];
        window.record_tokens(&attributes, 42);
        // Publishing twice must not change the published total.
        window.publish();
        window.publish();
        assert_eq!(
            window.tokens_for(&attributes),
            42,
            "publishing must not alter the recorded day total"
        );
    }
}
