// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-model daily request allowances.
//!
//! Some providers cap the number of *requests* a project may make per model per day,
//! independently of tokens (the Google AI Studio free tier publishes exactly that: a
//! daily RPD figure whose value differs per model). The cap is the operator's hard
//! budget, so it has to be spent with an actual counter rather than inferred from a
//! provider error after the fact — a rejected request is still a request the provider
//! may have counted, and the whole point of the budget is to stay inside it.
//!
//! # A limit is a budget, not a balance
//!
//! The configured limit is a *ceiling on requests this deployment may make between two
//! provider midnights*. It is not a reading of the provider's remaining allowance, and
//! this module never presents it as one: [`DailyRequestAllowance::remaining`] reports
//! what this counter itself has left, which is only the same number as the provider's
//! balance when every request that counted was made through this deployment from a fresh
//! counter. Nothing here contacts the provider to find out otherwise.
//!
//! # Where the count is taken
//!
//! [`DailyRequestAllowance::try_reserve`] is called at the top of one HTTP attempt, the
//! single point every call to a backend passes through. That is deliberately *not* the
//! routing decision: a candidate can be reached by a composite, by a direct leaf
//! address, or by an algorithm that never consults ordered candidates, and only the
//! send path sees all of them.
//!
//! A reservation is never released. An attempt that fails at the transport layer may
//! still have reached the provider, so refunding it would let a retrying client exceed
//! the cap it is trying to respect. Attempts that were never sent (an exhausted
//! allowance) reserve nothing, so a declined leg costs no budget.
//!
//! # The provider day
//!
//! The reset boundary is the provider's civil midnight in Pacific time, including its
//! daylight-saving transitions: midnight PST is 08:00 UTC and midnight PDT is 07:00 UTC.
//! A fixed offset would be wrong for eight months of the year — the counter would reopen
//! an hour after the provider's day actually started, and would keep counting requests
//! against yesterday for that hour. The rules implemented here are the United States
//! ones the provider observes: daylight time begins on the second Sunday in March at
//! 02:00 local and ends on the first Sunday in November at 02:00 local. Midnight itself
//! is never skipped or repeated by a transition, so the day boundary is unambiguous.
//!
//! No timezone database is consulted; the rule is the rule, in civil arithmetic.
//!
//! # State
//!
//! One JSON file per (quota group, model), holding the day the count belongs to and the
//! number of requests spent on it. Writers hold an exclusive advisory lock on a sibling
//! `.lock` file for the whole read-modify-write, so two processes that share a state
//! directory cannot both spend the same last request. The counter is written to a
//! temporary file, flushed, and renamed, and the directory entry is flushed afterwards:
//! a reservation this code reports as granted must survive a crash, because the request
//! it was granted for may already have reached the provider.
//!
//! Unreadable, unwritable or corrupt state fails **closed**: the request is refused with
//! a reason naming the fault, which the caller reports and routes around, rather than
//! spending budget nobody is counting.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use switchyard_protocol::ModelId;

/// Length of one civil day.
const DAY_SECONDS: i64 = 86_400;

/// Offset of Pacific standard time from UTC, in seconds (negative: west of Greenwich).
const PACIFIC_STANDARD_SECONDS: i64 = -8 * 3_600;

/// Offset of Pacific daylight time from UTC, in seconds.
const PACIFIC_DAYLIGHT_SECONDS: i64 = -7 * 3_600;

/// Days since 1970-01-01 for a civil date.
///
/// Howard Hinnant's `days_from_civil`, which is exact for the whole proleptic Gregorian
/// range the arithmetic needs.
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The civil date for a count of days since 1970-01-01.
///
/// Hinnant's `civil_from_days`, the inverse of [`days_from_civil`].
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_position = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_position + 2) / 5 + 1;
    let month = if month_position < 10 {
        month_position + 3
    } else {
        month_position - 9
    };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// Day of the week for a count of days since 1970-01-01, with Sunday as 0.
fn weekday(days: i64) -> i64 {
    // 1970-01-01 was a Thursday, which is 4 in a Sunday-first week.
    (days + 4).rem_euclid(7)
}

/// The instants daylight time starts and ends in `year`, in seconds since the Unix epoch.
///
/// Daylight time begins on the second Sunday in March at 02:00 local standard time
/// (10:00 UTC) and ends on the first Sunday in November at 02:00 local daylight time
/// (09:00 UTC).
fn daylight_bounds(year: i64) -> (i64, i64) {
    let march_first = days_from_civil(year, 3, 1);
    let first_sunday_march = march_first + (7 - weekday(march_first)) % 7;
    let start = (first_sunday_march + 7) * DAY_SECONDS + 10 * 3_600;

    let november_first = days_from_civil(year, 11, 1);
    let first_sunday_november = november_first + (7 - weekday(november_first)) % 7;
    let end = first_sunday_november * DAY_SECONDS + 9 * 3_600;
    (start, end)
}

/// The Pacific offset in effect at `seconds` since the Unix epoch.
///
/// The year is read from the UTC date. A daylight transition is never within a day of a
/// year boundary, so the UTC year and the Pacific year agree wherever this decision is
/// made.
pub fn pacific_offset_seconds(seconds: i64) -> i64 {
    let year = civil_from_days(seconds.div_euclid(DAY_SECONDS)).0;
    let (start, end) = daylight_bounds(year);
    if seconds >= start && seconds < end {
        PACIFIC_DAYLIGHT_SECONDS
    } else {
        PACIFIC_STANDARD_SECONDS
    }
}

/// The provider's civil day number containing `now`.
///
/// Days are counted from the Unix epoch so the value is directly comparable between
/// processes and across a restart, and so it equals [`days_from_civil`] of the Pacific
/// calendar date — which is what a clearance header states.
pub fn provider_day(now: SystemTime) -> u64 {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let seconds = i64::try_from(seconds).unwrap_or(i64::MAX);
    let local = seconds + pacific_offset_seconds(seconds);
    u64::try_from(local.div_euclid(DAY_SECONDS)).unwrap_or(0)
}

/// The Pacific calendar date of a provider day, as `YYYY-MM-DD`.
pub fn provider_day_label(day: u64) -> String {
    let (year, month, date) = civil_from_days(i64::try_from(day).unwrap_or(i64::MAX));
    format!("{year:04}-{month:02}-{date:02}")
}

/// Why a reservation could not be granted.
#[derive(Debug)]
pub enum ReserveError {
    /// The allowance for the current provider day is spent.
    Exhausted,
    /// The accounting state could not be read or written, so no budget may be spent.
    Unusable(String),
}

/// Counts requests against one target's daily allowance.
#[derive(Debug)]
pub struct DailyRequestAllowance {
    /// The quota group the allowance belongs to — the provider-side project whose counter
    /// this is. Every client that bills to the same project shares one group, and so one
    /// counter per model.
    group: String,
    /// The target model whose allowance this is.
    model: ModelId,
    /// Requests permitted per provider day.
    limit: u64,
    /// Where this allowance's counter lives.
    path: PathBuf,
}

/// The persisted counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counter {
    day: u64,
    spent: u64,
}

impl DailyRequestAllowance {
    /// An allowance of `limit` requests per provider day for `model` under the quota group
    /// `group`, with its counter persisted at `path`.
    pub fn new(group: impl Into<String>, model: ModelId, limit: u64, path: PathBuf) -> Self {
        Self {
            group: group.into(),
            model,
            limit,
            path,
        }
    }

    /// The configured daily budget.
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// The target model this allowance governs.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// The quota group this allowance belongs to.
    pub fn group(&self) -> &str {
        &self.group
    }

    /// The provider-day number the counter currently belongs to.
    pub fn current_day(&self) -> u64 {
        provider_day(SystemTime::now())
    }

    /// Requests this counter still permits in the current provider day.
    ///
    /// This is the budget's own remainder, not the provider's balance; see the module
    /// documentation. Reports `0` rather than an error when the state is unusable: a
    /// caller asking how much budget is left must not be told there is some when it
    /// cannot be read.
    pub fn remaining(&self) -> u64 {
        self.remaining_at(SystemTime::now())
    }

    /// [`Self::remaining`] against an explicit clock, for tests and replays.
    pub fn remaining_at(&self, now: SystemTime) -> u64 {
        let day = provider_day(now);
        match self.read_under_lock() {
            Ok(counter) => self.limit.saturating_sub(spent_on(counter, day)),
            Err(_) => 0,
        }
    }

    /// Reserves one request against the current provider day.
    ///
    /// Returns [`ReserveError::Exhausted`] when the allowance is spent, in which case
    /// nothing was consumed, or [`ReserveError::Unusable`] when the counter cannot be
    /// trusted, in which case nothing was sent.
    pub fn try_reserve(&self) -> Result<(), ReserveError> {
        self.try_reserve_at(SystemTime::now())
    }

    /// [`Self::try_reserve`] against an explicit clock, for tests and for replays.
    pub fn try_reserve_at(&self, now: SystemTime) -> Result<(), ReserveError> {
        let day = provider_day(now);
        // The lock is held across the read, the decision and the write. Without it two
        // processes sharing a state directory would each read the same count, each decide
        // it may spend, and together exceed the budget by one request per race.
        let lock = self.lock()?;
        let counter = self.read()?;
        let spent = spent_on(counter, day);
        if spent >= self.limit {
            drop(lock);
            return Err(ReserveError::Exhausted);
        }
        let next = Counter {
            day,
            spent: spent + 1,
        };
        // Persist before sending. A counter that disagrees with the provider is the
        // failure this exists to prevent, so a failed write must also undo the spend.
        self.persist(next)?;
        drop(lock);
        Ok(())
    }

    /// The counter's lock file, beside the counter it guards.
    fn lock_path(&self) -> PathBuf {
        self.path.with_extension("json.lock")
    }

    /// Takes the exclusive advisory lock for this allowance's counter.
    fn lock(&self) -> Result<File, ReserveError> {
        self.create_parent()?;
        let path = self.lock_path();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                ReserveError::Unusable(format!("{} cannot be opened: {error}", path.display()))
            })?;
        file.lock().map_err(|error| {
            ReserveError::Unusable(format!("{} cannot be locked: {error}", path.display()))
        })?;
        Ok(file)
    }

    /// Creates the directory holding the counter, when there is one.
    fn create_parent(&self) -> Result<(), ReserveError> {
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|error| {
                ReserveError::Unusable(format!(
                    "the directory for {} cannot be created ({}): {error}",
                    self.path.display(),
                    parent.display()
                ))
            })?;
        }
        Ok(())
    }

    /// Reads the counter as it stands on disk.
    fn read(&self) -> Result<Counter, ReserveError> {
        match fs::read(&self.path) {
            Ok(bytes) => parse_counter(&bytes).map_err(|reason| {
                ReserveError::Unusable(format!("{} is unreadable: {reason}", self.path.display()))
            }),
            // A missing file is a fresh allowance, not a fault: the first reservation of a
            // deployment's life creates it.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(Counter { day: 0, spent: 0 })
            }
            Err(error) => Err(ReserveError::Unusable(format!(
                "{} cannot be read: {error}",
                self.path.display()
            ))),
        }
    }

    /// [`Self::read`] for callers that do not intend to write.
    fn read_under_lock(&self) -> Result<Counter, ReserveError> {
        let lock = self.lock()?;
        let counter = self.read();
        drop(lock);
        counter
    }

    /// Atomically and durably replaces the counter file.
    fn persist(&self, counter: Counter) -> Result<(), ReserveError> {
        self.create_parent()?;
        // A body the operator can read without this code: the numbers are the record.
        let body = format!(
            "{{\"day\":{},\"spent\":{},\"limit\":{},\"group\":{},\"model\":{},\"basis\":\"local_budget\",\"provider_remaining\":null}}\n",
            counter.day,
            counter.spent,
            self.limit,
            serde_json::Value::String(self.group.clone()),
            serde_json::Value::String(self.model.to_string()),
        );
        // Same-directory temporary name so the rename is a replace, never a cross-device
        // copy that could leave a truncated counter behind.
        let temporary = self.path.with_extension("json.tmp");
        let mut file = File::create(&temporary).map_err(|error| {
            ReserveError::Unusable(format!(
                "{} cannot be written: {error}",
                temporary.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|error| {
            ReserveError::Unusable(format!(
                "{} cannot be written: {error}",
                temporary.display()
            ))
        })?;
        // The reservation must survive a crash of this process: the request it was granted
        // for may already have left, and a lost reservation is the one failure mode that
        // silently overspends the provider's budget.
        file.sync_all().map_err(|error| {
            ReserveError::Unusable(format!(
                "{} cannot be flushed: {error}",
                temporary.display()
            ))
        })?;
        drop(file);
        fs::rename(&temporary, &self.path).map_err(|error| {
            let _ = fs::remove_file(&temporary);
            ReserveError::Unusable(format!(
                "{} cannot be replaced: {error}",
                self.path.display()
            ))
        })?;
        self.sync_parent()
    }

    /// Flushes the directory entry, so the rename itself is durable.
    fn sync_parent(&self) -> Result<(), ReserveError> {
        let Some(parent) = self.path.parent() else {
            return Ok(());
        };
        if parent.as_os_str().is_empty() {
            return Ok(());
        }
        let directory = File::open(parent).map_err(|error| {
            ReserveError::Unusable(format!("{} cannot be opened: {error}", parent.display()))
        })?;
        directory.sync_all().map_err(|error| {
            ReserveError::Unusable(format!("{} cannot be flushed: {error}", parent.display()))
        })
    }
}

/// Requests already spent on `day`, given a counter that may belong to an earlier day.
fn spent_on(counter: Counter, day: u64) -> u64 {
    if counter.day == day { counter.spent } else { 0 }
}

/// Reads `day` and `spent` from a counter file.
fn parse_counter(bytes: &[u8]) -> Result<Counter, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let day = value
        .get("day")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "missing or non-numeric \"day\"".to_string())?;
    let spent = value
        .get("spent")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "missing or non-numeric \"spent\"".to_string())?;
    Ok(Counter { day, spent })
}

/// The counter file for one (quota group, model) pair under `directory`.
///
/// The name keeps the pair legible and appends a stable hash of the *exact* pair, so two
/// groups whose names differ only in characters the legible part replaces cannot share a
/// counter. A shared counter would let one group spend another's budget while each
/// believed it was within its own.
pub fn allowance_path(directory: &Path, group: &str, model: &str) -> PathBuf {
    let mut key = String::with_capacity(group.len() + model.len() + 17);
    key.push_str(group);
    key.push('\u{1f}');
    key.push_str(model);
    directory.join(format!(
        "{}-{}-{:016x}.json",
        legible(group),
        legible(model),
        fnv1a(&key)
    ))
}

/// `value` reduced to characters that are safe in a path component.
///
/// Only for legibility; identity comes from the hash beside it.
fn legible(value: &str) -> String {
    let reduced: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect();
    // Bound the component: a long slug would push the name past the filesystem limit.
    reduced.chars().take(48).collect()
}

/// 64-bit FNV-1a.
///
/// A hash function whose output is fixed by its definition, so a counter file keeps the
/// same name across builds and Rust versions. `std`'s `DefaultHasher` is not specified to
/// be stable, which would silently orphan every existing counter on a toolchain bump.
fn fnv1a(value: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Length of one pacing window.
///
/// A minute is what the providers that publish window limits publish them over. It is a
/// sliding window, not a fixed clock minute: a fixed boundary would let a deployment send
/// a full window either side of it, doubling the burst the limit exists to prevent.
const WINDOW: Duration = Duration::from_secs(60);

/// A provider's per-minute pacing limits for one model.
///
/// Both axes are optional and independent: a model may cap requests, input tokens, or both.
/// Tokens are counted on the request side only — output is not known before the call, and
/// the provider's own input budget is what this mirrors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowLimits {
    /// Requests permitted in any rolling minute.
    pub requests_per_minute: Option<u64>,
    /// Input tokens permitted in any rolling minute.
    pub input_tokens_per_minute: Option<u64>,
}

impl WindowLimits {
    /// No window limits: nothing is recorded and nothing is refused.
    pub const fn unlimited() -> Self {
        Self {
            requests_per_minute: None,
            input_tokens_per_minute: None,
        }
    }

    /// Whether this declares no limit at all.
    pub fn is_unlimited(&self) -> bool {
        self.requests_per_minute.is_none() && self.input_tokens_per_minute.is_none()
    }
}

/// Identity of one paced budget.
///
/// The provider counts per project and physical model, so the key is exactly that and not
/// the alias: `smart-nt`, `smart_gemini` and a direct leaf address to the same model must
/// draw on one window, or each alias would be handed the full allowance. `feature` keeps a
/// second family on the same model (an embedding call on a chat model, say) from spending
/// the chat window unless the deployment says they share it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GovernorKey {
    group: String,
    model: ModelId,
    feature: &'static str,
}

impl GovernorKey {
    /// The paced budget for `model` in `group`, on the `feature` axis.
    pub fn new(group: impl Into<String>, model: ModelId, feature: &'static str) -> Self {
        Self {
            group: group.into(),
            model,
            feature,
        }
    }

    /// The quota group (provider project) this budget belongs to.
    pub fn group(&self) -> &str {
        &self.group
    }

    /// The physical model this budget belongs to.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// The feature axis this budget belongs to.
    pub fn feature(&self) -> &'static str {
        self.feature
    }
}

/// A granted minute-window reservation.
///
/// Returned so the caller can release it if the request never reaches the provider — the
/// daily allowance refusing afterwards is the case that matters, because spending a window
/// slot for a request that was never sent would turn a paced-out day into a burst.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PacingTicket {
    key: GovernorKey,
    sequence: u64,
}

/// Which limit refused a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacingScope {
    /// The rolling request count.
    Requests,
    /// The rolling input-token count.
    InputTokens,
}

impl PacingScope {
    /// Stable label for logs and error bodies.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Requests => "requests_per_minute",
            Self::InputTokens => "input_tokens_per_minute",
        }
    }
}

/// A window with no room right now, and when it next will have room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Throttled {
    /// The limit that refused.
    pub scope: PacingScope,
    /// How long until enough of the window has expired to admit this request.
    pub retry_after: Duration,
}

/// Why a minute-window reservation could not be granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacingError {
    /// The window is full now but refills shortly: route elsewhere or report the interval.
    Throttled(Throttled),
    /// The request alone is larger than a whole window's input budget, so waiting cannot
    /// help. A deployment that declares `input_tokens_per_minute` must also declare a
    /// per-request ceiling at or below it; configuration validation enforces that, and this
    /// variant is the defensive answer if a call arrives over it anyway.
    ExceedsWindow {
        /// The request's estimated input tokens.
        needed: u64,
        /// The whole window's input budget.
        limit: u64,
    },
}

/// Tracks rolling per-minute spend per [`GovernorKey`].
///
/// One instance is shared by every backend of a deployment, which is what makes the
/// accounting cross-alias: two aliases of one physical model resolve to the same key and so
/// to the same ledger. The state is in-process and deliberately not persisted — a window is
/// a minute wide, and a restart is not a thing a provider's per-minute counter refunds, so
/// the only cost of forgetting is that a restart may under-count for the remainder of the
/// current window. Persisting it would add a lock and a disk write to every request to
/// protect a value that expires in under a minute.
#[derive(Debug, Default)]
pub struct QuotaGovernor {
    ledgers: Mutex<HashMap<GovernorKey, Ledger>>,
}

#[derive(Debug, Default)]
struct Ledger {
    spends: VecDeque<Spend>,
    next_sequence: u64,
}

#[derive(Clone, Copy, Debug)]
struct Spend {
    sequence: u64,
    at: SystemTime,
    tokens: u64,
}

impl QuotaGovernor {
    /// An empty governor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves one request of `input_tokens` against `key`'s window, on the real clock.
    pub fn reserve(
        &self,
        key: &GovernorKey,
        limits: WindowLimits,
        input_tokens: u64,
    ) -> Result<Option<PacingTicket>, PacingError> {
        self.reserve_at(key, limits, input_tokens, SystemTime::now())
    }

    /// [`Self::reserve`] against an explicit clock, for tests and replays.
    pub fn reserve_at(
        &self,
        key: &GovernorKey,
        limits: WindowLimits,
        input_tokens: u64,
        now: SystemTime,
    ) -> Result<Option<PacingTicket>, PacingError> {
        if limits.is_unlimited() {
            return Ok(None);
        }
        if let Some(limit) = limits.input_tokens_per_minute
            && input_tokens > limit
        {
            return Err(PacingError::ExceedsWindow {
                needed: input_tokens,
                limit,
            });
        }
        let mut ledgers = self
            .ledgers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let ledger = ledgers.entry(key.clone()).or_default();
        let cutoff = now.checked_sub(WINDOW).unwrap_or(UNIX_EPOCH);
        while ledger.spends.front().is_some_and(|spend| spend.at <= cutoff) {
            ledger.spends.pop_front();
        }

        // Both axes are evaluated before anything is recorded: a request refused for one
        // reason must not have consumed room on the other.
        let mut refused: Option<Throttled> = None;
        if let Some(limit) = limits.requests_per_minute {
            let used = u64::try_from(ledger.spends.len()).unwrap_or(u64::MAX);
            if used >= limit
                && let Some(front) = ledger.spends.front()
            {
                refused = Some(Throttled {
                    scope: PacingScope::Requests,
                    retry_after: (front.at + WINDOW).duration_since(now).unwrap_or_default(),
                });
            }
        }
        if let Some(limit) = limits.input_tokens_per_minute {
            let outstanding = ledger.spends.iter().fold(0_u64, |total, spend| {
                total.saturating_add(spend.tokens)
            });
            if outstanding.saturating_add(input_tokens) > limit {
                let mut remaining = outstanding;
                let mut retry_after = WINDOW;
                for spend in &ledger.spends {
                    remaining = remaining.saturating_sub(spend.tokens);
                    if remaining.saturating_add(input_tokens) <= limit {
                        retry_after = (spend.at + WINDOW).duration_since(now).unwrap_or_default();
                        break;
                    }
                }
                // When both axes refuse, report the one that clears last: that is the wait
                // the caller actually faces.
                if refused.is_none_or(|existing| retry_after > existing.retry_after) {
                    refused = Some(Throttled {
                        scope: PacingScope::InputTokens,
                        retry_after,
                    });
                }
            }
        }
        if let Some(throttled) = refused {
            return Err(PacingError::Throttled(throttled));
        }

        ledger.next_sequence += 1;
        let sequence = ledger.next_sequence;
        ledger.spends.push_back(Spend {
            sequence,
            at: now,
            tokens: input_tokens,
        });
        Ok(Some(PacingTicket {
            key: key.clone(),
            sequence,
        }))
    }

    /// Returns a granted reservation that never reached the provider.
    ///
    /// Only correct before the request is sent. An attempt that left the process keeps its
    /// slot, for the same reason a daily reservation is never refunded: the provider may
    /// have counted it.
    pub fn release(&self, ticket: &PacingTicket) {
        let mut ledgers = self
            .ledgers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(ledger) = ledgers.get_mut(&ticket.key) {
            ledger
                .spends
                .retain(|spend| spend.sequence != ticket.sequence);
        }
    }

    /// The window's current spend for `key`, for reporting and tests.
    pub fn used(&self, key: &GovernorKey, now: SystemTime) -> (u64, u64) {
        let mut ledgers = self
            .ledgers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(ledger) = ledgers.get_mut(key) else {
            return (0, 0);
        };
        let cutoff = now.checked_sub(WINDOW).unwrap_or(UNIX_EPOCH);
        while ledger.spends.front().is_some_and(|spend| spend.at <= cutoff) {
            ledger.spends.pop_front();
        }
        let requests = u64::try_from(ledger.spends.len()).unwrap_or(u64::MAX);
        let tokens = ledger
            .spends
            .iter()
            .fold(0_u64, |total, spend| total.saturating_add(spend.tokens));
        (requests, tokens)
    }
}

/// One target's paced budget: the shared governor, the key it bills to, and the limits.
///
/// Carried by the backend because one HTTP attempt is the granularity the provider counts
/// in, and the send path is the one place every candidate passes through.
#[derive(Debug)]
pub struct PacedQuota {
    governor: std::sync::Arc<QuotaGovernor>,
    key: GovernorKey,
    limits: WindowLimits,
}

impl PacedQuota {
    /// Binds `key` and `limits` to a shared governor.
    pub fn new(governor: std::sync::Arc<QuotaGovernor>, key: GovernorKey, limits: WindowLimits) -> Self {
        Self {
            governor,
            key,
            limits,
        }
    }

    /// The declared limits.
    pub fn limits(&self) -> WindowLimits {
        self.limits
    }

    /// The budget this target draws on.
    pub fn key(&self) -> &GovernorKey {
        &self.key
    }

    /// Reserves one request against the window.
    pub fn reserve(&self, input_tokens: u64) -> Result<Option<PacingTicket>, PacingError> {
        self.governor.reserve(&self.key, self.limits, input_tokens)
    }

    /// Releases a reservation that never reached the provider.
    pub fn release(&self, ticket: &PacingTicket) {
        self.governor.release(ticket);
    }
}

/// A conservative estimate of a request body's input tokens, for pacing only.
///
/// The governor needs a number before the call, and the only body it can see is the exact
/// one about to be sent. Counting characters over every string in the body and dividing by
/// four is the standard approximation; it is deliberately crude, it is never reported as a
/// measurement, and it over-counts JSON structure rather than under-counting text.
///
/// Strings that are themselves inline binary payloads (`data:` URIs) are skipped: their
/// base64 length says nothing about the provider's image accounting, and counting it would
/// refuse every vision request against a text budget. Those requests are paced by the
/// request axis alone, which is what a per-request ceiling is for.
pub fn estimate_input_tokens(body: &serde_json::Value) -> u64 {
    fn count(value: &serde_json::Value, characters: &mut u64) {
        match value {
            serde_json::Value::String(text) => {
                if !text.starts_with("data:") {
                    *characters += u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    count(item, characters);
                }
            }
            serde_json::Value::Object(fields) => {
                for field in fields.values() {
                    count(field, characters);
                }
            }
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_) => {}
        }
    }
    let mut characters = 0_u64;
    count(body, &mut characters);
    (characters / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A directory that no other test in this process shares.
    fn scratch(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            label
        );
        let directory = std::env::temp_dir()
            .join("switchyard-allowance-tests")
            .join(unique);
        let _ = fs::remove_dir_all(&directory);
        directory
    }

    fn allowance(directory: &Path, limit: u64) -> DailyRequestAllowance {
        DailyRequestAllowance::new(
            "google-free",
            ModelId::from("google/gemini-3.8-flash"),
            limit,
            allowance_path(directory, "google-free", "google/gemini-3.8-flash"),
        )
    }

    /// Seconds since the epoch for a UTC instant, for readable test clocks.
    fn utc(year: i64, month: i64, day: i64, hour: i64, minute: i64) -> SystemTime {
        let seconds = days_from_civil(year, month, day) * DAY_SECONDS + hour * 3_600 + minute * 60;
        UNIX_EPOCH + std::time::Duration::from_secs(seconds.unsigned_abs())
    }

    /// The counter starts empty, counts each reservation, and stops exactly at the limit.
    #[test]
    fn reservations_stop_exactly_at_the_limit() {
        let directory = scratch("limit");
        let allowance = allowance(&directory, 3);
        assert_eq!(allowance.remaining(), 3);
        for expected in [2, 1, 0] {
            allowance.try_reserve().expect("within the limit");
            assert_eq!(allowance.remaining(), expected);
        }
        assert!(matches!(
            allowance.try_reserve(),
            Err(ReserveError::Exhausted)
        ));
        // A refused reservation consumes nothing.
        assert_eq!(allowance.remaining(), 0);
    }

    /// A spend survives a restart: a new instance over the same file continues the count.
    #[test]
    fn the_count_survives_a_restart() {
        let directory = scratch("restart");
        allowance(&directory, 5).try_reserve().expect("first");
        let restarted = allowance(&directory, 5);
        assert_eq!(restarted.remaining(), 4);
    }

    /// Two instances over one file cannot both spend the last request, however they race.
    ///
    /// Each thread opens its own lock file handle, which is what two processes do, so this
    /// fails if the read-modify-write is not serialised: without the lock the threads read
    /// the same count and together exceed the budget.
    #[test]
    fn concurrent_writers_cannot_overspend() {
        let directory = scratch("concurrent");
        let limit = 20_u64;
        let attempts_per_thread = 10_usize;
        let granted = std::sync::Arc::new(AtomicU64::new(0));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let directory = directory.clone();
            let granted = granted.clone();
            threads.push(std::thread::spawn(move || {
                for _ in 0..attempts_per_thread {
                    let allowance = allowance(&directory, limit);
                    if allowance.try_reserve().is_ok() {
                        granted.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }
        for thread in threads {
            thread.join().expect("thread");
        }
        assert_eq!(granted.load(Ordering::Relaxed), limit);
        // And the file agrees with what was granted.
        assert_eq!(allowance(&directory, limit).remaining(), 0);
    }

    /// The grant is durable: it is on disk, not only in memory, before the send happens.
    #[test]
    fn a_grant_is_flushed_to_disk() {
        let directory = scratch("durable");
        let counter = allowance(&directory, 5);
        counter.try_reserve().expect("first");
        let bytes = fs::read(&counter.path).expect("counter file exists on disk");
        let parsed = parse_counter(&bytes).expect("counter parses");
        assert_eq!(parsed.spent, 1);
        assert_eq!(parsed.day, provider_day(SystemTime::now()));
    }

    /// Crossing the provider's midnight reopens the allowance without operator action.
    #[test]
    fn a_new_provider_day_reopens_the_allowance() {
        let directory = scratch("rollover");
        let allowance = allowance(&directory, 1);
        // 2026-11-20T20:00:00Z — midday Pacific on 2026-11-20.
        let midday = utc(2026, 11, 20, 20, 0);
        allowance.try_reserve_at(midday).expect("within the limit");
        assert!(matches!(
            allowance.try_reserve_at(midday),
            Err(ReserveError::Exhausted)
        ));
        let next_day = utc(2026, 11, 21, 20, 0);
        assert_ne!(provider_day(midday), provider_day(next_day));
        allowance
            .try_reserve_at(next_day)
            .expect("a new provider day reopens the allowance");
        assert_eq!(allowance.remaining_at(next_day), 0);
    }

    /// In standard time the day boundary is midnight Pacific = 08:00 UTC.
    #[test]
    fn the_boundary_is_midnight_pacific_in_standard_time() {
        // 2026-12-01 is outside daylight time, so its first instant is 08:00Z.
        let boundary = utc(2026, 12, 1, 8, 0);
        let last_instant_of_the_previous_day = boundary - std::time::Duration::from_secs(1);
        assert_eq!(
            provider_day_label(provider_day(last_instant_of_the_previous_day)),
            "2026-11-30"
        );
        assert_eq!(provider_day_label(provider_day(boundary)), "2026-12-01");
        assert_eq!(
            provider_day(boundary),
            provider_day(last_instant_of_the_previous_day) + 1
        );
        // 23:59Z on 30 November is still 15:59 Pacific on the same day.
        assert_eq!(
            provider_day(utc(2026, 11, 30, 23, 59)),
            provider_day(last_instant_of_the_previous_day)
        );
        // The day begins at 08:00Z, so 07:59Z still belongs to yesterday.
        assert_eq!(
            provider_day(utc(2026, 12, 1, 7, 59)),
            provider_day(last_instant_of_the_previous_day)
        );
    }

    /// In daylight time the day boundary is midnight PDT = 07:00 UTC.
    ///
    /// This is the case a fixed UTC-8 offset gets wrong: it would keep counting the
    /// previous provider day until 08:00Z, an hour after the provider had reset.
    #[test]
    fn the_boundary_is_midnight_pacific_in_daylight_time() {
        // 2026-07-15 is inside daylight time: midnight PDT is 07:00Z.
        let boundary = utc(2026, 7, 15, 7, 0);
        assert_eq!(
            provider_day_label(provider_day(boundary - std::time::Duration::from_secs(1))),
            "2026-07-14"
        );
        assert_eq!(provider_day_label(provider_day(boundary)), "2026-07-15");
        // The fixed-offset implementation would have called 07:30Z the previous day.
        assert_ne!(
            provider_day(utc(2026, 7, 15, 7, 30)),
            provider_day(utc(2026, 7, 15, 6, 30))
        );
        // And 08:00Z is mid-day, not a boundary.
        assert_eq!(
            provider_day(utc(2026, 7, 15, 8, 0)),
            provider_day(utc(2026, 7, 15, 9, 0))
        );
    }

    /// The two transition days roll over exactly once, at the correct instant.
    #[test]
    fn the_transition_days_roll_over_once() {
        // 2026: daylight time starts on the second Sunday in March (the 8th) at 10:00Z
        // and ends on the first Sunday in November (the 1st) at 09:00Z.
        let (start, end) = daylight_bounds(2026);
        assert_eq!(
            start,
            utc(2026, 3, 8, 10, 0)
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        );
        assert_eq!(
            end,
            utc(2026, 11, 1, 9, 0)
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        );

        // Before the spring transition, midnight is 08:00Z; after it, midnight is 07:00Z.
        assert_eq!(pacific_offset_seconds(start - 1), PACIFIC_STANDARD_SECONDS);
        assert_eq!(pacific_offset_seconds(start), PACIFIC_DAYLIGHT_SECONDS);
        // 2026-03-08 midnight PST is 08:00Z, and the day after is midnight PDT = 07:00Z.
        assert_eq!(
            provider_day_label(provider_day(utc(2026, 3, 8, 8, 0))),
            "2026-03-08"
        );
        assert_eq!(
            provider_day_label(provider_day(utc(2026, 3, 9, 7, 0))),
            "2026-03-09"
        );
        // Once per local day either side of the transition.
        assert_eq!(
            provider_day(utc(2026, 3, 9, 7, 0)),
            provider_day(utc(2026, 3, 8, 8, 0)) + 1
        );

        // Before the autumn transition, midnight is 07:00Z; after it, midnight is 08:00Z.
        assert_eq!(pacific_offset_seconds(end - 1), PACIFIC_DAYLIGHT_SECONDS);
        assert_eq!(pacific_offset_seconds(end), PACIFIC_STANDARD_SECONDS);
        assert_eq!(
            provider_day_label(provider_day(utc(2026, 11, 1, 7, 0))),
            "2026-11-01"
        );
        assert_eq!(
            provider_day_label(provider_day(utc(2026, 11, 2, 8, 0))),
            "2026-11-02"
        );
        assert_eq!(
            provider_day(utc(2026, 11, 2, 8, 0)),
            provider_day(utc(2026, 11, 1, 7, 0)) + 1
        );

        // 2027's boundaries follow the rule, not the previous year's dates.
        let (start_2027, end_2027) = daylight_bounds(2027);
        assert_eq!(
            start_2027,
            utc(2027, 3, 14, 10, 0)
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        );
        assert_eq!(
            end_2027,
            utc(2027, 11, 7, 9, 0)
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        );
    }

    /// Exactly one boundary per civil day across a whole year, including both transitions.
    #[test]
    fn every_day_of_the_year_has_one_boundary() {
        let start = utc(2026, 1, 1, 12, 0);
        let mut previous = provider_day(start);
        let mut boundaries = 0;
        for hour in 1..=(365 * 24) {
            let now = start + std::time::Duration::from_secs(hour * 3_600);
            let day = provider_day(now);
            assert!(day >= previous, "the day counter never moves backwards");
            if day != previous {
                boundaries += 1;
                previous = day;
            }
        }
        assert_eq!(boundaries, 365, "one boundary per civil day of 2026");
    }

    /// A UTC midnight is not a provider midnight, so it does not reopen the allowance.
    #[test]
    fn a_utc_midnight_is_not_a_provider_midnight() {
        let utc_midnight = utc(2026, 7, 15, 0, 0);
        assert_eq!(
            provider_day(utc_midnight + std::time::Duration::from_secs(3_600)),
            provider_day(utc_midnight)
        );
    }

    /// The clearance header's day format and the counter's day agree by construction.
    #[test]
    fn the_day_label_round_trips_through_civil_arithmetic() {
        for day in [0_u64, 20_000, 20_800, 20_830] {
            let label = provider_day_label(day);
            let mut parts = label.split('-');
            let year: i64 = parts.next().unwrap().parse().unwrap();
            let month: i64 = parts.next().unwrap().parse().unwrap();
            let date: i64 = parts.next().unwrap().parse().unwrap();
            assert_eq!(days_from_civil(year, month, date), day as i64, "{label}");
        }
    }

    /// Unreadable state refuses instead of granting budget nobody is counting.
    #[test]
    fn corrupt_state_fails_closed() {
        let directory = scratch("corrupt");
        fs::create_dir_all(&directory).expect("scratch directory");
        let counter = allowance(&directory, 10);
        fs::write(counter.path.clone(), b"not json").expect("write corruption");
        assert!(matches!(
            counter.try_reserve(),
            Err(ReserveError::Unusable(_))
        ));
        // `remaining` reports nothing left rather than claiming the full limit.
        let fresh = allowance(&directory, 10);
        assert_eq!(fresh.remaining(), 0);
    }

    /// Two groups whose names collapse to the same legible slug keep separate counters.
    #[test]
    fn distinct_pairs_never_share_a_counter() {
        let directory = scratch("collision");
        let first = allowance_path(&directory, "a/b", "m");
        let second = allowance_path(&directory, "a_b", "m");
        assert_ne!(first, second);
        // The same pair resolves to the same path every time.
        assert_eq!(first, allowance_path(&directory, "a/b", "m"));
    }

    // ---- Per-minute pacing -------------------------------------------------

    fn governor_key(model: &str) -> GovernorKey {
        GovernorKey::new("google-free", ModelId::from(model), "generate")
    }

    fn minute_limits(requests: Option<u64>, tokens: Option<u64>) -> WindowLimits {
        WindowLimits {
            requests_per_minute: requests,
            input_tokens_per_minute: tokens,
        }
    }

    fn instant(offset: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_800_000_000 + offset)
    }

    /// The case the field exists for: Gemma's free window is 16K input tokens per minute,
    /// so two ten-thousand-token prompts cannot both depart inside one window.
    #[test]
    fn two_ten_thousand_token_prompts_cannot_share_a_sixteen_thousand_window() {
        let governor = QuotaGovernor::new();
        let key = governor_key("google/gemma-4-26b-a4b-it");
        let limits = minute_limits(None, Some(16_000));
        governor
            .reserve_at(&key, limits, 10_000, instant(0))
            .expect("the first prompt fits");
        let refused = governor
            .reserve_at(&key, limits, 10_000, instant(5))
            .expect_err("the second cannot fit");
        assert_eq!(
            refused,
            PacingError::Throttled(Throttled {
                scope: PacingScope::InputTokens,
                // The first reservation leaves the window sixty seconds after it was made.
                retry_after: Duration::from_secs(55),
            })
        );
    }

    /// A burst of small requests is refused by the request axis even though the token
    /// budget would have held them.
    #[test]
    fn the_thirty_first_request_is_refused_when_the_window_admits_thirty() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(Some(30), None);
        for second in 0..30 {
            governor
                .reserve_at(&key, limits, 1, instant(second))
                .expect("admits thirty");
        }
        assert_eq!(
            governor
                .reserve_at(&key, limits, 1, instant(30))
                .expect_err("the window is full"),
            PacingError::Throttled(Throttled {
                scope: PacingScope::Requests,
                retry_after: Duration::from_secs(30),
            })
        );
    }

    /// The two axes are separate resources: room on one says nothing about the other.
    #[test]
    fn the_request_and_token_axes_are_independent() {
        let tokens_full = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(Some(100), Some(1_000));
        tokens_full
            .reserve_at(&key, limits, 900, instant(0))
            .expect("fits");
        assert!(matches!(
            tokens_full.reserve_at(&key, limits, 200, instant(1)),
            Err(PacingError::Throttled(Throttled {
                scope: PacingScope::InputTokens,
                ..
            }))
        ));

        let requests_full = QuotaGovernor::new();
        requests_full
            .reserve_at(&key, limits, 1, instant(0))
            .expect("fits");
        // One request of one token leaves plenty of token room, so the request count is
        // what refuses.
        assert!(matches!(
            requests_full.reserve_at(
                &key,
                minute_limits(Some(1), Some(1_000)),
                1,
                instant(1)
            ),
            Err(PacingError::Throttled(Throttled {
                scope: PacingScope::Requests,
                ..
            }))
        ));
    }

    /// The window slides: it opens again sixty seconds after the requests that filled it.
    #[test]
    fn a_window_refills_sixty_seconds_after_the_requests_that_filled_it() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(Some(1), Some(16_000));
        governor
            .reserve_at(&key, limits, 100, instant(0))
            .expect("fits");
        assert!(governor.reserve_at(&key, limits, 100, instant(59)).is_err());
        governor
            .reserve_at(&key, limits, 100, instant(60))
            .expect("the window rolled over");
    }

    /// One physical model has one window however many names it is served under; a different
    /// model, or the same model in another project, has its own.
    #[test]
    fn aliases_of_one_physical_model_share_one_window() {
        let governor = QuotaGovernor::new();
        let limits = minute_limits(Some(1), None);
        let direct = GovernorKey::new(
            "google-free",
            ModelId::from("google/gemma-4-26b-a4b-it"),
            "generate",
        );
        let alias = GovernorKey::new(
            "google-free",
            ModelId::from("google/gemma-4-26b-a4b-it"),
            "generate",
        );
        governor
            .reserve_at(&direct, limits, 10, instant(0))
            .expect("fits");
        assert!(matches!(
            governor.reserve_at(&alias, limits, 10, instant(1)),
            Err(PacingError::Throttled(_))
        ));

        let sibling = GovernorKey::new(
            "google-free",
            ModelId::from("google/gemma-4-31b-it"),
            "generate",
        );
        governor
            .reserve_at(&sibling, limits, 10, instant(1))
            .expect("a different model has its own window");

        let other_project = GovernorKey::new(
            "google-second-project",
            ModelId::from("google/gemma-4-26b-a4b-it"),
            "generate",
        );
        governor
            .reserve_at(&other_project, limits, 10, instant(1))
            .expect("another project has its own window");
    }

    /// A reservation returned before the request left gives its room back.
    #[test]
    fn a_released_reservation_gives_the_room_back() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(Some(1), Some(16_000));
        let ticket = governor
            .reserve_at(&key, limits, 5_000, instant(0))
            .expect("fits")
            .expect("a limited key yields a ticket");
        assert_eq!(governor.used(&key, instant(0)), (1, 5_000));
        governor.release(&ticket);
        assert_eq!(governor.used(&key, instant(0)), (0, 0));
        governor
            .reserve_at(&key, limits, 5_000, instant(0))
            .expect("the room was returned");
    }

    /// A request larger than a whole window can never be admitted, so waiting is not the
    /// answer and is not offered.
    #[test]
    fn a_request_larger_than_a_whole_window_is_refused_rather_than_queued() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        assert_eq!(
            governor.reserve_at(&key, minute_limits(None, Some(16_000)), 20_000, instant(0)),
            Err(PacingError::ExceedsWindow {
                needed: 20_000,
                limit: 16_000
            })
        );
        assert_eq!(governor.used(&key, instant(0)), (0, 0));
    }

    /// A target that declares no window records nothing, so an undeclared deployment
    /// behaves exactly as it did before the field existed.
    #[test]
    fn no_declared_window_records_nothing() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        assert_eq!(
            governor.reserve_at(&key, WindowLimits::unlimited(), 1_000_000, instant(0)),
            Ok(None)
        );
        assert_eq!(governor.used(&key, instant(0)), (0, 0));
    }

    /// The estimate counts message text and never treats a base64 payload as prompt text.
    #[test]
    fn the_token_estimate_counts_text_and_ignores_inline_binary() {
        let plain = serde_json::json!({
            "messages": [{"role": "user", "content": "a".repeat(4_000)}]
        });
        let with_image = serde_json::json!({
            "messages": [{"role": "user", "content": "a".repeat(4_000)}],
            "image_url": format!("data:image/png;base64,{}", "A".repeat(40_000)),
        });
        // 4,000 characters of content plus the four-character role, divided by four.
        assert_eq!(estimate_input_tokens(&plain), 1_001);
        assert_eq!(estimate_input_tokens(&with_image), 1_001);
        // An empty body is never zero tokens: it is still a request.
        assert_eq!(estimate_input_tokens(&serde_json::json!({})), 1);
    }
}
