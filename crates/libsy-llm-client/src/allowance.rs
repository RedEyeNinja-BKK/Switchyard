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

/// How long a provider-reported request count stays actionable.
///
/// Two collection intervals, not one. The collector in this deployment reads accounts
/// every 300 s, so a tolerance equal to that interval would expire a good reading in the
/// window between the instant it ages out and the instant the next one is written --
/// including whenever a single round is slow or fails. The reading would then be discarded
/// for being stale, which is the one thing the three-state design exists to avoid.
///
/// It stays bounded rather than generous: a collector that has been dead for ten minutes
/// should stop being the authority, because the provider's day may have rolled over and
/// nothing here would know. Two intervals tolerate exactly one missed round.
///
/// See [`DailyRequestAllowance::provider_tolerance`] for the state this produces and why
/// it is always reported rather than assumed.
const DEFAULT_PROVIDER_TOLERANCE: Duration = Duration::from_secs(600);

/// How long a confirmed exhaustion keeps refusing after readings stop arriving.
///
/// A reading that says the account is empty is the only thing standing between this lane
/// and a retry storm: once it ages out the counter falls back to its local budget, which
/// still looks unspent, and every retry earns another provider 429. The retention is
/// therefore kept for a bounded time after the reading's own freshness ends.
///
/// Bounded, because the opposite failure is a wall that outlives the outage: a collector
/// that dies while the account is empty must not refuse the lane until someone notices.
/// Two collection intervals is long enough to ride out a missing round and short enough
/// that a recovered account serves again within minutes of the provider restoring it.
const EXHAUSTION_RETENTION: Duration = Duration::from_secs(600);

/// How far ahead of the local clock a reading may be stamped and still be believed.
///
/// A reading is stamped by the collector on this host, so a genuine one is never future.
/// The allowance exists for clock adjustment between the two threads and nothing else;
/// beyond it the reading describes an instant that has not happened.
const OBSERVATION_SKEW: Duration = Duration::from_secs(60);

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

/// The UTC civil day number containing `now`.
///
/// A second day boundary exists because provider families do not agree on when "today"
/// starts. Google's free tier turns over at midnight Pacific; OpenRouter's free-model
/// counter turns over at midnight UTC. Counting one on the other's clock either spends
/// the new day's budget a few hours early or refuses traffic the provider has already
/// restored, so the boundary belongs to the allowance rather than being a global
/// constant.
pub fn utc_day(now: SystemTime) -> u64 {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let seconds = i64::try_from(seconds).unwrap_or(i64::MAX);
    u64::try_from(seconds.div_euclid(DAY_SECONDS)).unwrap_or(0)
}

/// Which midnight a daily counter turns over on.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DayBoundary {
    /// Midnight Pacific — Google's AI Studio free tier. The historical default, so a
    /// deployment that names no boundary keeps exactly its previous behaviour.
    #[default]
    Pacific,
    /// Midnight UTC — OpenRouter's free-model daily counter.
    Utc,
}

impl DayBoundary {
    /// The civil day number containing `now` on this boundary's clock.
    pub fn day(self, now: SystemTime) -> u64 {
        match self {
            DayBoundary::Pacific => provider_day(now),
            DayBoundary::Utc => utc_day(now),
        }
    }

    /// The `YYYY-MM-DD` date of `day` on this boundary's clock.
    ///
    /// Both boundaries count whole civil days from the epoch, so the calendar date of a
    /// day number is the same arithmetic; only which day `now` falls in differs.
    pub fn label(self, day: u64) -> String {
        provider_day_label(day)
    }

    /// The stable spelling used in configuration and telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            DayBoundary::Pacific => "pacific",
            DayBoundary::Utc => "utc",
        }
    }
}

/// Why a reservation could not be granted.
#[derive(Debug)]
pub enum ReserveError {
    /// This deployment's own budget for the current provider day is spent.
    ///
    /// A statement about the counter this process keeps: the requests it has itself
    /// reserved. It says nothing about what anyone else spent on the same account.
    Exhausted,
    /// The provider itself reported that the account's allowance for the current day
    /// is spent.
    ///
    /// A stronger and different claim from [`Self::Exhausted`]: the provider is the
    /// authority on the account, it counts traffic from anywhere, and a local budget
    /// that still looks unspent cannot make the account servable again. Kept separate
    /// so that a reader can tell "we spent our share" from "the account is empty", and
    /// so a stale or unknown reading can never be reported as confirmed exhaustion.
    ProviderExhausted {
        /// The pool's stated daily limit, for the refusal message.
        limit: u64,
    },
    /// The accounting state could not be read or written, so no budget may be spent.
    Unusable(String),
    /// An observation was offered that cannot be applied without making the counter
    /// disagree with the provider: it is stamped in the future, it describes a provider day
    /// that has ended, or the counter already holds a later reading for the same day.
    ///
    /// Refusing is the whole point. Adopting a delayed observation resets the spend of the
    /// day that replaced it; adopting an older one restores headroom that has been used.
    /// Neither is recoverable once the counter has been written, so the write is the place
    /// to say no.
    ObservationRejected {
        /// Why the reading was not applied, in the terms of the rule that refused it.
        reason: &'static str,
    },
}

/// A request-count the provider itself reported for one account pool.
///
/// This is the provider's own answer, not this deployment's bookkeeping. It counts
/// every request the account made, including other clients and other keys, so it is
/// the only figure that can confirm an *account* is empty. It is also a snapshot: the
/// provider reports it when asked, and it is wrong the moment anything else spends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProviderReading {
    /// Requests the provider counted against the pool for its current day.
    pub used: u64,
    /// The pool's stated daily limit.
    pub limit: u64,
    /// Requests the provider says the pool still permits.
    pub remaining: u64,
    /// When this reading was taken, on this host's clock.
    pub observed_at: SystemTime,
    /// When the provider says the pool resets, when it states one.
    pub reset_at: Option<SystemTime>,
}

impl ProviderReading {
    /// Whether this reading is recent enough to act on.
    ///
    /// A reading older than the tolerance is not evidence of anything: the provider
    /// may have served a thousand requests since. Treating a stale "0 remaining" as
    /// exhaustion would refuse a lane that has recovered, and treating a stale "500
    /// remaining" as headroom would spend budget that is already gone. Stale is
    /// therefore neither, and the local budget governs.
    pub fn is_fresh_at(&self, now: SystemTime, tolerance: Duration) -> bool {
        match now.duration_since(self.observed_at) {
            Ok(age) => age <= tolerance,
            // `now` before the reading: the clock moved backwards. Not fresh, because
            // nothing here can tell how old the reading really is.
            Err(_) => false,
        }
    }

    /// Seconds until the provider's stated reset, when it stated one in the future.
    pub fn seconds_to_reset_at(&self, now: SystemTime) -> Option<u64> {
        self.reset_at
            .and_then(|reset| reset.duration_since(now).ok())
            .map(|left| left.as_secs())
    }
}

/// Why an exhaustion is being reported.
///
/// The two are kept apart because they licence different decisions. [`Self::Confirmed`] is
/// the provider saying, in a reading that is still current, that the pool has no headroom
/// left. [`Self::Retained`] is that same statement carried past the point where the reading
/// is current, for a bounded time, so that a collection round which never arrives does not
/// turn into a retry storm.
///
/// A retained refusal is still a refusal: data this stale is never a reason to spend. But it
/// is not evidence about the account *now*, so nothing outside the decision to send may read
/// it as a confirmed exhaustion -- in particular, a recovered account must not be described
/// as empty on the strength of a reading that has stopped being current.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExhaustionBasis {
    /// A current reading reports no headroom.
    Confirmed,
    /// A confirmed exhaustion held past the freshness of the reading that reported it.
    Retained,
}

/// What is known about the provider's own accounting for this allowance.
///
/// Three states, kept apart because collapsing any two of them produces a wrong
/// admission decision: an unknown reading cannot refute a local budget, and a stale
/// one cannot confirm exhaustion. Only [`Self::Exhausted`] may refuse on the
/// provider's behalf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderState {
    /// No reading, or one older than the tolerance. The provider has not told us
    /// anything current, which is not the same as telling us there is headroom.
    Unknown,
    /// The provider reported headroom, recently enough to act on.
    Available {
        /// Requests the provider said remained at that moment.
        remaining: u64,
    },
    /// The provider reported no headroom, recently enough to act on, or recently enough that
    /// its statement is still being honoured.
    Exhausted {
        /// The pool's stated limit, for the refusal message.
        limit: u64,
        /// Whether the provider said so in a current reading, or whether that statement is
        /// being retained past its freshness. Reported because a retained refusal describes
        /// the past, not the account.
        basis: ExhaustionBasis,
    },
}

/// Counts requests against one target's daily allowance.
#[derive(Debug)]
pub struct DailyRequestAllowance {
    /// The quota group the allowance belongs to — the provider-side project whose counter
    /// this is. Every client that bills to the same project shares one group, and so one
    /// counter per model — or one counter for the whole group when the cap is stated at
    /// account scope.
    group: String,
    /// The target model whose allowance this is. At account scope this is the group
    /// sentinel rather than a model, because one counter serves every model in the group.
    model: ModelId,
    /// Requests permitted per provider day.
    limit: u64,
    /// Where this allowance's counter lives.
    path: PathBuf,
    /// Which midnight the counter turns over on.
    boundary: DayBoundary,
    /// How old a provider reading may be and still be acted on.
    ///
    /// Five minutes is a deliberate trade: shorter, and a provider that is briefly
    /// unreachable leaves the lane admitting on local bookkeeping while the account is
    /// actually empty; longer, and a recovered account stays refused after the provider
    /// has restored it. Neither error is silent -- [`Self::provider_state_at`] exposes
    /// which of the three states was in force -- so the trade is visible rather than
    /// baked in.
    provider_tolerance: Duration,
}

/// The persisted counter.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Counter {
    day: u64,
    spent: u64,
    /// The provider's last reported request count for this pool, if one was taken.
    ///
    /// Persisted beside the local count on purpose: the two answer different questions
    /// and a restart must not forget what the provider said. It is stored in the same
    /// file under the same lock, so a reading and the reservations made against it
    /// cannot disagree about ordering.
    provider: Option<ProviderReading>,
    /// This deployment's own `spent` at the instant `provider` was recorded.
    ///
    /// A provider reading is a snapshot: `spent` minus this anchor is exactly the number
    /// of requests made since it was taken, and those are requests the provider has not
    /// yet counted. Subtracting them is what stops the deployment admitting
    /// `remaining` fresh requests *on top of* the ones it has already sent since the
    /// last collection -- an over-admission bounded only by the poll interval, which is
    /// precisely the traffic pattern that empties an account.
    provider_anchor: u64,
    /// How long a confirmed exhaustion keeps refusing after its reading ages out.
    ///
    /// Kept beside the reading rather than derived from it so the retention outlives the
    /// freshness window without the reading being treated as current. Its day is stored
    /// with it: yesterday's retention says nothing about today.
    retention: Option<Retention>,
}

/// A confirmed exhaustion, and the day it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Retention {
    /// The provider day the exhaustion was observed in.
    day: u64,
    /// The instant after which the retention no longer refuses anything.
    until: SystemTime,
}

/// The local spend at the moment an account read was started.
///
/// Taken before the network call and applied after it. A reading is a snapshot, so any
/// request this deployment sends while the read is outstanding is absent from the
/// provider's count; anchoring at the *pre-read* figure is what keeps those requests
/// subtracted instead of forgiven. Anchoring at application time would hand back one
/// request of headroom per in-flight send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationTicket {
    /// `(provider day, spend on that day)` when the read began.
    watermark: Option<(u64, u64)>,
}

impl ObservationTicket {
    /// A ticket for a reading taken without a pre-read watermark.
    ///
    /// Deliberately conservative: with no watermark the whole of the current day's spend
    /// is treated as unsent by the provider, so the headroom is understated rather than
    /// overstated. Used by callers that read the provider themselves and by tests.
    pub fn unwatermarked() -> Self {
        Self { watermark: None }
    }

    /// What the ticket captured, when it captured anything.
    fn watermark(self) -> Option<(u64, u64)> {
        self.watermark
    }
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
            boundary: DayBoundary::Pacific,
            provider_tolerance: DEFAULT_PROVIDER_TOLERANCE,
        }
    }

    /// [`Self::new`] with an explicit day boundary.
    pub fn new_on(
        group: impl Into<String>,
        model: ModelId,
        limit: u64,
        path: PathBuf,
        boundary: DayBoundary,
    ) -> Self {
        Self {
            group: group.into(),
            model,
            limit,
            path,
            boundary,
            provider_tolerance: DEFAULT_PROVIDER_TOLERANCE,
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

    /// Which midnight this counter turns over on.
    pub fn boundary(&self) -> DayBoundary {
        self.boundary
    }

    /// The provider-day number the counter currently belongs to.
    pub fn current_day(&self) -> u64 {
        self.boundary.day(SystemTime::now())
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
        let day = self.boundary.day(now);
        match self.read_under_lock() {
            Ok(counter) => {
                let local = self.limit.saturating_sub(spent_on(counter, day));
                // The stricter of the two, never the looser. Callers of this method
                // decide whether a leg is worth attempting, so answering with the local
                // remainder while the provider reports fewer would advertise capacity
                // that does not exist. A stale or absent reading contributes nothing.
                match self.provider_headroom(&counter, day, now) {
                    Some(headroom) => local.min(headroom),
                    None => local,
                }
            }
            Err(_) => 0,
        }
    }

    /// Whether this allowance's remainder is knowably zero for the current provider day.
    ///
    /// Three answers, because the two-answer form is the one that spends money. `Some(true)`
    /// only when the counter can be read and its remainder is zero; `Some(false)` when it can
    /// be read and still has room, or when no counter exists yet; `None` when a counter
    /// exists but cannot be read. A caller deciding whether to escalate on "this allowance is
    /// spent" must treat `None` as unknown: an unreadable ledger is not an exhausted one, and
    /// [`Self::remaining`] answers `0` for both, which is right for admission and wrong for
    /// this claim.
    ///
    /// Reads without taking the lock and creates nothing: the counter is replaced atomically,
    /// so a reader sees a whole file either way, and a judgement about what the deployment's
    /// accounting says must not write to the directory it is judging.
    pub fn spent_today(&self) -> Option<bool> {
        self.spent_today_at(SystemTime::now())
    }

    /// [`Self::spent_today`] against an explicit clock, for tests and replays.
    pub fn spent_today_at(&self, now: SystemTime) -> Option<bool> {
        let day = self.boundary.day(now);
        let counter = match fs::read(&self.path) {
            Ok(bytes) => match parse_counter(&bytes) {
                Ok(counter) => counter,
                Err(_) => return None,
            },
            // No counter yet: the day's allowance is untouched, which is a fact about this
            // deployment rather than an absence of evidence.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Some(false),
            Err(_) => return None,
        };
        let local = self.limit.saturating_sub(spent_on(counter, day));
        let remaining = match self.provider_headroom(&counter, day, now) {
            Some(headroom) => local.min(headroom),
            None => local,
        };
        Some(remaining == 0)
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
        let day = self.boundary.day(now);
        // The lock is held across the read, the decision and the write. Without it two
        // processes sharing a state directory would each read the same count, each decide
        // it may spend, and together exceed the budget by one request per race.
        let lock = self.lock()?;
        let counter = self.read()?;
        // The provider's own statement comes first. When it says the pool is empty, a
        // local counter that still has room is simply wrong about the account -- it
        // counts this deployment's requests, not everyone's -- and admitting on it would
        // spend requests that do not exist. Checked before the local budget so the
        // refusal names the authority that decided.
        if let Some(headroom) = self.provider_headroom(&counter, day, now)
            && headroom == 0
        {
            let limit = counter.provider.map(|reading| reading.limit).unwrap_or(0);
            drop(lock);
            return Err(ReserveError::ProviderExhausted { limit });
        }
        let spent = spent_on(counter, day);
        if spent >= self.limit {
            drop(lock);
            return Err(ReserveError::Exhausted);
        }
        // A reservation is not an observation: the provider reading keeps whatever it said
        // and its age keeps growing, so it expires on its own schedule instead of being
        // refreshed by traffic that did not check it.
        let provider = counter.provider.filter(|reading| {
            // Drop a reading from a previous provider day: it describes a window
            // that has ended and cannot constrain this one.
            self.boundary.day(reading.observed_at) == day
        });
        // A reservation can be what empties the pool: the provider's figure was current
        // when it said three remained, and this deployment has now sent all three. Arming
        // the same bounded retention here is what stops the next request, once that reading
        // ages out, from being admitted into a pool that is already empty. Without it the
        // protection covers only a provider that volunteers the zero, and misses the very
        // common case where the last unit is spent locally.
        let mut retention = counter.retention.filter(|retention| retention.day == day);
        if retention.is_none()
            && let Some(reading) = provider
            && reading.is_fresh_at(now, self.provider_tolerance)
        {
            let sent_since = if counter.day == day {
                (spent + 1).saturating_sub(counter.provider_anchor)
            } else {
                0
            };
            if reading.remaining.saturating_sub(sent_since) == 0 {
                retention =
                    Some(self.retention_for(&reading)).filter(|retention| retention.until > now);
            }
        }
        let next = Counter {
            day,
            spent: spent + 1,
            provider,
            provider_anchor: counter.provider_anchor,
            retention,
        };
        // Persist before sending. A counter that disagrees with the provider is the
        // failure this exists to prevent, so a failed write must also undo the spend.
        self.persist(next)?;
        drop(lock);
        Ok(())
    }

    /// Requests spent on the current provider day, as this counter records them.
    ///
    /// This is the local figure only: it says what this deployment sent, not what the
    /// account has spent. Reported for diagnostics beside the provider's own figure.
    pub fn spent_at(&self, now: SystemTime) -> u64 {
        self.read_under_lock()
            .map(|counter| spent_on(counter, self.boundary.day(now)))
            .unwrap_or(0)
    }

    /// Takes the watermark for an account read that is about to start.
    ///
    /// The lock is held for the read and released before returning, so no network call
    /// ever happens under it. What the ticket carries is the position of the counter at
    /// this instant; applying the reading against it is what keeps requests made while the
    /// read is outstanding subtracted from the provider's figure.
    pub fn begin_observation(&self) -> Result<ObservationTicket, ReserveError> {
        self.begin_observation_at(SystemTime::now())
    }

    /// [`Self::begin_observation`] against an explicit clock, for tests and replays.
    pub fn begin_observation_at(&self, now: SystemTime) -> Result<ObservationTicket, ReserveError> {
        let counter = self.read_under_lock()?;
        let day = self.boundary.day(now);
        Ok(ObservationTicket {
            watermark: Some((day, spent_on(counter, day))),
        })
    }

    /// How old a provider reading may be and still be acted on.
    pub fn provider_tolerance(&self) -> Duration {
        self.provider_tolerance
    }

    /// The counter file this allowance governs.
    ///
    /// Two reconcilers that resolve to the same path are the same budget, however many
    /// credential labels name them. A reading is applied to that budget once, not once per
    /// label: applying it twice would report a duplicate write as a refusal and, worse,
    /// invites the belief that the labels hold separate budgets when they do not.
    pub fn counter_path(&self) -> &Path {
        &self.path
    }

    /// Sets the tolerance for provider readings, for tests and for a deployment whose
    /// collector runs on a known cadence.
    pub fn with_provider_tolerance(mut self, tolerance: Duration) -> Self {
        self.provider_tolerance = tolerance;
        self
    }

    /// Records what the provider reported about this pool.
    ///
    /// This is the only way provider truth reaches admission, and it is deliberately a
    /// separate call from both [`Self::try_reserve`] and selection: the collector that
    /// reads the provider runs on its own schedule, and writing through this method keeps
    /// the reading durable and ordered with the reservations made against it. A reading
    /// for a pool this allowance does not govern is refused rather than stored, because
    /// accepting it would let one pool's exhaustion refuse another's traffic.
    pub fn reconcile(&self, reading: ProviderReading) -> Result<(), ReserveError> {
        self.reconcile_at(
            reading,
            ObservationTicket::unwatermarked(),
            reading.observed_at,
        )
    }

    /// [`Self::reconcile`] with the watermark taken before the read, and an explicit clock.
    ///
    /// A reading is applied only when it can be ordered against what the counter already
    /// holds. Four shapes are refused, each for a different reason:
    ///
    /// * Stamped in the future — it cannot have been observed.
    /// * Belonging to a provider day that has ended — applying it would move the counter
    ///   back a day and discard the spend made since the boundary.
    /// * Not newer than the reading already held for this day — an older figure restores
    ///   headroom that has already been spent, and a duplicate would move the anchor.
    /// * The counter itself already ahead of `now` — the clock moved backwards and nothing
    ///   here can tell which spend is real.
    ///
    /// The anchor comes from `ticket`, not from the counter at application time, so a
    /// request sent while the read was outstanding is still subtracted from the provider's
    /// figure instead of being forgiven.
    pub fn reconcile_at(
        &self,
        reading: ProviderReading,
        ticket: ObservationTicket,
        now: SystemTime,
    ) -> Result<(), ReserveError> {
        let lock = self.lock()?;
        let counter = self.read()?;
        let now_day = self.boundary.day(now);
        let reading_day = self.boundary.day(reading.observed_at);

        let reject = if reading.observed_at > now + OBSERVATION_SKEW {
            Some("the reading is stamped in the future")
        } else if reading_day != now_day {
            Some("the reading belongs to a provider day that has ended")
        } else if counter.day > now_day {
            Some("the counter has advanced past the reading's day")
        } else if counter.provider.is_some_and(|existing| {
            self.boundary.day(existing.observed_at) == reading_day
                && existing.observed_at >= reading.observed_at
        }) {
            Some("the counter already holds a later reading for this provider day")
        } else {
            None
        };
        if let Some(reason) = reject {
            drop(lock);
            return Err(ReserveError::ObservationRejected { reason });
        }

        // Spend on the day the reading describes. A counter still holding an earlier day
        // has none for this one, so the figure is zero rather than yesterday's.
        let spent = spent_on(counter, reading_day);
        // The anchor is the pre-read watermark when the read began on the same day the
        // counter is on. Anything else — a read that straddled the boundary, or a caller
        // with no watermark — anchors at zero, which subtracts the whole day's spend and
        // therefore understates headroom instead of overstating it.
        let anchor = match ticket.watermark() {
            Some((ticket_day, spent_then))
                if ticket_day == reading_day && counter.day == reading_day =>
            {
                spent_then.min(spent)
            }
            _ => 0,
        };
        let retention = (reading
            .remaining
            .saturating_sub(spent.saturating_sub(anchor))
            == 0)
            .then(|| self.retention_for(&reading))
            .filter(|retention| retention.until > now);
        let next = Counter {
            day: reading_day,
            spent,
            provider: Some(reading),
            provider_anchor: anchor,
            retention,
        };
        self.persist(next)?;
        drop(lock);
        Ok(())
    }

    /// Requests the provider's reading still permits, after subtracting local traffic
    /// sent since that reading was taken.
    ///
    /// `None` when nothing current is known, which is not zero: a caller must not treat
    /// an absent reading as an empty account.
    fn provider_headroom(&self, counter: &Counter, day: u64, now: SystemTime) -> Option<u64> {
        // A confirmed exhaustion keeps refusing after its reading ages out. Without this a
        // collector that stops answering turns a known-empty account into a retry storm:
        // the reading becomes Unknown, the local budget looks unspent, and every retry
        // earns another provider 429.
        if let Some(retention) = counter.retention
            && retention.day == day
            && now < retention.until
        {
            return Some(0);
        }
        let reading = counter.provider?;
        if !reading.is_fresh_at(now, self.provider_tolerance) {
            return None;
        }
        self.effective_headroom(counter, &reading, day)
    }

    /// The headroom one reading still permits, after subtracting local traffic.
    ///
    /// `None` when the reading describes another provider day, where the anchor means
    /// nothing.
    fn effective_headroom(
        &self,
        counter: &Counter,
        reading: &ProviderReading,
        day: u64,
    ) -> Option<u64> {
        if self.boundary.day(reading.observed_at) != day {
            return None;
        }
        // `spent` on a different day belongs to a window that has ended; the anchor is
        // only meaningful beside the same day's spend.
        let sent_since = if counter.day == day {
            counter.spent.saturating_sub(counter.provider_anchor)
        } else {
            0
        };
        Some(reading.remaining.saturating_sub(sent_since))
    }

    /// Which of the two reasons is producing an exhaustion.
    ///
    /// [`ExhaustionBasis::Confirmed`] requires a reading that is current *and* whose
    /// adjusted headroom is zero. Anything else that still refuses is a retention, which
    /// describes a past observation rather than the account.
    fn exhaustion_basis(
        &self,
        counter: &Counter,
        day: u64,
        now: SystemTime,
    ) -> Option<ExhaustionBasis> {
        if let Some(reading) = counter.provider
            && reading.is_fresh_at(now, self.provider_tolerance)
            && self.effective_headroom(counter, &reading, day) == Some(0)
        {
            return Some(ExhaustionBasis::Confirmed);
        }
        counter
            .retention
            .filter(|retention| retention.day == day && now < retention.until)
            .map(|_| ExhaustionBasis::Retained)
    }

    /// The bounded retention for a pool confirmed to have no headroom.
    ///
    /// Measured from the END OF THE READING'S FRESHNESS, not from `now`. A retention counted
    /// from `now` expires at the same instant the reading does whenever the reading is
    /// applied promptly -- which is the normal case -- so it would protect nothing against
    /// exactly the failure it exists for: one collection round that never arrives. Counting
    /// from the end of freshness gives a full retention window of protection *after* the
    /// reading stops being current.
    ///
    /// Capped by the provider's own stated reset, because a window that has ended cannot
    /// refuse the next one. The provider day is the second cap, and it is enforced where the
    /// retention is read: a retention belongs to the day it was recorded in.
    fn retention_for(&self, reading: &ProviderReading) -> Retention {
        let day = self.boundary.day(reading.observed_at);
        let until = reading.observed_at + self.provider_tolerance + EXHAUSTION_RETENTION;
        let until = match reading.reset_at {
            Some(reset) => reset.min(until),
            None => until,
        };
        Retention { day, until }
    }

    /// What the provider last reported, adjusted for traffic sent since.
    ///
    /// Three states, never two. [`ProviderState::Unknown`] covers both "no reading" and
    /// "a reading that is too old", and neither may be used to refuse traffic: a daemon
    /// that never asked the provider is not evidence that the account is empty.
    ///
    /// The remainder reported is the *effective* headroom -- the provider's figure less
    /// the requests this counter has sent since that figure was taken. Preselection reads
    /// this, so answering with the raw figure would advertise requests that have already
    /// gone out. [`Self::provider_reading_at`] exposes the unadjusted observation for
    /// telemetry.
    pub fn provider_state_at(&self, now: SystemTime) -> ProviderState {
        let counter = match self.read_under_lock() {
            Ok(counter) => counter,
            // An unreadable counter is reported by the reservation path as Unusable.
            // Here it is simply nothing known about the provider, so admission does not
            // claim an exhaustion it cannot substantiate.
            Err(_) => return ProviderState::Unknown,
        };
        let day = self.boundary.day(now);
        match self.provider_headroom(&counter, day, now) {
            None => ProviderState::Unknown,
            Some(0) => ProviderState::Exhausted {
                limit: counter
                    .provider
                    .map(|reading| reading.limit)
                    .unwrap_or(self.limit),
                // A refusal answers the same question either way, but a caller reading the
                // state is told which: a current reading, or a statement retained past its
                // freshness. Defaulting to `Retained` is the conservative answer for a state
                // this function cannot attribute.
                basis: self
                    .exhaustion_basis(&counter, day, now)
                    .unwrap_or(ExhaustionBasis::Retained),
            },
            Some(remaining) => ProviderState::Available { remaining },
        }
    }

    /// The provider's own figure, exactly as it was reported, while it is fresh.
    ///
    /// Not adjusted for local traffic and not the admission answer: this is the observation
    /// a dashboard shows and a reconciliation is written from, kept separate so the raw
    /// number and the headroom it produces can be compared.
    pub fn provider_reading_at(&self, now: SystemTime) -> Option<ProviderReading> {
        let counter = self.read_under_lock().ok()?;
        let reading = counter.provider?;
        reading
            .is_fresh_at(now, self.provider_tolerance)
            .then_some(reading)
    }

    /// [`Self::provider_reading_at`] against the current clock.
    pub fn provider_reading(&self) -> Option<ProviderReading> {
        self.provider_reading_at(SystemTime::now())
    }

    /// [`Self::provider_state_at`] against the current clock.
    pub fn provider_state(&self) -> ProviderState {
        self.provider_state_at(SystemTime::now())
    }

    /// Whether both of this allowance's files can be used *by this process*, without
    /// changing anything.
    ///
    /// Answers the question a directory probe cannot: the directory may be writable while a
    /// counter left behind by a different account is not. Files that are present are opened
    /// exactly as a real call will, so an unreadable counter is a configuration error with a
    /// remedy in the message rather than a 503 discovered by the first caller.
    ///
    /// **Absent files are not a fault and are not created here.** Creation belongs to the
    /// account that spends: the first reservation creates the directory, the lock and the
    /// counter as the process that will own them. Doing it at load looks equivalent and is
    /// not - a deployment preflight runs the same loader as root, so a load-time create
    /// leaves root-owned files under a unit that runs as its own user, and the first request
    /// then fails with a permission error that reads like a provider fault. A dry run must
    /// be able to validate a fresh deployment without leaving any state behind.
    pub fn check_usable(&self) -> Result<(), String> {
        for path in [self.path.clone(), self.lock_path()] {
            match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(_) => {}
                // Nothing exists yet: the first reservation creates both files, as the
                // account that will spend them.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(self.explain(&path, &error)),
            }
        }
        Ok(())
    }

    /// A refusal that names the file, its owner and the one command that fixes it.
    ///
    /// The ownership case is not hypothetical: a deployment path that runs as root creates
    /// these files under a unit that runs as another account, and the first caller then
    /// fails with a bare `Permission denied` that reads like a provider fault.
    fn explain(&self, path: &Path, error: &std::io::Error) -> String {
        let owner = std::fs::metadata(path).ok().map(|metadata| {
            use std::os::unix::fs::MetadataExt;
            format!("{}:{}", metadata.uid(), metadata.gid())
        });
        match owner {
            Some(owner) if error.kind() == std::io::ErrorKind::PermissionDenied => format!(
                "{} (owner {owner}) cannot be opened read-write by this process: {error}; the \
                 accounting files must belong to the account the service runs as, for example \
                 `chown -R <service-user>:<service-group> {}`",
                path.display(),
                self.path.parent().unwrap_or(Path::new(".")).display(),
            ),
            Some(owner) => format!("{} (owner {owner}) is unusable: {error}", path.display()),
            None => format!("{} is unusable: {error}", path.display()),
        }
    }

    /// The counter file this allowance persists at.
    pub fn path(&self) -> &Path {
        &self.path
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
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Counter {
                day: 0,
                spent: 0,
                provider: None,
                provider_anchor: 0,
                retention: None,
            }),
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
        // `basis` states which authority the admission decision rests on, so a reader
        // can tell a locally-budgeted counter from one reconciled with the provider.
        let basis = if counter.provider.is_some() {
            "local_budget+provider_reconciled"
        } else {
            "local_budget"
        };
        let provider = match counter.provider {
            None => "null".to_string(),
            Some(reading) => format!(
                "{{\"used\":{},\"limit\":{},\"remaining\":{},\"observed_at_secs\":{},\"reset_at_secs\":{},\"spent_at_observation\":{}}}",
                reading.used,
                reading.limit,
                reading.remaining,
                unix_secs(reading.observed_at),
                reading
                    .reset_at
                    .map(|at| unix_secs(at).to_string())
                    .unwrap_or_else(|| "null".to_string()),
                counter.provider_anchor,
            ),
        };
        let retention = match counter.retention {
            None => "null".to_string(),
            Some(retention) => format!(
                "{{\"day\":{},\"until_secs\":{}}}",
                retention.day,
                unix_secs(retention.until)
            ),
        };
        let body = format!(
            "{{\"day\":{},\"spent\":{},\"limit\":{},\"group\":{},\"model\":{},\"basis\":{},\"provider\":{},\"retention\":{}}}\n",
            counter.day,
            counter.spent,
            self.limit,
            serde_json::Value::String(self.group.clone()),
            serde_json::Value::String(self.model.to_string()),
            serde_json::Value::String(basis.to_string()),
            provider,
            retention,
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
    // A counter written before provider reconciliation existed has no `provider` key,
    // which is an absence of information rather than a fault. A key that IS present but
    // unreadable is a fault: it means the account's authority figure was written by
    // something that disagrees with this reader, and admitting on a half-read counter
    // would silently ignore an exhaustion it may be carrying.
    let (provider, provider_anchor) = match value.get("provider") {
        None | Some(serde_json::Value::Null) => (None, 0),
        Some(block) => {
            let reading = parse_provider(block)?;
            let anchor = block
                .get("spent_at_observation")
                .and_then(serde_json::Value::as_u64)
                // A block written without the anchor cannot be reconciled against local
                // traffic. Treat the anchor as the current spend, which makes the reading
                // conservative: it claims no extra headroom rather than over-admitting.
                .unwrap_or(spent);
            (Some(reading), anchor)
        }
    };
    // A counter written before the retention existed has no `retention` key, which is an
    // absence of a hold rather than a fault: nothing was known to be exhausted.
    let retention = match value.get("retention") {
        None | Some(serde_json::Value::Null) => None,
        Some(block) => Some(parse_retention(block)?),
    };
    Ok(Counter {
        day,
        spent,
        provider,
        provider_anchor,
        retention,
    })
}

/// Reads the retention block of a counter file.
///
/// A malformed block is a fault rather than a missing hold: a counter that says it was
/// holding an exhaustion but cannot say when the hold ends must not be trusted to admit.
fn parse_retention(block: &serde_json::Value) -> Result<Retention, String> {
    let day = block
        .get("day")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "retention.\"day\" is missing or not a whole number".to_string())?;
    let until = block
        .get("until_secs")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "retention.\"until_secs\" is missing or not a whole number".to_string())?;
    Ok(Retention {
        day,
        until: UNIX_EPOCH + Duration::from_secs(until),
    })
}

/// Reads the provider block of a counter file.
fn parse_provider(block: &serde_json::Value) -> Result<ProviderReading, String> {
    let number = |key: &str| -> Result<u64, String> {
        block
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("provider.\"{key}\" is missing or not a whole number"))
    };
    let seconds = |key: &str| -> Result<u64, String> {
        block
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("provider.\"{key}\" is missing or not a whole number"))
    };
    let observed = seconds("observed_at_secs")?;
    let reset_at = match block.get("reset_at_secs") {
        None | Some(serde_json::Value::Null) => None,
        Some(_) => Some(UNIX_EPOCH + Duration::from_secs(seconds("reset_at_secs")?)),
    };
    Ok(ProviderReading {
        used: number("used")?,
        limit: number("limit")?,
        remaining: number("remaining")?,
        observed_at: UNIX_EPOCH + Duration::from_secs(observed),
        reset_at,
    })
}

/// Seconds since the epoch for a clock reading, saturating rather than panicking.
fn unix_secs(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
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

/// The model component an account-scoped counter uses.
///
/// A provider that caps free requests per *account* shares one budget across every model,
/// so that counter must not be keyed by model. The sentinel is spelled so it cannot
/// collide with a real model id: provider model ids use `[a-z0-9./:_-]` and never begin
/// with `*`.
pub const ACCOUNT_SCOPE_MODEL: &str = "*account*";

/// The counter file for an account-scoped allowance: one per quota group, not per model.
pub fn account_allowance_path(directory: &Path, group: &str) -> PathBuf {
    allowance_path(directory, group, ACCOUNT_SCOPE_MODEL)
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

    /// The largest *raw estimate* whose margin-adjusted reservation still fits the window.
    ///
    /// The inverse of [`AdmissionMargin::apply`]: `apply(estimate) <= limit` holds exactly
    /// when `estimate <= floor(limit * 1000 / permille)`. The pre-dispatch filter uses this
    /// so a request it admits is one the send path will also admit, without the filter
    /// needing to know the margin arithmetic.
    pub fn safe_input_ceiling(&self, margin: AdmissionMargin) -> Option<u64> {
        self.input_tokens_per_minute.map(|limit| {
            u64::try_from(u128::from(limit) * 1_000 / u128::from(margin.permille()))
                .unwrap_or(u64::MAX)
        })
    }
}

/// How much larger than its estimate a request is assumed to be before it is dispatched.
///
/// The estimate is characters over the outbound JSON divided by four: a shape, not a
/// measurement. Two measurements of that estimator disagreed in both directions on real
/// prompts (an over-count of about 24 % and an under-count of about 16 %), so no fixed
/// accuracy figure is claimed here — the margin is a protective pad an operator
/// configures, not a proven bound and not a substitute for counting.
///
/// It exists because the failure it prevents is asymmetric. Reserving too little lets a
/// provider charge more input tokens in one minute than its published window allows, which
/// is a real overrun against a real account; reserving too much only refuses a request
/// slightly early, and the route can hand it to a sibling with a wider window. Only values
/// at or above 1.0 are constructible, so the pad can never become an under-count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdmissionMargin(u32);

impl AdmissionMargin {
    /// No pad: the estimate is reserved as stated. The default for every target that does
    /// not declare one, so pre-existing manifests behave exactly as they did.
    pub const NONE: Self = Self(1_000);

    /// A pad of `permille` thousandths, or `None` below 1.0.
    pub const fn from_permille(permille: u32) -> Option<Self> {
        if permille < 1_000 {
            None
        } else {
            Some(Self(permille))
        }
    }

    /// The pad in thousandths.
    pub const fn permille(self) -> u32 {
        self.0
    }

    /// The reservation for `estimate`: the estimate scaled up and rounded up.
    ///
    /// Rounded up because a fractional token is a token the provider counts, and because
    /// rounding down would silently undo part of the pad on every request.
    pub fn apply(self, estimate: u64) -> u64 {
        let scaled = u128::from(estimate) * u128::from(self.0);
        let rounded = scaled.div_ceil(1_000);
        u64::try_from(rounded).unwrap_or(u64::MAX).max(1)
    }

    /// The wider of two pads.
    #[must_use]
    pub fn widest(self, other: Self) -> Self {
        Self(self.0.max(other.0))
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
    /// per-request ceiling the same request fits under its margin; configuration validation
    /// enforces that, and this variant is the defensive answer if a call arrives over it
    /// anyway - or if a learned ratio has since widened the pad past the declared ceiling.
    ExceedsWindow {
        /// The request's reservation after the margin.
        needed: u64,
        /// The whole window's input budget.
        limit: u64,
        /// The margin the reservation was computed with, in permille.
        margin_permille: u32,
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
    /// The largest ratio of provider-reported input tokens to the raw estimate this budget
    /// has ever shown, in permille, floored at 1.0.
    ///
    /// Learned rather than declared, and never decreased: each provider answer that reports
    /// its own prompt count is evidence about how the estimator behaves on this model, and
    /// the next request should be reserved at the worst ratio seen rather than at the pad
    /// the operator guessed before any traffic existed. A ratio below 1.0 is discarded
    /// because it only appears when the estimate over-counted, and an over-count needs no
    /// protection.
    observed_ratio: u32,
}

#[derive(Clone, Copy, Debug)]
struct Spend {
    sequence: u64,
    at: SystemTime,
    /// What this request holds against the window: the margin-adjusted reservation, raised
    /// in place if the provider reports more.
    tokens: u64,
    /// The raw estimate the reservation came from, kept so a reported actual is compared
    /// against the estimate rather than against the pad, which would hide the estimator's
    /// own error.
    estimated: u64,
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

    /// [`Self::reserve_with_margin`], on the real clock.
    pub fn reserve_with_margin(
        &self,
        key: &GovernorKey,
        limits: WindowLimits,
        margin: AdmissionMargin,
        input_tokens: u64,
    ) -> Result<Option<PacingTicket>, PacingError> {
        self.reserve_with_margin_at(key, limits, margin, input_tokens, SystemTime::now())
    }

    /// [`Self::reserve`] against an explicit clock, for tests and replays.
    pub fn reserve_at(
        &self,
        key: &GovernorKey,
        limits: WindowLimits,
        input_tokens: u64,
        now: SystemTime,
    ) -> Result<Option<PacingTicket>, PacingError> {
        self.reserve_with_margin_at(key, limits, AdmissionMargin::NONE, input_tokens, now)
    }

    /// [`Self::reserve_with_margin`] against an explicit clock, for tests and replays.
    ///
    /// The reservation is the estimate scaled by whichever is wider - the operator's pad or
    /// the worst ratio this model has actually reported. That ordering matters: a pad chosen
    /// before any traffic cannot know that one model's tokenizer runs 19 % above
    /// characters/4, and the learned ratio supplies what the declaration could not.
    pub fn reserve_with_margin_at(
        &self,
        key: &GovernorKey,
        limits: WindowLimits,
        margin: AdmissionMargin,
        input_tokens: u64,
        now: SystemTime,
    ) -> Result<Option<PacingTicket>, PacingError> {
        if limits.is_unlimited() {
            return Ok(None);
        }
        let mut ledgers = self.ledgers.lock().unwrap_or_else(PoisonError::into_inner);
        let ledger = ledgers.entry(key.clone()).or_default();
        let effective = margin.widest(AdmissionMargin(ledger.observed_ratio.max(1_000)));
        let reserved = effective.apply(input_tokens);
        if let Some(limit) = limits.input_tokens_per_minute
            && reserved > limit
        {
            return Err(PacingError::ExceedsWindow {
                needed: reserved,
                limit,
                margin_permille: effective.permille(),
            });
        }
        let cutoff = now.checked_sub(WINDOW).unwrap_or(UNIX_EPOCH);
        while ledger
            .spends
            .front()
            .is_some_and(|spend| spend.at <= cutoff)
        {
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
            let outstanding = ledger
                .spends
                .iter()
                .fold(0_u64, |total, spend| total.saturating_add(spend.tokens));
            if outstanding.saturating_add(reserved) > limit {
                let mut remaining = outstanding;
                let mut retry_after = WINDOW;
                for spend in &ledger.spends {
                    remaining = remaining.saturating_sub(spend.tokens);
                    if remaining.saturating_add(reserved) <= limit {
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
            tokens: reserved,
            estimated: input_tokens,
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
        let mut ledgers = self.ledgers.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(ledger) = ledgers.get_mut(&ticket.key) {
            ledger
                .spends
                .retain(|spend| spend.sequence != ticket.sequence);
        }
    }

    /// Charges a sent request with what the provider says it actually read.
    ///
    /// Called once per answered request, with `usage.prompt_tokens` (or the provider's
    /// equivalent) from the response. Two things happen, and only ever upward:
    ///
    /// - The request's own slot rises to the reported count if that is larger than what was
    ///   reserved. The extra tokens are spent from this window now rather than discovered as
    ///   an overrun by the provider's meter later, which is the point: the next request on
    ///   this model must see less headroom than the estimate implied.
    /// - The ledger's learned ratio rises to cover this sample, so the next reservation on
    ///   this model is padded to the worst observed under-count instead of waiting for an
    ///   operator to guess a pad.
    ///
    /// Nothing is ever reduced. A request that reached the provider spent what it spent, and
    /// a reported count below the reservation is an over-count that needs no correction -
    /// refunding it would let a burst of short prompts buy back room a long one already used.
    pub fn reconcile(&self, ticket: &PacingTicket, actual_tokens: u64) {
        if actual_tokens == 0 {
            return;
        }
        let mut ledgers = self.ledgers.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(ledger) = ledgers.get_mut(&ticket.key) else {
            return;
        };
        let mut raised_ratio: Option<u32> = None;
        if let Some(spend) = ledger
            .spends
            .iter_mut()
            .find(|spend| spend.sequence == ticket.sequence)
        {
            spend.tokens = spend.tokens.max(actual_tokens);
            if spend.estimated > 0 {
                let ratio = u128::from(actual_tokens) * 1_000 / u128::from(spend.estimated);
                let ratio = u32::try_from(ratio).unwrap_or(u32::MAX).max(1_000);
                if ratio > ledger.observed_ratio {
                    raised_ratio = Some(ratio);
                }
            }
        }
        if let Some(ratio) = raised_ratio {
            ledger.observed_ratio = ratio;
        }
    }

    /// The worst provider-reported input ratio this budget has observed, in permille.
    ///
    /// 1000 until a provider reports a count larger than the estimate. Reported so an
    /// operator can see what the estimator is actually doing on one model instead of
    /// reasoning from two unrelated samples.
    pub fn observed_ratio(&self, key: &GovernorKey) -> u32 {
        let ledgers = self.ledgers.lock().unwrap_or_else(PoisonError::into_inner);
        ledgers
            .get(key)
            .map_or(1_000, |ledger| ledger.observed_ratio.max(1_000))
    }

    /// The window's current spend for `key`, for reporting and tests.
    pub fn used(&self, key: &GovernorKey, now: SystemTime) -> (u64, u64) {
        let mut ledgers = self.ledgers.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(ledger) = ledgers.get_mut(key) else {
            return (0, 0);
        };
        let cutoff = now.checked_sub(WINDOW).unwrap_or(UNIX_EPOCH);
        while ledger
            .spends
            .front()
            .is_some_and(|spend| spend.at <= cutoff)
        {
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
    margin: AdmissionMargin,
}

impl PacedQuota {
    /// Binds `key`, `limits` and the deployment's admission pad to a shared governor.
    pub fn new(
        governor: std::sync::Arc<QuotaGovernor>,
        key: GovernorKey,
        limits: WindowLimits,
        margin: AdmissionMargin,
    ) -> Self {
        Self {
            governor,
            key,
            limits,
            margin,
        }
    }

    /// The declared limits.
    pub fn limits(&self) -> WindowLimits {
        self.limits
    }

    /// The declared admission pad.
    pub fn margin(&self) -> AdmissionMargin {
        self.margin
    }

    /// The budget this target draws on.
    pub fn key(&self) -> &GovernorKey {
        &self.key
    }

    /// The largest raw estimate this target can admit, from its declaration alone.
    ///
    /// A *static* bound: it ignores both the window's current occupancy and any ratio the
    /// governor has since learned, both of which can only tighten it. The pre-dispatch
    /// filter uses it to decide that a request belongs on a wider sibling; the send path
    /// remains the authority.
    pub fn declared_safe_ceiling(&self) -> Option<u64> {
        self.limits.safe_input_ceiling(self.margin)
    }

    /// Reserves one request against the window, at the deployment's pad.
    pub fn reserve(&self, input_tokens: u64) -> Result<Option<PacingTicket>, PacingError> {
        self.governor
            .reserve_with_margin(&self.key, self.limits, self.margin, input_tokens)
    }

    /// Releases a reservation that never reached the provider.
    pub fn release(&self, ticket: &PacingTicket) {
        self.governor.release(ticket);
    }

    /// Charges a sent request with the provider's own input-token count.
    pub fn reconcile(&self, ticket: &PacingTicket, actual_tokens: u64) {
        self.governor.reconcile(ticket, actual_tokens);
    }
}

/// A crude character-based estimate of a request body's input tokens, for pacing only.
///
/// The governor needs a number before the call, and the only body it can see is the exact
/// one about to be sent. Counting characters over every string in the body and dividing by
/// four is the standard approximation. **It is not a measurement and must not be presented
/// as one**: measured against the provider's own count it has erred by +24 % on one prompt
/// and -16 % on another, so no accuracy figure is claimed.
///
/// Strings that are themselves inline binary payloads (`data:` URIs) are skipped: their
/// base64 length says nothing about the provider's image accounting, and counting it would
/// refuse every vision request against a text budget. Those requests are paced by the
/// request axis alone.
///
/// The caller applies [`AdmissionMargin`] to the result before it reaches a window, and
/// [`QuotaGovernor::reconcile`] corrects the window afterwards with what the provider
/// reported. This function's job is only to be cheap, deterministic and available before
/// any bytes leave the process.
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
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            }
        }
    }
    let mut characters = 0_u64;
    count(body, &mut characters);
    (characters / 4).max(1)
}

/// The input-token count a provider reported for an answered request, when it reported one.
///
/// Read from the response body rather than from a header, because that is where every
/// provider on this deployment's wire formats publishes it, and because the caller already
/// holds the body. The shapes:
///
/// - `usage.prompt_tokens` - OpenAI chat completions and the OpenAI-compatible surfaces
///   Google exposes, which is how the hosted Gemma lanes reported `18054`.
/// - `usage.input_tokens` - the Responses surface and Anthropic, whose cache counters are
///   part of the same input budget and are therefore summed rather than treated separately:
///   a cached prefix is still input the provider counted.
/// - `usageMetadata.promptTokenCount` - Google's native generateContent.
///
/// `None` means the provider said nothing this code can read, which is not the same as
/// zero: the caller keeps its conservative reservation rather than recording a free request.
pub fn reported_input_tokens(body: &serde_json::Value) -> Option<u64> {
    fn reach<'a>(body: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
        path.iter().try_fold(body, |node, key| node.get(key))
    }
    if let Some(prompt) =
        reach(body, &["usage", "prompt_tokens"]).and_then(serde_json::Value::as_u64)
    {
        return Some(prompt);
    }
    if let Some(input) = reach(body, &["usage", "input_tokens"]).and_then(serde_json::Value::as_u64)
    {
        let cached = ["cache_creation_input_tokens", "cache_read_input_tokens"]
            .iter()
            .filter_map(|key| reach(body, &["usage", key]).and_then(serde_json::Value::as_u64))
            .fold(0_u64, u64::saturating_add);
        return Some(input.saturating_add(cached));
    }
    reach(body, &["usageMetadata", "promptTokenCount"]).and_then(serde_json::Value::as_u64)
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

    /// The three-state reading: absent is untouched, spent is spent, and a counter that cannot
    /// be parsed is neither. The last case is why the two-state form is unusable as evidence —
    /// `remaining` answers `0` for it, which a caller would read as exhaustion.
    #[test]
    fn spent_today_separates_unknown_from_spent() {
        let directory = scratch("spent-today");
        let counted = allowance(&directory, 1);
        assert_eq!(
            counted.spent_today(),
            Some(false),
            "no counter yet is an untouched day, not an absent answer"
        );
        assert_eq!(
            counted.remaining(),
            1,
            "and the same day is visible to admission"
        );

        counted.try_reserve().expect("the day's only request");
        assert_eq!(counted.spent_today(), Some(true));
        assert_eq!(counted.remaining(), 0);

        let corrupt = scratch("spent-today-corrupt");
        let unreadable = allowance(&corrupt, 1);
        fs::create_dir_all(unreadable.path().parent().expect("a parent")).expect("directory");
        fs::write(unreadable.path(), b"{not json").expect("a broken counter");
        assert_eq!(
            unreadable.spent_today(),
            None,
            "a counter that cannot be read is unknown, never spent"
        );
        assert_eq!(
            unreadable.remaining(),
            0,
            "the two-state reading collapses unknown into spent, which is the defect"
        );
    }

    /// Judging the evidence must not change what it judges: no counter, no lock file.
    #[test]
    fn spent_today_creates_nothing() {
        let directory = scratch("spent-today-inert");
        let judged = allowance(&directory, 1);
        let lock = judged.lock_path();
        assert_eq!(judged.spent_today(), Some(false));
        assert!(
            !judged.path().exists() && !lock.exists(),
            "a decision about the accounting must not write the accounting"
        );
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
            requests_full.reserve_at(&key, minute_limits(Some(1), Some(1_000)), 1, instant(1)),
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
                limit: 16_000,
                margin_permille: 1_000
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
    // ---- The measured Gemma overrun, as a regression -------------------------------

    /// The one measured wire call this whole mechanism exists for.
    ///
    /// 2026-11-21, `gemma-4-26b-a4b-it`, free tier 30 RPM / 16,000 input TPM:
    /// `estimate_input_tokens` said **15,202**, the governor admitted it against a 16,000
    /// window, the provider reported **18,054** prompt tokens, and the AI Studio console
    /// recorded an 18.05K/16K input-TPM peak. The request left the process on an estimate
    /// that was 18.8 % low, and the account was charged above its own published window.
    ///
    /// At a 1300 per-mille admission pad the reservation is 15,202 * 1.3 = 19,763 against a
    /// 16,000 window, so the request is refused **before egress** and the route can offer it
    /// to a 250K-TPM Lite leg instead. This test is the red/green for that fix: without the
    /// pad it dispatches, with it it cannot.
    #[test]
    fn the_measured_gemma_overrun_is_refused_before_egress_under_an_admission_margin() {
        let governor = QuotaGovernor::new();
        let key = governor_key("google/gemma-4-26b-a4b-it");
        let limit = minute_limits(Some(30), Some(16_000));
        let pad = AdmissionMargin::from_permille(1_300).expect("1300 is a valid pad");

        // The estimate that actually went out on the wire.
        assert_eq!(pad.apply(15_202), 19_763);
        assert_eq!(
            governor.reserve_with_margin_at(&key, limit, pad, 15_202, instant(0)),
            Err(PacingError::ExceedsWindow {
                needed: 19_763,
                limit: 16_000,
                margin_permille: 1_300,
            })
        );
        // Nothing was recorded for a request that never left.
        assert_eq!(governor.used(&key, instant(0)), (0, 0));

        // Without the pad the same estimate is admitted - which is the defect, preserved
        // here as a control so this test cannot pass by the request simply being too large.
        assert!(
            governor.reserve_at(&key, limit, 15_202, instant(0)).is_ok(),
            "the unpadded estimate must still be admissible, or the pad proves nothing"
        );
    }

    /// A request the estimate places comfortably inside the window keeps working.
    ///
    /// The pad must not make the lane unusable: 8,000 estimated tokens pad to 10,400, well
    /// inside 16,000, so ordinary traffic is admitted exactly as before.
    #[test]
    fn the_margin_still_admits_ordinary_traffic() {
        let governor = QuotaGovernor::new();
        let key = governor_key("google/gemma-4-26b-a4b-it");
        let limit = minute_limits(Some(30), Some(16_000));
        let pad = AdmissionMargin::from_permille(1_300).expect("valid pad");
        assert_eq!(pad.apply(8_000), 10_400);
        assert!(
            governor
                .reserve_with_margin_at(&key, limit, pad, 8_000, instant(0))
                .is_ok()
        );
        assert_eq!(governor.used(&key, instant(0)), (1, 10_400));
    }

    /// A provider count above the reservation is charged to the window immediately.
    #[test]
    fn a_reported_count_above_the_reservation_raises_the_slot() {
        let governor = QuotaGovernor::new();
        let key = governor_key("google/gemma-4-26b-a4b-it");
        let limit = minute_limits(Some(30), Some(16_000));
        let pad = AdmissionMargin::from_permille(1_300).expect("valid pad");
        let ticket = governor
            .reserve_with_margin_at(&key, limit, pad, 2_000, instant(0))
            .expect("fits")
            .expect("paced");
        assert_eq!(governor.used(&key, instant(0)), (1, 2_600));

        // The provider says this request read 4,000 tokens, not 2,600.
        governor.reconcile(&ticket, 4_000);
        assert_eq!(governor.used(&key, instant(0)), (1, 4_000));

        // And the learned ratio now pads the next reservation at 2.0x, not 1.3x.
        assert_eq!(governor.observed_ratio(&key), 2_000);
        let next = governor
            .reserve_with_margin_at(&key, limit, pad, 2_000, instant(1))
            .expect("fits")
            .expect("paced");
        assert_eq!(governor.used(&key, instant(1)), (2, 8_000));
        // The second reservation alone is 4,000 - twice the estimate.
        drop(next);
    }

    /// A reported count below the reservation does not buy back room.
    #[test]
    fn a_reported_count_below_the_reservation_never_refunds() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(None, Some(10_000));
        let pad = AdmissionMargin::from_permille(1_500).expect("valid pad");
        let ticket = governor
            .reserve_with_margin_at(&key, limits, pad, 4_000, instant(0))
            .expect("fits")
            .expect("paced");
        assert_eq!(governor.used(&key, instant(0)), (1, 6_000));
        governor.reconcile(&ticket, 1_000);
        assert_eq!(governor.used(&key, instant(0)), (1, 6_000));
        assert_eq!(governor.observed_ratio(&key), 1_000);
    }

    /// The learned ratio survives that request's own expiry: it is evidence about the
    /// estimator, not about the minute.
    #[test]
    fn the_learned_ratio_outlives_the_window_it_was_measured_in() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(Some(10), Some(100_000));
        let pad = AdmissionMargin::NONE;
        let ticket = governor
            .reserve_with_margin_at(&key, limits, pad, 1_000, instant(0))
            .expect("fits")
            .expect("paced");
        governor.reconcile(&ticket, 1_400);
        assert_eq!(governor.observed_ratio(&key), 1_400);
        // Two minutes later the spends have aged out; the ratio has not.
        assert_eq!(governor.used(&key, instant(120)), (0, 0));
        assert_eq!(governor.observed_ratio(&key), 1_400);
        let ticket = governor
            .reserve_with_margin_at(&key, limits, pad, 1_000, instant(120))
            .expect("fits")
            .expect("paced");
        assert_eq!(governor.used(&key, instant(120)), (1, 1_400));
        drop(ticket);
    }

    /// `reconcile` on a ticket the ledger no longer holds (released, or already expired)
    /// changes nothing rather than resurrecting a spend.
    #[test]
    fn reconciling_an_unknown_ticket_is_inert() {
        let governor = QuotaGovernor::new();
        let key = governor_key("m");
        let limits = minute_limits(None, Some(10_000));
        let ticket = governor
            .reserve_at(&key, limits, 1_000, instant(0))
            .expect("fits")
            .expect("paced");
        governor.release(&ticket);
        governor.reconcile(&ticket, 9_000);
        assert_eq!(governor.used(&key, instant(0)), (0, 0));
    }

    /// The static ceiling used by the pre-dispatch filter is the exact inverse of `apply`.
    #[test]
    fn the_declared_ceiling_is_the_inverse_of_the_applied_margin() {
        let limits = minute_limits(Some(30), Some(16_000));
        let pad = AdmissionMargin::from_permille(1_300).expect("valid pad");
        let ceiling = limits.safe_input_ceiling(pad).expect("window declared");
        assert_eq!(ceiling, 12_307); // floor(16000 * 1000 / 1300)
        assert!(pad.apply(ceiling) <= 16_000);
        assert!(pad.apply(ceiling + 1) > 16_000);
        assert_eq!(minute_limits(None, None).safe_input_ceiling(pad), None);
    }

    /// A pad below 1.0 is not constructible, so it cannot be used to under-reserve.
    #[test]
    fn a_margin_below_one_is_refused_at_construction() {
        assert_eq!(AdmissionMargin::from_permille(999), None);
        assert_eq!(AdmissionMargin::from_permille(0), None);
        assert_eq!(
            AdmissionMargin::from_permille(1_000),
            Some(AdmissionMargin::NONE)
        );
    }

    /// The pad rounds up, because a fractional token is a token the provider counts.
    #[test]
    fn the_margin_rounds_up() {
        let pad = AdmissionMargin::from_permille(1_300).expect("valid pad");
        assert_eq!(pad.apply(1), 2);
        assert_eq!(pad.apply(10), 13);
        assert_eq!(pad.apply(15_202), 19_763);
        // 10 * 1300 / 1000 = 13 exactly; 11 -> 14.3 -> 15.
        assert_eq!(pad.apply(11), 15);
    }

    // ---- Provider-reported counts ---------------------------------------------------

    /// Every wire shape the deployment actually sees, and the one it must not guess.
    #[test]
    fn the_reported_input_count_is_read_from_each_wire_shape() {
        // The exact shape of the measured Gemma answer.
        assert_eq!(
            reported_input_tokens(&serde_json::json!({
                "usage": {"prompt_tokens": 18_054, "completion_tokens": 0, "total_tokens": 18_067}
            })),
            Some(18_054)
        );
        // Responses / Anthropic: cache counters are part of the same input budget.
        assert_eq!(
            reported_input_tokens(&serde_json::json!({
                "usage": {
                    "input_tokens": 900,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 10,
                }
            })),
            Some(1_010)
        );
        // Google native generateContent.
        assert_eq!(
            reported_input_tokens(&serde_json::json!({
                "usageMetadata": {"promptTokenCount": 4017, "candidatesTokenCount": 1}
            })),
            Some(4_017)
        );
        // Nothing readable is `None`, not zero: the reservation must stand.
        assert_eq!(
            reported_input_tokens(&serde_json::json!({"choices": []})),
            None
        );
        assert_eq!(reported_input_tokens(&serde_json::Value::Null), None);
    }

    // --- the load-time usability gate --------------------------------------------
    //
    // A budget that cannot be accounted is not a budget. But the gate must not CREATE the
    // accounting state: a deployment preflight runs this same loader, often as root, and a
    // root-owned counter under a unit that runs as its own user is a production defect that
    // surfaces as a provider fault. Absent state is the spender's to create.

    /// A newly configured lane's first call works, and the first call is what creates the
    /// counter - as the account that spends it.
    #[test]
    fn a_fresh_lane_is_checked_without_being_created_and_works_on_first_use() {
        let directory = scratch("fresh-lane");
        let counter = allowance(&directory, 7);

        counter.check_usable().expect("absent state is not a fault");
        assert!(
            !counter.path().exists(),
            "a check must leave no accounting state behind: a root-run dry run does this"
        );

        // The first physical call creates the directory, the lock and the counter itself.
        counter
            .try_reserve()
            .expect("the first call of a fresh lane");
        assert!(counter.path().exists(), "the spender created the counter");
        assert_eq!(counter.remaining(), 6, "and it counts what it spent");
        // A fresh instance over the same files agrees: the document is valid, not empty.
        assert_eq!(allowance(&directory, 7).remaining(), 6);
    }

    /// An unusable counter must name the file, its owner and the one command that fixes it.
    ///
    /// The bare `Permission denied (os error 13)` this replaces was read as a provider
    /// fault for a production lane; the remedy belongs in the message.
    #[test]
    fn check_usable_explains_an_unwritable_counter() {
        use std::os::unix::fs::PermissionsExt;

        let directory = scratch("check-explain");
        let counter = allowance(&directory, 7);
        // Created by "someone else" for this test: a real spend, then made read-only.
        counter
            .try_reserve()
            .expect("the file must exist to be probed");
        std::fs::set_permissions(counter.path(), std::fs::Permissions::from_mode(0o400))
            .expect("make the counter read-only");

        // Root ignores the mode; the assertion below is about the account the suite runs as.
        let root = std::fs::OpenOptions::new()
            .write(true)
            .open(counter.path())
            .is_ok();
        let failure = counter.check_usable();
        std::fs::set_permissions(counter.path(), std::fs::Permissions::from_mode(0o600))
            .expect("restore");
        if root {
            assert!(
                failure.is_ok(),
                "root can open it, so there is nothing to explain"
            );
            return;
        }
        let reason = failure.expect_err("a read-only counter cannot be accounted");
        assert!(
            reason.contains(&counter.path().display().to_string()),
            "the refusal must name the file: {reason}"
        );
        assert!(
            reason.contains("chown"),
            "the refusal must name the remedy: {reason}"
        );
        assert!(
            reason.contains(&directory.display().to_string()),
            "the remedy must name the directory to hand over: {reason}"
        );
    }

    /// A counter whose path is occupied by a directory is unusable, and says so.
    #[test]
    fn check_usable_refuses_a_path_it_cannot_open() {
        let directory = scratch("check-blocked");
        let counter = allowance(&directory, 7);
        std::fs::create_dir_all(counter.path()).expect("plant a directory in the file's place");
        let reason = counter
            .check_usable()
            .expect_err("a directory is not a counter");
        assert!(
            reason.contains(&counter.path().display().to_string()),
            "the refusal must name the path: {reason}"
        );
    }
}
