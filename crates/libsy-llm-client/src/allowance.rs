// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-target daily request allowances.
//!
//! Some providers cap the number of *requests* a project may make per model per day,
//! independently of tokens (the Google AI Studio free tier publishes exactly that: a
//! daily RPD figure whose value differs per model). The cap is the operator's hard
//! budget, so it has to be spent with an actual counter rather than inferred from a
//! provider error after the fact — a rejected request is still a request the provider
//! may have counted, and the whole point of the budget is to stay inside it.
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
//! The reset boundary is the provider's civil day in Pacific time. This module does
//! not consult a timezone database: it uses a fixed UTC-8 offset, which is the *later*
//! of the two Pacific offsets (standard time; daylight time is UTC-7). Our day
//! therefore never starts before the provider's, so the counter can only ever lag the
//! real budget by the offset, never overspend it. The cost of that choice is at most
//! one hour of allowance left unused per day while daylight time is in effect, chosen
//! deliberately over the alternative of spending into a window the provider has not
//! opened yet.
//!
//! # State
//!
//! One JSON file per (project, model, day), rewritten atomically after every
//! reservation. A deployment is a single process, so one writer owns each file; there
//! is no cross-process lock. Unreadable or unwritable state fails **closed**: the
//! request is refused with a reason naming the fault, which the caller reports and
//! routes around, rather than spending budget nobody is counting.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use switchyard_protocol::ModelId;

/// Length of one provider day.
const DAY_SECONDS: u64 = 86_400;

/// Offset of the provider's civil day from UTC, in seconds (fixed UTC-8; see module docs).
const PROVIDER_OFFSET_SECONDS: u64 = 8 * 3_600;

/// The provider's civil day number containing `now`.
///
/// Days are counted from the Unix epoch so the value is directly comparable between
/// processes and across a restart.
pub fn provider_day(now: SystemTime) -> u64 {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    seconds.saturating_sub(PROVIDER_OFFSET_SECONDS) / DAY_SECONDS
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
    /// The credential or project the allowance belongs to, for the state file and logs.
    project: String,
    /// The target model whose allowance this is.
    model: ModelId,
    /// Requests permitted per provider day.
    limit: u64,
    /// Where this allowance's counter lives.
    path: PathBuf,
    /// Cached counter, loaded on first use.
    state: Mutex<Option<Counter>>,
}

/// The persisted counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counter {
    day: u64,
    spent: u64,
}

impl DailyRequestAllowance {
    /// An allowance of `limit` requests per provider day for `model` under `project`,
    /// with its counter persisted at `path`.
    pub fn new(project: impl Into<String>, model: ModelId, limit: u64, path: PathBuf) -> Self {
        Self {
            project: project.into(),
            model,
            limit,
            path,
            state: Mutex::new(None),
        }
    }

    /// The configured daily limit.
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// The target model this allowance governs.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// The provider-day number the counter currently belongs to.
    pub fn current_day(&self) -> u64 {
        provider_day(SystemTime::now())
    }

    /// Requests still permitted in the current provider day.
    ///
    /// Reports `0` rather than an error when the state is unusable: a caller asking how
    /// much budget is left must not be told there is some when it cannot be read.
    pub fn remaining(&self) -> u64 {
        self.remaining_at(SystemTime::now())
    }

    /// [`Self::remaining`] against an explicit clock, for tests and replays.
    pub fn remaining_at(&self, now: SystemTime) -> u64 {
        let day = provider_day(now);
        let mut state = self.state.lock();
        match self.load(&mut state) {
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
        let mut state = self.state.lock();
        let counter = self.load(&mut state)?;
        let spent = spent_on(counter, day);
        if spent >= self.limit {
            return Err(ReserveError::Exhausted);
        }
        let next = Counter {
            day,
            spent: spent + 1,
        };
        // Persist before sending. A counter that disagrees with the provider is the
        // failure this exists to prevent, so a failed write must also undo the spend.
        self.persist(next)?;
        *state = Some(next);
        Ok(())
    }

    /// Loads the counter into `state` if it is not cached, returning it.
    fn load(&self, state: &mut Option<Counter>) -> Result<Counter, ReserveError> {
        if let Some(counter) = *state {
            return Ok(counter);
        }
        let counter = match fs::read(&self.path) {
            Ok(bytes) => parse_counter(&bytes).map_err(|reason| {
                ReserveError::Unusable(format!("{} is unreadable: {reason}", self.path.display()))
            })?,
            // A missing file is a fresh allowance, not a fault: the first reservation
            // of a deployment's life creates it.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Counter { day: 0, spent: 0 }
            }
            Err(error) => {
                return Err(ReserveError::Unusable(format!(
                    "{} cannot be read: {error}",
                    self.path.display()
                )));
            }
        };
        *state = Some(counter);
        Ok(counter)
    }

    /// Atomically replaces the counter file.
    fn persist(&self, counter: Counter) -> Result<(), ReserveError> {
        let unusable = |reason: String| ReserveError::Unusable(reason);
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|error| {
                unusable(format!("{} cannot be created: {error}", parent.display()))
            })?;
        }
        // A body the operator can read without this code: the numbers are the record.
        let body = format!(
            "{{\"day\":{},\"spent\":{},\"limit\":{},\"project\":{},\"model\":{}}}\n",
            counter.day,
            counter.spent,
            self.limit,
            serde_json::Value::String(self.project.clone()),
            serde_json::Value::String(self.model.to_string()),
        );
        // Same-directory temporary name so the rename is a replace, never a cross-device
        // copy that could leave a truncated counter behind.
        let temporary = self.path.with_extension("json.tmp");
        fs::write(&temporary, body.as_bytes()).map_err(|error| {
            unusable(format!(
                "{} cannot be written: {error}",
                temporary.display()
            ))
        })?;
        fs::rename(&temporary, &self.path).map_err(|error| {
            let _ = fs::remove_file(&temporary);
            unusable(format!(
                "{} cannot be replaced: {error}",
                self.path.display()
            ))
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

/// The counter file for one (project, model) pair under `directory`.
///
/// The name keeps the pair legible and appends a stable hash of the *exact* pair, so two
/// projects whose names differ only in characters the legible part replaces cannot share
/// a counter. A shared counter would let one project spend another's budget while each
/// believed it was within its own.
pub fn allowance_path(directory: &Path, project: &str, model: &str) -> PathBuf {
    let mut key = String::with_capacity(project.len() + model.len() + 17);
    key.push_str(project);
    key.push('\u{1f}');
    key.push_str(model);
    directory.join(format!(
        "{}-{}-{:016x}.json",
        legible(project),
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

    /// Crossing the provider's midnight reopens the allowance without operator action.
    #[test]
    fn a_new_provider_day_reopens_the_allowance() {
        let directory = scratch("rollover");
        let allowance = allowance(&directory, 1);
        // 2026-11-20T12:00:00Z — mid-afternoon Pacific, well inside one provider day.
        let noon = UNIX_EPOCH + std::time::Duration::from_secs(1_795_176_000);
        allowance.try_reserve_at(noon).expect("within the limit");
        assert!(matches!(
            allowance.try_reserve_at(noon),
            Err(ReserveError::Exhausted)
        ));
        let next_day = noon + std::time::Duration::from_secs(DAY_SECONDS);
        assert_ne!(provider_day(noon), provider_day(next_day));
        allowance
            .try_reserve_at(next_day)
            .expect("a new provider day reopens the allowance");
        // Asked at that day's clock: the reopened allowance is now spent again.
        assert_eq!(allowance.remaining_at(next_day), 0);
    }

    /// The day boundary is the provider's Pacific midnight (08:00Z in standard time),
    /// so a UTC midnight does not reopen the allowance.
    #[test]
    fn the_provider_day_boundary_is_utc_0800() {
        // 2026-11-20T08:00:00Z is exactly one Pacific-midnight boundary.
        let boundary = UNIX_EPOCH + std::time::Duration::from_secs(1_795_161_600);
        let before = boundary - std::time::Duration::from_secs(1);
        assert_eq!(provider_day(boundary), provider_day(before) + 1);
        // A whole UTC day is one provider day, offset by the fixed eight hours.
        assert_eq!(
            provider_day(boundary + std::time::Duration::from_secs(DAY_SECONDS)),
            provider_day(boundary) + 1
        );
        // UTC midnight sits mid-day for the provider, not on its boundary.
        let utc_midnight = boundary - std::time::Duration::from_secs(8 * 3_600);
        assert_eq!(provider_day(utc_midnight), provider_day(boundary) - 1);
        assert_eq!(
            provider_day(utc_midnight - std::time::Duration::from_secs(1)),
            provider_day(utc_midnight)
        );
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

    /// Two projects whose names collapse to the same legible slug keep separate counters.
    #[test]
    fn distinct_pairs_never_share_a_counter() {
        let directory = scratch("collision");
        let first = allowance_path(&directory, "a/b", "m");
        let second = allowance_path(&directory, "a_b", "m");
        assert_ne!(first, second);
        // The same pair resolves to the same path every time.
        assert_eq!(first, allowance_path(&directory, "a/b", "m"));
    }
}
