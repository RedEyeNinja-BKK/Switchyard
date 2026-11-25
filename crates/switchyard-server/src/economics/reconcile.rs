// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writing a provider's own allowance reading into the counter that admits traffic.
//!
//! The collector reads provider accounts on its own thread. A reading is a fact about an
//! *account*: it counts every request the account made today, including traffic this
//! deployment never sent, which is exactly what a local counter cannot see. Admission
//! therefore has to consult it, and this module is the only path by which a provider
//! figure reaches a counter.
//!
//! Two boundaries are load-bearing:
//!
//! * **Account isolation.** A reconciler is built from a specific credential group, so a
//!   reading from one account's key can only ever be written into counters whose own
//!   clients present that same credential. Stephen's account emptying must not refuse the
//!   owner's traffic, and nothing here selects a counter by provider name alone.
//! * **A reading is data, not an instruction.** A malformed or stale reading is refused by
//!   the allowance itself (it parses to `None`, or it is not fresh, in which case admission
//!   falls back to the local budget). Nothing in this layer converts a failed refresh into
//!   a refusal, so a collector outage cannot become a serving outage.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use switchyard_llm_client::{DailyRequestAllowance, ProviderReading};

use super::providers::{AccountObservation, RequestAllowanceReading};

/// One account's reading, bound to the counter it governs.
pub struct AllowanceReconciler {
    /// The account label the reading must arrive under.
    ///
    /// Assigned from the credential group the account reader was built for, so it is an
    /// account identity rather than a provider name: two accounts of the same provider
    /// carry different labels and cannot cross.
    account: String,
    /// The provider's own name for the pool, used to select the reading.
    pool: &'static str,
    /// The counter this reading governs.
    allowance: Arc<DailyRequestAllowance>,
}

impl AllowanceReconciler {
    /// Binds one provider pool, read from `account`, to `allowance`.
    pub fn new(
        account: impl Into<String>,
        pool: &'static str,
        allowance: Arc<DailyRequestAllowance>,
    ) -> Self {
        Self {
            account: account.into(),
            pool,
            allowance,
        }
    }

    /// The account label this reconciler listens on.
    pub fn account(&self) -> &str {
        &self.account
    }

    /// The pool this reconciler selects.
    pub fn pool(&self) -> &str {
        self.pool
    }
}

/// The set of bound counters, applied to each successful account read.
#[derive(Default)]
pub struct Reconcilers {
    entries: Vec<AllowanceReconciler>,
    /// Pairs already reported, so a persistent refusal is logged once rather than every
    /// interval for the life of the process.
    reported: Mutex<BTreeSet<(String, String, String)>>,
}

/// What one application of a reading did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// Reconcilers whose reading was written and accepted.
    pub written: usize,
    /// Reconcilers that matched the account but found no reading for their pool.
    pub absent: usize,
    /// Reconcilers whose write was refused, with the reason.
    pub refused: Vec<String>,
}

impl Reconcilers {
    /// Builds the set from already-bound entries.
    pub fn new(entries: Vec<AllowanceReconciler>) -> Self {
        Self {
            entries,
            reported: Mutex::new(BTreeSet::new()),
        }
    }

    /// How many counters are bound.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no counter is bound.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The (account, pool) pairs currently bound, for a startup diagnostic.
    pub fn bindings(&self) -> Vec<(String, String)> {
        self.entries
            .iter()
            .map(|entry| (entry.account.clone(), entry.pool.to_string()))
            .collect()
    }

    /// Writes the account's reading into every counter bound to it.
    ///
    /// A refusal from one counter never stops the others, and never propagates to the
    /// caller: the collector's job is to observe, and a counter that cannot be written is
    /// a reported condition rather than a reason to abandon the round. The counter itself
    /// keeps admitting on its local budget when no current reading exists, so the effect
    /// of a write failure is less protection, never a wrong refusal.
    pub fn apply(&self, account: &str, observation: &AccountObservation) -> Applied {
        let mut applied = Applied::default();
        for entry in self.entries.iter().filter(|entry| entry.account == account) {
            let Some(reading) = observation
                .request_allowances
                .iter()
                .find(|reading| reading.pool == entry.pool)
            else {
                // The provider answered but did not report this pool. That is a missing
                // fact, not a spent one, so the counter is left as it was -- its existing
                // reading ages out on its own instead of being replaced by a zero.
                applied.absent += 1;
                self.report_once(
                    account,
                    entry.pool,
                    "the provider answered without reporting this pool; the counter keeps its \
                     previous reading, which ages out on its own",
                );
                continue;
            };
            match entry.allowance.reconcile(provider_reading(reading)) {
                Ok(()) => applied.written += 1,
                Err(error) => {
                    let reason = format!("{error:?}");
                    self.report_once(account, entry.pool, &reason);
                    applied.refused.push(reason);
                }
            }
        }
        applied
    }

    fn report_once(&self, account: &str, pool: &str, reason: &str) {
        let key = (account.to_string(), pool.to_string(), reason.to_string());
        let inserted = self
            .reported
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key);
        if inserted {
            tracing::warn!(
                account = %account,
                pool = %pool,
                %reason,
                "a provider allowance reading was not written into its counter"
            );
        }
    }
}

/// Converts a collector reading into the counter's own type.
///
/// The two exist separately on purpose: the collector type is what the account API
/// parses, and the counter type is what admission reads. Keeping them apart means a change
/// to one provider's wire shape cannot alter the admission contract.
fn provider_reading(reading: &RequestAllowanceReading) -> ProviderReading {
    ProviderReading {
        used: reading.used,
        limit: reading.limit,
        remaining: reading.remaining,
        observed_at: reading.observed_at,
        reset_at: reading.reset_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(seconds)
    }

    // 2026-01-15T23:00:00Z
    const LATE_UTC_DAY: u64 = 1_768_518_000;

    fn counter(label: &str, limit: u64) -> Arc<DailyRequestAllowance> {
        let directory = std::env::temp_dir()
            .join("switchyard-reconcile-tests")
            .join(format!(
                "{label}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
        Arc::new(DailyRequestAllowance::new_on(
            "openrouter-free",
            "*account*".into(),
            limit,
            switchyard_llm_client::account_allowance_path(&directory, "openrouter-free"),
            switchyard_llm_client::DayBoundary::Utc,
        ))
    }

    fn observation(
        account: Option<&str>,
        readings: Vec<RequestAllowanceReading>,
    ) -> AccountObservation {
        AccountObservation {
            account_identity: account.map(str::to_string),
            request_allowances: readings,
            ..AccountObservation::default()
        }
    }

    fn reading(pool: &str, used: u64, limit: u64, remaining: u64) -> RequestAllowanceReading {
        RequestAllowanceReading::new(pool, used, limit, remaining, at(LATE_UTC_DAY), None)
    }

    /// The reading reaches the counter: after a provider says the pool is spent, the
    /// counter refuses with the provider's reason rather than with its own.
    #[test]
    fn a_provider_reading_is_written_into_its_counter() {
        let allowance = counter("written", 1000);
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter",
            "free_model_daily_requests",
            Arc::clone(&allowance),
        )]);

        let applied = reconcilers.apply(
            "openrouter",
            &observation(
                Some("workspace:w"),
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );

        assert_eq!(applied.written, 1);
        assert!(applied.refused.is_empty());
        assert!(matches!(
            allowance.try_reserve_at(at(LATE_UTC_DAY)),
            Err(switchyard_llm_client::ReserveError::ProviderExhausted { limit: 1000 })
        ));
    }

    /// A reading for a different pool is not written. The account API reports several
    /// pools; selecting by name is what keeps the free-model bucket from gating a counter
    /// that is governed by another one.
    #[test]
    fn a_reading_for_another_pool_is_not_written() {
        let allowance = counter("other-pool", 1000);
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter",
            "free_model_daily_requests",
            Arc::clone(&allowance),
        )]);

        let applied = reconcilers.apply(
            "openrouter",
            &observation(None, vec![reading("some_other_pool", 1000, 1000, 0)]),
        );

        assert_eq!(applied.written, 0);
        assert_eq!(applied.absent, 1);
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("a counter must not be refused by a pool it does not govern");
    }

    /// A reading arriving under another account's label is not written. This is the
    /// isolation boundary: one account emptying must not refuse another account's traffic.
    #[test]
    fn a_reading_from_one_account_does_not_touch_another() {
        let owner = counter("owner", 1000);
        let stephen = counter("stephen", 1000);
        let reconcilers = Reconcilers::new(vec![
            AllowanceReconciler::new("owner", "free_model_daily_requests", Arc::clone(&owner)),
            AllowanceReconciler::new("stephen", "free_model_daily_requests", Arc::clone(&stephen)),
        ]);

        let applied = reconcilers.apply(
            "stephen",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );

        assert_eq!(applied.written, 1);
        stephen
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect_err("the account that reported exhaustion must refuse");
        owner
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("the other account must be untouched");
    }

    /// An account the collector no longer polls is not applied to. Without this, a
    /// provider that began answering for a different label would silently move traffic
    /// between budgets.
    #[test]
    fn an_unknown_account_label_writes_nothing() {
        let allowance = counter("unknown-label", 1000);
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter",
            "free_model_daily_requests",
            Arc::clone(&allowance),
        )]);

        let applied = reconcilers.apply(
            "someone-else",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );

        assert_eq!(applied, Applied::default());
        allowance
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("an unmatched label must change nothing");
    }

    /// A provider that answers without the pool leaves the previous reading in place. The
    /// counter is not reset to zero and not marked exhausted: an absent figure is not a
    /// spent one.
    #[test]
    fn an_absent_pool_leaves_the_previous_reading_in_place() {
        let allowance = counter("absent", 1000);
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter",
            "free_model_daily_requests",
            Arc::clone(&allowance),
        )]);

        reconcilers.apply(
            "openrouter",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );
        let applied = reconcilers.apply("openrouter", &observation(None, vec![]));

        assert_eq!(applied.written, 0);
        assert_eq!(applied.absent, 1);
        assert!(matches!(
            allowance.try_reserve_at(at(LATE_UTC_DAY)),
            Err(switchyard_llm_client::ReserveError::ProviderExhausted { .. })
        ));
    }

    /// The counters bound to one account and one provider are reported, so a startup
    /// diagnostic can state what was actually wired rather than what was intended.
    #[test]
    fn bindings_name_the_account_and_the_pool() {
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter-stephen",
            "free_model_daily_requests",
            counter("bindings", 10),
        )]);
        assert_eq!(
            reconcilers.bindings(),
            vec![(
                "openrouter-stephen".to_string(),
                "free_model_daily_requests".to_string()
            )]
        );
    }
}
