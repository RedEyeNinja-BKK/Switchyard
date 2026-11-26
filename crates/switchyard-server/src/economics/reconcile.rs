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
use std::time::SystemTime;

use switchyard_llm_client::{DailyRequestAllowance, ObservationTicket, ProviderReading};

use super::providers::{AccountObservation, RequestAllowanceReading};

/// One account's reading, bound to the counter it governs.
pub struct AllowanceReconciler {
    /// The label the reading must currently arrive under.
    ///
    /// Assigned from the credential group the account reader was built for, so it is an
    /// account identity rather than a provider name: two accounts of the same provider
    /// carry different labels and cannot cross. When the provider itself says two labels
    /// are one account, this follows the surviving label -- see [`Reconcilers::absorb`].
    account: String,
    /// The label this reconciler was configured with, so a shadow can be undone.
    home: String,
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
        let account = account.into();
        Self {
            home: account.clone(),
            account,
            pool,
            allowance,
        }
    }

    /// The account label this reconciler currently listens on.
    pub fn account(&self) -> &str {
        &self.account
    }

    /// The pool this reconciler selects.
    pub fn pool(&self) -> &'static str {
        self.pool
    }
}

/// The watermarks for counters bound to one account, taken before that account was read.
pub struct PendingObservation {
    /// One entry per reconciler, in the same order. `None` where no watermark could be
    /// taken, which makes the application anchor at zero rather than forgive traffic.
    tickets: Vec<Option<ObservationTicket>>,
}

impl PendingObservation {
    /// How many watermarks were captured.
    pub fn captured(&self) -> usize {
        self.tickets
            .iter()
            .filter(|ticket| ticket.is_some())
            .count()
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
    ///
    /// Reported as configured, not as currently pointed: a shadowed label still names the
    /// counter it owns, and a diagnostic that hid it would make a bound counter look
    /// unwired.
    pub fn bindings(&self) -> Vec<(String, String)> {
        self.entries
            .iter()
            .map(|entry| (entry.home.clone(), entry.pool.to_string()))
            .collect()
    }

    /// Takes a watermark for every counter bound to `account`, before it is read.
    ///
    /// Taken ahead of the network call so requests sent while the read is outstanding are
    /// still subtracted from the provider's figure. Nothing here touches the network, and
    /// each counter's lock is held only for its own read.
    pub fn begin_observation(&self, account: &str, now: SystemTime) -> PendingObservation {
        PendingObservation {
            tickets: self
                .entries
                .iter()
                .map(|entry| {
                    (entry.account == account)
                        .then(|| entry.allowance.begin_observation_at(now).ok())
                        .flatten()
                })
                .collect(),
        }
    }

    /// Follows the label that actually reads a shared account.
    ///
    /// Two credentials the provider says belong to one account are one quota pool. The
    /// collector stops polling the shadowed label, so unless its counters follow the
    /// surviving label they would silently stop being reconciled and admit forever on
    /// their local budget -- the exact failure the shadow exists to prevent.
    pub fn absorb(&mut self, shadowed: &str, owner: &str) {
        for entry in &mut self.entries {
            if entry.account == shadowed {
                entry.account = owner.to_string();
            }
        }
    }

    /// Restores the counters absorbed by `owner` to the labels they were configured with.
    ///
    /// Called when the owning label stops answering and the shadowed label is polled again.
    /// Entries whose own home is `owner` are unchanged, so this cannot detach a counter
    /// from the account it was actually bound to.
    pub fn release(&mut self, owner: &str) {
        for entry in &mut self.entries {
            if entry.account == owner {
                entry.account.clone_from(&entry.home);
            }
        }
    }

    /// Applies a reading that is already in hand, with no read outstanding.
    ///
    /// The watermark is taken at `now`, so this is only correct when nothing was sent
    /// between reading the provider and calling here. A collector whose read is a network
    /// call must use [`Self::apply_at`] with the watermark it took before the call, or the
    /// traffic sent during it is forgiven.
    pub fn apply(
        &self,
        account: &str,
        observation: &AccountObservation,
        now: SystemTime,
    ) -> Applied {
        let pending = self.begin_observation(account, now);
        self.apply_at(account, observation, &pending, now)
    }

    /// Writes the account's reading into every counter bound to it.
    ///
    /// A refusal from one counter never stops the others, and never propagates to the
    /// caller: the collector's job is to observe, and a counter that cannot be written is
    /// a reported condition rather than a reason to abandon the round. The counter itself
    /// keeps admitting on its local budget when no current reading exists, so the effect
    /// of a write failure is less protection, never a wrong refusal.
    pub fn apply_at(
        &self,
        account: &str,
        observation: &AccountObservation,
        pending: &PendingObservation,
        now: SystemTime,
    ) -> Applied {
        let mut applied = Applied::default();
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.account != account {
                continue;
            }
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
            // A missing watermark is not a reason to skip the write: it anchors at zero,
            // which understates headroom instead of forgiving in-flight traffic.
            let ticket = pending
                .tickets
                .get(index)
                .copied()
                .flatten()
                .unwrap_or_else(ObservationTicket::unwatermarked);
            match entry
                .allowance
                .reconcile_at(provider_reading(reading), ticket, now)
            {
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

    /// Applies a reading against a fixed clock, with the watermark taken at that instant.
    fn apply_all(
        reconcilers: &Reconcilers,
        account: &str,
        observation: &AccountObservation,
    ) -> Applied {
        let now = at(LATE_UTC_DAY);
        let pending = reconcilers.begin_observation(account, now);
        reconcilers.apply_at(account, observation, &pending, now)
    }

    /// A request sent while the account read was outstanding is subtracted from the
    /// provider's figure. The watermark is taken before the read, so the application
    /// measures in-flight traffic instead of forgiving it.
    #[test]
    fn a_reading_is_applied_against_the_watermark_taken_before_the_read() {
        let allowance = counter("watermark", 1000);
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter",
            "free_model_daily_requests",
            Arc::clone(&allowance),
        )]);

        let now = at(LATE_UTC_DAY);
        let pending = reconcilers.begin_observation("openrouter", now);
        assert_eq!(pending.captured(), 1);
        // Five requests leave while the collector is waiting for the provider.
        for _ in 0..5 {
            allowance
                .try_reserve_at(now)
                .expect("the local budget has room");
        }

        let applied = reconcilers.apply_at(
            "openrouter",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 10, 1000, 990)],
            ),
            &pending,
            now,
        );

        assert_eq!(applied.written, 1);
        assert_eq!(
            allowance.remaining_at(now),
            985,
            "the five in-flight sends are not in the provider's 990"
        );
    }

    /// A reading the counter cannot order against what it holds is refused, and the
    /// refusal is reported rather than silently dropped.
    #[test]
    fn an_out_of_order_reading_is_refused_and_reported() {
        let allowance = counter("out-of-order", 1000);
        let reconcilers = Reconcilers::new(vec![AllowanceReconciler::new(
            "openrouter",
            "free_model_daily_requests",
            Arc::clone(&allowance),
        )]);

        // The newer reading first.
        let applied = apply_all(
            &reconcilers,
            "openrouter",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 997, 1000, 3)],
            ),
        );
        assert_eq!(applied.written, 1);

        // The same reading delivered again is not newer, so it cannot move the anchor.
        let applied = apply_all(
            &reconcilers,
            "openrouter",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 997, 1000, 3)],
            ),
        );
        assert_eq!(applied.written, 0);
        assert_eq!(
            applied.refused.len(),
            1,
            "a refused reading must be reported, not dropped"
        );
        assert!(
            applied.refused[0].contains("later reading"),
            "{:?}",
            applied.refused
        );
    }

    /// A label the provider says belongs to another label's account follows that account.
    /// Its counter keeps being reconciled instead of silently admitting on its local budget.
    #[test]
    fn a_shadowed_label_follows_the_account_that_reads_it() {
        let owner = counter("follow-owner", 1000);
        let second = counter("follow-second", 1000);
        let mut reconcilers = Reconcilers::new(vec![
            AllowanceReconciler::new("owner", "free_model_daily_requests", Arc::clone(&owner)),
            AllowanceReconciler::new("second", "free_model_daily_requests", Arc::clone(&second)),
        ]);

        reconcilers.absorb("second", "owner");
        let applied = apply_all(
            &reconcilers,
            "owner",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );

        assert_eq!(applied.written, 2, "both counters are one account");
        assert!(matches!(
            second.try_reserve_at(at(LATE_UTC_DAY)),
            Err(switchyard_llm_client::ReserveError::ProviderExhausted { .. })
        ));
        // The diagnostic still names the counter the operator configured.
        assert_eq!(
            reconcilers.bindings(),
            vec![
                ("owner".to_string(), "free_model_daily_requests".to_string()),
                (
                    "second".to_string(),
                    "free_model_daily_requests".to_string()
                )
            ]
        );
    }

    /// Releasing a shadow returns its counter to its own label, so a second credential can
    /// take over an account whose first reader stopped answering.
    #[test]
    fn releasing_a_shadow_returns_the_counter_to_its_own_label() {
        let owner = counter("release-owner", 1000);
        let second = counter("release-second", 1000);
        let mut reconcilers = Reconcilers::new(vec![
            AllowanceReconciler::new("owner", "free_model_daily_requests", Arc::clone(&owner)),
            AllowanceReconciler::new("second", "free_model_daily_requests", Arc::clone(&second)),
        ]);

        reconcilers.absorb("second", "owner");
        reconcilers.release("owner");

        let applied = apply_all(
            &reconcilers,
            "owner",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );
        assert_eq!(
            applied.written, 1,
            "only the owner's own counter is written"
        );
        second
            .try_reserve_at(at(LATE_UTC_DAY))
            .expect("the released counter is no longer bound to the owner's reading");
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

        let applied = apply_all(
            &reconcilers,
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

        let applied = apply_all(
            &reconcilers,
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

        let applied = apply_all(
            &reconcilers,
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

        let applied = apply_all(
            &reconcilers,
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

        apply_all(
            &reconcilers,
            "openrouter",
            &observation(
                None,
                vec![reading("free_model_daily_requests", 1000, 1000, 0)],
            ),
        );
        let applied = apply_all(&reconcilers, "openrouter", &observation(None, vec![]));

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
