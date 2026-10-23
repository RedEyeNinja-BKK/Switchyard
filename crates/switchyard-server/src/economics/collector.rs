// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The background collector that reads provider account state.
//!
//! The collector runs on its own thread, entirely outside the request path. It
//! sleeps between rounds and reads each provider in turn. A provider that is slow,
//! down or unauthorized is recorded as such and the round continues, so one provider's
//! failure never starves the others and never reaches an inference request.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::providers::{AccountApi, AccountPublisher, FetchOutcome};

/// How often the collector reads the provider accounts.
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(300);

/// The provider accounts a deployment reads.
pub struct AccountCollector {
    apis: Vec<Arc<dyn AccountApi>>,
    publisher: AccountPublisher,
    interval: Duration,
    /// The account each label was verified to read, once the provider has said so.
    ///
    /// Two labels mapping to one identity are two credentials for one account. Their
    /// figures are the same money, so only one of them may be published.
    identity_of: BTreeMap<String, String>,
    /// The label that publishes each verified identity.
    owner_of: BTreeMap<String, String>,
    /// Labels no longer polled because another label already reads their account.
    shadowed: BTreeSet<String>,
    /// Whether a shadowing collision has already been reported, so the warning is
    /// emitted once per pair rather than once per round.
    reported: BTreeSet<(String, String)>,
}

impl AccountCollector {
    /// Builds a collector over the given provider accounts.
    pub fn new(apis: Vec<Arc<dyn AccountApi>>) -> Self {
        Self {
            apis,
            publisher: AccountPublisher::new(),
            interval: DEFAULT_INTERVAL,
            identity_of: BTreeMap::new(),
            owner_of: BTreeMap::new(),
            shadowed: BTreeSet::new(),
            reported: BTreeSet::new(),
        }
    }

    /// Overrides the collection interval.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// The labels currently polled.
    pub fn polled_labels(&self) -> Vec<String> {
        self.apis
            .iter()
            .map(|api| api.provider().to_string())
            .filter(|label| !self.shadowed.contains(label))
            .collect()
    }

    /// Stops polling a label whose account is already covered, and remembers why.
    ///
    /// The account is still read through the label that owns it, so no measurement is
    /// lost. If the owning label later stops answering, the shadowed label is brought
    /// back, because a second credential for a live account is a real fallback rather
    /// than a duplicate to be discarded.
    fn shadow(&mut self, label: &str, owner: &str) {
        self.shadowed.insert(label.to_string());
        if self.reported.insert((label.to_string(), owner.to_string())) {
            tracing::warn!(
                account = %label,
                covered_by = %owner,
                "two configured clients hold different credentials for one provider \
                 account; the account is read once, under one label, so its figures are \
                 not counted twice. The second client keeps serving traffic and its \
                 served usage is still attributed to its own label"
            );
        }
    }

    /// Brings back the labels that were shadowed by a label that has now failed.
    fn release_shadowed(&mut self, owner: &str) {
        let revived: Vec<String> = self
            .shadowed
            .iter()
            .filter(|label| {
                self.identity_of
                    .get(*label)
                    .and_then(|identity| self.owner_of.get(identity))
                    .is_some_and(|current| current == owner)
            })
            .cloned()
            .collect();
        if revived.is_empty() {
            return;
        }
        for label in &revived {
            self.shadowed.remove(label);
            if let Some(identity) = self.identity_of.get(label) {
                self.owner_of.remove(identity);
            }
            tracing::warn!(
                account = %label,
                was_covered_by = %owner,
                "the client that covered this provider account stopped answering; \
                 its second credential is polled again"
            );
        }
    }

    /// Runs one collection round, reading every provider once.
    ///
    /// The returned value is the number of providers that answered successfully.
    pub async fn collect_once(&mut self) -> usize {
        let mut successful = 0;
        let mut shadow: Option<(String, String)> = None;
        let mut release: Option<String> = None;
        for api in &self.apis {
            let provider = api.provider();
            // A label whose account another label already reads is not polled. The
            // account is still covered, so nothing is missing from the metrics; only
            // the second, redundant read is dropped.
            if self.shadowed.contains(provider) {
                continue;
            }
            let started = Instant::now();
            match api.fetch().await {
                FetchOutcome::Ok(observation) => {
                    // An account the provider identified, whose figures are already
                    // published under a different label, is the same money twice.
                    // Publishing it would double the account's reported balance,
                    // credit and usage, so it is not published at all.
                    if let Some(identity) = &observation.account_identity {
                        match self.owner_of.get(identity) {
                            Some(owner) if owner != provider => {
                                shadow = Some((provider.to_string(), owner.clone()));
                                continue;
                            }
                            Some(_) => {}
                            None => {
                                self.owner_of.insert(identity.clone(), provider.to_string());
                            }
                        }
                        self.identity_of
                            .insert(provider.to_string(), identity.clone());
                    }
                    successful += 1;
                    self.publisher.record_success(
                        provider,
                        observation.as_ref(),
                        started.elapsed(),
                    );
                }
                FetchOutcome::Unauthorized => {
                    self.publisher.record_unauthorized(provider);
                    release = Some(provider.to_string());
                }
                FetchOutcome::Failed(_) => {
                    self.publisher.record_failure(provider);
                    release = Some(provider.to_string());
                }
            }
        }
        if let Some((label, owner)) = shadow {
            self.shadow(&label, &owner);
        }
        if let Some(owner) = release {
            // A label that has stopped answering releases the clients it was covering,
            // so a second credential can take over the account instead of the account
            // going unread because its first reader broke.
            self.release_shadowed(&owner);
        }
        successful
    }

    /// Runs the collector until the process ends.
    ///
    /// Each round is separated by the configured interval. The sleep happens between
    /// rounds rather than before the first, so the first observation is immediate.
    pub async fn run(mut self) {
        loop {
            self.collect_once().await;
            tokio::time::sleep(self.interval).await;
        }
    }

    /// Starts the collector on its own thread and returns the handle.
    ///
    /// The thread carries a runtime of its own. It shares no state with the request
    /// path beyond the metric registry, which is process-wide and thread-safe, so a
    /// provider that is slow or down can never delay an inference request.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("switchyard-economics-accounts".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        tracing::error!(
                            %error,
                            "economics account collector runtime failed to start"
                        );
                        return;
                    }
                };
                runtime.block_on(self.run());
            })
            .expect("failed to start the economics account collector")
    }
}
