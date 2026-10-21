// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The background collector that reads provider account state.
//!
//! The collector runs on its own thread, entirely outside the request path. It
//! sleeps between rounds and reads each provider in turn. A provider that is slow,
//! down or unauthorized is recorded as such and the round continues, so one provider's
//! failure never starves the others and never reaches an inference request.

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
}

impl AccountCollector {
    /// Builds a collector over the given provider accounts.
    pub fn new(apis: Vec<Arc<dyn AccountApi>>) -> Self {
        Self {
            apis,
            publisher: AccountPublisher::new(),
            interval: DEFAULT_INTERVAL,
        }
    }

    /// Overrides the collection interval.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Runs one collection round, reading every provider once.
    ///
    /// The returned value is the number of providers that answered successfully.
    pub async fn collect_once(&mut self) -> usize {
        let mut successful = 0;
        for api in &self.apis {
            let provider = api.provider();
            let started = Instant::now();
            match api.fetch().await {
                FetchOutcome::Ok(observation) => {
                    successful += 1;
                    self.publisher
                        .record_success(provider, &observation, started.elapsed());
                }
                FetchOutcome::Unauthorized => {
                    self.publisher.record_unauthorized(provider);
                }
                FetchOutcome::Failed(_) => {
                    self.publisher.record_failure(provider);
                }
            }
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
