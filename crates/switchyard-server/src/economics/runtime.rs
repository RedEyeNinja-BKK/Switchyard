// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Starting and running the economics collection.
//!
//! Two background activities, both outside the request path:
//!
//! * The account collector reads provider account state with Switchyard's own
//!   credentials, on its own thread.
//! * The calendar publisher republishes the Bangkok calendar-day counters, so the
//!   day view does not depend on a dashboard query being issued.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::account_apis::{
    AccountSurface, CodexAllowanceAccount, DeepSeekAccount, OpenRouterAccount,
    account_surface_for_base_url,
};
use super::collector::AccountCollector;
use super::providers::{AccountApi, LabeledAccount};

/// How often the calendar counters are republished.
const CALENDAR_PUBLISH_INTERVAL: Duration = Duration::from_secs(30);

/// Builds the account readers a deployment's configuration authorizes.
///
/// A reader is only built for a provider whose credential the deployment declares
/// and whose environment variable actually holds a value. A provider with no
/// credential produces no reader and therefore no account series, which is the
/// honest outcome: Switchyard cannot read an account it has no credential for.
///
/// Exactly one reader is built per provider account. A deployment may name several
/// clients for the same provider — the `openai` and `openai_nt` clients share one
/// credential and one base URL and differ only in the reasoning dialect they speak.
/// They describe one account, so they must produce one reader: two readers would
/// read the same account endpoint twice per interval and report two account reads
/// for a single account, which is a doubled count of something that happened once.
pub fn build_account_apis(
    facts: &switchyard_runner::DeploymentFacts,
    env: &dyn Fn(&str) -> Option<String>,
) -> Vec<Arc<dyn AccountApi>> {
    // Group the clients by the account they read: one group per (surface,
    // credential). Two clients holding one credential read one account, so they
    // produce one reader; two clients holding different credentials are two readers
    // until the provider itself says otherwise.
    let mut groups: BTreeMap<(AccountSurface, String), Vec<String>> = BTreeMap::new();

    for client in facts.clients.values() {
        let Some(surface) = account_surface_for_base_url(&client.base_url) else {
            // A provider with no account surface produces no reader. It is not
            // recorded as a zero balance, because a provider that exposes nothing has
            // not reported a balance of zero.
            continue;
        };
        let Some(key_env) = &client.api_key_env else {
            continue;
        };
        let Some(credential) = env(key_env) else {
            continue;
        };
        if credential.is_empty() {
            continue;
        }
        groups
            .entry((surface, credential))
            .or_default()
            .push(client.name.clone());
    }

    let mut apis: Vec<Arc<dyn AccountApi>> = Vec::new();
    for ((surface, credential), mut names) in groups {
        names.sort();

        // The account's label in the metric namespace. Each label means one account,
        // so the label must be unique per account and stable across restarts.
        //
        // A client whose name already matches the surface is preferred, which keeps
        // the labels a dashboard selects on — `openai`, `deepseek`, `openrouter` —
        // unchanged for the deployments that already use them. Otherwise the first
        // name in sorted order is used, so the choice does not depend on the order the
        // configuration happened to declare the clients in.
        let canonical = surface.canonical_label();
        let label = if names.iter().any(|name| name == canonical) {
            canonical.to_string()
        } else {
            names[0].clone()
        };

        let api: Arc<dyn AccountApi> = match surface {
            AccountSurface::OpenRouter => Arc::new(OpenRouterAccount::new(credential)),
            AccountSurface::DeepSeek => Arc::new(DeepSeekAccount::new(credential)),
            AccountSurface::CodexAllowance => {
                let account_id = env("SY_OPENAI_ACCOUNT_ID");
                Arc::new(CodexAllowanceAccount::new(credential, account_id))
            }
        };
        apis.push(Arc::new(LabeledAccount::new(label, api)));
    }
    apis
}

/// Starts the background economics collection.
///
/// The returned handle keeps the calendar publisher thread alive. The caller owns it
/// for the life of the process.
///
/// The account collector is started only for the providers whose credential the
/// deployment declares and whose environment variable holds a value. A deployment
/// with no such credential starts no collector, and therefore publishes no account
/// series, which is the honest outcome: Switchyard cannot read an account it holds
/// no credential for.
pub fn start(
    facts: &switchyard_runner::DeploymentFacts,
    calendar: Arc<super::CalendarWindow>,
    env: &dyn Fn(&str) -> Option<String>,
) -> std::thread::JoinHandle<()> {
    let apis = build_account_apis(facts, env);
    if !apis.is_empty() {
        AccountCollector::new(apis).spawn();
    }
    std::thread::Builder::new()
        .name("switchyard-economics-calendar".to_string())
        .spawn(move || {
            loop {
                super::calendar::publish_calendar(&calendar);
                // The rolling windows are republished on the same cadence. They are
                // gauges over events the request path records, so a publish only
                // recomputes totals and drops the buckets no window can still reach.
                super::rolling::publish_global();
                std::thread::sleep(CALENDAR_PUBLISH_INTERVAL);
            }
        })
        .expect("failed to start the economics calendar publisher")
}
