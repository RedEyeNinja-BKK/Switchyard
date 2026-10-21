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

use super::account_apis::{CodexAllowanceAccount, DeepSeekAccount, OpenRouterAccount};
use super::collector::AccountCollector;
use super::providers::AccountApi;

/// How often the calendar counters are republished.
const CALENDAR_PUBLISH_INTERVAL: Duration = Duration::from_secs(30);

/// A credential, held so that two clients can be compared without the secret ever
/// becoming printable.
///
/// The type deliberately has no `Debug` implementation, so a structure holding one
/// cannot be logged by accident.
struct Credential(String);

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
    let mut apis: Vec<Arc<dyn AccountApi>> = Vec::new();
    // The provider label already read, with the credential it was read with. The
    // provider label is the account's identity in the metric namespace, so one
    // label means one account.
    let mut read: BTreeMap<String, Credential> = BTreeMap::new();

    for client in facts.clients.values() {
        let Some(key_env) = &client.api_key_env else {
            continue;
        };
        let Some(credential) = env(key_env) else {
            continue;
        };
        if credential.is_empty() {
            continue;
        }

        let (provider, api): (String, Arc<dyn AccountApi>) = match client.name.as_str() {
            "openrouter" => (
                "openrouter".to_string(),
                Arc::new(OpenRouterAccount::new(credential.clone())),
            ),
            "deepseek" => (
                "deepseek".to_string(),
                Arc::new(DeepSeekAccount::new(credential.clone())),
            ),
            "openai" | "openai_nt" => {
                let account_id = env("SY_OPENAI_ACCOUNT_ID");
                (
                    "openai".to_string(),
                    Arc::new(CodexAllowanceAccount::new(credential.clone(), account_id)),
                )
            }
            // A provider with no account surface produces no reader. It is not
            // recorded as a zero balance, because a provider that exposes nothing
            // has not reported a balance of zero.
            _ => continue,
        };

        if let Some(existing) = read.get(&provider) {
            // Two clients for one provider normally share one credential, which is
            // the ordinary case and needs no report. Different credentials mean two
            // accounts behind one label, which the metric namespace cannot
            // represent; that is a configuration ambiguity worth saying out loud
            // rather than resolving silently.
            if existing.0 != credential {
                tracing::warn!(
                    provider = %provider,
                    client = %client.name,
                    "several clients share one provider account label but hold different \
                     credentials; only the first is read, and the metric namespace cannot \
                     distinguish the accounts"
                );
            }
            continue;
        }

        read.insert(provider, Credential(credential));
        apis.push(api);
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
                std::thread::sleep(CALENDAR_PUBLISH_INTERVAL);
            }
        })
        .expect("failed to start the economics calendar publisher")
}
