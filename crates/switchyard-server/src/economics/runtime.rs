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
use super::reconcile::{AllowanceReconciler, Reconcilers};
use switchyard_protocol::ModelId;

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
/// One provider account reader, with the facts needed to bind it to counters.
///
/// The label is an account identity, not a provider name: it is derived from the
/// credential group, so two credentials for one provider carry two labels and one
/// credential shared by two clients carries one. That is what lets a reading be written
/// into exactly the counters of the account it describes.
pub struct AccountReader {
    /// The reader the collector polls.
    pub api: Arc<dyn AccountApi>,
    /// The account label the reading arrives under.
    pub label: String,
    /// The account surface this reader speaks.
    pub surface: AccountSurface,
    /// The clients whose credential built this reader.
    pub clients: Vec<String>,
}

pub fn build_account_readers(
    facts: &switchyard_runner::DeploymentFacts,
    env: &dyn Fn(&str) -> Option<String>,
) -> Vec<AccountReader> {
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

    let mut apis: Vec<AccountReader> = Vec::new();
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
        apis.push(AccountReader {
            api: Arc::new(LabeledAccount::new(label.clone(), api)),
            label,
            surface,
            clients: names,
        });
    }
    apis
}

/// The account readers a deployment authorizes, in the order they were built.
pub fn build_account_apis(
    facts: &switchyard_runner::DeploymentFacts,
    env: &dyn Fn(&str) -> Option<String>,
) -> Vec<Arc<dyn AccountApi>> {
    build_account_readers(facts, env)
        .into_iter()
        .map(|reader| reader.api)
        .collect()
}

/// Binds every declared provider pool to the counter it governs.
///
/// A binding is only made where all three facts meet: an account reader was built from a
/// credential, a client that presents that credential declares a pool, and that client's
/// quota group owns an account-scoped counter. Any one of them missing produces no binding
/// rather than a guessed one -- a counter reconciled from the wrong account is worse than
/// a counter that only knows its own traffic, because it would refuse real work.
pub fn build_reconcilers(
    readers: &[AccountReader],
    facts: &switchyard_runner::DeploymentFacts,
    allowances: &std::collections::HashMap<
        (String, ModelId),
        Arc<switchyard_llm_client::DailyRequestAllowance>,
    >,
) -> Reconcilers {
    let mut entries = Vec::new();
    for reader in readers {
        // Only an account surface that reports request pools can reconcile a counter.
        // Another provider's reader is simply not consulted.
        if reader.surface != AccountSurface::OpenRouter {
            continue;
        }
        for name in &reader.clients {
            let Some(client) = facts.clients.get(name) else {
                continue;
            };
            let Some(provider) = client.quota_provider else {
                continue;
            };
            let group = client.quota_group.clone().unwrap_or_else(|| name.clone());
            let counter = ModelId::from(switchyard_llm_client::ACCOUNT_SCOPE_MODEL);
            let Some(allowance) = allowances.get(&(group.clone(), counter)) else {
                // The loader refuses this shape, so reaching it means the two views of the
                // configuration disagree. Reported rather than guessed.
                tracing::warn!(
                    client = %name,
                    group = %group,
                    pool = %provider.as_str(),
                    "a provider pool is declared but no account-scoped counter was built for \
                     the group; nothing will be reconciled"
                );
                continue;
            };
            entries.push(AllowanceReconciler::new(
                reader.label.clone(),
                provider.pool(),
                Arc::clone(allowance),
            ));
        }
    }
    Reconcilers::new(entries)
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
    allowances: &std::collections::HashMap<
        (String, ModelId),
        Arc<switchyard_llm_client::DailyRequestAllowance>,
    >,
) -> std::thread::JoinHandle<()> {
    let readers = build_account_readers(facts, env);
    if !readers.is_empty() {
        let reconcilers = build_reconcilers(&readers, facts, allowances);
        if !reconcilers.is_empty() {
            // Stated at startup because a reconciliation that was configured but not wired
            // is indistinguishable, on a dashboard, from one that is working.
            for (account, pool) in reconcilers.bindings() {
                tracing::info!(%account, %pool, "provider allowance reconciliation is bound");
            }
        }
        let api_handles = readers.into_iter().map(|reader| reader.api).collect();
        AccountCollector::new(api_handles)
            .with_reconcilers(reconcilers)
            .spawn();
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
