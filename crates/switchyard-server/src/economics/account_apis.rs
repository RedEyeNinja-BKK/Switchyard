// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The concrete provider account APIs Switchyard reads.
//!
//! Each implementation uses the same credential Switchyard uses to serve inference
//! from that provider. Nothing here creates, rotates or broadens a credential.

use std::time::Duration;

use serde_json::Value;

use super::providers::{
    AccountAmount, AccountApi, AccountBalanceComponent, AccountObservation, AccountScope,
    FetchOutcome, ReportingPeriod,
};

/// The HTTP client the account readers use.
///
/// A dedicated client with a short timeout, so a provider that stalls cannot hold the
/// collection task and starve the other providers' observations.
fn account_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

/// Reads OpenRouter's key usage and account credits.
///
/// Both endpoints are documented read-only surfaces. The key endpoint reports usage
/// over several windows; the credits endpoint reports the account-wide purchased
/// credit and the usage drawn against it.
pub struct OpenRouterAccount {
    api_key: String,
    base_url: String,
}

impl OpenRouterAccount {
    /// Builds the reader from the credential Switchyard serves OpenRouter with.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
        }
    }

    /// Overrides the API base, for a test double.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    async fn get_json(&self, path: &str) -> Result<Value, FetchOutcome> {
        let response = account_client()
            .get(format!("{}{path}", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|error| FetchOutcome::Failed(error.to_string()))?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(FetchOutcome::Unauthorized);
        }
        if !status.is_success() {
            return Err(FetchOutcome::Failed(format!("HTTP {status}")));
        }
        response
            .json::<Value>()
            .await
            .map_err(|error| FetchOutcome::Failed(error.to_string()))
    }
}

#[async_trait::async_trait]
impl AccountApi for OpenRouterAccount {
    fn provider(&self) -> &str {
        "openrouter"
    }

    async fn fetch(&self) -> FetchOutcome {
        self.fetch_async().await
    }
}

impl OpenRouterAccount {
    /// Reads the key usage and the account credits.
    async fn fetch_async(&self) -> FetchOutcome {
        let mut observation = AccountObservation::default();

        // The key endpoint reports the usage windows. An authorization failure is
        // reported as such rather than swallowed: a provider that rejected the
        // credential must not be recorded as one that reported no account state,
        // because those are different facts and only one of them is actionable.
        match self.get_json("/key").await {
            Ok(body) => {
                if let Some(data) = body.get("data") {
                    // The account this key belongs to, as the provider itself reports it.
                    // Two keys on one account share this value, which is what lets the
                    // collector recognise them as one account instead of counting the
                    // same money twice. It is an opaque identifier, not a secret: it
                    // grants nothing and is never published.
                    if let Some(owner) = data.get("creator_user_id").and_then(Value::as_str) {
                        observation.account_identity = Some(owner.to_string());
                    } else if let Some(workspace) = data.get("workspace_id").and_then(Value::as_str)
                    {
                        // A provisioning key may report no creator. The workspace is
                        // the next identity the provider states outright, and is a
                        // weaker but still provider-reported grouping.
                        observation.account_identity = Some(format!("workspace:{workspace}"));
                    }
                    // The key endpoint reports the usage of this key over several
                    // windows. Each window is a different amount, so each is
                    // published under its own period rather than folded into one
                    // series.
                    for (field, period) in [
                        ("usage", ReportingPeriod::Total),
                        ("usage_daily", ReportingPeriod::Daily),
                        ("usage_weekly", ReportingPeriod::Weekly),
                        ("usage_monthly", ReportingPeriod::Monthly),
                    ] {
                        if let Some(usage) = data.get(field).and_then(Value::as_f64) {
                            observation.usage.push(AccountAmount::new(
                                "USD",
                                AccountScope::Key,
                                period,
                                usage,
                            ));
                        }
                    }
                }
            }
            Err(outcome) => return outcome,
        }

        // The credits endpoint reports the account-wide purchased credit and the usage
        // drawn against it. It is a separate request because a provider may authorize
        // one and refuse the other, so a failure here does not discard the reading
        // already taken.
        //
        // The purchased credit and the consumed credit are different facts. The
        // remaining balance is the difference between them, and is published as its
        // own figure so no reader has to subtract one series from another to know
        // what is left.
        if let Ok(body) = self.get_json("/credits").await
            && let Some(data) = body.get("data")
        {
            let credits = data.get("total_credits").and_then(Value::as_f64);
            let consumed = data.get("total_usage").and_then(Value::as_f64);
            if let Some(credits) = credits {
                observation.credits.push(AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    credits,
                ));
            }
            if let Some(consumed) = consumed {
                observation.usage.push(AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    consumed,
                ));
            }
            if let Some(remaining) = credits
                .zip(consumed)
                .map(|(credits, consumed)| credits - consumed)
            {
                observation.balances.push(AccountAmount::new(
                    "USD",
                    AccountScope::Account,
                    ReportingPeriod::Total,
                    remaining,
                ));
            }
        }

        if observation.is_empty() {
            return FetchOutcome::Failed("provider reported no account state".to_string());
        }
        FetchOutcome::Ok(Box::new(observation))
    }
}

/// Reads DeepSeek's prepaid balance.
///
/// DeepSeek's balance endpoint is the only account surface it exposes. It reports the
/// granted and topped-up balance per currency, which is enough to show whether the
/// lane is funded.
pub struct DeepSeekAccount {
    api_key: String,
    base_url: String,
}

impl DeepSeekAccount {
    /// Builds the reader from the credential Switchyard serves DeepSeek with.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.deepseek.com".to_string(),
        }
    }

    /// Overrides the API base, for a test double.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait::async_trait]
impl AccountApi for DeepSeekAccount {
    fn provider(&self) -> &str {
        "deepseek"
    }

    async fn fetch(&self) -> FetchOutcome {
        let response = account_client()
            .get(format!("{}/user/balance", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await;
        let body = match response {
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::UNAUTHORIZED
                    || status == reqwest::StatusCode::FORBIDDEN
                {
                    return FetchOutcome::Unauthorized;
                }
                if !status.is_success() {
                    return FetchOutcome::Failed(format!("HTTP {status}"));
                }
                match response.json::<Value>().await {
                    Ok(body) => body,
                    Err(error) => {
                        return FetchOutcome::Failed(error.to_string());
                    }
                }
            }
            Err(error) => return FetchOutcome::Failed(error.to_string()),
        };
        let mut observation = AccountObservation::default();
        if let Some(balances) = body.get("balance_infos").and_then(Value::as_array) {
            for balance in balances {
                let Some(currency) = balance.get("currency").and_then(Value::as_str) else {
                    continue;
                };
                // The balance endpoint reports the account's balance, not the key's.
                if let Some(total) = balance.get("total_balance").and_then(decode_amount) {
                    observation.balances.push(AccountAmount::new(
                        currency,
                        AccountScope::Account,
                        ReportingPeriod::Total,
                        total,
                    ));
                }
                // The same response breaks the balance into the two kinds of credit it
                // is made of. They are components of the total above, not additions
                // to it: granted credit was given by the provider, topped-up credit
                // was paid for, and `total_balance` is their sum.
                //
                // `topped_up_balance` is the topped-up money still remaining. It is
                // NOT the amount ever purchased, and must never be read as lifetime
                // top-ups: a heavily used account can hold a small remaining topped-up
                // balance after buying a great deal.
                for (field, component) in [
                    ("granted_balance", "granted"),
                    ("topped_up_balance", "topped_up"),
                ] {
                    if let Some(value) = balance.get(field).and_then(decode_amount) {
                        observation
                            .balance_components
                            .push(AccountBalanceComponent {
                                component: component.to_string(),
                                amount: AccountAmount::new(
                                    currency,
                                    AccountScope::Account,
                                    ReportingPeriod::Total,
                                    value,
                                ),
                            });
                    }
                }
            }
        }
        if observation.is_empty() {
            return FetchOutcome::Failed("provider reported no balance".to_string());
        }
        FetchOutcome::Ok(Box::new(observation))
    }
}

/// The provider account surface a client's base URL points at.
///
/// This is decided from the base URL rather than from the client's name, because the
/// name is the operator's label for a lane and carries no guarantee. A deployment
/// that adds a second OpenRouter client called anything at all still points at
/// `openrouter.ai`, and must still get an account reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccountSurface {
    /// OpenRouter's key and credits endpoints.
    OpenRouter,
    /// DeepSeek's balance endpoint.
    DeepSeek,
    /// The OAuth-backed Codex allowance windows.
    CodexAllowance,
}

impl AccountSurface {
    /// The label this surface's account carries when a deployment names a client
    /// after the provider itself.
    ///
    /// This is not a fallback name for the reader; it is the well-known label a
    /// dashboard already selects on, kept stable so an added client cannot rename an
    /// account that is already being read.
    pub const fn canonical_label(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter",
            Self::DeepSeek => "deepseek",
            Self::CodexAllowance => "openai",
        }
    }
}

/// Returns the account surface a base URL points at, if Switchyard can read one.
/// A base URL that matches none of these is a provider whose account surface
/// Switchyard does not read. That is not an error: a local inference server has no
/// account, and a provider may expose an inference API without an account API.
pub fn account_surface_for_base_url(base_url: &str) -> Option<AccountSurface> {
    let url = base_url.to_ascii_lowercase();
    if url.contains("openrouter.ai") {
        Some(AccountSurface::OpenRouter)
    } else if url.contains("api.deepseek.com") {
        Some(AccountSurface::DeepSeek)
    } else if url.contains("chatgpt.com") || url.contains("api.openai.com") {
        Some(AccountSurface::CodexAllowance)
    } else {
        None
    }
}

/// Reads an amount a provider reported.
///
/// A provider may report an amount as a JSON number or as a decimal string, and the
/// two are the same amount. The live DeepSeek balance endpoint reports strings, so
/// a reader that accepted only numbers would find no balance on a funded account
/// and report the lane as unmeasured. A value that is neither is not an amount, and
/// is left absent rather than being read as zero.
fn decode_amount(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Reads the allowance windows of an OAuth-backed provider.
///
/// The Codex backend reports a primary window and a secondary window, each with a
/// used percentage and a reset time. The observation keeps them as pools, so the
/// dashboard can show the five-hour window and the weekly window separately.
pub struct CodexAllowanceAccount {
    oauth_token: String,
    account_id: Option<String>,
    base_url: String,
}

impl CodexAllowanceAccount {
    /// Builds the reader from the credential Switchyard serves the Codex lane with.
    pub fn new(oauth_token: impl Into<String>, account_id: Option<String>) -> Self {
        Self {
            oauth_token: oauth_token.into(),
            account_id,
            base_url: "https://chatgpt.com/backend-api".to_string(),
        }
    }

    /// Overrides the API base, for a test double.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait::async_trait]
impl AccountApi for CodexAllowanceAccount {
    fn provider(&self) -> &str {
        "openai"
    }

    async fn fetch(&self) -> FetchOutcome {
        let mut request = account_client()
            .get(format!("{}/wham/usage", self.base_url))
            .bearer_auth(&self.oauth_token);
        if let Some(account_id) = &self.account_id {
            request = request.header("chatgpt-account-id", account_id);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => return FetchOutcome::Failed(error.to_string()),
        };
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return FetchOutcome::Unauthorized;
        }
        if !status.is_success() {
            return FetchOutcome::Failed(format!("HTTP {status}"));
        }
        let body = match response.json::<Value>().await {
            Ok(body) => body,
            Err(error) => return FetchOutcome::Failed(error.to_string()),
        };
        let mut observation = AccountObservation::default();
        let Some(limit) = body.get("rate_limit") else {
            return FetchOutcome::Failed("provider reported no allowance".to_string());
        };
        for (pool, key) in [
            ("primary", "primary_window"),
            ("secondary", "secondary_window"),
        ] {
            let Some(window) = limit.get(key) else {
                continue;
            };
            if let Some(used) = window.get("used_percent").and_then(Value::as_f64) {
                observation
                    .allowance_remaining_percent
                    .insert(pool.to_string(), 100.0 - used);
            }
            if let Some(reset_at) = window.get("reset_at").and_then(Value::as_f64) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_secs_f64())
                    .unwrap_or(0.0);
                observation
                    .allowance_reset_seconds
                    .insert(pool.to_string(), (reset_at - now).max(0.0));
            }
        }
        if observation.is_empty() {
            return FetchOutcome::Failed("provider reported no allowance".to_string());
        }
        FetchOutcome::Ok(Box::new(observation))
    }
}
