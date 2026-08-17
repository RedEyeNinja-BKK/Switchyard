// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw integration fork.
// SPDX-License-Identifier: Apache-2.0

//! Production HTTP resource fetchers for the resource-aware router.
//!
//! Trust boundary:
//!
//! - OpenAI OAuth state: read from the sanitized loopback surface on :8645
//!   (`GET /resource/openai-codex`) using the local catalog bearer token.
//!   The OAuth credential never leaves :8645.
//! - DeepSeek balance: read from the official `GET /user/balance` endpoint
//!   using the server-owned DeepSeek API key from its own environment.
//!
//! No secrets are logged or exposed in resource snapshots or errors. Upstream
//! response bodies are never propagated to callers; errors carry only a
//! bounded coarse status.

use std::time::Duration;

use async_trait::async_trait;
use libsy::{
    DeepSeekResourceState, OpenAiResourceState, ResourceFetcher, ResourceSnapshot, Result,
};
use reqwest::redirect;

const RESOURCE_FETCH_TIMEOUT: Duration = Duration::from_secs(15);

fn resource_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(RESOURCE_FETCH_TIMEOUT)
        .redirect(redirect::Policy::none())
        .build()
        .expect("static reqwest client builder")
}

/// Fetcher that fans out to per-pool fetchers and merges snapshots.
pub struct CompositeResourceFetcher {
    fetchers: Vec<(&'static str, Box<dyn ResourceFetcher>)>,
}

impl CompositeResourceFetcher {
    pub fn new(fetchers: Vec<(&'static str, Box<dyn ResourceFetcher>)>) -> Self {
        Self { fetchers }
    }
}

#[async_trait]
impl ResourceFetcher for CompositeResourceFetcher {
    async fn fetch(&self) -> Result<ResourceSnapshot> {
        let mut snapshot = ResourceSnapshot::default();
        for (kind, fetcher) in &self.fetchers {
            let part = fetcher.fetch().await?;
            match *kind {
                "openai" => snapshot.openai = part.openai,
                "deepseek" => snapshot.deepseek = part.deepseek,
                _ => {}
            }
        }
        Ok(snapshot)
    }
}

/// Fetches the sanitized OpenAI/Codex usage state from the :8645 surface.
pub struct OpenAiHttpFetcher {
    url: String,
    token: String,
    client: reqwest::Client,
}

impl OpenAiHttpFetcher {
    pub fn new(url: String, token: String) -> Self {
        Self {
            url,
            token,
            client: resource_http_client(),
        }
    }
}

#[async_trait]
impl ResourceFetcher for OpenAiHttpFetcher {
    async fn fetch(&self) -> Result<ResourceSnapshot> {
        let response = self
            .client
            .get(&self.url)
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|error| libsy::LibsyError::external("openai-resource", error))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| libsy::LibsyError::external("openai-resource", error))?;
        if !status.is_success() {
            // Coarse error only; do not propagate the upstream body.
            return Err(libsy::LibsyError::external(
                "openai-resource",
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("upstream resource fetch failed (HTTP {status})"),
                ),
            ));
        }
        let value: serde_json::Value = serde_json::from_str(&body)
            .map_err(|error| libsy::LibsyError::external("openai-resource", error))?;
        let rate_limit = value.get("rate_limit").cloned().unwrap_or_default();
        let primary = rate_limit.get("primary_window").cloned().unwrap_or_default();
        let credits = value.get("credits").cloned().unwrap_or_default();
        let weekly_used = primary
            .get("used_percent")
            .and_then(serde_json::Value::as_f64);
        let weekly_reset_at = primary.get("reset_at").and_then(serde_json::Value::as_i64);
        let state = OpenAiResourceState {
            available: rate_limit
                .get("allowed")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            limit_reached: rate_limit
                .get("limit_reached")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            spend_control_reached: value
                .get("spend_control")
                .and_then(|sc| sc.get("reached"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            weekly_used_percent: weekly_used,
            weekly_reset_at,
            credits_balance: credits
                .get("balance")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned),
        };
        Ok(ResourceSnapshot {
            openai: Some(state),
            deepseek: None,
        })
    }
}

/// Fetches DeepSeek balance from the official `/user/balance` endpoint.
pub struct DeepSeekHttpFetcher {
    url: String,
    api_key: String,
    currency: String,
    client: reqwest::Client,
}

impl DeepSeekHttpFetcher {
    pub fn new(url: String, api_key: String, currency: String) -> Self {
        Self {
            url,
            api_key,
            currency,
            client: resource_http_client(),
        }
    }
}

#[async_trait]
impl ResourceFetcher for DeepSeekHttpFetcher {
    async fn fetch(&self) -> Result<ResourceSnapshot> {
        let response = self
            .client
            .get(&self.url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|error| libsy::LibsyError::external("deepseek-resource", error))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| libsy::LibsyError::external("deepseek-resource", error))?;
        if !status.is_success() {
            // Coarse error only; do not propagate the upstream body.
            return Err(libsy::LibsyError::external(
                "deepseek-resource",
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("upstream resource fetch failed (HTTP {status})"),
                ),
            ));
        }
        let value: serde_json::Value = serde_json::from_str(&body)
            .map_err(|error| libsy::LibsyError::external("deepseek-resource", error))?;
        let is_available = value
            .get("is_available")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let mut total = String::new();
        let mut granted = String::new();
        let mut topped_up = String::new();
        if let Some(balances) = value.get("balance_infos").and_then(serde_json::Value::as_array) {
            // Fail closed when the configured currency is absent: never use a
            // different currency's balance for eligibility.
            let matching = balances.iter().find(|b| {
                b.get("currency")
                    .and_then(serde_json::Value::as_str)
                    .map(|c| c.eq_ignore_ascii_case(&self.currency))
                    .unwrap_or(false)
            });
            if let Some(balance) = matching {
                total = balance
                    .get("total_balance")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                granted = balance
                    .get("granted_balance")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                topped_up = balance
                    .get("topped_up_balance")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
            } else {
                tracing::warn!(
                    currency = self.currency,
                    "deepseek balance_infos missing configured currency; treating pool as unavailable"
                );
            }
        }
        Ok(ResourceSnapshot {
            openai: None,
            deepseek: Some(DeepSeekResourceState {
                is_available: is_available && !total.is_empty(),
                currency: self.currency.clone(),
                total_balance: total,
                granted_balance: granted,
                topped_up_balance: topped_up,
            }),
        })
    }
}
