//! Fleet-readiness monitor configuration (LocalClaw extension).
//!
//! These sections declare the host-owned readiness monitor the stock CLI
//! constructs at startup. All sub-sections are optional; a fully-absent
//! `fleet_readiness` means no monitor is built and fleet routes fail closed.

use crate::route::RunnerError;
use serde::Deserialize;
use std::collections::HashMap;

/// Declares the fleet-readiness monitor components for the stock CLI runtime
/// owner. All sub-sections are optional; a fully-absent `fleet_readiness` means
/// no monitor is constructed and the server runs fail-closed for fleet routes.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetReadinessConfig {
    /// Seconds between observation cycles.
    #[serde(default = "default_fleet_observe_interval")]
    pub observe_interval_seconds: u64,
    /// ComfyNinja factual-readiness probe (read-only `/v1/resource`).
    #[serde(default)]
    pub comfy: Option<ComfyFactConfig>,
    /// HTPC factual-readiness probe (read-only `/health` + `/v1/models`).
    #[serde(default)]
    pub htpc: Option<HtpcFactConfig>,
    /// Live resource-state facts (OpenAI weekly allowance + DeepSeek balance).
    #[serde(default)]
    pub resource: Option<ResourceFactConfig>,
    /// Static `(model-id, ready)` base entries for cloud candidates that need
    /// no live resource/health probe (immediately attemptable).
    #[serde(default)]
    pub ready: Vec<String>,
    /// Static `(model-id, transition-required)` entries (e.g. a sealed-idle
    /// locally-resident model that is valid but not yet loaded).
    #[serde(default)]
    pub transition_required: Vec<String>,
}

const fn default_fleet_observe_interval() -> u64 {
    30
}

/// ComfyNinja factual-readiness probe (read-only `/v1/resource`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComfyFactConfig {
    /// Read-only resource endpoint URL.
    pub url: String,
    /// Environment variable holding the bearer token.
    pub auth_token_env: String,
    /// The candidate target model id this fact gates.
    pub model: String,
}

/// HTPC factual-readiness probe (read-only `/health` + `/v1/models`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HtpcFactConfig {
    /// HTPC base URL (health + models are joined onto it).
    pub base_url: String,
    /// Model basename that must be served for readiness.
    pub expected_model: String,
    /// The candidate target model id this fact gates.
    pub model: String,
}

/// Live resource-state facts (OpenAI weekly allowance + DeepSeek balance) that
/// gate cloud candidate readiness according to the operator
/// confirmed-exhaustion policy.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceFactConfig {
    #[serde(default)]
    pub openai_url: Option<String>,
    #[serde(default)]
    pub openai_auth_token_env: Option<String>,
    #[serde(default)]
    pub openclaw_openai_url: Option<String>,
    #[serde(default)]
    pub openclaw_openai_auth_token_env: Option<String>,
    #[serde(default)]
    pub deepseek_url: Option<String>,
    #[serde(default)]
    pub deepseek_api_key_env: Option<String>,
    #[serde(default)]
    pub deepseek_currency: Option<String>,
    /// Target model ids gated by the primary OpenAI weekly allowance.
    #[serde(default)]
    pub openai_gated: Vec<String>,
    /// Target model ids gated by the separate OpenClaw-owned OpenAI allowance.
    #[serde(default)]
    pub openclaw_openai_gated: Vec<String>,
    /// Target model ids gated by the DeepSeek configured-currency balance.
    #[serde(default)]
    pub deepseek_gated: Vec<String>,
}

impl FleetReadinessConfig {
    /// Validates the declared readiness sources: every gated model list must
    /// have its observation source configured (otherwise that candidate is
    /// silently excluded every cycle - fail loud at load time), and model ids
    /// must be disjoint across sources (a duplicate key would make every
    /// snapshot cycle fail and the fleet stay at the fail-closed empty
    /// snapshot).
    pub(crate) fn validate(&self) -> Result<(), crate::route::RunnerError> {
        // Disjointness across readiness sources (F2 review invariant).
        let mut declared: std::collections::HashMap<String, &'static str> =
            std::collections::HashMap::new();
        let mut disjoint =
            |model: &str, source: &'static str| -> Result<(), crate::route::RunnerError> {
                if let Some(origin) = declared.insert(model.to_string(), source) {
                    return Err(RunnerError::configuration(format!(
                        "[fleet_readiness] model {model:?} appears in both {origin} and {source}"
                    )));
                }
                Ok(())
            };
        for model in &self.ready {
            disjoint(model, "ready")?;
        }
        for model in &self.transition_required {
            disjoint(model, "transition_required")?;
        }
        if let Some(c) = &self.comfy {
            disjoint(c.model.as_str(), "comfy")?;
        }
        if let Some(h) = &self.htpc {
            disjoint(h.model.as_str(), "htpc")?;
        }
        if let Some(r) = &self.resource {
            for m in r
                .openai_gated
                .iter()
                .chain(r.openclaw_openai_gated.iter())
                .chain(r.deepseek_gated.iter())
            {
                disjoint(m, "resource-gated")?;
            }
        }
        // Gated lists require their observation source (F1 review invariant).
        if let Some(r) = &self.resource {
            if !r.openai_gated.is_empty()
                && (r.openai_url.is_none() || r.openai_auth_token_env.is_none())
            {
                return Err(RunnerError::configuration(
                    "openai_gated candidates require openai_url and openai_auth_token_env",
                ));
            }
            if !r.openclaw_openai_gated.is_empty()
                && (r.openclaw_openai_url.is_none() || r.openclaw_openai_auth_token_env.is_none())
            {
                return Err(RunnerError::configuration(
                    "openclaw_openai_gated candidates require openclaw_openai_url and openclaw_openai_auth_token_env",
                ));
            }
            if !r.deepseek_gated.is_empty()
                && (r.deepseek_url.is_none()
                    || r.deepseek_api_key_env.is_none()
                    || r.deepseek_currency.is_none())
            {
                return Err(RunnerError::configuration(
                    "deepseek_gated candidates require deepseek_url, deepseek_api_key_env and deepseek_currency",
                ));
            }
            // Both-or-neither for the optional OpenClaw pool credentials.
            match (&r.openclaw_openai_url, &r.openclaw_openai_auth_token_env) {
                (None, None) => {}
                (Some(_), Some(_)) => {}
                _ => {
                    return Err(RunnerError::configuration(
                        "openclaw_openai_url and openclaw_openai_auth_token_env must be set together",
                    ));
                }
            }
        }
        Ok(())
    }
}
