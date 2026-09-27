// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Fleet-readiness monitor configuration (LocalClaw extension).
//!
//! These sections declare the host-owned readiness monitor the serving runtime
//! constructs at startup. All sub-sections are optional; a fully-absent
//! `fleet_readiness` means no monitor is built and fleet routes fail closed.
//!
//! Readiness facts are **produced** here and **consumed** by the FleetRouter as
//! a single injected snapshot. This module only declares the schema; the probe
//! and the monitor loop are the producer's responsibility, and a host that has
//! no monitor simply publishes no observations, which the consumer treats as
//! fail-closed.

use crate::route::RunnerError;
use serde::Deserialize;
use std::collections::HashMap;

/// Declares the fleet-readiness monitor components. All sub-sections optional.
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
    /// Candidate target model ids gated by the same ComfyNinja observation.
    #[serde(default)]
    pub governed: Vec<ComfyGovernedConfig>,
    /// OPTIONAL exact expected resident profile for this gate.
    ///
    /// When present, the candidate is `ready` only while the resource surface
    /// reports EXACTLY this resident profile (case-insensitive basename match)
    /// and the remaining serving conditions hold. There is deliberately no
    /// wildcard, no substring/prefix match and no implicit q3/q4 equivalence: a
    /// gate that accepted a family name could mark the q3 candidate ready while a
    /// q4 backing is resident.
    ///
    /// When absent (the default) the legacy family-vocabulary classification
    /// semantics are preserved, including `transition_required` for the
    /// sealed-idle signature, so existing deployments keep their meaning.
    #[serde(default)]
    pub expected_resident: Option<String>,
}

/// An additional ComfyNinja target and its exact expected resident profile.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComfyGovernedConfig {
    /// The additionally-governed target model id.
    pub model: String,
    /// Its exact expected resident profile.
    pub expected_resident: String,
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
    /// Validates the declared readiness sources.
    ///
    /// Two invariants, both fail-loud at load time:
    /// * every gated model list must have its observation source configured,
    ///   otherwise that candidate is silently excluded every cycle;
    /// * model ids must be disjoint across sources, because a duplicate key
    ///   would make every snapshot cycle fail and strand the fleet on the
    ///   fail-closed empty snapshot.
    pub(crate) fn validate(&self) -> Result<(), RunnerError> {
        let mut declared: HashMap<String, &'static str> = HashMap::new();
        let mut disjoint = |model: &str, source: &'static str| -> Result<(), RunnerError> {
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
            for governed in &c.governed {
                disjoint(governed.model.as_str(), "comfy-governed")?;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
observe_interval_seconds = 30

[comfy]
url = "https://example.invalid/v1/resource"
auth_token_env = "SOME_TOKEN"
model = "comfyninja/qwen3.8-27b-q3-mtp"
"#;

    #[test]
    fn comfy_expected_resident_is_optional_and_accepted() {
        let cfg: FleetReadinessConfig = toml::from_str(BASE).expect("base config must parse");
        assert_eq!(cfg.comfy.as_ref().unwrap().expected_resident, None);
        cfg.validate().expect("legacy-shaped config must validate");

        let with = format!("{BASE}expected_resident = \"qwen3.8-27b-q3-mtp\"\n");
        let cfg: FleetReadinessConfig =
            toml::from_str(&with).expect("exact-gate config must parse");
        assert_eq!(
            cfg.comfy.as_ref().unwrap().expected_resident.as_deref(),
            Some("qwen3.8-27b-q3-mtp")
        );
        cfg.validate().expect("exact-gate config must validate");
    }

    #[test]
    fn comfy_unknown_field_is_still_rejected() {
        // `deny_unknown_fields` keeps guarding the schema: a typo in the field
        // name cannot be silently ignored, which would leave the gate in legacy
        // mode while the operator believes it is exact.
        for bogus in [
            "expected_residents = \"qwen3.8-27b-q3-mtp\"",
            "expected_resident_name = \"qwen3.8-27b-q3-mtp\"",
            "expected_resident = \"a\"\nexpected_resident = \"b\"",
        ] {
            let bad = format!("{BASE}{bogus}\n");
            let err = toml::from_str::<FleetReadinessConfig>(&bad)
                .expect_err(&format!("must reject: {bogus}"));
            let msg = err.to_string();
            assert!(
                msg.contains("unknown field") || msg.contains("duplicate"),
                "expected an unknown-field/duplicate rejection for {bogus:?}, got: {msg}"
            );
        }
    }
}
