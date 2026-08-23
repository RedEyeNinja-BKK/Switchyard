// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real local resource / readiness fact producers (S2-D).
//!
//! This module observes the existing local inference factual surfaces **outside**
//! the routing request path, maps the observations into the accepted
//! [`CandidateState`] vocabulary (`ready` / `transition_required` only), assembles
//! **one coherent** [`FleetSnapshot`] per observation cycle, and writes it
//! atomically via [`SharedFleetState::set`] so FleetRouter requests consume a
//! complete generation — never a half-assembled fleet.
//!
//! Boundaries preserved:
//! - ComfyNinja is the factual GPU/resource/current-state authority. We consume
//!   its snapshot read-only; we NEVER invoke a transition/activation.
//! - HTPC readiness is established from its live health + model-identity
//!   evidence, never from TCP reachability alone.
//! - The producer performs no lifecycle transition, no economics, no context
//!   admission, and it does not guess ready from uncertainty (fail closed).
//!
//! Cloud provider candidates are not live-health-probed in S2-D: they are
//! placed in a clearly-defined static "immediately attemptable / no resource
//! transition" base state. That is distinct from a live provider health
//! guarantee (documented, not health-monitored here).
//!
//! [`FleetReadinessMonitor`] is a reusable switchyard-server-owned / server-side
//! readiness component that an explicit runtime owner can construct and run in a
//! background Tokio task alongside the server using the same `SharedFleetState`.
//! Automatic stock server-startup wiring (so `run_server`/`BoundServer` would
//! start the monitor by itself) is intentionally deferred to the later migration
//! gate; in S2-D the monitor is constructed and run explicitly by the
//! development runtime proof, not auto-wired into the stock server startup.

use std::sync::Arc;
use std::time::Duration;

use libsy::{CandidateState, FleetSnapshot, FleetStateSource, LibsyError, SharedFleetState};
use parking_lot::Mutex;
use reqwest::Client;
use switchyard_protocol::ModelId;

/// A bounded shared HTTP client for read-only factual fetches.
///
/// 10s timeout, no system proxy (we trust direct tailnet/estate connectivity),
/// and redirects disabled so a bare credential is never replayed off-origin.
fn factual_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("static reqwest client")
}

/// One observation of a candidate target's factual state, mapped to
/// [`CandidateState`].
#[derive(Debug)]
pub struct Observed {
    /// The model id recorded in the snapshot.
    pub model: ModelId,
    /// The readiness classification derived from the live observation.
    pub state: CandidateState,
}

// --- ComfyNinja factual contract (read-only `/v1/resource`) -----------------

/// ComfyNinja snapshot fields relevant to Qwen3.8 inference readiness.
///
/// Matches the deployed schema E `/v1/resource` response (verified live). Only
/// the fields needed to classify readiness are retained.
#[derive(serde::Deserialize, Debug)]
struct ComfyResourceResponse {
    #[serde(default)]
    state: ComfyResourceState,
}

#[derive(serde::Deserialize, Debug, Default)]
struct ComfyResourceState {
    #[serde(default)]
    mode: Option<String>,
    #[serde(rename = "resident_qwen_profile", default)]
    resident_qwen_profile: Option<String>,
    #[serde(default)]
    llama_server: Option<String>,
}

/// Read-only authenticated client for the ComfyNinja `/v1/resource` edge.
///
/// The bearer credential is read from an env-var **name** at request time and
/// dropped after building the header; the value is never stored, logged, or
/// printed.
pub struct ComfyFactsClient {
    url: String,
    auth_token_env: String,
    client: Client,
}

impl std::fmt::Debug for ComfyFactsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComfyFactsClient")
            .field("url", &self.url)
            .field("auth_token_env", &self.auth_token_env)
            .finish()
    }
}

impl ComfyFactsClient {
    /// A client observing `GET {url}/v1/resource`, authenticated with the
    /// bearer credential named by `auth_token_env` (a var NAME, never a value).
    pub fn new(url: String, auth_token_env: String) -> Self {
        Self {
            url,
            auth_token_env,
            client: factual_client(),
        }
    }

    /// One read-only observation. Maps the live ComfyNinja state to a
    /// [`CandidateState`] for the Qwen inference surface (the caller associates
    /// the model id it is classifying).
    ///
    /// Strict vocabulary (only the live-installed domain values map to a
    /// non-fail-closed state; see `classify_comfy` for the exact rules):
    /// - the complete **sealed-idle** signature (`mode=="idle"` and
    ///   `resident_qwen_profile=="unknown"` and `llama_server=="no"`)
    ///   → `transition_required` (valid but unloaded; requires an external
    ///   lifecycle transition that is never invoked here);
    /// - the complete **serving** signature (a known serving mode `"comfy"` or
    ///   `"studio"`, `llama_server=="yes"`, and a known `resident_qwen_profile`
    ///   in `{"unknown","FAST","LONG"}`) → `ready`;
    /// - any other value or combination → `not_ready` (fail closed).
    pub async fn observe(&self) -> CandidateState {
        self.fetch_snapshot()
            .await
            .map_or_else(|_| CandidateState::not_ready(), classify_comfy)
    }

    async fn fetch_snapshot(&self) -> Result<ComfyResourceResponse, String> {
        let token = std::env::var(&self.auth_token_env).map_err(|_| {
            format!(
                "credential env {} not set (fail closed)",
                self.auth_token_env
            )
        })?;
        if token.trim().is_empty() {
            return Err("credential env var is empty (fail closed)".into());
        }
        // The credential is read at observation time and used request-locally for
        // this one HTTP call, then dropped at the end of this function (normal
        // request-local lifetime). It is never stored/retained in the
        // `ComfyFactsClient` struct, and its value is never logged or debugged.
        let response = self
            .client
            .get(&self.url)
            .bearer_auth(&token)
            .send()
            .await
            .map_err(|e| format!("comfyninja snapshot request failed: {e}"))?;
        if !response.status().is_success() {
            let status = response.status();
            return Err(format!("comfyninja snapshot fetch failed (HTTP {status})"));
        }
        serde_json::from_str(
            &response
                .text()
                .await
                .map_err(|e| format!("read failed: {e}"))?,
        )
        .map_err(|e| format!("comfyninja snapshot parse failed: {e}"))
    }
}

// --- ComfyNinja strict factual vocabulary -----------------------------------

/// Known ComfyNinja `mode` values (canonical vocabulary from the live RT
/// contract: `idle | comfy | studio | conflict | unknown`).
const COMFY_MODE_IDLE: &str = "idle";
const COMFY_MODE_COMFY: &str = "comfy";
const COMFY_MODE_STUDIO: &str = "studio";

/// Known serving modes — the only `mode` values that indicate a load-bearing
/// Qwen surface. `idle` (unloaded) and `conflict`/`unknown` (ambiguous) are not.
const COMFY_SERVING_MODES: [&str; 2] = [COMFY_MODE_COMFY, COMFY_MODE_STUDIO];

/// Known `llama_server` values. `"yes"` is the proven llama-ready value
/// (`gpu-workload` sets it on model load); `"no"` is the unloaded value. Any
/// other/absent string is unknown and fails closed.
const COMFY_LLAMA_YES: &str = "yes";
const COMFY_LLAMA_NO: &str = "no";

/// Known `resident_qwen_profile` contract values. The factual producer leaves
/// this `"unknown"` even when a model is loaded (FAST/LONG are not factually
/// provable without an auth-gated seam), so `"unknown"` is the expected producer
/// value that, combined with a known serving mode + `llama_server=="yes"`,
/// indicates a loaded surface — NOT an arbitrary unknown string. Any string not
/// in this set fails closed.
const COMFY_RESIDENT_UNKNOWN: &str = "unknown";
const COMFY_RESIDENT_FAST: &str = "FAST";
const COMFY_RESIDENT_LONG: &str = "LONG";
const COMFY_RESIDENT_KNOWN: [&str; 3] = [
    COMFY_RESIDENT_UNKNOWN,
    COMFY_RESIDENT_FAST,
    COMFY_RESIDENT_LONG,
];

/// Strictly classifies one ComfyNinja snapshot into a [`CandidateState`].
///
/// Only the two documented, proven signatures map to a non-fail-closed state:
/// - **serving** → `ready` (mode ∈ serving modes AND `llama_server=="yes"` AND
///   `resident_qwen_profile` ∈ known set);
/// - **sealed-idle** → `transition_required` (mode=="idle" AND
///   `resident_qwen_profile=="unknown"` AND `llama_server=="no"`).
///
/// Every other value or combination — unknown mode, unknown llama value, an
/// arbitrary resident string outside the known set, or a contradictory
/// combination like `mode=idle + llama_server=yes` — → `not_ready`.
fn classify_comfy(snap: ComfyResourceResponse) -> CandidateState {
    let mode = snap.state.mode.as_deref().unwrap_or("");
    let resident = snap.state.resident_qwen_profile.as_deref().unwrap_or("");
    let llama = snap.state.llama_server.as_deref().unwrap_or("");
    // Profile-seam reconciliation (2026-08-24): the ComfyNinja seam now publishes
    // lowercase "fast"/"long" (readback-derived); the legacy constants are uppercase.
    // Match case-insensitively so both spellings classify truthfully.
    let resident_upper = resident.to_ascii_uppercase();
    let known_resident = COMFY_RESIDENT_KNOWN.contains(&resident)
        || COMFY_RESIDENT_KNOWN.contains(&resident_upper.as_str());
    if mode == COMFY_MODE_IDLE && resident == COMFY_RESIDENT_UNKNOWN && llama == COMFY_LLAMA_NO {
        CandidateState::transition_required()
    } else if COMFY_SERVING_MODES.contains(&mode) && llama == COMFY_LLAMA_YES && known_resident {
        CandidateState::ready()
    } else {
        CandidateState::not_ready()
    }
}

// --- HTPC health/model contract (read-only) --------------------------------

/// Read-only client for the HTPC generation/health surface (no authentication).
///
/// Readiness is established from actual health **and** model-identity evidence,
/// not TCP reachability alone.
pub struct HtpcFactsClient {
    base_url: String,
    expected_model: String,
    client: Client,
}

impl std::fmt::Debug for HtpcFactsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HtpcFactsClient")
            .field("base_url", &self.base_url)
            .field("expected_model", &self.expected_model)
            .finish()
    }
}

impl HtpcFactsClient {
    /// A client observing `{base_url}/health` and `{base_url}/v1/models`.
    ///
    /// `expected_model` is a canonical model identifier (e.g. the gguf
    /// basename `Qwen3.5-9B-Q4_0.gguf`). Readiness requires an **exact** match
    /// of a model's path basename to `expected_model` — never an arbitrary
    /// substring/family match.
    pub fn new(base_url: String, expected_model: String) -> Self {
        Self {
            base_url,
            expected_model,
            client: factual_client(),
        }
    }

    /// One read-only observation. `ready` only when the health endpoint reports
    /// an exact `{"status":"ok"}` AND `/v1/models` exposes a model whose path
    /// basename **exactly** equals the configured canonical `expected_model`
    /// (with no `"unknown"` entry). Any failure, malformed schema, different
    /// quant/file under the same family, or unexpected/unknown model
    /// → `not_ready` (HTPC has no transition-required state in this gate).
    pub async fn observe(&self) -> CandidateState {
        let health_url = format!("{}/health", self.base_url);
        let healthy = match self.client.get(&health_url).send().await {
            Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                Ok(v) => v.get("status").and_then(|s| s.as_str()) == Some("ok"),
                Err(_) => false,
            },
            _ => false,
        };
        if !healthy {
            return CandidateState::not_ready();
        }
        let models_url = format!("{}/v1/models", self.base_url);
        let models = match self.client.get(&models_url).send().await {
            Ok(r) if r.status().is_success() => r.json::<HtpcModelsResponse>().await.ok(),
            _ => None,
        };
        let Some(models) = models else {
            return CandidateState::not_ready();
        };
        // Exact canonical model-identity match: at least one entry's path
        // basename equals the configured expected model, and no entry is
        // "unknown".
        let has_expected = models
            .data
            .iter()
            .any(|m| fleet_basename(&m.id) == self.expected_model);
        let no_unknown = models.data.iter().all(|m| m.id != "unknown");
        if has_expected && no_unknown {
            CandidateState::ready()
        } else {
            CandidateState::not_ready()
        }
    }
}

/// The expected `/v1/models` response shape (OpenAI-style list).
#[derive(serde::Deserialize, Debug)]
struct HtpcModelsResponse {
    data: Vec<HtpcModelEntry>,
}

#[derive(serde::Deserialize, Debug)]
struct HtpcModelEntry {
    id: String,
}

/// The final path component of a model id (the text after the last `/`, or the
/// whole string when there is no `/`). This is the clean canonical identifier
/// used for strict model-identity comparison (never a machine-specific full
/// filesystem path).
fn fleet_basename(id: &str) -> &str {
    match id.rfind('/') {
        Some(pos) => &id[pos + 1..],
        None => id,
    }
}

// --- Resource-state facts (OpenAI weekly allowance / DeepSeek balance) -------

/// Sanitized OpenAI resource payload from a provider bridge's read-only
/// `/resource/openai-codex` surface (the same contract the old smart-routing
/// pool consumed). Fields are parsed for schema completeness; `available` and
/// `spend_control_reached` are intentionally NOT spill triggers (only confirmed
/// included-allowance exhaustion — `limit_reached` or `weekly used% >= 100` —
/// opens a fallback). No credential or account identity is parsed.
#[derive(serde::Deserialize, Debug, Default)]
#[allow(dead_code)]
struct OpenAiResourceState {
    #[serde(default)]
    available: Option<bool>,
    #[serde(default)]
    limit_reached: Option<bool>,
    #[serde(default)]
    spend_control_reached: Option<bool>,
    #[serde(default)]
    windows: Option<OpenAiWindows>,
}

#[derive(serde::Deserialize, Debug, Default)]
struct OpenAiWindows {
    #[serde(default)]
    primary: Option<OpenAiPrimaryWindow>,
}

#[derive(serde::Deserialize, Debug, Default)]
struct OpenAiPrimaryWindow {
    #[serde(rename = "used_percent", default)]
    used_percent: Option<f64>,
}

#[derive(serde::Deserialize, Debug, Default)]
struct DeepSeekResourceState {
    #[serde(rename = "is_available", default)]
    is_available: Option<bool>,
    #[serde(default)]
    balance_infos: Option<Vec<DeepSeekBalanceInfo>>,
}

#[derive(serde::Deserialize, Debug, Default)]
struct DeepSeekBalanceInfo {
    #[serde(default)]
    currency: Option<String>,
    #[serde(rename = "total_balance", default)]
    total_balance: Option<String>,
    #[serde(rename = "granted_balance", default)]
    granted_balance: Option<String>,
    #[serde(rename = "topped_up_balance", default)]
    topped_up_balance: Option<String>,
}

/// The DEPLOYED OpenAI included-allowance spill policy (operator-verbatim):
/// an OpenAI candidate becomes **resource-ineligible** (sanctioned DeepSeek
/// fallback may open) ONLY on **confirmed included-allowance exhaustion**:
///
/// ```text
///   limit_reached == true
///   OR
///   weekly_used_percent >= 100
/// ```
///
/// Everything else does NOT open a spill merely because resource telemetry is
/// uncertain:
///
/// - `spend_control_reached == true` — does NOT by itself sanction fallback
/// - `available == false` without confirmed included exhaustion — does NOT
///   sanction fallback
/// - resource endpoint fetch failure — does NOT sanction fallback
/// - malformed / stale / unknown telemetry — does NOT sanction fallback
/// - missing `used_percent` — does NOT sanction fallback
///
/// Those states intentionally FAIL TOWARD LUNA/OpenAI. A resource-monitor
/// failure is NOT confused with provider technical health: if the OpenAI
/// provider subsequently fails as an actual call, normal runtime/provider
/// failure behavior remains available. Do NOT open a DeepSeek spill merely
/// because OpenAI resource monitoring is uncertain or the pool reports
/// `spend_control_reached`/`available=false` without confirmed exhaustion.
fn classify_openai(state: &OpenAiResourceState) -> CandidateState {
    let limit_reached = state.limit_reached.unwrap_or(false);
    let used = state
        .windows
        .as_ref()
        .and_then(|w| w.primary.as_ref())
        .and_then(|p| p.used_percent);
    let confirmed_exhausted = limit_reached || used.is_some_and(|u| u >= 100.0);
    // NOT confirmed exhausted -> OpenAI candidate stays eligible (fail toward Luna).
    if confirmed_exhausted {
        CandidateState::not_ready()
    } else {
        CandidateState::ready()
    }
}

/// The preserved old smart-routing DeepSeek eligibility rule: READY only when the
/// provider reports the configured-currency balance available AND positive
/// (parsed `total_balance > 0.0`). Unknown/error state is not_ready (fail closed;
/// no cost engine is built here).
fn classify_deepseek(state: &DeepSeekResourceState, currency: &str) -> CandidateState {
    let is_available = state.is_available.unwrap_or(false);
    let balance = state
        .balance_infos
        .as_ref()
        .and_then(|infos| {
            infos.iter().find(|b| {
                b.currency
                    .as_deref()
                    .map(|c| c.eq_ignore_ascii_case(currency))
                    .unwrap_or(false)
            })
        })
        .and_then(|b| b.total_balance.as_deref())
        .and_then(|s| s.trim().parse::<f64>().ok());
    let eligible = is_available && balance.is_some_and(|b| b > 0.0);
    if eligible {
        CandidateState::ready()
    } else {
        CandidateState::not_ready()
    }
}

/// Sanitized last-successful DeepSeek balance observation, published by the
/// resource fact client for read-only observability (`GET /v1/resource/deepseek`).
///
/// Never contains credential material: no API key, no env-var value, no
/// Authorization header, no account identity.
#[derive(Clone, Debug, serde::Serialize)]
pub struct DeepSeekTelemetrySnapshot {
    pub is_available: Option<bool>,
    pub currency: Option<String>,
    pub total_balance: Option<f64>,
    pub granted_balance: Option<f64>,
    pub topped_up_balance: Option<f64>,
    pub observed_at: f64,
    pub error: String,
}

/// Shared slot holding the last successful sanitized DeepSeek observation.
///
/// Last-successful semantics: a failed observation cycle never overwrites the
/// previous successful snapshot, so a stale-but-truthful `observed_at` remains
/// detectable by downstream consumers instead of being replaced with invented
/// zeros.
#[derive(Clone, Default)]
pub struct SharedDeepSeekTelemetry(Arc<Mutex<Option<DeepSeekTelemetrySnapshot>>>);

impl std::fmt::Debug for SharedDeepSeekTelemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDeepSeekTelemetry")
            .field("state", &"<shared sanitized telemetry>")
            .finish()
    }
}

impl SharedDeepSeekTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publishes a successful observation snapshot (last-successful replace).
    pub fn publish(&self, snapshot: DeepSeekTelemetrySnapshot) {
        *self.0.lock() = Some(snapshot);
    }

    /// Returns the last successful observation, if any.
    pub fn get(&self) -> Option<DeepSeekTelemetrySnapshot> {
        self.0.lock().clone()
    }
}

/// Builds a sanitized telemetry snapshot from a successful DeepSeek factual
/// observation. The configured-currency balance info (when present) supplies the
/// balances; `observed_at` records the observation wall-clock.
fn deepseek_telemetry_snapshot(
    state: &DeepSeekResourceState,
    currency: &str,
    observed_at: f64,
) -> DeepSeekTelemetrySnapshot {
    let info = state.balance_infos.as_ref().and_then(|infos| {
        infos.iter().find(|b| {
            b.currency
                .as_deref()
                .is_some_and(|c| c.eq_ignore_ascii_case(currency))
        })
    });
    let parse = |v: &Option<String>| v.as_deref().and_then(|s| s.trim().parse::<f64>().ok());
    DeepSeekTelemetrySnapshot {
        is_available: state.is_available,
        currency: info.and_then(|b| b.currency.clone()),
        total_balance: info.and_then(|b| parse(&b.total_balance)),
        granted_balance: info.and_then(|b| parse(&b.granted_balance)),
        topped_up_balance: info.and_then(|b| parse(&b.topped_up_balance)),
        observed_at,
        error: String::new(),
    }
}

/// Current wall-clock as UNIX epoch seconds (for `observed_at`).
fn epoch_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Read-only client for the sanitized resource-state surfaces (OpenAI weekly
/// allowance via a provider bridge `/resource/openai-codex`, and the official
/// DeepSeek `/user/balance` endpoint).
///
/// The bearer/api credential is read from an env-var **name** at request time and
/// dropped on every path; values are never stored, logged, debugged, or printed.
pub struct ResourceStateFactsClient {
    openai_url: Option<String>,
    openai_auth_token_env: Option<String>,
    /// A second, caller-separated OpenAI OAuth pool (e.g. an OpenClaw-owned
    /// provider bridge). Kept separate so a turnstone/Hermes Luna lane and an
    /// OpenClaw Luna lane gate on their own allowance, never each other's.
    openclaw_openai_url: Option<String>,
    openclaw_openai_auth_token_env: Option<String>,
    deepseek_url: Option<String>,
    deepseek_api_key_env: Option<String>,
    deepseek_currency: Option<String>,
    /// Optional shared slot for the sanitized last-successful DeepSeek
    /// observation (observability only; see [`SharedDeepSeekTelemetry`]).
    deepseek_telemetry: Option<SharedDeepSeekTelemetry>,
    client: Client,
}

impl std::fmt::Debug for ResourceStateFactsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceStateFactsClient")
            .field("openai_url", &self.openai_url)
            .field("openai_auth_token_env", &self.openai_auth_token_env)
            .field("openclaw_openai_url", &self.openclaw_openai_url)
            .field(
                "openclaw_openai_auth_token_env",
                &self.openclaw_openai_auth_token_env,
            )
            .field("deepseek_url", &self.deepseek_url)
            .field("deepseek_api_key_env", &self.deepseek_api_key_env)
            .field("deepseek_currency", &self.deepseek_currency)
            .finish_non_exhaustive()
    }
}

impl ResourceStateFactsClient {
    /// A client observing the given sanitized OpenAI and/or DeepSeek resource
    /// surfaces. Pass `None` for a source not configured. `openai_*` is the
    /// primary OpenAI OAuth pool; `openclaw_openai_*` is an optional separately
    /// owned OpenAI OAuth pool.
    pub fn new(
        openai_url: Option<String>,
        openai_auth_token_env: Option<String>,
        deepseek_url: Option<String>,
        deepseek_api_key_env: Option<String>,
        deepseek_currency: Option<String>,
    ) -> Self {
        Self {
            openai_url,
            openai_auth_token_env,
            openclaw_openai_url: None,
            openclaw_openai_auth_token_env: None,
            deepseek_url,
            deepseek_api_key_env,
            deepseek_currency,
            deepseek_telemetry: None,
            client: factual_client(),
        }
    }

    /// Attaches a shared sanitized-telemetry slot published on each successful
    /// DeepSeek observation (read-only observability; never a fetch trigger).
    pub fn with_deepseek_telemetry(mut self, telemetry: SharedDeepSeekTelemetry) -> Self {
        self.deepseek_telemetry = Some(telemetry);
        self
    }

    /// Configures the optional second (OpenClaw-owned) OpenAI OAuth pool.
    pub fn with_openclaw_openai(mut self, url: String, auth_token_env: String) -> Self {
        self.openclaw_openai_url = Some(url);
        self.openclaw_openai_auth_token_env = Some(auth_token_env);
        self
    }

    /// One read-only observation of the OpenClaw-owned OpenAI weekly allowance.
    pub async fn observe_openai_openclaw(&self) -> CandidateState {
        let (Some(url), Some(token_env)) = (
            &self.openclaw_openai_url,
            &self.openclaw_openai_auth_token_env,
        ) else {
            return CandidateState::not_ready();
        };
        observe_openai_generic(&self.client, url, token_env).await
    }

    /// One read-only observation of the OpenAI weekly allowance via the bridge
    /// surface. Classifies eligibility per `classify_openai` (confirmed
    /// exhaustion only); any fetch failure or malformed payload fails toward
    /// `ready` (does NOT sanction a DeepSeek spill).
    pub async fn observe_openai(&self) -> CandidateState {
        let (Some(url), Some(token_env)) = (&self.openai_url, &self.openai_auth_token_env) else {
            return CandidateState::not_ready();
        };
        observe_openai_generic(&self.client, url, token_env).await
    }

    /// One read-only observation of the DeepSeek configured-currency balance.
    /// Classifies per `classify_deepseek`; fetch failure or missing currency
    /// entry resolves to `not_ready` (fail closed).
    pub async fn observe_deepseek(&self) -> CandidateState {
        let (Some(url), Some(key_env), Some(currency)) = (
            &self.deepseek_url,
            &self.deepseek_api_key_env,
            &self.deepseek_currency,
        ) else {
            return CandidateState::not_ready();
        };
        let key = std::env::var(key_env)
            .map_err(|_| "credential env not set")
            .ok();
        let Some(key) = key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty()) else {
            return CandidateState::not_ready();
        };
        let response = self.client.get(url).bearer_auth(&key).send().await;
        match response {
            Ok(r) if r.status().is_success() => match r.json::<DeepSeekResourceState>().await {
                Ok(state) => {
                    // Observability publication: only a SUCCESSFUL factual
                    // observation updates the last-successful telemetry slot.
                    // Failed cycles keep the prior snapshot (no invented zeros).
                    if let Some(telemetry) = &self.deepseek_telemetry {
                        telemetry.publish(deepseek_telemetry_snapshot(
                            &state,
                            currency,
                            epoch_seconds(),
                        ));
                    }
                    classify_deepseek(&state, currency)
                }
                Err(_) => CandidateState::not_ready(),
            },
            _ => CandidateState::not_ready(),
        }
    }
}

// --- Producer -----------------------------------------------------------------

/// Shared read-only OpenAI weekly-allowance observation for one OpenAI OAuth
/// bridge surface. Reads the credential from the env var NAME at request time
/// and drops it on every path.
///
/// Deployed spill policy (fail-toward-Luna): an OpenAI candidate is made
/// resource-ineligible ONLY on confirmed included-allowance exhaustion
/// (`limit_reached == true` OR `used_percent >= 100`). Any fetch failure,
/// malformed payload, missing/empty credential, non-2xx status, unknown
/// telemetry, `spend_control_reached`, `available == false`, or missing
/// `used_percent` resolves to `ready` — i.e. it does NOT sanction a DeepSeek
/// spill merely because OpenAI resource monitoring is uncertain.
async fn observe_openai_generic(client: &Client, url: &str, token_env: &str) -> CandidateState {
    let token = std::env::var(token_env)
        .map_err(|_| "credential env not set")
        .ok();
    let Some(token) = token
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
    else {
        // Missing/empty credential is not confirmed included-allowance exhaustion:
        // fail toward Luna (no spill).
        return CandidateState::ready();
    };
    let response = client.get(url).bearer_auth(&token).send().await;
    match response {
        Ok(r) if r.status().is_success() => match r.json::<OpenAiResourceState>().await {
            Ok(state) => classify_openai(&state),
            // Malformed payload is not confirmed exhaustion: fail toward Luna.
            Err(_) => CandidateState::ready(),
        },
        // Fetch/HTTP failure is not confirmed exhaustion: fail toward Luna.
        _ => CandidateState::ready(),
    }
}

/// Assembles one coherent [`FleetSnapshot`] from observed facts and writes it
/// atomically to a shared [`SharedFleetState`].
///
/// Atomic **whole-snapshot** publication is provided by [`SharedFleetState::set`]:
/// a single `set` swaps the complete immutable snapshot, so a reader never
/// observes a half-assembled generation.
///
/// Stale-cycle ordering is **not** this producer's guarantee (it has no
/// observation-generation identity, so a later `apply` of an older observation
/// could in principle overwrite a newer snapshot). Ordering is instead a
/// structural property of the single serialized [`FleetReadinessMonitor`] loop,
/// which never lets the next observation cycle begin before the prior one has
/// fully completed and published.
#[derive(Clone)]
pub struct FleetSnapshotProducer {
    state: Arc<SharedFleetState>,
}

impl std::fmt::Debug for FleetSnapshotProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FleetSnapshotProducer")
            .field("state", &"<Arc<SharedFleetState>>")
            .finish()
    }
}

impl FleetSnapshotProducer {
    /// A producer writing to `state`.
    pub fn new(state: Arc<SharedFleetState>) -> Self {
        Self { state }
    }

    /// Atomically replaces the whole snapshot with the observations of one
    /// generation.
    ///
    /// The caller assembles the complete observed set — including the static
    /// "immediately attemptable / no resource transition" base entries for cloud
    /// provider candidates, which are documented as configured/attemptable, not a
    /// live provider health guarantee. `SharedFleetState::set` swaps the whole
    /// snapshot atomically; ordering between generations is the monitor loop's
    /// structural responsibility (see the type docs).
    pub fn apply(&self, observations: Vec<Observed>) -> Result<(), LibsyError> {
        let states = observations
            .into_iter()
            .map(|o| (o.model, o.state))
            .collect();
        let snapshot = FleetSnapshot::new(states)?;
        self.state.set(snapshot);
        Ok(())
    }
}

// --- FleetReadinessMonitor ----------------------------------------------------

/// The server-owned asynchronous readiness monitor.
///
/// A [`FleetReadinessMonitor`] owns a **single serialized observation loop** that
/// runs in its own background task alongside the server runtime. Each cycle:
///
/// 1. observes the real ComfyNinja and HTPC factual surfaces (read-only),
/// 2. assembles **one complete** [`FleetSnapshot`] from those observations plus
///    the static cloud base states,
/// 3. publishes it **exactly once** via [`SharedFleetState::set`].
///
/// Because the loop awaits each cycle to completion before starting the next,
/// observation cycles cannot overlap, so a stale older generation can never
/// overwrite a newer one (structural no-stale-overwrite guarantee). The monitor
/// performs no lifecycle transition, no economics, and no context admission.
pub struct FleetReadinessMonitor {
    /// ComfyNinja fact probe; `None` when no Comfy fact is configured.
    comfy: Option<ComfyFactsClient>,
    /// Candidate target model id gated by the (optional) Comfy fact.
    comfy_model: Option<ModelId>,
    /// HTPC fact probe; `None` when no HTPC fact is configured.
    htpc: Option<HtpcFactsClient>,
    /// Candidate target model id gated by the (optional) HTPC fact.
    htpc_model: Option<ModelId>,
    /// Static cloud base states: "configured / immediately attemptable", never a
    /// live provider health guarantee.
    cloud_base: Vec<Observed>,
    /// Optional live resource-state facts (OpenAI weekly / DeepSeek balance) that
    /// gate the readiness of cloud resource-gated candidates on the same cycle.
    resource: Option<ResourceStateFactsClient>,
    /// Cloud candidates whose readiness is gated by the OpenAI weekly allowance
    /// (e.g. a premium Luna lane): ready only while the allowance is eligible.
    openai_gated: Vec<ModelId>,
    /// Cloud candidates whose readiness is gated by the separate OpenClaw-owned
    /// OpenAI OAuth allowance.
    openclaw_openai_gated: Vec<ModelId>,
    /// Cloud candidates whose readiness is gated by the DeepSeek balance.
    deepseek_gated: Vec<ModelId>,
    state: Arc<SharedFleetState>,
    interval: Duration,
}

impl std::fmt::Debug for FleetReadinessMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FleetReadinessMonitor")
            .field("comfy", &self.comfy)
            .field("comfy_model", &self.comfy_model)
            .field("htpc", &self.htpc)
            .field("htpc_model", &self.htpc_model)
            .field("cloud_base_count", &self.cloud_base.len())
            .field("resource", &self.resource)
            .field("openai_gated", &self.openai_gated)
            .field("openclaw_openai_gated", &self.openclaw_openai_gated)
            .field("deepseek_gated", &self.deepseek_gated)
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

impl FleetReadinessMonitor {
    /// Constructs a monitor over `state` (which must be the same
    /// [`SharedFleetState`] injected into the FleetRouter server). Local
    /// candidates are observed from `comfy`/`htpc` (each optional; `None` skips
    /// that fact); `cloud_base` supplies the static cloud base states. `interval`
    /// is the spacing between observation cycles.
    pub fn new(
        comfy: Option<ComfyFactsClient>,
        comfy_model: Option<ModelId>,
        htpc: Option<HtpcFactsClient>,
        htpc_model: Option<ModelId>,
        cloud_base: Vec<Observed>,
        state: Arc<SharedFleetState>,
        interval: Duration,
    ) -> Self {
        Self {
            comfy,
            comfy_model,
            htpc,
            htpc_model,
            cloud_base,
            resource: None,
            openai_gated: Vec::new(),
            openclaw_openai_gated: Vec::new(),
            deepseek_gated: Vec::new(),
            state,
            interval,
        }
    }

    /// Opts this monitor into live OpenAI/DeepSeek resource-state readiness for
    /// the listed cloud candidates. `openai_gated` names candidates whose
    /// readiness depends on the primary OpenAI weekly allowance (e.g. a premium
    /// Luna lane); `openclaw_openai_gated` names candidates gated by the separate
    /// OpenClaw-owned OpenAI OAuth allowance; `deepseek_gated` names candidates
    /// gated by the DeepSeek balance. Each cycle observes the resource surfaces
    /// once and applies the resulting eligibility to the corresponding candidates.
    pub fn with_resource(
        mut self,
        resource: ResourceStateFactsClient,
        openai_gated: Vec<ModelId>,
        openclaw_openai_gated: Vec<ModelId>,
        deepseek_gated: Vec<ModelId>,
    ) -> Self {
        self.resource = Some(resource);
        self.openai_gated = openai_gated;
        self.openclaw_openai_gated = openclaw_openai_gated;
        self.deepseek_gated = deepseek_gated;
        self
    }

    /// One complete observation cycle.
    ///
    /// Observes ComfyNinja and HTPC concurrently, assembles a complete snapshot
    /// (static cloud base + the two local observations), and calls
    /// [`SharedFleetState::set`] **exactly once**. A failed/schema-malformed local
    /// observation already resolves to a `not_ready` candidate (fail closed), so
    /// that candidate becomes `not_ready` in the newly-published generation
    /// rather than retaining a stale prior ready. Returns `Err` only if snapshot
    /// assembly itself fails (e.g. a duplicate model key, which cannot occur for
    /// distinct local models).
    pub async fn observe_once(&self) -> Result<(), LibsyError> {
        // Observe the local surfaces concurrently. Resource-state observations
        // (OpenAI weekly / DeepSeek balance) run only when a resource client is
        // configured, also concurrently with the local probes.
        let resource_fut = async {
            if let Some(resource) = &self.resource {
                // Contributes the primary OpenAI, the optional OpenClaw-owned
                // OpenAI, and the DeepSeek facts; the gated candidate lists
                // assemble the snapshot from them below.
                tokio::join!(
                    resource.observe_openai(),
                    resource.observe_openai_openclaw(),
                    resource.observe_deepseek(),
                )
            } else {
                (
                    CandidateState::not_ready(),
                    CandidateState::not_ready(),
                    CandidateState::not_ready(),
                )
            }
        };
        let (comfy_state, htpc_state, (openai_state, openclaw_state, deepseek_state)) =
            tokio::join!(self.comfy_observe(), self.htpc_observe(), resource_fut);
        let configured_local =
            usize::from(self.comfy_model.is_some()) + usize::from(self.htpc_model.is_some());
        let mut states = Vec::with_capacity(
            self.cloud_base.len()
                + configured_local
                + self.openai_gated.len()
                + self.openclaw_openai_gated.len()
                + self.deepseek_gated.len(),
        );
        for observed in &self.cloud_base {
            states.push((observed.model.clone(), observed.state));
        }
        if let Some(model) = &self.comfy_model {
            states.push((model.clone(), comfy_state));
        }
        if let Some(model) = &self.htpc_model {
            states.push((model.clone(), htpc_state));
        }
        for model in &self.openai_gated {
            states.push((model.clone(), openai_state));
        }
        for model in &self.openclaw_openai_gated {
            states.push((model.clone(), openclaw_state));
        }
        for model in &self.deepseek_gated {
            states.push((model.clone(), deepseek_state));
        }
        let snapshot = FleetSnapshot::new(states)?;
        self.state.set(snapshot);
        Ok(())
    }

    /// Observes the Comfy fact when configured (else a fail-closed not_ready).
    async fn comfy_observe(&self) -> CandidateState {
        match &self.comfy {
            Some(client) => client.observe().await,
            None => CandidateState::not_ready(),
        }
    }

    /// Observes the HTPC fact when configured (else a fail-closed not_ready).
    async fn htpc_observe(&self) -> CandidateState {
        match &self.htpc {
            Some(client) => client.observe().await,
            None => CandidateState::not_ready(),
        }
    }

    /// The fleet readiness snapshot most recently published by this monitor
    /// (i.e. the current value of the shared [`SharedFleetState`] it owns).
    pub fn snapshot(&self) -> Arc<FleetSnapshot> {
        self.state.snapshot()
    }

    /// The background observation loop.
    ///
    /// Observes immediately, then repeatedly at `interval`, until `shutdown`
    /// resolves. The serialize-one-cycle-at-a-time structure (each cycle is fully
    /// awaited before the next begins) is what guarantees no overlapping
    /// generations and hence no stale-cycle overwrite.
    pub async fn run(self, shutdown: impl Future<Output = ()> + Send + 'static) {
        let mut shutdown = Box::pin(shutdown);
        loop {
            if let Err(error) = self.observe_once().await {
                tracing::warn!(error = %error, "fleet readiness observation cycle failed; skipped");
            }
            tokio::select! {
                _ = &mut shutdown => break,
                _ = tokio::time::sleep(self.interval) => {}
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::Router;
    use axum::http::StatusCode;
    use axum::routing::{get, post};
    use libsy::{CandidateState, FleetSnapshot, FleetStateSource, SharedFleetState};
    use switchyard_protocol::ModelId;
    use tokio::net::TcpListener;

    use super::{
        ComfyFactsClient, DeepSeekBalanceInfo, DeepSeekResourceState, FleetReadinessMonitor,
        FleetSnapshotProducer, HtpcFactsClient, Observed, OpenAiPrimaryWindow, OpenAiResourceState,
        OpenAiWindows, ResourceStateFactsClient, SharedDeepSeekTelemetry, classify_deepseek,
        classify_openai,
    };

    async fn mock_server(routes: Vec<(&'static str, &'static str)>) -> String {
        let router = routes.into_iter().fold(Router::new(), |r, (path, body)| {
            let body = body.to_string();
            r.route(
                path,
                get(move || {
                    let body = body.clone();
                    async move { (axum::http::StatusCode::OK, body) }
                }),
            )
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = router;
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    // --- Comfy mapping (A, B, C) ---

    #[tokio::test]
    async fn comfy_ready_when_known_serving_signature() {
        // A: the documented loaded signature (mode=studio, llama=yes, resident
        // profile "unknown" as the factual producer leaves it) => ready.
        let url = mock_server(vec![(
            "/v1/resource",
            r#"{"producer_epoch":"e","state_generation":1,"state_fingerprint":"f",
                "state":{"mode":"studio","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
        )])
        .await;
        let client = ComfyFactsClient::new(format!("{url}/v1/resource"), "S2D_TEST_TOKEN_A".into());
        unsafe { std::env::set_var("S2D_TEST_TOKEN_A", "dummy") };
        assert_eq!(client.observe().await, CandidateState::ready());
        unsafe { std::env::remove_var("S2D_TEST_TOKEN_A") };
    }

    #[tokio::test]
    async fn comfy_fails_closed_on_unknown_or_contradictory_vocabulary() {
        // G: strict enums — only the documented signatures map to ready /
        // transition_required; any unknown or contradictory value is not_ready.
        let cases: Vec<(&str, &str)> = vec![
            // unknown mode (not a serving/idle keyword)
            (
                "unknown_mode",
                r#"{"state":{"mode":"bogus","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
            ),
            // unknown llama_server value
            (
                "unknown_llama",
                r#"{"state":{"mode":"studio","resident_qwen_profile":"unknown","llama_server":"maybe"}}"#,
            ),
            // arbitrary unknown resident profile string (not in the contract set)
            (
                "unknown_resident",
                r#"{"state":{"mode":"studio","resident_qwen_profile":"weird","llama_server":"yes"}}"#,
            ),
            // contradictory: idle mode but llama up
            (
                "contradictory",
                r#"{"state":{"mode":"idle","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
            ),
            // contradictory: serving mode but llama down
            (
                "serving_but_no_llama",
                r#"{"state":{"mode":"studio","resident_qwen_profile":"unknown","llama_server":"no"}}"#,
            ),
            // conflict mode
            (
                "conflict_mode",
                r#"{"state":{"mode":"conflict","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
            ),
        ];
        for (name, body) in cases {
            let url = mock_server(vec![("/v1/resource", body)]).await;
            let client = ComfyFactsClient::new(
                format!("{url}/v1/resource"),
                format!("S2D_TEST_TOKEN_{name}"),
            );
            unsafe { std::env::set_var(format!("S2D_TEST_TOKEN_{name}"), "dummy") };
            assert_eq!(
                client.observe().await,
                CandidateState::not_ready(),
                "case {name} must fail closed to not_ready"
            );
            unsafe { std::env::remove_var(format!("S2D_TEST_TOKEN_{name}")) };
        }
    }

    #[tokio::test]
    async fn comfy_transition_required_when_sealed_idle() {
        // B: valid but unloaded - idle mode (sealed idle) => transition required.
        let url = mock_server(vec![(
            "/v1/resource",
            r#"{"producer_epoch":"e","state_generation":2,"state_fingerprint":"g",
                "state":{"mode":"idle","resident_qwen_profile":"unknown","llama_server":"no"}}"#,
        )])
        .await;
        let client = ComfyFactsClient::new(format!("{url}/v1/resource"), "S2D_TEST_TOKEN_B".into());
        unsafe { std::env::set_var("S2D_TEST_TOKEN_B", "dummy") };
        assert_eq!(
            client.observe().await,
            CandidateState::transition_required()
        );
        unsafe { std::env::remove_var("S2D_TEST_TOKEN_B") };
    }

    #[tokio::test]
    async fn comfy_failure_or_malformed_fails_closed() {
        // C: unavailable / malformed => not_ready, not transition-required.
        // Malformed body (not JSON).
        let url = mock_server(vec![("/v1/resource", "not-json")]).await;
        let client = ComfyFactsClient::new(format!("{url}/v1/resource"), "S2D_TEST_TOKEN_C".into());
        unsafe { std::env::set_var("S2D_TEST_TOKEN_C", "dummy") };
        assert_eq!(client.observe().await, CandidateState::not_ready());
        unsafe { std::env::remove_var("S2D_TEST_TOKEN_C") };

        // Missing credential env => fail closed (not_ready).
        let client2 = ComfyFactsClient::new(
            format!("{url}/v1/resource"),
            "S2D_TEST_TOKEN_MISSING".into(),
        );
        assert_eq!(client2.observe().await, CandidateState::not_ready());
    }

    // --- HTPC mapping (D, E) ---

    #[tokio::test]
    async fn htpc_ready_when_healthy_and_expected_model() {
        // D: healthy endpoint serving the exact canonical model basename => ready.
        let url = mock_server(vec![
            ("/health", r#"{"status":"ok"}"#),
            (
                "/v1/models",
                r#"{"object":"list","data":[{"id":"/srv/htpc-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"}]}"#,
            ),
        ])
        .await;
        let client = HtpcFactsClient::new(url.clone(), "Qwen3.5-9B-Q4_0.gguf".into());
        assert_eq!(client.observe().await, CandidateState::ready());
        let _ = &client;
    }

    #[tokio::test]
    async fn htpc_not_ready_on_health_failure_or_wrong_model() {
        // E: health down, or unexpected/unknown model => not_ready.
        let url = mock_server(vec![
            ("/health", r#"{"status":"down"}"#),
            (
                "/v1/models",
                r#"{"object":"list","data":[{"id":"/srv/htpc-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"}]}"#,
            ),
        ])
        .await;
        let c_health = HtpcFactsClient::new(url.clone(), "Qwen3.5-9B-Q4_0.gguf".into());
        assert_eq!(c_health.observe().await, CandidateState::not_ready());

        // Health ok but a different/unknown model => not_ready.
        let url2 = mock_server(vec![
            ("/health", r#"{"status":"ok"}"#),
            (
                "/v1/models",
                r#"{"object":"list","data":[{"id":"some/other"}]}"#,
            ),
        ])
        .await;
        let c_model = HtpcFactsClient::new(url2.clone(), "Qwen3.5-9B-Q4_0.gguf".into());
        assert_eq!(c_model.observe().await, CandidateState::not_ready());
        let _ = &c_model;
    }

    #[tokio::test]
    async fn htpc_strict_exact_identity_not_family_substring() {
        // F: same family directory but different quant/file, or a substring that
        // is not the exact canonical basename => not_ready. Only the exact
        // canonical model basename is ready.
        // Family present but wrong quant => not_ready.
        let url = mock_server(vec![
            ("/health", r#"{"status":"ok"}"#),
            (
                "/v1/models",
                r#"{"object":"list","data":[{"id":"/srv/htpc-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q8_0.gguf"}]}"#,
            ),
        ])
        .await;
        let client = HtpcFactsClient::new(url.clone(), "Qwen3.5-9B-Q4_0.gguf".into());
        assert_eq!(
            client.observe().await,
            CandidateState::not_ready(),
            "a different quant under the same family must not be ready"
        );

        // An "unknown" model entry must fail closed even with the expected one.
        let url2 = mock_server(vec![
            ("/health", r#"{"status":"ok"}"#),
            (
                "/v1/models",
                r#"{"object":"list","data":[
                    {"id":"/srv/htpc-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"},
                    {"id":"unknown"}
                ]}"#,
            ),
        ])
        .await;
        let client2 = HtpcFactsClient::new(url2.clone(), "Qwen3.5-9B-Q4_0.gguf".into());
        assert_eq!(
            client2.observe().await,
            CandidateState::not_ready(),
            "an unknown model id entry must fail closed"
        );
        let _ = &client;
        let _ = &client2;
    }

    // --- Atomic generation + failure recovery + transition purity (F, G, H) ---

    #[tokio::test]
    async fn producer_applies_one_coherent_snapshot() {
        // F: one apply() replaces the whole snapshot; a reader sees a complete
        // generation (here we read it back via SharedFleetState::snapshot).
        let state = Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap()));
        let producer = FleetSnapshotProducer::new(Arc::clone(&state));
        let luna = Observed {
            model: ModelId::from("model/luna"),
            state: CandidateState::ready(),
        };
        let comfy = Observed {
            model: ModelId::from("model/comfyninja-qwen3_8"),
            state: CandidateState::transition_required(),
        };
        producer.apply(vec![luna, comfy]).unwrap();
        let snap = state.snapshot();
        assert_eq!(
            snap.state_for(&ModelId::from("model/luna")),
            CandidateState::ready()
        );
        assert_eq!(
            snap.state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::transition_required()
        );
    }

    #[tokio::test]
    async fn producer_fails_closed_then_recovers() {
        // G: cycle 1 ready, cycle 2 failure => not-ready, cycle 3 healthy => ready
        // again on the same producer/state (transition purity is inherent).
        let state = Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap()));
        let producer = FleetSnapshotProducer::new(Arc::clone(&state));

        // Cycle 1: ready.
        producer
            .apply(vec![Observed {
                model: ModelId::from("model/luna"),
                state: CandidateState::ready(),
            }])
            .unwrap();
        assert_eq!(
            state.snapshot().state_for(&ModelId::from("model/luna")),
            CandidateState::ready()
        );

        // Cycle 2: failure => not-ready.
        producer
            .apply(vec![Observed {
                model: ModelId::from("model/luna"),
                state: CandidateState::not_ready(),
            }])
            .unwrap();
        assert_eq!(
            state.snapshot().state_for(&ModelId::from("model/luna")),
            CandidateState::not_ready()
        );

        // Cycle 3: healthy => ready again.
        producer
            .apply(vec![Observed {
                model: ModelId::from("model/luna"),
                state: CandidateState::ready(),
            }])
            .unwrap();
        assert_eq!(
            state.snapshot().state_for(&ModelId::from("model/luna")),
            CandidateState::ready()
        );
    }

    #[tokio::test]
    async fn comfy_transition_required_never_invokes_transition() {
        // H: transition-required observation never triggers a transition call.
        // The mock records every path touched; we assert only the read resource
        // path was called and no POST /v1/transitions mutation fired.
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen_res = seen.clone();
        let seen_tx = seen.clone();
        let app = Router::new()
            .route(
                "/v1/resource",
                get(move || {
                    let seen = seen_res.clone();
                    async move {
                        seen.lock().unwrap().push("GET /v1/resource".into());
                        (
                            axum::http::StatusCode::OK,
                            r#"{"state":{"mode":"idle","resident_qwen_profile":"unknown","llama_server":"no"}}"#,
                        )
                    }
                }),
            )
            .route(
                "/v1/transitions",
                post(move || {
                    let seen = seen_tx.clone();
                    async move {
                        seen.lock().unwrap().push("POST /v1/transitions".into());
                        (
                            axum::http::StatusCode::OK,
                            "transition mutation (must NOT be called)",
                        )
                    }
                }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let url = format!("http://{addr}");

        let client = ComfyFactsClient::new(format!("{url}/v1/resource"), "S2D_TEST_TOKEN_H".into());
        unsafe { std::env::set_var("S2D_TEST_TOKEN_H", "dummy") };
        assert_eq!(
            client.observe().await,
            CandidateState::transition_required()
        );
        // Only the read path may have been called; the transitions POST must
        // never fire because observation is read-only.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert!(
            seen.lock()
                .unwrap()
                .iter()
                .all(|p| p.starts_with("GET ") && p.contains("/resource")),
            "only the read resource path may be called: {:?}",
            *seen.lock().unwrap()
        );
        unsafe { std::env::remove_var("S2D_TEST_TOKEN_H") };
    }

    // --- Monitor tests (A, B, D) + E/background loop -------------------------

    /// Shared gauge tracking global concurrent HTTP reads and call counts across
    /// the Comfy + HTPC mock servers. `active` is the number of request handlers
    /// currently in flight; `max_seen` records its high-water mark; `calls`
    /// counts total handler invocations.
    #[derive(Clone, Default)]
    struct Gauge {
        active: Arc<std::sync::atomic::AtomicUsize>,
        max_seen: Arc<std::sync::atomic::AtomicUsize>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Gauge {
        fn snapshot(&self) -> (usize, usize, usize) {
            (
                self.active.load(std::sync::atomic::Ordering::SeqCst),
                self.max_seen.load(std::sync::atomic::Ordering::SeqCst),
                self.calls.load(std::sync::atomic::Ordering::SeqCst),
            )
        }
    }

    /// One mutable endpoint route: body + HTTP status are swappable between
    /// cycles; handler bumps a shared [`Gauge`] for concurrency tracking. A
    /// short fixed sleep makes any overlapping whole-cycle observable in
    /// `max_seen`.
    #[derive(Clone)]
    struct Endpoint {
        body: Arc<std::sync::Mutex<String>>,
        status: Arc<std::sync::Mutex<u16>>,
    }

    impl Endpoint {
        fn new(body: &str) -> Self {
            Self {
                body: Arc::new(std::sync::Mutex::new(body.to_string())),
                status: Arc::new(std::sync::Mutex::new(200)),
            }
        }
        fn set(&self, status: u16, body: &str) {
            *self.status.lock().unwrap() = status;
            *self.body.lock().unwrap() = body.to_string();
        }
    }

    /// Spawns an axum server serving the given GET routes, each with its mutable
    /// [`Endpoint`] and the shared [`Gauge`]. Returns the server base URL.
    async fn spawn_facts_server(routes: Vec<(&'static str, Endpoint)>, gauge: Gauge) -> String {
        let router = routes.into_iter().fold(Router::new(), |r, (path, ep)| {
            let gauge = gauge.clone();
            r.route(
                path,
                get(move || {
                    let gauge = gauge.clone();
                    let ep_body = ep.body.clone();
                    let ep_status = ep.status.clone();
                    async move {
                        let cur = gauge
                            .active
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                            + 1;
                        gauge
                            .max_seen
                            .fetch_max(cur, std::sync::atomic::Ordering::SeqCst);
                        gauge
                            .calls
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        tokio::time::sleep(tokio::time::Duration::from_millis(4)).await;
                        let status =
                            axum::http::StatusCode::from_u16(*ep_status.lock().unwrap()).unwrap();
                        let body = ep_body.lock().unwrap().clone();
                        gauge
                            .active
                            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        (status, body)
                    }
                }),
            )
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    /// Builds a ready-to-use monitor over `state` with the given comfy/htpc model
    /// ids, a fixed static cloud base (luna + deepseek), and a small interval.
    fn test_monitor(
        comfy: ComfyFactsClient,
        comfy_model: ModelId,
        htpc: HtpcFactsClient,
        htpc_model: ModelId,
        state: Arc<SharedFleetState>,
        interval: Duration,
    ) -> FleetReadinessMonitor {
        FleetReadinessMonitor::new(
            Some(comfy),
            Some(comfy_model),
            Some(htpc),
            Some(htpc_model),
            vec![
                Observed {
                    model: ModelId::from("model/luna"),
                    state: CandidateState::ready(),
                },
                Observed {
                    model: ModelId::from("model/deepseek-flash"),
                    state: CandidateState::ready(),
                },
            ],
            state,
            interval,
        )
    }

    #[tokio::test]
    async fn monitor_starts_fail_closed_before_first_observation() {
        // A: before any successful observe_once publication, local candidates
        // resolve to a fail-closed not-ready (absent from the empty snapshot).
        let state = Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap()));
        let snap = state.snapshot();
        assert_eq!(
            snap.state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::not_ready(),
            "comfy must be fail-closed not-ready before first observation"
        );
        assert_eq!(
            snap.state_for(&ModelId::from("model/htpc-qwen3_5")),
            CandidateState::not_ready(),
            "htpc must be fail-closed not-ready before first observation"
        );
    }

    #[tokio::test]
    async fn monitor_one_cycle_publishes_one_complete_snapshot() {
        // B: one observe_once assembles cloud base + comfy(sealed-idle =>
        // transition_required) + htpc(ready) into a single complete snapshot.
        let gauge = Gauge::default();
        let comfy_url = spawn_facts_server(
            vec![(
                "/v1/resource",
                Endpoint::new(
                    r#"{"state":{"mode":"idle","resident_qwen_profile":"unknown","llama_server":"no"}}"#,
                ),
            )],
            gauge.clone(),
        )
        .await;
        let htpc_url = spawn_facts_server(
            vec![
                ("/health", Endpoint::new(r#"{"status":"ok"}"#)),
                (
                    "/v1/models",
                    Endpoint::new(
                        r#"{"object":"list","data":[{"id":"/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"}]}"#,
                    ),
                ),
            ],
            gauge,
        )
        .await;

        let state = Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap()));
        let comfy = ComfyFactsClient::new(
            format!("{comfy_url}/v1/resource"),
            "S2D_TEST_TOKEN_MONB".to_string(),
        );
        unsafe { std::env::set_var("S2D_TEST_TOKEN_MONB", "dummy") };
        let htpc = HtpcFactsClient::new(htpc_url.clone(), "Qwen3.5-9B-Q4_0.gguf".to_string());
        let monitor = test_monitor(
            comfy,
            ModelId::from("model/comfyninja-qwen3_8"),
            htpc,
            ModelId::from("model/htpc-qwen3_5"),
            Arc::clone(&state),
            Duration::from_secs(3600),
        );

        monitor.observe_once().await.unwrap();
        let snap = state.snapshot();
        // Cloud base retained.
        assert_eq!(
            snap.state_for(&ModelId::from("model/luna")),
            CandidateState::ready()
        );
        assert_eq!(
            snap.state_for(&ModelId::from("model/deepseek-flash")),
            CandidateState::ready()
        );
        // Real local observations.
        assert_eq!(
            snap.state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::transition_required()
        );
        assert_eq!(
            snap.state_for(&ModelId::from("model/htpc-qwen3_5")),
            CandidateState::ready()
        );
        unsafe { std::env::remove_var("S2D_TEST_TOKEN_MONB") };
        let _ = &htpc_url;
    }

    #[tokio::test]
    async fn monitor_fails_closed_then_recovers_through_real_observers() {
        // D: the mocked factual endpoints actually produce ready -> failure ->
        // ready across three observe_once cycles on the same monitor/state
        // (never a manual producer.apply). SharedFleetState follows the facts.
        let gauge = Gauge::default();
        let comfy_ep = Endpoint::new(
            r#"{"state":{"mode":"studio","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
        );
        let comfy_url =
            spawn_facts_server(vec![("/v1/resource", comfy_ep.clone())], gauge.clone()).await;
        let htpc_health_ep = Endpoint::new(r#"{"status":"ok"}"#);
        let htpc_models_ep = Endpoint::new(
            r#"{"object":"list","data":[{"id":"/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"}]}"#,
        );
        let htpc_url = spawn_facts_server(
            vec![
                ("/health", htpc_health_ep.clone()),
                ("/v1/models", htpc_models_ep.clone()),
            ],
            gauge,
        )
        .await;

        let state = Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap()));
        let comfy = ComfyFactsClient::new(
            format!("{comfy_url}/v1/resource"),
            "S2D_TEST_TOKEN_MOND".to_string(),
        );
        unsafe { std::env::set_var("S2D_TEST_TOKEN_MOND", "dummy") };
        let htpc = HtpcFactsClient::new(htpc_url.clone(), "Qwen3.5-9B-Q4_0.gguf".to_string());
        let monitor = test_monitor(
            comfy,
            ModelId::from("model/comfyninja-qwen3_8"),
            htpc,
            ModelId::from("model/htpc-qwen3_5"),
            Arc::clone(&state),
            Duration::from_secs(3600),
        );

        // Cycle 1: both ready.
        monitor.observe_once().await.unwrap();
        assert_eq!(
            state
                .snapshot()
                .state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::ready()
        );
        assert_eq!(
            state
                .snapshot()
                .state_for(&ModelId::from("model/htpc-qwen3_5")),
            CandidateState::ready()
        );

        // Cycle 2: make Comfy fail (HTTP 503 => fail closed to not_ready).
        comfy_ep.set(503, "unavailable");
        monitor.observe_once().await.unwrap();
        assert_eq!(
            state
                .snapshot()
                .state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::not_ready(),
            "a failed poll must replace a prior ready with not_ready"
        );

        // Cycle 3: Comfy recovers => ready again (same monitor/state).
        comfy_ep.set(
            200,
            r#"{"state":{"mode":"studio","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
        );
        monitor.observe_once().await.unwrap();
        assert_eq!(
            state
                .snapshot()
                .state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::ready(),
            "recovery must restore ready on the same monitor"
        );

        unsafe { std::env::remove_var("S2D_TEST_TOKEN_MOND") };
        let _ = &htpc_url;
    }

    #[tokio::test]
    async fn monitor_background_loop_no_overlapping_cycles_and_clean_shutdown() {
        // E + background loop: run() performs repeated observations, whole cycles
        // never overlap (max concurrent HTTP reaches only the in-cycle Comfy+HTPC
        // pair, never more), and a shutdown signal stops the loop cleanly so no
        // further cycles run.
        let gauge = Gauge::default();
        let comfy_url = spawn_facts_server(
            vec![(
                "/v1/resource",
                Endpoint::new(
                    r#"{"state":{"mode":"studio","resident_qwen_profile":"unknown","llama_server":"yes"}}"#,
                ),
            )],
            gauge.clone(),
        )
        .await;
        let htpc_url = spawn_facts_server(
            vec![
                ("/health", Endpoint::new(r#"{"status":"ok"}"#)),
                (
                    "/v1/models",
                    Endpoint::new(
                        r#"{"object":"list","data":[{"id":"/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"}]}"#,
                    ),
                ),
            ],
            gauge.clone(),
        )
        .await;

        let state = Arc::new(SharedFleetState::new(FleetSnapshot::new(vec![]).unwrap()));
        let comfy = ComfyFactsClient::new(
            format!("{comfy_url}/v1/resource"),
            "S2D_TEST_TOKEN_MONE".to_string(),
        );
        unsafe { std::env::set_var("S2D_TEST_TOKEN_MONE", "dummy") };
        let htpc = HtpcFactsClient::new(htpc_url.clone(), "Qwen3.5-9B-Q4_0.gguf".to_string());
        let monitor = test_monitor(
            comfy,
            ModelId::from("model/comfyninja-qwen3_8"),
            htpc,
            ModelId::from("model/htpc-qwen3_5"),
            Arc::clone(&state),
            Duration::from_millis(10),
        );

        // Shutdown via a one-shot that fires after a couple of intervals.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(monitor.run(async move {
            let _ = rx.await;
        }));

        // Let a few cycles run, then request shutdown.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let _ = tx.send(());
        // Bound the wait; the loop must exit promptly after shutdown.
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("monitor loop must exit cleanly on shutdown")
            .expect("monitor task must not panic");

        let (_active, max_seen, calls) = gauge.snapshot();
        // Each completed cycle does one comfy + one (health+models) read, so
        // within a single cycle the peak is 2 concurrent handlers (comfy + one
        // sequential htpc read). Overlapping whole cycles would raise this above 2.
        assert!(
            max_seen <= 2,
            "whole observation cycles must never overlap; saw {max_seen} concurrent reads"
        );
        // The loop performed several cycles and both comfy + htpc (health+models)
        // were exercised.
        assert!(
            calls >= 6,
            "expected repeated observations, saw {calls} reads"
        );
        // The final published snapshot reflects the (still-ready) facts.
        assert_eq!(
            state
                .snapshot()
                .state_for(&ModelId::from("model/comfyninja-qwen3_8")),
            CandidateState::ready()
        );
        assert_eq!(
            state
                .snapshot()
                .state_for(&ModelId::from("model/htpc-qwen3_5")),
            CandidateState::ready()
        );

        unsafe { std::env::remove_var("S2D_TEST_TOKEN_MONE") };
        let _ = &htpc_url;
    }

    // --- Resource-state classification (OpenAI weekly / DeepSeek balance) -----

    // --- OpenAI confirmed-exhaustion-only spill policy (Fix C) ----------------

    #[test]
    fn openai_healthy_under_100_is_ready() {
        let state = OpenAiResourceState {
            available: Some(true),
            limit_reached: Some(false),
            spend_control_reached: Some(false),
            windows: Some(OpenAiWindows {
                primary: Some(OpenAiPrimaryWindow {
                    used_percent: Some(11.0),
                }),
            }),
        };
        assert_eq!(classify_openai(&state), CandidateState::ready());
    }

    #[test]
    fn openai_confirmed_exhaustion_opens_fallback() {
        // limit_reached == true => confirmed exhaustion => not ready (fallback may open).
        let limit = OpenAiResourceState {
            available: Some(true),
            limit_reached: Some(true),
            spend_control_reached: Some(false),
            windows: Some(OpenAiWindows {
                primary: Some(OpenAiPrimaryWindow {
                    used_percent: Some(90.0),
                }),
            }),
        };
        assert_eq!(classify_openai(&limit), CandidateState::not_ready());

        // weekly_used_percent >= 100 => confirmed exhaustion => not ready.
        let weekly_full = OpenAiResourceState {
            available: Some(true),
            limit_reached: Some(false),
            spend_control_reached: Some(false),
            windows: Some(OpenAiWindows {
                primary: Some(OpenAiPrimaryWindow {
                    used_percent: Some(100.0),
                }),
            }),
        };
        assert_eq!(classify_openai(&weekly_full), CandidateState::not_ready());
    }

    #[test]
    fn openai_spend_control_alone_does_not_open_fallback() {
        // spend_control_reached == true WITHOUT confirmed exhaustion => NOT a
        // spill trigger: OpenAI candidate stays ready (fail toward Luna).
        let spend = OpenAiResourceState {
            available: Some(true),
            limit_reached: Some(false),
            spend_control_reached: Some(true),
            windows: Some(OpenAiWindows {
                primary: Some(OpenAiPrimaryWindow {
                    used_percent: Some(60.0),
                }),
            }),
        };
        assert_eq!(classify_openai(&spend), CandidateState::ready());
    }

    #[test]
    fn openai_unavailable_without_confirmed_exhaustion_does_not_open_fallback() {
        // available == false alone does NOT sanction fallback.
        let unavailable = OpenAiResourceState {
            available: Some(false),
            ..Default::default()
        };
        assert_eq!(classify_openai(&unavailable), CandidateState::ready());
    }

    #[test]
    fn openai_unknown_or_missing_telemetry_does_not_open_fallback() {
        // missing used_percent / no windows => not confirmed exhaustion => ready.
        let unknown_used = OpenAiResourceState {
            available: Some(true),
            limit_reached: Some(false),
            spend_control_reached: Some(false),
            windows: None,
        };
        assert_eq!(classify_openai(&unknown_used), CandidateState::ready());

        // limit_reached NOT present but no used% => still recognized as not
        // confirmed exhausted (missing used_percent does NOT sanction fallback).
        let missing_used = OpenAiResourceState {
            available: Some(true),
            limit_reached: Some(false),
            spend_control_reached: Some(false),
            windows: Some(OpenAiWindows { primary: None }),
        };
        assert_eq!(classify_openai(&missing_used), CandidateState::ready());
    }

    #[test]
    fn deepseek_eligible_only_when_available_and_positive_balance() {
        let healthy = DeepSeekResourceState {
            is_available: Some(true),
            balance_infos: Some(vec![DeepSeekBalanceInfo {
                currency: Some("CNY".to_string()),
                total_balance: Some("90.75".to_string()),
                ..Default::default()
            }]),
        };
        assert_eq!(classify_deepseek(&healthy, "CNY"), CandidateState::ready());

        let empty = DeepSeekResourceState {
            is_available: Some(true),
            balance_infos: Some(vec![DeepSeekBalanceInfo {
                currency: Some("CNY".to_string()),
                total_balance: Some("0.00".to_string()),
                ..Default::default()
            }]),
        };
        assert_eq!(
            classify_deepseek(&empty, "CNY"),
            CandidateState::not_ready()
        );

        let unknown = DeepSeekResourceState::default();
        assert_eq!(
            classify_deepseek(&unknown, "CNY"),
            CandidateState::not_ready()
        );
    }

    #[test]
    fn deepseek_wrong_currency_or_negative_is_not_ready() {
        let wrong_currency = DeepSeekResourceState {
            is_available: Some(true),
            balance_infos: Some(vec![DeepSeekBalanceInfo {
                currency: Some("USD".to_string()),
                total_balance: Some("90.75".to_string()),
                ..Default::default()
            }]),
        };
        assert_eq!(
            classify_deepseek(&wrong_currency, "CNY"),
            CandidateState::not_ready()
        );

        let negative = DeepSeekResourceState {
            is_available: Some(true),
            balance_infos: Some(vec![DeepSeekBalanceInfo {
                currency: Some("CNY".to_string()),
                total_balance: Some("-1.0".to_string()),
                ..Default::default()
            }]),
        };
        assert_eq!(
            classify_deepseek(&negative, "CNY"),
            CandidateState::not_ready()
        );
    }

    // Exercises the live observe path of the resource-state client against a
    // real (loopback) sanitized OpenAI + DeepSeek surface, with the credential
    // read from an env var name at request time and dropped on every path.
    #[tokio::test]
    async fn resource_state_client_observes_openai_and_deepseek_live() {
        let openai_url = mock_server(vec![(
            "/resource/openai-codex",
            r#"{"available":true,"limit_reached":false,"spend_control_reached":false,
                "windows":{"primary":{"used_percent":11}}} "#,
        )])
        .await;
        let deepseek_url = mock_server(vec![(
            "/user/balance",
            r#"{"is_available":true,"balance_infos":[
                {"currency":"CNY","total_balance":"90.75"}
            ]}"#,
        )])
        .await;

        let client = ResourceStateFactsClient::new(
            Some(format!("{openai_url}/resource/openai-codex")),
            Some("S2D_TEST_OPENAI_TOKEN".into()),
            Some(format!("{deepseek_url}/user/balance")),
            Some("S2D_TEST_DEEPSEEK_KEY".into()),
            Some("CNY".into()),
        );

        unsafe { std::env::set_var("S2D_TEST_OPENAI_TOKEN", "dummy") };
        unsafe { std::env::set_var("S2D_TEST_DEEPSEEK_KEY", "dummy") };
        assert_eq!(client.observe_openai().await, CandidateState::ready());
        assert_eq!(client.observe_deepseek().await, CandidateState::ready());
        unsafe { std::env::remove_var("S2D_TEST_OPENAI_TOKEN") };
        unsafe { std::env::remove_var("S2D_TEST_DEEPSEEK_KEY") };

        // Missing credential env => NOT confirmed included-allowance exhaustion:
        // fail toward Luna (no DeepSeek spill), never panic.
        let client2 = ResourceStateFactsClient::new(
            Some(format!("{openai_url}/resource/openai-codex")),
            Some("S2D_TEST_MISSING_TOKEN".into()),
            None,
            None,
            None,
        );
        assert_eq!(client2.observe_openai().await, CandidateState::ready());
    }

    // F4 (Hermes review): an OpenAI resource FETCH FAILURE (HTTP 500) or a
    // MALFORMED payload is NOT confirmed included-allowance exhaustion and must
    // fail toward Luna (ready) — it must NOT open a DeepSeek spill.
    #[tokio::test]
    async fn openai_fetch_failure_or_malformed_fails_toward_luna_no_spill() {
        // A 500-returning OpenAI resource endpoint.
        let fail_router = axum::Router::new().route(
            "/resource/openai-codex",
            axum::routing::get(|| async {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json("boom".to_string()),
                )
            }),
        );
        let fail_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let fail_addr = fail_listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(fail_listener, fail_router).await.unwrap() });

        let fail_client = ResourceStateFactsClient::new(
            Some(format!("http://{fail_addr}/resource/openai-codex")),
            Some("S2D_TEST_FETCH_FAIL_TOKEN".into()),
            None,
            None,
            None,
        );
        unsafe { std::env::set_var("S2D_TEST_FETCH_FAIL_TOKEN", "dummy") };
        assert_eq!(
            fail_client.observe_openai().await,
            CandidateState::ready(),
            "an OpenAI resource fetch failure must fail toward Luna (no DeepSeek spill)"
        );
        unsafe { std::env::remove_var("S2D_TEST_FETCH_FAIL_TOKEN") };

        // A garbage/malformed-JSON OpenAI resource endpoint.
        let garbage_router = axum::Router::new().route(
            "/resource/openai-codex",
            axum::routing::get(|| async { (StatusCode::OK, axum::body::Body::from("not-json{")) }),
        );
        let garbage_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let garbage_addr = garbage_listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(garbage_listener, garbage_router).await.unwrap() });

        let garbage_client = ResourceStateFactsClient::new(
            Some(format!("http://{garbage_addr}/resource/openai-codex")),
            Some("S2D_TEST_GARBAGE_TOKEN".into()),
            None,
            None,
            None,
        );
        unsafe { std::env::set_var("S2D_TEST_GARBAGE_TOKEN", "dummy") };
        assert_eq!(
            garbage_client.observe_openai().await,
            CandidateState::ready(),
            "a malformed OpenAI resource payload must fail toward Luna (no DeepSeek spill)"
        );
        unsafe { std::env::remove_var("S2D_TEST_GARBAGE_TOKEN") };
    }

    // --- DeepSeek observability telemetry (regression-fix) ---

    #[tokio::test]
    async fn deepseek_success_publishes_last_successful_telemetry() {
        let url = mock_server(vec![(
            "/user/balance",
            r#"{"is_available":true,"balance_infos":[{"currency":"CNY","total_balance":"55.14","granted_balance":"10.0","topped_up_balance":"45.14"}]}"#,
        )])
        .await;
        let telemetry = SharedDeepSeekTelemetry::new();
        let client = ResourceStateFactsClient::new(
            None,
            None,
            Some(format!("{url}/user/balance")),
            Some("S2D_TEST_DEEPSEEK_KEY".into()),
            Some("CNY".into()),
        )
        .with_deepseek_telemetry(telemetry.clone());
        unsafe { std::env::set_var("S2D_TEST_DEEPSEEK_KEY", "dummy") };
        let state = client.observe_deepseek().await;
        unsafe { std::env::remove_var("S2D_TEST_DEEPSEEK_KEY") };
        assert_eq!(state, CandidateState::ready());
        let snap = telemetry
            .get()
            .expect("successful observation must publish telemetry");
        assert_eq!(snap.is_available, Some(true));
        assert_eq!(snap.currency.as_deref(), Some("CNY"));
        assert_eq!(snap.total_balance, Some(55.14));
        assert_eq!(snap.granted_balance, Some(10.0));
        assert_eq!(snap.topped_up_balance, Some(45.14));
        assert!(snap.observed_at > 0.0);
        assert!(snap.error.is_empty());
    }

    #[tokio::test]
    async fn deepseek_failure_keeps_last_successful_telemetry() {
        let gauge = Gauge::default();
        let ep = Endpoint::new(
            r#"{"is_available":true,"balance_infos":[{"currency":"CNY","total_balance":"55.14"}]}"#,
        );
        let url = spawn_facts_server(vec![("/user/balance", ep.clone())], gauge).await;
        let telemetry = SharedDeepSeekTelemetry::new();
        let client = ResourceStateFactsClient::new(
            None,
            None,
            Some(format!("{url}/user/balance")),
            Some("S2D_TEST_DEEPSEEK_KEY".into()),
            Some("CNY".into()),
        )
        .with_deepseek_telemetry(telemetry.clone());
        unsafe { std::env::set_var("S2D_TEST_DEEPSEEK_KEY", "dummy") };
        assert_eq!(client.observe_deepseek().await, CandidateState::ready());
        let before = telemetry
            .get()
            .expect("successful observation must publish telemetry");
        // Flip the factual surface to a 500: the cycle fails closed for
        // readiness and MUST NOT overwrite the last-successful telemetry.
        ep.set(500, r#"{"error":"boom"}"#);
        assert_eq!(client.observe_deepseek().await, CandidateState::not_ready());
        unsafe { std::env::remove_var("S2D_TEST_DEEPSEEK_KEY") };
        let after = telemetry
            .get()
            .expect("last-successful telemetry must survive a failed cycle");
        assert_eq!(after.total_balance, before.total_balance);
        assert_eq!(after.observed_at, before.observed_at);
        assert_eq!(after.currency.as_deref(), Some("CNY"));
    }

    #[tokio::test]
    async fn deepseek_unconfigured_publishes_no_telemetry() {
        let telemetry = SharedDeepSeekTelemetry::new();
        let client = ResourceStateFactsClient::new(None, None, None, None, None)
            .with_deepseek_telemetry(telemetry.clone());
        assert_eq!(
            client.observe_deepseek().await,
            CandidateState::not_ready(),
            "unconfigured DeepSeek must fail closed for readiness"
        );
        assert!(
            telemetry.get().is_none(),
            "unconfigured DeepSeek must not fabricate a telemetry snapshot"
        );
    }
}
