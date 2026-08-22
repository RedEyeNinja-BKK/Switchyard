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
//! The server-owned seam is exercised by the normal runtime path via
//! [`FleetReadinessMonitor`], so its types are not dead code.

use std::sync::Arc;
use std::time::Duration;

use libsy::{CandidateState, FleetSnapshot, FleetStateSource, LibsyError, SharedFleetState};
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
    /// non-fail-closed state; see [`classify_comfy`] for the exact rules):
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
        // Guard drops the credential on every path (success or error) before the
        // future completes; it is never retained in any struct.
        struct DropBearer(String);
        impl Drop for DropBearer {
            fn drop(&mut self) {
                self.0.clear();
            }
        }
        let _bearer = DropBearer(token.clone());
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
    let known_resident = COMFY_RESIDENT_KNOWN.contains(&resident);
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

// --- Producer -----------------------------------------------------------------

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
    comfy: ComfyFactsClient,
    comfy_model: ModelId,
    htpc: HtpcFactsClient,
    htpc_model: ModelId,
    /// Static cloud base states: "configured / immediately attemptable", never a
    /// live provider health guarantee.
    cloud_base: Vec<Observed>,
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
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

impl FleetReadinessMonitor {
    /// Constructs a monitor over `state` (which must be the same
    /// [`SharedFleetState`] injected into the FleetRouter server). Local
    /// candidates are observed from `comfy`/`htpc`; `cloud_base` supplies the
    /// static cloud base states. `interval` is the spacing between observation
    /// cycles.
    pub fn new(
        comfy: ComfyFactsClient,
        comfy_model: ModelId,
        htpc: HtpcFactsClient,
        htpc_model: ModelId,
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
            state,
            interval,
        }
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
        let (comfy_state, htpc_state) = tokio::join!(self.comfy.observe(), self.htpc.observe());
        let mut states = Vec::with_capacity(self.cloud_base.len() + 2);
        for observed in &self.cloud_base {
            states.push((observed.model.clone(), observed.state));
        }
        states.push((self.comfy_model.clone(), comfy_state));
        states.push((self.htpc_model.clone(), htpc_state));
        let snapshot = FleetSnapshot::new(states)?;
        self.state.set(snapshot);
        Ok(())
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
    use axum::routing::{get, post};
    use libsy::{CandidateState, FleetSnapshot, FleetStateSource, SharedFleetState};
    use switchyard_protocol::ModelId;
    use tokio::net::TcpListener;

    use super::{
        ComfyFactsClient, FleetReadinessMonitor, FleetSnapshotProducer, HtpcFactsClient, Observed,
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
            comfy,
            comfy_model,
            htpc,
            htpc_model,
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
}
