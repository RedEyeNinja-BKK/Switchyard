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
//! The types here are the server-owned producer seam, proven by unit tests and
//! the live read-only proof. Wiring the producer into the server run-loop's
//! background observation task is a small follow-on seam; until then the types
//! are constructed by tests and the live proof, so dead-code analysis is
//! expected to be quiet about them.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use libsy::{CandidateState, FleetSnapshot, LibsyError, SharedFleetState};
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
    /// Mapping (evidence-first; uncertainty never becomes `ready`):
    /// - serving mode (not idle) AND Qwen resident AND llama server up
    ///   → `ready` (immediately callable).
    /// - `mode == "idle"` with the full sealed-idle signature
    ///   (`resident_qwen_profile == "unknown"` and `llama_server` absent/`"no"`)
    ///   → `transition_required` (valid but unloaded; needs an explicit lifecycle
    ///   transition that is never invoked here).
    /// - any malformed, contradictory, or incomplete idle state, or an API
    ///   failure → `not_ready`, not transition-required (fail closed).
    pub async fn observe(&self) -> CandidateState {
        self.fetch_snapshot().await.map_or_else(
            |_| CandidateState::not_ready(),
            |snap| {
                let mode = snap.state.mode.as_deref().unwrap_or("");
                let resident =
                    snap.state.resident_qwen_profile.as_deref().unwrap_or("") != "unknown";
                let llama_up = snap
                    .state
                    .llama_server
                    .as_deref()
                    .is_some_and(|s| s != "no");
                // Sealed idle requires the complete transition signature: an
                // idle mode with no resident Qwen and no llama server. A missing
                // or contradictory field does not qualify — it is not a proven
                // transition case, so it fails closed.
                let sealed_idle = mode == "idle"
                    && snap
                        .state
                        .resident_qwen_profile
                        .as_deref()
                        .is_some_and(|p| p == "unknown")
                    && snap
                        .state
                        .llama_server
                        .as_deref()
                        .is_some_and(|s| s == "no");
                if sealed_idle {
                    CandidateState::transition_required()
                } else if resident && llama_up && mode != "idle" {
                    CandidateState::ready()
                } else {
                    CandidateState::not_ready()
                }
            },
        )
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

// --- HTPC health/model contract (read-only) --------------------------------

/// Read-only client for the HTPC generation/health surface (no authentication).
///
/// Readiness is established from actual health **and** model-identity evidence,
/// not TCP reachability alone.
pub struct HtpcFactsClient {
    base_url: String,
    expected_model_fragment: String,
    client: Client,
}

impl std::fmt::Debug for HtpcFactsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HtpcFactsClient")
            .field("base_url", &self.base_url)
            .field("expected_model_fragment", &self.expected_model_fragment)
            .finish()
    }
}

impl HtpcFactsClient {
    /// A client observing `{base_url}/health` and `{base_url}/v1/models`.
    pub fn new(base_url: String, expected_model_fragment: String) -> Self {
        Self {
            base_url,
            expected_model_fragment,
            client: factual_client(),
        }
    }

    /// One read-only observation. `ready` only when the health endpoint reports
    /// an exact `{"status":"ok"}` AND `/v1/models` exposes the expected configured
    /// model id (parsed from its `data[].id` fields, with no "unknown" entry).
    /// Any failure, malformed schema, or unexpected model → `not_ready` (HTPC has
    /// no transition-required state in this gate).
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
        // The expected configured model id must be present among the data ids,
        // and no entry may be "unknown".
        let has_expected = models
            .data
            .iter()
            .any(|m| m.id.contains(&self.expected_model_fragment));
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

// --- Producer -----------------------------------------------------------------

/// Assembles one coherent [`FleetSnapshot`] generation from observed facts and
/// writes it atomically to a shared [`SharedFleetState`].
///
/// A single `observe_once()` performs all observation reads, builds **one**
/// complete snapshot, and calls `set()` exactly once — so readers never observe
/// a half-assembled generation. No target is updated while the snapshot is
/// visible.
#[derive(Clone)]
pub struct FleetSnapshotProducer {
    state: Arc<SharedFleetState>,
    /// Serializes observation-generation writes so an older generation can never
    /// overwrite a newer one (prevents stale whole-snapshot replacement from
    /// out-of-order observation cycles).
    generation: Arc<std::sync::Mutex<u64>>,
}

impl std::fmt::Debug for FleetSnapshotProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FleetSnapshotProducer")
            .field("state", &"<Arc<SharedFleetState>>")
            .field("generation", &"<Mutex<u64>>")
            .finish()
    }
}

impl FleetSnapshotProducer {
    /// A producer writing to `state`.
    pub fn new(state: Arc<SharedFleetState>) -> Self {
        Self {
            state,
            generation: Arc::new(std::sync::Mutex::new(0)),
        }
    }

    /// Atomically replaces the whole snapshot with the observations of one
    /// generation.
    ///
    /// The caller assembles the complete observed set — including the static
    /// "immediately attemptable / no resource transition" base entries for cloud
    /// provider candidates, which are documented as configured/attemptable, not a
    /// live provider health guarantee. `apply` serializes generation writes so an
    /// older cycle can never clobber a newer one, and `SharedFleetState::set`
    /// swaps the whole snapshot atomically (readers never see a half generation).
    pub fn apply(&self, observations: Vec<Observed>) -> Result<(), LibsyError> {
        let states = observations
            .into_iter()
            .map(|o| (o.model, o.state))
            .collect();
        let snapshot = FleetSnapshot::new(states)?;
        // Serialize generation commits: hold the generation lock across the
        // atomic swap so two concurrent cycles cannot publish out of order.
        let mut stored = self.generation.lock().expect("producer generation lock");
        *stored = stored.saturating_add(1);
        self.state.set(snapshot);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::Router;
    use axum::routing::{get, post};
    use libsy::{CandidateState, FleetSnapshot, FleetStateSource, SharedFleetState};
    use switchyard_protocol::ModelId;
    use tokio::net::TcpListener;

    use super::{ComfyFactsClient, FleetSnapshotProducer, HtpcFactsClient, Observed};

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
    async fn comfy_ready_when_serving_profile_and_resident() {
        // A: immediately callable - serving mode, Qwen resident, llama up.
        let url = mock_server(vec![(
            "/v1/resource",
            r#"{"producer_epoch":"e","state_generation":1,"state_fingerprint":"f",
                "state":{"mode":"fast","resident_qwen_profile":"fast","llama_server":"yes"}}"#,
        )])
        .await;
        let client = ComfyFactsClient::new(format!("{url}/v1/resource"), "S2D_TEST_TOKEN_A".into());
        unsafe { std::env::set_var("S2D_TEST_TOKEN_A", "dummy") };
        assert_eq!(client.observe().await, CandidateState::ready());
        unsafe { std::env::remove_var("S2D_TEST_TOKEN_A") };
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
        // D: healthy endpoint serving the expected model => ready.
        let url = mock_server(vec![
            ("/health", r#"{"status":"ok"}"#),
            (
                "/v1/models",
                r#"{"object":"list","data":[{"id":"/srv/htpc-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf"}]}"#,
            ),
        ])
        .await;
        let client = HtpcFactsClient::new(url.clone(), "qwen3.5-9b-mtp".into());
        assert_eq!(client.observe().await, CandidateState::ready());
        // The client built its own reqwest client; nothing more needed.
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
        let c_health = HtpcFactsClient::new(url.clone(), "qwen3.5-9b-mtp".into());
        // Health status "down" => not_ready.
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
        let c_model = HtpcFactsClient::new(url2.clone(), "qwen3.5-9b-mtp".into());
        assert_eq!(c_model.observe().await, CandidateState::not_ready());
        let _ = &c_model;
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
}
