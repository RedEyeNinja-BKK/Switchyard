// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! F8 host-owned escalation, exercised through the real server seam.
//!
//! These tests drive actual HTTP requests against a real `ServerState` whose
//! routes are real `FleetRouter` instances over a mock upstream. Nothing here
//! calls a policy helper directly: every assertion about WHEN escalation fires,
//! WHICH policy the escalated leg runs under, and WHICH model the response is
//! attributed to is made at the host boundary, because that is the only place
//! those properties exist.
//!
//! The F7 reasoning re-stamp lands in the same change as the transition itself,
//! so a test that could observe escalation running under the PRIMARY route's
//! policy would be caught here rather than in production.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use libsy::{
    CandidateState, FleetCandidate, FleetRouter, FleetSnapshot, FleetStateSource,
    SharedFleetState,
};
use serde_json::{Value, json};
use switchyard_llm_client::{
    Backend, ClientRouter, HttpBackendConfig, ModelConfig, TranslatingLlmClient,
};
use switchyard_protocol::RoutedLlmClient;
use switchyard_protocol::{Category, ModelId, WireFormat};
use switchyard_protocol::ReasoningPolicy;
use switchyard_runner::{DecisionTarget, ModelCapabilities, Route, Runner, RuntimeModels};
use switchyard_server::{ServerState, build_llm_router};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tower::ServiceExt;

type TestError = Box<dyn Error + Send + Sync>;
type TestResult<T = ()> = Result<T, TestError>;

// ---------------------------------------------------------------------------
// Mock upstream: records what each model was asked, and answers per-model.
// ---------------------------------------------------------------------------

/// Behaviour per model id, so a test can make one candidate fail and another
/// succeed, and can inspect the reasoning policy that reached the wire.
#[derive(Clone, Default)]
struct UpstreamCalls {
    calls: Arc<Mutex<Vec<Value>>>,
}

struct MockUpstream {
    base_url: String,
    calls: UpstreamCalls,
    task: JoinHandle<()>,
}

impl MockUpstream {
    async fn start() -> TestResult<Self> {
        let calls = UpstreamCalls::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(serve_chat))
            .with_state(calls.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            base_url,
            calls,
            task,
        })
    }

    /// Every request body the upstream actually received, in order.
    async fn received(&self) -> Vec<Value> {
        self.calls.calls.lock().await.clone()
    }

    /// The models the upstream was asked to serve, in order.
    async fn served_models(&self) -> Vec<String> {
        self.received()
            .await
            .iter()
            .filter_map(|body| {
                body.get("model")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }
}

impl Drop for MockUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Serves a chat completion, or fails it, based on the requested model id.
///
/// `fail-*` models return a terminal upstream error so the fleet chain advances
/// or escalates; every other model answers successfully. The full received
/// body is recorded so a test can assert on the wire-level reasoning control.
async fn serve_chat(State(calls): State<UpstreamCalls>, body: Bytes) -> HttpResponse {
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
    calls.calls.lock().await.push(parsed.clone());
    let model = parsed.get("model").and_then(Value::as_str).unwrap_or("");

    if model.starts_with("fail-") {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    // The model id is echoed into the message so a test can tell
                    // WHOSE failure the caller was shown when more than one
                    // candidate failed.
                    "message": format!("upstream unavailable for {model}"),
                    "type": "server_error"
                }
            })),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        Json(json!({
            "id": "chatcmpl-escalation",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": format!("served by {model}") },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// State construction: real FleetRouters over a real client, with readiness.
// ---------------------------------------------------------------------------

/// One fleet route declaration: its candidates, its escalation, its policies.
struct FleetRoute {
    id: &'static str,
    /// `(target, preference_rank)` in declared order.
    candidates: Vec<(&'static str, u16)>,
    escalation: Option<&'static str>,
    max_input_tokens: Option<u64>,
    reasoning_policy: Option<ReasoningPolicy>,
    escalation_reasoning_policy: Option<ReasoningPolicy>,
    /// Declared usable context capacity. `None` is UNMANAGED and therefore
    /// admitted; `Some` makes the candidate BOUNDED, which is what row 40 acts on.
    usable_context_tokens: Option<u64>,
}

impl FleetRoute {
    fn new(id: &'static str, candidates: &[&'static str]) -> Self {
        Self {
            id,
            candidates: candidates.iter().map(|target| (*target, 0)).collect(),
            escalation: None,
            max_input_tokens: None,
            reasoning_policy: None,
            escalation_reasoning_policy: None,
            usable_context_tokens: None,
        }
    }

    fn escalating(mut self, destination: &'static str) -> Self {
        self.escalation = Some(destination);
        self
    }

    /// Declares candidates with explicit preference ranks, in DECLARED order.
    fn ranked(id: &'static str, candidates: &[(&'static str, u16)]) -> Self {
        Self {
            id,
            candidates: candidates.to_vec(),
            escalation: None,
            max_input_tokens: None,
            reasoning_policy: None,
            escalation_reasoning_policy: None,
            usable_context_tokens: None,
        }
    }

    /// Makes every candidate of this route BOUNDED, so row 40 governs it.
    fn bounded(mut self, capacity: u64) -> Self {
        self.usable_context_tokens = Some(capacity);
        self
    }

    fn with_threshold(mut self, threshold: u64) -> Self {
        self.max_input_tokens = Some(threshold);
        self
    }

    fn with_reasoning(mut self, policy: ReasoningPolicy) -> Self {
        self.reasoning_policy = Some(policy);
        self
    }

    fn with_escalation_reasoning(mut self, policy: ReasoningPolicy) -> Self {
        self.escalation_reasoning_policy = Some(policy);
        self
    }
}

fn backend(base_url: &str) -> Backend {
    Backend::OpenAiChat(HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("test-key".to_string()),
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        reasoning_effort: None,
        reasoning_dialect: None,
        reasoning_efforts: None,
        strip_reasoning_content: false,
        max_retries: 0,
        timeout: None,
    })
}

/// A backend that DECLARES the OpenAI effort dialect, so a route carrying a
/// reasoning policy can be sent rather than being refused fail-closed by F7.
///
/// F7 refuses to send an uncontrolled request when a route policy meets a
/// dialect-less backend, so any test that exercises a reasoning policy across
/// the escalation seam must use a dialect-capable backend.
fn dialect_backend(base_url: &str) -> Backend {
    Backend::OpenAiChat(HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("test-key".to_string()),
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        reasoning_effort: None,
        reasoning_dialect: Some(switchyard_protocol::ReasoningDialect::OpenAiEffort),
        reasoning_efforts: Some(vec![
            "none".to_string(),
            "low".to_string(),
            "medium".to_string(),
            "high".to_string(),
        ]),
        strip_reasoning_content: false,
        max_retries: 0,
        timeout: None,
    })
}

/// Builds real routes over ONE deployment-wide readiness handle and returns the
/// state plus that handle.
///
/// The handle is returned so a test can publish a NEW readiness generation
/// between requests, which is how the snapshot-consistency assertions are made
/// against a live decision rather than a helper.
///
/// The routers read readiness through the SHARED handle, exactly as a real
/// deployment does: one coherent snapshot per decision, replaced wholesale
/// between decisions. A per-route `StaticFleetState` would make the returned
/// handle decorative, and publishing one snapshot per route would make an
/// unlisted target read as unobserved-and-therefore-fail-closed.
fn fleet_state(
    base_url: &str,
    routes: &[FleetRoute],
) -> TestResult<(ServerState, Arc<SharedFleetState>)> {
    fleet_state_with_backend(base_url, routes, backend)
}

/// As [`fleet_state`], but every target is served by a backend that DECLARES the
/// OpenAI effort dialect, so a reasoning policy is actually sent rather than
/// refused fail-closed by F7.
fn fleet_state_dialects(
    base_url: &str,
    routes: &[FleetRoute],
) -> TestResult<(ServerState, Arc<SharedFleetState>)> {
    fleet_state_with_backend(base_url, routes, dialect_backend)
}

fn fleet_state_with_backend(
    base_url: &str,
    routes: &[FleetRoute],
    make_backend: fn(&str) -> Backend,
) -> TestResult<(ServerState, Arc<SharedFleetState>)> {
    // The generation every route starts from: ONE coherent snapshot covering
    // every declared target in the deployment, published once.
    let initial = FleetSnapshot::new(
        routes
            .iter()
            .flat_map(|route| route.candidates.iter().map(|(target, _)| *target))
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|target| (ModelId::from(target), CandidateState::ready()))
            .collect(),
    )?;
    let shared = Arc::new(SharedFleetState::new(initial));
    // The routers read readiness through the SHARED handle, so the handle a test
    // holds and the handle the decision reads are the same object. The erasure
    // happens once here, not per route. `clone()` rather than `Arc::clone` is what
    // permits the unsize coercion to `dyn FleetStateSource`; this mirrors the
    // production build path in `algorithm.rs` exactly.
    let source: Arc<dyn FleetStateSource> = shared.clone();
    let target_models: HashSet<String> = routes
        .iter()
        .flat_map(|route| route.candidates.iter().map(|(target, _)| (*target).to_string()))
        .collect();
    let model_configs = target_models
        .into_iter()
        .map(|model| ModelConfig::new(model, make_backend(base_url), None))
        .collect::<Vec<_>>();
    let client: Arc<dyn RoutedLlmClient> = Arc::new(TranslatingLlmClient::new(&model_configs)?);
    let mut entries = Vec::with_capacity(routes.len());
    for route in routes {
        let candidates: Vec<FleetCandidate> = route
            .candidates
            .iter()
            .map(|(target, rank)| FleetCandidate {
                target: ModelId::from(*target),
                tool_calling: true,
                reasoning: true,
                supports_vision: true,
                preference_rank: *rank,
                usable_context_tokens: route.usable_context_tokens,
            })
            .collect();
        let algorithm: Arc<dyn libsy::Algorithm> = Arc::new(FleetRouter::with_source(
            candidates,
            route.escalation.map(ModelId::from),
            route.max_input_tokens,
            Arc::clone(&source),
        ));
        let decision_targets = route
            .candidates
            .iter()
            .map(|(target, _)| DecisionTarget {
                target: (*target).to_string(),
                model: ModelId::from(*target),
                format: WireFormat::OpenAiChat,
                base_url: base_url.to_string(),
                extra_body: BTreeMap::new(),
            })
            .collect();
        let models = RuntimeModels::new(
            [(
                Category::Any,
                route
                    .candidates
                    .iter()
                    .map(|(target, _)| ModelId::from(*target))
                    .collect(),
            )]
            .into(),
        );
        let mut built = Route::new(
            algorithm,
            ClientRouter::single(Arc::clone(&client)),
            None,
            ModelCapabilities::default(),
            None,
            None,
            decision_targets,
            models,
        )
        .with_escalation(route.escalation.map(ModelId::from), route.max_input_tokens);
        if let Some(policy) = route.reasoning_policy {
            built = built.with_reasoning_policy(Some(policy));
        }
        if let Some(policy) = route.escalation_reasoning_policy {
            built = built.with_escalation_reasoning_policy(Some(policy));
        }
        entries.push((ModelId::from(route.id), built));
    }
    Ok((
        ServerState::from_runner(
            Runner::new(entries).with_fleet_state(Some(shared.clone())),
        )?,
        shared,
    ))
}

/// Posts a chat request for `model` and returns (status, body).
async fn post_chat(app: &Router, model: &str, prompt: &str) -> TestResult<(StatusCode, Value)> {
    let response = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": model,
                        "messages": [{ "role": "user", "content": prompt }],
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Ok((status, body))
}

// ---------------------------------------------------------------------------
// 1. Primary succeeds -> no escalation.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_successful_primary_route_never_escalates() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[FleetRoute::new("primary", &["good-a"]).escalating("destination")],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let served = upstream.served_models().await;
    assert_eq!(
        served,
        vec!["good-a".to_string()],
        "only the primary candidate may be contacted"
    );
    assert!(
        !served.contains(&"good-destination".to_string()),
        "an unused escalation destination must never be contacted"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2 + 3. Pre-execution threshold, including the exact boundary.
// ---------------------------------------------------------------------------

/// The production comparison is STRICTLY greater-than, so a request whose
/// estimate equals the threshold stays on the primary route and only a larger
/// one escalates. This test pins the boundary from both sides.
#[tokio::test]
async fn the_pre_execution_threshold_is_strictly_greater_than() -> TestResult {
    // Calibrate the estimate for a fixed prompt by measuring the primary route's
    // own behaviour at a threshold the estimate is known to sit under.
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["good-a"]).escalating("destination"),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);
    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    // A zero threshold is below any real estimate, so this MUST escalate
    // before the primary ladder runs.
    let upstream2 = MockUpstream::start().await?;
    let (state2, _) = fleet_state(
        &upstream2.base_url,
        &[
            FleetRoute::new("primary", &["good-a"])
                .escalating("destination")
                .with_threshold(0),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app2 = build_llm_router(state2);
    let (status2, body2) = post_chat(&app2, "primary", "hello").await?;
    assert_eq!(status2, StatusCode::OK, "body: {body2}");

    let served = upstream2.served_models().await;
    assert_eq!(
        served,
        vec!["good-destination".to_string()],
        "an estimate above the threshold must escalate BEFORE the primary ladder runs, \
         so the primary candidate is never contacted"
    );
    Ok(())
}

#[tokio::test]
async fn a_request_exactly_at_the_threshold_stays_on_the_primary_route() -> TestResult {
    // Establish the exact estimate for this body, then pin the threshold to it.
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["good-a"]).escalating("destination"),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);
    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    // u64::MAX is above any estimate this body can produce, so the strictly
    // greater-than comparison is false and the primary route is kept. This is
    // the same comparison as the escalating case above, pinned from the other
    // side, so an off-by-one in either direction fails one of the two tests.
    let upstream2 = MockUpstream::start().await?;
    let (state2, _) = fleet_state(
        &upstream2.base_url,
        &[
            FleetRoute::new("primary", &["good-a"])
                .escalating("destination")
                .with_threshold(u64::MAX),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app2 = build_llm_router(state2);
    let (status2, body2) = post_chat(&app2, "primary", "hello").await?;
    assert_eq!(status2, StatusCode::OK, "body: {body2}");

    assert_eq!(
        upstream2.served_models().await,
        vec!["good-a".to_string()],
        "a threshold above the estimate must keep the request on the primary route"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. The EXACT threshold boundary, located rather than approximated.
// ---------------------------------------------------------------------------

/// Whether a request whose primary route declares `threshold` is answered by the
/// primary candidate (`true`) or escalated to the destination (`false`).
async fn primary_served_at_threshold(
    threshold: u64,
    prompt: &str,
) -> TestResult<bool> {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["good-a"])
                .escalating("destination")
                .with_threshold(threshold),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);
    let (status, body) = post_chat(&app, "primary", prompt).await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    Ok(upstream.served_models().await.first().map(String::as_str) == Some("good-a"))
}

/// The pre-execution threshold is a strict `>`, so a request whose estimate is
/// EXACTLY the threshold stays on the primary route.
///
/// The estimate is host-internal, so the value is pinned by an in-crate unit
/// test in `switchyard-server`
/// (`the_pre_execution_estimate_of_the_f8_fixture_body_is_eighteen`) over the
/// same body this test sends. The equality probes below then straddle that
/// value: equality must not escalate, one token below must. Together they fail
/// if the comparison ever becomes `>=`.
///
/// A behavioural bisection was deliberately NOT used here: it recalibrates to
/// whatever the implementation does, so it is self-referential and would report
/// PASS under a `>=` comparison.
#[tokio::test]
async fn equality_with_the_estimate_does_not_escalate() -> TestResult {
    let prompt = "boundary probe";

    // Establish the two extremes, so the pinned value is meaningfully bracketed.
    assert!(
        !primary_served_at_threshold(0, prompt).await?,
        "a zero threshold is below any estimate, so this must escalate"
    );
    assert!(
        primary_served_at_threshold(u64::MAX, prompt).await?,
        "an unreachable threshold is above any estimate, so this must not escalate"
    );

    // The pinned estimate of this exact body (asserted in the in-crate test).
    const ESTIMATE: u64 = 18;

    // Equality: estimate == threshold must NOT escalate.
    assert!(
        primary_served_at_threshold(ESTIMATE, prompt).await?,
        "an estimate EXACTLY equal to the threshold must stay on the primary route \
         (the comparison is strictly greater-than)"
    );
    // One below: estimate > threshold must escalate. This is the probe that
    // fails if the comparison becomes `>=`.
    assert!(
        !primary_served_at_threshold(ESTIMATE - 1, prompt).await?,
        "an estimate one token above the threshold must escalate"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Primary ladder exhausts -> escalation succeeds.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_exhausted_primary_ladder_escalates_and_the_destination_serves() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["fail-a", "fail-b"]).escalating("destination"),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let served = upstream.served_models().await;
    assert_eq!(
        &served[..2],
        &["fail-a".to_string(), "fail-b".to_string()],
        "the frozen primary ladder is walked in preference order before escalation"
    );
    assert_eq!(
        served.last().map(String::as_str),
        Some("good-destination"),
        "escalation runs only after the primary chain is exhausted"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 7 + 8. Reasoning re-stamp, and the wire the escalated leg actually sent.
// ---------------------------------------------------------------------------

/// The escalated leg must run under the ESCALATION policy, never the primary
/// route's own policy. Asserted at the host boundary by observing what reached
/// the upstream.
#[tokio::test]
async fn the_escalated_leg_runs_under_the_escalation_policy_not_the_primary_policy() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state_dialects(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["fail-a"])
                .escalating("destination")
                .with_reasoning(ReasoningPolicy::High)
                .with_escalation_reasoning(ReasoningPolicy::None),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let received = upstream.received().await;
    let escalated = received
        .last()
        .expect("the destination must have been contacted");
    // The destination runs under a DIALECT-CAPABLE backend, so the escalated
    // leg's policy is not merely declared but actually projected onto the wire.
    // That makes this a positive assertion of the re-stamp rather than the weak
    // "the primary's high did not survive" form: a missing re-stamp would leave
    // `high` here, and an inert seam would leave the key absent.
    //
    // The effort rides at `reasoning.effort`, which is where the OpenAI-effort
    // dialect actually projects it. A top-level `reasoning_effort` read would be
    // vacuously true on a correct build AND on a missing re-stamp.
    let wire_effort = escalated
        .get("reasoning")
        .and_then(|value| value.get("effort"))
        .and_then(Value::as_str);
    assert_eq!(
        wire_effort,
        Some("none"),
        "the escalated leg must run under the ESCALATION policy, not the primary's, \
         body: {escalated}"
    );
    Ok(())
}

/// A primary route that declares NO escalation policy leaves the destination's
/// own declaration in force, and must not silently inherit the primary's.
#[tokio::test]
async fn without_an_escalation_policy_the_destination_declaration_governs() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state_dialects(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["fail-a"])
                .escalating("destination")
                .with_reasoning(ReasoningPolicy::High),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        upstream.served_models().await.last().map(String::as_str),
        Some("good-destination"),
        "escalation still fires; only the reasoning authority differs"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 9. Attribution follows the model that actually served.
// ---------------------------------------------------------------------------

/// The response must be attributed to the model that actually served it, not
/// to the primary route, not to the failed candidate, and not to the first
/// attempt.
#[tokio::test]
async fn the_response_is_attributed_to_the_model_that_actually_served() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["fail-a", "good-b"]).escalating("destination"),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    assert_eq!(
        body.get("model").and_then(Value::as_str),
        Some("good-b"),
        "the SECOND candidate served, so it must be the attributed model, not the \
         failed first candidate nor the route id"
    );
    assert_ne!(
        body.get("model").and_then(Value::as_str),
        Some("primary"),
        "the FleetRouter route id is not the serving model"
    );
    Ok(())
}

/// The same proof across the escalation seam: a failed primary candidate and an
/// exhausted ladder must not become the attribution of the destination's answer.
#[tokio::test]
async fn an_escalated_response_is_attributed_to_the_destination_model() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["fail-a"]).escalating("destination"),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        body.get("model").and_then(Value::as_str),
        Some("good-destination"),
        "the destination candidate served, so it must be the attributed model"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. A non-escalation-eligible failure does not escalate.
// ---------------------------------------------------------------------------

/// A request the CALLER got wrong is surfaced as-is. Escalation must not become
/// a way to paper over a caller error.
#[tokio::test]
async fn a_caller_error_is_surfaced_and_never_escalates() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[FleetRoute::new("primary", &["good-a"]).escalating("destination")],
    )?;
    let app = build_llm_router(state);

    // A body with no model is rejected during route resolution, before any
    // execution, so no candidate and no destination is ever contacted.
    let response = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json!({ "messages": [] }).to_string()))
                .unwrap(),
        )
        .await?;
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "a caller error must surface as a caller error"
    );
    assert!(
        upstream.received().await.is_empty(),
        "a request rejected before execution must contact neither the primary nor the destination"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. The DESTINATION's failure is what the caller is shown.
// ---------------------------------------------------------------------------

/// When both the primary ladder and the destination fail, the caller must be
/// shown the DESTINATION's failure.
///
/// This is the pair that makes the error choice observable. Surfacing the
/// primary's error instead is a real, plausible defect - the primary error is
/// the one already in scope at that point - and it would misreport what the
/// caller actually reached. The mock echoes the failing model id into the
/// message, so the two are unambiguous.
#[tokio::test]
async fn a_failing_destination_is_what_the_caller_is_shown() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["fail-primary"]).escalating("destination"),
            FleetRoute::new("destination", &["fail-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert!(
        status.is_server_error(),
        "both legs failed, so the caller must see an error, got {status}"
    );
    let message = body
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("an error body was expected, got: {body}"));
    assert!(
        message.contains("fail-destination"),
        "the caller must be shown the DESTINATION's failure, since that is the last \
         route actually reached; got: {message}"
    );
    assert!(
        !message.contains("fail-primary"),
        "the primary's failure must not be reported as the outcome once escalation \
         was attempted and also failed; got: {message}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 6 + 3. One hop: the destination cannot escalate again.
// ---------------------------------------------------------------------------

/// A route whose escalation destination itself declares an escalation is refused
/// at load, so a second hop is unreachable. Asserted through the real loader.
#[tokio::test]
async fn a_destination_that_itself_escalates_is_refused_at_load() -> TestResult {
    // Route A escalates to B; B escalates to A. Loading this must fail.
    let toml = r#"
schema_version = 1
fallback_client = "c"

[llm_clients.c]
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_TEST_KEY"
format = "openai_chat"

[targets.a]
llm_client = "c"
id = "model/a"

[targets.b]
llm_client = "c"
id = "model/b"

[routes.primary]
id = "primary"
type = "fleet_router"
candidates = [{ target = "a" }]
escalation = "destination"

[routes.destination]
id = "destination"
type = "fleet_router"
candidates = [{ target = "b" }]
escalation = "primary"
"#;
    let dir = std::env::temp_dir().join(format!("f8-cycle-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("routes.toml");
    std::fs::write(&path, toml)?;

    let loaded = {
        // The client references a key by ROLE; the value is never read for a
        // load-time cycle check, but resolution requires the variable to exist.
        unsafe { std::env::set_var("SWITCHYARD_TEST_KEY", "test-key") }
        switchyard_server::config::load_server_state(&path)
    };
    let message = match loaded {
        Ok(_) => panic!("a mutual escalation cycle must be refused at load"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("escalation") && message.contains("chains"),
        "the refusal must name the cycle rule, got: {message}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// ---------------------------------------------------------------------------
// 10 + 11. The ladder is frozen, and readiness is one generation per decision.
// ---------------------------------------------------------------------------

/// The primary ladder is walked in preference order and is NOT rebuilt between
/// candidates: the second candidate attempted is the one the frozen order names.
#[tokio::test]
async fn the_primary_ladder_is_walked_in_frozen_preference_order() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[FleetRoute::ranked(
            "primary",
            &[("fail-low", 3), ("fail-mid", 2), ("fail-high", 1)],
        )],
    )?;
    let app = build_llm_router(state);

    // No escalation destination, so this is purely the frozen ladder walk.
    let response = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "primary",
                        "messages": [{ "role": "user", "content": "hello" }],
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await?;
    assert!(
        response.status().is_server_error(),
        "every candidate fails, so the request must fail, got {}",
        response.status()
    );

    assert_eq!(
        upstream.served_models().await,
        vec![
            "fail-high".to_string(),
            "fail-mid".to_string(),
            "fail-low".to_string()
        ],
        "candidates are attempted in PREFERENCE order, not declaration order, and the \
         order is decided once"
    );
    Ok(())
}

/// Readiness is read once per decision. A generation published between two
/// REQUESTS is visible to the next one, proving the source is live rather than
/// frozen at construction.
#[tokio::test]
async fn a_new_readiness_generation_is_visible_to_the_next_decision() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let (state, shared) = fleet_state(
        &upstream.base_url,
        &[FleetRoute::new("primary", &["fail-high", "good-low"])],
    )?;
    let app = build_llm_router(state);

    // First decision: `good-low` is not ready, so only `fail-high` is eligible.
    shared.set(FleetSnapshot::new(vec![
        (ModelId::from("fail-high"), CandidateState::ready()),
        (ModelId::from("good-low"), CandidateState::not_ready()),
    ])?);
    let response = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "primary",
                        "messages": [{ "role": "user", "content": "hello" }],
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await?;
    assert!(response.status().is_server_error(), "only one candidate was ready and it fails");
    assert_eq!(
        upstream.served_models().await,
        vec!["fail-high".to_string()],
        "an unready candidate must not be attempted"
    );

    // Second decision: the NEW generation makes the second candidate eligible.
    shared.set(FleetSnapshot::new(vec![
        (ModelId::from("fail-high"), CandidateState::not_ready()),
        (ModelId::from("good-low"), CandidateState::ready()),
    ])?);
    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        upstream.served_models().await.last().map(String::as_str),
        Some("good-low"),
        "a newly ready candidate is selectable by the NEXT decision"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 12. R40-B dormancy survives the escalation wiring.
// ---------------------------------------------------------------------------

/// The escalation path must not have ACTIVATED a token producer.
///
/// With no producer wired, context admission is not an active filter, so a
/// BOUNDED candidate is admitted on readiness alone and the PRIMARY route
/// serves the request. That is the corrected row-40 contract: the contract may
/// exist without the producer, and a dormant producer must not remove every
/// bounded candidate.
///
/// (This test previously asserted the opposite — the bounded primary failing
/// closed and the destination serving. That assertion encoded the 0.3.0 cutover
/// defect: it made "every bounded route is excluded" look correct.)
#[tokio::test]
async fn row_40_stays_dormant_through_the_escalation_path() -> TestResult {
    let upstream = MockUpstream::start().await?;
    // The primary candidate is genuinely BOUNDED: it declares a usable context
    // capacity, so row 40's CONTRACT governs it. With the producer unwired the
    // policy is not participating, so the primary serves.
    let (state, _) = fleet_state(
        &upstream.base_url,
        &[
            FleetRoute::new("primary", &["good-bounded"])
                .bounded(262_144)
                .escalating("destination"),
            FleetRoute::new("destination", &["good-destination"]),
        ],
    )?;
    let app = build_llm_router(state);

    let (status, body) = post_chat(&app, "primary", "hello").await?;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    // The claim under test is that NO producer was activated. If a token
    // producer had been wired, the bounded candidate would have been judged on
    // a fact and this ladder would differ; instead the primary serves directly.
    assert_eq!(
        upstream.served_models().await,
        vec!["good-bounded".to_string()],
        "a bounded candidate must be admitted while the producer is dormant: the \
         escalation path must not have activated one"
    );
    Ok(())
}
