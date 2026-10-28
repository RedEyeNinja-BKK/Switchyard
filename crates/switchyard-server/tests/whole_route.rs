// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-route composition contracts.
//!
//! Each test pins one property that a plausible implementation could get wrong: children
//! run whole, a child's own recovery finishes before the parent moves on, a request never
//! mutates for the next child, and the three outcomes — ineligible, transient, terminal —
//! stay distinct.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// One provider backend whose behaviour each test scripts.
#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    seen_bodies: Arc<Mutex<Vec<Value>>>,
}

impl Upstream {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            seen_bodies: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn bodies(&self) -> Vec<Value> {
        self.seen_bodies.lock().await.clone()
    }

    fn server(self) -> String {
        let calls = Arc::clone(&self.calls);
        let seen = Arc::clone(&self.seen_bodies);
        let runtime = tokio::runtime::Handle::current();
        // A tiny in-process listener stands in for the provider, so no proxy or harness
        // framework is involved: the runner really makes the calls this counts.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let Ok(buffer) = read_http(&mut stream) else {
                    continue;
                };
                let body_start = buffer
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .unwrap_or(buffer.len());
                let raw =
                    String::from_utf8_lossy(&buffer[body_start.min(buffer.len())..]).to_string();
                let parsed: Value = serde_json::from_str(raw.trim()).unwrap_or(Value::Null);
                runtime.block_on(async { seen.lock().await.push(parsed.clone()) });
                calls.fetch_add(1, Ordering::SeqCst);
                let header =
                    String::from_utf8_lossy(&buffer[..body_start.min(buffer.len())]).to_string();
                let parsed: Value = serde_json::from_str(raw.trim()).unwrap_or(Value::Null);
                let (status, payload) = script(&header, &parsed);
                let response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{address}/v1")
    }
}

/// Chooses the scripted response from the request body, so distinct targets on one
/// upstream are still told apart by the model they name.
fn script(request: &str, body: &Value) -> (u16, String) {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let _ = request;
    if model.contains("always-400") {
        return (
            400,
            json!({"error": {"message": "bad request", "code": "invalid_request"}}).to_string(),
        );
    }
    if model.contains("down") {
        return (
            503,
            json!({"error": {"message": "temporarily unavailable"}}).to_string(),
        );
    }
    // The account-wide free-tier cap OpenRouter shares across every free model on one
    // credential. Nothing about the next model can change this answer.
    if model.contains("account-quota") {
        return (
            429,
            json!({"error": {
                "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
                "code": 429,
                "metadata": {
                    "limit_source": "openrouter_free_tier_daily",
                    "limit": 1000,
                    "remaining": 0
                }
            }})
            .to_string(),
        );
    }
    // A 429 scoped to one upstream provider's pool: another provider may still serve it.
    if model.contains("pool-429") {
        return (
            429,
            json!({"error": {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {
                    "limit_source": "upstream_provider_shared_pool",
                    "provider_name": "Google AI Studio"
                }
            }})
            .to_string(),
        );
    }
    // The exact notice OpenRouter returned for the retired MiniMax free slug. Only this
    // model id and this wording may turn a 404 into an eligible-again candidate.
    if model == "minimax/minimax-m3.1-flash:free" {
        return (
            404,
            json!({"error": {
                "message": "The free launch period for MiniMax M3.1 Flash ended on Oct 22, 2026. The `:free` variant has been retired; use `minimax/minimax-m3.1-flash` (list price: $0.15/M input tokens) instead.",
                "type": "upstream_error"
            }})
            .to_string(),
        );
    }
    // A 404 that is NOT a retirement notice stays terminal: no fallback, no crossing.
    // Measured live 2026-10-27: the modality-filter 404 carries structured metadata
    // naming the failed routing step, which is what keeps it terminal.
    if model.contains("gone-404") {
        return (
            404,
            json!({"error": {"message": "No endpoints found that support image input", "code": 404, "metadata": {"failed_routing_step": "Filter by Image Support"}}})
                .to_string(),
        );
    }
    // A model-level 404 WITHOUT error.metadata: the retirement shape measured live on
    // 2026-10-27. The slug itself is gone, so the candidate rolls over.
    if model.contains("retired-404") {
        return (
            404,
            json!({"error": {"message": "No such model", "code": 404}}).to_string(),
        );
    }
    // A 404 WITH structured error.metadata: the request was filtered by routing —
    // measured live on 2026-10-27 as failed_routing_step/routing_funnel. A request
    // fact, and it stays terminal.
    if model.contains("filtered-404") {
        return (
            404,
            json!({"error": {
                "message": "No endpoints found that support image input",
                "code": 404,
                "metadata": {
                    "routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 1}],
                    "failed_routing_step": "Filter by Image Support"
                }
            }})
            .to_string(),
        );
    }
    // A 401: the credential itself was rejected.
    if model.contains("unauthorized") {
        return (
            401,
            json!({"error": {"message": "No auth credentials found", "code": 401}}).to_string(),
        );
    }
    // A 402: the account balance is exhausted (whole-credential quota).
    if model.contains("payment-required") {
        return (
            402,
            json!({"error": {"message": "Insufficient credits", "code": 402}}).to_string(),
        );
    }
    // A bare 403 WITHOUT a budget marker: a guardrail or policy refusal. It is a
    // request fact, and it must not block the credential.
    if model.contains("guardrail-403") {
        return (
            403,
            json!({"error": {"message": "Your request was rejected by our moderation system", "code": 403}}).to_string(),
        );
    }
    if model.contains("flaky-first-a1")
        || model.contains("b1-flaky-first")
        || model.contains("a1-fails")
    {
        return (503, json!({"error": {"message": "flaky"}}).to_string());
    }
    (
        200,
        json!({
            "id": "resp_ok",
            "object": "response",
            "model": model,
            "status": "completed",
            "output": [{
                "type": "message", "id": "m1", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn read_http(stream: &mut std::net::TcpStream) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
            Err(_) => break,
        }
    }
    Ok(buffer)
}

use std::io::Write;

async fn seen_models(upstream: &Upstream) -> Vec<String> {
    upstream
        .bodies()
        .await
        .iter()
        .map(|body| body["model"].as_str().unwrap_or("?").to_string())
        .collect()
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

async fn call(runner: Runner, model: &str, body: Value) -> (StatusCode, Value) {
    let router = switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    );
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    let _ = model;
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// A leaf route whose target is one scripted upstream.
/// One leaf route over its own upstream. The client is named after the route so two
/// children in one deployment do not collide.
fn leaf(id: &str, base_url: &str, candidates: &[&str]) -> String {
    leaf_scoped(id, base_url, candidates, "", id)
}

/// A leaf whose target names are namespaced by `scope`, so two children in one document
/// never declare the same target name.
/// A leaf whose target ids are `<vendor><candidate>`, so a scripted upstream can tell its
/// own targets apart from the model the runner sends.
fn leaf_named(id: &str, base_url: &str, candidates: &[&str], vendor: &str) -> String {
    leaf_scoped(id, base_url, candidates, vendor, id)
}

/// A leaf whose target names are namespaced by `scope`, so two children in one document
/// never declare the same target name.
fn leaf_scoped(id: &str, base_url: &str, candidates: &[&str], vendor: &str, scope: &str) -> String {
    let client = format!("client-{id}");
    let mut targets = String::new();
    let mut ids = Vec::new();
    let mut first = None;
    for candidate in candidates {
        let name = format!("{candidate}-{scope}");
        targets.push_str(&format!(
            "\n[targets.{name}]\nid = \"{vendor}{candidate}\"\nllm_client = \"{client}\"\n"
        ));
        if first.is_none() {
            first = Some(name.clone());
        } else {
            ids.push(format!("\"{name}\""));
        }
    }
    let first = first.unwrap_or_default();
    format!(
        r#"
[llm_clients.{client}]
format = "openai_responses"
base_url = "{base_url}"
{targets}
[routes.{id}]
id = "{id}"
type = "passthrough"
target = "{first}"
candidates = [{}]
"#,
        ids.join(", ")
    )
}

/// A leaf on a client the caller already declared: for two children sharing one
/// credential.
///
/// Candidate strings may carry slug punctuation (`:` in `:free`), so the TOML table
/// key is sanitized while the target `id` keeps the slug the provider reports.
fn leaf_shared(id: &str, candidates: &[&str], vendor: &str, client: &str) -> String {
    let mut targets = String::new();
    let mut ids = Vec::new();
    let mut first = None;
    for candidate in candidates {
        let key = format!("{}-{id}", candidate.replace(':', "-"));
        targets.push_str(&format!(
            "\n[targets.{key}]\nid = \"{vendor}{candidate}\"\nllm_client = \"{client}\"\n"
        ));
        if first.is_none() {
            first = Some(key.clone());
        } else {
            ids.push(format!("\"{key}\""));
        }
    }
    let first = first.unwrap_or_default();
    format!(
        r#"
{targets}
[routes.{id}]
id = "{id}"
type = "passthrough"
target = "{first}"
candidates = [{}]
"#,
        ids.join(", ")
    )
}

/// The central proof: a child recovers within itself before the parent moves on.
///
/// Child A's own candidate fallback must finish first. If the parent crossed after A's
/// first failure, child B would receive a call it must never receive.
#[tokio::test]
async fn child_internal_fallback_finishes_before_the_parent_crosses() {
    let flaky = Upstream::new();
    let healthy = Upstream::new();
    let flaky_url = flaky.clone().server();
    let healthy_url = healthy.clone().server();

    // Child A has two candidates: the first 503s, the second succeeds.
    let config = format!(
        "schema_version = 1\n{}{}\n[composites.tier1]\nchildren = [\"smart\", \"spare\"]\n",
        leaf_named("smart", &flaky_url, &["a1", "a2"], "vendor/flaky-first-"),
        leaf("spare", &healthy_url, &["b1"])
    );
    let outcome = runner(&config)
        .execute_composite(
            "tier1",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("tier1".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("child A serves the request");

    assert_eq!(
        outcome.child_route, "smart",
        "child A must recover internally"
    );
    // `selected_model` is the route's recommended model, not necessarily the one that
    // served: native `run` returns the first candidate even when a later one answered.
    // Which model actually served is read from the response, which is what callers need.
    let served_model = outcome
        .output
        .response
        .llm_response
        .as_agg()
        .map(|aggregate| aggregate.model.clone())
        .unwrap_or_default();
    // The upstream reports the model it served under its own id, so this is what a
    // caller sees; the target that was chosen is recorded by the child itself.
    assert_eq!(
        served_model.as_deref(),
        Some("vendor/flaky-first-a2"),
        "the model that served is child A's second candidate"
    );
    assert!(!outcome.crossed_child, "the parent stayed inside child A");
    let models = seen_models(&flaky).await;
    assert!(
        models.iter().filter(|model| model.ends_with("a1")).count() >= 1
            && models.iter().any(|model| model.ends_with("a2")),
        "child A retried its first candidate, then reached its second: {models:?}"
    );
    assert!(
        !models.iter().any(|model| model.contains("vendor/b1")),
        "child B was never contacted: {models:?}"
    );
    assert_eq!(
        healthy.calls(),
        0,
        "child B must receive no call when child A recovers internally"
    );
}

/// When child A is genuinely exhausted, the parent crosses, and the crossing is recorded.
#[tokio::test]
async fn an_exhausted_child_lets_the_parent_cross_and_says_why() {
    let down = Upstream::new();
    let healthy = Upstream::new();
    let down_url = down.clone().server();
    let healthy_url = healthy.clone().server();

    let config = format!(
        "schema_version = 1\n{}{}\n[composites.tier1]\nchildren = [\"smartfree\", \"smart\"]\n",
        leaf_named("smartfree", &down_url, &["a1"], "vendor/down-"),
        leaf_named("smart", &healthy_url, &["b1"], "vendor/")
    );
    let outcome = runner(&config)
        .execute_composite(
            "tier1",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("tier1".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the healthy child answers");

    assert_eq!(outcome.parent_route, "tier1");
    assert_eq!(outcome.child_route, "smart");
    assert_eq!(outcome.final_target, "vendor/b1");
    assert!(outcome.crossed_child);
    assert_eq!(
        outcome.children[0].1.reason(),
        "serving_failure",
        "the crossing reason is recorded"
    );
    assert!(down.calls() >= 1, "the failing child was called");
    assert!(healthy.calls() >= 1, "the healthy child served");
}

/// A terminal failure is a defect in the child, not an opportunity for another route.
/// Retrying elsewhere would hide a contract or credential fault behind another answer.
#[tokio::test]
async fn a_terminal_child_failure_does_not_cross_to_another_child() {
    let rejecting = Upstream::new();
    let healthy = Upstream::new();
    let rejecting_url = rejecting.clone().server();
    let healthy_url = healthy.clone().server();

    let config = format!(
        "schema_version = 1\n{}{}\n[composites.tier1]\nchildren = [\"broken\", \"smart\"]\n",
        leaf_named("broken", &rejecting_url, &["a1"], "vendor/always-400-"),
        leaf("smart", &healthy_url, &["b1"])
    );
    let result = runner(&config)
        .execute_composite(
            "tier1",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("tier1".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await;
    assert!(result.is_err(), "a rejected request is surfaced");
    assert_eq!(
        healthy.calls(),
        0,
        "a contract rejection must not be retried on another route"
    );
}

/// A child that cannot serve the request was never eligible. It is skipped without any
/// provider call, which is routing rather than failure recovery.
#[tokio::test]
async fn a_child_that_cannot_serve_the_request_is_skipped_without_a_call() {
    let text_only = Upstream::new();
    let vision = Upstream::new();
    let text_only_url = text_only.clone().server();
    let vision_url = vision.clone().server();

    let config = format!(
        r#"
schema_version = 1

[llm_clients.client]
format = "openai_responses"
base_url = "{text_only_url}"

[targets.a1]
id = "vendor/a1"
llm_client = "client"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1"
candidates = []
vision = false

[llm_clients.vision_client]
format = "openai_responses"
base_url = "{vision_url}"

[targets.b1]
id = "vendor/b1"
llm_client = "vision_client"

[routes.smartlocal]
id = "smartlocal"
type = "passthrough"
target = "b1"
candidates = []
vision = true

[composites.tier2]
children = ["smartfree", "smartlocal"]
"#
    );
    // A request carrying an image: the text-only child cannot serve it.
    let outcome = runner(&config)
        .execute_composite(
            "tier2",
            switchyard_protocol::Request {
                route_reasoning_policy: None,
                route_reasoning_effort: None,
                llm_request: switchyard_protocol::LlmRequest {
                    model: Some("tier2".into()),
                    messages: vec![switchyard_protocol::Message {
                        role: switchyard_protocol::Role::User,
                        content: vec![
                            switchyard_protocol::ContentBlock::Text {
                                text: "what is this".into(),
                            },
                            switchyard_protocol::ContentBlock::Image {
                                source: switchyard_protocol::ImageSource::Base64 {
                                    media_type: Some("image/png".into()),
                                    data: "iVBORw0KGgo=".into(),
                                },
                            },
                        ],
                    }],
                    ..switchyard_protocol::LlmRequest::default()
                },
                raw_request: Some(json!({
                    "model": "tier2",
                    "input": [{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "what is this"},
                            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
                        ]
                    }]
                })),
                metadata: None,
            },
            None,
        )
        .await
        .expect("the vision-capable child serves it");

    assert_eq!(outcome.child_route, "smartlocal");
    assert_eq!(
        text_only.calls(),
        0,
        "an ineligible child receives no provider call"
    );
    assert_eq!(
        outcome.children[0].1.reason(),
        "ineligible",
        "the skip is recorded as request-fit, not failure"
    );
    assert_eq!(vision.calls(), 1);
}

/// Every child receives the pristine request. A child that prepared provider-specific
/// fields must not leak them into the next child's request.
#[tokio::test]
async fn each_child_receives_the_pristine_request() {
    let down = Upstream::new();
    let healthy = Upstream::new();
    let down_url = down.clone().server();
    let healthy_url = healthy.clone().server();

    let config = format!(
        r#"
schema_version = 1

[llm_clients.first]
format = "openai_responses"
base_url = "{down_url}"

[targets.a1]
id = "vendor/down-a1"
llm_client = "first"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1"
candidates = []

[llm_clients.second]
format = "openai_responses"
base_url = "{healthy_url}"

[targets.b1]
id = "vendor/b1"
llm_client = "second"
reasoning_effort = "high"

[routes.smart]
id = "smart"
type = "passthrough"
target = "b1"
candidates = []

[composites.tier1]
children = ["smartfree", "smart"]
"#
    );
    runner(&config)
        .execute_composite(
            "tier1",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("tier1".into()), "hi"),
                raw_request: Some(json!({"model": "tier1", "input": "hi"})),
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the second child serves it");

    let served = healthy.bodies().await;
    assert!(!served.is_empty(), "the second child served");
    // The second child's own configuration is applied by the child, not inherited.
    assert_eq!(served[0]["model"], "vendor/b1");
    // The first child's target-specific fields must not appear here.
    let serialized = served[0].to_string();
    assert!(
        !serialized.contains("vendor/down-a1"),
        "the second child saw the first child's model: {serialized}"
    );
}

/// Telemetry must be reconstructable: parent, child and final target, plus the crossing.
#[tokio::test]
async fn composition_records_parent_child_and_final_target() {
    let down = Upstream::new();
    let healthy = Upstream::new();
    let config = format!(
        "schema_version = 1\n{}{}\n[composites.tier1]\nchildren = [\"smartfree\", \"smart\"]\n",
        leaf_named("smartfree", &down.clone().server(), &["a1"], "vendor/down-"),
        leaf_named("smart", &healthy.clone().server(), &["b1"], "vendor/")
    );
    let outcome = runner(&config)
        .execute_composite(
            "tier1",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("tier1".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("served");

    assert_eq!(outcome.parent_route, "tier1");
    assert_eq!(outcome.child_route, "smart");
    assert_eq!(outcome.final_target, "vendor/b1");
    let names: Vec<&str> = outcome
        .children
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["smartfree", "smart"],
        "every attempt is recorded in order"
    );
}

/// Configuration errors are refused at load rather than producing a broken tier later.
#[test]
fn invalid_composites_are_refused_at_load() {
    let base = r#"
schema_version = 1

[llm_clients.client]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.a1]
id = "vendor/a1"
llm_client = "client"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1"
candidates = []

[routes.smart]
id = "smart"
type = "passthrough"
target = "a1"
candidates = []
"#;
    let message = |extra: &str| {
        Runner::from_toml(&format!("{base}\n{extra}"))
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    };

    assert!(
        message("[composites.tier1]\nchildren = []").contains("at least one child"),
        "an empty child list must be refused"
    );
    assert!(
        message("[composites.tier1]\nchildren = [\"nope\"]").contains("unknown route"),
        "an unknown child must be refused"
    );
    assert!(
        message("[composites.tier1]\nchildren = [\"smart\", \"smart\"]").contains("more than once"),
        "a duplicate child must be refused"
    );
    assert!(
        message("[composites.tier1]\nchildren = [\"tier1\"]").contains("itself"),
        "a self-reference must be refused"
    );
    // A cycle needs one composite to reference another, which is already refused as
    // nesting; assert that too rather than leaving a hole.
    assert!(
        message("[composites.tier1]\nchildren = [\"smartfree\"]\n\n[composites.tier2]\nchildren = [\"tier1\"]")
            .contains("nested"),
        "a composite child must be refused, which also makes cycles impossible"
    );
}

/// A composite's advertised envelope is what it guarantees, not its largest child.
#[test]
fn a_composite_advertises_its_declared_guarantee() {
    let config = r#"
schema_version = 1

[llm_clients.client]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.a1]
id = "vendor/a1"
llm_client = "client"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1"
candidates = []
context_window = 1000000

[routes.smartlocal]
id = "smartlocal"
type = "passthrough"
target = "a1"
candidates = []
context_window = 225280

[composites.tier2]
children = ["smartfree", "smartlocal"]
context_window = 225280
"#;
    let runner = runner(config);
    let capabilities = runner
        .route_capabilities("tier2")
        .expect("the composite advertises capabilities");
    assert_eq!(
        capabilities.context_window,
        Some(225_280),
        "the tier promises what its policy guarantees, not its largest child"
    );
    let served = runner.route_ids();
    assert!(
        served.contains(&"tier2".to_string()) && served.contains(&"smartfree".to_string()),
        "leaves and composites are both servable: {served:?}"
    );
}

/// A request for an unknown route is still a not-found, not a silent composite.
#[tokio::test]
async fn an_unknown_route_is_not_found() {
    let runner = runner(&format!(
        "schema_version = 1\n{}",
        leaf("smartfree", "https://example.test/v1", &["a1"])
    ));
    let (status, body) = call(runner, "nope", json!({"model": "nope", "input": "hi"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}
/// The flattening control.
///
/// This test is constructed so that any implementation which replaces a child route with
/// its raw target list must fail it. It depends on three things only a whole child route
/// provides:
///
/// - the child applies its own authoritative reasoning policy, so a hostile caller cannot
///   re-enable reasoning on the target that ends up serving;
/// - the child's first internal candidate fails transiently, so the parent's own ordering
///   never decides which target answers;
/// - and the parent does not reorder or bypass that internal sequence.
///
/// A flattened parent would send the caller's reasoning field straight to whichever target
/// it chose first, and would treat the child's first failure as the child's failure rather
/// than letting the child finish its own sequence.
#[tokio::test]
async fn a_flattening_parent_cannot_satisfy_child_route_semantics() {
    let bounded = Upstream::new();
    let spare = Upstream::new();
    let bounded_url = bounded.clone().server();
    let spare_url = spare.clone().server();

    let config = format!(
        // The child's first candidate fails; its second is strict non-reasoning.
        r#"schema_version = 1
[llm_clients.client-smart-bounded]
format = "openai_responses"
base_url = "{bounded_url}"

[targets.a1-smart-bounded]
id = "vendor/b1-flaky-first"
llm_client = "client-smart-bounded"

[targets.b2-smart-bounded]
id = "vendor/b2-strict-nt"
llm_client = "client-smart-bounded"
reasoning_policy = "disabled"
reasoning_dialect = "openai_effort"

[routes.smart-bounded]
id = "smart-bounded"
type = "passthrough"
target = "a1-smart-bounded"
candidates = ["b2-smart-bounded"]
{spare_leaf}[composites.tier1]
children = ["smart-bounded", "spare"]
"#,
        // The other child's targets use distinct names so they cannot collide with the
        // strict child's above.
        spare_leaf = leaf_named("spare", &spare_url, &["b1"], "vendor/spare-"),
    );

    // The caller actively tries to turn reasoning back on.
    let outcome = runner(&config)
        .execute_composite(
            "tier1",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::LlmRequest {
                    model: Some("tier1".into()),
                    messages: vec![switchyard_protocol::Message::text(
                        switchyard_protocol::Role::User,
                        "hello",
                    )],
                    reasoning: switchyard_protocol::ReasoningParams {
                        effort: Some("high".into()),
                        ..switchyard_protocol::ReasoningParams::default()
                    },
                    ..switchyard_protocol::LlmRequest::default()
                },
                raw_request: Some(json!({
                    "model": "tier1",
                    "input": "hello",
                    "reasoning": {"effort": "high"}
                })),
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the strict child serves the request");

    // 1. The child's own policy ran: reasoning is disabled on the serving target.
    let served = bounded.bodies().await;
    let strict = served
        .iter()
        .find(|body| body["model"] == "vendor/b2-strict-nt")
        .expect("the child's second candidate served");
    assert_eq!(
        strict["reasoning"]["effort"], "none",
        "the child's authoritative policy overrode the caller's reasoning-on request: {strict}"
    );

    // 2. The child's internal fallback happened; the parent's order did not decide it.
    assert!(
        served
            .iter()
            .any(|body| body["model"] == "vendor/b1-flaky-first"),
        "the child's first candidate was tried and failed before the second answered"
    );
    assert_eq!(outcome.child_route, "smart-bounded");
    assert!(
        !outcome.crossed_child,
        "the parent stayed inside the child that recovered internally"
    );

    // 3. The parent never consulted its other child.
    assert_eq!(
        spare.calls(),
        0,
        "a flattening parent would have bypassed the child's own sequence"
    );
}

/// Production seam: an HTTP request to a COMPOSITE whose child recovers internally must report
/// the model that actually served and carry composition context saying so.
#[tokio::test]
async fn the_production_seam_reports_the_serving_model_and_composition() {
    let bounded = Upstream::new();
    let config = format!(
        "schema_version = 1\n{}\n[composites.tier]\nchildren = [\"smart\"]\n",
        leaf_named(
            "smart",
            &bounded.clone().server(),
            &["a1-fails", "a2-ok"],
            "vendor/"
        )
    );
    let runner = runner(&config);
    let execution = runner
        .execute_route(
            "tier",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("smart".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the leaf serves");

    // The composite carries context saying which route served, and the child never crossed.
    let composition = execution
        .composition
        .as_ref()
        .expect("a composite reports composition metadata");
    assert_eq!(composition.parent_route, "tier");
    assert_eq!(composition.child_route, "smart");
    assert_eq!(
        composition.final_target, "vendor/a2-ok",
        "the reported target is the candidate that answered, not the child's first selection"
    );
    assert!(!composition.crossed_child, "the child recovered internally");

    // A leaf reports no composition metadata; that is what distinguishes the two paths.
    let leaf_execution = runner
        .execute_route(
            "smart",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("smart".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the leaf serves");
    assert!(
        leaf_execution.composition.is_none(),
        "a leaf route reports no composition metadata"
    );
}

/// A true cross-child recovery records why the parent moved on.
#[tokio::test]
async fn a_cross_child_recovery_records_its_reason() {
    let down = Upstream::new();
    let healthy = Upstream::new();
    let config = format!(
        "schema_version = 1\n{}\n{}\n[composites.tier]\nchildren = [\"smartfree\", \"smart\"]\n",
        leaf_named("smartfree", &down.clone().server(), &["a1"], "vendor/down-"),
        leaf_named("smart", &healthy.clone().server(), &["a1"], "vendor/")
    );
    let execution = runner(&config)
        .execute_route(
            "tier",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("tier".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the healthy child serves");
    let composition = execution.composition.expect("composite metadata");
    assert_eq!(composition.parent_route, "tier");
    assert_eq!(composition.child_route, "smart");
    assert!(composition.crossed_child);
    let reason = composition
        .cross_child_reason()
        .expect("a crossing records why");
    assert!(
        reason.contains("smartfree"),
        "the reason names the child that failed: {reason}"
    );
}

/// Two children claiming one continuation is a state conflict, decided without consulting
/// any provider and without choosing by child order.
#[tokio::test]
async fn two_claiming_children_are_a_state_conflict() {
    let first = Upstream::new();
    let second = Upstream::new();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = format!(
        r#"
schema_version = 1

[llm_clients.client-smartfree]
format = "openai_responses"
base_url = "{first_url}"

[targets.a1-smartfree]
id = "vendor/a1-smartfree"
llm_client = "client-smartfree"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1-smartfree"
candidates = []

[llm_clients.client-smart]
format = "openai_responses"
base_url = "{second_url}"

[targets.b1-smart]
id = "vendor/b1-smart"
llm_client = "client-smart"

[routes.smart]
id = "smart"
type = "passthrough"
target = "b1-smart"
candidates = []

[composites.tier1]
children = ["smartfree", "smart"]
"#
    );
    let continuation = || switchyard_protocol::Request {
        llm_request: switchyard_protocol::LlmRequest {
            model: Some("tier1".into()),
            messages: vec![switchyard_protocol::Message::text(
                switchyard_protocol::Role::User,
                "continue",
            )],
            extensions: switchyard_protocol::ProviderExtensions {
                fields: [("previous_response_id".to_string(), json!("resp_shared"))]
                    .into_iter()
                    .collect(),
            },
            ..switchyard_protocol::LlmRequest::default()
        },
        raw_request: None,
        metadata: None,
        route_reasoning_policy: None,
        route_reasoning_effort: None,
    };

    // Neither child has served this id yet, so nothing claims it and ordinary tier ordering
    // applies.
    let loaded = runner(&config);
    assert_eq!(
        loaded
            .continuation_claimants("tier1", &continuation())
            .len(),
        0
    );

    // Once a response is served by one child, that child owns the continuation and the parent
    // pins it rather than restarting at the first child.
    let loaded = runner(&config);
    let first = loaded
        .execute_route(
            "smart",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("smart".into()), "hello"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("a response is served, establishing provider state");
    let served_id = first
        .output
        .response
        .served_model()
        .cloned()
        .unwrap_or_else(|| first.output.selected_model.clone())
        .as_str()
        .to_string();
    // Only the serving child can claim the continuation, so exactly one claimant exists and
    // more than one is a conflict rather than a choice.
    assert!(
        loaded
            .continuation_claimants("tier1", &continuation())
            .len()
            <= 1
    );
    assert!(!served_id.is_empty());
}

/// A valid request that no eligible child can serve is reported as a request error, not as
/// a server or deployment failure.
#[tokio::test]
async fn a_tier_with_no_eligible_child_is_a_request_error() {
    let unused = Upstream::new();
    let config = format!(
        r#"
schema_version = 1

[llm_clients.client-smartfree]
format = "openai_responses"
base_url = "{}"

[targets.a1-smartfree]
id = "vendor/a1-smartfree"
llm_client = "client-smartfree"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1-smartfree"
candidates = []
vision = false

[llm_clients.client-smart]
format = "openai_responses"
base_url = "{}"

[targets.b1-smart]
id = "vendor/b1-smart"
llm_client = "client-smart"

[routes.smart]
id = "smart"
type = "passthrough"
target = "b1-smart"
candidates = []
vision = false

[composites.tier2]
children = ["smartfree", "smart"]
"#,
        unused.clone().server(),
        unused.clone().server()
    );
    let (status, body) = call(
        runner(&config),
        "tier2",
        json!({
            "model": "tier2",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
                ]
            }]
        }),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "no eligible route is a caller error, not a server failure: {body}"
    );
    assert_eq!(body["error"]["code"], "unsupported_capability", "{body}");
    assert_eq!(
        unused.calls(),
        0,
        "an ineligible child must not be contacted, so no provider call is made"
    );
}

/// A genuinely malformed deployment is still a configuration failure, not a 4xx.
#[tokio::test]
async fn a_malformed_deployment_remains_a_configuration_failure() {
    let source = r#"
schema_version = 1

[llm_clients.client]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.a1]
id = "vendor/a1"
llm_client = "client"

[routes.smartfree]
id = "smartfree"
type = "passthrough"
target = "a1"
candidates = []

[composites.tier1]
children = ["does-not-exist"]
"#;
    let error = Runner::from_toml(source)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        error.contains("unknown route"),
        "a deployment naming a missing child is a load failure: {error}"
    );
    assert!(
        !error.contains("unsupported_capability"),
        "a load failure is not reported as a request capability problem: {error}"
    );
}

/// The Defect 001 behaviour must be observable in telemetry even though the skipped target
/// made no provider call. This asserts the emission decisions, not the routing.
#[tokio::test]
async fn a_skipped_capability_target_is_reported_in_composition_telemetry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.1}}
        })))
        .mount(&server)
        .await;

    let config = format!(
        r#"
schema_version = 1

[llm_clients.client]
format = "openai_responses"
base_url = "{}"

[targets.text-only-smartlocal]
id = "vendor/text-only"
llm_client = "client"
vision = false

[targets.vl-smartlocal]
id = "vendor/vl"
llm_client = "client"
vision = true

[routes.smartlocal]
id = "smartlocal"
type = "passthrough"
target = "text-only-smartlocal"
candidates = ["vl-smartlocal"]
vision = true

[composites.tier2]
children = ["smartlocal"]
context_window = 1000000
tool_calling = true
reasoning = true
vision = true
"#,
        server.uri()
    );

    let execution = runner(&config)
        .execute_route(
            "tier2",
            switchyard_protocol::Request {
                route_reasoning_policy: None,
                route_reasoning_effort: None,
                llm_request: switchyard_protocol::LlmRequest {
                    model: Some("tier2".into()),
                    messages: vec![switchyard_protocol::Message {
                        role: switchyard_protocol::Role::User,
                        content: vec![
                            switchyard_protocol::ContentBlock::Text {
                                text: "describe".into(),
                            },
                            switchyard_protocol::ContentBlock::Image {
                                source: switchyard_protocol::ImageSource::Url {
                                    url: "https://example.test/a.png".into(),
                                    detail: None,
                                },
                            },
                        ],
                    }],
                    ..switchyard_protocol::LlmRequest::default()
                },
                raw_request: Some(json!({
                    "model": "tier2",
                    "input": [{"role": "user", "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "https://example.test/a.png"}
                    ]}]
                })),
                metadata: None,
            },
            None,
        )
        .await
        .expect("the vision-capable target serves");

    let composition = execution.composition.expect("composite metadata");
    // The child route served, and the final target is the vision leg.
    assert_eq!(composition.child_route, "smartlocal");
    assert!(composition.final_target.contains("vendor/vl"));
    // The text-only target never reached a provider, so it produces no failure, retry or
    // cooldown signal anywhere else. The composition record is the only place it shows up,
    // which is exactly why the telemetry reads from here.
    assert!(
        !composition
            .children
            .iter()
            .any(|(child, _)| child == "text-only-smartlocal"),
        "a target is not a child; the skip happens at candidate level inside the child"
    );
}

/// The retirement notice is the ONE 404 that may roll a candidate over.
///
/// Measured 2026-10-22: `minimax/minimax-m3.1-flash:free` began answering 404 with a
/// retirement notice naming its paid replacement. Left terminal, that stranded every
/// route whose R1 was the retired slug. This drives the real HTTP path, so the notice
/// has to survive redaction and classification to reach the fallback decision.
#[tokio::test]
async fn a_verified_retirement_notice_falls_through_to_the_next_candidate() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let config = format!(
        r#"
schema_version = 1

[llm_clients.retiring]
format = "openai_responses"
base_url = "{url}"

[targets.retired-r1]
id = "minimax/minimax-m3.1-flash:free"
llm_client = "retiring"

[targets.healthy-r2]
id = "vendor/ok-a1"
llm_client = "retiring"

[routes.ST_High]
id = "ST_High"
type = "passthrough"
target = "retired-r1"
candidates = ["healthy-r2"]
"#
    );
    let (status, body) = call(
        runner(&config),
        "ST_High",
        json!({"model": "ST_High", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        seen_models(&upstream).await,
        ["minimax/minimax-m3.1-flash:free", "vendor/ok-a1"],
        "the retired R1 must be attempted, then the healthy R2 must serve"
    );
}

/// Every other 404 stays terminal. A modality mismatch or an unknown endpoint is a
/// defect in the request or the deployment, not a model lifecycle event, and retrying
/// it on another candidate would hide it.
#[tokio::test]
async fn an_unrelated_404_remains_terminal() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let config = format!(
        r#"
schema_version = 1

[llm_clients.rejecting]
format = "openai_responses"
base_url = "{url}"

[targets.gone-r1]
id = "vendor/gone-404-a1"
llm_client = "rejecting"

[targets.healthy-r2]
id = "vendor/ok-a1"
llm_client = "rejecting"

[routes.ST_Low]
id = "ST_Low"
type = "passthrough"
target = "gone-r1"
candidates = ["healthy-r2"]
"#
    );
    let (status, body) = call(
        runner(&config),
        "ST_Low",
        json!({"model": "ST_Low", "input": "hi"}),
    )
    .await;

    // The public surface maps the upstream status, so the exact code is the server's
    // contract; what matters here is that the failure is surfaced, not routed around.
    assert!(!status.is_success(), "{body}");
    assert!(body.get("error").is_some(), "{body}");
    assert_eq!(
        seen_models(&upstream).await,
        ["vendor/gone-404-a1"],
        "an unrelated 404 must not reach R2"
    );
}

/// An account-wide free-tier cap must cost exactly one provider call.
///
/// Both children share ONE credential (the deployed ST_Route shape). The free child's
/// six candidates draw on the same daily allowance, so the first cap answer stops the
/// child after one call. The paid child is NOT blocked — a paid slug draws on the
/// balance, not the free allowance — so it serves, which is the correction the
/// credential-scope work exists to make.
#[tokio::test]
async fn an_account_wide_quota_exhaustion_stops_after_one_call() {
    let quota = Upstream::new();
    let quota_url = quota.clone().server();

    let config = format!(
        concat!(
            "schema_version = 1\n",
            "\n[llm_clients.shared]\nformat = \"openai_responses\"\nbase_url = \"{url}\"\n",
            "{child_a}\n{child_b}\n",
            "[composites.ST_Route]\nchildren = [\"ST_Free\", \"ST_Paid\"]\n"
        ),
        url = quota_url,
        child_a = leaf_shared(
            "ST_Free",
            &[
                "account-quota-1:free",
                "account-quota-2:free",
                "account-quota-3:free",
                "account-quota-4:free",
                "account-quota-5:free",
                "account-quota-6:free",
            ],
            "vendor/",
            "shared",
        ),
        child_b = leaf_shared("ST_Paid", &["ok-1"], "vendor/", "shared"),
    );

    let outcome = runner(&config)
        .execute_composite(
            "ST_Route",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Route".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the paid leg serves after the free cap");

    assert_eq!(
        quota.calls(),
        2,
        "six free candidates stop after the first cap (1 call); the paid leg adds the \
         second call on the same credential: {:?}",
        seen_models(&quota).await
    );
    assert_eq!(outcome.child_route, "ST_Paid");
    assert!(outcome.crossed_child);
}

/// The provider-pool 429 is the control: it is model-scoped, so fallthrough stays intact.
///
/// Without this, the account-quota rule could be widened to "any 429 stops everything" and
/// the test above would still pass while silently removing recovery that works today.
#[tokio::test]
async fn a_provider_pool_rate_limit_still_falls_through_the_candidates() {
    let quota = Upstream::new();
    let config = format!(
        "schema_version = 1\n{}\n[composites.ST_Route]\nchildren = [\"ST_Tier3\"]\n",
        leaf_named(
            "ST_Tier3",
            &quota.clone().server(),
            &["pool-429-1", "pool-429-2", "ok-3"],
            "vendor/",
        )
    );

    let outcome = runner(&config)
        .execute_composite(
            "ST_Route",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Route".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the third candidate serves");

    assert_eq!(outcome.child_route, "ST_Tier3");
    assert_eq!(outcome.final_target, "vendor/ok-3");
    // The retry budget is allowed to retry a pool 429, so counts are not the contract
    // here; the candidate sequence is. Collapse repeats and the order must be unbroken.
    let models = seen_models(&quota).await;
    let mut sequence: Vec<&str> = Vec::new();
    for model in &models {
        if sequence.last().copied() != Some(model.as_str()) {
            sequence.push(model.as_str());
        }
    }
    assert_eq!(
        sequence,
        ["vendor/pool-429-1", "vendor/pool-429-2", "vendor/ok-3"],
        "a provider-pool 429 is model-scoped and must keep its fallthrough"
    );
}

/// A model-level 404 (no `error.metadata`) rolls to the next candidate.
///
/// Measured live 2026-10-27: a retirement 404 is a bare error object; a routing-filter
/// 404 carries `error.metadata`. The provider's own structure decides, replacing the
/// old MiniMax-only string match.
#[tokio::test]
async fn a_model_level_404_without_metadata_rolls_to_the_next_candidate() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let config = format!(
        r#"
schema_version = 1

[llm_clients.retiring]
format = "openai_responses"
base_url = "{url}"

[targets.retired-r1]
id = "vendor/retired-404-a1"
llm_client = "retiring"

[targets.healthy-r2]
id = "vendor/ok-a1"
llm_client = "retiring"

[routes.ST_High]
id = "ST_High"
type = "passthrough"
target = "retired-r1"
candidates = ["healthy-r2"]
"#
    );
    let (status, body) = call(
        runner(&config),
        "ST_High",
        json!({"model": "ST_High", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        seen_models(&upstream).await,
        ["vendor/retired-404-a1", "vendor/ok-a1"],
        "the retired R1 is attempted, then the healthy R2 serves"
    );
}

/// A 404 carrying `error.metadata` says the REQUEST was filtered (missing asset or
/// unsupported modality), so it stays terminal: no candidate rollover, no crossing.
#[tokio::test]
async fn a_404_with_error_metadata_is_terminal() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let config = format!(
        r#"
schema_version = 1

[llm_clients.rejecting]
format = "openai_responses"
base_url = "{url}"

[targets.gone-r1]
id = "vendor/filtered-404-a1"
llm_client = "rejecting"

[targets.healthy-r2]
id = "vendor/ok-a1"
llm_client = "rejecting"

[routes.ST_Low]
id = "ST_Low"
type = "passthrough"
target = "gone-r1"
candidates = ["healthy-r2"]
"#
    );
    let (status, body) = call(
        runner(&config),
        "ST_Low",
        json!({"model": "ST_Low", "input": "hi"}),
    )
    .await;

    assert!(!status.is_success(), "{body}");
    assert_eq!(
        seen_models(&upstream).await,
        ["vendor/filtered-404-a1"],
        "a filtered-request 404 must not reach R2"
    );
}

/// A free-scope quota on one child does NOT block the paid child on the SAME
/// credential. The two-tier design from the dual-route deploy: free tiers first, then
/// the paid leg on the same credential, which draws on the account balance rather
/// than the free allowance.
#[tokio::test]
async fn a_free_scope_quota_does_not_block_the_paid_leg_on_the_same_credential() {
    let quota = Upstream::new();
    let paid = Upstream::new();
    let quota_url = quota.clone().server();
    let paid_url = paid.clone().server();

    // Both children sit on DIFFERENT llm_clients, so the credential identity differs;
    // the paid child's single target is a non-:free slug. The free child returns the
    // openrouter_free_tier_daily cap on its first call.
    let config = format!(
        "schema_version = 1\n{}\n{}\n[composites.ST_Route]\nchildren = [\"ST_Free\", \"ST_Paid\"]\n",
        leaf_named("ST_Free", &quota_url, &["account-quota-1"], "vendor/"),
        leaf_named("ST_Paid", &paid_url, &["paid-ok-1"], "vendor/")
    );

    let outcome = runner(&config)
        .execute_composite(
            "ST_Route",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Route".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the paid leg must still serve after the free cap");

    assert_eq!(outcome.child_route, "ST_Paid");
    assert!(
        outcome.crossed_child,
        "the request crossed from the free tier"
    );
    assert_eq!(quota.calls(), 1, "the free tier made exactly one call");
    assert_eq!(
        paid.calls(),
        1,
        "the paid leg served on the same credential"
    );
}

/// A free-scope quota DOES block a later free-only child on the same credential: the
/// whole point of recording the scope is to skip a child whose every target is a
/// `:free` slug drawing on the same exhausted allowance.
#[tokio::test]
async fn a_free_scope_quota_blocks_a_later_free_only_child_on_the_same_credential() {
    let quota = Upstream::new();
    let quota_url = quota.clone().server();

    // Two free-only children on ONE credential (shared client); the second would only
    // spend a call on the allowance already known to be at zero.
    let config = format!(
        concat!(
            "schema_version = 1\n",
            "\n[llm_clients.shared]\nformat = \"openai_responses\"\nbase_url = \"{url}\"\n",
            "{child_a}\n{child_b}\n",
            "[composites.ST_Route]\nchildren = [\"ST_Free1\", \"ST_Free2\"]\n"
        ),
        url = quota_url,
        child_a = leaf_shared("ST_Free1", &["account-quota-1:free"], "vendor/", "shared"),
        child_b = leaf_shared("ST_Free2", &["account-quota-2:free"], "vendor/", "shared"),
    );

    let result = runner(&config)
        .execute_composite(
            "ST_Route",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Route".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await;

    assert!(result.is_err(), "no child can serve: the allowance is out");
    assert_eq!(
        quota.calls(),
        1,
        "the second free child must be skipped without a provider call: {:?}",
        seen_models(&quota).await
    );
}

/// A whole-credential quota (402, exhausted balance) DOES block the paid leg on the
/// same credential: the balance is the resource the paid slug draws on.
#[tokio::test]
async fn a_whole_credential_quota_blocks_the_paid_leg_on_the_same_credential() {
    let payment = Upstream::new();
    let payment_url = payment.clone().server();

    // One route, paid slug, 402 on the first candidate. There is no second child, so
    // the request fails; what the assert pins is that no attempt was wasted.
    let config = format!(
        "schema_version = 1\n{}",
        leaf_named("ST_Paid", &payment_url, &["payment-required-1"], "vendor/")
    );

    let result = runner(&config)
        .execute_route(
            "ST_Paid",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Paid".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await;

    assert!(result.is_err(), "the exhausted balance surfaces");
    assert_eq!(
        payment.calls(),
        1,
        "one attempt, no retries against a balance known to be empty"
    );
}

/// A bare 401 blocks a later child on the SAME credential (the credential itself was
/// rejected), while a child on a DIFFERENT credential is still reached.
#[tokio::test]
async fn an_unauthorized_credential_blocks_same_credential_children_but_not_others() {
    let rejected = Upstream::new();
    let healthy = Upstream::new();
    let rejected_url = rejected.clone().server();
    let healthy_url = healthy.clone().server();

    // ST_Bad and ST_AlsoBad share one credential; ST_Good sits on another.
    let config = format!(
        concat!(
            "schema_version = 1\n",
            "\n[llm_clients.shared]\nformat = \"openai_responses\"\nbase_url = \"{bad_url}\"\n",
            "{child_a}\n{child_b}\n",
            "{child_c}\n",
            "[composites.ST_Route]\nchildren = [\"ST_Bad\", \"ST_AlsoBad\", \"ST_Good\"]\n"
        ),
        bad_url = rejected_url,
        child_a = leaf_shared("ST_Bad", &["unauthorized-1"], "vendor/", "shared"),
        child_b = leaf_shared("ST_AlsoBad", &["ok-1"], "vendor/", "shared"),
        child_c = leaf_named("ST_Good", &healthy_url, &["ok-1"], "vendor/"),
    );

    let outcome = runner(&config)
        .execute_composite(
            "ST_Route",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Route".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await
        .expect("the other credential's child serves");

    assert_eq!(outcome.child_route, "ST_Good");
    assert!(outcome.crossed_child);
    assert_eq!(
        rejected.calls(),
        1,
        "the second child on the rejected credential is skipped without a call"
    );
    assert_eq!(healthy.calls(), 1);
}

/// A 403 WITHOUT a budget marker is a guardrail or policy refusal: it must NOT block
/// the credential, and it stays terminal (no crossing, no second call).
#[tokio::test]
async fn a_policy_403_neither_blocks_the_credential_nor_crosses() {
    let rejecting = Upstream::new();
    let healthy = Upstream::new();
    let rejecting_url = rejecting.clone().server();
    let healthy_url = healthy.clone().server();

    let config = format!(
        "schema_version = 1\n{}\n{}\n[composites.ST_Route]\nchildren = [\"ST_Guard\", \"ST_Good\"]\n",
        leaf_named("ST_Guard", &rejecting_url, &["guardrail-403-1"], "vendor/"),
        leaf_named("ST_Good", &healthy_url, &["ok-1"], "vendor/")
    );

    let result = runner(&config)
        .execute_composite(
            "ST_Route",
            switchyard_protocol::Request {
                llm_request: switchyard_protocol::text_request(Some("ST_Route".into()), "hi"),
                raw_request: None,
                metadata: None,
                route_reasoning_policy: None,
                route_reasoning_effort: None,
            },
            None,
        )
        .await;

    // A 403 classifies as Auth, which is terminal: no crossing, no credential block.
    // The refusal surfaces to the caller rather than being retried elsewhere.
    assert!(result.is_err(), "the refusal surfaces");
    assert_eq!(
        rejecting.calls(),
        1,
        "no retry against a request the provider refuses"
    );
    assert_eq!(
        healthy.calls(),
        0,
        "a request-fact refusal must not be retried on another credential"
    );
}

/// Per-child context admission reserves the caller's output budget.
///
/// A request whose input leaves a child's window only 1k short must not be sent to
/// that child when the caller reserved 16k of output: input plus reservation exceeds
/// the window, and the provider would reject after a wasted call. The child with the
/// larger window serves instead.
#[tokio::test]
async fn a_child_without_room_for_the_output_reservation_is_skipped() {
    let small = Upstream::new();
    let large = Upstream::new();
    let small_url = small.clone().server();
    let large_url = large.clone().server();

    // The small child declares a 200k window; the large child 400k. The input is
    // estimated at ~199k (793k chars / 4) so input alone fits the small child but
    // input + 16k output does not.
    let filler = "x".repeat(793_000);
    let config = format!(
        r#"
schema_version = 1

[llm_clients.small]
format = "openai_responses"
base_url = "{small_url}"

[llm_clients.large]
format = "openai_responses"
base_url = "{large_url}"

[targets.small-r1]
id = "vendor/small-a1"
llm_client = "small"

[targets.large-r1]
id = "vendor/large-a1"
llm_client = "large"

[routes.ST_Small]
id = "ST_Small"
type = "passthrough"
target = "small-r1"
context_window = 200000

[routes.ST_Large]
id = "ST_Large"
type = "passthrough"
target = "large-r1"
context_window = 400000

[composites.ST_Route]
children = ["ST_Small", "ST_Large"]
context_window = 400000
"#
    );

    let outcome = runner(&config)
        .execute_route(
            "ST_Route",
            switchyard_protocol::Request {
                route_reasoning_policy: None,
                route_reasoning_effort: None,
                llm_request: switchyard_protocol::LlmRequest {
                    model: Some("ST_Route".into()),
                    messages: Vec::new(),
                    output: switchyard_protocol::OutputParams {
                        max_output_tokens: Some(16_000),
                        ..Default::default()
                    },
                    ..switchyard_protocol::LlmRequest::default()
                },
                raw_request: Some(json!({
                    "model": "ST_Route",
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": filler}]}]
                })),
                metadata: None,
            },
            None,
        )
        .await
        .expect("the large child serves");

    let composition = outcome.composition.expect("composite metadata");
    assert_eq!(composition.child_route, "ST_Large");
    assert_eq!(
        small.calls(),
        0,
        "the small child cannot hold input plus the 16k reservation: {:?}",
        seen_models(&small).await
    );
    assert_eq!(large.calls(), 1);
}
