// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `smart-bounded` shape: an NT primary with an interchangeable fallback.
//!
//! These tests pin the policy the operator set for this route — the advertised window, the
//! candidate order, and the rule that a fallback's ability to honour "no thinking" decides
//! nothing about its eligibility. They use the live route's real model ids so a change to
//! the shipped manifest has a counterpart here.
//!
//! No provider is contacted: every upstream is a scripted local listener.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// How one scripted upstream answers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    /// Answer 200 with a complete Responses body.
    Serve,
    /// Answer 503, so the route has to move to its next candidate.
    Unavailable,
    /// Answer 200, open an SSE stream, and end it without a terminal event.
    IncompleteStream,
}

#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<Value>>>,
    behaviour: Behaviour,
}

impl Upstream {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            bodies: Arc::new(Mutex::new(Vec::new())),
            behaviour,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn bodies(&self) -> Vec<Value> {
        self.bodies.lock().await.clone()
    }

    async fn models(&self) -> Vec<String> {
        self.bodies()
            .await
            .iter()
            .map(|body| body["model"].as_str().unwrap_or("?").to_string())
            .collect()
    }

    fn url(self) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let calls = Arc::clone(&self.calls);
        let seen = Arc::clone(&self.bodies);
        let behaviour = self.behaviour;
        let runtime = tokio::runtime::Handle::current();
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

                let model = parsed["model"].as_str().unwrap_or_default().to_string();
                let response = match behaviour {
                    Behaviour::Unavailable => format!(
                        "HTTP/1.1 503 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        json!({"error": {"message": "unavailable"}})
                            .to_string()
                            .len(),
                        json!({"error": {"message": "unavailable"}})
                    ),
                    Behaviour::Serve => {
                        let payload = json!({
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
                        .to_string();
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
                            payload.len()
                        )
                    }
                    Behaviour::IncompleteStream => {
                        // A well-formed opening frame and then EOF, with no terminal event.
                        let payload = concat!(
                            "event: response.created\n",
                            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_x\",\"model\":\"x\",\"status\":\"in_progress\",\"output\":[]}}\n\n"
                        );
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{payload}",
                            payload.len()
                        )
                    }
                };
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{address}/v1")
    }
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

/// The live `smart-bounded` route, with its real target names and model ids.
///
/// `fallback` is the candidate target block; the caller decides whether that fallback
/// declares a reasoning dialect at all.
fn config(luna_url: &str, openrouter_url: &str, fallback: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.openai_nt]
format = "openai_responses"
base_url = "{luna_url}"

[llm_clients.openrouter_stephen]
format = "openai_responses"
base_url = "{openrouter_url}"

[targets.openai-gpt-6-luna-nt]
id = "gpt-6-luna"
llm_client = "openai_nt"
vision = true
reasoning_policy = "disabled"
reasoning_dialect = "openai_effort"

{fallback}

[routes.smart-bounded]
id = "smart-bounded"
type = "passthrough"
target = "openai-gpt-6-luna-nt"
candidates = ["stephen-dots-3-note-preview-free"]
context_window = 262144
tool_calling = true
reasoning = true
vision = true
route_reasoning_policy = "none"
route_reasoning_policy_best_effort = true

# An unrelated bounded route: it must keep its own window and its own target, so an
# opt-in on one route cannot be mistaken for a deployment-wide change.
[targets.unrelated-primary]
id = "unrelated/model"
llm_client = "openai_nt"
reasoning_dialect = "openrouter_enabled"

[routes.unrelated-bounded]
id = "unrelated-bounded"
type = "passthrough"
target = "unrelated-primary"
candidates = []
context_window = 1048576
route_reasoning_policy = "medium"
"#
    )
}

/// The live fallback target: OpenRouter DOTS, which declares a boolean reasoning dialect.
const DOTS_TARGET: &str = r#"
[targets.stephen-dots-3-note-preview-free]
id = "dots-studio/dots-3-note-preview:free"
llm_client = "openrouter_stephen"
vision = true
tool_calling = true
reasoning_policy = "enabled"
reasoning_dialect = "openrouter_enabled"
"#;

/// A future fallback on a provider that declares nothing about reasoning, so it cannot
/// carry "no thinking" in any form.
const NO_NT_TARGET: &str = r#"
[targets.stephen-dots-3-note-preview-free]
id = "future/dots-replacement:free"
llm_client = "openrouter_stephen"
vision = true
tool_calling = true
"#;

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

async fn advertised_models(runner: Runner) -> Value {
    let router = switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    );
    let response = router
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body reads");
    serde_json::from_slice(&bytes).expect("models payload")
}

fn advertised_context(payload: &Value, route: &str) -> Option<u64> {
    payload["data"]
        .as_array()?
        .iter()
        .find(|entry| entry["id"].as_str() == Some(route))
        .and_then(|entry| entry["context_length"].as_u64())
}

/// The primary keeps its own NT behaviour: the route's "no thinking" policy reaches the
/// OpenAI target as the effort-off control that dialect actually understands.
#[tokio::test]
async fn the_primary_receives_the_no_thinking_control() {
    let luna = Upstream::new(Behaviour::Serve);
    let url = luna.clone().url();
    let deployment = runner(&config(&url, &url, DOTS_TARGET));

    let (status, body) = call(
        deployment,
        "smart-bounded",
        json!({"model": "smart-bounded", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    let bodies = luna.bodies().await;
    assert_eq!(bodies.len(), 1, "the primary answered on its own");
    assert_eq!(bodies[0]["model"], "gpt-6-luna");
    assert_eq!(
        bodies[0]["reasoning"]["effort"], "none",
        "the OpenAI dialect carries 'no thinking' as effort=none: {}",
        bodies[0]
    );
}

/// An unavailable primary hands the request to the declared fallback, and the removed
/// candidate is never contacted.
#[tokio::test]
async fn an_unavailable_primary_falls_back_to_the_declared_fallback() {
    let luna = Upstream::new(Behaviour::Unavailable);
    let dots = Upstream::new(Behaviour::Serve);
    let (luna_url, dots_url) = (luna.clone().url(), dots.clone().url());
    let deployment = runner(&config(&luna_url, &dots_url, DOTS_TARGET));

    let (status, body) = call(
        deployment,
        "smart-bounded",
        json!({"model": "smart-bounded", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "the fallback served: {body}");
    let served = dots.models().await;
    assert_eq!(
        served,
        vec!["dots-studio/dots-3-note-preview:free"],
        "the fallback is the configured DOTS target"
    );
    assert!(
        !served.iter().any(|model| model.contains("qwen")),
        "the removed Qwen candidate must not be reachable: {served:?}"
    );
}

/// A fallback that cannot express "no thinking" is still eligible, and receives no
/// reasoning field it did not ask for.
///
/// This is the operator's rule for this route: NT is best-effort per target, never an
/// eligibility requirement. The target below declares no reasoning dialect at all, which is
/// the hardest case — there is no field to send, so its native behaviour is used.
#[tokio::test]
async fn a_fallback_that_cannot_express_no_thinking_is_still_eligible() {
    let luna = Upstream::new(Behaviour::Unavailable);
    let future = Upstream::new(Behaviour::Serve);
    let (luna_url, future_url) = (luna.clone().url(), future.clone().url());
    // Loading at all is half the assertion: the strict default would refuse this document.
    let deployment = runner(&config(&luna_url, &future_url, NO_NT_TARGET));

    let (status, body) = call(
        deployment,
        "smart-bounded",
        json!({"model": "smart-bounded", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    let bodies = future.bodies().await;
    assert_eq!(
        bodies.len(),
        1,
        "the NT-incapable fallback must still be tried"
    );
    assert_eq!(bodies[0]["model"], "future/dots-replacement:free");
    assert!(
        bodies[0].get("reasoning").is_none(),
        "no reasoning field may be invented for a provider that declares none: {}",
        bodies[0]
    );
    assert!(
        bodies[0].get("reasoning_effort").is_none(),
        "no effort field may be invented either: {}",
        bodies[0]
    );
}

/// The route advertises the window the operator set, and an unrelated route keeps its own.
#[tokio::test]
async fn the_route_advertises_the_declared_context_window() {
    let luna = Upstream::new(Behaviour::Serve);
    let url = luna.clone().url();
    let payload = advertised_models(runner(&config(&url, &url, DOTS_TARGET))).await;

    assert_eq!(
        advertised_context(&payload, "smart-bounded"),
        Some(262_144),
        "smart-bounded must advertise 262144: {payload}"
    );
    assert_eq!(
        advertised_context(&payload, "unrelated-bounded"),
        Some(1_048_576),
        "an unrelated route must keep its own window: {payload}"
    );
}

/// The opt-in is scoped: the same deployment still enforces strictness on a route that did
/// not ask for best-effort semantics.
#[test]
fn best_effort_is_scoped_to_the_route_that_declares_it() {
    let strict = r#"
[targets.no-dialect]
id = "no/dialect"
llm_client = "openrouter_stephen"

[routes.strict-bounded]
id = "strict-bounded"
type = "passthrough"
target = "no-dialect"
candidates = []
route_reasoning_policy = "none"
"#;
    let source = format!(
        "{}\n{strict}",
        config(
            "https://example.test/v1",
            "https://example.test/v1",
            DOTS_TARGET
        )
    );

    let message = match switchyard_runner::Runner::from_toml(&source) {
        Ok(_) => panic!("a route without the opt-in must stay strict"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("route strict-bounded reasoning policy cannot be applied"),
        "got: {message}"
    );
}

/// A stream that dies after it has already started is not replayed.
///
/// The route may fall back before anything is delivered; once bytes are on the wire to the
/// caller, re-issuing the request would show them a second beginning.
#[tokio::test]
async fn a_partially_delivered_stream_is_not_replayed() {
    let broken = Upstream::new(Behaviour::IncompleteStream);
    let candidate = Upstream::new(Behaviour::Serve);
    let (broken_url, candidate_url) = (broken.clone().url(), candidate.clone().url());
    let deployment = runner(&format!(
        r#"
schema_version = 1

[llm_clients.openai_nt]
format = "openai_responses"
base_url = "{broken_url}"

[llm_clients.openrouter_stephen]
format = "openai_responses"
base_url = "{candidate_url}"

[targets.broken-primary]
id = "broken/primary"
llm_client = "openai_nt"

[targets.spare]
id = "spare/model"
llm_client = "openrouter_stephen"

[routes.replay-probe]
id = "replay-probe"
type = "passthrough"
target = "broken-primary"
candidates = ["spare"]
"#
    ));

    let (_status, _body) = call(
        deployment,
        "replay-probe",
        json!({"model": "replay-probe", "input": "hi", "stream": true}),
    )
    .await;

    assert_eq!(
        broken.calls(),
        1,
        "the partially delivered stream must not be re-requested"
    );
    assert_eq!(
        candidate.calls(),
        0,
        "a stream that already started must not fall back: {:?}",
        candidate.models().await
    );
}
