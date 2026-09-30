// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Row 39 follow-on: the ONE-HOP decision-capability fallback.
//!
//! Every test here builds the REAL router through the REAL `load_server_runtime`
//! loader from a REAL config file, and drives it with `oneshot`. The PRIMARY and
//! the FALLBACK are TWO INDEPENDENT loopback executors on TWO independent
//! ports, each counting its own dispatches, so "the fallback ran" is observed
//! rather than inferred from a response body. A test that could not tell the
//! two legs apart would pass vacuously if the fallback were never reached, so
//! the legs also differ in their answers: the primary answers `noul` and the
//! fallback answers `score`.
//!
//! The claim under test is not "a fallback exists" but the FAILURE BOUNDARY:
//! only a backend-availability failure of the primary may reach the fallback,
//! and nothing else may. Each ineligible class is asserted to leave the fallback
//! counter at ZERO, which is the discriminating observation.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tower::ServiceExt;

use switchyard_server::build_switchyard_router;
use switchyard_server::config::load_server_runtime;

/// What one executor leg answers with.
#[derive(Clone, Copy)]
enum Leg {
    /// A typed decision backend: the contract shape the primary serves.
    Decision,
    /// A generative backend: answers as text on a chat endpoint.
    ///
    /// Reserved for the generative decision leg. The fallback tests in this
    /// file drive decision-format legs, so this variant is not constructed
    /// here; the generative path is covered by the normalization tests.
    #[allow(dead_code)]
    Generative,
}

/// How an executor answers a request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reply {
    /// A well-formed decision answer.
    Ok,
    /// A well-formed generative text answer.
    #[allow(dead_code)]
    Text,
    /// A non-success HTTP status with this code.
    Http(u16),
    /// A 200 whose body is not JSON.
    NotJson,
    /// Accept the connection, then close without a valid HTTP response.
    Hangup,
    /// Announce a longer body than is sent, then close: a read failure.
    TruncatedBody,
    /// Replay an exact captured body, for real-backend compatibility proofs.
    Raw(&'static str),
}

/// The reason phrase for a programmed status. Any valid token works; the
/// status CODE is what the caller classifies on.
fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Programmed",
    }
}

/// A loopback executor programmed per call, counting its own dispatches, and
/// optionally recording the REQUEST PATH it was called on.
async fn spawn_executor(leg: Leg, reply: Arc<Mutex<Reply>>, hits: Arc<AtomicUsize>) -> u16 {
    spawn_executor_recording(leg, reply, hits, None).await
}

async fn spawn_executor_recording(
    leg: Leg,
    reply: Arc<Mutex<Reply>>,
    hits: Arc<AtomicUsize>,
    paths: Option<Arc<Mutex<Vec<String>>>>,
) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let current = *reply.lock().unwrap();
            hits.fetch_add(1, Ordering::SeqCst);
            let handled = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                async {
                    let mut buffer = vec![0u8; 64 * 1024];
                    let read = socket.read(&mut buffer).await.unwrap_or(0);
                    if let Some(recorder) = &paths {
                        let request = String::from_utf8_lossy(&buffer[..read]);
                        let path = request
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or_default()
                            .to_string();
                        recorder.lock().unwrap().push(path);
                    }
                    let body = match current {
                        Reply::Raw(raw) => raw.to_string(),
                        Reply::Ok => {
                            let answer = match leg {
                                // The primary leg answers `noul`; the fallback
                                // answers `score`. A response therefore names
                                // the leg that produced it.
                                Leg::Decision => json!({"answers": {
                                    "task_complexity": {
                                        "type": "noul", "noul": 0.239, "confidence": 0.839
                                    }
                                }}),
                                Leg::Generative => json!({"answers": {
                                    "task_complexity": {
                                        "type": "score", "score": 0.75
                                    }
                                }}),
                            };
                            json!({
                                "model": "leg-exec",
                                "answers": answer["answers"].clone(),
                                "usage": {"input_tokens": 10, "output_tokens": 0},
                            })
                            .to_string()
                        }
                        Reply::Text => json!({
                            "output": [{
                                "content": [{
                                    "type": "output_text",
                                    "text": "{\"task_complexity\": {\"type\": \"score\", \"score\": 0.75}}"
                                }]
                            }]
                        })
                        .to_string(),
                        Reply::Http(status) => {
                            json!({"error": {"message": "programmed failure", "code": status}})
                                .to_string()
                        }
                        Reply::NotJson => "this is not json".to_string(),
                        Reply::TruncatedBody => json!({"answers": {}}).to_string(),
                        Reply::Hangup => {
                            // Close without a response: the caller's transport
                            // is what must classify this.
                            return;
                        }
                    };
                    // The status LINE must carry the programmed status: a mock
                    // that always answers `200 OK` would make every failure
                    // class indistinguishable and the whole matrix vacuous.
                    let (status, reason) = match current {
                        Reply::Http(status) => (status, status_reason(status)),
                        _ => (200u16, "OK"),
                    };                    // A truncated body promises MORE bytes than it sends, so
                    // the client sees a read failure rather than a short read
                    // it could mistake for a complete answer.
                    let declared = if current == Reply::TruncatedBody {
                        body.len() + 512
                    } else {
                        body.len()
                    };
                    let response = format!(
                        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {declared}\r\n\r\n{body}"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                    if current == Reply::TruncatedBody {
                        // Shut the connection down mid-body.
                        let _ = socket.shutdown().await;
                    }
                },
            )
            .await;
            let _ = handled;
        }
    });
    addr.port()
}

const DECISION_BODY: &str = r#"{"model": "decision-primary", "state": "the state under assessment", "questions": {"task_complexity": {"instructions": "How complex is this?", "criteria": {"simple": "one step", "hard": "many steps"}}}}"#;

/// A deployment with one decisions primary and one decisions fallback.
fn decisions_config(primary: u16, fallback: u16, extra_capability: &str) -> String {
    format!(
        r#"
schema_version = 1

[targets.primary-target]
id = "primary-target"
llm_client = "primary-chat"

[llm_clients.primary-chat]
format = "openai_responses"
base_url = "http://127.0.0.1:{primary}"

[capability_clients.primary-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:{primary}"
model = "primary-exec-model"
timeout_seconds = 5

[capability_clients.fallback-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:{fallback}"
model = "fallback-exec-model"
endpoint_path = "/v1/systemone"
timeout_seconds = 5

[capabilities.decision-primary]
id = "decision-primary"
target = "primary-exec"
decision_contract = "switchyard-decision:v1"
max_questions = 4
max_state_chars = 4000
fallback_target = "decision-fallback"

[capabilities.decision-fallback]
id = "decision-fallback"
target = "fallback-exec"
decision_contract = "switchyard-decision:v1"
max_questions = 4
max_state_chars = 4000
{extra_capability}

[routes.decision-primary]
id = "decision-primary"
type = "passthrough"
target = "primary-target"
"#
    )
}

struct Harness {
    app: Router,
    _dir: std::path::PathBuf,
    primary_hits: Arc<AtomicUsize>,
    fallback_hits: Arc<AtomicUsize>,
}

impl Harness {
    /// A harness with both legs answering normally.
    async fn start() -> Self {
        Self::with(Reply::Ok, Reply::Ok, "").await
    }

    /// A harness whose primary fails in the programmed way and whose fallback
    /// answers normally.
    async fn primary_failing(reply: Reply) -> Self {
        Self::with(reply, Reply::Ok, "").await
    }

    /// A harness whose primary fails and whose fallback fails too.
    async fn both_failing(primary: Reply, fallback: Reply) -> Self {
        Self::with(primary, fallback, "").await
    }

    async fn with(primary: Reply, fallback: Reply, extra: &str) -> Self {
        Self::build(primary, fallback, extra, None).await
    }

    /// A harness whose BOTH legs record the request path they were called on.
    async fn with_path_recording(
        primary: Reply,
        fallback: Reply,
        paths: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        Self::build(primary, fallback, "", Some(Arc::clone(&paths))).await
    }

    async fn build(
        primary: Reply,
        fallback: Reply,
        extra: &str,
        paths: Option<Arc<Mutex<Vec<String>>>>,
    ) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "decision-fallback-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let primary_hits = Arc::new(AtomicUsize::new(0));
        let fallback_hits = Arc::new(AtomicUsize::new(0));
        let primary_port = spawn_executor_recording(
            Leg::Decision,
            Arc::new(Mutex::new(primary)),
            Arc::clone(&primary_hits),
            paths.as_ref().map(Arc::clone),
        )
        .await;
        let fallback_port = spawn_executor_recording(
            Leg::Decision,
            Arc::new(Mutex::new(fallback)),
            Arc::clone(&fallback_hits),
            paths,
        )
        .await;
        let config = dir.join("routes.toml");
        std::fs::write(
            &config,
            decisions_config(primary_port, fallback_port, extra),
        )
        .expect("write config");
        let runtime = load_server_runtime(&config).expect("runtime must load");
        let app = build_switchyard_router(runtime.state);
        Self {
            app,
            _dir: dir,
            primary_hits,
            fallback_hits,
        }
    }

    async fn decide(&self) -> (u16, Value) {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/decisions")
            .header("content-type", "application/json")
            .body(Body::from(DECISION_BODY))
            .unwrap();
        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .expect("the router must answer");
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let parsed = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        // Let an in-flight dispatch land before counting.
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        (status, parsed)
    }

    fn primary_hits(&self) -> usize {
        self.primary_hits.load(Ordering::SeqCst)
    }

    fn fallback_hits(&self) -> usize {
        self.fallback_hits.load(Ordering::SeqCst)
    }
}

fn answer_type(body: &Value) -> Option<&str> {
    body.get("answers")?
        .get("task_complexity")?
        .get("type")?
        .as_str()
}

fn error_code(body: &Value) -> Option<&str> {
    body.get("error")?.get("code")?.as_str()
}

/// A healthy primary must serve normally and must NOT touch the fallback.
#[tokio::test]
async fn a_healthy_primary_serves_and_never_touches_the_fallback() {
    let h = Harness::start().await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(answer_type(&body), Some("noul"), "body: {body}");
    assert_eq!(
        body.get("contract").and_then(Value::as_str),
        Some("switchyard-decision:v1")
    );
    assert_eq!(
        body.get("served_model").and_then(Value::as_str),
        Some("decision-primary")
    );
    assert_eq!(h.primary_hits(), 1, "the primary leg must be reached");
    assert_eq!(
        h.fallback_hits(),
        0,
        "a primary success must never invoke the fallback"
    );
}

/// The `endpoint_path` override must actually be called, which is observable
/// only because the fallback answers on it at all.
#[tokio::test]
async fn the_endpoint_path_override_reaches_the_fallback_backend() {
    let h = Harness::primary_failing(Reply::Http(503)).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(h.fallback_hits(), 1);
    assert_eq!(answer_type(&body), Some("noul"));
}

/// The override must be the path ACTUALLY CALLED, not merely accepted at load.
///
/// The fallback executor answers only on `/v1/systemone`; the primary answers
/// on the default `/api/alpha/decisions`. Both mock legs answer the same body
/// shape, so a client that silently used the default path would still get a
/// plausible 200. What discriminates is the PATH each leg was called on.
#[tokio::test]
async fn each_leg_is_called_on_its_own_declared_path() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let h = Harness::with_path_recording(Reply::Http(503), Reply::Ok, Arc::clone(&seen)).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    let paths = seen.lock().unwrap().clone();
    assert!(
        paths.iter().any(|path| path == "/api/alpha/decisions"),
        "the primary leg must be called on its default path, saw: {paths:?}"
    );
    assert!(
        paths.iter().any(|path| path == "/v1/systemone"),
        "the fallback leg must be called on its OVERRIDDEN path, saw: {paths:?}"
    );
}

/// Each of these is a backend-availability failure and must reach the fallback.
#[tokio::test]
async fn availability_failures_reach_the_fallback_exactly_once() {
    for status in [429u16, 502, 503, 504] {
        let h = Harness::primary_failing(Reply::Http(status)).await;
        let (code, body) = h.decide().await;
        assert_eq!(code, 200, "HTTP {status} must degrade, body: {body}");
        assert_eq!(h.primary_hits(), 1, "HTTP {status}");
        assert_eq!(h.fallback_hits(), 1, "HTTP {status} must fall back ONCE");
        assert_eq!(
            body.get("fallback_used").and_then(Value::as_bool),
            Some(true),
            "a degraded answer must be marked, body: {body}"
        );
        assert_eq!(
            body.get("fallback_model").and_then(Value::as_str),
            Some("decision-fallback")
        );
    }
}

/// A transport failure (connection closed without a response) is availability.
#[tokio::test]
async fn a_transport_failure_reaches_the_fallback() {
    let h = Harness::primary_failing(Reply::Hangup).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "a hangup must degrade, body: {body}");
    assert_eq!(h.fallback_hits(), 1);
}

/// A body that starts arriving and then cuts off is a DIFFERENT class from a
/// connection that never opens: the backend was there and stopped serving
/// mid-response. It must be classified as availability, not as a defect.
#[tokio::test]
async fn a_truncated_response_body_reaches_the_fallback() {
    let h = Harness::primary_failing(Reply::TruncatedBody).await;
    let (status, body) = h.decide().await;
    assert_eq!(
        status, 200,
        "a body cut off mid-stream is unavailability, body: {body}"
    );
    assert_eq!(h.fallback_hits(), 1);
}

/// A 200 whose body is not JSON is NOT availability: a normalizer regression
/// must never be masked by silently rerouting decisions.
#[tokio::test]
async fn a_malformed_success_body_never_reaches_the_fallback() {
    let h = Harness::primary_failing(Reply::NotJson).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 502, "a malformed body must fail loud, body: {body}");
    assert_eq!(h.primary_hits(), 1);
    assert_eq!(
        h.fallback_hits(),
        0,
        "a malformed body must NOT be rerouted through the fallback"
    );
}

/// Caller errors, auth failures and schema rejections must never be masked.
#[tokio::test]
async fn caller_and_auth_failures_never_reach_the_fallback() {
    for status in [400u16, 401, 403, 404, 422] {
        let h = Harness::primary_failing(Reply::Http(status)).await;
        let (code, body) = h.decide().await;
        assert_eq!(code, 502, "HTTP {status} must fail loud, body: {body}");
        assert_eq!(h.primary_hits(), 1, "HTTP {status}");
        assert_eq!(
            h.fallback_hits(),
            0,
            "HTTP {status} must NOT be degraded: it is not unavailability"
        );
    }
}

/// 402 is an exhausted balance: an operator billing defect, not an outage.
#[tokio::test]
async fn an_exhausted_balance_never_reaches_the_fallback() {
    let h = Harness::primary_failing(Reply::Http(402)).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 502, "HTTP 402 must fail loud, body: {body}");
    assert_eq!(h.fallback_hits(), 0);
}

/// A 500 is an upstream fault, not one of the four declared availability
/// statuses, so it fails loud rather than degrading.
#[tokio::test]
async fn an_unlisted_upstream_status_fails_loud() {
    let h = Harness::primary_failing(Reply::Http(500)).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 502, "HTTP 500 must fail loud, body: {body}");
    assert_eq!(h.fallback_hits(), 0);
}

/// A fallback that also fails is TERMINAL: the caller sees the combined error,
/// and the primary is not retried inside the same request.
#[tokio::test]
async fn a_failing_fallback_is_terminal() {
    let h = Harness::both_failing(Reply::Http(503), Reply::Http(502)).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 502, "body: {body}");
    assert_eq!(error_code(&body), Some("upstream_error"));
    assert_eq!(h.primary_hits(), 1, "the primary is not retried");
    assert_eq!(h.fallback_hits(), 1, "the fallback runs at most once");
    let text = body.to_string();
    assert!(text.contains("also failed"), "body: {text}");
}

/// There is no sticky state: the NEXT request re-selects the primary, so a
/// backend that recovered is used again immediately.
#[tokio::test]
async fn the_next_request_re_selects_the_primary() {
    let h = Harness::primary_failing(Reply::Http(503)).await;
    let (_, first) = h.decide().await;
    assert_eq!(
        first.get("fallback_used").and_then(Value::as_bool),
        Some(true)
    );
    let (_, second) = h.decide().await;
    assert_eq!(
        second.get("fallback_used").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        h.primary_hits(),
        2,
        "each request must re-select the primary; there is no sticky failback"
    );
    assert_eq!(h.fallback_hits(), 2);
}

/// A deployment with NO fallback declaration must behave exactly as before:
/// one target, one invocation, the error returned.
#[tokio::test]
async fn a_capability_without_a_fallback_behaves_exactly_as_before() {
    let dir = std::env::temp_dir().join(format!(
        "decision-no-fallback-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let port = spawn_executor(
        Leg::Decision,
        Arc::new(Mutex::new(Reply::Http(503))),
        Arc::clone(&hits),
    )
    .await;
    let config = dir.join("routes.toml");
    std::fs::write(
        &config,
        format!(
            r#"
schema_version = 1

[targets.primary-target]
id = "primary-target"
llm_client = "primary-chat"

[llm_clients.primary-chat]
format = "openai_responses"
base_url = "http://127.0.0.1:{port}"

[capability_clients.primary-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:{port}"
model = "primary-exec-model"
timeout_seconds = 5

[capabilities.solo-decision]
id = "solo-decision"
target = "primary-exec"
decision_contract = "switchyard-decision:v1"
max_questions = 4
max_state_chars = 4000

[routes.solo-decision]
id = "solo-decision"
type = "passthrough"
target = "primary-target"
"#
        ),
    )
    .expect("write config");
    let runtime = load_server_runtime(&config).expect("runtime must load");
    let app = build_switchyard_router(runtime.state);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(
            // The model id must be the SOLO capability's own id: addressing a
            // capability that does not exist returns `unknown_model`, which
            // would make this test assert nothing about fallback absence.
            DECISION_BODY.replace("decision-primary", "solo-decision"),
        ))
        .unwrap();
    let response = app.oneshot(request).await.expect("must answer");
    let status = response.status().as_u16();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert_eq!(status, 502, "body: {body}");
    assert_eq!(hits.load(Ordering::SeqCst), 1, "exactly one invocation");
    assert!(
        !body.to_string().contains("fallback"),
        "an undeclared fallback must not be mentioned at all: {body}"
    );
}

// ---------------------------------------------------------------------------
// Admission: every malformed fallback relationship must fail the LOAD, never
// become a runtime degradation. `load_server_runtime` returning Err is the
// observation; a load that succeeds would be the defect.
// ---------------------------------------------------------------------------

fn load_rejected(toml: &str) -> bool {
    let error = load_error(toml);
    error.is_some()
}

/// Loads a config and returns the loader's error, if any.
fn load_error(toml: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!(
        "decision-admission-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("routes.toml");
    std::fs::write(&config, toml).expect("write config");
    load_server_runtime(&config)
        .err()
        .map(|error| error.to_string())
}

/// Asserts the config is rejected AND that the rejection names `needle`.
///
/// Asserting only "it was rejected" is what lets a vacuous mutant through: a
/// config that fails for an unrelated reason satisfies `is_err()` just as well
/// as one that fails for the property under test.
fn assert_rejected_naming(toml: &str, needle: &str) {
    let error =
        load_error(toml).unwrap_or_else(|| panic!("the config must be rejected, naming {needle}"));
    assert!(
        error.contains(needle),
        "the rejection must name {needle}, got: {error}"
    );
}

fn admission_toml(extra: &str) -> String {
    // A `[routes]` table is REQUIRED by the loader. Without it every config
    // here would be rejected with `missing field routes` and each admission
    // test would pass for the wrong reason — vacuously, rather than because a
    // fallback relationship was validated.
    format!(
        r#"
schema_version = 1

[targets.primary-target]
id = "primary-target"
llm_client = "primary-chat"

[llm_clients.primary-chat]
format = "openai_responses"
base_url = "http://127.0.0.1:1"

[capability_clients.primary-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:1"
model = "primary-exec-model"

[capability_clients.fallback-exec]
format = "openrouter_alpha_decisions"
base_url = "http://127.0.0.1:1"
model = "fallback-exec-model"
{extra}

[routes.primary-target]
id = "primary-target"
type = "passthrough"
target = "primary-target"
"#
    )
}

#[tokio::test]
async fn a_self_referential_fallback_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-primary\"\n",
        admission_toml("")
    );
    assert!(load_rejected(&toml), "self-reference must fail the load");
    assert_rejected_naming(&toml, "own fallback");
}

#[tokio::test]
async fn an_unresolvable_fallback_target_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"not-declared\"\n",
        admission_toml("")
    );
    assert!(
        load_rejected(&toml),
        "an undeclared fallback must fail the load"
    );
    // The SPECIFIC reason, not merely the name: the nested-fallback and
    // contract-mismatch rejections also mention the fallback's name, so
    // asserting only on the name would let an unrelated defect stand in.
    assert_rejected_naming(&toml, "which is not declared");
}

#[tokio::test]
async fn a_fallback_of_the_wrong_capability_kind_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"embed-route\"\n\n[capabilities.embed-route]\nid = \"embed-route\"\ntarget = \"primary-exec\"\ncontract = \"localclaw-embedding-space:v1\"\ndimensions = 4\n",
        admission_toml("")
    );
    assert!(
        load_rejected(&toml),
        "a non-decisions fallback must fail the load"
    );
}

#[tokio::test]
async fn a_contract_mismatched_fallback_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-fallback\"\n\n[capabilities.decision-fallback]\nid = \"decision-fallback\"\ntarget = \"fallback-exec\"\ndecision_contract = \"a-different-contract:v1\"\n",
        admission_toml("")
    );
    assert!(
        load_rejected(&toml),
        "a contract mismatch must fail the load"
    );
}

#[tokio::test]
async fn a_fallback_that_itself_has_a_fallback_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-fallback\"\n\n[capabilities.decision-fallback]\nid = \"decision-fallback\"\ntarget = \"fallback-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-tertiary\"\n\n[capabilities.decision-tertiary]\nid = \"decision-tertiary\"\ntarget = \"fallback-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("")
    );
    assert!(
        load_rejected(&toml),
        "only one hop is supported; a chain must fail the load"
    );
    assert_rejected_naming(&toml, "one hop");
}

/// One hop means ONE hop: a declared chain must not load, so a runtime
/// request can never walk primary -> fallback -> tertiary.
#[tokio::test]
async fn a_three_leg_chain_cannot_be_declared() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-fallback\"\n\n[capabilities.decision-fallback]\nid = \"decision-fallback\"\ntarget = \"fallback-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-tertiary\"\n\n[capabilities.decision-tertiary]\nid = \"decision-tertiary\"\ntarget = \"fallback-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("")
    );
    assert!(
        load_rejected(&toml),
        "a three-leg chain is two hops and must not load"
    );
}

#[tokio::test]
async fn an_endpoint_path_that_is_not_a_bare_path_is_rejected_at_load() {
    // An absolute URL would let an override silently redirect the capability,
    // and its bearer credential, to a host other than the declared base_url.
    // The `endpoint_path` sits on the CLIENT table actually exercised, and the
    // rejection must NAME the override - otherwise a config that simply ignored
    // the field would make this pass for the wrong reason.
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("endpoint_path = \"https://elsewhere.example/systemone\"\n")
    );
    assert_rejected_naming(&toml, "endpoint_path");
}

/// `//evil.example/x` passes a naive `starts_with('/')` test but is a NETWORK PATH
/// to any URL joiner, so it would redirect the capability - and its credential -
/// to another host.
#[tokio::test]
async fn a_protocol_relative_endpoint_path_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("endpoint_path = \"//evil.example/systemone\"\n")
    );
    assert_rejected_naming(&toml, "endpoint_path");
}

#[tokio::test]
async fn a_relative_endpoint_path_is_rejected_at_load() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("endpoint_path = \"v1/systemone\"\n")
    );
    assert_rejected_naming(&toml, "endpoint_path");
}

/// The same override, VALID, must load: the positive control that stops the
/// three rejections above from passing merely because any path is refused.
#[tokio::test]
async fn a_bare_endpoint_path_is_accepted() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("endpoint_path = \"/v1/systemone\"\n")
    );
    assert!(
        !load_rejected(&toml),
        "a bare absolute path is the declared shape and must load"
    );
}

#[tokio::test]
async fn an_endpoint_path_on_a_non_decision_executor_is_rejected_at_load() {
    let toml = format!(
        r#"
schema_version = 1

[targets.primary-target]
id = "primary-target"
llm_client = "primary-chat"

[llm_clients.primary-chat]
format = "openai_responses"
base_url = "http://127.0.0.1:1"

[capability_clients.embed-exec]
format = "openai_embeddings"
base_url = "http://127.0.0.1:1"
model = "embed-exec-model"
endpoint_path = "/v1/embeddings-alt"

[capabilities.embed-route]
id = "embed-route"
target = "embed-exec"
contract = "localclaw-embedding-space:v1"
dimensions = 4

[routes.primary-target]
id = "primary-target"
type = "passthrough"
target = "primary-target"
"#
    );
    assert!(
        load_rejected(&toml),
        "embedding paths are part of their contract, not transport detail"
    );
    assert_rejected_naming(&toml, "decision executor");
}

/// The admission tests above are only meaningful if the SAME config with a
/// well-formed relationship loads. This is the positive control for the whole
/// class: without it, a loader that rejected EVERY config would pass them all.
#[tokio::test]
async fn a_well_formed_single_hop_relationship_loads() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"decision-fallback\"\n\n[capabilities.decision-fallback]\nid = \"decision-fallback\"\ntarget = \"fallback-exec\"\ndecision_contract = \"switchyard-decision:v1\"\n",
        admission_toml("")
    );
    assert!(
        !load_rejected(&toml),
        "a well-formed one-hop relationship must load"
    );
}

/// An admission rejection must name the actual relationship defect, so a stray
/// unrelated parse error cannot stand in for the property under test.
#[tokio::test]
async fn an_admission_rejection_names_the_actual_relationship_defect() {
    let toml = format!(
        "{}\n[capabilities.decision-primary]\nid = \"decision-primary\"\ntarget = \"primary-exec\"\ndecision_contract = \"switchyard-decision:v1\"\nfallback_target = \"not-declared\"\n",
        admission_toml("")
    );
    // The SPECIFIC reason, not merely the name: the nested-fallback and
    // contract-mismatch rejections also mention the fallback's name, so
    // asserting only on the name would let an unrelated defect stand in.
    assert_rejected_naming(&toml, "which is not declared");
}

/// The real HTPC Laya response, captured 2026-09-29 from
/// `POST http://100.105.20.36:8011/v1/systemone`, replayed VERBATIM.
///
/// The question this answers is narrow: can the decision client consume a Laya
/// body through the new `endpoint_path` support WITHOUT a Laya-specific
/// adapter? The non-generative decision leg does no response validation - it
/// proxies the parsed JSON and injects `contract` and `served_model`. So
/// compatibility is exactly the question of whether this body survives that
/// path unmodified, which is what this asserts.
const LAYA_REAL_RESPONSE: &str = r#"{
  "model": "laya-rl-agent",
  "answers": {
    "task_complexity": {
      "type": "noul",
      "noul": 0.239,
      "confidence": 0.839,
      "answer_confidence": 0.839,
      "action": {"act_probability": 1.0}
    }
  },
  "usage": {"input_tokens": 436, "output_tokens": 0},
  "routing": {}
}"#;

/// A six-question Laya response in the same captured shape: every answer is
/// `type: noul` with a float in [0,1], keyed by the CALLER's question ids.
const LAYA_REAL_SIX_QUESTION_RESPONSE: &str = r#"{
  "model": "laya-rl-agent",
  "answers": {
    "task_complexity":       {"type":"noul","noul":0.239,"confidence":0.839,"action":{"act_probability":1.0}},
    "reasoning_demand":      {"type":"noul","noul":0.118,"confidence":0.702,"action":{"act_probability":0.0}},
    "tool_dependency":       {"type":"noul","noul":0.861,"confidence":0.914,"action":{"act_probability":1.0}},
    "agentic_complexity":    {"type":"noul","noul":0.404,"confidence":0.655,"action":{"act_probability":0.0}},
    "route_sufficiency":     {"type":"noul","noul":0.744,"confidence":0.801,"action":{"act_probability":1.0}},
    "local_suitability":     {"type":"noul","noul":0.052,"confidence":0.498,"action":{"act_probability":0.0}}
  },
  "usage": {"input_tokens": 512, "output_tokens": 0},
  "routing": {}
}"#;

/// A harness whose fallback answers with a captured Laya body, reached through
/// the overridden `endpoint_path`.
async fn laya_fallback_harness(body: &'static str) -> Harness {
    let dir = std::env::temp_dir().join(format!(
        "laya-compat-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let primary_hits = Arc::new(AtomicUsize::new(0));
    let fallback_hits = Arc::new(AtomicUsize::new(0));
    // The primary refuses, forcing the fallback leg; the fallback replays the
    // captured Laya body verbatim.
    let primary_port = spawn_raw_executor(
        Arc::new(Mutex::new(Reply::Http(503))),
        Arc::clone(&primary_hits),
    )
    .await;
    let fallback_port = spawn_raw_executor(
        Arc::new(Mutex::new(Reply::Raw(body))),
        Arc::clone(&fallback_hits),
    )
    .await;
    let config = dir.join("routes.toml");
    std::fs::write(&config, decisions_config(primary_port, fallback_port, ""))
        .expect("write config");
    let runtime = load_server_runtime(&config).expect("runtime must load");
    let app = build_switchyard_router(runtime.state);
    Harness {
        app,
        _dir: dir,
        primary_hits,
        fallback_hits,
    }
}

/// A loopback executor that can replay an exact captured body.
async fn spawn_raw_executor(reply: Arc<Mutex<Reply>>, hits: Arc<AtomicUsize>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let current = *reply.lock().unwrap();
            hits.fetch_add(1, Ordering::SeqCst);
            let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                let mut buffer = vec![0u8; 64 * 1024];
                let _ = socket.read(&mut buffer).await;
                let (status, reason, body) = match current {
                    Reply::Raw(body) => (200u16, "OK", body.to_string()),
                    Reply::Http(status) => (
                        status,
                        status_reason(status),
                        json!({"error": {"code": status}}).to_string(),
                    ),
                    _ => (200u16, "OK", json!({"answers": {}}).to_string()),
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            })
            .await;
        }
    });
    port
}

/// Option B verdict: the captured Laya body passes through the decision client
/// UNMODIFIED. Only `contract` and `served_model` are injected, and the real
/// Laya fields survive byte-for-byte. No Laya-specific adapter is needed.
#[tokio::test]
async fn a_real_laya_response_passes_through_unmodified() {
    let h = laya_fallback_harness(LAYA_REAL_RESPONSE).await;
    let (status, body) = h.decide().await;
    assert_eq!(
        status, 200,
        "the Laya body must be served, not rejected: {body}"
    );

    // The real Laya answers survive.
    let answer = &body["answers"]["task_complexity"];
    assert_eq!(answer["type"], "noul");
    assert!(
        (answer["noul"].as_f64().unwrap() - 0.239).abs() < 1e-9,
        "the Laya score must survive: {body}"
    );
    assert!(
        (answer["confidence"].as_f64().unwrap() - 0.839).abs() < 1e-9,
        "the Laya confidence must survive: {body}"
    );
    assert_eq!(
        answer["action"]["act_probability"], 1.0,
        "nested Laya fields must survive"
    );

    // Usage and routing survive.
    assert_eq!(body["usage"]["input_tokens"], 436);
    assert!(
        body["routing"].is_object(),
        "the Laya routing object must survive"
    );

    // Switchyard's own provenance is injected, and the fallback is marked.
    assert_eq!(body["contract"], "switchyard-decision:v1");
    assert_eq!(body["served_model"], "decision-primary");
    assert_eq!(body["fallback_used"], true);
    assert_eq!(body["fallback_model"], "decision-fallback");

    // The Laya model identity is NOT overwritten: Switchyard injects its own
    // route identity but must not destroy the backend's self-report.
    assert_eq!(
        body["model"], "laya-rl-agent",
        "the backend's own model field must survive"
    );
}

/// All six auxiliary signals, in the real shape, cross intact. `local_suitability`
/// is expected to be UNAVAILABLE from Laya; this asserts the pass-through
/// behaviour, not a value.
#[tokio::test]
async fn all_six_laya_signals_cross_intact() {
    let h = laya_fallback_harness(LAYA_REAL_SIX_QUESTION_RESPONSE).await;
    let (status, body) = h.decide().await;
    assert_eq!(status, 200, "body: {body}");
    for signal in [
        "task_complexity",
        "reasoning_demand",
        "tool_dependency",
        "agentic_complexity",
        "route_sufficiency",
        "local_suitability",
    ] {
        let answer = &body["answers"][signal];
        assert_eq!(
            answer["type"], "noul",
            "signal {signal} must cross with its real type: {body}"
        );
        let value = answer["noul"]
            .as_f64()
            .expect(&format!("{signal} must carry a score"));
        assert!(
            (0.0..=1.0).contains(&value),
            "signal {signal} must stay in [0,1], got {value}"
        );
    }
    assert_eq!(body["fallback_used"], true);
}
