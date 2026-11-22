// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The terminal leg of a route.
//!
//! `smart-nt` is a free pool of Google legs with one provider-neutral terminal leg behind
//! it. The terminal leg must behave like a last resort and not like a low-ranked peer:
//!
//! * it is never ranked, so a busy primary pool cannot leak requests onto it;
//! * it is reached only when the primary pool's unavailability is *provable* — every
//!   capable leg blocked, or every dispatched capable leg out of capacity with no
//!   unattempted leg left unexplained;
//! * a failure that is an answer about the request (a policy refusal, a malformed
//!   request, an empty answer) never authorizes it, because retrying elsewhere would
//!   either evade the refusal or fail identically at a price;
//! * the declared configuration is checked at load, so a terminal target that is also a
//!   candidate, unknown, held or duplicated is a startup error rather than a silent
//!   extra candidate.
//!
//! Each test names the observation that rules out the alternative: which upstream was
//! contacted, how many times, and for which model.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// Chooses the scripted answer for one model. `None` serves the normal success body.
type Script = fn(&str) -> Option<(u16, String)>;

/// One provider backend. It records every model it was asked for, so a test can say which
/// credential was actually spent instead of inferring it from the response.
#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    seen_models: Arc<Mutex<Vec<String>>>,
    script: Script,
}

impl Upstream {
    fn new(script: Script) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            seen_models: Arc::new(Mutex::new(Vec::new())),
            script,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn models(&self) -> Vec<String> {
        self.seen_models.lock().await.clone()
    }

    fn server(self) -> String {
        let calls = Arc::clone(&self.calls);
        let seen = Arc::clone(&self.seen_models);
        let script = self.script;
        let runtime = tokio::runtime::Handle::current();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let Ok(buffer) = read_http(&mut stream) else {
                    continue;
                };
                let split = buffer
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .unwrap_or(buffer.len());
                let raw = String::from_utf8_lossy(&buffer[split.min(buffer.len())..]).to_string();
                let body: Value = serde_json::from_str(raw.trim()).unwrap_or(Value::Null);
                let model = body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                runtime.block_on(async { seen.lock().await.push(model.clone()) });
                calls.fetch_add(1, Ordering::SeqCst);
                let (status, payload) = script(&model).unwrap_or_else(|| success(&model));
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = std::io::Write::write_all(&mut stream, response.as_bytes());
                let _ = std::io::Write::flush(&mut stream);
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

fn success(model: &str) -> (u16, String) {
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

/// The account-wide free-model daily allowance: every free slug on this credential is spent
/// until midnight, so no peer on it can serve either.
fn account_wide_free_cap(_model: &str) -> Option<(u16, String)> {
    Some((
        429,
        json!({"error": {
            "message": "Rate limit exceeded: free-models-per-day",
            "code": 429,
            "metadata": {"limit_source": "openrouter_free_tier_daily", "limit": 1000}
        }})
        .to_string(),
    ))
}

/// A Google daily quota scoped by the body to ONE model inside the project.
fn model_scoped_daily_quota(model: &str) -> Option<(u16, String)> {
    if !model.contains("lite") {
        // The sibling has its own per-model budget on the same credential.
        return None;
    }
    Some((
        429,
        json!({"error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    "quotaDimensions": {"location": "global", "model": model},
                    "quotaValue": "20"
                }]
            }]
        }})
        .to_string(),
    ))
}

/// A policy refusal: an answer about the request, never a capacity fact.
fn permission_denied(_model: &str) -> Option<(u16, String)> {
    Some((
        403,
        json!({"error": {
            "code": 403,
            "message": "The caller does not have permission",
            "status": "PERMISSION_DENIED"
        }})
        .to_string(),
    ))
}

fn client(name: &str, base_url: &str) -> String {
    format!("\n[llm_clients.{name}]\nformat = \"openai_responses\"\nbase_url = \"{base_url}\"\n")
}

fn target(name: &str, client: &str, slug: &str) -> String {
    format!("\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = true\n")
}

fn target_with_window(name: &str, client: &str, slug: &str, window: u64) -> String {
    format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = true\ncontext_window = {window}\n"
    )
}

/// A declared-order route over two legs on one credential, with a terminal leg on another.
fn pool_with_terminal(primary: &str, sibling: &str, terminal: &str) -> String {
    format!(
        "\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"{primary}\"\ncandidates = [\"{sibling}\"]\nterminal_fallback = [\"{terminal}\"]\n"
    )
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

async fn call(runner: Runner, input: &str) -> (StatusCode, Value) {
    let router = switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    );
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "smart-nt", "input": input}).to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// Two credentialed backends: the pool's credential and the terminal leg's credential.
/// Returns the primary pool upstreams (one credential) and the terminal upstream.
fn pool_and_terminal(
    pool_script: Script,
    terminal_script: Script,
) -> (Upstream, Upstream, String) {
    let pool = Upstream::new(pool_script);
    let terminal = Upstream::new(terminal_script);
    let pool_url = pool.clone().server();
    let terminal_url = terminal.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}",
        client("google", &pool_url),
        client("neutral", &terminal_url),
        target("p1", "google", "aaa/gemini-lite"),
        target("p2", "google", "bbb/gemini-flash"),
        target("last", "neutral", "zzz/neutral-nt"),
        pool_with_terminal("p1", "p2", "last"),
    );
    (pool, terminal, config)
}

/// An account-wide cap on the pool's credential authorizes the terminal leg, and the
/// sibling on that same credential is not contacted: the failure already said the whole
/// credential is spent, so calling `p2` could only repeat it.
#[tokio::test]
async fn an_exhausted_pool_credential_serves_from_the_terminal_leg() {
    let (pool, terminal, config) = pool_and_terminal(account_wide_free_cap, |model| {
        Some(success(model))
    });
    let (status, body) = call(runner(&config), "hi").await;

    assert_eq!(status, StatusCode::OK, "terminal leg answered: {body}");
    assert_eq!(body["model"], "zzz/neutral-nt");
    assert_eq!(pool.calls(), 1, "the spent credential is not retried");
    assert_eq!(pool.models().await, vec!["aaa/gemini-lite"]);
    assert_eq!(terminal.models().await, vec!["zzz/neutral-nt"]);
}

/// A policy refusal is an answer about the request. It must not buy a terminal call.
#[tokio::test]
async fn a_policy_refusal_never_reaches_the_terminal_leg() {
    let (pool, terminal, config) =
        pool_and_terminal(permission_denied, |model| Some(success(model)));
    let (status, _) = call(runner(&config), "hi").await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(terminal.calls(), 0, "the refusal is not retried on a paid leg");
    assert_eq!(pool.calls(), 1);
}

/// A quota scoped to one model leaves its sibling servable, so the pool is not exhausted
/// and the terminal leg stays out of the request.
#[tokio::test]
async fn a_model_scoped_quota_is_served_by_the_sibling() {
    let (pool, terminal, config) = pool_and_terminal(model_scoped_daily_quota, |model| {
        Some(success(model))
    });
    let (status, body) = call(runner(&config), "hi").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "bbb/gemini-flash");
    assert_eq!(terminal.calls(), 0);
    assert_eq!(pool.models().await, vec!["aaa/gemini-lite", "bbb/gemini-flash"]);
}

/// A healthy primary pool never reaches the terminal leg. This is the property that keeps
/// a last resort from becoming a second candidate.
#[tokio::test]
async fn a_healthy_primary_pool_never_touches_the_terminal_leg() {
    let (pool, terminal, config) =
        pool_and_terminal(|model| Some(success(model)), |model| Some(success(model)));
    let (status, body) = call(runner(&config), "hi").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "aaa/gemini-lite");
    assert_eq!(terminal.calls(), 0);
    assert_eq!(pool.calls(), 1);
}

/// The terminal leg is subject to the same request-fit gates as a candidate. A request its
/// window cannot hold is refused rather than sent, even though the pool is exhausted.
#[tokio::test]
async fn a_request_the_terminal_leg_cannot_hold_is_not_sent_to_it() {
    let pool = Upstream::new(account_wide_free_cap);
    let terminal = Upstream::new(|model| Some(success(model)));
    let pool_url = pool.clone().server();
    let terminal_url = terminal.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}",
        client("google", &pool_url),
        client("neutral", &terminal_url),
        target("p1", "google", "aaa/gemini-lite"),
        target("p2", "google", "bbb/gemini-flash"),
        target_with_window("last", "neutral", "zzz/neutral-nt", 16),
        pool_with_terminal("p1", "p2", "last"),
    );

    let (status, _) = call(runner(&config), &"x".repeat(2_000)).await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(terminal.calls(), 0, "an unfit request is not dispatched");
    assert_eq!(pool.calls(), 1);
}

/// The terminal leg must not be rankable. Naming it as a candidate too is ambiguous — it
/// would serve whenever the pool was merely busy — so the deployment refuses at load.
#[tokio::test]
async fn a_terminal_target_cannot_also_be_a_candidate() {
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = [\"last\"]\nterminal_fallback = [\"last\"]\n",
        client("google", "http://127.0.0.1:1/v1"),
        target("p1", "google", "aaa/gemini-lite"),
        target("last", "google", "zzz/neutral-nt"),
        "",
        "",
    );
    let error = Runner::from_toml(&config).err().expect("ambiguous terminal leg is refused");
    let text = error.to_string();
    assert!(
        text.contains("also a candidate"),
        "refusal names the conflict: {text}"
    );
}

/// An unknown terminal target would make the route silently unwilling to use its last
/// resort, so it is a startup error.
#[tokio::test]
async fn an_unknown_terminal_target_is_refused() {
    let config = format!(
        "schema_version = 1\n{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = []\nterminal_fallback = [\"ghost\"]\n",
        client("google", "http://127.0.0.1:1/v1"),
        target("p1", "google", "aaa/gemini-lite"),
    );
    let error = Runner::from_toml(&config).err().expect("unknown terminal target is refused");
    assert!(
        error.to_string().contains("unknown target ghost"),
        "{error}"
    );
}

/// An empty list reads as "a terminal leg exists" while naming none.
#[tokio::test]
async fn an_empty_terminal_list_is_refused() {
    let config = format!(
        "schema_version = 1\n{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = []\nterminal_fallback = []\n",
        client("google", "http://127.0.0.1:1/v1"),
        target("p1", "google", "aaa/gemini-lite"),
    );
    let error = Runner::from_toml(&config).err().expect("empty terminal list is refused");
    assert!(error.to_string().contains("names no target"), "{error}");
}

/// The decision record separates "served by a peer" from "served by the last resort": the
/// declared terminal leg, whether it was crossed, and why.
#[tokio::test]
async fn the_routing_log_records_the_terminal_crossing() {
    let (pool, terminal, config) = pool_and_terminal(account_wide_free_cap, |model| {
        Some(success(model))
    });
    let directory = tempfile::tempdir().expect("tempdir");
    let log_path = directory.path().join("routing.jsonl");
    let decision_path = directory.path().join("routing.decisions.jsonl");
    let state = switchyard_server::ServerState::from_runner(runner(&config))
        .expect("state builds")
        .with_routing_log(&log_path)
        .expect("routing log");
    let router = switchyard_server::build_switchyard_router(state);
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "smart-nt", "input": "hi"}).to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");

    let text = std::fs::read_to_string(&decision_path).expect("decision log written");
    let record: Value = serde_json::from_str(text.lines().last().expect("one record"))
        .expect("record is json");
    assert_eq!(record["terminal"], json!(["zzz/neutral-nt"]));
    assert_eq!(record["terminal_crossed"], json!("primary_pool_exhausted"));
    assert_eq!(record["served"], json!("zzz/neutral-nt"));
    assert_eq!(pool.calls(), 1);
    assert_eq!(terminal.calls(), 1);
}
