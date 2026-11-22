// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A route that narrows which evidence may spend its terminal leg.
//!
//! `smart-judge` runs inside the owner's free Gemma allowance and carries a metered Luna NT
//! leg that exists for one situation: both Gemma primaries are genuinely out of allowance.
//! The historical rule is broader — it also escalates on a 5xx, a transport fault, a timeout
//! or a rejected credential — which is the right answer for a provider-neutral pool and the
//! wrong one here: those say the deployment could not reach the pool, not that its allowance
//! ran out.
//!
//! Each test names the observation that rules out the alternative: how many times each
//! upstream was contacted and for which model. The control at the end runs the *same*
//! failures through a route that declares no policy, so a passing suite cannot be explained
//! by the failures having stopped crossing at all.

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

/// A normal answer, as a script: the terminal leg must serve when it is crossed.
fn serves(model: &str) -> Option<(u16, String)> {
    Some(success(model))
}

/// A per-minute throttle on one model: a quota fact the policy accepts, and one that does
/// not stop the candidate walk, so both primaries are actually attempted.
fn throttled(_model: &str) -> Option<(u16, String)> {
    Some((
        429,
        json!({"error": {"code": 429, "message": "Too many requests"}}).to_string(),
    ))
}

/// A Google daily quota stated per model, which is what the free suite really returns.
fn daily_quota(_model: &str) -> Option<(u16, String)> {
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
                    "quotaDimensions": {"location": "global", "model": "gemma-4-26b"},
                    "quotaValue": "14400"
                }]
            }]
        }})
        .to_string(),
    ))
}

/// The provider is down. Not a statement about the allowance.
fn unavailable(_model: &str) -> Option<(u16, String)> {
    Some((
        503,
        json!({"error": {"code": 503, "message": "The model is overloaded"}}).to_string(),
    ))
}

/// The credential is rejected. The allowance may be untouched.
fn rejected(_model: &str) -> Option<(u16, String)> {
    Some((
        401,
        json!({"error": {"code": 401, "message": "API key not valid"}}).to_string(),
    ))
}

/// The account cannot pay. A balance fact, never an allowance fact.
fn payment_required(_model: &str) -> Option<(u16, String)> {
    Some((
        402,
        json!({"error": {"code": 402, "message": "Insufficient credit"}}).to_string(),
    ))
}

fn client(name: &str, base_url: &str) -> String {
    format!("\n[llm_clients.{name}]\nformat = \"openai_responses\"\nbase_url = \"{base_url}\"\n")
}

fn target(name: &str, client: &str, slug: &str) -> String {
    format!("\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = true\n")
}

/// The judge lane exactly as production declares it, except for the terminal policy the
/// test is about. `policy` empty means the route declares none, which is the control.
fn judge_route(policy: &str) -> String {
    format!(
        "\n[routes.smart-judge]\nid = \"smart-judge\"\ntype = \"passthrough\"\ntarget = \"gemma26\"\ncandidates = [\"gemma31\"]\nselection = \"measured\"\nterminal_fallback = [\"luna\"]\n{policy}"
    )
}

/// Two Gemma primaries on one credential and a Luna NT terminal on another. Returns the
/// primary upstream, the terminal upstream, and the deployment source.
fn judge(
    pool_script: Script,
    terminal_script: Script,
    policy: &str,
) -> (Upstream, Upstream, String) {
    let pool = Upstream::new(pool_script);
    let terminal = Upstream::new(terminal_script);
    let pool_url = pool.clone().server();
    let terminal_url = terminal.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}",
        client("google", &pool_url),
        client("neutral", &terminal_url),
        target("gemma26", "google", "google/gemma-4-26b-a4b-it"),
        target("gemma31", "google", "google/gemma-4-31b-it"),
        target("luna", "neutral", "openai/gpt-6-luna-nt"),
        judge_route(policy),
    );
    (pool, terminal, config)
}

const QUOTA_ONLY: &str = "terminal_policy = \"quota_only\"\n";

/// The distinct models an upstream was asked for, so a test states coverage rather than a
/// retry count: the retry layer may repeat one leg, which is not a second primary.
async fn contacted(upstream: &Upstream) -> Vec<String> {
    let mut seen = upstream.models().await;
    seen.sort();
    seen.dedup();
    seen
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

async fn call(runner: Runner) -> (StatusCode, Value) {
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
                    json!({"model": "smart-judge", "input": "judge this"}).to_string(),
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

/// Both Gemma legs throttled is the one situation Luna NT exists for.
#[tokio::test]
async fn both_primaries_throttled_spends_the_terminal_leg() {
    let (pool, terminal, config) = judge(throttled, serves, QUOTA_ONLY);
    let (status, body) = call(runner(&config)).await;

    assert_eq!(status, StatusCode::OK, "the terminal leg answered: {body}");
    assert_eq!(body["model"], "openai/gpt-6-luna-nt");
    assert_eq!(
        contacted(&pool).await,
        vec!["google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"],
        "both primaries were really asked before the allowance claim was made"
    );
    assert_eq!(terminal.calls(), 1);
}

/// A Google daily quota is the same claim as a throttle: the allowance is spent.
#[tokio::test]
async fn both_primaries_out_of_daily_quota_spends_the_terminal_leg() {
    let (pool, terminal, config) = judge(daily_quota, serves, QUOTA_ONLY);
    let (status, body) = call(runner(&config)).await;

    assert_eq!(status, StatusCode::OK, "the terminal leg answered: {body}");
    assert_eq!(body["model"], "openai/gpt-6-luna-nt");
    assert_eq!(terminal.calls(), 1);
    assert_eq!(pool.calls(), 2, "each primary's own budget is judged");
}

/// The policy's discriminator: an outage is not an exhausted allowance.
#[tokio::test]
async fn both_primaries_unavailable_do_not_spend_the_terminal_leg() {
    let (pool, terminal, config) = judge(unavailable, serves, QUOTA_ONLY);
    let (status, _) = call(runner(&config)).await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(terminal.calls(), 0, "a 5xx is not a spent allowance");
    assert_eq!(
        contacted(&pool).await,
        vec!["google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"],
        "both primaries were tried before the refusal"
    );
}

/// One leg out of quota and the other down is not evidence that the allowance ran out.
#[tokio::test]
async fn an_outage_beside_a_throttle_does_not_spend_the_terminal_leg() {
    let script: Script = |model| {
        if model.contains("26b") {
            unavailable(model)
        } else {
            throttled(model)
        }
    };
    let (pool, terminal, config) = judge(script, serves, QUOTA_ONLY);
    let (status, _) = call(runner(&config)).await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(terminal.calls(), 0);
    assert_eq!(
        contacted(&pool).await,
        vec!["google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"],
        "the outage and the throttle were both observed"
    );
}

/// A rejected credential leaves the allowance untouched, so it must not read as spent.
#[tokio::test]
async fn a_rejected_credential_does_not_spend_the_terminal_leg() {
    let (_pool, terminal, config) = judge(rejected, serves, QUOTA_ONLY);
    let (status, _) = call(runner(&config)).await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(terminal.calls(), 0);
}

/// An account that cannot pay is a balance fact, not an allowance fact.
#[tokio::test]
async fn a_payment_required_does_not_spend_the_terminal_leg() {
    let (_pool, terminal, config) = judge(payment_required, serves, QUOTA_ONLY);
    let (status, _) = call(runner(&config)).await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(terminal.calls(), 0);
}

/// A healthy primary is never replaced, whichever policy the route declares.
#[tokio::test]
async fn a_healthy_primary_never_spends_the_terminal_leg() {
    let (pool, terminal, config) = judge(|model| Some(success(model)), serves, QUOTA_ONLY);
    let (status, body) = call(runner(&config)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "google/gemma-4-26b-a4b-it");
    assert_eq!(pool.calls(), 1);
    assert_eq!(terminal.calls(), 0);
}

/// The control: the same outage on a route that declares no policy still crosses, so the
/// refusals above are the policy's effect and not a harness that cannot cross at all.
#[tokio::test]
async fn the_same_outage_still_crosses_without_a_terminal_policy() {
    let (pool, terminal, config) = judge(unavailable, serves, "");
    let (status, body) = call(runner(&config)).await;

    assert_eq!(status, StatusCode::OK, "the default rule crossed: {body}");
    assert_eq!(body["model"], "openai/gpt-6-luna-nt");
    assert_eq!(
        contacted(&pool).await,
        vec!["google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"]
    );
    assert_eq!(terminal.calls(), 1);
}
