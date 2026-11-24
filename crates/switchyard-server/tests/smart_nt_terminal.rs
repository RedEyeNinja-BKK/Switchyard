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
    call_body(runner, json!({"model": "smart-nt", "input": input})).await
}

/// Sends an arbitrary request body, so a test can add the fields the real caller sends.
async fn call_body(runner: Runner, body: Value) -> (StatusCode, Value) {
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
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// Two credentialed backends: the pool's credential and the terminal leg's credential.
/// Returns the primary pool upstreams (one credential) and the terminal upstream.
fn pool_and_terminal(pool_script: Script, terminal_script: Script) -> (Upstream, Upstream, String) {
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
    let (pool, terminal, config) =
        pool_and_terminal(account_wide_free_cap, |model| Some(success(model)));
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
    assert_eq!(
        terminal.calls(),
        0,
        "the refusal is not retried on a paid leg"
    );
    assert_eq!(pool.calls(), 1);
}

/// A quota scoped to one model leaves its sibling servable, so the pool is not exhausted
/// and the terminal leg stays out of the request.
#[tokio::test]
async fn a_model_scoped_quota_is_served_by_the_sibling() {
    let (pool, terminal, config) =
        pool_and_terminal(model_scoped_daily_quota, |model| Some(success(model)));
    let (status, body) = call(runner(&config), "hi").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "bbb/gemini-flash");
    assert_eq!(terminal.calls(), 0);
    assert_eq!(
        pool.models().await,
        vec!["aaa/gemini-lite", "bbb/gemini-flash"]
    );
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
    let error = Runner::from_toml(&config)
        .err()
        .expect("ambiguous terminal leg is refused");
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
    let error = Runner::from_toml(&config)
        .err()
        .expect("unknown terminal target is refused");
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
    let error = Runner::from_toml(&config)
        .err()
        .expect("empty terminal list is refused");
    assert!(error.to_string().contains("names no target"), "{error}");
}

/// The decision record separates "served by a peer" from "served by the last resort": the
/// declared terminal leg, whether it was crossed, and why.
#[tokio::test]
async fn the_routing_log_records_the_terminal_crossing() {
    let (pool, terminal, config) =
        pool_and_terminal(account_wide_free_cap, |model| Some(success(model)));
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
    let record: Value =
        serde_json::from_str(text.lines().last().expect("one record")).expect("record is json");
    assert_eq!(record["terminal"], json!(["zzz/neutral-nt"]));
    assert_eq!(record["terminal_crossed"], json!("primary_pool_exhausted"));
    assert_eq!(record["served"], json!("zzz/neutral-nt"));
    assert_eq!(pool.calls(), 1);
    assert_eq!(terminal.calls(), 1);
}

// ---------------------------------------------------------------------------
// Ordered tiers.
//
// `smart-judge` declares more than one last resort: a free Gemma pool, then a
// provider-neutral NT leg, then a local one. Those are not peers. Tier n + 1 exists because
// tier n was proved out, and the declaration order is the escalation order. The tests below
// name the observation that rules out the alternative for each boundary.
// ---------------------------------------------------------------------------

/// A backend that answers "the service is unavailable" - capacity evidence for an
/// `availability` route and evidence of nothing for a quota-only one.
fn upstream_unavailable(_model: &str) -> Option<(u16, String)> {
    Some((
        503,
        json!({"error": {
            "code": 503,
            "message": "The service is currently unavailable",
            "status": "UNAVAILABLE"
        }})
        .to_string(),
    ))
}

/// Renders declared tiers of TARGET NAMES as the manifest form. Each tier is written with
/// brackets so the declaration is unambiguously ordered, even when it has one member.
fn rendered_tiers(tiers: &[Vec<String>]) -> String {
    let body = tiers
        .iter()
        .map(|tier| {
            let names = tier
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{names}]")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

/// A pool of two legs on one credential, plus one independent backend per terminal tier.
///
/// `tier_scripts[i]` answers for every target in tier `i`; `tiers[i][j]` is the slug the
/// target reports, so an assertion names the tier that answered rather than a position.
fn cascade(
    pool_script: Script,
    tier_scripts: &[Script],
    tiers: &[&[&str]],
    policy: &str,
) -> (Upstream, Vec<Upstream>, String) {
    let declared: Vec<Vec<String>> = (0..tiers.len())
        .map(|index| {
            (0..tiers[index].len())
                .map(|position| format!("t{index}{position}"))
                .collect()
        })
        .collect();
    assert_eq!(
        tier_scripts.len(),
        tiers.len(),
        "every declared tier needs a backend"
    );
    let pool = Upstream::new(pool_script);
    let pool_url = pool.clone().server();
    let mut config = format!(
        "schema_version = 1\n{}{}{}",
        client("google", &pool_url),
        target("p1", "google", "aaa/gemini-lite"),
        target("p2", "google", "bbb/gemini-flash"),
    );
    let mut backends = Vec::new();
    for (index, script) in tier_scripts.iter().copied().enumerate() {
        let backend = Upstream::new(script);
        let url = backend.clone().server();
        let name = format!("tier{index}");
        config.push_str(&client(&name, &url));
        for (position, slug) in tiers[index].iter().enumerate() {
            let target_name = format!("t{index}{position}");
            config.push_str(&target(&target_name, &name, &format!("tier{index}/{slug}")));
        }
        backends.push(backend);
    }
    config.push_str(&format!(
        "\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = [\"p2\"]\nterminal_fallback = {}\nterminal_policy = \"{policy}\"\n",
        rendered_tiers(&declared)
    ));
    (pool, backends, config)
}

/// Runs one request through the decision log and returns the record the route emitted.
async fn record(config: &str) -> Value {
    record_with(config, json!({"model": "smart-nt", "input": "hi"})).await
}

/// Runs one request body through the decision log and returns the record the route emitted.
async fn record_with(config: &str, body: Value) -> Value {
    let directory = tempfile::tempdir().expect("tempdir");
    let log_path = directory.path().join("routing.jsonl");
    let decision_path = directory.path().join("routing.decisions.jsonl");
    let state = switchyard_server::ServerState::from_runner(runner(config))
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
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let _ = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    let text = std::fs::read_to_string(&decision_path).expect("decision log written");
    serde_json::from_str(text.lines().last().expect("one record")).expect("record is json")
}

/// The historical flat declaration is ONE tier whose members are peers. Written any other
/// way it would become an escalation, which is the precise meaning this field must not drift
/// into: two peers of one leg are ranked by the route's own policy, and the record says tier
/// one answered because both names were consulted inside the same escalation step.
#[tokio::test]
async fn a_flat_terminal_declaration_is_one_tier_of_peers() {
    let pool = Upstream::new(account_wide_free_cap);
    let pool_url = pool.clone().server();
    // Two peers of one declared leg. They sit on separate credentials on purpose: a 429
    // marks the credential that answered it spent, so putting both peers on one backend
    // would test that rule instead of the one under test here.
    let first = Upstream::new(account_wide_free_cap);
    let first_url = first.clone().server();
    let second = Upstream::new(|model| Some(success(model)));
    let second_url = second.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = [\"p2\"]\nterminal_fallback = [\"first\", \"second\"]\n",
        client("google", &pool_url),
        client("neutral1", &first_url),
        client("neutral2", &second_url),
        target("p1", "google", "aaa/gemini-lite"),
        target("p2", "google", "bbb/gemini-flash"),
        target("first", "neutral1", "peer/first"),
        target("second", "neutral2", "peer/second"),
    );

    let decision = record(&config).await;

    assert_eq!(decision["served"], json!("peer/second"));
    assert_eq!(
        decision["terminal_tier"],
        json!(1),
        "both names were consulted inside one escalation step: {decision}"
    );
    assert!(
        decision.get("terminal_tiers").is_none(),
        "a one-tier declaration has no order to report: {decision}"
    );
    assert_eq!(first.models().await, vec!["peer/first"]);
    assert_eq!(second.models().await, vec!["peer/second"]);
}

/// Tier two is reached only after tier one is proved out, and the record says so.
#[tokio::test]
async fn the_second_tier_is_consulted_only_after_the_first_is_proved_out() {
    let (pool, backends, config) = cascade(
        account_wide_free_cap,
        &[account_wide_free_cap, |model| Some(success(model))],
        &[&["first"], &["second"]],
        "availability",
    );

    let decision = record(&config).await;

    assert_eq!(decision["served"], json!("tier1/second"));
    assert_eq!(decision["terminal_tier"], json!(2));
    assert_eq!(
        decision["terminal_tiers"],
        json!([["tier0/first"], ["tier1/second"]])
    );
    assert_eq!(backends[0].models().await, vec!["tier0/first"]);
    assert_eq!(backends[1].models().await, vec!["tier1/second"]);
    assert_eq!(pool.calls(), 1);
}

/// A healthy first tier answers, and the later ones are never contacted. This is the
/// property that keeps a declared order from becoming a preference list.
#[tokio::test]
async fn a_healthy_first_tier_never_reaches_the_second() {
    let (_pool, backends, config) = cascade(
        account_wide_free_cap,
        &[|model| Some(success(model)), |model| Some(success(model))],
        &[&["first"], &["second"]],
        "availability",
    );

    let decision = record(&config).await;

    assert_eq!(decision["served"], json!("tier0/first"));
    assert_eq!(decision["terminal_tier"], json!(1));
    assert_eq!(backends[0].calls(), 1);
    assert_eq!(backends[1].calls(), 0, "the later tier is never touched");
}

/// A refusal in tier one is an answer about the request. It does not prove tier one is out
/// of capacity, so it must not advance the cascade - the same rule that governs the primary
/// pool, applied at each boundary rather than only at the first.
#[tokio::test]
async fn a_policy_refusal_in_the_first_tier_never_reaches_the_second() {
    let (_pool, backends, config) = cascade(
        account_wide_free_cap,
        &[permission_denied, |model| Some(success(model))],
        &[&["first"], &["second"]],
        "availability",
    );

    let (status, _) = call(runner(&config), "hi").await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(backends[0].calls(), 1);
    assert_eq!(
        backends[1].calls(),
        0,
        "a policy refusal does not buy the next leg"
    );
}

/// Declared order is the escalation order, not a ranking. Swapping the same two targets
/// changes which one is consulted first, which is only true if no selector is choosing.
#[tokio::test]
async fn the_declared_order_is_the_escalation_order_not_a_ranking() {
    let succeed = |model: &str| Some(success(model));

    let (_p1, first_order, config_a) = cascade(
        account_wide_free_cap,
        &[succeed, succeed],
        &[&["alpha"], &["beta"]],
        "availability",
    );
    assert_eq!(record(&config_a).await["served"], json!("tier0/alpha"));
    assert_eq!(first_order[1].calls(), 0);

    let (_p2, second_order, config_b) = cascade(
        account_wide_free_cap,
        &[succeed, succeed],
        &[&["beta"], &["alpha"]],
        "availability",
    );
    assert_eq!(record(&config_b).await["served"], json!("tier0/beta"));
    assert_eq!(
        second_order[1].calls(),
        0,
        "the cheaper or faster leg does not jump the declared order"
    );
}

/// Every earlier tier is out of capacity, so the last one answers. A 5xx counts for an
/// `availability` route, which is the difference between this declaration and a quota-only
/// one: without the wider policy the cascade would stop at tier two.
#[tokio::test]
async fn an_unavailable_tier_advances_an_availability_route() {
    let (_pool, backends, config) = cascade(
        account_wide_free_cap,
        &[account_wide_free_cap, upstream_unavailable, |model| {
            Some(success(model))
        }],
        &[&["gemma"], &["luna"], &["local"]],
        "availability",
    );

    let decision = record(&config).await;

    assert_eq!(decision["served"], json!("tier2/local"));
    assert_eq!(decision["terminal_tier"], json!(3));
    assert_eq!(backends[0].calls(), 1);
    assert_eq!(backends[1].calls(), 1);
    assert_eq!(backends[2].calls(), 1);
}

/// The control for the test above. The same failures under the historical policy stop at
/// tier two, because a 5xx is not quota evidence. If this control ever passes, the wider
/// policy is not what advanced the cascade and the test above proves nothing.
#[tokio::test]
async fn a_quota_only_route_does_not_advance_on_a_5xx() {
    let (_pool, backends, config) = cascade(
        account_wide_free_cap,
        &[account_wide_free_cap, upstream_unavailable, |model| {
            Some(success(model))
        }],
        &[&["gemma"], &["luna"], &["local"]],
        "quota_only",
    );

    let (status, _) = call(runner(&config), "hi").await;

    assert_ne!(status, StatusCode::OK);
    assert_eq!(backends[1].calls(), 1);
    assert_eq!(
        backends[2].calls(),
        0,
        "a 5xx is not quota evidence for a quota-only route"
    );
}

/// A target named in two tiers would be tried twice under two different exhaustion claims,
/// so the declaration is refused at load.
#[tokio::test]
async fn a_target_repeated_across_tiers_is_refused() {
    let config = format!(
        "schema_version = 1\n{}{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = []\nterminal_fallback = [[\"last\"], [\"last\"]]\n",
        client("google", "http://127.0.0.1:1/v1"),
        target("p1", "google", "aaa/gemini-lite"),
        target("last", "google", "zzz/neutral-nt"),
    );
    let error = Runner::from_toml(&config)
        .err()
        .expect("a repeated terminal target is refused");
    assert!(
        error.to_string().contains("more than once"),
        "refusal names the repetition: {error}"
    );
}

/// An empty tier would escalate past a leg that was never consulted, so it is a startup
/// error rather than a silent skip.
#[tokio::test]
async fn an_empty_tier_is_refused() {
    let config = format!(
        "schema_version = 1\n{}{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = []\nterminal_fallback = [[\"last\"], []]\n",
        client("google", "http://127.0.0.1:1/v1"),
        target("p1", "google", "aaa/gemini-lite"),
        target("last", "google", "zzz/neutral-nt"),
    );
    let error = Runner::from_toml(&config)
        .err()
        .expect("an empty tier is refused");
    assert!(
        error.to_string().contains("tier 2 names no target"),
        "refusal names the empty tier: {error}"
    );
}

/// The judge's real turns carry tools, and a tool-bearing request must still reach the later
/// tiers. This is the contract that decides whether the third tier is usable at all: a leg
/// excluded from every tool-bearing turn would answer nothing that matters. The target here
/// declares no tool-calling fact, which the gate reads as uncharacterised rather than
/// incapable - the same rule that lets a newly added leg serve before it is characterised.
#[tokio::test]
async fn a_tool_bearing_judge_turn_reaches_the_later_tier() {
    let (_pool, backends, config) = cascade(
        account_wide_free_cap,
        &[account_wide_free_cap, |model| Some(success(model))],
        &[&["luna"], &["local"]],
        "availability",
    );

    let (status, body) = call_body(
        runner(&config),
        json!({
            "model": "smart-nt",
            "input": "verdict please",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {"type": "object", "properties": {}}
            }]
        }),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::OK,
        "the tool-bearing turn was served: {body}"
    );
    assert_eq!(body["model"], "tier1/local");
    assert_eq!(backends[1].calls(), 1);
}

/// The paired control for the test above: a tier whose target explicitly refuses tools is not
/// dispatchable for that request, so it is never called and the cascade stops rather than
/// escalating past a leg it could not use. Without this, "the tool-bearing turn was served"
/// could be true because the tool check never ran.
#[tokio::test]
async fn an_unfit_tier_is_neither_dispatched_nor_skipped_over() {
    let pool = Upstream::new(account_wide_free_cap);
    let pool_url = pool.clone().server();
    let luna = Upstream::new(account_wide_free_cap);
    let luna_url = luna.clone().server();
    // The local leg cannot carry tools, and says so.
    let local = Upstream::new(|model| Some(success(model)));
    let local_url = local.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}{}\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\ntarget = \"p1\"\ncandidates = [\"p2\"]\nterminal_fallback = [[\"t00\"], [\"t10\"]]\nterminal_policy = \"availability\"\n",
        client("google", &pool_url),
        client("tier0", &luna_url),
        client("tier1", &local_url),
        target("p1", "google", "aaa/gemini-lite"),
        target("p2", "google", "bbb/gemini-flash"),
        target("t00", "tier0", "tier0/luna"),
        format!(
            "\n[targets.t10]\nid = \"tier1/local\"\nllm_client = \"tier1\"\ntool_calling = false\n"
        ),
    );

    let decision = record_with(
        &config,
        json!({
            "model": "smart-nt",
            "input": "verdict please",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {"type": "object", "properties": {}}
            }]
        }),
    )
    .await;

    // `calls()` alone cannot tell "never dispatched" from "dispatched and refused before the
    // wire", because the send path applies the same gate. The tier index can: a cascade that
    // entered the unfit tier reports tier 2 even though no request left the process.
    assert_eq!(decision["outcome"], json!("failed"));
    assert_eq!(
        decision["terminal_tier"],
        json!(1),
        "the cascade stopped at the tier it could use: {decision}"
    );
    assert_eq!(luna.calls(), 1, "the first tier was still consulted");
    assert_eq!(
        local.calls(),
        0,
        "a tier that cannot carry this request is not dispatched"
    );
}
