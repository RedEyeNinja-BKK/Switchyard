// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Account-scoped daily budgets, measured through real dispatch.
//!
//! An account budget is one counter shared by every model on that account. The failure mode
//! this file exists to catch is silent: a target whose lookup key does not match the key the
//! counter was filed under resolves to *no allowance*, and "no allowance" is indistinguishable
//! from "this target declares no budget". The manifest advertises a limit, the reconciler keeps
//! the counter, and every request is admitted.
//!
//! So none of these tests assert on map keys alone. Each sends real requests through the
//! router to an offline upstream and reads the counter file the reservation wrote, which is
//! the only evidence that a specific target actually charged the budget.
//!
//! Covered, in the order the owner asked for them:
//!
//! * every applicable free target is bound (four targets, four routes, one counter);
//! * sequential reservations across distinct targets share that counter;
//! * exhaustion refuses *every* affected target before it reaches the provider;
//! * independent controls: Stephen's credential, paid OpenRouter, model-scoped Google;
//! * exactly-once accounting: one dispatched request, one increment.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// One provider backend. It records every model it was asked for and how many requests it
/// received, so a test can assert what was *not* sent as precisely as what was.
#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<String>>>,
    /// Status every request is answered with, so a test can drive the retry path
    /// deterministically instead of waiting for a real provider to misbehave.
    status: Arc<AtomicU16>,
}

impl Upstream {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            seen: Arc::new(Mutex::new(Vec::new())),
            status: Arc::new(AtomicU16::new(200)),
        }
    }

    /// A provider that refuses every request with this status.
    fn failing(status: u16) -> Self {
        let upstream = Self::new();
        upstream.status.store(status, Ordering::SeqCst);
        upstream
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn models(&self) -> Vec<String> {
        self.seen.lock().await.clone()
    }

    fn url(self) -> String {
        let calls = Arc::clone(&self.calls);
        let seen = Arc::clone(&self.seen);
        let status = Arc::clone(&self.status);
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
                let code = status.load(Ordering::SeqCst);
                let payload = if (200..300).contains(&code) {
                    success(&model)
                } else {
                    json!({"error": {"code": code, "message": "upstream refused"}}).to_string()
                };
                let response = format!(
                    "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
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

fn success(model: &str) -> String {
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
    .to_string()
}

/// A fresh state directory per test, so a counter left by another test cannot be mistaken
/// for this one's spend.
fn state_dir(name: &str) -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("/var/tmp").join(format!(
        "switchyard-quota-binding-{name}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("state dir");
    dir
}

/// Every counter file in the state dir, parsed. Named rather than counted so a test can say
/// which account spent what.
fn counters(dir: &std::path::Path) -> Vec<(String, u64, u64)> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        found.push((
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string(),
            value.get("spent").and_then(Value::as_u64).unwrap_or(0),
            value.get("limit").and_then(Value::as_u64).unwrap_or(0),
        ));
    }
    found.sort();
    found
}

/// The single counter's spend, asserting there is exactly one so a test cannot pass on a
/// per-model counter that should not exist.
fn only_counter(dir: &std::path::Path) -> (String, u64, u64) {
    let all = counters(dir);
    assert_eq!(all.len(), 1, "expected exactly one counter, found {all:?}");
    all.into_iter().next().expect("one counter")
}

async fn call(config: &str, route: &str) -> (StatusCode, Value) {
    let runner = Runner::from_toml(config).expect("deployment builds");
    call_runner(runner, route).await
}

async fn call_runner(runner: Runner, route: &str) -> (StatusCode, Value) {
    call_body(runner, json!({"model": route, "input": "ping"})).await
}

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

#[tokio::test(flavor = "multi_thread")]
async fn every_free_target_shares_one_counter_and_the_budget_refuses_them_all() {
    let dir = state_dir("free-pool");
    let free = Upstream::new();
    let free_url = free.clone().url();

    // Four targets on one account, one route each so the test decides which target is
    // dispatched instead of letting a declared order hide a broken binding behind the first
    // leg. Limit 3: three requests spend it, the fourth must be refused locally.
    let mut config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n\n\
         [llm_clients.openrouter_free]\nformat = \"openai_responses\"\nbase_url = \"{free_url}\"\n\
         quota_group = \"openrouter-free-owner\"\nquota_scope = \"account\"\nquota_reset = \"utc\"\n\n",
        dir.display()
    );
    let slugs = [
        "vendor/free-a",
        "vendor/free-b",
        "vendor/free-c",
        "vendor/free-d",
    ];
    for (index, slug) in slugs.iter().enumerate() {
        config.push_str(&format!(
            "\n[targets.free-{index}]\nid = \"{slug}\"\nllm_client = \"openrouter_free\"\n\
             vision = false\nvideo = false\nfree_tier = true\ndaily_request_limit = 3\n"
        ));
        config.push_str(&format!(
            "\n[routes.free-{index}]\nid = \"free-{index}\"\ntype = \"passthrough\"\ntarget = \"free-{index}\"\n"
        ));
    }

    Runner::from_toml(&config).expect("deployment builds");

    // Three distinct targets spend the shared budget, one request each.
    for index in 0..3 {
        let (status, body) = call(&config, &format!("free-{index}")).await;
        assert_eq!(status, StatusCode::OK, "free-{index} should serve: {body}");
        assert_eq!(
            free.calls(),
            index + 1,
            "free-{index} must reach the provider exactly once"
        );
        let (_, spent, limit) = only_counter(&dir);
        assert_eq!(limit, 3);
        assert_eq!(
            spent,
            index as u64 + 1,
            "each dispatched request must charge the one account counter exactly once"
        );
    }
    assert_eq!(
        free.models().await,
        slugs[..3].to_vec(),
        "the three served requests must be three different targets"
    );

    // The fourth target was never dispatched, yet it must be refused on the same spent
    // budget. This is the assertion the broken shape fails: an unbound target dispatches.
    let before = free.calls();
    let (status, body) = call(&config, "free-3").await;
    assert_ne!(status, StatusCode::OK, "an exhausted pool must not serve");
    assert_eq!(
        free.calls(),
        before,
        "refusal must happen before egress, not as a provider 429"
    );
    assert_eq!(only_counter(&dir).1, 3, "a refusal must not spend");

    // And the targets that already served are refused too, not just the one that never ran.
    for index in 0..3 {
        let (status, _) = call(&config, &format!("free-{index}")).await;
        assert_ne!(status, StatusCode::OK, "free-{index} must be refused too");
    }
    assert_eq!(free.calls(), before, "no refusal may reach the provider");

    let text = format!("{body}");
    assert!(
        text.to_lowercase().contains("quota")
            || text.to_lowercase().contains("allowance")
            || text.to_lowercase().contains("exhaust"),
        "the refusal should name the budget, got {body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn accounting_is_exactly_once_per_dispatched_request() {
    let dir = state_dir("exactly-once");
    let free = Upstream::new();
    let free_url = free.clone().url();
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n\n\
         [llm_clients.openrouter_free]\nformat = \"openai_responses\"\nbase_url = \"{free_url}\"\n\
         quota_group = \"openrouter-free-owner\"\nquota_scope = \"account\"\nquota_reset = \"utc\"\n\n\
         [targets.free-a]\nid = \"vendor/free-a\"\nllm_client = \"openrouter_free\"\n\
         vision = false\nvideo = false\ndaily_request_limit = 50\n\n\
         [routes.free-a]\nid = \"free-a\"\ntype = \"passthrough\"\ntarget = \"free-a\"\n",
        dir.display()
    );
    Runner::from_toml(&config).expect("deployment builds");

    for expected in 1..=5u64 {
        let (status, _) = call(&config, "free-a").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(free.calls(), expected as usize);
        assert_eq!(
            only_counter(&dir).1,
            expected,
            "one request, one increment -- no double count from ranking or fallback"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn independent_credentials_are_not_metered_or_blocked_by_the_free_pool() {
    let dir = state_dir("independent");
    let free = Upstream::new();
    let stephen = Upstream::new();
    let paid = Upstream::new();
    let google = Upstream::new();

    // Stephen has its own credential and declares no group at all, so it must not be folded
    // into the owner's account counter just because it also serves :free slugs. Paid
    // OpenRouter declares a limit-less target: no counter, no refusal, ever.
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n\n\
         [llm_clients.openrouter_free]\nformat = \"openai_responses\"\nbase_url = \"{free_url}\"\n\
         quota_group = \"openrouter-free-owner\"\nquota_scope = \"account\"\nquota_reset = \"utc\"\n\n\
         [llm_clients.openrouter_stephen]\nformat = \"openai_responses\"\nbase_url = \"{stephen_url}\"\n\n\
         [llm_clients.openrouter]\nformat = \"openai_responses\"\nbase_url = \"{paid_url}\"\n\n\
         [llm_clients.google]\nformat = \"openai_responses\"\nbase_url = \"{google_url}\"\n\
         quota_group = \"google-free\"\nquota_scope = \"model\"\nquota_reset = \"pacific\"\n\n\
         [targets.free-a]\nid = \"vendor/free-a\"\nllm_client = \"openrouter_free\"\n\
         vision = false\nvideo = false\ndaily_request_limit = 2\n\n\
         [targets.stephen-free]\nid = \"vendor/stephen-free\"\nllm_client = \"openrouter_stephen\"\n\
         vision = false\nvideo = false\nfree_tier = true\n\n\
         [targets.paid]\nid = \"vendor/paid\"\nllm_client = \"openrouter\"\nvision = false\nvideo = false\n\n\
         [targets.google-a]\nid = \"models/gemini-a\"\nllm_client = \"google\"\n\
         vision = false\nvideo = false\ndaily_request_limit = 1\n\n\
         [targets.google-b]\nid = \"models/gemini-b\"\nllm_client = \"google\"\n\
         vision = false\nvideo = false\ndaily_request_limit = 1\n\n\
         [routes.free]\nid = \"free\"\ntype = \"passthrough\"\ntarget = \"free-a\"\n\n\
         [routes.stephen]\nid = \"stephen\"\ntype = \"passthrough\"\ntarget = \"stephen-free\"\n\n\
         [routes.paid]\nid = \"paid\"\ntype = \"passthrough\"\ntarget = \"paid\"\n\n\
         [routes.google-a]\nid = \"google-a\"\ntype = \"passthrough\"\ntarget = \"google-a\"\n\n\
         [routes.google-b]\nid = \"google-b\"\ntype = \"passthrough\"\ntarget = \"google-b\"\n",
        dir.display(),
        free_url = free.clone().url(),
        stephen_url = stephen.clone().url(),
        paid_url = paid.clone().url(),
        google_url = google.clone().url(),
    );
    Runner::from_toml(&config).expect("deployment builds");

    // Spend the owner's free budget exactly.
    for _ in 0..2 {
        assert_eq!(call(&config, "free").await.0, StatusCode::OK);
    }
    assert_ne!(call(&config, "free").await.0, StatusCode::OK);

    // Stephen, paid and Google are untouched by that exhaustion.
    assert_eq!(call(&config, "stephen").await.0, StatusCode::OK);
    assert_eq!(call(&config, "paid").await.0, StatusCode::OK);
    assert_eq!(stephen.calls(), 1);
    assert_eq!(paid.calls(), 1);

    // Model scope stays per model: exhausting one does not exhaust its sibling.
    assert_eq!(call(&config, "google-a").await.0, StatusCode::OK);
    assert_ne!(call(&config, "google-a").await.0, StatusCode::OK);
    assert_eq!(call(&config, "google-b").await.0, StatusCode::OK);
    assert_eq!(
        google.calls(),
        2,
        "one per model, the refusals never egress"
    );

    // Counters: one for the owner account, one per Google model. None for Stephen or paid,
    // because neither declares a budget and neither may be given one implicitly.
    let all = counters(&dir);
    let names: Vec<&str> = all.iter().map(|(name, _, _)| name.as_str()).collect();
    assert_eq!(all.len(), 3, "counters were {all:?}");
    assert!(
        names
            .iter()
            .any(|name| name.contains("openrouter-free-owner")),
        "the owner account counter must exist: {names:?}"
    );
    assert_eq!(
        names.iter().filter(|name| name.contains("gemini")).count(),
        2,
        "model scope is one counter per model: {names:?}"
    );
    assert!(
        !names.iter().any(|name| name.contains("stephen")),
        "Stephen must never acquire a counter: {names:?}"
    );
    for (name, spent, _) in &all {
        if name.contains("openrouter-free-owner") {
            assert_eq!(*spent, 2, "the owner account spent exactly its two");
        }
        if name.contains("gemini") {
            assert_eq!(*spent, 1, "each Google model spent its one");
        }
    }
}

/// Reversal: the same fixture with the targets declared in the opposite order must produce
/// the same accounting. A binding that depends on which target was visited first is the bug.
#[tokio::test(flavor = "multi_thread")]
async fn binding_does_not_depend_on_declaration_order() {
    let dir = state_dir("reversed");
    let free = Upstream::new();
    let free_url = free.clone().url();
    let slugs = ["vendor/z-last", "vendor/m-middle", "vendor/a-first"];
    let mut config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n\n\
         [llm_clients.openrouter_free]\nformat = \"openai_responses\"\nbase_url = \"{free_url}\"\n\
         quota_group = \"openrouter-free-owner\"\nquota_scope = \"account\"\nquota_reset = \"utc\"\n\n",
        dir.display()
    );
    for (index, slug) in slugs.iter().enumerate() {
        config.push_str(&format!(
            "\n[targets.t{index}]\nid = \"{slug}\"\nllm_client = \"openrouter_free\"\n\
             vision = false\nvideo = false\ndaily_request_limit = 2\n\
             \n[routes.t{index}]\nid = \"t{index}\"\ntype = \"passthrough\"\ntarget = \"t{index}\"\n"
        ));
    }
    Runner::from_toml(&config).expect("deployment builds");
    // Spend on the two targets that are NOT first in declared order.
    assert_eq!(call(&config, "t2").await.0, StatusCode::OK);
    assert_eq!(only_counter(&dir).1, 1);
    assert_eq!(call(&config, "t1").await.0, StatusCode::OK);
    assert_eq!(only_counter(&dir).1, 2);
    // Every target, including the one declared first, is now refused.
    for index in 0..3 {
        assert_ne!(
            call(&config, &format!("t{index}")).await.0,
            StatusCode::OK,
            "t{index} must be refused on the shared budget"
        );
    }
    assert_eq!(free.calls(), 2);
}

/// A free leaf must not turn one refusal into four reservations.
///
/// This is the measured defect the bounded-retry policy exists for: the retry loop
/// reserved once per attempt against the account budget, so a single client request that
/// drew a 429 charged four reservations while the routing evidence recorded one attempt.
/// Both numbers are asserted together - reservations in the counter file and requests the
/// provider actually saw - because either one alone can look correct.
///
/// The paid control matters as much as the rule: a lane with no daily allowance must keep
/// its configured retries, or this change silently removes recovery on the lanes that are
/// not metered.
#[tokio::test(flavor = "multi_thread")]
async fn a_free_leaf_spends_one_reservation_on_a_throttle_but_a_paid_lane_keeps_retries() {
    let dir = state_dir("retry-throttle");
    let throttled_free = Upstream::failing(429);
    let throttled_free_url = throttled_free.clone().url();
    let throttled_paid = Upstream::failing(429);
    let throttled_paid_url = throttled_paid.clone().url();

    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n\n\
         [llm_clients.openrouter_free]\nformat = \"openai_responses\"\n\
         base_url = \"{throttled_free_url}\"\nquota_group = \"openrouter-free-owner\"\n\
         quota_scope = \"account\"\nquota_reset = \"utc\"\nmax_retries = 2\n\n\
         [llm_clients.openrouter]\nformat = \"openai_responses\"\n\
         base_url = \"{throttled_paid_url}\"\nmax_retries = 2\n\n\
         [targets.free-throttle]\nid = \"vendor/free-a\"\nllm_client = \"openrouter_free\"\n\
         vision = false\nvideo = false\nfree_tier = true\ndaily_request_limit = 100\n\n\
         [routes.free-throttle]\nid = \"free-throttle\"\ntype = \"passthrough\"\n\
         target = \"free-throttle\"\n\n\
         [targets.paid-control]\nid = \"vendor/paid-a\"\nllm_client = \"openrouter\"\n\
         vision = false\nvideo = false\n\n\
         [routes.paid-control]\nid = \"paid-control\"\ntype = \"passthrough\"\n\
         target = \"paid-control\"\n",
        dir.display()
    );

    // One request, one dispatch, one reservation - even though the lane allows two retries.
    let (status, body) = call(&config, "free-throttle").await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    assert_eq!(
        throttled_free.calls(),
        1,
        "a 429 must not be resent to the same quota-limited target"
    );
    let (counter, spent, limit) = only_counter(&dir);
    assert_eq!(limit, 100);
    assert_eq!(
        spent, 1,
        "{counter}: one dispatched attempt must reserve exactly once"
    );
    assert_eq!(
        throttled_free.models().await,
        vec!["vendor/free-a".to_string()],
        "the provider must have seen the free target exactly once"
    );

    // The control: same status, same configured budget, no daily allowance.
    let (status, body) = call(&config, "paid-control").await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    assert_eq!(
        throttled_paid.calls(),
        3,
        "an unmetered lane must keep its configured retries"
    );
    let (_, spent, _) = only_counter(&dir);
    assert_eq!(spent, 1, "the paid lane must not touch the free counter");
}

/// A transient 5xx costs a free leaf at most one extra reservation.
///
/// The budget is not "no retry": a single retry is what recovers from a genuinely bad
/// second of upstream health. It is only the unbounded multiplication that the policy
/// refuses, and the reservation is not refunded on failure - a refused attempt still
/// reached the provider and still drew on the allowance.
#[tokio::test(flavor = "multi_thread")]
async fn a_free_leaf_retries_a_5xx_once_and_keeps_both_reservations() {
    let dir = state_dir("retry-5xx");
    let unavailable = Upstream::failing(503);
    let unavailable_url = unavailable.clone().url();

    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n\n\
         [llm_clients.openrouter_free]\nformat = \"openai_responses\"\n\
         base_url = \"{unavailable_url}\"\nquota_group = \"openrouter-free-owner\"\n\
         quota_scope = \"account\"\nquota_reset = \"utc\"\nmax_retries = 2\n\n\
         [targets.free-unavailable]\nid = \"vendor/free-b\"\nllm_client = \"openrouter_free\"\n\
         vision = false\nvideo = false\nfree_tier = true\ndaily_request_limit = 100\n\n\
         [routes.free-unavailable]\nid = \"free-unavailable\"\ntype = \"passthrough\"\n\
         target = \"free-unavailable\"\n",
        dir.display()
    );

    let (status, body) = call(&config, "free-unavailable").await;
    assert!(
        status.is_server_error(),
        "the refusal must reach the caller, got {status}: {body}"
    );
    assert_eq!(
        unavailable.calls(),
        2,
        "a 5xx gets exactly one further attempt on a metered leaf"
    );
    let (counter, spent, _) = only_counter(&dir);
    assert_eq!(
        spent, 2,
        "{counter}: both dispatched attempts reserve, and neither is refunded"
    );
}
