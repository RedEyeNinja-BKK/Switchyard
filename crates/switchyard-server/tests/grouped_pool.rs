// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A pool of declared groups: Google free and the OpenAI subscription as equal primaries.
//!
//! These tests exist because the flat peer pool cannot state this topology honestly. Under
//! one peer set the only Google leaf that declares a daily budget outranks the OpenAI leaf
//! whose allowance the deployment cannot count, so four Google leaves would win every tie and
//! the "equal primary groups" the owner asked for would be a Google-first waterfall in
//! disguise. Each test names the observation that rules that out:
//!
//! * two groups with no evidence are both served, so leaf count is not priority;
//! * a spent group steps aside for a ready one;
//! * both primary groups spent, with evidence, is what opens the local tier — and nothing
//!   else is: a refusal about the request, or an outage with no allowance evidence, stops at
//!   the pool;
//! * the metered tier behind the local one is reachable when the local capacity is gone, and
//!   the local tier's own boundary is the only place that judgment is made.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// Scripted upstream answer. `None` serves the normal success body.
type Script = fn(&str) -> Option<(u16, String)>;

/// One provider backend. It records every model it was asked for, so an assertion can say
/// which credential was actually spent instead of inferring it from the response.
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
                let (status, payload) =
                    script(&model).unwrap_or_else(|| (200, success(&model)));
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
                    payload.len()
                );
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

fn success(model: &str) -> String {
    json!({
        "id": format!("resp_{}", model.replace('/', "_")),
        "object": "response",
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message", "id": "m1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "ok"}]
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    })
    .to_string()
}

/// The account-wide free-model daily allowance: every free slug on this credential is spent
/// until midnight, so no peer on it can serve either.
fn account_wide_cap(_model: &str) -> Option<(u16, String)> {
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

/// An answer about the request, not about capacity.
fn policy_refusal(_model: &str) -> Option<(u16, String)> {
    Some((
        400,
        json!({"error": {"message": "unsupported input", "code": 400}}).to_string(),
    ))
}

/// An outage: the provider could not be reached, which says nothing about the allowance.
fn outage(_model: &str) -> Option<(u16, String)> {
    Some((
        503,
        json!({"error": {"message": "upstream unavailable", "code": 503}}).to_string(),
    ))
}

fn client(name: &str, base_url: &str) -> String {
    format!("\n[llm_clients.{name}]\nformat = \"openai_responses\"\nbase_url = \"{base_url}\"\n")
}

/// A free leaf that belongs to a declared group.
fn grouped_leaf(name: &str, client: &str, slug: &str, group: &str, daily: u64) -> String {
    format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = true\n\
         groups = [\"{group}\"]\ndaily_request_limit = {daily}\n"
    )
}

/// A terminal leaf: no group, no declared budget.
fn terminal_leaf(name: &str, client: &str, slug: &str) -> String {
    format!("\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\n")
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

/// The load error for a configuration that must be refused. `Runner` has no `Debug`, so the
/// refusal is read through its message rather than through `expect_err`.
fn load_error(source: &str) -> String {
    match Runner::from_toml(source) {
        Ok(_) => panic!("configuration unexpectedly loaded"),
        Err(error) => error.to_string(),
    }
}

async fn call(router: &Router, model: &str, input: &str) -> (StatusCode, Value) {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": model, "input": input}).to_string(),
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

/// The deployed smart-nt shape: four Google leaves and one OpenAI leaf as two equal primary
/// groups, a local NT tier behind them, and a metered NT tier behind that.
struct Fixture {
    google: Upstream,
    openai: Upstream,
    local: Upstream,
    metered: Upstream,
    router: Router,
}

fn smart_nt(
    google_script: Script,
    openai_script: Script,
    local_script: Script,
    metered_script: Script,
) -> Fixture {
    let google = Upstream::new(google_script);
    let openai = Upstream::new(openai_script);
    let local = Upstream::new(local_script);
    let metered = Upstream::new(metered_script);
    let google_url = google.clone().server();
    let openai_url = openai.clone().server();
    let local_url = local.clone().server();
    let metered_url = metered.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}{}{}{}{}{}",
        quota.path().display(),
        client("google", &google_url),
        client("openai", &openai_url),
        client("local", &local_url),
        client("metered", &metered_url),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 500),
        grouped_leaf("g2", "google", "vendor/g2", "smart-gemini-bounded", 500),
        grouped_leaf("g3", "google", "vendor/g3", "smart-gemini-bounded", 500),
        grouped_leaf("g4", "google", "vendor/g4", "smart-gemini-bounded", 500),
        format!(
            "{}{}{}",
            grouped_leaf("o1", "openai", "vendor/o1", "smart-openai-nt", 500),
            terminal_leaf("loc", "local", "local/nt"),
            terminal_leaf("ds", "metered", "metered/nt"),
        ),
    ) + "\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\n\
         pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
         target = \"g1\"\ncandidates = [\"g2\", \"g3\", \"g4\", \"o1\"]\n\
         terminal_fallback = [[\"loc\"], [\"ds\"]]\n\
         terminal_policy = \"quota_only\"\n\
         terminal_policies = [\"availability\", \"quota_only\"]\n";
    // The temp dir must outlive the runner; these tests are about routing.
    std::mem::forget(quota);
    let state = switchyard_server::ServerState::from_runner(runner(&config)).expect("state");
    Fixture {
        google,
        openai,
        local,
        metered,
        router: switchyard_server::build_switchyard_router(state),
    }
}

/// Two requests with no evidence at all must not both land on the provider that happens to
/// own four of the five leaves.
#[tokio::test]
async fn two_equally_evidenced_groups_are_both_served() {
    let fixture = smart_nt(
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (first_status, first) = call(&fixture.router, "smart-nt", "hi").await;
    let (second_status, second) = call(&fixture.router, "smart-nt", "hi").await;

    assert!(first_status.is_success(), "{first}");
    assert!(second_status.is_success(), "{second}");
    let served: Vec<String> = [&first, &second]
        .iter()
        .map(|body| body["model"].as_str().unwrap_or_default().to_string())
        .collect();
    let google = served.iter().filter(|model| model.starts_with("vendor/g")).count();
    let openai = served.iter().filter(|model| model.starts_with("vendor/o")).count();
    assert_eq!(
        (google, openai),
        (1, 1),
        "one request per group: {served:?}"
    );
    assert_eq!(fixture.local.calls(), 0);
    assert_eq!(fixture.metered.calls(), 0);
}

/// A group whose allowance is provably spent steps aside for the other primary group, and
/// the terminal tiers are not touched: the pool still answered.
#[tokio::test]
async fn a_spent_google_group_yields_to_openai_without_touching_the_terminals() {
    let fixture = smart_nt(
        account_wide_cap,
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;

    assert!(status.is_success(), "{body}");
    assert!(
        body["model"].as_str().unwrap_or_default().starts_with("vendor/o"),
        "served by the ready group: {body}"
    );
    assert_eq!(
        fixture.google.calls(),
        1,
        "the first 429 says the whole Google credential is spent, so the other three leaves are not retried"
    );
    assert_eq!(fixture.local.calls(), 0);
    assert_eq!(fixture.metered.calls(), 0);
}

/// Both primary allowance domains spent, with evidence, is exactly what the local tier is
/// for. The metered tier behind it stays untouched: local answered.
#[tokio::test]
async fn both_primary_groups_spent_crosses_to_the_local_tier() {
    let fixture = smart_nt(
        account_wide_cap,
        account_wide_cap,
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(body["model"], "local/nt");
    assert_eq!(fixture.local.calls(), 1);
    assert_eq!(fixture.metered.calls(), 0);
    assert_eq!(fixture.google.calls(), 1);
    assert_eq!(fixture.openai.calls(), 1);
}

/// The metered tier exists for the local outage. Once the allowance is exhausted and the
/// local model cannot answer, that is a capacity fact, and the declared boundary for the
/// local tier is the one that decides it.
#[tokio::test]
async fn a_local_outage_passes_the_walk_to_the_metered_tier() {
    let fixture = smart_nt(
        account_wide_cap,
        account_wide_cap,
        outage,
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(body["model"], "metered/nt");
    assert!(
        fixture.local.calls() >= 1,
        "the local tier was consulted and could not answer"
    );
    assert_eq!(
        fixture.metered.calls(),
        1,
        "the metered tier answered on its first dispatch"
    );
}

/// A refusal about the request never buys a terminal call. Retrying it elsewhere would either
/// evade the refusal or fail identically at a price.
#[tokio::test]
async fn a_policy_refusal_never_reaches_the_local_or_metered_tier() {
    let fixture = smart_nt(
        policy_refusal,
        policy_refusal,
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;

    assert!(!status.is_success(), "{body}");
    assert_eq!(fixture.local.calls(), 0);
    assert_eq!(fixture.metered.calls(), 0);
}

/// An outage with no allowance evidence is not exhaustion. The local tier is reserved for the
/// allowance claim, so it is not spent on "the provider could not be reached".
#[tokio::test]
async fn an_outage_without_allowance_evidence_does_not_open_the_local_tier() {
    let fixture = smart_nt(
        outage,
        outage,
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;

    assert!(!status.is_success(), "{body}");
    assert_eq!(fixture.local.calls(), 0);
    assert_eq!(fixture.metered.calls(), 0);
}

/// A grouped pool with one group is a peer pool written the long way, and accepting it would
/// let an operator believe a boundary exists that the deployment never draws.
#[test]
fn a_grouped_pool_with_a_single_group_is_refused() {
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}",
        quota.path().display(),
        client("google", "http://127.0.0.1:1/v1"),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 500),
        grouped_leaf("g2", "google", "vendor/g2", "smart-gemini-bounded", 500),
        "\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\npool_groups = [\"smart-gemini-bounded\"]\n\
         target = \"g1\"\ncandidates = [\"g2\"]\n"
    );
    std::mem::forget(quota);
    let error = load_error(&config);
    assert!(error.contains("fewer than two pool_groups"), "{error}");
}

/// A grouped pool with no groups has nothing to compare, so it is refused rather than
/// silently ranked flat.
#[test]
fn a_grouped_pool_without_groups_is_refused() {
    let config = format!(
        "schema_version = 1\n{}{}",
        client("google", "http://127.0.0.1:1/v1"),
        "\n[targets.g1]\nid = \"vendor/g1\"\nllm_client = \"google\"\nfree_tier = true\n\
         \n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\ntarget = \"g1\"\ncandidates = []\n"
    );
    let error = load_error(&config);
    assert!(error.contains("but no pool_groups"), "{error}");
}

/// A boundary policy must state one value per tier; a short list would silently leave the
/// rest on the route policy.
#[test]
fn a_boundary_policy_list_must_match_the_tiers() {
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}{}",
        quota.path().display(),
        client("google", "http://127.0.0.1:1/v1"),
        client("local", "http://127.0.0.1:2/v1"),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 500),
        grouped_leaf("o1", "google", "vendor/o1", "smart-openai-nt", 500),
        format!(
            "{}",
            terminal_leaf("loc", "local", "local/nt")
        ),
    ) + "\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
        selection = \"grouped\"\n\
        pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
        target = \"g1\"\ncandidates = [\"o1\"]\nterminal_fallback = [[\"loc\"]]\n\
        terminal_policies = [\"quota_only\", \"availability\"]\n";
    std::mem::forget(quota);
    let error = load_error(&config);
    assert!(error.contains("one value per tier"), "{error}");
}

/// An ordered pool is the cost shape: the declared group sequence is the priority, so the
/// second group is not offered while the first still has an eligible member - and, unlike the
/// equal-peer pool, repeated requests do not rotate between them.
#[tokio::test]
async fn an_ordered_pool_walks_its_first_group_before_the_second() {
    let first = Upstream::new(|model| Some((200, success(model))));
    let second = Upstream::new(|model| Some((200, success(model))));
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}{}\n\
         [routes.smart-route]\nid = \"smart-route\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\ngroup_ordering = \"ordered\"\n\
         pool_groups = [\"smart-gemini-chat\", \"smart-openrouter-free\"]\n\
         target = \"c1\"\ncandidates = [\"c2\", \"f1\"]\n",
        quota.path().display(),
        client("google", &first_url),
        client("openrouter", &second_url),
        grouped_leaf("c1", "google", "vendor/c1", "smart-gemini-chat", 500),
        grouped_leaf("c2", "google", "vendor/c2", "smart-gemini-chat", 500),
        grouped_leaf("f1", "openrouter", "free/f1", "smart-openrouter-free", 500),
    );
    std::mem::forget(quota);
    let state = switchyard_server::ServerState::from_runner(runner(&config)).expect("state");
    let router = switchyard_server::build_switchyard_router(state);

    for _ in 0..3 {
        let (status, body) = call(&router, "smart-route", "hi").await;
        assert!(status.is_success(), "{body}");
        assert!(
            body["model"].as_str().unwrap_or_default().starts_with("vendor/"),
            "the cheap group answers while it can: {body}"
        );
    }
    assert_eq!(
        second.calls(),
        0,
        "the paid group is not offered while the cheap group has a member left"
    );
    assert_eq!(
        first.calls(),
        3,
        "no rotation between ordered groups: every request starts at the first group"
    );
}

/// The group shape is a deployment decision with two exact words; anything else is refused
/// rather than defaulted, because a typo would silently deploy the other policy.
#[test]
fn an_unknown_group_ordering_is_refused() {
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}\n\
         [routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\ngroup_ordering = \"cheapest\"\n\
         pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
         target = \"g1\"\ncandidates = [\"o1\"]\n",
        quota.path().display(),
        client("google", "http://127.0.0.1:1/v1"),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 500),
        grouped_leaf("o1", "google", "vendor/o1", "smart-openai-nt", 500),
    );
    std::mem::forget(quota);
    let error = load_error(&config);
    assert!(error.contains("unknown group_ordering"), "{error}");
}

/// `flat` is the absence of grouping, so it cannot be stated for a grouped pool.
#[test]
fn a_flat_group_ordering_is_refused() {
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}\n\
         [routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\ngroup_ordering = \"flat\"\n\
         pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
         target = \"g1\"\ncandidates = [\"o1\"]\n",
        quota.path().display(),
        client("google", "http://127.0.0.1:1/v1"),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 500),
        grouped_leaf("o1", "google", "vendor/o1", "smart-openai-nt", 500),
    );
    std::mem::forget(quota);
    let error = load_error(&config);
    assert!(error.contains("always orders by group"), "{error}");
}

/// A group ordering on a route that does not group would be read by nobody; refusing it
/// keeps the manifest from claiming a policy the route does not have.
#[test]
fn a_group_ordering_without_grouped_selection_is_refused() {
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}\n\
         [routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         group_ordering = \"ordered\"\ntarget = \"g1\"\ncandidates = []\n",
        quota.path().display(),
        client("google", "http://127.0.0.1:1/v1"),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 500),
    );
    std::mem::forget(quota);
    let error = load_error(&config);
    assert!(error.contains("without selection"), "{error}");
}
