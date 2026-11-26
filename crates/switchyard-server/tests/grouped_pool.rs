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
                let (status, payload) = script(&model).unwrap_or_else(|| (200, success(&model)));
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
    smart_nt_with_boundaries(
        "quota_only",
        "\"availability\", \"quota_only\"",
        google_script,
        openai_script,
        local_script,
        metered_script,
    )
}

/// The same deployed shape, with the boundary policies spelled out so a test can vary the
/// tier-leaving list and prove which slot the walk actually reads.
fn smart_nt_with_boundaries(
    route_policy: &str,
    terminal_policies: &str,
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
    ) + &format!(
        "\n[routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\n\
         pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
         target = \"g1\"\ncandidates = [\"g2\", \"g3\", \"g4\", \"o1\"]\n\
         terminal_fallback = [[\"loc\"], [\"ds\"]]\n\
         terminal_policy = \"{route_policy}\"\n\
         terminal_policies = [{terminal_policies}]\n"
    );
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
    let google = served
        .iter()
        .filter(|model| model.starts_with("vendor/g"))
        .count();
    let openai = served
        .iter()
        .filter(|model| model.starts_with("vendor/o"))
        .count();
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
        body["model"]
            .as_str()
            .unwrap_or_default()
            .starts_with("vendor/o"),
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
    let fixture = smart_nt(account_wide_cap, account_wide_cap, outage, |model| {
        Some((200, success(model)))
    });

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
        format!("{}", terminal_leaf("loc", "local", "local/nt")),
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
            body["model"]
                .as_str()
                .unwrap_or_default()
                .starts_with("vendor/"),
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

// ---------------------------------------------------------------------------
// ORDERED groups: the cost-efficiency ladder
// ---------------------------------------------------------------------------
//
// `smart-route` is not a set of equal peers. Its groups are a declared economic
// sequence - Google free chat, then OpenRouter free, then DeepSeek direct, then
// OpenRouter paid - and no evidence may let a later, more expensive group jump
// ahead of an earlier one that can still answer. The tests below pin the three
// properties that shape depends on:
//
// * a later group may be paid, so the free-only pool gate must not apply here;
// * a ready earlier group is never displaced by a later one, however the later
//   one measures;
// * inside a group the leaf facts still choose the best eligible member, so the
//   declared sequence never becomes a fixed leaf order.

/// A leaf in an ordered ladder. `free` is what makes the group free or metered;
/// `max_input_tokens` is what lets a test make one member ineligible for size.
fn ladder_leaf(
    name: &str,
    client: &str,
    slug: &str,
    group: &str,
    free: bool,
    fit: Option<u64>,
) -> String {
    let mut out = format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = {free}\n\
         groups = [\"{group}\"]\n"
    );
    if let Some(fit) = fit {
        out.push_str(&format!("max_input_tokens = {fit}\n"));
    }
    out
}

/// The smart-route shape in miniature: a free chat group, then a metered group.
struct Ladder {
    chat: Upstream,
    paid: Upstream,
    router: Router,
}

fn ladder(chat_script: Script, paid_script: Script) -> Ladder {
    let chat = Upstream::new(chat_script);
    let paid = Upstream::new(paid_script);
    let chat_url = chat.clone().server();
    let paid_url = paid.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}{}\n\
         [routes.smart-route]\nid = \"smart-route\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\ngroup_ordering = \"ordered\"\n\
         pool_groups = [\"smart-gemini-chat\", \"smart-openrouter-paid\"]\n\
         target = \"g1\"\ncandidates = [\"g2\", \"z1\"]\n",
        quota.path().display(),
        client("chat", &chat_url),
        client("paid", &paid_url),
        ladder_leaf("g1", "chat", "vendor/g1", "smart-gemini-chat", true, None),
        ladder_leaf("g2", "chat", "vendor/g2", "smart-gemini-chat", true, None),
        ladder_leaf(
            "z1",
            "paid",
            "paid/z1",
            "smart-openrouter-paid",
            false,
            None
        ),
    );
    std::mem::forget(quota);
    let state = switchyard_server::ServerState::from_runner(runner(&config)).expect("state");
    Ladder {
        chat,
        paid,
        router: switchyard_server::build_switchyard_router(state),
    }
}

/// The metered group is a legitimate member of the ladder: it must load, and it must be
/// reachable only once the free group above it cannot answer.
#[tokio::test]
async fn an_ordered_pool_admits_a_metered_group_behind_a_free_one() {
    let fixture = ladder(
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-route", "hi").await;
    assert!(status.is_success(), "{body}");
    assert!(
        body["model"]
            .as_str()
            .unwrap_or_default()
            .starts_with("vendor/"),
        "the free group answers while it can: {body}"
    );
    assert_eq!(
        fixture.paid.calls(),
        0,
        "the metered group is not offered while free capacity remains"
    );
}

/// A ready earlier group keeps the request however the later group measures. Under the
/// adaptive policy two equal groups rotate; under the ordered policy they do not, so three
/// requests that all land on the free group is the observation that separates the two.
#[tokio::test]
async fn an_ordered_pool_never_promotes_a_later_group_over_a_ready_earlier_one() {
    let fixture = ladder(
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let mut served = Vec::new();
    for _ in 0..3 {
        let (status, body) = call(&fixture.router, "smart-route", "hi").await;
        assert!(status.is_success(), "{body}");
        served.push(body["model"].as_str().unwrap_or_default().to_string());
    }
    assert_eq!(
        served
            .iter()
            .filter(|model| model.starts_with("vendor/"))
            .count(),
        3,
        "every request stays on the earlier group: {served:?}"
    );
    assert_eq!(fixture.chat.calls(), 3, "each request spent one free leg");
    assert_eq!(fixture.paid.calls(), 0);
}

/// Inside a group the eligible member wins, not the first row. Here the first declared
/// member cannot hold the request, so the group is exhausted for this request by a fact
/// about fit - and the sibling answers. That is the "best eligible within group" rule; a
/// strict declared order would have sent the request to the metered group instead.
#[tokio::test]
async fn an_ordered_pool_chooses_the_best_eligible_member_inside_its_group() {
    let chat = Upstream::new(|model| Some((200, success(model))));
    let paid = Upstream::new(|model| Some((200, success(model))));
    let chat_url = chat.clone().server();
    let paid_url = paid.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}{}\n\
         [routes.smart-route]\nid = \"smart-route\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\ngroup_ordering = \"ordered\"\n\
         pool_groups = [\"smart-gemini-chat\", \"smart-openrouter-paid\"]\n\
         target = \"g1\"\ncandidates = [\"g2\", \"z1\"]\n",
        quota.path().display(),
        client("chat", &chat_url),
        client("paid", &paid_url),
        ladder_leaf(
            "g1",
            "chat",
            "vendor/g1",
            "smart-gemini-chat",
            true,
            Some(1)
        ),
        ladder_leaf("g2", "chat", "vendor/g2", "smart-gemini-chat", true, None),
        ladder_leaf(
            "z1",
            "paid",
            "paid/z1",
            "smart-openrouter-paid",
            false,
            None
        ),
    );
    std::mem::forget(quota);
    let state = switchyard_server::ServerState::from_runner(runner(&config)).expect("state");
    let router = switchyard_server::build_switchyard_router(state);

    let (status, body) = call(&router, "smart-route", "a request well over one token").await;
    assert!(status.is_success(), "{body}");
    assert_eq!(
        body["model"].as_str().unwrap_or_default(),
        "vendor/g2",
        "the eligible sibling inside the group answers: {body}"
    );
    assert_eq!(
        chat.calls(),
        1,
        "the ineligible first member is never dispatched"
    );
    assert_eq!(
        paid.calls(),
        0,
        "the group answered, so the metered group is not consulted"
    );
}

/// The exemption is scoped to the ordered shape. An adaptive pool is still a free peer
/// pool, so a metered member there is a load error rather than a paid escalation.
#[test]
fn an_adaptive_pool_still_refuses_a_metered_group_member() {
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}\n\
         [routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\n\
         pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
         target = \"g1\"\ncandidates = [\"z1\"]\n",
        quota.path().display(),
        client("chat", "http://127.0.0.1:1/v1"),
        client("paid", "http://127.0.0.1:1/v1"),
        ladder_leaf(
            "g1",
            "chat",
            "vendor/g1",
            "smart-gemini-bounded",
            true,
            None
        ),
        ladder_leaf("z1", "paid", "paid/z1", "smart-openai-nt", false, None),
    );
    std::mem::forget(quota);
    let error = load_error(&config);
    assert!(error.contains("does not declare free_tier"), "{error}");
}

// ---------------------------------------------------------------------------
// Tier-leaving indexing: which slot the walk actually reads
// ---------------------------------------------------------------------------
//
// `terminal_policies[i]` governs the boundary LEAVING tier i. The route-level policy governs
// entry into tier 1, and for a two-tier route the second slot is never read - it is positional
// padding so the list can be checked against the tier count. Both facts are easy to get wrong
// and expensive when wrong, so each is pinned by an observation that only holds under the
// correct indexing.

/// Two primary domains with a declared budget of one request each, a local tier and a metered
/// tier. A budget is the only way to produce `daily_allowance_exhausted` from the deployment's
/// own accounting: a bare 429 is unclassified, and the strict rule is entitled to refuse it.
fn two_domains_with_budgets(
    route_policy: &str,
    terminal_policies: &str,
    local_script: Script,
    metered_script: Script,
) -> Fixture {
    let google = Upstream::new(|model| Some((200, success(model))));
    let openai = Upstream::new(|model| Some((200, success(model))));
    let local = Upstream::new(local_script);
    let metered = Upstream::new(metered_script);
    let google_url = google.clone().server();
    let openai_url = openai.clone().server();
    let local_url = local.clone().server();
    let metered_url = metered.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}{}{}{}{}\n\
         [routes.smart-nt]\nid = \"smart-nt\"\ntype = \"passthrough\"\n\
         selection = \"grouped\"\n\
         pool_groups = [\"smart-gemini-bounded\", \"smart-openai-nt\"]\n\
         target = \"g1\"\ncandidates = [\"o1\"]\n\
         terminal_fallback = [[\"loc\"], [\"ds\"]]\n\
         terminal_policy = \"{route_policy}\"\n\
         terminal_policies = [{terminal_policies}]\n",
        quota.path().display(),
        client("google", &google_url),
        client("openai", &openai_url),
        client("local", &local_url),
        client("metered", &metered_url),
        grouped_leaf("g1", "google", "vendor/g1", "smart-gemini-bounded", 1),
        grouped_leaf("o1", "openai", "vendor/o1", "smart-openai-nt", 1),
        terminal_leaf("loc", "local", "local/nt"),
        terminal_leaf("ds", "metered", "metered/nt"),
    );
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

/// Spends the one declared request on each primary domain, so the next call finds both
/// allowances affirmatively exhausted by the deployment's own accounting.
async fn exhaust_both_domains(router: &Router) {
    for _ in 0..2 {
        let (status, body) = call(router, "smart-nt", "warm").await;
        assert!(status.is_success(), "warm-up request: {body}");
    }
}

/// Leaving the local tier on an outage crosses to the metered tier, because slot 0 - the
/// boundary being left - says `availability`. Slot 1 says the strict allowance rule, which no
/// outage could satisfy, so this also proves slot 1 is not read: if the walk consulted it, the
/// crossing would be refused.
#[tokio::test]
async fn the_boundary_leaving_the_local_tier_is_the_one_that_decides() {
    let fixture = two_domains_with_budgets(
        "allowance_only",
        "\"availability\", \"allowance_only\"",
        outage,
        |model| Some((200, success(model))),
    );
    exhaust_both_domains(&fixture.router).await;

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;
    assert!(status.is_success(), "{body}");
    assert_eq!(
        body["model"].as_str().unwrap_or_default(),
        "metered/nt",
        "a local outage crosses on slot 0 regardless of slot 1: {body}"
    );
    assert_eq!(fixture.metered.calls(), 1);
}

/// The same configuration with the two slots swapped behaves differently, which is what makes
/// the first test meaningful: `allowance_only` at slot 0 refuses a local outage, so the request
/// stops at the local tier instead of spending the metered one.
#[tokio::test]
async fn swapping_the_boundary_slots_changes_the_outcome() {
    let fixture = two_domains_with_budgets(
        "allowance_only",
        "\"allowance_only\", \"availability\"",
        outage,
        |model| Some((200, success(model))),
    );
    exhaust_both_domains(&fixture.router).await;

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;
    assert!(
        !status.is_success(),
        "an outage is not allowance evidence: {body}"
    );
    assert_eq!(
        fixture.metered.calls(),
        0,
        "the metered tier is not spent on a local outage"
    );
}

/// A minute throttle is not allowance exhaustion, so the strict route policy refuses to open
/// the local tier on it even though the Google domain really is spent. This is the defect the
/// previous `quota_only` entry rule carried: one domain exhausted plus one domain throttled
/// looked the same as two domains exhausted.
#[tokio::test]
async fn a_minute_throttle_does_not_open_the_local_tier_under_the_strict_rule() {
    let fixture = smart_nt_with_boundaries(
        "allowance_only",
        "\"availability\", \"availability\"",
        account_wide_cap,
        |_model| {
            Some((
                429,
                json!({"error": {"message": "minute limit", "code": 429}}).to_string(),
            ))
        },
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;
    assert!(
        !status.is_success(),
        "not served while one domain is only throttled: {body}"
    );
    assert_eq!(
        fixture.local.calls(),
        0,
        "the local tier is not opened by a minute throttle"
    );
    assert_eq!(fixture.metered.calls(), 0);
}

/// The same request under the previous entry rule does open the local tier, which is what makes
/// the strict rule a real change rather than a rename. Kept as a regression witness: if the
/// strict predicate ever widens back to `quota_only`, this test still passes while the one above
/// fails, and an operator can see which rule the deployment is actually running.
#[tokio::test]
async fn the_previous_quota_only_rule_would_have_opened_the_local_tier() {
    let fixture = smart_nt_with_boundaries(
        "quota_only",
        "\"availability\", \"availability\"",
        account_wide_cap,
        |_model| {
            Some((
                429,
                json!({"error": {"message": "minute limit", "code": 429}}).to_string(),
            ))
        },
        |model| Some((200, success(model))),
        |model| Some((200, success(model))),
    );

    let (status, body) = call(&fixture.router, "smart-nt", "hi").await;
    assert!(status.is_success(), "{body}");
    assert_eq!(
        body["model"].as_str().unwrap_or_default(),
        "local/nt",
        "the historical rule treats a throttle as exhaustion: {body}"
    );
}
