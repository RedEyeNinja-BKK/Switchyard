// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `smart-judge` escalates a request its own candidate pool cannot hold.
//!
//! The pool is two Gemma legs inside the owner's free allowance, each admitting at most
//! 12,307 estimated tokens because that is what its minute window leaves after the admission
//! margin. A judge prompt that needs more than that admitted nothing at all, and the route
//! refused the caller with a 400 before either declared tier was looked at — measured in
//! production on 2026-11-24: the same route served a 4,946-token request on the local tier
//! 54 ms before refusing a 13,200-token request as `no_eligible_candidate` with
//! `dispatched: []`.
//!
//! Two claims are being separated here, and the tests are written so a passing suite cannot
//! be explained by the refusal having simply moved:
//!
//! * **size** — no candidate in the pool can hold the request, and a declared tier can. The
//!   ordered tiers are consulted, and the tier that answers is the first one that could hold
//!   it. This is authorized by `fit_escalation`, and recorded under its own reason
//!   (`primary_pool_cannot_hold_request`), never as capacity exhaustion.
//! * **everything else** — a capability the request needs, a credential or balance the
//!   deployment lost, an accounting fault, an invalid judgment. None of these becomes true by
//!   asking a metered leg, so none of them may escalate. Each control asserts the *upstream
//!   call count*, because "the response was a 400" is also satisfied by a route that spent a
//!   tier to learn the same thing.
//!
//! Every input size is constructed exactly: `judged()` asserts the estimate it produced before
//! the request is sent, so a fixture that drifted cannot quietly test a different boundary.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;

const JUDGE: &str = "smart-judge";

/// The production admission ceiling of one Gemma leg: 16,000 TPM less the 1,300-per-mille
/// admission margin. Hard-coded on purpose — this is the number the refusal boundary was
/// measured at, and a fixture that derived it from the code under test could not catch the
/// code changing.
const GEMMA_CEILING: u64 = 12_307;

type Script = fn(&str) -> Option<(u16, String)>;

/// One provider backend. It records every model it was asked for, so a test can state which
/// credential was spent instead of inferring it from a response.
#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    seen_models: Arc<Mutex<Vec<String>>>,
    script: Script,
    delay: Option<Duration>,
}

impl Upstream {
    fn new(script: Script) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            seen_models: Arc::new(Mutex::new(Vec::new())),
            script,
            delay: None,
        }
    }

    /// The same upstream, but it takes `delay` to answer. Used to cancel a request while a
    /// tier is executing.
    fn slow(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
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
        let delay = self.delay;
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
                if let Some(delay) = delay {
                    std::thread::sleep(delay);
                }
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
                "content": [{"type": "output_text", "text": "{\"verdict\":\"accept\"}"}]
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

/// A normal answer: the tier must serve when it is reached.
fn serves(model: &str) -> Option<(u16, String)> {
    Some(success(model))
}

/// The first tier is down. Not a statement about the allowance, and not a statement about
/// whether the request fits.
fn unavailable(_model: &str) -> Option<(u16, String)> {
    Some((
        503,
        json!({"error": {"code": 503, "message": "The model is overloaded"}}).to_string(),
    ))
}

/// A 200 whose whole output is empty. A judge verdict that is not there is a fact about the
/// answer, never about capacity, so the walk must stop.
fn empty_answer(model: &str) -> Option<(u16, String)> {
    Some((
        200,
        json!({
            "id": "resp_empty",
            "object": "response",
            "model": model,
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}
        })
        .to_string(),
    ))
}

fn client(name: &str, base_url: &str) -> String {
    // One attempt per target on purpose. A retry inside a tier would make "which tier was
    // spent" ambiguous in a call count, and the escalation decision is about tiers, not about
    // how many times one provider was asked.
    format!(
        "\n[llm_clients.{name}]\nformat = \"openai_responses\"\nbase_url = \"{base_url}\"\n\
         max_retries = 0\n"
    )
}

/// One Gemma leg at the production admission ceiling.
fn gemma(name: &str, slug: &str, extra: &str) -> String {
    format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"google\"\nfree_tier = true\n\
         max_input_tokens = {GEMMA_CEILING}\nrequests_per_minute = 30\n\
         input_tokens_per_minute = 16000\nadmission_margin_permille = 1300\n{extra}"
    )
}

/// A terminal target, with a per-request ceiling only when the test needs one to be unable
/// to hold the request.
fn terminal_target(
    name: &str,
    client: &str,
    slug: &str,
    ceiling: Option<u64>,
    extra: &str,
) -> String {
    let limits = ceiling.map_or_else(String::new, |ceiling| {
        format!(
            "max_input_tokens = {ceiling}\nrequests_per_minute = 30\n\
             input_tokens_per_minute = {ceiling}\nadmission_margin_permille = 1000\n"
        )
    });
    format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = true\n{limits}{extra}"
    )
}

/// The judge lane as production declares it: two Gemma candidates in one tiered pool and two
/// ordered terminal tiers. Only the facts a test varies are parameters.
struct Shape<'a> {
    /// Lines added to both Gemma targets.
    gemma_extra: &'a str,
    /// Lines added to both terminal targets.
    terminal_extra: &'a str,
    /// Lines added to the first terminal target only, overriding `terminal_extra` for it.
    first_extra: Option<&'a str>,
    /// Lines added to the route.
    route_extra: &'a str,
    fit_escalation: bool,
    luna_ceiling: Option<u64>,
    htpc_ceiling: Option<u64>,
}

impl Default for Shape<'_> {
    fn default() -> Self {
        Self {
            gemma_extra: "",
            terminal_extra: "",
            first_extra: None,
            route_extra: "",
            fit_escalation: true,
            luna_ceiling: None,
            htpc_ceiling: None,
        }
    }
}

fn deployment(shape: &Shape<'_>, pool_url: &str, luna_url: &str, htpc_url: &str) -> String {
    let escalation = if shape.fit_escalation {
        "fit_escalation = true\n"
    } else {
        ""
    };
    let route_extra = shape.route_extra;
    format!(
        "schema_version = 1\n{}{}{}{}{}{}{}\n[routes.smart-judge]\nid = \"smart-judge\"\n\
         type = \"passthrough\"\ntarget = \"gemma26\"\ncandidates = [\"gemma31\"]\n\
         selection = \"measured\"\nterminal_fallback = [[\"luna\"], [\"htpc-nt\"]]\n\
         terminal_policy = \"availability\"\ncontext_window = {GEMMA_CEILING}\n\
         reasoning = false\nvision = false\nvideo = false\n{escalation}{route_extra}",
        client("google", pool_url),
        client("luna", luna_url),
        client("htpc", htpc_url),
        gemma("gemma26", "google/gemma-4-26b-a4b-it", shape.gemma_extra),
        gemma("gemma31", "google/gemma-4-31b-it", shape.gemma_extra),
        terminal_target(
            "luna",
            "luna",
            "openai/gpt-6-luna-nt",
            shape.luna_ceiling,
            shape.first_extra.unwrap_or(shape.terminal_extra),
        ),
        terminal_target(
            "htpc-nt",
            "htpc",
            "Qwen3.5-9B-MTP",
            shape.htpc_ceiling,
            shape.terminal_extra,
        ),
    )
}

/// A deployment with two unlimited terminal tiers, which is the shape every escalation test
/// starts from.
fn tiers(pool: Script, first: Script, second: Script) -> (Upstream, Upstream, Upstream, String) {
    let pool = Upstream::new(pool);
    let first = Upstream::new(first);
    let second = Upstream::new(second);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(&Shape::default(), &pool_url, &first_url, &second_url);
    (pool, first, second, config)
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

fn router(runner: Runner) -> axum::Router {
    switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    )
}

async fn post(router: axum::Router, body: Value) -> (StatusCode, Value) {
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

async fn call(runner: Runner, body: Value) -> (StatusCode, Value) {
    post(router(runner), body).await
}

/// A judge request whose estimated input is exactly `tokens`.
///
/// The estimate counts every JSON string as `ceil(ascii/4) + 1` plus a fixed 8 for structure
/// (`request_fit::estimate_input_tokens`), so the filler length is solved for rather than
/// guessed, and the solved value is asserted before the request is used.
fn judged(tokens: u64, output: Option<u64>) -> Value {
    let model_cost = JUDGE.len().div_ceil(4) as u64 + 1;
    let mut chars = tokens.saturating_sub(8 + model_cost + 1).saturating_mul(4);
    loop {
        let body = json!({
            "model": JUDGE,
            "input": "x".repeat(chars as usize),
            "max_output_tokens": output,
        });
        match switchyard_runner::request_fit::estimate_input_tokens(&body).cmp(&tokens) {
            std::cmp::Ordering::Equal => {
                assert_eq!(
                    switchyard_runner::request_fit::estimate_input_tokens(&body),
                    tokens,
                    "the fixture must cost exactly the boundary it names"
                );
                return body;
            }
            std::cmp::Ordering::Less => chars += 1,
            std::cmp::Ordering::Greater => chars -= 1,
        }
    }
}

/// The distinct models an upstream was asked for: coverage, not a retry count.
async fn contacted(upstream: &Upstream) -> Vec<String> {
    let mut seen = upstream.models().await;
    seen.sort();
    seen.dedup();
    seen
}

// ---------------------------------------------------------------------------
// The boundary the production refusal was measured on.
// ---------------------------------------------------------------------------

/// 10,250 input plus a 2,048-token output reservation is 12,298, which the pool holds. Nothing
/// above the pool is contacted, so the wide tiers cannot become a general shortcut.
#[tokio::test]
async fn a_request_the_pool_can_hold_is_served_by_the_pool() {
    let (pool, first, second, config) = tiers(serves, serves, serves);
    let (status, body) = call(runner(&config), judged(10_250, Some(2_048))).await;

    assert_eq!(status, StatusCode::OK, "the pool should hold this: {body}");
    assert_eq!(body["model"], "google/gemma-4-26b-a4b-it");
    assert_eq!(first.calls(), 0, "the first tier must not be spent");
    assert_eq!(second.calls(), 0, "the second tier must not be spent");
    assert_eq!(contacted(&pool).await, vec!["google/gemma-4-26b-a4b-it"]);
}

/// 10,259 + 2,048 is exactly the 12,307 ceiling: the last request the pool can hold.
#[tokio::test]
async fn the_exact_ceiling_is_still_the_pools_request() {
    let (pool, first, second, config) = tiers(serves, serves, serves);
    let (status, body) = call(runner(&config), judged(10_259, Some(2_048))).await;

    assert_eq!(status, StatusCode::OK, "12,307 must fit: {body}");
    assert_eq!(body["model"], "google/gemma-4-26b-a4b-it");
    assert_eq!(first.calls(), 0);
    assert_eq!(second.calls(), 0);
    assert_eq!(contacted(&pool).await, vec!["google/gemma-4-26b-a4b-it"]);
}

/// One token past the ceiling. The pool admits nothing, and the first tier holds the request,
/// so the first tier serves it — with the pool contacted zero times, because a target that
/// cannot hold the request is never asked.
#[tokio::test]
async fn one_token_over_the_ceiling_reaches_the_first_tier() {
    let (pool, first, second, config) = tiers(serves, serves, serves);
    let (status, body) = call(runner(&config), judged(10_260, Some(2_048))).await;

    assert_eq!(status, StatusCode::OK, "the tier should hold this: {body}");
    assert_eq!(body["model"], "openai/gpt-6-luna-nt");
    assert_eq!(
        pool.calls(),
        0,
        "a pool that cannot hold the request is not called"
    );
    assert_eq!(first.calls(), 1, "the first tier is the wider leg");
    assert_eq!(second.calls(), 0, "the second tier is not reached");
}

/// The size that failed in production on 2026-11-24.
#[tokio::test]
async fn the_production_refusal_size_now_reaches_the_first_tier() {
    let (pool, first, second, config) = tiers(serves, serves, serves);
    let (status, body) = call(runner(&config), judged(13_200, Some(2_048))).await;

    assert_eq!(status, StatusCode::OK, "13,200 should reach a tier: {body}");
    assert_eq!(body["model"], "openai/gpt-6-luna-nt");
    assert_eq!(pool.calls(), 0);
    assert_eq!(first.calls(), 1);
    assert_eq!(second.calls(), 0);
}

/// The upper end of the observed refusal range. Still a size fact, still the first tier.
#[tokio::test]
async fn the_upper_end_of_the_refusal_range_reaches_the_first_tier() {
    let (pool, first, second, config) = tiers(serves, serves, serves);
    let (status, body) = call(runner(&config), judged(19_580, Some(2_048))).await;

    assert_eq!(status, StatusCode::OK, "19,580 should reach a tier: {body}");
    assert_eq!(body["model"], "openai/gpt-6-luna-nt");
    assert_eq!(pool.calls(), 0);
    assert_eq!(first.calls(), 1);
    assert_eq!(second.calls(), 0);
}

// ---------------------------------------------------------------------------
// The ordered walk.
// ---------------------------------------------------------------------------

/// The first tier is reachable but down. Exhaustion, not size, decides the second boundary,
/// and the second tier is the one that answers.
#[tokio::test]
async fn a_first_tier_that_is_down_hands_the_request_to_the_second() {
    let (pool, first, second, config) = tiers(serves, unavailable, serves);
    let (status, body) = call(runner(&config), judged(13_200, Some(2_048))).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "the second tier should answer: {body}"
    );
    assert_eq!(body["model"], "Qwen3.5-9B-MTP");
    assert_eq!(pool.calls(), 0);
    assert_eq!(
        first.calls(),
        1,
        "the first tier is tried before the second"
    );
    assert_eq!(second.calls(), 1);
}

/// The first tier cannot hold this request either. It is passed over without a call, because
/// the question at this boundary is the same size question the pool answered, and the walk
/// continues to the tier that can hold it.
#[tokio::test]
async fn a_first_tier_that_cannot_hold_the_request_is_passed_over() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    // Both terminal tiers admit at most 20,000 estimated tokens; the request needs 21,628.
    let config = deployment(
        &Shape {
            luna_ceiling: Some(20_000),
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );
    let (status, body) = call(runner(&config), judged(19_580, Some(2_048))).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "the local tier should answer: {body}"
    );
    assert_eq!(body["model"], "Qwen3.5-9B-MTP");
    assert_eq!(pool.calls(), 0);
    assert_eq!(
        first.calls(),
        0,
        "a tier that cannot hold the request must not be called to find out"
    );
    assert_eq!(second.calls(), 1);
}

/// The first tier admits nothing because it cannot serve a tool turn, and the second tier
/// could. That is not the size question the walk was opened on, so the walk ends and neither
/// tier is called: passing the request to the second tier would buy an answer to a question
/// about the request, and would credit a boundary the route never proved out.
#[tokio::test]
async fn a_tier_that_refuses_for_a_capability_ends_the_walk() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    // The pool permits tools and is emptied by size alone; the first tier refuses the tools
    // and the second tier would accept them.
    let config = deployment(
        &Shape {
            first_extra: Some("tool_calling = false\n"),
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );
    let mut body = judged(13_200, Some(2_048));
    body["tools"] = json!([{
        "type": "function",
        "name": "read_file",
        "parameters": {"type": "object", "properties": {}}
    }]);
    // The tool definitions add a little to the estimate; the point of the fixture is that the
    // pool still cannot hold it.
    assert!(
        switchyard_runner::request_fit::estimate_input_tokens(&body) > GEMMA_CEILING,
        "the fixture must still exceed the pool ceiling"
    );
    let (status, refusal) = call(runner(&config), body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{refusal}");
    assert_eq!(pool.calls(), 0);
    assert_eq!(
        first.calls(),
        0,
        "the tier refused on capability, not on size"
    );
    assert_eq!(
        second.calls(),
        0,
        "a capability refusal at one boundary must not buy the next tier"
    );
}

/// Nothing authorized can hold it. The refusal stands, nothing is dispatched at all, and the
/// caller is told what did not fit rather than given a success from somewhere else.
#[tokio::test]
async fn when_no_authorized_tier_can_hold_it_the_refusal_is_final() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(
        &Shape {
            luna_ceiling: Some(20_000),
            htpc_ceiling: Some(20_000),
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );
    let (status, body) = call(runner(&config), judged(19_580, Some(2_048))).await;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "the refusal is honest: {body}"
    );
    assert_eq!(pool.calls(), 0);
    assert_eq!(
        first.calls(),
        0,
        "nothing could hold it, so nothing was called"
    );
    assert_eq!(second.calls(), 0);
    let message = body.to_string();
    assert!(
        message.contains("input_tokens_per_minute"),
        "the refusal must name the limit that excluded the request: {message}"
    );
}

// ---------------------------------------------------------------------------
// Controls: what must NOT escalate.
// ---------------------------------------------------------------------------

/// The request needs a capability the lane never declared. Its size would also exclude every
/// candidate, so a rule that only looked at exclusion codes would escalate it — the route's own
/// declaration is what stops it, and nothing upstream is contacted.
#[tokio::test]
async fn a_request_the_route_never_promised_to_serve_is_not_escalated() {
    let (pool, first, second, config) = tiers(serves, serves, serves);
    // A data URI long enough that the size gate would also exclude both Gemma legs.
    let padding = "A".repeat(60_000);
    let body = json!({
        "model": JUDGE,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": format!("data:image/png;base64,{padding}")},
                {"type": "input_text", "text": "read this"}
            ]
        }],
        "max_output_tokens": 2_048,
    });
    let (status, refusal) = call(runner(&config), body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{refusal}");
    assert!(
        refusal.to_string().contains("vision"),
        "the refusal names the capability: {refusal}"
    );
    assert_eq!(pool.calls(), 0);
    assert_eq!(first.calls(), 0, "a capability refusal must not buy a tier");
    assert_eq!(second.calls(), 0);
}

/// The pool is empty because both legs refuse the request's tools. That is not a size fact, so
/// the tiers stay out of it even though the pool dispatched nothing.
#[tokio::test]
async fn an_empty_pool_for_a_reason_that_is_not_size_is_not_escalated() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    // Neither Gemma target can hold a tool turn; both tiers can. The route deliberately does
    // not claim `tool_calling` itself: the loader validates that claim against the candidate
    // pool and refuses a route-level claim no candidate honours. Leaving it unset still
    // permits a tool turn, which is what this control needs — the pool empties for a
    // capability reason, so both tiers must stay unspent.
    let config = deployment(
        &Shape {
            gemma_extra: "tool_calling = false\n",
            terminal_extra: "tool_calling = true\n",
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );
    let body = json!({
        "model": JUDGE,
        "input": "judge this",
        "tools": [{
            "type": "function",
            "name": "read_file",
            "parameters": {"type": "object", "properties": {}}
        }],
    });
    let (status, refusal) = call(runner(&config), body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{refusal}");
    assert!(
        refusal.to_string().contains("tool_calling"),
        "the refusal names the capability: {refusal}"
    );
    assert_eq!(pool.calls(), 0);
    assert_eq!(first.calls(), 0, "a capability refusal must not buy a tier");
    assert_eq!(second.calls(), 0);
}

/// The first tier answers with nothing. An absent verdict is a fact about the answer, so the
/// second tier is never asked to produce one.
#[tokio::test]
async fn an_invalid_answer_from_the_first_tier_does_not_escalate() {
    let (pool, first, second, config) = tiers(serves, empty_answer, serves);
    let (status, _body) = call(runner(&config), judged(13_200, Some(2_048))).await;

    assert_ne!(
        status,
        StatusCode::OK,
        "an empty tier answer is not a success"
    );
    assert_eq!(pool.calls(), 0);
    assert_eq!(first.calls(), 1);
    assert_eq!(
        second.calls(),
        0,
        "a quality refusal is not a reason to spend the next tier"
    );
}

/// Cancelling the caller while the first tier is executing must not continue the walk. The
/// second tier costs money and shares one local slot, so a request nobody is waiting for must
/// not reach it.
#[tokio::test]
async fn a_cancelled_caller_does_not_walk_to_the_next_tier() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves).slow(Duration::from_secs(3));
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(&Shape::default(), &pool_url, &first_url, &second_url);

    let pending = post(router(runner(&config)), judged(13_200, Some(2_048)));
    let cancelled = tokio::time::timeout(Duration::from_millis(400), pending).await;
    assert!(
        cancelled.is_err(),
        "the tier is slow, so the caller times out"
    );
    // Give the abandoned future and the upstream thread time to settle.
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert_eq!(pool.calls(), 0);
    assert_eq!(first.calls(), 1, "the first tier was executing");
    assert_eq!(second.calls(), 0, "the second tier must not be paid for");
}

/// The same topology without the policy: the historical refusal, unchanged, and no tier spent.
/// This is the control for the whole suite — the escalation above is opt-in, not a new default.
#[tokio::test]
async fn the_same_topology_without_the_policy_refuses_as_before() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(
        &Shape {
            fit_escalation: false,
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );
    let (status, body) = call(runner(&config), judged(13_200, Some(2_048))).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(pool.calls(), 0);
    assert_eq!(
        first.calls(),
        0,
        "without the policy the tiers stay out of it"
    );
    assert_eq!(second.calls(), 0);
}

// ---------------------------------------------------------------------------
// The routing log is the evidence an operator acts on.
// ---------------------------------------------------------------------------

/// The log has to name the tier that answered, the reason the walk was authorized and the
/// target that answered — read from the file the server actually writes, not from a trace
/// assembled by the test.
#[tokio::test]
async fn the_decision_record_names_the_crossed_tier_and_its_reason() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(unavailable);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(&Shape::default(), &pool_url, &first_url, &second_url);

    let dir = std::env::temp_dir().join(format!("sj-fit-log-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let attempts = dir.join("routing.jsonl");
    let decisions = dir.join("routing.decisions.jsonl");
    let _ = std::fs::remove_file(&decisions);
    let state = switchyard_server::ServerState::from_runner(runner(&config))
        .expect("state builds")
        .with_routing_log(&attempts)
        .expect("log opens");
    let (status, body) = post(
        switchyard_server::build_switchyard_router(state),
        judged(13_200, Some(2_048)),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "Qwen3.5-9B-MTP");

    let text = std::fs::read_to_string(&decisions).expect("decision log was written");
    let record: Value =
        serde_json::from_str(text.lines().next().expect("one record")).expect("record parses");
    assert_eq!(record["outcome"], "served");
    assert_eq!(record["served"], "Qwen3.5-9B-MTP");
    assert_eq!(
        record["terminal_crossed"], "primary_pool_cannot_hold_request",
        "the reason is the size claim, never capacity exhaustion: {record}"
    );
    assert_eq!(record["terminal_tier"], 2, "the second tier answered");
    assert_eq!(
        record["skipped"]["google/gemma-4-26b-a4b-it"], "max_input_tokens",
        "the pool's own refusal is still recorded: {record}"
    );
    assert_eq!(
        record["attempts"][0]["model"], "openai/gpt-6-luna-nt",
        "the first tier was attempted and failed: {record}"
    );
}

/// A tier that was judged and could not hold the request leaves evidence, so a refusal cannot
/// be read as "the wider legs do not exist".
#[tokio::test]
async fn the_decision_record_names_why_a_tier_was_passed_over() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(
        &Shape {
            luna_ceiling: Some(20_000),
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );

    let dir = std::env::temp_dir().join(format!("sj-fit-skip-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let attempts = dir.join("routing.jsonl");
    let decisions = dir.join("routing.decisions.jsonl");
    let _ = std::fs::remove_file(&decisions);
    let state = switchyard_server::ServerState::from_runner(runner(&config))
        .expect("state builds")
        .with_routing_log(&attempts)
        .expect("log opens");
    let (status, _body) = post(
        switchyard_server::build_switchyard_router(state),
        judged(19_580, Some(2_048)),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let text = std::fs::read_to_string(&decisions).expect("decision log was written");
    let record: Value =
        serde_json::from_str(text.lines().next().expect("one record")).expect("record parses");
    assert_eq!(record["terminal_tier"], 2);
    assert_eq!(
        record["terminal_skipped"]["openai/gpt-6-luna-nt"], "max_input_tokens",
        "the tier that was passed over says why: {record}"
    );
    assert_eq!(record["terminal_skipped"]["Qwen3.5-9B-MTP"], Value::Null);
}

/// `terminal_skipped` is a traversal fact, not an evaluation fact. When the option is off the
/// tiers are never consulted, so the field is absent — the refusal cannot be read as "the wider
/// legs were asked and could not hold this".
#[tokio::test]
async fn the_record_leaves_terminal_skipped_absent_when_tiers_are_never_consulted() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(
        &Shape {
            fit_escalation: false,
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );

    let dir = std::env::temp_dir().join(format!("sj-fit-never-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let attempts = dir.join("routing.jsonl");
    let decisions = dir.join("routing.decisions.jsonl");
    let _ = std::fs::remove_file(&decisions);
    let state = switchyard_server::ServerState::from_runner(runner(&config))
        .expect("state builds")
        .with_routing_log(&attempts)
        .expect("log opens");
    let (status, _body) = post(
        switchyard_server::build_switchyard_router(state),
        judged(13_200, Some(2_048)),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let text = std::fs::read_to_string(&decisions).expect("decision log was written");
    let record: Value =
        serde_json::from_str(text.lines().next().expect("one record")).expect("record parses");
    assert_eq!(record["outcome"], "no_eligible_candidate");
    assert!(
        record.get("terminal_skipped").is_none(),
        "tiers were never consulted, so terminal_skipped must be absent: {record}"
    );
}

/// Both tiers are reached, neither can hold the request, and the walk ends without a dispatch.
/// `terminal_skipped` names both, and `terminal_crossed` stays absent because no tier was ever
/// spent. That pair is the one the field exists to make readable.
#[tokio::test]
async fn the_record_names_every_reached_tier_that_could_not_hold_it() {
    let pool = Upstream::new(serves);
    let first = Upstream::new(serves);
    let second = Upstream::new(serves);
    let pool_url = pool.clone().server();
    let first_url = first.clone().server();
    let second_url = second.clone().server();
    let config = deployment(
        &Shape {
            luna_ceiling: Some(20_000),
            htpc_ceiling: Some(20_000),
            ..Shape::default()
        },
        &pool_url,
        &first_url,
        &second_url,
    );

    let dir = std::env::temp_dir().join(format!("sj-fit-both-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let attempts = dir.join("routing.jsonl");
    let decisions = dir.join("routing.decisions.jsonl");
    let _ = std::fs::remove_file(&decisions);
    let state = switchyard_server::ServerState::from_runner(runner(&config))
        .expect("state builds")
        .with_routing_log(&attempts)
        .expect("log opens");
    let (status, _body) = post(
        switchyard_server::build_switchyard_router(state),
        judged(19_580, Some(2_048)),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let text = std::fs::read_to_string(&decisions).expect("decision log was written");
    let record: Value =
        serde_json::from_str(text.lines().next().expect("one record")).expect("record parses");
    assert_eq!(record["outcome"], "no_eligible_candidate");
    assert_eq!(
        record["terminal_skipped"]["openai/gpt-6-luna-nt"], "max_input_tokens",
        "the first tier was reached and could not hold it: {record}"
    );
    assert_eq!(
        record["terminal_skipped"]["Qwen3.5-9B-MTP"], "max_input_tokens",
        "the second tier was reached and could not hold it: {record}"
    );
    assert!(
        record.get("terminal_crossed").is_none(),
        "no tier was dispatched, so terminal_crossed stays absent: {record}"
    );
}
