// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local no-thinking leaf, tested from both surfaces that can reach it.
//!
//! The leaf exists for one job: answer a strict-JSON verdict when the metered remote leg has
//! also run out. Two properties decide whether it can do that job, and neither is visible in
//! a response body:
//!
//! * the request Switchyard puts on the wire must carry `chat_template_kwargs.enable_thinking
//!   = false`, because the llama.cpp-family server behind the target reads that key and
//!   nothing else. A route that merely *declares* `reasoning = false` while forwarding the
//!   caller's default would spend its budget on a monologue and answer nothing;
//! * the declarations the P9 gate demands must be facts. `tool_calling = true` has to survive
//!   a request that actually carries tools, and `vision`/`video = false` have to refuse media
//!   before a provider call is spent on a request the deployment cannot serve.
//!
//! Every test therefore names which upstream was contacted, how many times, and — where the
//! claim is about the wire — what body it received. A test that only read the response could
//! not tell a truthful route from one that got lucky.

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

/// One provider backend. It records the model and the whole body it was asked for, so a test
/// can assert on the request rather than infer it from the answer.
#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<Value>>>,
    script: Script,
}

impl Upstream {
    fn new(script: Script) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            bodies: Arc::new(Mutex::new(Vec::new())),
            script,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    async fn bodies(&self) -> Vec<Value> {
        self.bodies.lock().await.clone()
    }

    /// The single body this backend received. Panics with the call count otherwise, so a
    /// test cannot silently assert against the first of several requests.
    async fn only_body(&self) -> Value {
        let bodies = self.bodies().await;
        assert_eq!(
            bodies.len(),
            1,
            "expected exactly one request, got {bodies:?}"
        );
        bodies.into_iter().next().expect("one body")
    }

    fn server(self) -> String {
        let calls = Arc::clone(&self.calls);
        let bodies = Arc::clone(&self.bodies);
        let script = self.script;
        let runtime = tokio::runtime::Handle::current();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let Some(body) = read_request_body(&mut stream) else {
                    continue;
                };
                let model = body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                runtime.block_on(async { bodies.lock().await.push(body) });
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

/// Reads one HTTP request and returns its JSON body.
///
/// Delimited by `Content-Length` rather than by end-of-stream: the client under test may keep
/// the connection open, and a read-until-close loop would either block or — worse — return a
/// body assembled from a later response.
fn read_request_body(stream: &mut std::net::TcpStream) -> Option<Value> {
    use std::io::Read;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok()?;
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    let split = loop {
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..split]).to_ascii_lowercase();
    let length: usize = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse().ok())?;
    while buffer.len() < split + length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
        }
    }
    serde_json::from_slice(&buffer[split..(split + length).min(buffer.len())]).ok()
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

/// An account-wide daily cap: one fact about the credential, not about the request.
fn exhausted(_model: &str) -> Option<(u16, String)> {
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

/// An answer about the request. Never a capacity fact, so it must not authorize a further leg.
fn malformed_request(_model: &str) -> Option<(u16, String)> {
    Some((
        400,
        json!({"error": {
            "code": 400,
            "message": "Invalid value for 'input': expected a string or an array of messages",
            "type": "invalid_request_error"
        }})
        .to_string(),
    ))
}

/// A transient provider fault: the leg is reachable but cannot serve this moment.
fn unavailable(_model: &str) -> Option<(u16, String)> {
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

/// A rejected credential says the deployment is wrong, not that capacity ran out.
fn rejected(_model: &str) -> Option<(u16, String)> {
    Some((
        401,
        json!({"error": {
            "code": 401,
            "message": "Incorrect API key provided",
            "type": "invalid_request_error"
        }})
        .to_string(),
    ))
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

async fn post(runner: Runner, body: Value) -> (StatusCode, Value) {
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

/// The deployed pair: one local server, two clients, two targets that name one model and
/// differ only in reasoning policy, and two 1:1 aliases.
///
/// The clients are separate on purpose. The loader refuses two targets that name one model on
/// one client with different policies, and the policy has to differ because that is the whole
/// point of the leaf.
fn local_pair(thinking_url: &str, no_thinking_url: &str) -> String {
    format!(
        r#"schema_version = 1

[llm_clients.htpc]
format = "openai_responses"
base_url = "{thinking_url}"
locality = "local"
billing = "local"

[llm_clients.htpc-nt]
format = "openai_responses"
base_url = "{no_thinking_url}"
locality = "local"
billing = "local"
quota_group = "htpc"

[targets.htpc-qwen3_5-9b-mtp]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc"
vision = false
video = false
tool_calling = true
reasoning_policy = "enabled"
reasoning_dialect = "llama_cpp_enable_thinking"

[targets.htpc-qwen3_5-9b-mtp-nt]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc-nt"
vision = false
video = false
tool_calling = true
reasoning_policy = "disabled"
reasoning_dialect = "llama_cpp_enable_thinking"

[routes.direct-htpc-qwen3_5-9b-mtp]
id = "direct-htpc-qwen3_5-9b-mtp"
type = "passthrough"
target = "htpc-qwen3_5-9b-mtp"
candidates = []
context_window = 81920
tool_calling = true
reasoning = true
vision = false
video = false

[routes.direct-htpc-qwen3_5-9b-mtp-nt]
id = "direct-htpc-qwen3_5-9b-mtp-nt"
type = "passthrough"
target = "htpc-qwen3_5-9b-mtp-nt"
candidates = []
context_window = 81920
tool_calling = true
reasoning = false
vision = false
video = false
"#
    )
}

fn local_pair_runner() -> (Upstream, Upstream, Runner) {
    let thinking = Upstream::new(|model| Some(success(model)));
    let no_thinking = Upstream::new(|model| Some(success(model)));
    let thinking_url = thinking.clone().server();
    let no_thinking_url = no_thinking.clone().server();
    let source = local_pair(&thinking_url, &no_thinking_url);
    (thinking, no_thinking, runner(&source))
}

/// The claim the leaf is built on, read off the wire: a request through the new alias asks the
/// local server not to think, and reaches nothing else.
#[tokio::test]
async fn the_no_thinking_leaf_sends_the_refusal_the_local_server_reads() {
    let (thinking, no_thinking, runner) = local_pair_runner();
    let (status, body) = post(
        runner,
        json!({"model": "direct-htpc-qwen3_5-9b-mtp-nt", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(no_thinking.calls(), 1, "the leaf served the request");
    assert_eq!(
        thinking.calls(),
        0,
        "the thinking sibling is a different route and must not be contacted"
    );

    let sent = no_thinking.only_body().await;
    assert_eq!(
        sent.get("model").and_then(Value::as_str),
        Some("Qwen3.5-9B-MTP"),
        "the leaf must name the model the local server serves: {sent}"
    );
    assert_eq!(
        sent.pointer("/chat_template_kwargs/enable_thinking"),
        Some(&json!(false)),
        "the refusal has to be on the wire as a boolean false; the server reads this key \
         and a caller-side default would silently re-enable reasoning: {sent}"
    );
}

/// The paired control. Without it, the test above could pass because *every* local request
/// carries the refusal, including the sibling that is supposed to think.
#[tokio::test]
async fn the_thinking_sibling_still_asks_to_think_and_is_otherwise_untouched() {
    let (thinking, no_thinking, runner) = local_pair_runner();
    let (status, body) = post(
        runner,
        json!({"model": "direct-htpc-qwen3_5-9b-mtp", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(thinking.calls(), 1, "the sibling served the request");
    assert_eq!(no_thinking.calls(), 0);

    let sent = thinking.only_body().await;
    assert_eq!(
        sent.pointer("/chat_template_kwargs/enable_thinking"),
        Some(&json!(true)),
        "the existing reasoning-enabled route must be unchanged by the new leaf: {sent}"
    );
}

/// A direct alias is 1:1. When its only leg fails there is no second leg to substitute, and
/// the failure the caller sees has to be that leg's own answer rather than a success invented
/// somewhere else.
#[tokio::test]
async fn the_leaf_is_one_to_one_so_a_failed_call_does_not_reach_another_leg() {
    let thinking = Upstream::new(|model| Some(success(model)));
    let no_thinking = Upstream::new(malformed_request);
    let thinking_url = thinking.clone().server();
    let no_thinking_url = no_thinking.clone().server();
    let source = local_pair(&thinking_url, &no_thinking_url);

    let (status, body) = post(
        runner(&source),
        json!({"model": "direct-htpc-qwen3_5-9b-mtp-nt", "input": "hi"}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(no_thinking.calls(), 1, "the leg was tried exactly once");
    assert_eq!(
        thinking.calls(),
        0,
        "a 1:1 route has no substitute, and in particular not the thinking sibling"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Invalid value for 'input'")),
        "the caller must see the leg's own reason: {body}"
    );
}

/// `video = false` is a fact about this deployment, so a video request is answered by routing
/// and costs no provider call.
#[tokio::test]
async fn the_leaf_refuses_video_because_it_declares_no_video_target() {
    let (thinking, no_thinking, runner) = local_pair_runner();
    let (status, body) = post(
        runner,
        json!({
            "model": "direct-htpc-qwen3_5-9b-mtp-nt",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "what happens here"},
                {"type": "input_video", "video_url": "https://example.test/clip.mp4"}
            ]}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_capability", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("cannot accept video")),
        "the refusal names the medium: {body}"
    );
    assert_eq!(thinking.calls(), 0);
    assert_eq!(
        no_thinking.calls(),
        0,
        "no provider call for a refused request"
    );
}

/// The vision counterpart, and the reason the route repeats the target's fact: the load
/// validates one against the other, so a route that narrowed vision silently would be a
/// different deployment from the one tested here.
#[tokio::test]
async fn the_leaf_refuses_images_because_vision_is_declared_false() {
    let (thinking, no_thinking, runner) = local_pair_runner();
    let (status, body) = post(
        runner,
        json!({
            "model": "direct-htpc-qwen3_5-9b-mtp-nt",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "what is this"},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
            ]}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_capability", "{body}");
    assert_eq!(thinking.calls(), 0);
    assert_eq!(no_thinking.calls(), 0);
}

/// `tool_calling = true` is what the P9 gate asks the manifest to state, so it has to be
/// demonstrated: a tool-bearing turn must be admitted and the tools must reach the server.
#[tokio::test]
async fn the_leaf_admits_a_tool_bearing_turn_because_tool_calling_was_measured() {
    let (thinking, no_thinking, runner) = local_pair_runner();
    let (status, body) = post(
        runner,
        json!({
            "model": "direct-htpc-qwen3_5-9b-mtp-nt",
            "input": "verdict please",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {"type": "object", "properties": {}}
            }],
            "tool_choice": "auto"
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(no_thinking.calls(), 1);
    assert_eq!(thinking.calls(), 0);

    let sent = no_thinking.only_body().await;
    assert!(
        sent.get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| !tools.is_empty()),
        "the tools must survive to the backend: {sent}"
    );
    assert_eq!(
        sent.pointer("/chat_template_kwargs/enable_thinking"),
        Some(&json!(false)),
        "a tool-bearing turn must not re-enable reasoning: {sent}"
    );
}

// ---------------------------------------------------------------------------------------
// The leaf as the third tier of `smart-judge`.
//
// Hierarchy, in order: Gemma 26B/31B measured pool -> Luna NT -> local Qwen3.5 9B MTP NT.
// The pool's silence is the reason tier one exists, and tier one's silence is the only
// reason tier two may be spent.
// ---------------------------------------------------------------------------------------

/// The deployed hierarchy, with one backend per credential.
fn judge(pool_url: &str, luna_url: &str, local_url: &str) -> String {
    format!(
        r#"schema_version = 1

[llm_clients.google]
format = "openai_responses"
base_url = "{pool_url}"

[targets.gemma26]
id = "google/gemma-4-26b-a4b-it"
llm_client = "google"
free_tier = true
tool_calling = true

[targets.gemma31]
id = "google/gemma-4-31b-it"
llm_client = "google"
free_tier = true
tool_calling = true

[llm_clients.luna]
format = "openai_responses"
base_url = "{luna_url}"

[targets.luna-nt]
id = "gpt-6-luna-nt"
llm_client = "luna"
tool_calling = true

[llm_clients.htpc-nt]
format = "openai_responses"
base_url = "{local_url}"
locality = "local"
billing = "local"
quota_group = "htpc"

[targets.htpc-qwen3_5-9b-mtp-nt]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc-nt"
vision = false
video = false
tool_calling = true
reasoning_policy = "disabled"
reasoning_dialect = "llama_cpp_enable_thinking"

[routes.smart-judge]
id = "smart-judge"
type = "passthrough"
target = "gemma26"
candidates = ["gemma31"]
terminal_fallback = [["luna-nt"], ["htpc-qwen3_5-9b-mtp-nt"]]
terminal_policy = "availability"
context_window = 12307
tool_calling = true
reasoning = false
vision = false
video = false
"#
    )
}

fn judge_runner(
    pool_script: Script,
    luna_script: Script,
    local_script: Script,
) -> (Upstream, Upstream, Upstream, Runner) {
    let pool = Upstream::new(pool_script);
    let luna = Upstream::new(luna_script);
    let local = Upstream::new(local_script);
    let pool_url = pool.clone().server();
    let luna_url = luna.clone().server();
    let local_url = local.clone().server();
    let source = judge(&pool_url, &luna_url, &local_url);
    (pool, luna, local, runner(&source))
}

/// Both metered legs out of allowance, so the local leg answers — and it answers without
/// thinking, which is the only reason it can answer inside a verdict's budget.
#[tokio::test]
async fn the_local_no_thinking_leg_is_the_third_tier_of_smart_judge() {
    let (pool, luna, local, runner) =
        judge_runner(exhausted, exhausted, |model| Some(success(model)));

    let (status, body) = post(runner, json!({"model": "smart-judge", "input": "verdict"})).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(pool.calls() >= 1, "the free pool was consulted first");
    assert_eq!(luna.calls(), 1, "the metered remote leg is always tier one");
    assert_eq!(local.calls(), 1, "the local leg answered");

    let sent = local.only_body().await;
    assert_eq!(
        sent.get("model").and_then(Value::as_str),
        Some("Qwen3.5-9B-MTP"),
        "the third tier names the local model: {sent}"
    );
    assert_eq!(
        sent.pointer("/chat_template_kwargs/enable_thinking"),
        Some(&json!(false)),
        "the cascade must not lose the no-thinking refusal on its way to the terminal leg: {sent}"
    );
}

/// A serving remote leg ends the cascade. The local leg costs nothing to call, which is
/// precisely why it must not be called on a cheaper-looking impulse.
#[tokio::test]
async fn a_serving_first_tier_never_spends_the_local_leg() {
    let (pool, luna, local, runner) = judge_runner(
        exhausted,
        |model| Some(success(model)),
        |model| Some(success(model)),
    );

    let (status, body) = post(runner, json!({"model": "smart-judge", "input": "verdict"})).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(pool.calls() >= 1);
    assert_eq!(luna.calls(), 1);
    assert_eq!(local.calls(), 0, "a healthy earlier tier is never skipped");
}

/// A transient outage in the metered tier is availability, not an exhausted allowance, so
/// the local leg is exactly the right answer: the remote leg could not serve now.
#[tokio::test]
async fn a_transient_outage_in_the_metered_tier_still_reaches_the_local_leg() {
    let (pool, luna, local, runner) =
        judge_runner(exhausted, unavailable, |model| Some(success(model)));

    let (status, body) = post(runner, json!({"model": "smart-judge", "input": "verdict"})).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(pool.calls() >= 1, "the free pool was consulted first");
    assert_eq!(luna.calls(), 1);
    assert_eq!(
        local.calls(),
        1,
        "a 503 says the remote leg cannot serve now, which is what the local leg is for"
    );
}

/// A rejected credential is a deployment fault. Answering it with a local call would turn a
/// configuration error into a silent, wrong answer.
#[tokio::test]
async fn a_rejected_credential_never_spends_the_local_leg() {
    let (pool, luna, local, runner) =
        judge_runner(exhausted, rejected, |model| Some(success(model)));

    let (status, body) = post(runner, json!({"model": "smart-judge", "input": "verdict"})).await;

    assert_ne!(status, StatusCode::OK, "{body}");
    assert!(pool.calls() >= 1, "the free pool was consulted first");
    assert_eq!(luna.calls(), 1);
    assert_eq!(
        local.calls(),
        0,
        "a rejected key is not exhausted capacity, so the third tier stays unspent"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Incorrect API key")),
        "the credential fault is reported rather than masked by a fallback: {body}"
    );
}

/// The last tier's own refusal is the caller's answer. There is nothing behind it, so a
/// fabricated success would be the one outcome the hierarchy must never produce.
#[tokio::test]
async fn a_local_refusal_is_reported_truthfully_and_not_replaced_by_a_success() {
    let (pool, luna, local, runner) = judge_runner(exhausted, exhausted, malformed_request);

    let (status, body) = post(runner, json!({"model": "smart-judge", "input": "verdict"})).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(pool.calls() >= 1);
    assert_eq!(luna.calls(), 1);
    assert_eq!(local.calls(), 1, "the last tier was tried once");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("Invalid value for 'input'")),
        "the terminal leg's own reason reaches the caller: {body}"
    );
}
