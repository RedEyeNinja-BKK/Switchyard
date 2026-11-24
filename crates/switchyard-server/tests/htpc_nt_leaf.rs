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
max_input_tokens = 12307

[targets.gemma31]
id = "google/gemma-4-31b-it"
llm_client = "google"
free_tier = true
tool_calling = true
max_input_tokens = 12307

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

// ---------------------------------------------------------------------------------------
// The judgment contract itself.
//
// The third tier exists to answer a *strict JSON verdict*. Everything above proves the leaf
// is reached and asks not to think; none of it proves the answer it returns is the answer
// Turnstone's judge can use. These tests pin the contract from both ends:
//
// * the prompt side — the tools Turnstone actually sends (`read_file`, `list_directory`) and
//   the no-thinking switch, since a thinking turn spends the verdict budget on a monologue;
// * the answer side — a real verdict body survives the cascade byte-for-byte and parses
//   through a *closed* schema, so a deployment cannot quietly accept a field Turnstone would
//   discard;
// * the refusal side — an answer that is not JSON reaches the caller verbatim. Turnstone's
//   own parser returns `None` for it and falls back to a heuristic verdict labelled
//   `llm_fallback`. Switchyard must not manufacture a verdict in its place, because a
//   fabricated `approve` is the one outcome the gate exists to prevent.
// ---------------------------------------------------------------------------------------

use serde::Deserialize;

/// The verdict shape Turnstone's `IntentVerdict` carries off the wire. Closed on purpose:
/// an unexpected key or an out-of-vocabulary enum is a contract break, and a test that
/// ignored it could not tell a real verdict from a plausible-looking one.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct JudgedVerdict {
    intent_summary: String,
    risk_level: RiskLevel,
    confidence: f64,
    recommendation: Recommendation,
    reasoning: String,
    evidence: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Recommendation {
    Approve,
    Review,
    Deny,
}

/// A verdict a judge model really emits, in the shape the contract requires.
const VERDICT: &str = r#"{"intent_summary":"Read the deployment file to confirm the timeout value.","risk_level":"low","confidence":0.93,"recommendation":"approve","reasoning":"The call reads one configuration file inside the workspace and changes nothing.","evidence":["path is inside /srv/switchyard","no write flags"]}"#;

/// The tools Turnstone's judge sends when `read_only_tools` is on. Frozen here so the
/// assertion is against the names the judge uses, not a name invented by this file.
fn judge_tools() -> Value {
    json!([
        {
            "type": "function",
            "name": "read_file",
            "description": "Read the contents of a file to verify what would be affected.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        },
        {
            "type": "function",
            "name": "list_directory",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    ])
}

/// A Responses envelope carrying `text` as the assistant's output, which is where the judge
/// reads the verdict from.
fn answer_text(model: &str, text: &str) -> (u16, String) {
    (
        200,
        json!({
            "id": "resp_verdict",
            "object": "response",
            "model": model,
            "status": "completed",
            "output": [{
                "type": "message", "id": "m1", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}]
            }],
            "usage": {"input_tokens": 2600, "output_tokens": 64, "total_tokens": 2664}
        })
        .to_string(),
    )
}

/// The output text of the first assistant message in a Responses body.
fn output_text(body: &Value) -> &str {
    body.pointer("/output/0/content/0/text")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("no output text in {body}"))
}

/// The request the judge sends: a verdict prompt plus its two read-only tools. The body is a
/// fixed shape so an input-fit boundary can be computed against it.
fn judge_request() -> Value {
    json!({
        "model": "smart-judge",
        "input": "Evaluate the following tool call and answer with a JSON verdict.",
        "tools": judge_tools(),
        "tool_choice": "auto"
    })
}

/// The tools and the no-thinking switch both reach the local leg on the verdict path.
#[tokio::test]
async fn the_third_tier_receives_the_judges_tools_and_asks_for_no_thought() {
    let (pool, luna, local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, VERDICT))
    });

    let (status, body) = post(runner, judge_request()).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(pool.calls() >= 1);
    assert_eq!(luna.calls(), 1);
    assert_eq!(local.calls(), 1);

    let sent = local.only_body().await;
    let names: Vec<&str> = sent
        .get("tools")
        .and_then(Value::as_array)
        .expect("the judge's tools reach the terminal leg")
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect();
    assert_eq!(
        names,
        vec!["read_file", "list_directory"],
        "the evidence-gathering tools the judge declares must survive the cascade: {sent}"
    );
    assert_eq!(
        sent.pointer("/chat_template_kwargs/enable_thinking"),
        Some(&json!(false)),
        "a tool-bearing verdict prompt must still refuse to think: {sent}"
    );
}

/// The verdict body survives the cascade unchanged and parses through a closed schema.
///
/// This is the property the third tier is for: the answer is usable by Turnstone without a
/// repair step, and no field was added or dropped on the way through.
#[tokio::test]
async fn the_local_verdict_round_trips_through_a_closed_schema() {
    let (_pool, _luna, _local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, VERDICT))
    });

    let (status, body) = post(runner, judge_request()).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    let delivered = output_text(&body);
    assert_eq!(
        delivered, VERDICT,
        "the verdict must arrive byte-for-byte; any edit here is a fabricated verdict"
    );

    // Parsed from the bytes that came off the wire, not from the constant that produced
    // them: a schema check on `VERDICT` would restate the fixture's own input and could
    // pass even if the cascade rewrote or truncated the answer.
    let verdict: JudgedVerdict =
        serde_json::from_str(delivered).expect("a real verdict satisfies the closed schema");
    assert_eq!(verdict.risk_level, RiskLevel::Low);
    assert_eq!(verdict.recommendation, Recommendation::Approve);
    assert_eq!(verdict.confidence, 0.93);
    assert_eq!(verdict.evidence.len(), 2);
}

/// An answer that is not JSON is delivered verbatim, and the closed schema refuses it.
///
/// Turnstone's `_parse_verdict` returns `None` for this text and the caller substitutes a
/// heuristic verdict it labels `llm_fallback`. The one thing that must not happen is
/// Switchyard inventing a verdict — a plausible-looking `approve` produced on the caller's
/// behalf would defeat the gate while looking like a success.
#[tokio::test]
async fn a_non_json_answer_is_delivered_verbatim_and_refused_by_the_contract() {
    const PROSE: &str = "I believe this call is safe and I would approve it.";
    let (_pool, _luna, _local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, PROSE))
    });

    let (status, body) = post(runner, judge_request()).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        output_text(&body),
        PROSE,
        "the answer reaches the caller unchanged so the judge can classify it"
    );
    assert!(
        serde_json::from_str::<JudgedVerdict>(output_text(&body)).is_err(),
        "prose is not a verdict, and a contract that accepted it would accept anything"
    );
}

/// Out-of-vocabulary fields are delivered as written rather than repaired locally.
///
/// Turnstone normalises them (`risk_level` to `medium`, `recommendation` to `review`,
/// `confidence` into 0..1). That normalisation is Turnstone's decision and has to happen on
/// Turnstone's side of the boundary, on the bytes the model produced. A closed consumer sees
/// the break, which is exactly how the test knows no one silently rewrote the answer.
#[tokio::test]
async fn out_of_vocabulary_fields_are_not_repaired_inside_switchyard() {
    const ODD: &str = r#"{"intent_summary":"x","risk_level":"severe","confidence":5,"recommendation":"maybe","reasoning":"","evidence":[]}"#;
    let (_pool, _luna, _local, runner) =
        judge_runner(exhausted, exhausted, |model| Some(answer_text(model, ODD)));

    let (status, body) = post(runner, judge_request()).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        output_text(&body),
        ODD,
        "Switchyard must not normalise a field it does not own: {body}"
    );
    assert!(
        serde_json::from_str::<JudgedVerdict>(output_text(&body)).is_err(),
        "a closed schema refuses the vocabulary break, which is what keeps the repair honest"
    );
}

// ---------------------------------------------------------------------------------------
// Input fit at the declared ceiling, measured on the real estimator.
//
// `max_input_tokens = 12307` is not an arbitrary number: it is
// `floor(input_tokens_per_minute 16000 * 1000 / admission_margin_permille 1300)`, the largest
// single request the minute governor will admit at a 1.3x pad. The gate compares
// `estimate_input_tokens(body) + max_output_tokens` against it, and the estimate covers the
// whole outbound body — tools, schemas and formatting included — so a boundary test written
// against it is a test of the number production actually uses.
// ---------------------------------------------------------------------------------------

const JUDGE_CEILING: u64 = 12307;

fn with_output_budget(base: &Value, budget: u64) -> Value {
    let mut body = base.clone();
    body["max_output_tokens"] = json!(budget);
    body
}

/// A body whose own estimate plus its own output reservation is exactly `ceiling`.
///
/// Solved by iteration rather than assumed, because writing `max_output_tokens` changes the
/// body and therefore its estimate. A one-shot subtraction would be wrong by the width of the
/// digits it inserted.
fn exact_fit_body(base: &Value, ceiling: u64) -> (Value, u64) {
    let mut budget = ceiling;
    for _ in 0..8 {
        let body = with_output_budget(base, budget);
        let estimate = switchyard_runner::request_fit::estimate_input_tokens(&body);
        if estimate + budget == ceiling {
            return (body, budget);
        }
        budget = ceiling.saturating_sub(estimate);
    }
    panic!("output budget did not converge on an exact fit at {ceiling}");
}

/// The boundary is inclusive: a request that lands exactly on the ceiling is served, and one
/// token over it is refused before any provider is contacted.
#[tokio::test]
async fn the_judge_gate_admits_exactly_at_its_ceiling_and_refuses_one_token_over() {
    let base = judge_request();
    let (fits, budget) = exact_fit_body(&base, JUDGE_CEILING);
    let over = with_output_budget(&base, budget + 1);
    let over_required = switchyard_runner::request_fit::estimate_input_tokens(&over) + budget + 1;
    assert!(
        over_required > JUDGE_CEILING,
        "the over-ceiling body must actually be over: {over_required}"
    );

    let (_pool, _luna, _local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, VERDICT))
    });
    let (status, body) = post(runner, fits).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (pool, luna, local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, VERDICT))
    });
    let (status, body) = post(runner, over).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "unsupported_capability",
        "an unfittable request is a request fact, not a server fault: {body}"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("input_tokens_per_minute")),
        "the refusal names the binding limit. The gate compared `max_input_tokens`, but that \
         ceiling and the margin-adjusted minute bound are one limit and are reported to the \
         caller under one name; the routing log keeps the internal field name: {body}"
    );
    assert_eq!(
        pool.calls(),
        0,
        "no provider call for a request that cannot fit"
    );
    assert_eq!(luna.calls(), 0);
    assert_eq!(local.calls(), 0);
}

/// The declared ceiling is derivable, and the loader refuses one that pretends to be wider
/// than the minute window allows. This is the provenance of `12307`, asserted rather than
/// asserted-to.
#[test]
fn the_declared_ceiling_is_the_margin_adjusted_minute_window() {
    let ceiling = switchyard_llm_client::WindowLimits {
        requests_per_minute: Some(30),
        input_tokens_per_minute: Some(16000),
    }
    .safe_input_ceiling(
        switchyard_llm_client::AdmissionMargin::from_permille(1300)
            .expect("1.3x is a constructible margin"),
    )
    .expect("a declared minute window has a safe ceiling");
    assert_eq!(
        ceiling, JUDGE_CEILING,
        "12307 is floor(16000 * 1000 / 1300), not a hand-picked number"
    );

    let deploy = |max_input: u64| -> String {
        format!(
            r#"schema_version = 1

[llm_clients.google]
format = "openai_responses"
base_url = "http://127.0.0.1:1/v1"

[targets.gemma]
id = "gemma-4-26b-a4b-it"
llm_client = "google"
free_tier = true
max_input_tokens = {max_input}
requests_per_minute = 30
input_tokens_per_minute = 16000
admission_margin_permille = 1300

[routes.smart-judge]
id = "smart-judge"
type = "passthrough"
target = "gemma"
candidates = []
"#
        )
    };

    assert!(
        Runner::from_toml(&deploy(JUDGE_CEILING)).is_ok(),
        "the live ceiling sits exactly on the margin-adjusted bound and loads"
    );
    let error = Runner::from_toml(&deploy(JUDGE_CEILING + 1))
        .err()
        .expect("above the margin-adjusted bound must be refused");
    assert!(
        error.to_string().contains("max_input_tokens"),
        "the refusal names the field: {error}"
    );
}

// ---------------------------------------------------------------------------------------
// What the routing log says, and which counter the two profiles share.
// ---------------------------------------------------------------------------------------

/// Runs one request with a decision log attached and returns the record the route emitted.
async fn post_logged(runner: Runner, body: Value) -> (StatusCode, Value, Value) {
    let directory = tempfile::tempdir().expect("tempdir");
    let log_path = directory.path().join("routing.jsonl");
    let decision_path = directory.path().join("routing.decisions.jsonl");
    let state = switchyard_server::ServerState::from_runner(runner)
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
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    let answered = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let text = std::fs::read_to_string(&decision_path).expect("decision log written");
    let decision =
        serde_json::from_str(text.lines().last().expect("one record")).expect("record is json");
    (status, answered, decision)
}

/// The local leg is tier two, and the record says so. A flat list and a tiered list are
/// different declarations, and the difference has to be readable after the fact.
#[tokio::test]
async fn the_decision_record_names_the_local_leg_as_the_second_tier() {
    let (_pool, _luna, _local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, VERDICT))
    });

    let (status, body, decision) = post_logged(runner, judge_request()).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(decision["outcome"], json!("served"), "{decision}");
    assert_eq!(
        decision["served"],
        json!("Qwen3.5-9B-MTP"),
        "the record names the model that answered: {decision}"
    );
    assert_eq!(
        decision["terminal_tier"],
        json!(2),
        "Luna is tier one and the local leg is tier two: {decision}"
    );
    assert_eq!(
        decision["terminal_tiers"],
        json!([["gpt-6-luna-nt"], ["Qwen3.5-9B-MTP"]]),
        "the declared order is reported as declared: {decision}"
    );
    assert_eq!(
        decision["terminal_crossed"],
        json!("primary_pool_exhausted"),
        "the record says why the terminal leg was spent: {decision}"
    );
}

/// A request that cannot fit its own route is recorded as such, with no tier claimed.
#[tokio::test]
async fn an_unfittable_judge_request_records_no_eligible_candidate() {
    let base = judge_request();
    let (_, budget) = exact_fit_body(&base, JUDGE_CEILING);
    let over = with_output_budget(&base, budget + 1);

    let (_pool, _luna, _local, runner) = judge_runner(exhausted, exhausted, |model| {
        Some(answer_text(model, VERDICT))
    });
    let (status, body, decision) = post_logged(runner, over).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        decision["outcome"],
        json!("no_eligible_candidate"),
        "the record and the caller name the same reason: {decision}"
    );
    assert!(
        decision.get("terminal_crossed").is_none(),
        "no terminal leg was spent, so there is no crossing to report: {decision}"
    );
    assert_eq!(decision["served"], Value::Null, "{decision}");
}

/// `quota_group` is what makes the two profiles one budget.
///
/// The two clients name one physical server, so a per-client counter would hand the
/// deployment twice the allowance the provider actually grants. The loader keys the counter
/// by `(quota_group, model)`, and refuses a group whose members disagree about the limit.
fn quota_deploy(state_dir: &str, first_limit: u64, second_limit: u64, share_group: bool) -> String {
    let first_group = if share_group {
        "quota_group = \"htpc\"\n"
    } else {
        ""
    };
    let second_group = if share_group {
        "quota_group = \"htpc\"\n"
    } else {
        ""
    };
    format!(
        r#"schema_version = 1

[quota]
state_dir = "{state_dir}"

[llm_clients.htpc]
format = "openai_responses"
base_url = "http://127.0.0.1:1/v1"
locality = "local"
billing = "local"
{first_group}
[llm_clients.htpc-nt]
format = "openai_responses"
base_url = "http://127.0.0.1:1/v1"
locality = "local"
billing = "local"
{second_group}
[targets.thinking]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc"
daily_request_limit = {first_limit}

[targets.no-thinking]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc-nt"
daily_request_limit = {second_limit}

[routes.direct-htpc-qwen3_5-9b-mtp]
id = "direct-htpc-qwen3_5-9b-mtp"
type = "passthrough"
target = "thinking"
candidates = []
"#
    )
}

/// Two profiles of one model in one group are one budget, so they cannot disagree about it.
#[test]
fn two_profiles_in_one_quota_group_must_declare_one_limit() {
    let directory = tempfile::tempdir().expect("tempdir");
    let state_dir = directory.path().join("quota");
    let state_dir = state_dir.display().to_string();

    assert!(
        Runner::from_toml(&quota_deploy(&state_dir, 500, 500, true)).is_ok(),
        "one model, one group, one limit is a single counter and must load"
    );
    let error = Runner::from_toml(&quota_deploy(&state_dir, 500, 501, true))
        .err()
        .expect("a group that disagrees about its budget must be refused");
    assert!(
        error.to_string().contains("different daily_request_limit"),
        "the refusal names the disagreement: {error}"
    );

    // The control. Without a shared group the same two limits are two clients and two
    // counters, so the deployment loads. This is what proves the grouping — not the limits —
    // is what collapsed them above.
    assert!(
        Runner::from_toml(&quota_deploy(&state_dir, 500, 501, false)).is_ok(),
        "separate groups are separate counters, which is the status quo for every other pair"
    );
}

/// The two aliases are two policies over one physical model, and the wire proves it: same
/// server, same model id, and only the thinking switch moves.
#[tokio::test]
async fn the_two_aliases_are_one_model_that_differs_only_in_its_thinking_switch() {
    let physical = Upstream::new(|model| Some(success(model)));
    let url = physical.clone().server();
    let source = local_pair(&url, &url);
    let router = || {
        switchyard_server::build_switchyard_router(
            switchyard_server::ServerState::from_runner(runner(&source)).expect("state builds"),
        )
    };

    for model in [
        "direct-htpc-qwen3_5-9b-mtp",
        "direct-htpc-qwen3_5-9b-mtp-nt",
    ] {
        let response = router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"model": model, "input": "hi"}).to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("router responds");
        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("body reads");
    }

    let bodies = physical.bodies().await;
    assert_eq!(
        bodies.len(),
        2,
        "one physical server answered both aliases: {bodies:?}"
    );
    let models: Vec<&str> = bodies
        .iter()
        .filter_map(|body| body.get("model").and_then(Value::as_str))
        .collect();
    assert_eq!(
        models,
        vec!["Qwen3.5-9B-MTP", "Qwen3.5-9B-MTP"],
        "both aliases name the same served model: {bodies:?}"
    );
    let switches: Vec<Option<bool>> = bodies
        .iter()
        .map(|body| {
            body.pointer("/chat_template_kwargs/enable_thinking")
                .and_then(Value::as_bool)
        })
        .collect();
    assert_eq!(
        switches,
        vec![Some(true), Some(false)],
        "the profiles differ in the thinking switch and in nothing else: {bodies:?}"
    );
}
