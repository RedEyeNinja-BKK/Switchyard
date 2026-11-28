// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mixed-credential composites.
//!
//! A child route may rank candidates that draw on different credentials — an OpenRouter
//! free leaf beside a Google free leaf is the deployed shape, and one provider's project
//! holds several models. The composite must therefore treat a credential as a property of
//! the target that failed, not of the child:
//!
//! * a spent allowance on one provider leaves every peer on another provider servable;
//! * a per-model Google budget retires that model and nothing else;
//! * a child is skipped only when blocks cover *every* target it would call;
//! * a scope the provider only stated for one model never spends the paid leg, while a
//!   scope it stated for the whole credential still does (the documented crossing);
//! * an outcome that is an answer — a policy refusal — is never handed to another
//!   provider, because that would evade it rather than serve the caller.
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

/// OpenRouter's account-wide free-model daily allowance, reported on the account rather
/// than on the model: every `:free` variant on that credential is spent until midnight.
fn openrouter_free_cap(_model: &str) -> Option<(u16, String)> {
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

/// A Google daily quota the body scopes to ONE model inside the project — the shape its
/// free tier's `generate_content_free_tier_requests` allowance actually has.
fn google_model_daily_quota(model: &str) -> (u16, String) {
    (
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
    )
}

/// The same condition, scoped by Google to the whole project: no model may be tried.
fn google_project_daily_quota(_model: &str) -> (u16, String) {
    (
        429,
        json!({"error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": "GenerateRequestsPerDayPerProject-FreeTier",
                    "quotaValue": "0"
                }]
            }]
        }})
        .to_string(),
    )
}

/// A policy refusal. It is an answer about the request, so no other provider may be asked.
fn google_permission_denied(_model: &str) -> Option<(u16, String)> {
    Some((
        403,
        json!({"error": {
            "code": 403,
            "message": "The caller does not have permission",
            "status": "PERMISSION_DENIED",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_SERVICE_BLOCKED",
                "domain": "googleapis.com"
            }]
        }})
        .to_string(),
    ))
}

fn client(name: &str, base_url: &str) -> String {
    format!("\n[llm_clients.{name}]\nformat = \"openai_responses\"\nbase_url = \"{base_url}\"\n")
}

fn target(name: &str, client: &str, slug: &str, group: &str) -> String {
    format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\nfree_tier = true\ngroups = [\"{group}\"]\n"
    )
}

/// A leaf that serves one target. Used as the "first child" that records a block.
fn leaf_route(name: &str, target: &str) -> String {
    format!(
        "\n[routes.{name}]\nid = \"{name}\"\ntype = \"passthrough\"\ntarget = \"{target}\"\ncandidates = []\n"
    )
}

/// A provider-neutral pool over `group`. Ranking is by declared facts; with no declared
/// budget the tie falls to the canonical model id, so each test can state which leg the
/// pool would have tried first by naming the slugs in that order.
fn pool_route(name: &str, target: &str, candidates: &[&str], group: &str) -> String {
    let candidates: Vec<String> = candidates
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect();
    format!(
        "\n[routes.{name}]\nid = \"{name}\"\ntype = \"passthrough\"\ntarget = \"{target}\"\ncandidates = [{}]\nselection = \"peer\"\npool_groups = [\"{group}\"]\n",
        candidates.join(", ")
    )
}

fn composite(name: &str, children: &[&str]) -> String {
    let children: Vec<String> = children.iter().map(|name| format!("\"{name}\"")).collect();
    format!(
        "\n[composites.{name}]\nchildren = [{}]\ncross_on = \"credential_block\"\n",
        children.join(", ")
    )
}

fn runner(source: &str) -> Runner {
    Runner::from_toml(source).expect("deployment builds")
}

async fn call(runner: Runner, model: &str) -> (StatusCode, Value) {
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
                    json!({"model": model, "input": "hi"}).to_string(),
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

/// A free pool over two credentials. The OpenRouter leaf is named so the peer ranking
/// reaches it first (canonical-id tie-break), which makes every assertion about the
/// second provider an assertion about fallback rather than about declaration order.
fn two_provider_pool(or_script: Script, gem_script: Script) -> (Upstream, Upstream, String) {
    let or = Upstream::new(or_script);
    let gem = Upstream::new(gem_script);
    let or_url = or.clone().server();
    let gem_url = gem.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}",
        client("or", &or_url),
        client("gem", &gem_url),
        target("or-leaf", "or", "aaa/vendor-or:free", "ST_Free_Pool"),
        target("gem-leaf", "gem", "zzz/vendor-gem", "ST_Free_Pool"),
        pool_route("ST_Free", "or-leaf", &["gem-leaf"], "ST_Free_Pool"),
    );
    (or, gem, config)
}

/// An account-wide free allowance spent on the provider the pool would try first leaves
/// the other provider — a different credential — servable.
///
/// The first leg's failure is terminal for that CREDENTIAL: OpenRouter reports the cap on
/// the account, so no other `:free` model on it can succeed. It says nothing about the
/// Google credential, whose allowance is a separate account's project. Treating the
/// failure as terminal for the ROUTE is the mixed-credential defect: the request failed
/// although a healthy peer was one candidate away.
#[tokio::test]
async fn a_spent_free_allowance_on_one_provider_leaves_the_other_servable() {
    let (or, gem, config) = two_provider_pool(openrouter_free_cap, |model| Some(success(model)));

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        or.calls(),
        1,
        "the spent credential must be tried once and not retried"
    );
    assert_eq!(
        gem.models().await,
        ["zzz/vendor-gem"],
        "the peer on the independent credential must serve"
    );
}

/// The symmetric direction: a Google project-wide quota is terminal for the Google
/// credential and must not end a pool that also holds an OpenRouter leaf.
///
/// Direction matters because the two providers phrase their terminal quota very
/// differently — Google states a gRPC status, OpenRouter a metadata limit source — and a
/// fix that only reads one spelling would pass in one direction and fail in the other.
#[tokio::test]
async fn a_google_project_quota_leaves_the_openrouter_peer_servable() {
    let gem = Upstream::new(|model| Some(google_project_daily_quota(model)));
    let or = Upstream::new(|model| Some(success(model)));
    let gem_url = gem.clone().server();
    let or_url = or.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}",
        client("gem", &gem_url),
        client("or", &or_url),
        target("gem-leaf", "gem", "aaa/vendor-gem", "ST_Free_Pool"),
        target("or-leaf", "or", "zzz/vendor-or:free", "ST_Free_Pool"),
        pool_route("ST_Free", "gem-leaf", &["or-leaf"], "ST_Free_Pool"),
    );

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(gem.calls(), 1);
    assert_eq!(
        or.models().await,
        ["zzz/vendor-or:free"],
        "an independent credential's peer must serve after the project quota"
    );
}

/// A Google daily quota scoped to one model retires that model and no other.
///
/// This is the deployed shape of Google's free tier: `generate_content_free_tier_requests`
/// is allocated per model inside the project, so a spent `flash` does not touch `lite`.
/// The observation that rules out a project-wide reading is that the second model was
/// called at all.
#[tokio::test]
async fn a_model_scoped_google_quota_retires_only_that_model() {
    let gem = Upstream::new(|model| {
        if model == "aaa/gemini-flash" {
            Some(google_model_daily_quota(model))
        } else {
            Some(success(model))
        }
    });
    let gem_url = gem.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}",
        client("gem", &gem_url),
        target("gem-flash", "gem", "aaa/gemini-flash", "ST_Gemini_Free"),
        target("gem-lite", "gem", "zzz/gemini-lite", "ST_Gemini_Free"),
        pool_route(
            "ST_Gemini_Free",
            "gem-flash",
            &["gem-lite"],
            "ST_Gemini_Free"
        ),
    );

    let (status, body) = call(runner(&config), "ST_Gemini_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        gem.models().await,
        ["aaa/gemini-flash", "zzz/gemini-lite"],
        "the spent model is tried once, then its sibling on the same project serves"
    );
}

/// A per-model Google quota never spends the paid leg.
///
/// The composite crosses children only on a scope the provider stated for the WHOLE
/// credential. One model's daily allocation is not that: the project still has budget for
/// its other models, so paying to work around this one is a cost the owner did not
/// authorize. The request fails closed, and the observation that rules out crossing is
/// that the paid upstream was never contacted.
#[tokio::test]
async fn a_model_scoped_quota_does_not_spend_the_paid_leg() {
    let gem = Upstream::new(|model| Some(google_model_daily_quota(model)));
    let paid = Upstream::new(|model| Some(success(model)));
    let gem_url = gem.clone().server();
    let paid_url = paid.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}",
        client("gem", &gem_url),
        client("paid", &paid_url),
        target("gem-only", "gem", "aaa/gemini-flash", "ST_Gemini_Free"),
        target("paid-leg", "paid", "vendor/paid-1017", "ST_Paid"),
        leaf_route("ST_Gemini_Free", "gem-only"),
        leaf_route("ST_Paid", "paid-leg"),
    ) + &composite("ST_Route", &["ST_Gemini_Free", "ST_Paid"]);

    let (status, body) = call(runner(&config), "ST_Route").await;

    assert_ne!(status, StatusCode::OK, "the request must fail closed: {body}");
    assert_eq!(
        paid.calls(),
        0,
        "a per-model budget is not an account-wide exhaustion and must not buy a paid call"
    );
}

/// A block recorded on one provider must not skip a child whose other candidate is
/// healthy.
///
/// The first child spends the OpenRouter credential's free allowance, which covers every
/// `:free` slug on it — including the mixed child's first candidate. The mixed child is
/// still servable through its Gemini peer, so it must run, with the blocked candidate
/// filtered out before ranking. Judging the block against the child's first candidate
/// only, and skipping the whole child, is the defect: a healthy peer never gets asked.
#[tokio::test]
async fn a_block_on_one_provider_does_not_skip_a_mixed_child() {
    let or = Upstream::new(openrouter_free_cap);
    let gem = Upstream::new(|model| Some(success(model)));
    let or_url = or.clone().server();
    let gem_url = gem.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}{}{}",
        client("or", &or_url),
        client("gem", &gem_url),
        target("or-only", "or", "aaa/vendor-or:free", "ST_OR_Free"),
        target("mixed-or", "or", "aab/vendor-or-2:free", "ST_Free_Pool"),
        target("mixed-gem", "gem", "zzz/vendor-gem", "ST_Free_Pool"),
        leaf_route("ST_OR_Only", "or-only"),
        pool_route(
            "ST_Free",
            "mixed-or",
            &["mixed-gem"],
            "ST_Free_Pool"
        ),
        composite("ST_Route", &["ST_OR_Only", "ST_Free"]),
    );

    let (status, body) = call(runner(&config), "ST_Route").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        or.calls(),
        1,
        "only the first child may spend the blocked credential"
    );
    assert_eq!(
        gem.models().await,
        ["zzz/vendor-gem"],
        "the mixed child must serve through the credential that is not blocked"
    );
}

/// Both free credits exhausted still reaches the paid leg — the documented crossing.
///
/// This is the positive control for the previous test: the same composite, with each
/// free child blocked on a scope the provider stated for the whole credential, must
/// cross. Without it, "never spend the paid leg" could be satisfied by refusing
/// everything.
#[tokio::test]
async fn both_free_credentials_exhausted_crosses_to_the_paid_leg() {
    let or = Upstream::new(openrouter_free_cap);
    let gem = Upstream::new(|model| Some(google_project_daily_quota(model)));
    let paid = Upstream::new(|model| Some(success(model)));
    let or_url = or.clone().server();
    let gem_url = gem.clone().server();
    let paid_url = paid.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}{}{}{}{}{}",
        client("or", &or_url),
        client("gem", &gem_url),
        client("paid", &paid_url),
        target("or-only", "or", "aaa/vendor-or:free", "ST_OR_Free"),
        target("gem-only", "gem", "aaa/vendor-gem", "ST_Gemini_Free"),
        target("paid-leg", "paid", "vendor/paid-1017", "ST_Paid"),
        leaf_route("ST_OR_Only", "or-only"),
        leaf_route("ST_Gemini_Only", "gem-only"),
        leaf_route("ST_Paid", "paid-leg"),
        composite("ST_Route", &["ST_OR_Only", "ST_Gemini_Only", "ST_Paid"]),
    );

    let (status, body) = call(runner(&config), "ST_Route").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        paid.models().await,
        ["vendor/paid-1017"],
        "confirmed exhaustion of both free credentials must reach the paid leg"
    );
}

/// A policy refusal is an answer about the request, not a fault of the credential, so it
/// is never handed to another provider.
///
/// Google states `PERMISSION_DENIED` with `keep_outcome`: the bytes must change before
/// anything is resent. Evading it by re-sending the same payload to a second provider is
/// how a refusal becomes a leak, so the observation that rules out evasion is that the
/// peer was never contacted.
#[tokio::test]
async fn a_policy_refusal_is_not_evaded_by_another_provider() {
    let gem = Upstream::new(|model| {
        google_permission_denied(model).or_else(|| Some(success(model)))
    });
    let or = Upstream::new(|model| Some(success(model)));
    let gem_url = gem.clone().server();
    let or_url = or.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}{}{}{}",
        client("gem", &gem_url),
        client("or", &or_url),
        target("gem-leaf", "gem", "aaa/vendor-gem", "ST_Free_Pool"),
        target("or-leaf", "or", "zzz/vendor-or:free", "ST_Free_Pool"),
        pool_route("ST_Free", "gem-leaf", &["or-leaf"], "ST_Free_Pool"),
    );

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert_ne!(status, StatusCode::OK, "the refusal must surface: {body}");
    assert_eq!(
        or.calls(),
        0,
        "an outcome about the request must not be re-sent to another provider"
    );
}
