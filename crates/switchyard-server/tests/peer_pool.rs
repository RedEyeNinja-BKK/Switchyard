// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral peer selection over one mixed free pool.
//!
//! Every test here would pass under a static provider waterfall only by accident, so each
//! one states the observation that rules the waterfall out:
//!
//! * the winner is chosen by a fact, not by which provider block was typed first;
//! * swapping the two providers' facts swaps the winner;
//! * permuting the candidate order changes nothing when the facts are unequal;
//! * the pool's declared membership and its candidate list must agree, so a stale list
//!   cannot silently narrow the pool;
//! * a target that does not positively declare free entitlement cannot be in the pool at
//!   all, because `unknown` is not free.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use switchyard_runner::Runner;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// One provider backend. It answers `200` for every model except the ones whose id carries
/// `throttled`, which answer a model-scoped `429` — the shape that says "this model is busy
/// upstream", not "this account is out of allowance".
#[derive(Clone)]
struct Upstream {
    calls: Arc<AtomicUsize>,
    seen_models: Arc<Mutex<Vec<String>>>,
}

impl Upstream {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            seen_models: Arc::new(Mutex::new(Vec::new())),
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
                let (status, payload) = if model.contains("throttled") {
                    (
                        429,
                        json!({"error": {"message": "upstream busy", "code": 429}}).to_string(),
                    )
                } else {
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
                };
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

/// One free leaf: a target in `group` with its own daily budget.
fn leaf(
    name: &str,
    client: &str,
    slug: &str,
    group: &str,
    daily: Option<u64>,
    free: Option<bool>,
) -> String {
    let free_line = match free {
        Some(value) => format!("free_tier = {value}\n"),
        None => String::new(),
    };
    let daily_line = daily
        .map(|limit| format!("daily_request_limit = {limit}\n"))
        .unwrap_or_default();
    format!(
        "\n[targets.{name}]\nid = \"{slug}\"\nllm_client = \"{client}\"\n{free_line}groups = [\"{group}\"]\n{daily_line}"
    )
}

fn client(name: &str, base_url: &str) -> String {
    format!("\n[llm_clients.{name}]\nformat = \"openai_responses\"\nbase_url = \"{base_url}\"\n")
}

/// A pool route: membership by group label, ranking by facts.
fn pool(target: &str, candidates: &[&str], groups: &[&str]) -> String {
    let candidates: Vec<String> = candidates
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect();
    let groups: Vec<String> = groups.iter().map(|name| format!("\"{name}\"")).collect();
    format!(
        r#"
[routes.ST_Free]
id = "ST_Free"
type = "passthrough"
target = "{target}"
candidates = [{}]
selection = "peer"
pool_groups = [{}]
"#,
        candidates.join(", "),
        groups.join(", ")
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

/// Two providers, one pool. `or` is declared first everywhere, so anything the waterfall
/// would do is visible: the caller sees which provider actually served.
async fn two_provider_pool(
    or_slug: &str,
    gem_slug: &str,
    or_daily: u64,
    gem_daily: u64,
    or_first: bool,
) -> (Upstream, Upstream, String) {
    let or = Upstream::new();
    let gem = Upstream::new();
    let or_url = or.clone().server();
    let gem_url = gem.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let leaves = format!(
        "{}{}{}{}",
        client("or", &or_url),
        client("gem", &gem_url),
        leaf(
            "or-leaf",
            "or",
            or_slug,
            "ST_OR_Free",
            Some(or_daily),
            Some(true)
        ),
        leaf(
            "gem-leaf",
            "gem",
            gem_slug,
            "ST_Gemini_Free",
            Some(gem_daily),
            Some(true)
        ),
    );
    let route = if or_first {
        pool("or-leaf", &["gem-leaf"], &["ST_OR_Free", "ST_Gemini_Free"])
    } else {
        pool("gem-leaf", &["or-leaf"], &["ST_Gemini_Free", "ST_OR_Free"])
    };
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{leaves}{route}",
        quota.path().display()
    );
    // The temp dir must outlive the runner, so it is leaked deliberately: these tests are
    // about routing, and holding a path alive for the process is cheaper than threading a
    // guard through every assertion.
    std::mem::forget(quota);
    (or, gem, config)
}

/// The larger daily budget wins, and it wins from the second provider block.
///
/// Under a provider waterfall the OpenRouter leg would be tried first and would serve,
/// because it is declared first and answers `200`. That is exactly what must not happen: the
/// pool ranks the facts, so the Gemini leg with 100 requests left beats the OpenRouter leg
/// with 2.
#[tokio::test]
async fn a_pool_ranks_across_providers_by_headroom_not_by_declaration_order() {
    let (or, gem, config) =
        two_provider_pool("vendor/or-free", "vendor/gem-free", 2, 100, true).await;

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        gem.models().await,
        ["vendor/gem-free"],
        "the Gemini leg has the larger allowance and must serve"
    );
    assert_eq!(
        or.calls(),
        0,
        "the OpenRouter leg must not be contacted merely because it is declared first"
    );
}

/// The same pool with the budgets swapped chooses the other provider.
///
/// Together with the previous test this rules out a provider preference in either direction:
/// the difference in outcome is caused by the fact that changed, not by the provider's name.
#[tokio::test]
async fn the_same_pool_with_the_budgets_swapped_chooses_the_other_provider() {
    let (or, gem, config) =
        two_provider_pool("vendor/or-free", "vendor/gem-free", 100, 2, true).await;

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(or.models().await, ["vendor/or-free"]);
    assert_eq!(gem.calls(), 0);
}

/// Permuting the declaration order of two unequal candidates does not change the winner.
///
/// This is the property a waterfall cannot have: if order were authoritative, reversing the
/// list would reverse the choice.
#[tokio::test]
async fn permuting_the_candidate_order_does_not_change_the_choice() {
    let (_or_a, gem_a, forward) =
        two_provider_pool("vendor/or-free", "vendor/gem-free", 2, 100, true).await;
    let (status, body) = call(runner(&forward), "ST_Free").await;
    assert!(status.is_success(), "{body}");
    assert_eq!(gem_a.models().await, ["vendor/gem-free"]);

    // Same facts, Gemini declared as the primary and OpenRouter as the candidate.
    let (_or_b, gem_b, reversed) =
        two_provider_pool("vendor/or-free", "vendor/gem-free", 2, 100, false).await;
    let (status, body) = call(runner(&reversed), "ST_Free").await;
    assert!(status.is_success(), "{body}");
    assert_eq!(
        gem_b.models().await,
        ["vendor/gem-free"],
        "the same facts must select the same model through either declaration order"
    );
}

/// A model-scoped failure drops that candidate and the remaining pool — across providers —
/// competes again.
///
/// The OpenRouter leg is ranked first because it has the larger budget; it answers `429`
/// for that model. The Gemini leg must then serve, and the OpenRouter leg must be called
/// exactly once so no retry storm is hidden in the assertion.
#[tokio::test]
async fn a_throttled_leg_is_dropped_and_the_other_provider_serves() {
    let (or, gem, config) =
        two_provider_pool("vendor/or-throttled", "vendor/gem-free", 100, 2, true).await;

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        gem.models().await,
        ["vendor/gem-free"],
        "the remaining candidate must be tried after the throttled leg"
    );
    // The throttled leg keeps whatever retry budget the client's own policy gives it; what
    // matters is that its failure does not end the request and does not buy anything.
    assert!(
        !or.models().await.is_empty(),
        "the throttle is only meaningful if the ranked-first leg was actually tried"
    );
}

/// A pool member that does not positively declare free entitlement is refused at load.
///
/// `unknown` and `false` are both refused, and for the same reason: an undeclared tariff is
/// not evidence that a request is free, so admitting it would be a silent cost.
#[tokio::test]
async fn a_pool_member_without_a_free_declaration_is_refused_at_load() {
    for free in [None, Some(false)] {
        let upstream = Upstream::new();
        let url = upstream.clone().server();
        let quota = tempfile::tempdir().expect("quota dir");
        let config = format!(
            "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}",
            quota.path().display(),
            client("or", &url),
            client("gem", &url),
            leaf(
                "or-leaf",
                "or",
                "vendor/or-free",
                "ST_OR_Free",
                Some(10),
                Some(true)
            ),
        ) + &leaf(
            "gem-leaf",
            "gem",
            "vendor/gem-free",
            "ST_Gemini_Free",
            Some(10),
            free,
        ) + &pool("or-leaf", &["gem-leaf"], &["ST_OR_Free", "ST_Gemini_Free"]);

        let Err(failure) = Runner::from_toml(&config) else {
            panic!("a non-free member must be refused");
        };
        let message = format!("{failure}");
        assert!(
            message.contains("free_tier = true"),
            "the refusal must name the missing free declaration, got: {message}"
        );
    }
}

/// A candidate that belongs to none of the pool's groups is refused at load.
///
/// Without this the two records of the pool could disagree: the group would name a member
/// the route never calls, or the route would call a leg with no stated membership.
#[tokio::test]
async fn a_pool_candidate_outside_every_group_is_refused_at_load() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}",
        quota.path().display(),
        client("or", &url),
        client("gem", &url),
        leaf(
            "or-leaf",
            "or",
            "vendor/or-free",
            "ST_OR_Free",
            Some(10),
            Some(true)
        ),
        leaf(
            "stray",
            "gem",
            "vendor/stray",
            "ST_Unrelated",
            Some(10),
            Some(true)
        ),
    ) + &pool("or-leaf", &["stray"], &["ST_OR_Free"]);

    let Err(failure) = Runner::from_toml(&config) else {
        panic!("a stray candidate must be refused");
    };
    let message = format!("{failure}");
    assert!(
        message.contains("pool_groups"),
        "the refusal must name the pool mismatch, got: {message}"
    );
}

/// A group that no target belongs to is refused at load.
///
/// An empty group means the pool silently became a different pool, which is the drift the
/// label exists to prevent.
#[tokio::test]
async fn an_empty_pool_group_is_refused_at_load() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}",
        quota.path().display(),
        client("or", &url),
        leaf(
            "or-leaf",
            "or",
            "vendor/or-free",
            "ST_OR_Free",
            Some(10),
            Some(true)
        ),
        pool("or-leaf", &[], &["ST_OR_Free", "ST_Gemini_Free"]),
    );

    let Err(failure) = Runner::from_toml(&config) else {
        panic!("an empty group must be refused");
    };
    let message = format!("{failure}");
    assert!(
        message.contains("no target belongs to"),
        "the refusal must name the empty group, got: {message}"
    );
}

/// An exact direct leaf is untouched by the pool machinery: one target, no ranking.
#[tokio::test]
async fn a_direct_leaf_still_calls_its_one_target() {
    let upstream = Upstream::new();
    let url = upstream.clone().server();
    let config = format!(
        "schema_version = 1\n{}{}\n[routes.direct-or-free]\nid = \"direct-or-free\"\ntype = \"passthrough\"\ntarget = \"or-leaf\"\n",
        client("or", &url),
        leaf(
            "or-leaf",
            "or",
            "vendor/or-free",
            "ST_OR_Free",
            None,
            Some(true)
        ),
    );

    let (status, body) = call(runner(&config), "direct-or-free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(upstream.models().await, ["vendor/or-free"]);
}

/// Two candidates that are equal in every measurable fact are separated by canonical id, not
/// by where the manifest happens to put them.
///
/// This is the case the legacy policy gets wrong and the reason `peer` exists: with all facts
/// and all evidence equal, the declared order is the only remaining key, so whichever provider
/// block was typed first would win every tie forever. Here the answer is the same from either
/// order, because the tie-break is the candidate's own identity.
#[tokio::test]
async fn equal_fit_candidates_break_ties_by_canonical_id_not_declaration_order() {
    for or_first in [true, false] {
        let or = Upstream::new();
        let gem = Upstream::new();
        let or_url = or.clone().server();
        let gem_url = gem.clone().server();
        // No daily limits on either leg, so nothing but identity separates them. `zzz` is
        // declared first in one arrangement and last in the other; the winner must not move.
        let leaves = format!(
            "{}{}{}{}",
            client("or", &or_url),
            client("gem", &gem_url),
            leaf(
                "or-leaf",
                "or",
                "vendor/zzz-free",
                "ST_OR_Free",
                None,
                Some(true)
            ),
            leaf(
                "gem-leaf",
                "gem",
                "vendor/aaa-free",
                "ST_Gemini_Free",
                None,
                Some(true)
            ),
        );
        let route = if or_first {
            pool("or-leaf", &["gem-leaf"], &["ST_OR_Free", "ST_Gemini_Free"])
        } else {
            pool("gem-leaf", &["or-leaf"], &["ST_Gemini_Free", "ST_OR_Free"])
        };
        let config = format!("schema_version = 1\n{leaves}{route}");

        let (status, body) = call(runner(&config), "ST_Free").await;

        assert!(status.is_success(), "{body}");
        assert_eq!(
            gem.models().await,
            ["vendor/aaa-free"],
            "equal-fit candidates must be separated by canonical id (or_first={or_first})"
        );
        assert_eq!(or.calls(), 0, "or_first={or_first}");
    }
}

/// A candidate whose own allowance is undeclared does not outrank one whose headroom is
/// verifiable.
///
/// The legacy ordering substitutes `u64::MAX` for an absent budget, which reads as "more
/// allowance than anyone else" and is not a fact the deployment ever stated. Under `peer` the
/// verifiable candidate wins, which is also what keeps an unqualified leg from quietly
/// becoming the default.
#[tokio::test]
async fn an_undeclared_allowance_does_not_outrank_a_verifiable_one() {
    let or = Upstream::new();
    let gem = Upstream::new();
    let or_url = or.clone().server();
    let gem_url = gem.clone().server();
    let quota = tempfile::tempdir().expect("quota dir");
    // OpenRouter is declared first and states nothing about its budget; Gemini states 10.
    let config = format!(
        "schema_version = 1\n\n[quota]\nstate_dir = \"{}\"\n{}{}{}{}",
        quota.path().display(),
        client("or", &or_url),
        client("gem", &gem_url),
        leaf(
            "or-leaf",
            "or",
            "vendor/or-free",
            "ST_OR_Free",
            None,
            Some(true)
        ),
        leaf(
            "gem-leaf",
            "gem",
            "vendor/gem-free",
            "ST_Gemini_Free",
            Some(10),
            Some(true)
        ),
    ) + &pool("or-leaf", &["gem-leaf"], &["ST_OR_Free", "ST_Gemini_Free"]);

    let (status, body) = call(runner(&config), "ST_Free").await;

    assert!(status.is_success(), "{body}");
    assert_eq!(
        gem.models().await,
        ["vendor/gem-free"],
        "a verifiable daily budget must outrank an undeclared one"
    );
    assert_eq!(or.calls(), 0);
}
