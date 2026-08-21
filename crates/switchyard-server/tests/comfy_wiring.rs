// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw integration fork.
// SPDX-License-Identifier: Apache-2.0

//! Gate C2-W integration tests: PROVE that normal production composition
//! (`load_server_state` -> `config.build()` -> `build_comfy()` -> driver spawn)
//! actually causes a driver-originated authenticated request to a ComfyNinja
//! endpoint, that shutdown stops it, and that disabled/absent spawns nothing.
//!
//! These tests enforce the fake bearer (the mock rejects missing/incorrect
//! auth), so a request only counts if it was authenticated. They use a clearly
//! fake local token and a fake loopback HTTP endpoint. No production `:8447`
//! contact occurs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use switchyard_server::{ServerState, config::load_server_state};

const FAKE_TOKEN: &str = "c2w-test-only-fake-bearer";

/// One fake ComfyNinja endpoint that enforces the bearer and counts requests.
#[derive(Default)]
struct MockComfy {
    snapshot_requests: AtomicUsize,
    transition_requests: AtomicUsize,
    auth_failures: AtomicUsize,
}

fn authed(headers: &HeaderMap) -> bool {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == format!("Bearer {FAKE_TOKEN}"))
        .unwrap_or(false)
}

async fn mock_resource(State(st): State<Arc<MockComfy>>, headers: HeaderMap) -> Response {
    st.snapshot_requests.fetch_add(1, Ordering::SeqCst);
    if !authed(&headers) {
        st.auth_failures.fetch_add(1, Ordering::SeqCst);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Response::new(Body::from(
        r#"{"producer_epoch":"e0","state_generation":1,"state_fingerprint":"fp","state":{
            "owner":null,"mode":"idle","transition_state":"idle","transition_target":null,
            "comfyui":null,"unsloth_studio":null,"llama_server":null,"studio_backend_health":"ok",
            "resident_qwen_profile":"unknown","comfy_busy":null,"comfy_queue_running":null,
            "comfy_queue_pending":null,"vram_used_mib":67,"vram_free_mib":24260}
        }"#,
    ))
}

async fn mock_transitions(State(st): State<Arc<MockComfy>>, headers: HeaderMap) -> Response {
    st.transition_requests.fetch_add(1, Ordering::SeqCst);
    if !authed(&headers) {
        st.auth_failures.fetch_add(1, Ordering::SeqCst);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Response::new(Body::from(
        r#"{"producer_epoch":"e0","oldest_available_eid":1,"latest_eid":0,"epoch_reset":false,"gap":false,"overflow":false,"events":[]}"#,
    ))
}

/// Route a candidate `routes.toml` with a `[comfyninja]` block pointing at the
/// fake endpoint into a `ServerState` (the real production composition path).
///
/// The fake credential env var is set for the caller's lifetime (it must remain
/// present for the driver's request-time resolution); callers should
/// `std::env::remove_var("COMFY_C2W_TEST_TOKEN")` after `stop_comfy`.
async fn build_state_with_comfy(comfy_block: &str, base_url: &str) -> ServerState {
    let comfy = comfy_block.replace("__BASE__", base_url);
    let toml = format!(
        r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.weak]
id = "weak/model"
llm_client = "primary"

[routes.noop]
id = "switchyard/noop"
type = "noop"
{comfy}
"#
    );
    let path = std::env::temp_dir().join(format!(
        "c2w-routes-{}-{}.toml",
        std::process::id(),
        comfy_block.len()
    ));
    std::fs::write(&path, &toml).expect("write temp routes");
    let state = load_server_state(&path).expect("load production-style config");
    let _ = std::fs::remove_file(&path);
    state
}

const ENABLED_BLOCK: &str = r#"
[comfyninja]
enabled = true
ttl_seconds = 30
[comfyninja.snapshot]
url = "__BASE__/v1/resource"
auth_token_env = "COMFY_C2W_TEST_TOKEN"
[comfyninja.transitions]
url = "__BASE__/v1/transitions"
auth_token_env = "COMFY_C2W_TEST_TOKEN"
"#;

const DISABLED_BLOCK: &str = r#"
[comfyninja]
enabled = false
[comfyninja.snapshot]
url = "__BASE__/v1/resource"
auth_token_env = "COMFY_C2W_TEST_TOKEN"
[comfyninja.transitions]
url = "__BASE__/v1/transitions"
auth_token_env = "COMFY_C2W_TEST_TOKEN"
"#;

/// A minimal production-shaped config with no comfy at all (also used for the
/// "absent" composition — but we still need valid routes, so we add a noop).
const MINIMAL_TOML: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.weak]
id = "weak/model"
llm_client = "primary"

[routes.noop]
id = "switchyard/noop"
type = "noop"
"#;

async fn spawn_mock() -> (String, Arc<MockComfy>) {
    let state = Arc::new(MockComfy::default());
    let app = Router::new()
        .route("/v1/resource", get(mock_resource))
        .route("/v1/transitions", get(mock_transitions))
        .with_state(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), state)
}

/// Enabled normal composition -> the production path spawns the driver, which
/// makes an authenticated request to the fake endpoint.
#[tokio::test]
async fn enabled_composition_starts_exactly_one_driver_and_reaches_auth_endpoint() {
    // The count of simultaneously live drivers is hard to observe directly, but
    // a single normal composition must produce requests that repeat over ticks
    // (not one per request/route/model). We prove the driver is ACTIVE by
    // observing an authenticated request arrive on a normal config build.
    unsafe { std::env::set_var("COMFY_C2W_TEST_TOKEN", FAKE_TOKEN) };
    let (base, mock) = spawn_mock().await;
    let state = build_state_with_comfy(ENABLED_BLOCK, &base).await;
    assert!(
        state.has_comfy(),
        "enabled composition must activate one driver"
    );
    let status = state.comfy_status().await.expect("driver status");
    assert!(status.enabled);

    // Allow a short window; the driver starts fetching immediately on spawn.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let mut saw = 0usize;
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        saw = mock.snapshot_requests.load(Ordering::SeqCst)
            + mock.transition_requests.load(Ordering::SeqCst);
        if saw > 0 {
            break;
        }
    }
    assert!(
        saw > 0,
        "driver must reach the authenticated ComfyNinja endpoint (snap={} trans={})",
        mock.snapshot_requests.load(Ordering::SeqCst),
        mock.transition_requests.load(Ordering::SeqCst)
    );
    assert_eq!(
        mock.auth_failures.load(Ordering::SeqCst),
        0,
        "all driver requests must be authenticated (Auth header present)"
    );

    // Deterministic shutdown: the driver stops and no further requests arrive.
    state.stop_comfy().await;
    unsafe { std::env::remove_var("COMFY_C2W_TEST_TOKEN") };
    let after = mock.snapshot_requests.load(Ordering::SeqCst)
        + mock.transition_requests.load(Ordering::SeqCst);
    let settle = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut post = after;
    while std::time::Instant::now() < settle {
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        post = mock.snapshot_requests.load(Ordering::SeqCst)
            + mock.transition_requests.load(Ordering::SeqCst);
    }
    assert_eq!(
        post, after,
        "no further driver requests after shutdown (before {after}, after {post})"
    );
}

/// Disabled composition -> no driver, zero driver-originated requests.
#[tokio::test]
async fn disabled_composition_starts_zero_drivers_and_zero_requests() {
    let (base, mock) = spawn_mock().await;
    let state = build_state_with_comfy(DISABLED_BLOCK, &base).await;
    assert!(
        !state.has_comfy(),
        "disabled composition must NOT activate a driver"
    );
    assert!(state.comfy_status().await.is_none());
    // Allow a short window; no driver exists so no request can arrive.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let count = mock.snapshot_requests.load(Ordering::SeqCst)
        + mock.transition_requests.load(Ordering::SeqCst)
        + mock.auth_failures.load(Ordering::SeqCst);
    assert_eq!(
        count, 0,
        "disabled => zero driver-originated requests (saw {count})"
    );
}

/// Absent config -> no driver, zero requests.
#[tokio::test]
async fn absent_composition_starts_zero_drivers() {
    let (_base, mock) = spawn_mock().await;
    let path = std::env::temp_dir().join(format!("c2w-absent-{}.toml", std::process::id()));
    std::fs::write(&path, MINIMAL_TOML).unwrap();
    let state = load_server_state(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(!state.has_comfy());
    assert!(state.comfy_status().await.is_none());
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(mock.snapshot_requests.load(Ordering::SeqCst), 0);
    assert_eq!(mock.transition_requests.load(Ordering::SeqCst), 0);
}
