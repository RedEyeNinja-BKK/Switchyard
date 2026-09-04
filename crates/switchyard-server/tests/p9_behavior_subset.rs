// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P9 behavior-critical subset: server-level tests for the two behavior areas
//! whose semantics live at the SERVER boundary and were not already covered by
//! the ported suites:
//!
//! 1. Typed capability endpoints (`/v1/embeddings`, `/v1/rerank`): contract
//!    rewrite (route id -> executor model), admission bounds (max_batch,
//!    max_candidates, non-empty inputs), and fail-closed response validation
//!    (dimension mismatch).
//! 2. Buffered-caller SSE aggregation: stream-mandatory strict-Codex legs are
//!    forced to upstream `stream = true`, and callers that asked for buffered
//!    output get aggregated JSON back (never raw SSE frames).
//!
//! Fleet admission/eligibility, fallback purity, first-byte failover, and
//! Responses-profile encoding are covered by the ported suites (59 FleetRouter
//! algorithm tests, 26 fleet-readiness monitor tests, client reliability and
//! profile tests, translation profile tests) and are deliberately not
//! duplicated here.

use std::error::Error;
use std::io::Write as _;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_server::{ServerState, build_switchyard_router};
use tokio::sync::Mutex as AsyncMutex;
use tower::ServiceExt;

type TestError = Box<dyn Error + Send + Sync>;
type TestResult<T = ()> = Result<T, TestError>;

async fn start_upstream(handler: Router) -> TestResult<(String, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, handler).await;
    });
    Ok((format!("http://{address}"), task))
}

fn load_state(toml: &str) -> TestResult<ServerState> {
    // Seed the fail-loud credential guard with placeholder values; names are
    // configuration, values are never real credentials.
    for line in toml.lines() {
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            if key == "api_key_env" || key == "auth_token_env" {
                let name = value.trim().trim_matches('"').to_string();
                if std::env::var(&name).is_err() {
                    // SAFETY: single-writer test process, placeholder-only.
                    unsafe {
                        std::env::set_var(&name, "p9-placeholder");
                    }
                }
            }
        }
    }
    let mut config = tempfile::Builder::new()
        .prefix("p9-config-")
        .suffix(".toml")
        .tempfile()?;
    config.write_all(toml.as_bytes())?;
    config.flush()?;
    Ok(switchyard_server::config::load_server_state(config.path())?)
}

async fn send(app: &Router, path: &str, body: Value) -> TestResult<(StatusCode, Value)> {
    let response = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body)?))?,
        )
        .await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok((status, serde_json::from_slice(&bytes)?))
}

// --- 1. embeddings capability endpoint ------------------------------------

#[tokio::test]
async fn embeddings_endpoint_rewrites_model_enforces_admission_and_validates_dimensions()
-> TestResult {
    let captured: Arc<AsyncMutex<Option<Value>>> = Arc::new(AsyncMutex::new(None));
    // A WRONG-dimension response first (fail-closed), fixed by a flag flip
    // for the happy path.
    let wrong = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let handler = {
        let captured = captured.clone();
        let wrong = wrong.clone();
        Router::new().route(
            "/v1/embeddings",
            post(move |Json(body): Json<Value>| {
                let captured = captured.clone();
                let wrong = wrong.clone();
                async move {
                    *captured.lock().await = Some(body);
                    let dims = if wrong.load(std::sync::atomic::Ordering::SeqCst) {
                        3
                    } else {
                        2
                    };
                    let vector: Vec<f64> = (0..dims).map(|i| 0.1 + i as f64).collect();
                    (
                        StatusCode::OK,
                        Json(json!({
                            "data": [{"embedding": vector, "index": 0, "object": "embedding"}],
                            "model": "executor-embed",
                            "usage": {"prompt_tokens": 1, "total_tokens": 1}
                        })),
                    )
                        .into_response()
                }
            }),
        )
    };
    let (base_url, task) = start_upstream(handler).await?;

    let state = load_state(&format!(
        r#"
schema_version = 1

[llm_clients.up]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "P9_EMBED_KEY"

[targets.served]
id = "vendor/served"
llm_client = "up"

[routes.route]
id = "vendor/served"
type = "passthrough"
target = "served"

[capability_clients.embed]
format = "openai_embeddings"
base_url = "{base_url}"
model = "executor-embed"
api_key_env = "P9_EMBED_KEY"

[capabilities."switchyard/test/embedding"]
id = "switchyard/test/embedding"
target = "embed"
contract = "p9-space:v1"
dimensions = 2
max_batch = 2
"#
    ))?;
    let app = build_switchyard_router(state);

    // Happy path: route id rewritten to the executor model, dimensions valid.
    wrong.store(false, std::sync::atomic::Ordering::SeqCst);
    let (status, body) = send(
        &app,
        "/v1/embeddings",
        json!({"model": "switchyard/test/embedding", "input": ["hi"]}),
    )
    .await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 2);
    let executor_body = captured.lock().await.clone().expect("upstream captured");
    assert_eq!(
        executor_body["model"],
        json!("executor-embed"),
        "route id must be rewritten to the executor model identity"
    );

    // Admission: batch bound enforced at the server, before the executor.
    let (status, body) = send(
        &app,
        "/v1/embeddings",
        json!({"model": "switchyard/test/embedding", "input": ["a", "b", "c"]}),
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "batch_too_large");

    // Empty input rejected.
    let (status, body) = send(
        &app,
        "/v1/embeddings",
        json!({"model": "switchyard/test/embedding", "input": []}),
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_input");

    // Fail-closed: dimension mismatch from the executor is a 502.
    wrong.store(true, std::sync::atomic::Ordering::SeqCst);
    let (status, body) = send(
        &app,
        "/v1/embeddings",
        json!({"model": "switchyard/test/embedding", "input": "hi"}),
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
    assert_eq!(body["error"]["code"], "embedding_invalid");

    // Unknown capability id -> 404.
    let (status, body) = send(
        &app,
        "/v1/embeddings",
        json!({"model": "nope", "input": "hi"}),
    )
    .await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "unknown_model");

    task.abort();
    Ok(())
}

// --- 2. rerank capability endpoint -----------------------------------------

#[tokio::test]
async fn rerank_endpoint_rewrites_model_and_enforces_candidate_bounds() -> TestResult {
    let captured: Arc<AsyncMutex<Option<Value>>> = Arc::new(AsyncMutex::new(None));
    let handler = {
        let captured = captured.clone();
        Router::new().route(
            "/v1/rerank",
            post(move |Json(body): Json<Value>| {
                let captured = captured.clone();
                async move {
                    *captured.lock().await = Some(body);
                    (
                        StatusCode::OK,
                        Json(json!({
                            "results": [
                                {"index": 0, "relevance_score": 0.97},
                                {"index": 1, "relevance_score": 0.42}
                            ],
                            "model": "executor-rerank",
                            "usage": {"total_tokens": 5}
                        })),
                    )
                        .into_response()
                }
            }),
        )
    };
    let (base_url, task) = start_upstream(handler).await?;

    let state = load_state(&format!(
        r#"
schema_version = 1

[llm_clients.up]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "P9_RERANK_KEY"

[targets.served]
id = "vendor/served"
llm_client = "up"

[routes.route]
id = "vendor/served"
type = "passthrough"
target = "served"

[capability_clients.rerank]
format = "cohere_jina_rerank"
base_url = "{base_url}"
model = "executor-rerank"
api_key_env = "P9_RERANK_KEY"

[capabilities."switchyard/test/rerank"]
id = "switchyard/test/rerank"
target = "rerank"
max_candidates = 2
top_n = 1
"#
    ))?;
    let app = build_switchyard_router(state);

    // Happy path.
    let (status, body) = send(
        &app,
        "/v1/rerank",
        json!({"model": "switchyard/test/rerank", "query": "q", "documents": ["a", "b"]}),
    )
    .await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"][0]["relevance_score"], 0.97);
    let executor_body = captured.lock().await.clone().expect("upstream captured");
    assert_eq!(executor_body["model"], json!("executor-rerank"));

    // Candidate bound enforced before the executor.
    let (status, body) = send(
        &app,
        "/v1/rerank",
        json!({"model": "switchyard/test/rerank", "query": "q", "documents": ["a", "b", "c"]}),
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "too_many_candidates");

    // Empty documents rejected.
    let (status, body) = send(
        &app,
        "/v1/rerank",
        json!({"model": "switchyard/test/rerank", "query": "q", "documents": []}),
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_documents");

    // Empty query rejected.
    let (status, body) = send(
        &app,
        "/v1/rerank",
        json!({"model": "switchyard/test/rerank", "query": "  ", "documents": ["a"]}),
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_query");

    task.abort();
    Ok(())
}

// --- 3. buffered-caller SSE aggregation ------------------------------------

// Stream-mandatory strict-Codex leg: the client forces upstream stream=true;
// a caller that asked for buffered output must receive aggregated JSON, and
// the chat-only stream_options decoration must never leak upstream.
#[tokio::test]
async fn buffered_caller_gets_aggregated_json_from_stream_mandatory_codex_leg() -> TestResult {
    let captured: Arc<AsyncMutex<Option<Value>>> = Arc::new(AsyncMutex::new(None));
    let sse_body = "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\
\"output_index\":0,\"content_index\":0,\"delta\":\"ok\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\
\"status\":\"completed\",\"model\":\"gpt\",\"output\":[],\"usage\":{\"input_tokens\":1,\
\"output_tokens\":1,\"total_tokens\":2}}}\n\n\
data: [DONE]\n\n";
    let handler = {
        let captured = captured.clone();
        Router::new().route(
            "/backend-api/codex/responses",
            post(move |Json(body): Json<Value>| {
                let captured = captured.clone();
                async move {
                    *captured.lock().await = Some(body);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(Body::from(sse_body))
                        .unwrap()
                }
            }),
        )
    };
    let (base_url, task) = start_upstream(handler).await?;

    let state = load_state(&format!(
        r#"
schema_version = 1

[llm_clients.codex]
format = "openai_responses"
base_url = "{base_url}/backend-api/codex"
api_key_env = "P9_CODEX_KEY"

[targets.served]
id = "vendor/served"
llm_client = "codex"

[routes.route]
id = "vendor/served"
type = "passthrough"
target = "served"
"#
    ))?;
    let app = build_switchyard_router(state);

    // Caller asks for BUFFERED output (no stream field).
    let response = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&json!({
                    "model": "vendor/served",
                    "messages": [{"role": "user", "content": "hi"}]
                }))?))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .split(';')
            .next()
            .unwrap_or_default(),
        "application/json",
        "buffered caller must never receive raw SSE frames"
    );
    let bytes = response.into_body().collect().await?.to_bytes();
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["choices"][0]["message"]["content"], json!("ok"));
    assert_eq!(body["model"], json!("vendor/served"));

    let upstream_body = captured.lock().await.clone().expect("upstream captured");
    assert_eq!(
        upstream_body["stream"],
        json!(true),
        "stream-mandatory leg must be forced upstream"
    );
    assert!(
        upstream_body.get("stream_options").is_none(),
        "chat-only stream_options leaked onto a Codex Responses request: {upstream_body}"
    );
    assert!(
        upstream_body.get("max_output_tokens").is_none(),
        "Codex-incompatible max_output_tokens leaked upstream"
    );

    task.abort();
    Ok(())
}
