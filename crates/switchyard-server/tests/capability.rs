// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the capability endpoints (embeddings + rerank).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_server::build_switchyard_router;
use switchyard_server::config::load_server_state;
use tokio::net::TcpListener;
use tower::util::ServiceExt;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const EMBED_TOML: &str = r#"
schema_version = 1

[targets]

[routes.noop]
id = "switchyard/noop"
type = "noop"

[capability_clients.test-embed]
format = "openai_embeddings"
base_url = "http://127.0.0.1:PORT"
model = "Qwen3-Embedding-0.6B"

[capability_clients.test-rerank]
format = "cohere_jina_rerank"
base_url = "http://127.0.0.1:PORT"
model = "qwen3-reranker-0.6b-q8_0"

[capabilities.embed]
id = "localclaw/embed"
target = "test-embed"
contract = "localclaw-embedding-space:v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.rerank]
id = "localclaw/rerank"
target = "test-rerank"
max_candidates = 32
top_n = 5
max_doc_chars = 4096
max_query_chars = 2048
"#;

fn vector_1024() -> Vec<f64> {
    let mut v = vec![0.0f64; 1024];
    v[0] = 0.1;
    v[1] = -0.2;
    v
}

async fn embed_upstream(State(calls): State<Arc<tokio::sync::Mutex<Vec<Value>>>>, Json(_): Json<Value>) -> Response {
    calls.lock().await.push(json!({"kind": "embed"}));
    let body = json!({
        "object": "list",
        "model": "Qwen3-Embedding-0.6B",
        "data": [{"object": "embedding", "embedding": vector_1024(), "index": 0}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1}
    });
    (StatusCode::OK, Json(body)).into_response()
}

async fn embed_upstream_bad_dims(State(calls): State<Arc<tokio::sync::Mutex<Vec<Value>>>>, Json(_): Json<Value>) -> Response {
    calls.lock().await.push(json!({"kind": "embed-bad-dims"}));
    let body = json!({
        "object": "list",
        "model": "Qwen3-Embedding-0.6B",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1}
    });
    (StatusCode::OK, Json(body)).into_response()
}

async fn rerank_upstream(State(calls): State<Arc<tokio::sync::Mutex<Vec<Value>>>>, Json(_): Json<Value>) -> Response {
    calls.lock().await.push(json!({"kind": "rerank"}));
    let body = json!({
        "results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]
    });
    (StatusCode::OK, Json(body)).into_response()
}

async fn rerank_upstream_malformed(State(calls): State<Arc<tokio::sync::Mutex<Vec<Value>>>>, Json(_): Json<Value>) -> Response {
    calls.lock().await.push(json!({"kind": "rerank-malformed"}));
    let body = json!({"results": [{"index": "x", "relevance_score": 0.9}]});
    (StatusCode::OK, Json(body)).into_response()
}

async fn upstream_404() -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": "nope"}))).into_response()
}

async fn build_test_app(upstream_port: u16, _route: &str) -> TestResult<Router> {
    let toml = EMBED_TOML.replace("PORT", &upstream_port.to_string());
    let state = load_server_state_from(toml)?;
    Ok(build_switchyard_router(state))
}

fn load_server_state_from(toml: String) -> TestResult<switchyard_server::ServerState> {
    // Unique path per caller+pid: tests run concurrently and previously shared one
    // routes.toml, racing each other's config writes (caused intermittent
    // "missing field" / empty-file parse failures).
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "switchyard-cap-test-{}-{}",
        std::process::id(),
        uniq
    ));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("routes.toml");
    std::fs::write(&path, toml)?;
    Ok(load_server_state(&path)?)
}

async fn start_upstream(app: Router) -> TestResult<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(addr.port())
}

async fn send(app: &Router, method: &str, path: &str, body: Value) -> TestResult<(StatusCode, Value)> {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))?;
    let response = app.clone().oneshot(request).await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Ok((status, value))
}

#[tokio::test]
async fn embeddings_route_proxies_and_validates_dims() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    let (status, value) = send(&app, "POST", "/v1/embeddings", json!({
        "model": "localclaw/embed",
        "input": "hello world"
    })).await?;
    assert_eq!(status, StatusCode::OK, "embedding should succeed: {value}");
    let dims = value["data"][0]["embedding"].as_array().map(|a| a.len());
    assert_eq!(dims, Some(1024));
    Ok(())
}

#[tokio::test]
async fn embeddings_unknown_model_404() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    let (status, _) = send(&app, "POST", "/v1/embeddings", json!({
        "model": "localclaw/nope", "input": "x"
    })).await?;
    assert_eq!(status, StatusCode::NOT_FOUND);
    Ok(())
}

#[tokio::test]
async fn embeddings_dimension_mismatch_fails_closed() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream_bad_dims).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    let (status, value) = send(&app, "POST", "/v1/embeddings", json!({
        "model": "localclaw/embed", "input": "x"
    })).await?;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "dim mismatch must fail closed: {value}");
    Ok(())
}

#[tokio::test]
async fn rerank_route_proxies_and_validates() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    let (status, value) = send(&app, "POST", "/v1/rerank", json!({
        "model": "localclaw/rerank",
        "query": "how do I restart turnstone",
        "documents": [{"id": "a", "text": "restart with systemctl"}, {"id": "b", "text": "gold price"}],
        "top_n": 2
    })).await?;
    assert_eq!(status, StatusCode::OK, "rerank should succeed: {value}");
    assert_eq!(value["results"][0]["index"], 0);
    Ok(())
}

#[tokio::test]
async fn rerank_malformed_response_fails_closed() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream_malformed).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    let (status, _) = send(&app, "POST", "/v1/rerank", json!({
        "model": "localclaw/rerank",
        "query": "q",
        "documents": ["a", "b"]
    })).await?;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    Ok(())
}

#[tokio::test]
async fn rerank_bounds_enforced() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    // 40 candidates > max_candidates 32
    let docs = (0..40).map(|i| format!("doc {i}")).collect::<Vec<_>>();
    let (status, _) = send(&app, "POST", "/v1/rerank", json!({
        "model": "localclaw/rerank", "query": "q", "documents": docs
    })).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    Ok(())
}

#[tokio::test]
async fn capability_routes_advertised_in_models() -> TestResult {
    let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/embeddings", post(embed_upstream).with_state(Arc::clone(&calls)))
        .route("/v1/rerank", post(rerank_upstream).with_state(Arc::clone(&calls)))
        .route("/missing", post(upstream_404));
    let port = start_upstream(app).await?;
    let toml = EMBED_TOML.replace("PORT", &port.to_string());
    let state = load_server_state_from(toml)?;
    let app = build_switchyard_router(state);

    let (status, value) = send(&app, "GET", "/v1/models", json!({})).await?;
    assert_eq!(status, StatusCode::OK);
    let ids = value["data"].as_array().map(|a| a.iter().filter_map(|m| m["id"].as_str()).collect::<Vec<_>>()).unwrap_or_default();
    assert!(ids.contains(&"localclaw/embed"), "models should advertise localclaw/embed: {ids:?}");
    assert!(ids.contains(&"localclaw/rerank"), "models should advertise localclaw/rerank: {ids:?}");
    Ok(())
}
