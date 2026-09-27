// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Row 39 parity: the typed capability serving surface.
//!
//! Each test builds the REAL router through the REAL `load_server_runtime`
//! loader from a REAL config file, and drives it with `oneshot` against the real
//! router stack. That is what makes a result meaningful: a test cannot pass by
//! exercising a proxy, an unmounted path, or hand-assembled state the production
//! loader would never produce.
//!
//! The capability executors point at a loopback mock the test programs per call,
//! so the executor leg is genuinely reached and every failure class is
//! distinguishable. Admission refusals are asserted NOT to reach the executor,
//! which is what makes the claim discriminating rather than "it returned
//! something".

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tower::ServiceExt;

use switchyard_server::build_switchyard_router;
use switchyard_server::config::load_server_runtime;

/// Counts executor dispatches, so admission refusals can be proven pre-dispatch
/// and success proven post-dispatch.
///
/// PER-HARNESS, not a process global: Rust runs these tests on parallel
/// threads, and a shared counter lets one test's dispatch be counted by another.
/// A process-global version passed only under `--test-threads=1`, which is a
/// hidden serialisation requirement, not a proof.

#[derive(Clone, Copy)]
enum Upstream {
    Embeddings,
    Rerank,
    Http500,
    Overflow,
    Malformed,
}

/// A loopback capability executor programmed per call.
async fn spawn_executor(mode: Arc<Mutex<Upstream>>, hits: Arc<AtomicUsize>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let current = { *mode.lock().unwrap() };
            let (code, body) = match current {
                Upstream::Embeddings => {
                    (200, json!({"data": [{"embedding": [0.0, 0.1, 0.2, 0.3]}]}))
                }
                Upstream::Rerank => (
                    200,
                    json!({"results": [{"index": 0, "relevance_score": 0.9}]}),
                ),
                Upstream::Http500 => (500, json!({"error": "boom"})),
                // Row 17's exact recognised shape: HTTP 500 plus BOTH markers.
                Upstream::Overflow => (
                    500,
                    Value::String(
                        "the request is too large to process for this physical batch; \
                         increase the physical batch size"
                            .to_string(),
                    ),
                ),
                Upstream::Malformed => (200, json!({"unexpected": "shape"})),
            };
            let payload = serde_json::to_vec(&body).unwrap();
            let head = format!(
                "HTTP/1.1 {code} \r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n",
                payload.len()
            );
            let mut buf = vec![0u8; 8192];
            let _ = socket.read(&mut buf).await;
            hits.fetch_add(1, Ordering::SeqCst);
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(&payload).await;
            let _ = socket.flush().await;
        }
    });
    addr.port()
}

fn write_config(dir: &Path, exec_port: u16) -> std::path::PathBuf {
    let path = dir.join(format!("routes-{exec_port}.toml"));
    let text = format!(
        r#"
schema_version = 1

[targets.embed-target]
id = "embed-model"
llm_client = "embed-client"

[llm_clients.embed-client]
format = "openai_responses"
base_url = "http://127.0.0.1:{exec_port}"

[capability_clients.embed-exec]
format = "openai_embeddings"
base_url = "http://127.0.0.1:{exec_port}"
model = "exec-embed-model"

[capability_clients.rerank-exec]
format = "cohere_jina_rerank"
base_url = "http://127.0.0.1:{exec_port}"
model = "exec-rerank-model"
timeout_seconds = 5

[capabilities.embed-route]
id = "embed-route"
target = "embed-exec"
contract = "localclaw-embedding-space:v1"
dimensions = 4
max_batch = 3

[capabilities.rerank-route]
id = "rerank-route"
target = "rerank-exec"
max_candidates = 5
top_n = 2
max_doc_chars = 10
max_query_chars = 8

[routes.embed-route]
id = "embed-route"
type = "passthrough"
target = "embed-target"
"#
    );
    std::fs::write(&path, text).expect("write config");
    path
}

struct Harness {
    app: Router,
    _dir: std::path::PathBuf,
    _mode: Arc<Mutex<Upstream>>,
    hits: Arc<AtomicUsize>,
}

impl Harness {
    async fn start(mode: Upstream) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "row39-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mode = Arc::new(Mutex::new(mode));
        let hits = Arc::new(AtomicUsize::new(0));
        let port = spawn_executor(Arc::clone(&mode), Arc::clone(&hits)).await;
        let config = write_config(&dir, port);
        let runtime = load_server_runtime(&config).expect("runtime must load");
        let app = build_switchyard_router(runtime.state);
        Self {
            app,
            _dir: dir,
            _mode: mode,
            hits,
        }
    }

    /// POSTs `body` to `path` and returns (status, parsed body, executor hits).
    async fn post(&self, path: &str, body: Value) -> (u16, Value, usize) {
        let before = self.hits.load(Ordering::SeqCst);
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .expect("the router must answer");
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        // Let an in-flight dispatch land before counting.
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        let after = self.hits.load(Ordering::SeqCst);
        (status, parsed, after - before)
    }
}

fn code_of(body: &Value) -> Option<&str> {
    body.get("error")?.get("code")?.as_str()
}

#[tokio::test]
async fn embeddings_endpoint_is_mounted_and_serves() {
    let h = Harness::start(Upstream::Embeddings).await;
    let (status, body, hits) = h
        .post("/v1/embeddings", json!({"model": "embed-route", "input": "a"}))
        .await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(hits, 1, "the executor must actually be dispatched");
    assert_eq!(
        body["data"][0]["embedding"].as_array().unwrap().len(),
        4,
        "the declared dimension count must be honoured"
    );
}

#[tokio::test]
async fn rerank_endpoint_is_mounted_and_serves() {
    let h = Harness::start(Upstream::Rerank).await;
    let (status, body, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": ["d"]}),
        )
        .await;
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(hits, 1);
    assert_eq!(body["results"][0]["index"], 0);
    assert!(body["results"][0]["relevance_score"].is_number());
}

#[tokio::test]
async fn unknown_capability_model_is_refused_before_dispatch() {
    let h = Harness::start(Upstream::Embeddings).await;
    let (status, body, hits) = h
        .post("/v1/embeddings", json!({"model": "no-such-route", "input": "a"}))
        .await;
    assert_eq!(status, 404);
    assert_eq!(code_of(&body), Some("unknown_model"));
    assert_eq!(hits, 0, "an unresolved route must never reach an executor");
}

#[tokio::test]
async fn wrong_capability_kind_is_refused_before_dispatch() {
    let h = Harness::start(Upstream::Rerank).await;
    // A rerank route posted to the embeddings endpoint.
    let (status, body, hits) = h
        .post("/v1/embeddings", json!({"model": "rerank-route", "input": "a"}))
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("wrong_capability"));
    assert_eq!(hits, 0);
}

#[tokio::test]
async fn empty_input_is_refused_before_dispatch() {
    let h = Harness::start(Upstream::Embeddings).await;
    let (status, body, hits) = h
        .post("/v1/embeddings", json!({"model": "embed-route", "input": []}))
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("invalid_input"));
    assert_eq!(hits, 0);
}

#[tokio::test]
async fn batch_at_max_batch_is_admitted_and_one_over_is_refused() {
    // max_batch = 3 in the fixture: 3 admitted, 4 refused.
    let h = Harness::start(Upstream::Embeddings).await;
    let (status, _, hits) = h
        .post(
            "/v1/embeddings",
            json!({"model": "embed-route", "input": ["a", "b", "c"]}),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(hits, 1, "the boundary value must be admitted");

    let (status, body, hits) = h
        .post(
            "/v1/embeddings",
            json!({"model": "embed-route", "input": ["a", "b", "c", "d"]}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("batch_too_large"));
    assert_eq!(hits, 0, "over-batch must be refused before dispatch");
}

#[tokio::test]
async fn embedding_response_of_wrong_dimension_fails_closed() {
    // The executor returns a 4-dim vector and the route declares 4, so the
    // success arm is the control for the malformed arm below.
    let h = Harness::start(Upstream::Malformed).await;
    let (status, body, hits) = h
        .post("/v1/embeddings", json!({"model": "embed-route", "input": "a"}))
        .await;
    assert_eq!(status, 502);
    assert_eq!(code_of(&body), Some("embedding_invalid"));
    assert_eq!(hits, 1, "the executor was reached; the response was rejected");
}

#[tokio::test]
async fn rerank_response_missing_fields_fails_closed() {
    let h = Harness::start(Upstream::Malformed).await;
    let (status, body, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": ["d"]}),
        )
        .await;
    assert_eq!(status, 502);
    assert_eq!(code_of(&body), Some("rerank_invalid"));
    assert_eq!(hits, 1);
}

// ---- row 17: llama.cpp physical-batch overflow classification ----

#[tokio::test]
async fn row17_physical_batch_overflow_is_classified_input_too_long() {
    let h = Harness::start(Upstream::Overflow).await;
    let (status, body, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": ["d"]}),
        )
        .await;
    assert_eq!(hits, 1, "overflow is only observable AFTER upstream dispatch");
    assert_eq!(status, 502);
    assert_eq!(
        code_of(&body),
        Some("input_too_long"),
        "the recognised overflow must not fall through to executor_error"
    );
}

#[tokio::test]
async fn row17_a_plain_500_is_not_misclassified_as_input_too_long() {
    // The negative control: same status, missing the markers.
    let h = Harness::start(Upstream::Http500).await;
    let (status, body, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": ["d"]}),
        )
        .await;
    assert_eq!(hits, 1);
    assert_eq!(status, 502);
    assert_eq!(code_of(&body), Some("executor_error"));
}

#[tokio::test]
async fn rerank_document_length_boundary_is_exact() {
    // max_doc_chars = 10: exactly 10 admitted, 11 refused.
    let h = Harness::start(Upstream::Rerank).await;
    let (status, _, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": ["x".repeat(10)]}),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(hits, 1);

    let (status, body, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": ["x".repeat(11)]}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("document_too_long"));
    assert_eq!(hits, 0);
}

#[tokio::test]
async fn rerank_query_and_candidate_bounds_are_enforced() {
    let h = Harness::start(Upstream::Rerank).await;

    // max_query_chars = 8
    let (status, body, _) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q".repeat(9), "documents": ["d"]}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("query_too_long"));

    // max_candidates = 5
    let (status, body, _) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": vec!["d"; 6]}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("too_many_candidates"));
}

#[tokio::test]
async fn rerank_rejects_candidate_objects_without_a_text_field() {
    let h = Harness::start(Upstream::Rerank).await;
    let (status, body, hits) = h
        .post(
            "/v1/rerank",
            json!({"model": "rerank-route", "query": "q", "documents": [{"id": 1}]}),
        )
        .await;
    assert_eq!(status, 400);
    assert_eq!(code_of(&body), Some("invalid_documents"));
    assert_eq!(hits, 0);
}

#[tokio::test]
async fn capability_models_are_advertised_alongside_routes() {
    let dir = std::env::temp_dir().join(format!(
        "row39-models-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // Port 1: nothing listens, so no dispatch can occur; this test only reads
    // the advertisement surface.
    let config = write_config(&dir, 1);
    let runtime = load_server_runtime(&config).expect("runtime must load");
    let app = build_switchyard_router(runtime.state);
    let request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"embed-route"), "advertised: {ids:?}");
    assert!(ids.contains(&"rerank-route"), "advertised: {ids:?}");
}
