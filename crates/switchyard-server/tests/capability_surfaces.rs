// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the capability serving surfaces.
//!
//! These drive the real router against a mock engine, so they assert what a caller
//! actually receives: which engine was chosen, which contract the response is tagged
//! with, and whether a declared limit is enforced before anything reaches an engine.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use switchyard_runner::Runner;
use tower::ServiceExt;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn serve(config: &str) -> axum::Router {
    let runner = Runner::from_toml(config).expect("runner builds");
    switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    )
}

/// Config with two embedding spaces and one rerank capability, each pointing at the
/// engine the test chooses.
fn config(engine: &str) -> String {
    config_with(engine, engine, engine)
}

fn config_with(engine_a: &str, engine_b: &str, reranker: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.responses]
format = "openai_responses"
base_url = "https://completion.test/v1"

[targets.only]
id = "vendor/only"
llm_client = "responses"

[routes.only]
id = "probe/only"
type = "passthrough"
target = "only"

[capability_clients.engine_a]
format = "openai_embeddings"
base_url = "{engine_a}"
model = "Qwen3-Embedding-0.6B"

[capability_clients.engine_b]
format = "openai_embeddings"
base_url = "{engine_b}"
model = "lfm-350m"

[capability_clients.reranker]
format = "cohere_jina_rerank"
base_url = "{reranker}"
model = "qwen3-reranker"

[capabilities.space_a]
id = "switchyard-embedding-local"
target = "engine_a"
contract = "localclaw-embedding-space:v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.space_b]
id = "switchyard-embedding-free"
target = "engine_b"
contract = "localclaw-embedding-space:lfm2.5-350m-v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.rerank]
id = "switchyard-rerank-local"
target = "reranker"
max_candidates = 64
top_n = 5
max_doc_chars = 1024
max_query_chars = 512
"#
    )
}

async fn post(router: axum::Router, uri: &str, body: Value) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    (status, headers, bytes.to_vec())
}

#[tokio::test]
async fn embedding_dispatches_to_the_named_space_and_returns_its_contract() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_partial_json(json!({"model": "Qwen3-Embedding-0.6B"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": vec![0.1, 0.2, 0.3]}],
            "model": "Qwen3-Embedding-0.6B",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        })))
        .expect(1)
        .mount(&engine)
        .await;

    let router = serve(&config(&engine.uri())).await;
    let (status, headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-local", "input": ["hello"]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers
            .get("x-switchyard-embedding-contract")
            .and_then(|value| value.to_str().ok()),
        Some("localclaw-embedding-space:v1"),
        "the response must say which space produced the vector"
    );
    assert_eq!(
        headers
            .get("x-switchyard-embedding-dimensions")
            .and_then(|value| value.to_str().ok()),
        Some("1024")
    );
    // The public capability id must not be forwarded as the provider model name.
    assert!(String::from_utf8_lossy(&body).contains("Qwen3-Embedding-0.6B"));
}

#[tokio::test]
async fn two_embedding_spaces_reach_their_own_engines_and_keep_separate_contracts() {
    let engine_a = MockServer::start().await;
    let engine_b = MockServer::start().await;
    for (server, model) in [(&engine_a, "Qwen3-Embedding-0.6B"), (&engine_b, "lfm-350m")] {
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(body_partial_json(json!({"model": model})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": vec![0.5, 0.5]}],
            })))
            .expect(1)
            .mount(server)
            .await;
    }

    let router = serve(&config_with(&engine_a.uri(), &engine_b.uri(), &engine_a.uri())).await;

    let (_status, local_headers, _) = post(
        router.clone(),
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-local", "input": ["same text"]}),
    )
    .await;
    let (_status, free_headers, _) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-free", "input": ["same text"]}),
    )
    .await;

    let contract = |headers: &axum::http::HeaderMap| {
        headers
            .get("x-switchyard-embedding-contract")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string()
    };
    assert_eq!(contract(&local_headers), "localclaw-embedding-space:v1");
    assert_eq!(contract(&free_headers), "localclaw-embedding-space:lfm2.5-350m-v1");
    assert_ne!(
        contract(&local_headers),
        contract(&free_headers),
        "equal width must not collapse two spaces into one identity"
    );
    // engine_b must have received exactly one call, so `free` was not served by engine_a.
    assert_eq!(engine_b.received_requests().await.unwrap_or_default().len(), 1);
}

#[tokio::test]
async fn rerank_returns_engine_scores_and_honours_top_n() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .and(body_partial_json(json!({"model": "qwen3-reranker", "top_n": 2})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "model": "qwen3-reranker",
            "results": [
                {"index": 0, "relevance_score": 0.9974},
                {"index": 2, "relevance_score": 0.9848},
            ],
            "usage": {"prompt_tokens": 244, "total_tokens": 244},
        })))
        .mount(&engine)
        .await;

    let router = serve(&config(&engine.uri())).await;
    let (status, _headers, body) = post(
        router,
        "/v1/rerank",
        json!({
            "model": "switchyard-rerank-local",
            "query": "routing policy",
            "documents": ["routes requests", "a cat", "capability config"],
            "top_n": 2,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let parsed: Value = serde_json::from_slice(&body).expect("rerank response is JSON");
    let results = parsed["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2);
    // Scores pass through untouched: they are the engine's contract, not ours to reshape.
    assert_eq!(results[0]["index"], 0);
    assert_eq!(results[0]["relevance_score"], 0.9974);
    assert_eq!(results[1]["index"], 2);
}

#[tokio::test]
async fn declared_limits_are_rejected_before_reaching_an_engine() {
    let engine = MockServer::start().await;
    let router = serve(&config(&engine.uri())).await;

    let too_many_texts: Vec<String> = (0..65).map(|index| format!("text{index}")).collect();
    let (status, _headers, body) = post(
        router.clone(),
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-local", "input": too_many_texts}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("accepts at most 64"),
        "{}",
        String::from_utf8_lossy(&body)
    );

    let too_many_documents: Vec<String> = (0..65).map(|index| format!("doc{index}")).collect();
    let (status, _headers, body) = post(
        router.clone(),
        "/v1/rerank",
        json!({"model": "switchyard-rerank-local", "query": "q", "documents": too_many_documents}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("accepts at most 64"));

    let (status, _headers, body) = post(
        router.clone(),
        "/v1/rerank",
        json!({"model": "switchyard-rerank-local", "query": "q", "documents": ["x".repeat(1025)]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("1024 characters"));

    let (status, _headers, body) = post(
        router,
        "/v1/rerank",
        json!({"model": "switchyard-rerank-local", "query": "q", "documents": ["d"], "top_n": 99}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("exceeds"));

    // Nothing above should have reached an engine.
    assert_eq!(engine.received_requests().await.unwrap_or_default().len(), 0);
}

/// Probes immediately across the contract boundary rather than only obviously invalid
/// values. `top_n = 99` passing or failing says nothing about whether the declared
/// `top_n = 5` ceiling is enforced; `top_n = 6` is the discriminating case.
#[tokio::test]
async fn rerank_top_n_ceiling_is_enforced_at_the_boundary() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "results": [{"index": 0, "relevance_score": 0.5}],
        })))
        .mount(&engine)
        .await;

    let router = serve(&config(&engine.uri())).await;
    let documents: Vec<String> = (0..8).map(|index| format!("doc{index}")).collect();
    let request = |top_n: u32| {
        json!({
            "model": "switchyard-rerank-local",
            "query": "q",
            "documents": documents,
            "top_n": top_n,
        })
    };

    // At the declared ceiling: accepted.
    let (status, _headers, _body) = post(
        router.clone(),
        "/v1/rerank",
        request(5),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "top_n 5 is the declared ceiling");

    // One past it: rejected, even though it is far below max_candidates 64.
    let (status, _headers, body) = post(router.clone(), "/v1/rerank", request(6)).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "top_n 6 must not pass a contract declaring top_n 5"
    );
    let message = String::from_utf8_lossy(&body).to_string();
    assert!(message.contains("top_n 6"), "{message}");
    assert!(
        message.contains("max_candidates") == false || message.contains("top_n 5"),
        "the error must name the declared ceiling, not the candidate pool: {message}"
    );

    // Omitting top_n uses the declared value and is accepted.
    let (status, _headers, _body) = post(
        router,
        "/v1/rerank",
        json!({"model": "switchyard-rerank-local", "query": "q", "documents": documents}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn an_unknown_or_mistyped_capability_is_refused_rather_than_guessed() {
    let router = serve(&config("https://engine.test")).await;

    // Unknown capability id: a 404 naming what the deployment does expose.
    let (status, _headers, body) = post(
        router.clone(),
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-nope", "input": ["x"]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let message = String::from_utf8_lossy(&body).to_string();
    assert!(message.contains("switchyard-embedding-local"), "{message}");
    assert!(message.contains("switchyard-rerank-local"), "{message}");

    // A completion route id is not a capability.
    let (status, _headers, _) = post(
        router.clone(),
        "/v1/rerank",
        json!({"model": "probe/only", "query": "q", "documents": ["d"]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Asking the rerank capability for embeddings is a contract error, not a guess.
    let (status, _headers, body) = post(
        router.clone(),
        "/v1/embeddings",
        json!({"model": "switchyard-rerank-local", "input": ["x"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("does not serve embeddings"));

    // Asking the embedding capability to rerank, likewise.
    let (status, _headers, body) = post(
        router,
        "/v1/rerank",
        json!({"model": "switchyard-embedding-local", "query": "q", "documents": ["d"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("does not serve reranking"));
}

#[tokio::test]
async fn an_unreachable_engine_fails_rather_than_returning_an_empty_result() {
    let router = serve(&config("http://127.0.0.1:1")).await;
    let (status, _headers, _body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-local", "input": ["x"]}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_GATEWAY,
        "an unreachable engine must be reported, not silently empty"
    );
}
