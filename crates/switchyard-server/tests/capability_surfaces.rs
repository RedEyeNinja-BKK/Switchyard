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
use wiremock::matchers::{body_partial_json, header, method, path};
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

async fn post(
    router: axum::Router,
    uri: &str,
    body: Value,
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
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

    let router = serve(&config_with(
        &engine_a.uri(),
        &engine_b.uri(),
        &engine_a.uri(),
    ))
    .await;

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
    assert_eq!(
        contract(&free_headers),
        "localclaw-embedding-space:lfm2.5-350m-v1"
    );
    assert_ne!(
        contract(&local_headers),
        contract(&free_headers),
        "equal width must not collapse two spaces into one identity"
    );
    // engine_b must have received exactly one call, so `free` was not served by engine_a.
    assert_eq!(
        engine_b.received_requests().await.unwrap_or_default().len(),
        1
    );
}

#[tokio::test]
async fn rerank_returns_engine_scores_and_honours_top_n() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .and(body_partial_json(
            json!({"model": "qwen3-reranker", "top_n": 2}),
        ))
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
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
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
    let (status, _headers, _body) = post(router.clone(), "/v1/rerank", request(5)).await;
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
        !message.contains("max_candidates") || message.contains("top_n 5"),
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
async fn stephen_embedding_rejects_wrong_engine_and_dimension_without_claiming_a_contract() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list", "model": "nvidia/nemotron-3.5-embed-1b:free",
            "data": [{"index": 0, "embedding": vec![0.1; 2048]}],
        })))
        .mount(&engine)
        .await;
    let cfg = config(&engine.uri())
        .replace("Qwen3-Embedding-0.6B", "nvidia/nemotron-3.5-embed-8b:free")
        .replace("switchyard-embedding-local", "ST_Embed")
        .replace("dimensions = 1024", "dimensions = 4096");
    let (status, headers, body) = post(
        serve(&cfg).await,
        "/v1/embeddings",
        json!({"model": "ST_Embed", "input": ["hello"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(headers.get("x-switchyard-embedding-contract").is_none());
    assert!(String::from_utf8_lossy(&body).contains("ST_Embed"));
}

#[tokio::test]
async fn stephen_embedding_valid_batch_passes_with_declared_identity_and_width() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_partial_json(json!({
            "model": "nvidia/nemotron-3.5-embed-8b:free", "input": ["one", "two"],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list", "model": "nvidia/nemotron-3.5-embed-8b:free",
            "data": [
                {"index": 0, "embedding": vec![0.1; 4096]},
                {"index": 1, "embedding": vec![0.2; 4096]},
            ],
        })))
        .expect(1)
        .mount(&engine)
        .await;
    let cfg = config(&engine.uri())
        .replace("Qwen3-Embedding-0.6B", "nvidia/nemotron-3.5-embed-8b:free")
        .replace("switchyard-embedding-local", "ST_Embed")
        .replace("dimensions = 1024", "dimensions = 4096");
    let (status, headers, body) = post(
        serve(&cfg).await,
        "/v1/embeddings",
        json!({"model": "ST_Embed", "input": ["one", "two"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(headers["x-switchyard-embedding-dimensions"], "4096");
    assert_eq!(
        headers["x-switchyard-embedding-contract"],
        "localclaw-embedding-space:v1"
    );
}

#[tokio::test]
async fn stephen_rerank_rejects_chat_completion_instead_of_results() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "chat.completion", "model": "nvidia/nemotron-3.5-super-vl:free",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
        })))
        .mount(&engine)
        .await;
    let cfg = config(&engine.uri())
        .replace("switchyard-rerank-local", "ST_Rerank")
        .replace("qwen3-reranker", "nvidia/nemotron-3.5-rerank-8b:free");
    let (status, _, body) = post(
        serve(&cfg).await,
        "/v1/rerank",
        json!({"model": "ST_Rerank", "query": "capital of France", "documents": ["Paris", "a cat"], "top_n": 2}),
    ).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(String::from_utf8_lossy(&body).contains("ST_Rerank"));
}

#[tokio::test]
async fn stephen_rerank_valid_results_pass_with_top_n_and_ranked_scores() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .and(body_partial_json(json!({
            "model": "nvidia/nemotron-3.5-rerank-8b:free", "top_n": 2,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "nvidia/nemotron-3.5-rerank-8b:free",
            "results": [
                {"index": 0, "relevance_score": 0.98},
                {"index": 1, "relevance_score": 0.03},
            ],
        })))
        .expect(1)
        .mount(&engine)
        .await;
    let cfg = config(&engine.uri())
        .replace("switchyard-rerank-local", "ST_Rerank")
        .replace("qwen3-reranker", "nvidia/nemotron-3.5-rerank-8b:free");
    let (status, _, body) = post(
        serve(&cfg).await, "/v1/rerank",
        json!({"model": "ST_Rerank", "query": "capital of France", "documents": ["Paris is the capital of France", "A cat sleeps"], "top_n": 2}),
    ).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let parsed: Value = serde_json::from_slice(&body).expect("rerank JSON");
    assert_eq!(parsed["results"][0]["index"], 0);
    assert_eq!(parsed["results"][0]["relevance_score"], 0.98);
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

/// The substitution that matters: Embed 1B answering at the requested width. Width
/// alone cannot detect it, so identity is checked independently of dimension.
#[tokio::test]
async fn stephen_embed_rejects_a_substituted_sibling_even_at_the_right_width() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list", "model": "nvidia/nemotron-3.5-embed-1b:free",
            "data": [{"index": 0, "embedding": vec![0.1; 4096]}],
        })))
        .mount(&engine)
        .await;
    let cfg = config(&engine.uri())
        .replace("Qwen3-Embedding-0.6B", "nvidia/nemotron-3.5-embed-8b:free")
        .replace("switchyard-embedding-local", "ST_Embed")
        .replace("dimensions = 1024", "dimensions = 4096");
    let (status, headers, body) = post(
        serve(&cfg).await,
        "/v1/embeddings",
        json!({"model": "ST_Embed", "input": ["hello"]}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_GATEWAY,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(
        headers.get("x-switchyard-embedding-contract").is_none(),
        "a substituted engine must not be attested under the agreed contract"
    );
}

/// A provider may name the same weights by their canonical slug. That is the same
/// model, so it must be accepted rather than failing every real call.
#[tokio::test]
async fn stephen_embed_accepts_the_canonical_slug_for_the_same_weights() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list", "model": "nvidia/nemotron-3.5-embed-8b",
            "data": [{"index": 0, "embedding": vec![0.25; 4096]}],
        })))
        .mount(&engine)
        .await;
    let cfg = config(&engine.uri())
        .replace("Qwen3-Embedding-0.6B", "nvidia/nemotron-3.5-embed-8b:free")
        .replace("switchyard-embedding-local", "ST_Embed")
        .replace("dimensions = 1024", "dimensions = 4096");
    let (status, headers, body) = post(
        serve(&cfg).await,
        "/v1/embeddings",
        json!({"model": "ST_Embed", "input": ["hello"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(headers["x-switchyard-embedding-dimensions"], "4096");
}

/// A capability whose credential is absent must fail closed. Previously the request
/// was sent unauthenticated and the provider's 401 was relayed, which reads like a
/// credential problem at the engine rather than a missing deployment value.
#[tokio::test]
async fn a_missing_capability_credential_fails_closed_without_an_upstream_call() {
    let engine = MockServer::start().await;
    let cfg = config(&engine.uri()).replace(
        "model = \"Qwen3-Embedding-0.6B\"",
        "model = \"Qwen3-Embedding-0.6B\"\napi_key_env = \"SWITCHYARD_TEST_UNSET_CAPABILITY_KEY\"",
    );
    assert!(
        std::env::var("SWITCHYARD_TEST_UNSET_CAPABILITY_KEY").is_err(),
        "the fixture assumes the variable is unset in this process"
    );
    let (status, _headers, body) = post(
        serve(&cfg).await,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-local", "input": ["x"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        String::from_utf8_lossy(&body).contains("credential unavailable"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0,
        "an unauthenticated call must not be attempted"
    );
}

/// A Google embedding capability. Google has no bulk embed endpoint: one content per
/// request, named in the path, authenticated with `x-goog-api-key`.
fn google_config(engine: &str, max_input_tokens: u32) -> String {
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

[capability_clients.google]
format = "google_embed_content"
base_url = "{engine}"
model = "gemini-embedding-001"
task_type = "RETRIEVAL_DOCUMENT"
api_key_env = "SWITCHYARD_TEST_GOOGLE_KEY"

[capabilities.gem]
id = "switchyard-embedding-google"
target = "google"
contract = "google-embedding-001:v1"
dimensions = 3
normalization = "L2"
max_batch = 8
max_input_tokens = {max_input_tokens}
"#
    )
}

/// A caller sends one OpenAI-shaped batch; Google answers one content at a time. The
/// surface must therefore fan out and reassemble in input order, or the caller receives
/// vectors whose `index` no longer means what it says.
#[tokio::test]
async fn a_google_embedding_capability_assembles_one_response_from_one_call_per_input() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE_KEY", "test-key") };
    let engine = MockServer::start().await;
    for (text, values) in [
        ("alpha", json!([0.1, 0.2, 0.3])),
        ("beta", json!([0.4, 0.5, 0.6])),
    ] {
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-embedding-001:embedContent"))
            .and(header("x-goog-api-key", "test-key"))
            .and(body_partial_json(json!({
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "task_type": "RETRIEVAL_DOCUMENT",
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"embedding": {"values": values}})),
            )
            .expect(1)
            .mount(&engine)
            .await;
    }

    let router = serve(&google_config(&engine.uri(), 2048)).await;
    let (status, headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google", "input": ["alpha", "beta"]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: Value = serde_json::from_slice(&body).expect("response is json");
    assert_eq!(value["data"][0]["index"], json!(0));
    assert_eq!(value["data"][0]["embedding"], json!([0.1, 0.2, 0.3]));
    assert_eq!(value["data"][1]["index"], json!(1));
    assert_eq!(value["data"][1]["embedding"], json!([0.4, 0.5, 0.6]));
    assert_eq!(
        headers
            .get("x-switchyard-embedding-contract")
            .and_then(|value| value.to_str().ok()),
        Some("google-embedding-001:v1"),
        "the response must say which space produced the vectors"
    );
}

/// A token count cannot be computed here, but a token never covers less than one byte,
/// so bytes bound tokens from above. The refusal must happen before the wire, and must
/// say which input and which limit, because "too long" is not actionable.
#[tokio::test]
async fn a_google_capability_refuses_an_input_past_its_token_limit_without_calling_the_engine() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE_KEY", "test-key") };
    let engine = MockServer::start().await;
    // Nothing is mounted: a leaked call would answer 404 and be visible below.
    let router = serve(&google_config(&engine.uri(), 16)).await;
    let (status, _headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google", "input": ["ok", "a".repeat(17)]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("input[1]"), "{text}");
    assert!(text.contains("16"), "{text}");
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0,
        "an over-limit input must not reach the engine"
    );
}

/// Returning the vectors that succeeded would hand back a shorter, silently re-indexed
/// batch. A partial batch is a failed batch, and it must not carry the contract header,
/// which would otherwise attest to a vector space the caller did not fully receive.
#[tokio::test]
async fn a_google_capability_returns_no_vectors_when_one_input_fails() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"content": {"parts": [{"text": "alpha"}]}}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embedding": {"values": [0.1, 0.2, 0.3]}})),
        )
        .mount(&engine)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"content": {"parts": [{"text": "beta"}]}}),
        ))
        .respond_with(
            ResponseTemplate::new(500)
                .set_body_json(json!({"error": {"message": "backend exploded"}})),
        )
        .mount(&engine)
        .await;

    let router = serve(&google_config(&engine.uri(), 2048)).await;
    let (status, headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google", "input": ["alpha", "beta"]}),
    )
    .await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("input[1]"), "{text}");
    assert!(text.contains("backend exploded"), "{text}");
    assert!(
        !text.contains("\"embedding\""),
        "a failed batch must not carry vectors: {text}"
    );
    assert!(
        headers.get("x-switchyard-embedding-contract").is_none(),
        "the contract must not be attested on a failed batch"
    );
}

/// `gemini-embedding-2` speaks a different codec: width is a request field and a task is a
/// text prefix rather than a `task_type`. The caller's contract is unchanged.
fn gemini2_config(engine: &str, dims: u32, prefix: &str, extra_client: &str) -> String {
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

[capability_clients.google2]
format = "google_embed_content_v2"
base_url = "{engine}"
model = "gemini-embedding-2"
api_key_env = "SWITCHYARD_TEST_GOOGLE2_KEY"
{extra_client}

[capabilities.gem2]
id = "switchyard-embedding-google2"
target = "google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = {dims}
normalization = "L2"
max_batch = 8
max_input_tokens = 8192
{task_prefix}
"#,
        task_prefix = if prefix.is_empty() {
            String::new()
        } else {
            format!("task_prefix = \"{prefix}\"")
        }
    )
}

fn unit_vector(dims: usize, scale: f64) -> Value {
    let each = 1.0 / (dims as f64).sqrt();
    json!(vec![each * scale; dims])
}

#[tokio::test]
async fn a_gemini_embedding_2_capability_sends_the_prefix_and_the_requested_width() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    for text in ["alpha", "beta"] {
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-embedding-2:embedContent"))
            .and(header("x-goog-api-key", "test-key"))
            .and(body_partial_json(json!({
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": format!("task: search result | query: {text}")}]},
                "output_dimensionality": 128,
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"embedding": {"values": unit_vector(128, 1.0)}})),
            )
            .expect(1)
            .mount(&engine)
            .await;
    }

    let router = serve(&gemini2_config(
        &engine.uri(),
        128,
        "task: search result | query: ",
        "",
    ))
    .await;
    let (status, headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["alpha", "beta"]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let value: Value = serde_json::from_slice(&body).expect("response is json");
    assert_eq!(value["data"][0]["index"], json!(0));
    assert_eq!(value["data"][1]["index"], json!(1));
    assert_eq!(
        headers
            .get("x-switchyard-embedding-contract")
            .and_then(|value| value.to_str().ok()),
        Some("google-embedding-space:gemini-embedding-2-v1")
    );
}

/// The prefix is sent with the text, so it counts against the model's input limit. A budget
/// that ignored it would let an over-limit request reach the engine.
#[tokio::test]
async fn a_gemini_embedding_2_capability_counts_the_prefix_against_the_input_limit() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    let mut config = gemini2_config(&engine.uri(), 128, "task: x | query: ", "");
    config = config.replace("max_input_tokens = 8192", "max_input_tokens = 17");
    let router = serve(&config).await;
    // 26-byte prefix + a 10-byte text is over 17 bytes even though the text alone is not.
    let (status, _headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["0123456789"]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("input[0]"), "{text}");
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0,
        "an over-limit input must not reach the engine"
    );
}

/// Width is negotiated per call, so a vector of a different width is not the vector the
/// contract promises. Returning it would make the caller's index silently wrong.
#[tokio::test]
async fn a_gemini_embedding_2_capability_refuses_a_vector_at_the_wrong_width() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embedding": {"values": unit_vector(127, 1.0)}})),
        )
        .mount(&engine)
        .await;
    let router = serve(&gemini2_config(&engine.uri(), 128, "", "")).await;
    let (status, headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["alpha"]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(String::from_utf8_lossy(&body).contains("dimension"));
    assert!(
        headers.get("x-switchyard-embedding-contract").is_none(),
        "a refused vector must not be attested to the caller"
    );
}

/// A declared normalization is a claim about the value space. If the engine's vector is not
/// unit length, the contract is not what the caller was promised.
#[tokio::test]
async fn a_gemini_embedding_2_capability_refuses_a_vector_that_is_not_normalised() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embedding": {"values": unit_vector(128, 10.0)}})),
        )
        .mount(&engine)
        .await;
    let router = serve(&gemini2_config(&engine.uri(), 128, "", "")).await;
    let (status, _headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["alpha"]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        String::from_utf8_lossy(&body).contains("L2-normalised"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

/// A fan-out makes one provider request per input, so the budget must be charged per
/// physical call. With one request authorised, the second input must not be sent.
#[tokio::test]
async fn a_gemini_embedding_2_capability_charges_the_daily_budget_per_physical_call() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embedding": {"values": unit_vector(128, 1.0)}})),
        )
        .mount(&engine)
        .await;
    let state =
        std::env::temp_dir().join(format!("switchyard-gemini2-budget-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&state);
    let config = format!(
        "{}\n[quota]\nstate_dir = \"{}\"\n",
        gemini2_config(&engine.uri(), 128, "", "daily_request_limit = 1"),
        state.display()
    );
    let router = serve(&config).await;
    let (status, _headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["alpha", "beta"]}),
    )
    .await;

    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("daily request allowance"), "{text}");
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        1,
        "the spent budget authorised exactly one physical call"
    );
}

/// A minute window is taken before the call as well, so a caller cannot fan one request out
/// into a burst the provider would refuse.
#[tokio::test]
async fn a_gemini_embedding_2_capability_charges_the_minute_window_per_physical_call() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embedding": {"values": unit_vector(128, 1.0)}})),
        )
        .mount(&engine)
        .await;
    let router = serve(&gemini2_config(
        &engine.uri(),
        128,
        "",
        "requests_per_minute = 1",
    ))
    .await;
    let (status, _headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["alpha", "beta"]}),
    )
    .await;

    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert!(
        String::from_utf8_lossy(&body).contains("per-minute requests_per_minute"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        1
    );
}

/// A held capability is inventory, not a fallback: naming it must fail locally, name what is
/// actually available, and reach no engine.
#[tokio::test]
async fn a_held_capability_is_refused_locally_with_no_egress() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_GOOGLE2_KEY", "test-key") };
    let engine = MockServer::start().await;
    let config = format!(
        "{gemini2_config}\nheld = \"not qualified: response shape unobserved\"\n",
        gemini2_config = gemini2_config(&engine.uri(), 128, "", "")
    );
    let router = serve(&config).await;
    let (status, _headers, body) = post(
        router,
        "/v1/embeddings",
        json!({"model": "switchyard-embedding-google2", "input": ["alpha"]}),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("No capability registered"), "{text}");
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0,
        "a held capability must never reach its engine"
    );
}

// =====================================================================
// Speech synthesis and transcription
// =====================================================================

/// Config with one speech capability and one transcription capability, each pointing at the
/// engine the test chooses.
fn audio_config(engine: &str, speech_extra: &str, transcribe_extra: &str) -> String {
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

[capability_clients.speech]
format = "google_speech"
base_url = "{engine}"
model = "gemini-2.5-flash-preview-tts"
api_key_env = "SWITCHYARD_TEST_AUDIO_KEY"
{speech_extra}

[capability_clients.transcriber]
format = "google_transcribe"
base_url = "{engine}"
model = "gemini-3.5-transcribe"
api_key_env = "SWITCHYARD_TEST_AUDIO_KEY"
{transcribe_extra}

[capabilities.tts]
id = "switchyard-tts"
target = "speech"
voice = "Kore"
audio_format = "wav"
audio_sample_rate = 24000
max_input_tokens = 8192

[capabilities.asr]
id = "switchyard-transcribe"
target = "transcriber"
task_prefix = "Transcribe this recording verbatim."
max_input_tokens = 98304
max_audio_seconds = 300
max_audio_bytes = 15728640
"#
    )
}

/// A 16-bit mono WAV of the requested length, so a test can state a duration exactly.
fn wav_bytes(seconds: f64, sample_rate: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let bits: u16 = 16;
    let frames = (seconds * f64::from(sample_rate)).round() as usize;
    let data_len = frames * 2;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * u32::from(block_align);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    out.extend(vec![7u8; data_len]);
    out
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn pcm16(samples: usize) -> Vec<u8> {
    vec![0u8; samples * 2]
}

#[tokio::test]
async fn speech_asks_for_the_audio_modality_and_returns_a_wav_container() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    let pcm = pcm16(480); // 20 ms at 24 kHz
    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.5-flash-preview-tts:generateContent",
        ))
        .and(header("x-goog-api-key", "test-key"))
        .and(body_partial_json(json!({
            "contents": [{"parts": [{"text": "hello there"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"inlineData": {
                "mimeType": "audio/L16;codec=pcm;rate=24000",
                "data": b64(&pcm)
            }}]}}]
        })))
        .expect(1)
        .mount(&engine)
        .await;

    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, headers, body) = post(
        router,
        "/v1/audio/speech",
        json!({"model": "switchyard-tts", "input": "hello there"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "audio/wav");
    assert_eq!(headers["x-switchyard-audio-format"], "wav");
    assert_eq!(headers["x-switchyard-audio-rate"], "24000");
    assert_eq!(headers["x-switchyard-audio-voice"], "Kore");
    assert_eq!(&body[0..4], b"RIFF");
    assert_eq!(&body[8..12], b"WAVE");
    let riff_size = u32::from_le_bytes(body[4..8].try_into().expect("riff size"));
    assert_eq!(riff_size as usize, body.len() - 8);
    let channels = u16::from_le_bytes(body[22..24].try_into().expect("channels"));
    let rate = u32::from_le_bytes(body[24..28].try_into().expect("rate"));
    let bits = u16::from_le_bytes(body[34..36].try_into().expect("bits"));
    assert_eq!((channels, rate, bits), (1, 24_000, 16));
    let data_len = u32::from_le_bytes(body[40..44].try_into().expect("data length"));
    assert_eq!(data_len as usize, pcm.len());
    assert_eq!(&body[44..], &pcm[..]);
}

/// A declared format the engine does not produce must be refused before dispatch: the
/// engine returns PCM, so a caller asking for a compressed container would otherwise be
/// handed bytes labelled with a format they are not.
#[tokio::test]
async fn speech_refuses_a_container_it_cannot_produce_without_calling_out() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, _headers, body) = post(
        router,
        "/v1/audio/speech",
        json!({
            "model": "switchyard-tts",
            "input": "hello",
            "response_format": "mp3"
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("produces wav and pcm only"), "{text}");
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

/// Resampling is not implemented, so a speed other than 1.0 must be refused rather than
/// silently ignored: the audio would carry a rate the caller did not ask for.
#[tokio::test]
async fn speech_refuses_a_speed_it_cannot_apply() {
    let engine = MockServer::start().await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, _headers, body) = post(
        router,
        "/v1/audio/speech",
        json!({"model": "switchyard-tts", "input": "hello", "speed": 1.5}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("does not resample"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

/// A byte budget bounds the token count from above, so an input past it is refused locally.
#[tokio::test]
async fn speech_refuses_input_past_the_declared_byte_budget() {
    let engine = MockServer::start().await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, headers, body) = post(
        router,
        "/v1/audio/speech",
        json!({"model": "switchyard-tts", "input": "x".repeat(8193)}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(headers.get("x-switchyard-audio-format").is_none());
    assert!(
        String::from_utf8_lossy(&body).contains("accepts at most 8192"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

/// The engine's own status travels back, and no audio header may attest to audio that was
/// never received.
#[tokio::test]
async fn speech_passes_an_upstream_refusal_back_without_audio_headers() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_string("quota exceeded"))
        .mount(&engine)
        .await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, headers, _body) = post(
        router,
        "/v1/audio/speech",
        json!({"model": "switchyard-tts", "input": "hello"}),
    )
    .await;

    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    // The error is JSON, not audio, and nothing may attest to a container or rate.
    assert_eq!(headers["content-type"], "application/json");
    assert!(headers.get("x-switchyard-audio-format").is_none());
    assert!(headers.get("x-switchyard-audio-rate").is_none());
    assert!(headers.get("x-switchyard-audio-voice").is_none());
}

/// The rate the engine reports is checked against the declared one. A WAV header written at
/// the wrong rate would play back at the wrong speed, which is worse than a refusal.
#[tokio::test]
async fn speech_refuses_audio_whose_reported_rate_differs_from_the_contract() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"inlineData": {
                "mimeType": "audio/L16;codec=pcm;rate=16000",
                "data": b64(&pcm16(16))
            }}]}}]
        })))
        .mount(&engine)
        .await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, _headers, body) = post(
        router,
        "/v1/audio/speech",
        json!({"model": "switchyard-tts", "input": "hello"}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        String::from_utf8_lossy(&body).contains("does not match the declared rate"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

/// A held speech lane is inventory: naming it fails locally and reaches no engine.
#[tokio::test]
async fn a_held_speech_capability_reaches_no_engine() {
    let engine = MockServer::start().await;
    let config = audio_config(&engine.uri(), "", "").replace(
        "[capabilities.tts]",
        "[capabilities.tts]\nheld = \"not qualified: no authenticated audio response observed\"",
    );
    let router = serve(&config).await;
    let (status, _headers, body) = post(
        router,
        "/v1/audio/speech",
        json!({"model": "switchyard-tts", "input": "hello"}),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        String::from_utf8_lossy(&body).contains("No capability registered"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

#[tokio::test]
async fn transcription_sends_the_audio_and_returns_the_text() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    let wav = wav_bytes(0.5, 16_000);
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-3.5-transcribe:generateContent"))
        .and(header("x-goog-api-key", "test-key"))
        .and(body_partial_json(json!({
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": b64(&wav)}},
                {"text": "Transcribe this recording verbatim."}
            ]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "hello"}, {"text": " world"}]}}]
        })))
        .expect(1)
        .mount(&engine)
        .await;

    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let (status, headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(&wav)}
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-switchyard-audio-seconds"], "0.500");
    let value: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(value["text"], "hello world");
}

/// A container the surface cannot measure is refused: forwarding it would spend a token
/// budget nobody bounded.
#[tokio::test]
async fn transcription_refuses_a_container_it_cannot_measure() {
    let engine = MockServer::start().await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;

    let (status, _headers, body) = post(
        router.clone(),
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/mpeg", "data": b64(b"ID3notreallyaudio")}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("audio.mime_type audio/mpeg is not accepted"),
        "{}",
        String::from_utf8_lossy(&body)
    );

    let (status, _headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(b"not a riff file at all")}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("not a RIFF/WAVE container"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

/// Duration is parsed from the container, so a recording past the contract is refused even
/// though its byte count is small.
#[tokio::test]
async fn transcription_refuses_audio_longer_than_the_contract() {
    let engine = MockServer::start().await;
    let config = audio_config(&engine.uri(), "", "")
        .replace("max_audio_seconds = 300", "max_audio_seconds = 2");
    let router = serve(&config).await;
    let wav = wav_bytes(3.0, 8_000);
    let (status, _headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(&wav)}
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("accepts at most 2s"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

/// A header whose byte rate disagrees with its sample rate would let a long recording be
/// admitted as a short one, and charged for the short one. Both figures are the caller's, so
/// they have to agree before either is trusted.
#[tokio::test]
async fn transcription_refuses_a_byte_rate_that_understates_the_audio() {
    let engine = MockServer::start().await;
    let config = audio_config(&engine.uri(), "", "")
        .replace("max_audio_seconds = 300", "max_audio_seconds = 2");
    let router = serve(&config).await;
    // Three seconds of 16-bit 8 kHz mono, then a byte rate that claims 4 MB/s — which would
    // read as twelve milliseconds and slip under the two-second ceiling.
    let mut wav = wav_bytes(3.0, 8_000);
    wav[28..32].copy_from_slice(&4_000_000u32.to_le_bytes());
    let (status, _headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(&wav)}
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("byte rate does not agree"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

/// The byte ceiling is separate from the duration ceiling and is enforced first.
#[tokio::test]
async fn transcription_refuses_audio_larger_than_the_contract() {
    let engine = MockServer::start().await;
    let config = audio_config(&engine.uri(), "", "")
        .replace("max_audio_bytes = 15728640", "max_audio_bytes = 1000");
    let router = serve(&config).await;
    let wav = wav_bytes(0.1, 48_000);
    assert!(wav.len() > 1000);
    let (status, _headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(&wav)}
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("accepts at most 1000"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0
    );
}

/// An answer with no text is a contract mismatch, not an empty transcription: the caller
/// must not be told the recording was silent when the engine said something else.
#[tokio::test]
async fn transcription_reports_an_answer_without_text_as_a_contract_mismatch() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]
        })))
        .mount(&engine)
        .await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let wav = wav_bytes(0.2, 16_000);
    let (status, _headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(&wav)}
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    let text = String::from_utf8_lossy(&body).to_string();
    assert!(text.contains("returned no text"), "{text}");
    assert!(text.contains("SAFETY"), "{text}");
}

/// The engine's status travels back on the transcription surface too.
#[tokio::test]
async fn transcription_passes_an_upstream_failure_back() {
    unsafe { std::env::set_var("SWITCHYARD_TEST_AUDIO_KEY", "test-key") };
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_string("model overloaded"))
        .mount(&engine)
        .await;
    let router = serve(&audio_config(&engine.uri(), "", "")).await;
    let wav = wav_bytes(0.2, 16_000);
    let (status, _headers, body) = post(
        router,
        "/v1/audio/transcriptions",
        json!({
            "model": "switchyard-transcribe",
            "audio": {"mime_type": "audio/wav", "data": b64(&wav)}
        }),
    )
    .await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        String::from_utf8_lossy(&body).contains("model overloaded"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

// ---------------------------------------------------------------------------------------------
// Package D: typed rerank selection and pre-egress data admission.
//
// The selector walks interchangeable engines in declared order. Admission is evaluated before
// anything is sent, from the caller's own declaration rather than from the payload's contents:
// the broad content filter was removed on 2026-11-21 and must not return through this surface.
// ---------------------------------------------------------------------------------------------

/// A rerank selector over a local engine and a third-party free engine, exactly the deployed
/// shape: `smartlocal-rerank` first, `smartfree-rerank` second.
fn selector_config(local: &str, free: &str) -> String {
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

[capability_clients.reranker-local]
format = "cohere_jina_rerank"
base_url = "{local}"
model = "qwen3-reranker-0.6b-q8_0"

[capability_clients.reranker-free]
format = "cohere_jina_rerank"
base_url = "{free}"
model = "nvidia/nemotron-3.5-rerank-8b:free"
privacy = "third_party_free"

[capabilities.smart-rerank]
id = "smart-rerank"
target = "reranker-local"
candidates = ["reranker-free"]
max_candidates = 64
top_n = 5
max_doc_chars = 1024
max_query_chars = 512
"#
    )
}

/// A rerank answer shaped the way these engines actually reply.
fn rerank_answer() -> Value {
    json!({
        "id": "rerank-1",
        "results": [{"index": 0, "relevance_score": 0.93, "document": "alpha"}],
    })
}

/// Post with an explicit data-class declaration (or none, when `class` is `None`).
async fn post_with_class(
    router: axum::Router,
    uri: &str,
    body: Value,
    class: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(class) = class {
        builder = builder.header("x-switchyard-data-class", class);
    }
    let response = router
        .oneshot(builder.body(Body::from(body.to_string())).expect("request builds"))
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    (status, bytes.to_vec())
}

fn rerank_body() -> Value {
    json!({"model": "smart-rerank", "query": "find alpha", "documents": ["alpha", "beta"]})
}

// An unstated class is not evidence that the material is public, so it may only reach an
// endpoint that declares no restriction. The retentive engine in the same selector must stay
// untouched - but the request itself is served, because the internal engine may hold it.
#[tokio::test]
async fn an_unstated_data_class_never_reaches_a_third_party_free_engine() {
    let local = MockServer::start().await;
    let free = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(1)
        .mount(&local)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(0)
        .mount(&free)
        .await;

    let router = serve(&selector_config(&local.uri(), &free.uri())).await;
    let (status, body) = post_with_class(router, "/v1/rerank", rerank_body(), None).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "the internal engine serves an unclassified payload: {}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        free.received_requests().await.unwrap_or_default().len(),
        0,
        "no request may reach the third-party free engine"
    );
}

// When the only engine is third-party free, an unstated class is refused rather than guessed,
// and the refusal names the class and the engine instead of pretending the capability is absent.
#[tokio::test]
async fn an_unstated_data_class_is_refused_when_the_only_engine_is_third_party_free() {
    let free = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(0)
        .mount(&free)
        .await;

    let config = format!(
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

[capability_clients.reranker-free]
format = "cohere_jina_rerank"
base_url = "{}"
model = "nvidia/nemotron-3.5-rerank-8b:free"
privacy = "third_party_free"

[capabilities.smart-rerank]
id = "smart-rerank"
target = "reranker-free"
max_candidates = 64
top_n = 5
max_doc_chars = 1024
max_query_chars = 512
"#,
        free.uri()
    );

    let router = serve(&config).await;
    let (status, body) = post_with_class(router, "/v1/rerank", rerank_body(), None).await;

    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "an unstated class must not reach a retentive endpoint: {}",
        String::from_utf8_lossy(&body)
    );
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("egress_not_admitted"), "{text}");
    assert!(text.contains("unknown"), "the refusal names the class: {text}");
    assert!(
        text.contains("reranker-free"),
        "the refusal names the engine that was skipped: {text}"
    );
    assert_eq!(
        free.received_requests().await.unwrap_or_default().len(),
        0,
        "the payload never leaves the process"
    );
}

// The local engine keeps working exactly as before for ordinary internal material, because its
// endpoint declares no restriction. This is what stops the gate from becoming a broad filter.
#[tokio::test]
async fn internal_data_still_reaches_an_engine_that_declares_no_restriction() {
    let local = MockServer::start().await;
    let free = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"model": "qwen3-reranker-0.6b-q8_0"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(1)
        .mount(&local)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(0)
        .mount(&free)
        .await;

    let router = serve(&selector_config(&local.uri(), &free.uri())).await;
    let (status, body) =
        post_with_class(router, "/v1/rerank", rerank_body(), Some("internal")).await;

    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(
        local.received_requests().await.unwrap_or_default().len(),
        1,
        "the local engine answers, and only once"
    );
}

// A public payload may use the free engine, which is the whole reason it is attached.
#[tokio::test]
async fn public_data_is_admitted_to_a_third_party_free_engine_when_the_local_one_fails() {
    let local = MockServer::start().await;
    let free = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&local)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"model": "nvidia/nemotron-3.5-rerank-8b:free"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(1)
        .mount(&free)
        .await;

    let router = serve(&selector_config(&local.uri(), &free.uri())).await;
    let (status, body) = post_with_class(router, "/v1/rerank", rerank_body(), Some("public")).await;

    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(
        local.received_requests().await.unwrap_or_default().len(),
        1,
        "the declared engine is tried first"
    );
    assert_eq!(
        free.received_requests().await.unwrap_or_default().len(),
        1,
        "the selector moves on only after the first engine fails"
    );
}

// A caller error is the caller's answer: the walk stops rather than spending a second provider
// call to receive the same complaint from a different engine.
#[tokio::test]
async fn a_caller_error_does_not_move_the_selector_to_the_next_engine() {
    let local = MockServer::start().await;
    let free = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(422).set_body_json(json!({"error": "bad documents"})))
        .expect(1)
        .mount(&local)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(rerank_answer()))
        .expect(0)
        .mount(&free)
        .await;

    let router = serve(&selector_config(&local.uri(), &free.uri())).await;
    let (status, _) = post_with_class(router, "/v1/rerank", rerank_body(), Some("public")).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

// An embedding has no compatible fallback: a vector from another space is a different space.
// Unstated data is therefore refused outright rather than re-homed to another engine.
#[tokio::test]
async fn an_embedding_refuses_a_third_party_free_endpoint_for_unstated_data() {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": vec![0.1, 0.2]}],
            "model": "nvidia/nemotron-3.5-embed-8b:free",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })))
        .expect(0)
        .mount(&engine)
        .await;

    let config = format!(
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

[capability_clients.embedding-free-nemotron]
format = "openai_embeddings"
base_url = "{}"
model = "nvidia/nemotron-3.5-embed-8b:free"
privacy = "third_party_free"

[capabilities.smartfree-embedding]
id = "smartfree-embedding"
target = "embedding-free-nemotron"
contract = "openrouter-embedding-space:nemotron3.5-embed8b-v1"
dimensions = 4096
normalization = "L2"
max_batch = 64
"#,
        engine.uri()
    );

    let router = serve(&config).await;
    let (status, body) = post_with_class(
        router,
        "/v1/embeddings",
        json!({"model": "smartfree-embedding", "input": ["private note"]}),
        None,
    )
    .await;

    assert_eq!(status, StatusCode::FORBIDDEN, "{}", String::from_utf8_lossy(&body));
    assert!(
        String::from_utf8_lossy(&body).contains("egress_not_admitted"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert_eq!(
        engine.received_requests().await.unwrap_or_default().len(),
        0,
        "the payload never leaves the process"
    );
}

// `smart-gemini-embedding` is an alias over the existing Gemini Embedding 2 client and its
// existing space: one client, one contract, no new space and no cross-space fallback.
#[tokio::test]
async fn smart_gemini_embedding_is_an_alias_over_the_existing_space() {
    let config = r#"
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

[capability_clients.embedding-google2]
format = "google_embed_content_v2"
base_url = "https://generativelanguage.test"
model = "gemini-embedding-2"

[capabilities.gemini-embedding-2]
id = "gemini-embedding-2"
target = "embedding-google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = 3072
normalization = "L2"
max_batch = 8
max_input_tokens = 8192

[capabilities.smart-gemini-embedding]
id = "smart-gemini-embedding"
target = "embedding-google2"
contract = "google-embedding-space:gemini-embedding-2-v1"
dimensions = 3072
normalization = "L2"
max_batch = 8
max_input_tokens = 8192
"#;
    let runner = Runner::from_toml(config).expect("the alias is expressible against the schema");

    let canonical = runner
        .capability("gemini-embedding-2")
        .expect("the canonical capability still exists");
    let alias = runner
        .capability("smart-gemini-embedding")
        .expect("the alias resolves");

    assert_eq!(
        alias.contract, canonical.contract,
        "the alias must name the same value space, not a new one"
    );
    assert_eq!(alias.dimensions, canonical.dimensions);
    assert_eq!(alias.normalization, canonical.normalization);
    assert_eq!(
        alias.format, canonical.format,
        "the alias must speak the same wire"
    );
    assert_eq!(alias.model, canonical.model);
    assert_eq!(
        alias.targets[0].client, canonical.targets[0].client,
        "one client: the alias is not a second endpoint"
    );
    assert_eq!(alias.url, canonical.url, "one endpoint, not two");
    assert_eq!(
        alias.targets.len(),
        1,
        "no candidates: an embedding selector would be a cross-space fallback"
    );
}

// The selector is a rerank-only shape. Declaring candidates on an embedding space is refused at
// load rather than silently resolving to whichever engine answered first.
#[test]
fn an_embedding_capability_may_not_declare_candidates() {
    let config = r#"
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

[capability_clients.embedding-local]
format = "openai_embeddings"
base_url = "https://local.test"
model = "Qwen3-Embedding-0.6B"

[capability_clients.embedding-free]
format = "openai_embeddings"
base_url = "https://free.test"
model = "nvidia/nemotron-3.5-embed-8b:free"

[capabilities.smart-embedding]
id = "smart-embedding"
target = "embedding-local"
candidates = ["embedding-free"]
contract = "local-embedding-space:v1"
dimensions = 1024
normalization = "L2"
"#;
    let error = match Runner::from_toml(config) {
        Ok(_) => panic!("an embedding may not select among engines"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("only a rerank capability may select among engines"),
        "the refusal explains why: {error}"
    );
}

// Two engines of different wire formats do not answer the same question, so the selector refuses
// the pairing instead of discovering it on the first failover.
#[test]
fn a_selector_refuses_candidates_that_speak_a_different_wire() {
    let config = r#"
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

[capability_clients.reranker-local]
format = "cohere_jina_rerank"
base_url = "https://local.test"
model = "qwen3-reranker"

[capability_clients.embedding-free]
format = "openai_embeddings"
base_url = "https://free.test"
model = "nvidia/nemotron-3.5-embed-8b:free"

[capabilities.mixed]
id = "mixed"
target = "reranker-local"
candidates = ["embedding-free"]
max_candidates = 64
"#;
    let error = match Runner::from_toml(config) {
        Ok(_) => panic!("mismatched candidates must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("answer the same request shape"),
        "the refusal names the mismatch: {error}"
    );
}
