// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the spatial surface (`POST /v1/robotics/spatial`).
//!
//! These drive the real router against a mock Gemini endpoint, so they assert what a caller
//! actually receives and what the provider actually sees: a validated item list in place of
//! the model's prose, the documented JSON output contract on the wire, an error when the
//! model answers outside it, and local refusal of a scene the capability cannot accept.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use serde_json::{Value, json};
use switchyard_runner::Runner;
use tower::ServiceExt;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY_ENV: &str = "SY_TEST_ROBOTICS_KEY";
const KEY_VALUE: &str = "server-owned-spatial-key";

fn config(engine: &str, max_image_bytes: u64) -> String {
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

[capability_clients.robotics]
format = "google_robotics_spatial"
base_url = "{engine}"
model = "gemini-robotics-test"
api_key_env = "{KEY_ENV}"
timeout_seconds = 30

[capabilities.spatial-cap]
id = "spatial-cap"
target = "robotics"
max_input_tokens = 100000
max_image_bytes = {max_image_bytes}
"#
    )
}

fn router(config: &str) -> axum::Router {
    let runner = Runner::from_toml(config).expect("runner builds");
    switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    )
}

/// A PNG header with the stated dimensions: 8-byte signature plus a 25-byte IHDR chunk.
fn png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    bytes.extend_from_slice(&[0, 0, 0, 13]);
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes
}

fn body(model: &str, task: &str, image: &[u8]) -> Value {
    json!({
        "model": model,
        "task": task,
        "scene": {
            "data": base64::engine::general_purpose::STANDARD.encode(image),
            "mime_type": "image/png",
        },
    })
}

async fn post(router: axum::Router, body: &Value) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/robotics/spatial")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router answers");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, value)
}

fn gemini_answer(text: &str, prompt_tokens: u64) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": prompt_tokens},
    }))
}

#[tokio::test]
async fn a_point_answer_reaches_the_caller_as_validated_items() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };
    let engine = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-robotics-test:generateContent"))
        .and(header("x-goog-api-key", KEY_VALUE))
        .and(body_partial_json(json!({
            "generationConfig": {"responseMimeType": "application/json"}
        })))
        .respond_with(gemini_answer(
            r#"[{"label": "cup", "point": [512, 300]}, {"label": "handle", "point": [480, 340]}]"#,
            1100,
        ))
        .expect(1)
        .mount(&engine)
        .await;

    let (status, headers, value) = post(
        router(&config(&engine.uri(), 1_000_000)),
        &body("spatial-cap", "point", &png(800, 600)),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {value}");
    assert_eq!(value["model"], "spatial-cap");
    assert_eq!(value["task"], "point");
    assert_eq!(value["count"], 2);
    assert_eq!(value["items"][0], json!({"label": "cup", "point": [512, 300]}));
    assert_eq!(value["usage"]["provider_prompt_tokens"], 1100);
    assert!(
        value["usage"]["estimated_prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "the local estimate travels with the provider's own count: {value}"
    );
    assert_eq!(
        headers.get("x-switchyard-spatial-task").and_then(|v| v.to_str().ok()),
        Some("point")
    );
    assert_eq!(
        headers.get("x-switchyard-spatial-count").and_then(|v| v.to_str().ok()),
        Some("2")
    );

    // The instruction the provider saw states the contract this surface validates against,
    // so the model is never asked for a shape the caller would then have refused.
    let requests = engine.received_requests().await.expect("requests recorded");
    let sent: Value = serde_json::from_slice(&requests[0].body).expect("request json");
    let instruction = sent["contents"][0]["parts"][1]["text"]
        .as_str()
        .expect("instruction part");
    assert!(
        instruction.contains("[y, x]"),
        "the point contract is stated to the provider: {instruction}"
    );
    assert_eq!(
        sent["contents"][0]["parts"][0]["inlineData"]["mimeType"],
        "image/png"
    );
    assert!(
        sent.to_string().find(KEY_VALUE).is_none(),
        "the credential travels in the header, never in the body"
    );
}

#[tokio::test]
async fn prose_is_refused_rather_than_returned_as_a_spatial_answer() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };
    let engine = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-robotics-test:generateContent"))
        .respond_with(gemini_answer("I can see a cup on the left side of the table.", 1000))
        .expect(1)
        .mount(&engine)
        .await;

    let (status, _headers, value) = post(
        router(&config(&engine.uri(), 1_000_000)),
        &body("spatial-cap", "point", &png(800, 600)),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::BAD_GATEWAY,
        "a provider that answers in prose has broken the contract: {value}"
    );
    assert_eq!(value["error"]["type"], "spatial_contract_error");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("spatial contract")),
        "the refusal names the contract: {value}"
    );
}

#[tokio::test]
async fn an_oversize_scene_is_refused_before_the_engine_is_called() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };
    let engine = MockServer::start().await;

    // No mock is mounted: any request that reaches the engine fails the test.
    let image = png(800, 600);
    let (status, _headers, value) = post(
        router(&config(&engine.uri(), 16)),
        &body("spatial-cap", "boxes", &image),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("accepts at most 16")),
        "the refusal names the declared ceiling: {value}"
    );
    let requests = engine
        .received_requests()
        .await
        .expect("requests recorded");
    assert!(
        requests.is_empty(),
        "a refused scene must not cost a provider call: {requests:?}"
    );
}

#[tokio::test]
async fn an_unmeasurable_container_is_refused_before_the_engine_is_called() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };
    let engine = MockServer::start().await;

    let mut request = body("spatial-cap", "point", b"not an image at all");
    request["scene"]["mime_type"] = json!("image/gif");
    let (status, _headers, value) = post(router(&config(&engine.uri(), 1_000_000)), &request).await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("measurand")),
        "the refusal explains why the container is unsupported: {value}"
    );
    let requests = engine
        .received_requests()
        .await
        .expect("requests recorded");
    assert!(requests.is_empty());
}
