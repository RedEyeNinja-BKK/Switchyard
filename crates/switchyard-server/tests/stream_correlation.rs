// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Correlation join across the real HTTP path.
//!
//! A streamed request answers `200` before the stream ends, so a stream that dies has no
//! status code to carry its failure. The only trace is the terminal request line and the
//! stream warning, and tying one request's warning to one request's line is the whole
//! point of the correlation id.
//!
//! These tests therefore drive `/v1/responses` through `build_switchyard_router` over a
//! real socket-backed upstream and read the events the server emits, rather than calling
//! the reporter directly. A unit test of the reporter can show it renders whatever id it
//! is handed; only the HTTP path can show that the id it is handed is the same one the
//! terminal line names, including the case where the caller supplied none and the server
//! had to mint one.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_protocol::correlation::{MAX_REQUEST_ID_LEN, validated_request_id};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The prompt in every request. A distinctive string, so a leak into a log line is
/// unambiguous rather than a coincidence of common words.
const PROMPT: &str = "PROMPT-MUST-NEVER-BE-LOGGED-4f2a91";

/// The caller-supplied id used by the join tests.
const CALLER_ID: &str = "caller-correlation-7f3c";

/// One Responses stream: reasoning, several answer deltas, then a neutral terminal.
///
/// Long enough that a consumer which drops after the first frame is unambiguously
/// before `response.completed`, and sanitised: fixed ids, no provider or prompt text.
fn responses_sse() -> String {
    let mut frames = vec![
        "event: response.created".to_string(),
        r#"data: {"type":"response.created","response":{"id":"resp_fixture","model":"fixture-model","status":"in_progress","output":[]}}"#.to_string(),
        String::new(),
        "event: response.output_item.added".to_string(),
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_fixture","type":"reasoning","summary":[]}}"#.to_string(),
        String::new(),
        r#"data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_fixture","output_index":0,"delta":"thinking"}"#.to_string(),
        String::new(),
    ];
    for index in 0..40 {
        frames.push(format!(
            r#"data: {{"type":"response.output_text.delta","item_id":"msg_fixture","output_index":1,"delta":"tok{index} "}}"#
        ));
        frames.push(String::new());
    }
    frames.push(
        r#"data: {"type":"response.completed","response":{"id":"resp_fixture","model":"fixture-model","status":"completed","output":[{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#.to_string(),
    );
    frames.push(String::new());
    frames.push("data: [DONE]".to_string());
    frames.push(String::new());
    frames.join("\n")
}

/// The same stream, but it fails instead of completing.
///
/// `response.failed` is the Responses shape that decodes to an explicit stream error,
/// which is the other way a stream can end early: the consumer keeps reading and is told
/// why, rather than leaving. The warning for that path must name the request too.
fn responses_sse_failing() -> String {
    [
        r#"data: {"type":"response.created","response":{"id":"resp_fixture","model":"fixture-model","status":"in_progress","output":[]}}"#,
        "",
        r#"data: {"type":"response.output_text.delta","item_id":"msg_fixture","output_index":0,"delta":"partial"}"#,
        "",
        r#"data: {"type":"response.failed","response":{"id":"resp_fixture","status":"failed","error":{"code":"upstream_error","message":"provider stopped mid-answer"}}}"#,
        "",
        "data: [DONE]",
        "",
    ]
    .join("\n")
}

/// Serves the failing fixture.
async fn failing_upstream() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(responses_sse_failing().into_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;
    server
}

/// Serves the fixture on the path an `openai_responses` client calls.
async fn upstream() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(responses_sse().into_bytes(), "text/event-stream"),
        )
        .mount(&server)
        .await;
    server
}

/// A leaf passthrough route whose single target is the mock upstream.
///
/// Deliberately the smallest deployment that streams: no candidates, no composition, so
/// the only things that can appear in the log are the two lines this test is about.
fn deployment(base_url: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.fixture]
format = "openai_responses"
base_url = "{base_url}/v1"
max_retries = 0

[targets.fixture]
id = "fixture-model"
llm_client = "fixture"

[routes.leaf]
id = "leaf"
type = "passthrough"
target = "fixture"
"#
    )
}

/// Every event the server emits on its request-log target, rendered as `field=value` text.
///
/// Both the terminal line and the stream warning name `switchyard_server::request`, which
/// is what makes them one record per request rather than two unrelated ones.
#[derive(Clone, Default)]
struct CapturedEvents(Arc<Mutex<Vec<(tracing::Level, String)>>>);

impl tracing_subscriber::Layer<tracing_subscriber::Registry> for CapturedEvents {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, tracing_subscriber::Registry>,
    ) {
        if event.metadata().target() != "switchyard_server::request" {
            return;
        }
        let mut rendered = String::new();
        event.record(
            &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                rendered.push_str(&format!("{}={value:?} ", field.name()));
            },
        );
        self.0
            .lock()
            .unwrap()
            .push((*event.metadata().level(), rendered));
    }
}

/// Drives a streaming request through the real router and abandons the body.
///
/// Returns the status, the events emitted while the request and its stream were alive,
/// and the raw first frame so a caller can prove it really got a stream rather than an
/// error body.
async fn drive_and_abandon(
    base_url: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, Vec<(tracing::Level, String)>, Vec<u8>) {
    use tracing_subscriber::layer::SubscriberExt;

    let runner =
        switchyard_runner::Runner::from_toml(&deployment(base_url)).expect("deployment builds");
    let state = switchyard_server::ServerState::from_runner(runner).expect("state builds");
    let app = switchyard_server::build_switchyard_router(state);

    let captured = CapturedEvents::default();
    let subscriber = tracing_subscriber::registry().with(captured.clone());
    let guard = tracing::subscriber::set_default(subscriber);

    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let body = json!({
        "model": "leaf",
        "input": PROMPT,
        "max_output_tokens": 64,
        "stream": true
    });
    let response = app
        .oneshot(
            builder
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();

    // Read one frame and let go. This is the shape every live `consumer_dropped` had:
    // the caller took delivery of the response and then stopped reading.
    let mut body = response.into_body();
    let first = match body.frame().await {
        Some(Ok(frame)) => frame.into_data().unwrap_or_default().to_vec(),
        _ => Vec::new(),
    };
    drop(body);

    // The reporter is dropped with the stream body, so its event is recorded by now.
    drop(guard);
    let events = captured.0.lock().unwrap().clone();
    (status, events, first)
}

/// The value of one `field=value` pair in a rendered event.
fn field<'a>(event: &'a str, name: &str) -> Option<&'a str> {
    let start = event.find(&format!("{name}="))? + name.len() + 1;
    let rest = &event[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Some(rest[..end].trim_matches('"'))
}

/// The terminal line: the one that says the request was handled or failed.
fn terminal_line(events: &[(tracing::Level, String)]) -> &str {
    events
        .iter()
        .map(|(_, text)| text.as_str())
        .find(|text| text.contains("LLM request handled") || text.contains("LLM request failed"))
        .expect("every request emits one terminal line")
}

/// The stream warning: the one that says the stream did not reach its terminal event.
fn drop_line(events: &[(tracing::Level, String)]) -> &str {
    events
        .iter()
        .map(|(_, text)| text.as_str())
        .find(|text| text.contains("LLM stream ended before its terminal event"))
        .expect("abandoning a stream before its terminal event must be reported")
}

#[tokio::test]
async fn an_abandoned_stream_joins_its_request_line_without_a_caller_id() {
    let upstream = upstream().await;
    let (status, events, first_frame) = drive_and_abandon(&upstream.uri(), &[]).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "a stream is answered before it ends"
    );
    assert!(
        String::from_utf8_lossy(&first_frame).contains("data:"),
        "the caller must really receive SSE, got {:?}",
        String::from_utf8_lossy(&first_frame)
    );

    let terminal = terminal_line(&events);
    let dropped = drop_line(&events);
    let terminal_id = field(terminal, "correlation_id").expect("the terminal line names an id");
    let dropped_id = field(dropped, "correlation_id").expect("the warning names an id");

    assert!(
        !terminal_id.is_empty(),
        "a request with no caller id must still be nameable: {terminal}"
    );
    assert_eq!(
        terminal_id, dropped_id,
        "the warning must name the same request as the terminal line:\n  terminal: {terminal}\n  warning:  {dropped}"
    );
}

#[tokio::test]
async fn a_caller_supplied_id_is_the_one_both_lines_name() {
    let upstream = upstream().await;
    let (status, events, _) =
        drive_and_abandon(&upstream.uri(), &[("x-switchyard-request-id", CALLER_ID)]).await;

    assert_eq!(status, StatusCode::OK);
    let terminal = terminal_line(&events);
    let dropped = drop_line(&events);

    assert_eq!(
        field(terminal, "correlation_id"),
        Some(CALLER_ID),
        "a caller id must be used verbatim, not replaced: {terminal}"
    );
    assert_eq!(
        field(dropped, "correlation_id"),
        Some(CALLER_ID),
        "the warning must name the caller's id: {dropped}"
    );
}

#[tokio::test]
async fn a_minted_id_is_bounded_and_identifier_shaped() {
    let upstream = upstream().await;
    let (_, events, _) = drive_and_abandon(&upstream.uri(), &[]).await;
    let id = field(terminal_line(&events), "correlation_id").expect("an id");

    assert!(
        validated_request_id(id).is_some(),
        "a minted id must be identifier-shaped so it can be quoted anywhere a provider id goes: {id}"
    );
    assert!(
        id.len() <= MAX_REQUEST_ID_LEN,
        "a minted id must stay within the provider identifier bound: {} chars",
        id.len()
    );

    // Two requests must not collide, or the join is ambiguous exactly when it matters.
    let (_, second, _) = drive_and_abandon(&upstream.uri(), &[]).await;
    let second_id = field(terminal_line(&second), "correlation_id").expect("an id");
    assert_ne!(id, second_id, "two requests must not share an id");
}

#[tokio::test]
async fn the_prompt_never_reaches_a_log_line() {
    let upstream = upstream().await;
    let (_, events, _) = drive_and_abandon(&upstream.uri(), &[]).await;

    for (level, text) in &events {
        assert!(
            !text.contains(PROMPT),
            "request content must not be logged ({level}): {text}"
        );
        assert!(
            !text.to_ascii_lowercase().contains("authorization") && !text.contains("Bearer"),
            "credentials must not be logged ({level}): {text}"
        );
    }
}

#[tokio::test]
async fn adding_a_caller_id_does_not_change_the_upstream_or_downstream_body() {
    let upstream = upstream().await;
    let (_, _, without_id) = drive_and_abandon(&upstream.uri(), &[]).await;
    let (_, _, with_id) =
        drive_and_abandon(&upstream.uri(), &[("x-switchyard-request-id", CALLER_ID)]).await;

    assert_eq!(
        without_id, with_id,
        "the correlation id is a host-side join key, so it must not alter what a caller receives"
    );

    let requests = upstream
        .received_requests()
        .await
        .expect("the mock records what it was sent");
    assert_eq!(requests.len(), 2);
    let bodies: Vec<Value> = requests
        .iter()
        .map(|request| serde_json::from_slice(&request.body).expect("json request"))
        .collect();
    assert_eq!(
        bodies[0], bodies[1],
        "the correlation id must not be forwarded upstream as part of the request"
    );
    for body in &bodies {
        assert!(
            !body.to_string().contains(CALLER_ID),
            "the id must not leak into the provider request: {body}"
        );
    }
}

#[tokio::test]
async fn an_explicit_stream_failure_names_the_same_request_as_its_terminal_line() {
    use tracing_subscriber::layer::SubscriberExt;

    let upstream = failing_upstream().await;
    let runner = switchyard_runner::Runner::from_toml(&deployment(&upstream.uri()))
        .expect("deployment builds");
    let state = switchyard_server::ServerState::from_runner(runner).expect("state builds");
    let app = switchyard_server::build_switchyard_router(state);

    let captured = CapturedEvents::default();
    let subscriber = tracing_subscriber::registry().with(captured.clone());
    let guard = tracing::subscriber::set_default(subscriber);

    let body = json!({
        "model": "leaf",
        "input": PROMPT,
        "max_output_tokens": 64,
        "stream": true
    });
    let response = app
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
    assert_eq!(response.status(), StatusCode::OK);

    // Drain to the end: this failure is reported while the consumer is still reading,
    // so the guard is not the thing under test here.
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body reads");
    drop(guard);
    let _ = bytes;

    let events = captured.0.lock().unwrap().clone();
    let terminal = terminal_line(&events);
    let failed = events
        .iter()
        .map(|(_, text)| text.as_str())
        .find(|text| text.contains("LLM stream failed"))
        .expect("an explicit stream failure must be reported");

    assert_eq!(
        field(terminal, "correlation_id"),
        field(failed, "correlation_id"),
        "the explicit-failure warning must name the request its terminal line names:\n  terminal: {terminal}\n  warning:  {failed}"
    );
    assert!(
        !field(failed, "correlation_id")
            .unwrap_or_default()
            .is_empty(),
        "the warning must carry an id: {failed}"
    );
    assert!(
        failed.contains(r#"failure_stage="stream""#),
        "the warning must say which stage failed: {failed}"
    );
}
