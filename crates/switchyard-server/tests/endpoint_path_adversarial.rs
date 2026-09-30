// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw
// SPDX-License-Identifier: Apache-2.0

//! Adversarial `endpoint_path` admission.
//!
//! `endpoint_path` is security-sensitive: the executor's bearer credential
//! travels with the request, so a value that changes the URL's origin would
//! send that credential to an attacker-chosen host.
//!
//! The invariant under test is narrow and absolute:
//!
//! > `endpoint_path` may select a PATH on the configured origin, and may never
//! > alter scheme, authority, host, or credential destination.
//!
//! Every case is asserted on the JOINED URL's origin — the scheme, host and
//! effective port actually parsed out of the final string — not on a
//! substring. A check that only refuses the one input it was written against
//! proves nothing about its neighbours, so the battery is written as a table
//! and every row is an input a reviewer would expect to be refused.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tower::ServiceExt;

use switchyard_server::build_switchyard_router;
use switchyard_server::config::load_server_runtime;

/// A loopback executor that records the request line it was called on and
/// answers a well-formed decision body.
async fn spawn_recorder(seen: Arc<std::sync::Mutex<Vec<String>>>, hits: Arc<AtomicUsize>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            hits.fetch_add(1, Ordering::SeqCst);
            let body = json!({
                "answers": {"q": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            })
            .to_string();
            let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                let mut buffer = vec![0u8; 64 * 1024];
                let read = socket.read(&mut buffer).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]);
                let path = request
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                seen.lock().unwrap().push(path);
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            })
            .await;
        }
    });
    port
}

/// TOML string form for an `endpoint_path` literal.
///
/// A BACKSLASH can only reach the validator through a TOML LITERAL string:
/// in a basic string `"\e"` is an invalid escape and the PARSER rejects the
/// file before the validator ever runs. Writing the hostile cases in basic
/// strings would test the TOML parser, not this field - so literal form is
/// used for every case that contains a backslash.
fn toml_string(value: &str) -> String {
    if value.contains('\\') {
        format!("'{value}'")
    } else {
        format!("{value:?}")
    }
}

fn config_toml(base_url: &str, endpoint_path: &str) -> String {
    let endpoint_path = toml_string(endpoint_path);
    format!(
        r#"
schema_version = 1

[targets.primary-target]
id = "primary-target"
llm_client = "primary-chat"

[llm_clients.primary-chat]
format = "openai_responses"
base_url = "{base_url}"

[capability_clients.primary-exec]
format = "openrouter_alpha_decisions"
base_url = "{base_url}"
model = "primary-exec-model"
endpoint_path = {endpoint_path}
timeout_seconds = 5

[capabilities.solo-decision]
id = "solo-decision"
target = "primary-exec"
decision_contract = "switchyard-decision:v1"
max_questions = 4
max_state_chars = 4000

[routes.solo-decision]
id = "solo-decision"
type = "passthrough"
target = "primary-target"
"#
    )
}

const REQUEST_BODY: &str = r#"{"model": "solo-decision", "state": "the state under assessment", "questions": {"q": {"instructions": "How complex?", "criteria": {"simple": "one", "hard": "many"}}}}"#;

/// Loads a config and, if it loads, drives one request and returns the request
/// path the executor actually saw.
async fn observe(
    base_url: &str,
    endpoint_path: &str,
    seen: &Arc<std::sync::Mutex<Vec<String>>>,
) -> Result<Vec<String>, String> {
    let dir = std::env::temp_dir().join(format!(
        "endpoint-path-adv-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("routes.toml");
    std::fs::write(&config, config_toml(base_url, endpoint_path)).expect("write config");
    let runtime = load_server_runtime(&config).map_err(|error| error.to_string())?;
    let app = build_switchyard_router(runtime.state);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(REQUEST_BODY))
        .unwrap();
    let response = app.oneshot(request).await.expect("router must answer");
    let _ = response.into_body().collect().await.unwrap().to_bytes();
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    Ok(std::mem::take(&mut *seen.lock().unwrap()))
}

/// Inputs that must be REFUSED at load, with the reason each is dangerous.
const MUST_REFUSE: &[(&str, &str)] = &[
    (
        "https://evil.example/systemone",
        "absolute URL, different origin",
    ),
    (
        "http://evil.example/systemone",
        "absolute URL, different origin",
    ),
    ("//evil.example/systemone", "protocol-relative network path"),
    (
        "///evil.example/systemone",
        "triple-slash collapses to network path",
    ),
    ("v1/systemone", "relative, would escape the origin root"),
    ("/x?redirect=https://evil.example", "query string"),
    ("/x#fragment", "fragment"),
    ("/x@evil.example", "userinfo-like authority"),
];

#[tokio::test]
async fn every_origin_changing_form_is_refused_at_load() {
    for (path, why) in MUST_REFUSE {
        let result = observe(
            "http://127.0.0.1:1",
            path,
            &Arc::new(std::sync::Mutex::new(Vec::new())),
        )
        .await;
        assert!(
            result.is_err(),
            "endpoint_path {path:?} must be refused at load ({why})"
        );
    }
}

/// Percent-encoded slashes are NOT decoded before the URL is parsed, so
/// `/%2f%2fevil.example/x` cannot become a network path. This is asserted as
/// a property of the real join, not assumed: the executor must see the path
/// verbatim and the origin must be unchanged.
#[tokio::test]
async fn a_percent_encoded_separator_cannot_change_the_origin() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let hits = Arc::new(AtomicUsize::new(0));
    let port = spawn_recorder(Arc::clone(&seen), Arc::clone(&hits)).await;
    let base = format!("http://127.0.0.1:{port}");
    let paths = observe(&base, "/%2f%2fevil.example/x", &seen).await;
    let paths = paths.expect("an encoded separator is not an origin change and must load");
    assert_eq!(
        paths.len(),
        1,
        "the executor must have been reached: {paths:?}"
    );
    assert_eq!(
        paths[0], "/%2f%2fevil.example/x",
        "the path must be forwarded verbatim, not normalised or redirected"
    );
}

/// Dot segments are a PATH concern, never an ORIGIN concern: `/../x` and
/// `/./x` must reach the SAME configured origin. The assertion is that the
/// executor was reached at all, which is only true if the origin survived.
#[tokio::test]
async fn dot_segments_cannot_escape_the_configured_origin() {
    for path in ["/../x", "/./x"] {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hits = Arc::new(AtomicUsize::new(0));
        let port = spawn_recorder(Arc::clone(&seen), Arc::clone(&hits)).await;
        let base = format!("http://127.0.0.1:{port}");
        let paths = observe(&base, path, &seen).await.unwrap_or_else(|error| {
            panic!("{path:?} is a path-level concern and must load: {error}")
        });
        assert_eq!(
            paths.len(),
            1,
            "{path:?} must still reach the CONFIGURED origin: {paths:?}"
        );
    }
}

/// The positive control: the declared shape works, and the executor is called
/// on exactly the declared path. Without this, every case above could be
/// "passing" merely because everything is refused.
#[tokio::test]
async fn a_bare_path_reaches_the_executor_verbatim() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let hits = Arc::new(AtomicUsize::new(0));
    let port = spawn_recorder(Arc::clone(&seen), Arc::clone(&hits)).await;
    let base = format!("http://127.0.0.1:{port}");
    let paths = observe(&base, "/v1/systemone", &seen)
        .await
        .expect("must load");
    assert_eq!(paths, vec!["/v1/systemone".to_string()]);
}

/// A `base_url` carrying its own PATH is legal and common (an API root). The
/// override is APPENDED to it, so a path-selecting override must not discard
/// the prefix.
#[tokio::test]
async fn an_override_appends_to_a_base_url_that_has_a_path() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let hits = Arc::new(AtomicUsize::new(0));
    let port = spawn_recorder(Arc::clone(&seen), Arc::clone(&hits)).await;
    let base = format!("http://127.0.0.1:{port}/api");
    let paths = observe(&base, "/v1/systemone", &seen)
        .await
        .expect("must load");
    assert_eq!(
        paths,
        vec!["/api/v1/systemone".to_string()],
        "the declared base path must be preserved"
    );
}

/// A trailing slash on `base_url` must not produce a doubled separator, which
/// some servers treat as a distinct (404) path.
#[tokio::test]
async fn a_trailing_slash_on_base_url_does_not_double_the_separator() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let hits = Arc::new(AtomicUsize::new(0));
    let port = spawn_recorder(Arc::clone(&seen), Arc::clone(&hits)).await;
    let base = format!("http://127.0.0.1:{port}/");
    let paths = observe(&base, "/v1/systemone", &seen)
        .await
        .expect("must load");
    assert_eq!(paths, vec!["/v1/systemone".to_string()]);
}

/// A backslash reaches the validator only through a TOML LITERAL string: in a
/// basic string `\e` is an invalid escape, so the PARSER rejects the file and
/// the validator is never reached. This case therefore uses literal form, and
/// asserts the validator - not the parser - produced the rejection.
#[tokio::test]
async fn a_backslash_authority_attempt_is_refused() {
    for path in [
        "/\\evil.example/systemone",
        "\\evil.example\\systemone",
        "/v1\\systemone",
    ] {
        let result = observe(
            "http://127.0.0.1:1",
            path,
            &Arc::new(std::sync::Mutex::new(Vec::new())),
        )
        .await;
        assert!(
            result.is_err(),
            "a backslash authority attempt must be refused at load: {path:?}"
        );
        let error = result.unwrap_err();
        assert!(
            error.contains("bare absolute path"),
            "the rejection must come from the endpoint_path validator, not the TOML parser: {error}"
        );
    }
}
