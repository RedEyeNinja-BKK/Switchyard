// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the Live bridge (`/v1/live/{capability}`).
//!
//! A mock Google endpoint speaks the real WebSocket protocol, so these assert what a
//! caller and the provider each actually see: the setup frame is built from the
//! capability declaration, both directions are forwarded, Google's binary JSON frames
//! reach the caller intact, and the server-owned credential stays in the upstream URL.

use std::sync::{Arc, Mutex};

use axum::Router;
use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;
use serde_json::{Value, json};
use switchyard_runner::Runner;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{accept_hdr_async, connect_async};

const KEY_ENV: &str = "SY_TEST_LIVE_KEY";
const KEY_VALUE: &str = "server-owned-key";

fn config(engine: &str) -> String {
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

[capability_clients.live]
format = "google_live"
base_url = "{engine}"
model = "gemini-test-live"
api_key_env = "{KEY_ENV}"
timeout_seconds = 60

[capabilities.live-cap]
id = "live-cap"
target = "live"
response_modalities = ["AUDIO"]
system_instruction = "Answer briefly."
thinking_level = "HIGH"
"#
    )
}

/// A stand-in for Google's BidiGenerateContent endpoint.
///
/// Records the request URI (so the test can see the credential it received) and the
/// decoded frames the bridge sends, then answers with a `setupComplete` and one
/// `serverContent` carrying a transcription.
async fn mock_google() -> (TcpListener, Arc<Mutex<Vec<String>>>, Arc<Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let frames = Arc::new(Mutex::new(Vec::new()));
    let uri = Arc::new(Mutex::new(String::new()));
    (listener, frames, uri)
}

async fn serve_bridge(engine: &str) -> String {
    let runner = Runner::from_toml(&config(engine)).expect("runner builds");
    let app: Router = switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("ws://{address}")
}

#[tokio::test]
async fn the_setup_is_built_locally_and_credentials_stay_upstream() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };

    let (google_listener, frames, upstream_uri) = mock_google().await;
    let google_address = google_listener.local_addr().expect("addr").to_string();

    let uri_seen = Arc::clone(&upstream_uri);
    let frames_seen = Arc::clone(&frames);
    let google = tokio::spawn(async move {
        let (stream, _) = google_listener.accept().await.expect("accept tcp");
        let mut socket = accept_hdr_async(
            stream,
            |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
                *uri_seen.lock().expect("uri") = request.uri().to_string();
                Ok(response)
            },
        )
        .await
        .expect("accept ws");

        // 1. The setup must be the first frame the provider sees.
        if let Some(Ok(WsMessage::Text(text))) = socket.next().await {
            frames_seen
                .lock()
                .expect("frames")
                .push(format!("setup:{text}"));
        }
        socket
            .send(WsMessage::Text(json!({"setupComplete": {}}).to_string()))
            .await
            .expect("setupComplete");

        // 2. The caller's frame is forwarded verbatim.
        if let Some(Ok(WsMessage::Text(text))) = socket.next().await {
            frames_seen
                .lock()
                .expect("frames")
                .push(format!("client:{text}"));
        }

        // 3. Google answers with a transcript. It delivers JSON in BINARY frames.
        socket
            .send(WsMessage::Binary(
                json!({"serverContent": {"outputTranscription": {"text": "The quick brown fox"}}})
                    .to_string()
                    .into_bytes(),
            ))
            .await
            .expect("serverContent");
        socket.close(None).await.expect("close");
    });

    let bridge = serve_bridge(&format!("ws://{google_address}")).await;
    let (mut caller, _) = connect_async(format!("{bridge}/v1/live/live-cap"))
        .await
        .expect("caller connects");

    let first = caller.next().await.expect("frame").expect("frame");
    let first_text = match first {
        WsMessage::Text(text) => text.to_string(),
        WsMessage::Binary(bytes) => String::from_utf8(bytes.to_vec()).expect("utf8"),
        other => panic!("unexpected setup frame: {other:?}"),
    };
    assert!(
        first_text.contains("setupComplete"),
        "first caller frame must be setupComplete: {first_text}"
    );

    caller
        .send(WsMessage::Text(
            json!({"realtimeInput": {"text": "Say hello"}}).to_string(),
        ))
        .await
        .expect("client frame");

    let second = caller.next().await.expect("frame").expect("frame");
    let second_text = match second {
        WsMessage::Text(text) => text.to_string(),
        WsMessage::Binary(bytes) => String::from_utf8(bytes.to_vec()).expect("utf8"),
        other => panic!("unexpected content frame: {other:?}"),
    };
    let event: Value = serde_json::from_str(&second_text).expect("json event");
    assert_eq!(
        event
            .get("serverContent")
            .and_then(|content| content.get("outputTranscription"))
            .and_then(|transcript| transcript.get("text"))
            .and_then(Value::as_str),
        Some("The quick brown fox"),
        "provider transcript must reach the caller: {second_text}"
    );

    google.await.expect("google task");

    let frames = frames.lock().expect("frames").clone();
    let setup = frames
        .iter()
        .find(|frame| frame.starts_with("setup:"))
        .expect("setup frame recorded");
    let setup: Value = serde_json::from_str(&setup["setup:".len()..]).expect("setup json");
    assert_eq!(
        setup
            .get("setup")
            .and_then(|s| s.get("model"))
            .and_then(Value::as_str),
        Some("models/gemini-test-live"),
        "setup names the declared model"
    );
    assert_eq!(
        setup
            .get("setup")
            .and_then(|s| s.get("generationConfig"))
            .and_then(|c| c.get("responseModalities")),
        Some(&json!(["AUDIO"])),
        "setup carries the declared modalities"
    );
    assert_eq!(
        setup
            .get("setup")
            .and_then(|s| s.get("generationConfig"))
            .and_then(|c| c.get("thinkingConfig")),
        Some(&json!({"thinkingLevel": "HIGH"})),
        "a Gemini 3.x session takes a thinking level, not a budget"
    );
    assert!(
        !setup.to_string().contains(KEY_VALUE),
        "the credential must never appear in a provider-visible setup: {setup}"
    );

    let forwarded = frames
        .iter()
        .find(|frame| frame.starts_with("client:"))
        .expect("client frame recorded");
    assert_eq!(
        forwarded, "client:{\"realtimeInput\":{\"text\":\"Say hello\"}}",
        "the caller's frame is forwarded verbatim"
    );

    let uri = upstream_uri.lock().expect("uri").clone();
    assert!(
        uri.contains(&format!("key={KEY_VALUE}")),
        "the credential travels in the upstream URL: {uri}"
    );
}
