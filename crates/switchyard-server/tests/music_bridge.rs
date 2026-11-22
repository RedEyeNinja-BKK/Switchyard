// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the music bridge (`/v1/music/{capability}`).
//!
//! A mock Google endpoint speaks the real `BidiGenerateMusic` protocol, so these assert
//! what a caller and the provider each actually see: the setup carries the model and
//! nothing else, a caller prompt is translated into the documented weighted-prompt shape
//! rather than forwarded, the running audio total travels with the provider's frame, and
//! the declared ceiling closes the session instead of over-delivering.

use std::sync::{Arc, Mutex};

use axum::Router;
use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;
use serde_json::{Value, json};
use switchyard_runner::Runner;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{accept_hdr_async, connect_async};

const KEY_ENV: &str = "SY_TEST_MUSIC_KEY";
const KEY_VALUE: &str = "server-owned-music-key";

fn config(engine: &str, ceiling: u64) -> String {
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

[capability_clients.music]
format = "google_music"
base_url = "{engine}"
model = "lyria-test-realtime"
api_key_env = "{KEY_ENV}"
timeout_seconds = 30

[capabilities.music-cap]
id = "music-cap"
target = "music"
max_session_audio_bytes = {ceiling}
"#
    )
}

async fn serve_bridge(engine: &str, ceiling: u64) -> String {
    let runner = Runner::from_toml(&config(engine, ceiling)).expect("runner builds");
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

/// Accepts one bridge connection, records the URI and every frame it sees, then runs the
/// scripted provider conversation and returns.
async fn accept_provider(
    listener: TcpListener,
) -> (
    tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    Arc<Mutex<String>>,
    Arc<Mutex<Vec<String>>>,
) {
    let uri = Arc::new(Mutex::new(String::new()));
    let frames = Arc::new(Mutex::new(Vec::new()));
    let uri_seen = Arc::clone(&uri);
    let (stream, _) = listener.accept().await.expect("accept tcp");
    let socket = accept_hdr_async(
        stream,
        |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
            *uri_seen.lock().expect("uri") = request.uri().to_string();
            Ok(response)
        },
    )
    .await
    .expect("accept ws");
    (socket, uri, frames)
}

/// Reads one provider-visible frame as text, recording it.
async fn next_frame(
    socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    frames: &Arc<Mutex<Vec<String>>>,
    label: &str,
) -> String {
    let message = socket.next().await.expect("frame").expect("frame");
    let text = match message {
        WsMessage::Text(text) => text.to_string(),
        WsMessage::Binary(bytes) => String::from_utf8(bytes.to_vec()).expect("utf8"),
        other => panic!("unexpected provider frame: {other:?}"),
    };
    frames
        .lock()
        .expect("frames")
        .push(format!("{label}:{text}"));
    text
}

fn caller_text(message: WsMessage) -> String {
    match message {
        WsMessage::Text(text) => text.to_string(),
        WsMessage::Binary(bytes) => String::from_utf8(bytes.to_vec()).expect("utf8"),
        other => panic!("unexpected caller frame: {other:?}"),
    }
}

#[tokio::test]
async fn a_prompt_is_translated_and_audio_is_counted_for_the_caller() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let google_address = listener.local_addr().expect("addr").to_string();

    let provider = tokio::spawn(async move {
        let (mut socket, uri, frames) = accept_provider(listener).await;

        // 1. The setup is the first frame the provider sees and carries only the model.
        next_frame(&mut socket, &frames, "setup").await;
        socket
            .send(WsMessage::Text(json!({"setupComplete": {}}).to_string()))
            .await
            .expect("setupComplete");

        // 2. The caller's prompt arrives translated, not verbatim.
        next_frame(&mut socket, &frames, "prompt").await;

        // 3. Google delivers audio in binary JSON frames.
        socket
            .send(WsMessage::Binary(
                json!({"serverContent": {"audioChunks": [{"data": "AAECAwQFBgcICQoL"}]}})
                    .to_string()
                    .into_bytes(),
            ))
            .await
            .expect("audio chunk");
        socket.close(None).await.expect("close");
        (uri, frames)
    });

    let bridge = serve_bridge(&format!("ws://{google_address}"), 1024).await;
    let (mut caller, _) = connect_async(format!("{bridge}/v1/music/music-cap"))
        .await
        .expect("caller connects");

    let setup_complete = caller_text(caller.next().await.expect("frame").expect("frame"));
    assert!(
        setup_complete.contains("setupComplete"),
        "the provider's setupComplete reaches the caller: {setup_complete}"
    );

    caller
        .send(WsMessage::Text(
            json!({"clientContent": {"weightedPrompts": [
                {"text": "  warm lofi piano  ", "weight": 2.0}
            ]}})
            .to_string(),
        ))
        .await
        .expect("client frame");

    let audio = caller_text(caller.next().await.expect("frame").expect("frame"));
    let audio: Value = serde_json::from_str(&audio).expect("json audio frame");
    assert_eq!(
        audio
            .get("switchyardDeliveredAudioBytes")
            .and_then(Value::as_u64),
        Some(12),
        "the caller is told how much audio the session has delivered: {audio}"
    );
    assert_eq!(
        audio
            .get("switchyardAudioCeilingBytes")
            .and_then(Value::as_u64),
        Some(1024),
        "the ceiling travels beside the running total: {audio}"
    );

    let (uri, frames) = provider.await.expect("provider task");
    let frames = frames.lock().expect("frames").clone();

    let setup: Value = serde_json::from_str(
        frames
            .iter()
            .find(|frame| frame.starts_with("setup:"))
            .expect("setup recorded")
            .trim_start_matches("setup:"),
    )
    .expect("setup json");
    assert_eq!(
        setup,
        json!({"setup": {"model": "models/lyria-test-realtime"}}),
        "a BidiGenerateMusic setup carries the model and nothing else: {setup}"
    );
    assert!(
        !setup.to_string().contains(KEY_VALUE),
        "the credential must never appear in a provider-visible setup: {setup}"
    );

    let prompt: Value = serde_json::from_str(
        frames
            .iter()
            .find(|frame| frame.starts_with("prompt:"))
            .expect("prompt recorded")
            .trim_start_matches("prompt:"),
    )
    .expect("prompt json");
    assert_eq!(
        prompt,
        json!({"clientContent": {"weightedPrompts": [
            {"text": "warm lofi piano", "weight": 2.0}
        ]}}),
        "the caller's text is trimmed and rebuilt, never forwarded verbatim"
    );

    let uri = uri.lock().expect("uri").clone();
    assert!(
        uri.contains(&format!("key={KEY_VALUE}")),
        "the credential travels in the upstream URL: {uri}"
    );
    assert!(
        uri.contains("BidiGenerateMusic"),
        "the session opens the music method, not the Live one: {uri}"
    );
}

#[tokio::test]
async fn the_declared_audio_ceiling_closes_the_session_instead_of_over_delivering() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let google_address = listener.local_addr().expect("addr").to_string();

    let provider = tokio::spawn(async move {
        let (mut socket, _uri, frames) = accept_provider(listener).await;
        next_frame(&mut socket, &frames, "setup").await;
        socket
            .send(WsMessage::Text(json!({"setupComplete": {}}).to_string()))
            .await
            .expect("setupComplete");
        // 12 decoded bytes against a declared ceiling of 8.
        socket
            .send(WsMessage::Binary(
                json!({"serverContent": {"audioChunks": [{"data": "AAECAwQFBgcICQoL"}]}})
                    .to_string()
                    .into_bytes(),
            ))
            .await
            .expect("audio chunk");
        // The bridge should close rather than forward; swallow the close.
        let _ = socket.next().await;
    });

    let bridge = serve_bridge(&format!("ws://{google_address}"), 8).await;
    let (mut caller, _) = connect_async(format!("{bridge}/v1/music/music-cap"))
        .await
        .expect("caller connects");

    let setup_complete = caller_text(caller.next().await.expect("frame").expect("frame"));
    assert!(setup_complete.contains("setupComplete"));

    // The frame that would cross the ceiling is not forwarded: the caller's next frame is
    // the close, whose reason names the declared audio.
    let next = caller.next().await.expect("frame").expect("frame");
    match next {
        WsMessage::Close(Some(frame)) => {
            assert_eq!(
                frame.code,
                tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
                "a spent audio budget is not an error"
            );
            assert!(
                frame.reason.contains("delivered the audio"),
                "the close frame explains the ceiling: {}",
                frame.reason
            );
        }
        other => panic!("expected a close frame, got {other:?}"),
    }

    provider.await.expect("provider task");
}

#[tokio::test]
async fn a_caller_cannot_open_a_session_with_its_own_setup_frame() {
    unsafe { std::env::set_var(KEY_ENV, KEY_VALUE) };

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let google_address = listener.local_addr().expect("addr").to_string();

    let provider = tokio::spawn(async move {
        let (mut socket, _uri, frames) = accept_provider(listener).await;
        next_frame(&mut socket, &frames, "setup").await;
        socket
            .send(WsMessage::Text(json!({"setupComplete": {}}).to_string()))
            .await
            .expect("setupComplete");
        // Nothing but the close may arrive: the caller's setup attempt is refused locally,
        // so the upstream connection ends without carrying it.
        match socket.next().await {
            None | Some(Ok(WsMessage::Close(_))) => {}
            Some(Ok(other)) => {
                panic!("a caller's setup frame must never reach the provider: {other:?}")
            }
            // A reset while the bridge tears the connection down is the same outcome: the
            // frame did not travel as a protocol message.
            Some(Err(_)) => {}
        }
    });

    let bridge = serve_bridge(&format!("ws://{google_address}"), 1024).await;
    let (mut caller, _) = connect_async(format!("{bridge}/v1/music/music-cap"))
        .await
        .expect("caller connects");

    let setup_complete = caller_text(caller.next().await.expect("frame").expect("frame"));
    assert!(setup_complete.contains("setupComplete"));

    caller
        .send(WsMessage::Text(
            json!({"setup": {"model": "models/attacker-choice"}}).to_string(),
        ))
        .await
        .expect("client frame");

    let next = caller.next().await.expect("frame").expect("frame");
    match next {
        WsMessage::Close(Some(frame)) => {
            assert_eq!(
                frame.code,
                tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Policy,
                "a contract violation is a policy close"
            );
            assert!(
                frame.reason.contains("setup"),
                "the refusal names the field: {}",
                frame.reason
            );
        }
        other => panic!("expected a close frame, got {other:?}"),
    }

    provider.await.expect("provider task");
}
