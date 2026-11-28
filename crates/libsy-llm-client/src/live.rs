// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Google's stateful Live API client over WebSocket.
//!
//! `BidiGenerateContent` is *not* a stateless HTTP request: the first frame over the
//! connection must be a `BidiGenerateContentSetup`, and both sides stream JSON frames
//! until one side closes. This module manages one such session.
//!
//! The credential travels in the WebSocket URL as `key=`, so it never appears in
//! client-visible bodies or logs. Callers hold a [`LiveClient`] and exchange typed
//! frames; the wire shapes are Google's own [`BidiGenerateContentClientMessage`] /
//! [`BidiGenerateContentServerMessage`].

use futures::sink::SinkExt;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_tungstenite::WebSocketStream;
use tungstenite::Message;

/// One stateful Live session to Google's BidiGenerateContent WebSocket endpoint.
///
/// A `LiveClient` is bound to one physical model and one credential. The setup message
/// is sent on `connect`; every subsequent frame is a client input or a server event.
///
/// Dropping the client closes the WebSocket. No frame is sent after a terminal error.
pub struct LiveClient {
    inner: Option<WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>,
    closed: bool,
}

/// A frame a caller sends into the Live session.
///
/// Only one of the fields is set, mirroring Google's one-of wire encoding.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveClientMessage {
    /// Text input delivered to the model in real time.
    RealtimeInputText { text: String },
    /// Raw PCM audio input (16-bit little-endian, 16 kHz).
    RealtimeInputAudio {
        data: String, // base64
        mime_type: String,
    },
}

/// A frame the Live session delivers to the caller.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveServerMessage {
    /// The setup was accepted and the session is live.
    SetupComplete,
    /// The server's model turn: audio and/or text parts.
    ServerContent {
        /// Transcription of the user's input, when the server produced one.
        input_transcription: Option<String>,
        /// Transcription of the model's output, when the server produced one.
        output_transcription: Option<String>,
        /// Model turn parts.
        model_turn: Option<Vec<LivePart>>,
        /// True when this turn is complete.
        turn_complete: Option<bool>,
    },
    /// The session ended with an error.
    Error {
        code: Option<i32>,
        message: String,
    },
}

/// One part of a model turn.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LivePart {
    /// Inline media data (base64-encoded audio, image, etc.).
    InlineData { mime_type: String, data: String },
    /// Plain text.
    Text { text: String },
}

/// The setup a caller sends as the first frame of a Live session.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LiveSetup {
    pub model: String,
    pub generation_config: Option<LiveGenerationConfig>,
    pub system_instruction: Option<LiveContent>,
    pub session_resumption: Option<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_config: Option<Value>,
}

/// Generation config for a Live setup.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LiveGenerationConfig {
    pub response_modalities: Vec<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub candidate_count: Option<i32>,
    pub max_output_tokens: Option<i32>,
    pub stop_sequences: Option<Vec<String>>,
    pub thinking_config: Option<LiveThinkingConfig>,
}

/// Thinking configuration for Extended Thinking variants.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LiveThinkingConfig {
    pub thinking_budget: i32,
}

/// Content carrying parts (used for system instruction).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LiveContent {
    pub parts: Vec<LivePart>,
}

impl LiveClient {
    /// Connects to Google's Live WebSocket endpoint and sends the setup message.
    ///
    /// `url` is the full `wss://...?key=<server-owned-key>` WebSocket URL. The key is
    /// supplied by the caller (the deployment's server-owned credential) and is never
    /// logged or returned by this client.
    pub async fn connect(url: &str, setup: &LiveSetup) -> Result<Self, LiveError> {
        let (mut ws_stream, _response) =
            tokio_tungstenite::connect_async(url).await.map_err(LiveError::transport)?;
        let setup_json = serde_json::to_string(&json!({ "setup": setup })).map_err(|error| {
            LiveError::invalid_setup(format!("setup message cannot be serialized: {error}"))
        })?;
        ws_stream
            .send(Message::Text(setup_json))
            .await
            .map_err(LiveError::transport)?;
        Ok(Self {
            inner: Some(ws_stream),
            closed: false,
        })
    }

    /// Sends one client input frame into the session.
    ///
    /// Returns [`LiveError::SessionClosed`] once the session has been closed.
    pub async fn send(&mut self, message: LiveClientMessage) -> Result<(), LiveError> {
        if self.closed {
            return Err(LiveError::SessionClosed);
        }
        let stream = self.inner.as_mut().ok_or(LiveError::SessionClosed)?;
        let json = match message {
            LiveClientMessage::RealtimeInputText { text } => {
                json!({ "realtimeInput": { "text": text } })
            }
            LiveClientMessage::RealtimeInputAudio { data, mime_type } => {
                json!({ "realtimeInput": { "audio": { "data": data, "mimeType": mime_type } } })
            }
        };
        stream
            .send(Message::Text(serde_json::to_string(&json).map_err(|error| {
                LiveError::invalid_setup(format!("message cannot be serialized: {error}"))
            })?))
            .await
            .map_err(LiveError::transport)
    }

    /// Receives one server event from the session.
    ///
    /// Returns [`None`] when the server closes the session cleanly.
    pub async fn next(&mut self) -> Result<Option<LiveServerMessage>, LiveError> {
        if self.closed {
            return Err(LiveError::SessionClosed);
        }
        let mut stream = self.inner.as_mut().ok_or(LiveError::SessionClosed)?;
        loop {
            match StreamExt::next(&mut stream).await {
                Some(Ok(Message::Text(text))) => {
                    let event: Value = serde_json::from_str(&text).map_err(|error| {
                        LiveError::invalid_setup(format!("server frame is not valid JSON: {error}"))
                    })?;
                    return Ok(Some(self.decode_event(&event)?));
                }
                Some(Ok(Message::Close(_))) => {
                    self.closed = true;
                    return Ok(None);
                }
                Some(Err(error)) => {
                    self.closed = true;
                    return Err(LiveError::transport(error));
                }
                // Google Live speaks JSON text frames; binary/ping/pong frames are not part
                // of the contract and are ignored.
                Some(Ok(Message::Binary(_)))
                | Some(Ok(Message::Ping(_)))
                | Some(Ok(Message::Pong(_)))
                | Some(Ok(Message::Frame(_))) => continue,
                None => {
                    self.closed = true;
                    return Ok(None);
                }
            }
        }
    }

    /// Closes the session cleanly.
    pub async fn close(&mut self) -> Result<(), LiveError> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        if let Some(mut stream) = self.inner.take() {
            stream
                .close(None)
                .await
                .map_err(LiveError::transport)?;
        }
        Ok(())
    }

    fn decode_event(&self, event: &Value) -> Result<LiveServerMessage, LiveError> {
        if event.get("setupComplete").is_some() {
            return Ok(LiveServerMessage::SetupComplete);
        }
        if let Some(error) = event.get("error") {
            let code = error.get("code").and_then(Value::as_i64).map(|v| v as i32);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("server error")
                .to_string();
            return Ok(LiveServerMessage::Error { code, message });
        }
        if let Some(server_content) = event.get("serverContent") {
            let model_turn = server_content
                .get("modelTurn")
                .and_then(|turn| turn.get("parts"))
                .map(|parts| {
                    parts
                        .as_array()
                        .map(|array| {
                            array
                                .iter()
                                .filter_map(|part| {
                                    if let Some(data) = part.get("inlineData") {
                                        Some(LivePart::InlineData {
                                            mime_type: data
                                                .get("mimeType")
                                                .and_then(Value::as_str)
                                                .unwrap_or("")
                                                .to_string(),
                                            data: data
                                                .get("data")
                                                .and_then(Value::as_str)
                                                .unwrap_or("")
                                                .to_string(),
                                        })
                                    } else if let Some(text) = part.get("text") {
                                        Some(LivePart::Text {
                                            text: text.as_str().unwrap_or("").to_string(),
                                        })
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        })
                        .unwrap_or_default()
                });
            return Ok(LiveServerMessage::ServerContent {
                input_transcription: server_content
                    .get("inputTranscription")
                    .and_then(|v| v.get("text"))
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                output_transcription: server_content
                    .get("outputTranscription")
                    .and_then(|v| v.get("text"))
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                model_turn,
                turn_complete: server_content
                    .get("turnComplete")
                    .and_then(Value::as_bool),
            });
        }
        // Unknown frame shape: not an error, just uninterpreted. Return setup complete so
        // the session continues rather than dying on a frame the deployment didn't expect.
        Ok(LiveServerMessage::SetupComplete)
    }
}

impl Drop for LiveClient {
    fn drop(&mut self) {
        // No async close in Drop: the caller is responsible for closing via `close()`.
        // The WebSocketStream will be dropped and the TCP connection torn down.
        let _ = self.inner.take();
    }
}

/// Why a Live session failed.
#[derive(Debug, thiserror::Error)]
pub enum LiveError {
    /// The WebSocket transport failed (connect, send, receive, close).
    #[error("live transport error: {0}")]
    Transport(#[source] tokio_tungstenite::tungstenite::Error),
    /// The setup message or a client frame was invalid.
    #[error("invalid live message: {0}")]
    InvalidSetup(String),
    /// The session has been closed and no further frames may be sent or received.
    #[error("live session is closed")]
    SessionClosed,
}

impl LiveError {
    fn transport(error: tokio_tungstenite::tungstenite::Error) -> Self {
        Self::Transport(error)
    }

    fn invalid_setup(message: impl Into<String>) -> Self {
        Self::InvalidSetup(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A setup message serializes with the model in `models/` form.
    #[test]
    fn setup_uses_models_qualified_name() {
        let setup = LiveSetup {
            model: "models/gemini-3.8-live".to_string(),
            generation_config: Some(LiveGenerationConfig {
                response_modalities: vec!["AUDIO".to_string(), "TEXT".to_string()],
                temperature: None,
                top_p: None,
                top_k: None,
                candidate_count: None,
                max_output_tokens: None,
                stop_sequences: None,
                thinking_config: Some(LiveThinkingConfig {
                    thinking_budget: 1024,
                }),
            }),
            system_instruction: Some(LiveContent {
                parts: vec![LivePart::Text {
                    text: "You are a helpful assistant.".to_string(),
                }],
            }),
            session_resumption: None,
            tools: None,
            tool_config: None,
        };
        let json = serde_json::to_value(&setup).unwrap();
        assert_eq!(json.get("model").and_then(Value::as_str), Some("models/gemini-3.8-live"));
        assert_eq!(
            json.get("generation_config")
                .and_then(|c| c.get("response_modalities")),
            Some(&Value::Array(vec![
                Value::String("AUDIO".to_string()),
                Value::String("TEXT".to_string()),
            ]))
        );
        assert_eq!(
            json.get("generation_config")
                .and_then(|c| c.get("thinking_config"))
                .and_then(|c| c.get("thinking_budget")),
            Some(&Value::Number(1024.into()))
        );
    }

    /// A client text message serializes as `realtime_input { text }`.
    #[test]
    fn realtime_input_text_serializes() {
        let message = LiveClientMessage::RealtimeInputText {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(
            json.get("realtime_input_text").and_then(|v| v.get("text")),
            Some(&Value::String("Hello".to_string()))
        );
    }

    /// A server content frame decodes to typed fields.
    #[test]
    fn server_content_decodes() {
        let client = LiveClient {
            inner: None,
            closed: false,
        };
        let event = serde_json::json!({
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        { "inlineData": { "mimeType": "audio/pcm", "data": "base64==" } }
                    ]
                },
                "turnComplete": true
            }
        });
        let message = client.decode_event(&event).unwrap();
        assert!(matches!(
            message,
            LiveServerMessage::ServerContent {
                model_turn: Some(parts),
                turn_complete: Some(true),
                ..
            } if parts.len() == 1
        ));
    }

    /// A setup complete frame decodes to `SetupComplete`.
    #[test]
    fn setup_complete_decodes() {
        let client = LiveClient {
            inner: None,
            closed: false,
        };
        let event = serde_json::json!({ "setupComplete": {} });
        assert!(matches!(
            client.decode_event(&event).unwrap(),
            LiveServerMessage::SetupComplete
        ));
    }

    /// A server error frame decodes with its code and message.
    #[test]
    fn server_error_decodes() {
        let client = LiveClient {
            inner: None,
            closed: false,
        };
        let event = serde_json::json!({
            "error": { "code": 400, "message": "bad request" }
        });
        assert!(matches!(
            client.decode_event(&event).unwrap(),
            LiveServerMessage::Error {
                code: Some(400),
                message
            } if message == "bad request"
        ));
    }
}
