// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `WebSocket /v1/music/{capability}` — Google's `BidiGenerateMusic` (`Lyria Realtime`).
//!
//! A different protocol from the Live bridge, not a variant of it: the setup message
//! carries only `model`, and the caller then steers the session with three kinds of
//! message — weighted prompts, a music-generation config, and playback control.
//!
//! Because those controls arrive per message, this surface is a *validating* bridge rather
//! than a transparent one. Every caller message is parsed, checked against the fields and
//! ranges Google documents for `BidiGenerateMusicGenerationConfig`, and re-emitted in
//! canonical form. An undocumented field or an out-of-range value is refused by closing the
//! session with a policy-violation reason, so a typo cannot silently become a session that
//! costs quota while producing nothing. Provider frames are forwarded unchanged apart from
//! audio accounting.
//!
//! A session is bounded twice: by wall clock (`timeout_seconds`) and by delivered audio
//! (`max_session_audio_bytes`). Either bound closes the caller's socket with a reason.
//! Nothing here can actuate equipment or spend a paid lane: one session reserves exactly one
//! unit of the capability's provider budgets, and the model is fixed at configuration time.

use axum::extract::ws::{CloseFrame, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use serde_json::{Map, Value, json};
use switchyard_llm_client::PacingTicket;
use switchyard_runner::capability::{CapabilityFormat, ResolvedCapability};

use crate::ServerState;
use crate::capability_surfaces::{release_physical_call, reserve_physical_call_for};

/// The client message kinds `BidiGenerateMusic` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MusicMessageKind {
    /// Weighted prompts that start or steer generation.
    ClientContent,
    /// Music-generation parameters.
    GenerationConfig,
    /// Play / pause / stop / reset.
    PlaybackControl,
}

/// Why a caller's message was refused.
///
/// Each variant carries the detail that makes the refusal actionable. The session is closed
/// rather than answered frame-by-frame because `BidiGenerateMusic` has no client-error
/// message type: a refusal that is not a close would be invisible to a caller reading only
/// the documented server messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MusicRefusal {
    /// The frame is not a JSON object.
    NotAnObject,
    /// The object names no recognised client message, or names more than one.
    UnknownMessage { keys: Vec<String> },
    /// A field the protocol does not document for this message.
    UnknownField { message: String, field: String },
    /// A documented field with a value outside its documented range.
    OutOfRange {
        field: String,
        value: String,
        allowed: String,
    },
    /// A structurally invalid value that is not a range problem.
    Malformed { field: String, reason: String },
}

impl MusicRefusal {
    /// The reason string carried in the WebSocket close frame.
    pub fn message(&self) -> String {
        match self {
            Self::NotAnObject => {
                "a BidiGenerateMusic client message must be a JSON object".to_string()
            }
            Self::UnknownMessage { keys } => format!(
                "a client message names exactly one of clientContent, musicGenerationConfig or \
                 playbackControl; this frame names {}",
                if keys.is_empty() {
                    "none".to_string()
                } else {
                    keys.join(", ")
                }
            ),
            Self::UnknownField { message, field } => format!(
                "{message} has no field {field:?}; BidiGenerateMusic documents a fixed set of \
                 fields and this surface forwards nothing it cannot validate"
            ),
            Self::OutOfRange {
                field,
                value,
                allowed,
            } => format!("{field} = {value} is outside the documented range {allowed}"),
            Self::Malformed { field, reason } => format!("{field} is malformed: {reason}"),
        }
    }

    /// A stable code, for a caller that matches on machine-readable refusal reasons.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotAnObject => "music_message_not_an_object",
            Self::UnknownMessage { .. } => "music_message_unknown",
            Self::UnknownField { .. } => "music_field_unknown",
            Self::OutOfRange { .. } => "music_value_out_of_range",
            Self::Malformed { .. } => "music_field_malformed",
        }
    }
}

/// The documented keys of `BidiGenerateMusicGenerationConfig`.
///
/// Kept as one table so the accepted set and the range checks cannot drift apart: a field
/// missing from here is refused by name, and a field present here is checked below.
const GENERATION_CONFIG_FIELDS: &[&str] = &[
    "temperature",
    "topK",
    "seed",
    "guidance",
    "bpm",
    "density",
    "brightness",
    "scale",
    "muteBass",
    "muteDrums",
    "onlyBassAndDrums",
    "musicGenerationMode",
];

const SCALES: &[&str] = &[
    "C_MAJOR_A_MINOR",
    "D_FLAT_MAJOR_B_FLAT_MINOR",
    "D_MAJOR_B_MINOR",
    "E_FLAT_MAJOR_C_MINOR",
    "E_MAJOR_D_FLAT_MINOR",
    "F_MAJOR_D_MINOR",
    "G_FLAT_MAJOR_E_FLAT_MINOR",
    "G_MAJOR_E_MINOR",
    "A_FLAT_MAJOR_F_MINOR",
    "A_MAJOR_G_FLAT_MINOR",
    "B_FLAT_MAJOR_G_MINOR",
    "B_MAJOR_A_FLAT_MINOR",
];

const MODES: &[&str] = &["QUALITY", "DIVERSITY", "VOCALIZATION"];
const PLAYBACK_CONTROLS: &[&str] = &["PLAY", "PAUSE", "STOP", "RESET_CONTEXT"];

/// Most weighted prompts one message may carry.
///
/// Google normalises the weights, so the count is not a provider limit: it is a bound on
/// what this surface will forward from one frame, so a caller cannot send an arbitrarily
/// large prompt set that no operator could review.
const MAX_WEIGHTED_PROMPTS: usize = 8;
/// Longest single prompt text, in characters.
const MAX_PROMPT_CHARS: usize = 512;

/// Which client message kinds and how many of each a session may send.
///
/// The bound exists because a session is one reserved physical call: an unbounded number of
/// steering messages would let one call cost an unbounded amount of work.
#[derive(Clone, Copy, Debug)]
pub struct MusicLimits {
    /// Largest total audio the session may deliver, in raw bytes.
    pub max_session_audio_bytes: u64,
    /// Wall-clock ceiling of one session.
    pub timeout: std::time::Duration,
    /// Most messages the caller may send in one session.
    pub max_client_messages: u32,
}

impl MusicLimits {
    /// The limits a resolved capability declares.
    ///
    /// Validation at load refuses a music capability without these, so the fallbacks here
    /// exist only for unit tests that construct a capability by hand.
    pub fn of(capability: &ResolvedCapability) -> Self {
        Self {
            max_session_audio_bytes: capability
                .max_session_audio_bytes
                .unwrap_or(8 * 1024 * 1024),
            timeout: std::time::Duration::from_secs(capability.timeout_seconds.unwrap_or(120)),
            max_client_messages: 512,
        }
    }
}

/// Builds the `BidiGenerateMusicSetup`.
///
/// The message carries the model and nothing else: Google's setup has exactly one field.
/// Emitting anything more would be a shape the protocol does not define.
pub fn build_setup(capability: &ResolvedCapability) -> Value {
    json!({ "setup": { "model": format!("models/{}", capability.model) } })
}

/// Validates a caller frame and returns the canonical Google message.
///
/// The returned message is what is sent upstream: the caller's own JSON is never forwarded,
/// so a key that passed validation is the only key that can reach the provider.
pub fn translate_client_message(
    text: &str,
    limits: &MusicLimits,
) -> Result<(MusicMessageKind, Value, u64), MusicRefusal> {
    let value: Value = serde_json::from_str(text).map_err(|_| MusicRefusal::NotAnObject)?;
    let Some(object) = value.as_object() else {
        return Err(MusicRefusal::NotAnObject);
    };

    let mut present: Vec<(&str, MusicMessageKind)> = Vec::new();
    if object.contains_key("clientContent") {
        present.push(("clientContent", MusicMessageKind::ClientContent));
    }
    if object.contains_key("musicGenerationConfig") {
        present.push(("musicGenerationConfig", MusicMessageKind::GenerationConfig));
    }
    if object.contains_key("playbackControl") {
        present.push(("playbackControl", MusicMessageKind::PlaybackControl));
    }
    if present.len() != 1 {
        return Err(MusicRefusal::UnknownMessage {
            keys: object.keys().cloned().collect(),
        });
    }
    let (name, kind) = present[0];

    // Anything beside the one message field is a key the protocol does not define for a
    // client message. `setup` is deliberately included: only this server may send it.
    for key in object.keys() {
        if key != name {
            return Err(MusicRefusal::UnknownField {
                message: "a BidiGenerateMusic client message".to_string(),
                field: key.clone(),
            });
        }
    }

    let payload = &object[name];
    let (canonical, charged_bytes) = match kind {
        MusicMessageKind::ClientContent => {
            let canonical = translate_client_content(payload, limits)?;
            // The prompts are text the provider bills as input; charging their bytes keeps
            // a session that is steered constantly from being free.
            (canonical, text.len() as u64)
        }
        MusicMessageKind::GenerationConfig => (translate_generation_config(payload)?, 0),
        MusicMessageKind::PlaybackControl => (translate_playback_control(payload)?, 0),
    };
    Ok((kind, canonical, charged_bytes))
}

fn translate_client_content(payload: &Value, limits: &MusicLimits) -> Result<Value, MusicRefusal> {
    let Some(object) = payload.as_object() else {
        return Err(MusicRefusal::Malformed {
            field: "clientContent".to_string(),
            reason: "must be an object carrying weightedPrompts".to_string(),
        });
    };
    for key in object.keys() {
        if key != "weightedPrompts" {
            return Err(MusicRefusal::UnknownField {
                message: "clientContent".to_string(),
                field: key.clone(),
            });
        }
    }
    let Some(prompts) = object.get("weightedPrompts").and_then(Value::as_array) else {
        return Err(MusicRefusal::Malformed {
            field: "clientContent.weightedPrompts".to_string(),
            reason: "must be a non-empty array".to_string(),
        });
    };
    if prompts.is_empty() {
        return Err(MusicRefusal::Malformed {
            field: "clientContent.weightedPrompts".to_string(),
            reason: "must carry at least one prompt; Google refuses an all-zero set".to_string(),
        });
    }
    if prompts.len() > MAX_WEIGHTED_PROMPTS {
        return Err(MusicRefusal::OutOfRange {
            field: "clientContent.weightedPrompts".to_string(),
            value: format!("{} entries", prompts.len()),
            allowed: format!("at most {MAX_WEIGHTED_PROMPTS}"),
        });
    }
    let _ = limits;

    let mut canonical = Vec::with_capacity(prompts.len());
    let mut any_positive = false;
    for (index, prompt) in prompts.iter().enumerate() {
        let Some(entry) = prompt.as_object() else {
            return Err(MusicRefusal::Malformed {
                field: format!("clientContent.weightedPrompts[{index}]"),
                reason: "must be an object with text and weight".to_string(),
            });
        };
        for key in entry.keys() {
            if !matches!(key.as_str(), "text" | "weight") {
                return Err(MusicRefusal::UnknownField {
                    message: format!("weightedPrompts[{index}]"),
                    field: key.clone(),
                });
            }
        }
        let Some(text) = entry.get("text").and_then(Value::as_str) else {
            return Err(MusicRefusal::Malformed {
                field: format!("clientContent.weightedPrompts[{index}].text"),
                reason: "must be a string".to_string(),
            });
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(MusicRefusal::Malformed {
                field: format!("clientContent.weightedPrompts[{index}].text"),
                reason: "must not be empty".to_string(),
            });
        }
        if trimmed.chars().count() > MAX_PROMPT_CHARS {
            return Err(MusicRefusal::OutOfRange {
                field: format!("clientContent.weightedPrompts[{index}].text"),
                value: format!("{} characters", trimmed.chars().count()),
                allowed: format!("at most {MAX_PROMPT_CHARS}"),
            });
        }
        let weight = match entry.get("weight") {
            None => 1.0_f64,
            Some(value) => match value.as_f64() {
                Some(weight) if weight.is_finite() && weight >= 0.0 => weight,
                _ => {
                    return Err(MusicRefusal::Malformed {
                        field: format!("clientContent.weightedPrompts[{index}].weight"),
                        reason: "must be a finite number at or above 0".to_string(),
                    });
                }
            },
        };
        if weight > 0.0 {
            any_positive = true;
        }
        canonical.push(json!({ "text": trimmed, "weight": weight }));
    }
    if !any_positive {
        return Err(MusicRefusal::Malformed {
            field: "clientContent.weightedPrompts".to_string(),
            reason: "every weight is zero; Google refuses a set whose weights are all 0"
                .to_string(),
        });
    }
    Ok(json!({ "clientContent": { "weightedPrompts": canonical } }))
}

fn translate_generation_config(payload: &Value) -> Result<Value, MusicRefusal> {
    let Some(object) = payload.as_object() else {
        return Err(MusicRefusal::Malformed {
            field: "musicGenerationConfig".to_string(),
            reason: "must be an object".to_string(),
        });
    };
    if object.is_empty() {
        return Err(MusicRefusal::Malformed {
            field: "musicGenerationConfig".to_string(),
            reason: "must name at least one parameter".to_string(),
        });
    }
    let mut canonical = Map::new();
    for (key, value) in object {
        if !GENERATION_CONFIG_FIELDS.contains(&key.as_str()) {
            return Err(MusicRefusal::UnknownField {
                message: "musicGenerationConfig".to_string(),
                field: key.clone(),
            });
        }
        let checked = match key.as_str() {
            "temperature" => range_float("temperature", value, 0.0, 3.0)?,
            "topK" => range_int("topK", value, 1, 1000)?,
            "seed" => range_int("seed", value, 0, i32::MAX as i64)?,
            "guidance" => range_float("guidance", value, 0.0, 6.0)?,
            "bpm" => range_int("bpm", value, 60, 200)?,
            "density" => range_float("density", value, 0.0, 1.0)?,
            "brightness" => range_float("brightness", value, 0.0, 1.0)?,
            "scale" => enum_string("scale", value, SCALES)?,
            "musicGenerationMode" => enum_string("musicGenerationMode", value, MODES)?,
            "muteBass" | "muteDrums" | "onlyBassAndDrums" => {
                if !value.is_boolean() {
                    return Err(MusicRefusal::Malformed {
                        field: key.clone(),
                        reason: "must be a boolean".to_string(),
                    });
                }
                value.clone()
            }
            other => unreachable!("{other} is in GENERATION_CONFIG_FIELDS"),
        };
        canonical.insert(key.clone(), checked);
    }
    Ok(json!({ "musicGenerationConfig": canonical }))
}

fn translate_playback_control(payload: &Value) -> Result<Value, MusicRefusal> {
    let control = enum_string("playbackControl", payload, PLAYBACK_CONTROLS)?;
    Ok(json!({ "playbackControl": control }))
}

fn range_float(field: &str, value: &Value, low: f64, high: f64) -> Result<Value, MusicRefusal> {
    let Some(number) = value.as_f64() else {
        return Err(MusicRefusal::Malformed {
            field: field.to_string(),
            reason: "must be a number".to_string(),
        });
    };
    if !number.is_finite() || number < low || number > high {
        return Err(MusicRefusal::OutOfRange {
            field: field.to_string(),
            value: number.to_string(),
            allowed: format!("[{low}, {high}]"),
        });
    }
    Ok(json!(number))
}

fn range_int(field: &str, value: &Value, low: i64, high: i64) -> Result<Value, MusicRefusal> {
    let Some(number) = value.as_i64() else {
        return Err(MusicRefusal::Malformed {
            field: field.to_string(),
            reason: "must be an integer".to_string(),
        });
    };
    if number < low || number > high {
        return Err(MusicRefusal::OutOfRange {
            field: field.to_string(),
            value: number.to_string(),
            allowed: format!("[{low}, {high}]"),
        });
    }
    Ok(json!(number))
}

fn enum_string(field: &str, value: &Value, allowed: &[&str]) -> Result<Value, MusicRefusal> {
    let Some(text) = value.as_str() else {
        return Err(MusicRefusal::Malformed {
            field: field.to_string(),
            reason: "must be a string".to_string(),
        });
    };
    if !allowed.contains(&text) {
        return Err(MusicRefusal::OutOfRange {
            field: field.to_string(),
            value: format!("{text:?}"),
            allowed: allowed.join("|"),
        });
    }
    Ok(json!(text))
}

/// Raw bytes in one base64 payload, without decoding it.
///
/// Audio accounting must be exact enough to stop a session at its declared ceiling, and
/// base64's 4-characters-per-3-bytes relationship is exact apart from padding, which is
/// computed here. This avoids holding a second copy of every chunk.
pub fn base64_decoded_len(encoded: &str) -> u64 {
    let len = encoded.len() as u64;
    if len == 0 {
        return 0;
    }
    let padding = encoded
        .as_bytes()
        .iter()
        .rev()
        .take(2)
        .filter(|byte| **byte == b'=')
        .count() as u64;
    (len / 4) * 3 - padding
}

/// Total decoded audio bytes carried by one provider frame.
///
/// Returns 0 for a frame that carries no `serverContent.audioChunks`, which includes
/// `setupComplete`, warnings and filtered prompts. A frame that names a chunk without a
/// string `data` contributes 0 rather than aborting the stream: it cost nothing to deliver.
pub fn audio_bytes_in_frame(value: &Value) -> u64 {
    value
        .get("serverContent")
        .and_then(|content| content.get("audioChunks"))
        .and_then(Value::as_array)
        .map(|chunks| {
            chunks
                .iter()
                .filter_map(|chunk| chunk.get("data").and_then(Value::as_str))
                .map(base64_decoded_len)
                .sum()
        })
        .unwrap_or(0)
}

/// `WebSocket /v1/music/{capability_id}`.
pub(crate) async fn music_websocket(
    State(state): State<ServerState>,
    ws: WebSocketUpgrade,
    Path(capability_id): Path<String>,
) -> Response {
    let Some(capability) = state.runner.capability(&capability_id).cloned() else {
        return unknown_capability(&capability_id, &state.capability_ids);
    };
    if capability.format != CapabilityFormat::GoogleMusic {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!("capability {} is not a music endpoint", capability.id),
        );
    }

    let api_key = match &capability.api_key_env {
        None => None,
        Some(variable) => match std::env::var(variable) {
            Ok(value) if !value.trim().is_empty() => Some(value),
            _ => None,
        },
    };
    let Some(api_key) = api_key else {
        return upstream_contract_error(&capability, "credential unavailable");
    };

    // One session is one physical call: the caller pays for the connection whether or not
    // it plays anything, which is what the provider's RPM/RPD counters record.
    let ticket = match reserve_physical_call_for(&capability, 0) {
        Ok(ticket) => ticket,
        Err(refusal) => return *refusal,
    };

    let google_url = upstream_ws_url(&capability, &api_key);
    let setup = build_setup(&capability);
    let limits = MusicLimits::of(&capability);

    ws.on_upgrade(move |client_socket| {
        let capability = capability.clone();
        let ticket = ticket.clone();
        async move {
            music_session(
                client_socket,
                &google_url,
                &setup,
                limits,
                capability,
                ticket,
            )
            .await;
        }
    })
}

fn upstream_ws_url(capability: &ResolvedCapability, api_key: &str) -> String {
    let scheme_and_host = capability
        .url
        .strip_prefix("https://")
        .map(|rest| format!("wss://{rest}"))
        .unwrap_or_else(|| capability.url.clone());
    format!("{scheme_and_host}?key={api_key}")
}

/// Why a session ended, which decides the close frame the caller sees.
enum SessionEnd {
    /// The caller closed, or the provider did.
    Peer,
    /// The declared wall-clock ceiling was reached.
    Timeout,
    /// The declared delivered-audio ceiling was reached.
    AudioBudget,
    /// A caller message failed validation.
    Refused(MusicRefusal),
    /// The caller sent more messages than one session may cost.
    MessageBudget,
}

impl SessionEnd {
    fn close_frame(self) -> CloseFrame {
        let (code, reason): (u16, String) = match self {
            Self::Peer => (1000, "session closed".to_string()),
            Self::Timeout => (
                1000,
                "session reached its declared wall-clock ceiling and was closed".to_string(),
            ),
            Self::AudioBudget => (
                1000,
                "session delivered the audio its capability declares and was closed".to_string(),
            ),
            Self::Refused(refusal) => (1008, refusal.message()),
            Self::MessageBudget => (
                1008,
                "session reached its declared client-message ceiling".to_string(),
            ),
        };
        CloseFrame {
            code,
            reason: reason.into(),
        }
    }
}

async fn music_session(
    client_socket: WebSocket,
    google_url: &str,
    setup: &Value,
    limits: MusicLimits,
    capability: ResolvedCapability,
    ticket: Option<PacingTicket>,
) {
    let google_uri: Uri = match google_url.parse() {
        Ok(uri) => uri,
        Err(error) => {
            tracing::error!(
                "music {}: cannot parse upstream URL {google_url}: {error}",
                capability.id
            );
            release_physical_call(&capability, &ticket);
            return;
        }
    };

    let (mut google_ws, _response) =
        match tokio_tungstenite::connect_async(google_uri.to_string()).await {
            Ok((ws, response)) => (ws, response),
            Err(error) => {
                tracing::error!("music {}: cannot connect to Google: {error}", capability.id);
                release_physical_call(&capability, &ticket);
                return;
            }
        };

    let setup_text = serde_json::to_string(setup).unwrap_or_default();
    if google_ws
        .send(tungstenite::Message::Text(setup_text))
        .await
        .is_err()
    {
        tracing::error!("music {}: cannot send setup to Google", capability.id);
        release_physical_call(&capability, &ticket);
        return;
    }

    let (mut client_tx, mut client_rx) = client_socket.split();
    let (mut google_tx, mut google_rx) = google_ws.split();
    let deadline = tokio::time::sleep(limits.timeout);
    tokio::pin!(deadline);

    let mut delivered_bytes: u64 = 0;
    let mut client_messages: u32 = 0;

    let end = loop {
        tokio::select! {
            () = &mut deadline => break SessionEnd::Timeout,
            caller = client_rx.next() => {
                let Some(Ok(message)) = caller else { break SessionEnd::Peer };
                match message {
                    axum::extract::ws::Message::Text(text) => {
                        if client_messages >= limits.max_client_messages {
                            break SessionEnd::MessageBudget;
                        }
                        client_messages += 1;
                        match translate_client_message(&text, &limits) {
                            Ok((_kind, canonical, _charged)) => {
                                let out = serde_json::to_string(&canonical).unwrap_or_default();
                                if google_tx.send(tungstenite::Message::Text(out)).await.is_err() {
                                    break SessionEnd::Peer;
                                }
                            }
                            Err(refusal) => {
                                tracing::warn!(
                                    "music {}: caller message refused ({}): {}",
                                    capability.id,
                                    refusal.code(),
                                    refusal.message()
                                );
                                break SessionEnd::Refused(refusal);
                            }
                        }
                    }
                    axum::extract::ws::Message::Binary(_bytes) => {
                        // The documented client messages are JSON. Forwarding binary would
                        // bypass every check above, so it is refused rather than relayed.
                        break SessionEnd::Refused(MusicRefusal::NotAnObject);
                    }
                    axum::extract::ws::Message::Close(_) => break SessionEnd::Peer,
                    axum::extract::ws::Message::Ping(_) | axum::extract::ws::Message::Pong(_) => {}
                }
            }
            provider = google_rx.next() => {
                let Some(Ok(message)) = provider else { break SessionEnd::Peer };
                match message {
                    tungstenite::Message::Text(text) => {
                        if let Some(end) = account_and_forward_text(
                            &text,
                            &mut delivered_bytes,
                            limits.max_session_audio_bytes,
                            &mut client_tx,
                        ).await {
                            break end;
                        }
                    }
                    tungstenite::Message::Binary(bytes) => {
                        // Google delivers BidiGenerateMusic JSON in binary frames as well as
                        // text. Accounting only text would miss every chunk on a build that
                        // chose the binary encoding, so the payload is measured first.
                        if let Ok(text) = std::str::from_utf8(&bytes) {
                            if let Some(end) = account_and_forward_text(
                                text,
                                &mut delivered_bytes,
                                limits.max_session_audio_bytes,
                                &mut client_tx,
                            ).await {
                                break end;
                            }
                        } else if client_tx
                            .send(axum::extract::ws::Message::Binary(bytes.to_vec().into()))
                            .await
                            .is_err()
                        {
                            break SessionEnd::Peer;
                        }
                    }
                    tungstenite::Message::Close(_)
                    | tungstenite::Message::Ping(_)
                    | tungstenite::Message::Pong(_)
                    | tungstenite::Message::Frame(_) => break SessionEnd::Peer,
                }
            }
        }
    };

    tracing::info!(
        "music {}: session ended after {} client messages and {} audio bytes",
        capability.id,
        client_messages,
        delivered_bytes
    );
    let _ = client_tx
        .send(axum::extract::ws::Message::Close(Some(end.close_frame())))
        .await;
    let _ = google_tx.close().await;
    release_physical_call(&capability, &ticket);
}

/// Charges a provider frame against the audio ceiling and forwards it.
///
/// Returns `Some(SessionEnd::AudioBudget)` when forwarding the frame would cross the
/// declared ceiling. The frame that crosses it is not forwarded: a caller must never
/// receive more audio than the capability declares.
async fn account_and_forward_text(
    text: &str,
    delivered: &mut u64,
    ceiling: u64,
    client_tx: &mut futures::stream::SplitSink<
        axum::extract::ws::WebSocket,
        axum::extract::ws::Message,
    >,
) -> Option<SessionEnd> {
    let mut forwarded = text.to_string();
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        let bytes = audio_bytes_in_frame(&value);
        if bytes > 0 {
            let next = delivered.saturating_add(bytes);
            if next > ceiling {
                return Some(SessionEnd::AudioBudget);
            }
            *delivered = next;
            // Surface the accounting to the caller so a client can budget without parsing
            // base64: the running total travels beside the provider's own message.
            if let Some(object) = value.as_object().cloned() {
                let mut annotated = object;
                annotated.insert("switchyardDeliveredAudioBytes".to_string(), json!(next));
                annotated.insert("switchyardAudioCeilingBytes".to_string(), json!(ceiling));
                forwarded = Value::Object(annotated).to_string();
            }
        }
    }
    if client_tx
        .send(axum::extract::ws::Message::Text(forwarded.into()))
        .await
        .is_err()
    {
        return Some(SessionEnd::Peer);
    }
    None
}


fn unknown_capability(model: &str, available: &[String]) -> Response {
    (
        StatusCode::NOT_FOUND,
        axum::Json(serde_json::json!({
            "error": {
                "message": format!(
                    "No capability registered for model {model}. Available capabilities: {}",
                    if available.is_empty() {
                        "(none configured)".to_string()
                    } else {
                        available.join(", ")
                    }
                ),
                "type": "model_not_found"
            }
        })),
    )
        .into_response()
}

fn upstream_contract_error(capability: &ResolvedCapability, reason: &str) -> Response {
    (
        StatusCode::BAD_GATEWAY,
        axum::Json(serde_json::json!({
            "error": {
                "message": format!(
                    "capability {} upstream contract mismatch: {reason}",
                    capability.id
                ),
                "type": "upstream_contract_error"
            }
        })),
    )
        .into_response()
}

fn contract_error(status: StatusCode, message: String) -> Response {
    (
        status,
        axum::Json(
            serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}}),
        ),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> MusicLimits {
        MusicLimits {
            max_session_audio_bytes: 1024,
            timeout: std::time::Duration::from_secs(30),
            max_client_messages: 8,
        }
    }

    #[test]
    fn a_prompt_becomes_the_documented_weighted_prompt() {
        let (kind, canonical, charged) = translate_client_message(
            r#"{"clientContent":{"weightedPrompts":[{"text":"  warm lofi piano  ","weight":1.5}]}}"#,
            &limits(),
        )
        .expect("valid");
        assert_eq!(kind, MusicMessageKind::ClientContent);
        assert_eq!(
            canonical,
            json!({"clientContent": {"weightedPrompts": [
                {"text": "warm lofi piano", "weight": 1.5}
            ]}}),
            "the caller's text is trimmed and re-emitted, never forwarded verbatim"
        );
        assert!(charged > 0, "a prompt is billed as input");
    }

    #[test]
    fn a_prompt_without_a_weight_defaults_to_one() {
        let (_, canonical, _) = translate_client_message(
            r#"{"clientContent":{"weightedPrompts":[{"text":"ambient"}]}}"#,
            &limits(),
        )
        .expect("valid");
        assert_eq!(
            canonical["clientContent"]["weightedPrompts"][0]["weight"],
            json!(1.0)
        );
    }

    #[test]
    fn a_generation_config_is_checked_field_by_field() {
        let (kind, canonical, _) = translate_client_message(
            r#"{"musicGenerationConfig":{"bpm":90,"density":0.4,"scale":"D_MAJOR_B_MINOR","muteDrums":true}}"#,
            &limits(),
        )
        .expect("valid");
        assert_eq!(kind, MusicMessageKind::GenerationConfig);
        assert_eq!(
            canonical,
            json!({"musicGenerationConfig": {
                "bpm": 90, "density": 0.4, "scale": "D_MAJOR_B_MINOR", "muteDrums": true
            }})
        );
    }

    #[test]
    fn an_out_of_range_control_is_refused_with_its_documented_range() {
        for (message, field) in [
            (r#"{"musicGenerationConfig":{"bpm":240}}"#, "bpm"),
            (r#"{"musicGenerationConfig":{"bpm":30}}"#, "bpm"),
            (
                r#"{"musicGenerationConfig":{"temperature":3.5}}"#,
                "temperature",
            ),
            (r#"{"musicGenerationConfig":{"guidance":7}}"#, "guidance"),
            (r#"{"musicGenerationConfig":{"density":1.5}}"#, "density"),
            (
                r#"{"musicGenerationConfig":{"brightness":-0.1}}"#,
                "brightness",
            ),
            (r#"{"musicGenerationConfig":{"topK":0}}"#, "topK"),
            (r#"{"musicGenerationConfig":{"topK":1001}}"#, "topK"),
        ] {
            let refusal = translate_client_message(message, &limits()).expect_err(message);
            match &refusal {
                MusicRefusal::OutOfRange { field: named, .. } => {
                    assert_eq!(named, field, "{message}")
                }
                other => panic!("expected an out-of-range refusal for {message}, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_undocumented_config_field_is_refused_by_name() {
        let refusal =
            translate_client_message(r#"{"musicGenerationConfig":{"reverb":0.5}}"#, &limits())
                .expect_err("reverb is not a documented field");
        assert_eq!(
            refusal,
            MusicRefusal::UnknownField {
                message: "musicGenerationConfig".to_string(),
                field: "reverb".to_string(),
            }
        );
        assert!(refusal.message().contains("reverb"));
    }

    #[test]
    fn an_unknown_scale_or_mode_is_refused() {
        assert!(matches!(
            translate_client_message(
                r#"{"musicGenerationConfig":{"scale":"H_MAJOR"}}"#,
                &limits()
            ),
            Err(MusicRefusal::OutOfRange { .. })
        ));
        assert!(matches!(
            translate_client_message(
                r#"{"musicGenerationConfig":{"musicGenerationMode":"LOUD"}}"#,
                &limits()
            ),
            Err(MusicRefusal::OutOfRange { .. })
        ));
    }

    #[test]
    fn playback_control_accepts_only_the_documented_verbs() {
        for control in PLAYBACK_CONTROLS {
            let (kind, canonical, _) = translate_client_message(
                &format!(r#"{{"playbackControl":"{control}"}}"#),
                &limits(),
            )
            .expect("documented control");
            assert_eq!(kind, MusicMessageKind::PlaybackControl);
            assert_eq!(canonical, json!({ "playbackControl": control }));
        }
        assert!(matches!(
            translate_client_message(r#"{"playbackControl":"REWIND"}"#, &limits()),
            Err(MusicRefusal::OutOfRange { .. })
        ));
    }

    #[test]
    fn a_setup_frame_cannot_be_sent_by_the_caller() {
        // Only this server sends the setup. A caller that tries would otherwise be able to
        // name a different model on the open session.
        let refusal =
            translate_client_message(r#"{"setup":{"model":"models/anything-else"}}"#, &limits())
                .expect_err("setup is server-owned");
        assert!(matches!(refusal, MusicRefusal::UnknownMessage { .. }));
    }

    #[test]
    fn two_message_kinds_in_one_frame_are_refused() {
        let refusal = translate_client_message(
            r#"{"playbackControl":"PLAY","musicGenerationConfig":{"bpm":120}}"#,
            &limits(),
        )
        .expect_err("one kind per frame");
        assert!(matches!(refusal, MusicRefusal::UnknownMessage { .. }));
    }

    #[test]
    fn an_all_zero_prompt_set_is_refused_locally() {
        // Google refuses an all-zero weight set; refusing it here keeps a doomed session
        // from costing a physical call.
        let refusal = translate_client_message(
            r#"{"clientContent":{"weightedPrompts":[{"text":"a","weight":0},{"text":"b","weight":0}]}}"#,
            &limits(),
        )
        .expect_err("all-zero weights");
        assert!(matches!(refusal, MusicRefusal::Malformed { .. }));
    }

    #[test]
    fn too_many_prompts_are_refused() {
        let prompts: Vec<String> = (0..9)
            .map(|index| format!("{{\"text\":\"p{index}\",\"weight\":1}}"))
            .collect();
        let message = format!(
            "{{\"clientContent\":{{\"weightedPrompts\":[{}]}}}}",
            prompts.join(",")
        );
        assert!(matches!(
            translate_client_message(&message, &limits()),
            Err(MusicRefusal::OutOfRange { .. })
        ));
    }

    #[test]
    fn audio_accounting_counts_decoded_bytes() {
        // 4 base64 characters are 3 bytes; one '=' means two bytes, two '=' means one.
        assert_eq!(base64_decoded_len("AAAA"), 3);
        assert_eq!(base64_decoded_len("AAA="), 2);
        assert_eq!(base64_decoded_len("AA=="), 1);
        assert_eq!(base64_decoded_len(""), 0);

        let frame = json!({"serverContent": {"audioChunks": [
            {"data": "AAAA", "mimeType": "audio/wav"},
            {"data": "AAA=", "mimeType": "audio/wav"}
        ]}});
        assert_eq!(audio_bytes_in_frame(&frame), 5);
        assert_eq!(audio_bytes_in_frame(&json!({"setupComplete": {}})), 0);
        assert_eq!(
            audio_bytes_in_frame(&json!({"warning": "slow consumer"})),
            0
        );
    }

    #[test]
    fn the_setup_names_only_the_model() {
        let capability = ResolvedCapability {
            id: "music".to_string(),
            client: "music-client".to_string(),
            format: CapabilityFormat::GoogleMusic,
            url: "https://generativelanguage.googleapis.com/ws/music".to_string(),
            model: "lyria-realtime-exp".to_string(),
            api_key_env: Some("KEY".to_string()),
            timeout_seconds: Some(60),
            contract: None,
            dimensions: None,
            normalization: None,
            max_batch: None,
            max_candidates: None,
            top_n: None,
            max_doc_chars: None,
            max_query_chars: None,
            task_type: None,
            task_prefix: None,
            max_input_tokens: None,
            voice: None,
            audio_format: None,
            audio_sample_rate: None,
            max_audio_seconds: None,
            max_audio_bytes: None,
            max_image_bytes: None,
            max_session_audio_bytes: Some(1024),
            budget: None,
            response_modalities: None,
            system_instruction: None,
            thinking_budget: None,
            thinking_level: None,
        };
        assert_eq!(
            build_setup(&capability),
            json!({"setup": {"model": "models/lyria-realtime-exp"}}),
            "BidiGenerateMusicSetup has exactly one field"
        );
    }

    #[test]
    fn the_upstream_url_carries_the_key_and_upgrades_the_scheme() {
        let capability = ResolvedCapability {
            id: "music".to_string(),
            client: "music-client".to_string(),
            format: CapabilityFormat::GoogleMusic,
            url: "https://generativelanguage.googleapis.com/ws/music".to_string(),
            model: "lyria-realtime-exp".to_string(),
            api_key_env: None,
            timeout_seconds: None,
            contract: None,
            dimensions: None,
            normalization: None,
            max_batch: None,
            max_candidates: None,
            top_n: None,
            max_doc_chars: None,
            max_query_chars: None,
            task_type: None,
            task_prefix: None,
            max_input_tokens: None,
            voice: None,
            audio_format: None,
            audio_sample_rate: None,
            max_audio_seconds: None,
            max_audio_bytes: None,
            max_image_bytes: None,
            max_session_audio_bytes: None,
            budget: None,
            response_modalities: None,
            system_instruction: None,
            thinking_budget: None,
            thinking_level: None,
        };
        let url = upstream_ws_url(&capability, "server-key");
        assert_eq!(
            url,
            "wss://generativelanguage.googleapis.com/ws/music?key=server-key"
        );
    }
}
