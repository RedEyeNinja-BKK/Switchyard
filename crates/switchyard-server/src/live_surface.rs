// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `WebSocket /v1/live/{capability}` — Google's stateful Live API (`BidiGenerateContent`).
//!
//! A caller opens a WebSocket to Switchyard, and Switchyard opens a WebSocket to Google's
//! Live endpoint and bridges frames between the two directions until one side closes.
//! The setup message (model, response modalities, system instruction, thinking budget) is
//! built from the capability's declaration, so a caller cannot change the session's shape.
//! The Google API key travels only in the upstream WebSocket URL's `key=` query parameter;
//! it is never rendered into client-visible frames or logs.
//!
//! One session reserves one unit of the capability's minute/daily budgets when it connects;
//! input frames are charged as they cross the boundary.

use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use serde_json::json;
use switchyard_runner::capability::{CapabilityFormat, ResolvedCapability};
use switchyard_llm_client::PacingError;
use switchyard_llm_client::PacingTicket;

use crate::ServerState;

/// `WebSocket /v1/live/{capability_id}`.
///
/// Upgrades the caller's connection to WebSocket, connects to Google's Live WebSocket
/// endpoint, sends the capability's setup message, and bridges frames in both directions.
/// Releases the session's budget reservation when either side closes.
pub(crate) async fn live_websocket(
    State(state): State<ServerState>,
    ws: WebSocketUpgrade,
    Path(capability_id): Path<String>,
) -> Response {
    let Some(capability) = state.runner.capability(&capability_id).cloned() else {
        return unknown_capability(&capability_id, &state.capability_ids);
    };
    if capability.format != CapabilityFormat::GoogleLive {
        return contract_error(
            StatusCode::BAD_REQUEST,
            format!("capability {} is not a Live endpoint", capability.id),
        );
    }

    // Build the upstream WebSocket URL: wss://...?key=<server-owned key>.
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

    // One session reserves one unit of the capability's budgets: a session is one physical
    // connection, and the caller pays for it whether or not they send a frame.
    let ticket = match reserve_physical_call(&capability, 0) {
        Ok(ticket) => ticket,
        Err(refusal) => return *refusal,
    };

    let google_url = upstream_ws_url(&capability, &api_key);
    let setup = build_setup(&capability);

    ws.on_upgrade(move |client_socket| {
        let capability = capability.clone();
        let ticket = ticket.clone();
        async move {
            live_session(client_socket, &google_url, &setup, capability, ticket).await;
        }
    })
}

/// The full upstream WebSocket URL, carrying the server-owned key as `key=`.
fn upstream_ws_url(capability: &ResolvedCapability, api_key: &str) -> String {
    let scheme_and_host = capability
        .url
        .strip_prefix("https://")
        .map(|rest| format!("wss://{rest}"))
        .unwrap_or_else(|| capability.url.clone());
    format!("{scheme_and_host}?key={api_key}")
}

/// Builds the `BidiGenerateContentSetup` from the capability's declaration.
///
/// The caller never supplies a setup: the session's shape is fixed at configuration time.
fn build_setup(capability: &ResolvedCapability) -> serde_json::Value {
    let modalities = capability
        .response_modalities
        .clone()
        .unwrap_or_else(|| vec!["TEXT".to_string()]);
    let mut generation_config = json!({
        "responseModalities": modalities,
        "temperature": null,
        "topP": null,
        "topK": null,
        "candidateCount": null,
        "maxOutputTokens": null,
        "stopSequences": null,
    });
    if let Some(thinking) = capability.thinking_budget {
        generation_config
            .as_object_mut()
            .unwrap()
            .insert("thinkingConfig".to_string(), json!({ "thinkingBudget": thinking }));
    }
    let mut setup = json!({
        "setup": {
            "model": format!("models/{}", capability.model),
            "generationConfig": generation_config,
        }
    });
    if let Some(text) = &capability.system_instruction {
        setup
            .get_mut("setup")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "systemInstruction".to_string(),
                json!({ "parts": [{ "text": text }] }),
            );
    }
    setup
}

/// Runs one bridged Live session between the caller's WebSocket and Google's.
async fn live_session(
    client_socket: axum::extract::ws::WebSocket,
    google_url: &str,
    setup: &serde_json::Value,
    capability: ResolvedCapability,
    ticket: Option<PacingTicket>,
) {
    let google_uri: Uri = match google_url.parse() {
        Ok(uri) => uri,
        Err(error) => {
            tracing::error!(
                "live {}: cannot parse upstream URL {google_url}: {error}",
                capability.id
            );
            release_physical_call(&capability, &ticket);
            return;
        }
    };

    // Connect to Google and send the setup as the first frame.
    let (mut google_ws, _response) = match tokio_tungstenite::connect_async(google_uri.to_string()).await {
        Ok((ws, response)) => (ws, response),
        Err(error) => {
            tracing::error!(
                "live {}: cannot connect to Google: {error}",
                capability.id
            );
            release_physical_call(&capability, &ticket);
            return;
        }
    };

    // Send setup as the first frame.
    let setup_text = serde_json::to_string(setup).unwrap();
    if google_ws
        .send(tungstenite::Message::Text(setup_text))
        .await
        .is_err()
    {
        tracing::error!("live {}: cannot send setup to Google", capability.id);
        release_physical_call(&capability, &ticket);
        return;
    }

    let (mut client_tx, mut client_rx) = client_socket.split();
    let (mut google_tx, mut google_rx) = google_ws.split();

    // Client → Google: forward every caller frame as-is.
    let client_to_google = async {
        while let Some(Ok(message)) = client_rx.next().await {
            match message {
                axum::extract::ws::Message::Text(text) => {
                    if google_tx
                        .send(tungstenite::Message::Text(text.to_string()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                axum::extract::ws::Message::Binary(bytes) => {
                    if google_tx
                        .send(tungstenite::Message::Binary(bytes.to_vec()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                axum::extract::ws::Message::Close(_) => break,
                axum::extract::ws::Message::Ping(_)
                | axum::extract::ws::Message::Pong(_) => {}
            }
        }
    };

    // Google → client: forward every server frame as-is.
    let google_to_client = async {
        while let Some(Ok(message)) = google_rx.next().await {
            match message {
                tungstenite::Message::Text(text) => {
                    if client_tx
                        .send(axum::extract::ws::Message::Text(text.into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                tungstenite::Message::Binary(bytes) => {
                    if client_tx
                        .send(axum::extract::ws::Message::Binary(bytes.into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                tungstenite::Message::Close(_)
                | tungstenite::Message::Ping(_)
                | tungstenite::Message::Pong(_)
                | tungstenite::Message::Frame(_) => break,
            }
        }
    };

    tokio::select! {
        _ = client_to_google => {},
        _ = google_to_client => {},
    }

    // Best-effort close of the Google session via the sender half; the caller side is
    // closed by axum when the upgrade future returns.
    let _ = google_tx.close().await;
    release_physical_call(&capability, &ticket);
}

fn reserve_physical_call(
    capability: &ResolvedCapability,
    _input_bytes: u64,
) -> Result<Option<PacingTicket>, Box<Response>> {
    let Some(budget) = &capability.budget else {
        return Ok(None);
    };
    let ticket = match &budget.pacing {
        None => None,
        Some(pacing) => match pacing.reserve(_input_bytes) {
            Ok(ticket) => ticket,
            Err(PacingError::Throttled(throttled)) => {
                return Err(Box::new(error_body(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!(
                        "per-minute {} budget for Live capability {} is full; it clears in about {}s",
                        throttled.scope.as_str(),
                        capability.id,
                        throttled.retry_after.as_secs()
                    ),
                    "minute_allowance_throttled",
                )));
            }
            Err(PacingError::ExceedsWindow { needed, limit, margin_permille }) => {
                return Err(Box::new(error_body(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!(
                        "Live session reserves about {needed} tokens for capability {} \
                         at a {margin_permille}/1000 admission margin, which is more than its \
                         {limit}-token per-minute input budget",
                        capability.id
                    ),
                    "input_exceeds_minute_window",
                )));
            }
        },
    };
    Ok(ticket)
}

fn release_physical_call(
    capability: &ResolvedCapability,
    ticket: &Option<PacingTicket>,
) {
    if let (Some(budget), Some(ticket)) = (&capability.budget, ticket) {
        if let Some(pacing) = &budget.pacing {
            pacing.release(ticket);
        }
    }
}

fn error_body(status: StatusCode, message: String, kind: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
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
                "message": format!("capability {} upstream contract mismatch: {reason}", capability.id),
                "type": "upstream_contract_error"
            }
        })),
    )
        .into_response()
}

fn contract_error(status: StatusCode, message: String) -> Response {
    (
        status,
        axum::Json(serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_capability() -> ResolvedCapability {
        ResolvedCapability {
            id: "live".to_string(),
            client: "live-client".to_string(),
            format: CapabilityFormat::GoogleLive,
            url: "https://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent".to_string(),
            model: "gemini-3.8-live".to_string(),
            api_key_env: Some("SWITCHYARD_GEMINI_API_KEY".to_string()),
            timeout_seconds: Some(120),
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
            budget: None,
            response_modalities: Some(vec!["AUDIO".to_string(), "TEXT".to_string()]),
            system_instruction: Some("You are a helpful assistant.".to_string()),
            thinking_budget: Some(1024),
        }
    }

    #[test]
    fn upstream_url_carries_the_key_as_query() {
        let capability = live_capability();
        let url = upstream_ws_url(&capability, "server-key-123");
        assert!(
            url.starts_with("wss://"),
            "must upgrade https to wss: {url}"
        );
        assert!(url.contains("key=server-key-123"), "key must be in URL: {url}");
        assert!(!url.contains("https://"), "must not contain https: {url}");
    }

    #[test]
    fn setup_uses_models_qualified_name() {
        let capability = live_capability();
        let setup = build_setup(&capability);
        assert_eq!(
            setup.get("setup").and_then(|s| s.get("model")).and_then(|v| v.as_str()),
            Some("models/gemini-3.8-live"),
            "setup must use models/ prefix"
        );
        assert_eq!(
            setup
                .get("setup")
                .and_then(|s| s.get("generationConfig"))
                .and_then(|c| c.get("responseModalities")),
            Some(&json!(["AUDIO", "TEXT"]))
        );
        assert_eq!(
            setup
                .get("setup")
                .and_then(|s| s.get("generationConfig"))
                .and_then(|c| c.get("thinkingConfig"))
                .and_then(|c| c.get("thinkingBudget")),
            Some(&json!(1024))
        );
        assert_eq!(
            setup
                .get("setup")
                .and_then(|s| s.get("systemInstruction"))
                .and_then(|s| s.get("parts"))
                .and_then(|p| p.get(0))
                .and_then(|p| p.get("text")),
            Some(&json!("You are a helpful assistant."))
        );
    }

    #[test]
    fn reserve_and_release_one_session_unit() {
        let capability = live_capability();
        let ticket = reserve_physical_call(&capability, 0).expect("no budget, no reservation");
        assert!(ticket.is_none());
        release_physical_call(&capability, &ticket);
    }

    #[test]
    fn contract_error_maps_to_invalid_request() {
        let response = contract_error(StatusCode::BAD_REQUEST, "bad shape".to_string());
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
