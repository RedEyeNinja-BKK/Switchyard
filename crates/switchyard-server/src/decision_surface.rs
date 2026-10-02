// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/decisions` - the retained LocalClaw typed decision surface.
//!
//! This exists because current consumers use it. Upstream's own `/v1/decision` is a
//! routing preview taking `input_format` and `request`, and is untouched; the singular
//! and plural spellings are deliberately different endpoints and must not converge.
//!
//! The handler owns no semantics. It converts the compatibility body into a native
//! `DecisionRequest`, hands it to the same logical lane the internal decision call path
//! uses, and serializes the native answer back. Eligibility, candidate ordering,
//! fallback, transport and decoding all live in that one executor, so this is a
//! translation and nothing more.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use switchyard_protocol::{ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, ModelId};

use crate::ServerState;

/// The compatibility request body.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityDecisionRequest {
    /// Capability or backend identity, as the caller named it.
    pub model: String,
    /// Decision state. A string, or a structured value.
    pub state: Value,
    /// Questions keyed by id, each with a `type` of noul, choice or score.
    pub questions: BTreeMap<String, CompatibilityQuestion>,
}

/// One compatibility question.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityQuestion {
    /// `noul`, `choice` or `score`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Question instructions, as sent.
    #[serde(default)]
    pub instructions: Value,
    /// Optional criteria: an object for noul and choice, an ordered array for score.
    #[serde(default)]
    pub criteria: Option<Value>,
}

/// Converts a compatibility body into the native request.
///
/// This is the only translation in the path; anything it cannot express is refused here
/// rather than partially understood downstream.
pub fn to_native(request: CompatibilityDecisionRequest) -> Result<DecisionRequest, String> {
    let mut questions = BTreeMap::new();
    for (id, question) in request.questions {
        let kind = match question.kind.as_str() {
            "noul" => DecisionKind::Boolean {
                true_description: question
                    .criteria
                    .as_ref()
                    .and_then(|criteria| criteria.get("true").cloned()),
                false_description: question
                    .criteria
                    .as_ref()
                    .and_then(|criteria| criteria.get("false").cloned()),
            },
            "choice" => {
                let Some(Value::Object(criteria)) = question.criteria else {
                    return Err(format!(
                        "question {id} needs an object of options for a choice question"
                    ));
                };
                DecisionKind::Choice {
                    options: criteria
                        .into_iter()
                        .map(|(option, description)| ChoiceOption {
                            id: option,
                            description: if description.is_null() { None } else { Some(description) },
                        })
                        .collect(),
                }
            }
            "score" => {
                let Some(Value::Array(levels)) = question.criteria else {
                    return Err(format!(
                        "question {id} needs an ordered array of levels for a score question"
                    ));
                };
                DecisionKind::Score { levels }
            }
            other => {
                return Err(format!(
                    "question {id} has unsupported type {other}; expected noul, choice or score"
                ));
            }
        };
        questions.insert(
            id,
            DecisionQuestion {
                instructions: question.instructions,
                kind,
            },
        );
    }
    if questions.is_empty() {
        return Err("questions must not be empty".to_string());
    }
    Ok(DecisionRequest {
        model: Some(switchyard_protocol::ModelId::from(request.model)),
        context: request.state,
        questions,
    })
}

pub(crate) async fn decisions(
    State(state): State<ServerState>,
    body: std::result::Result<Json<CompatibilityDecisionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(parsed) => parsed,
        Err(error) => return error_body(StatusCode::BAD_REQUEST, error.body_text()),
    };

    let native = match to_native(request) {
        Ok(native) => native,
        Err(message) => return error_body(StatusCode::BAD_REQUEST, message),
    };

    let Some(lane) = state.runner.decision_lane().cloned() else {
        return error_body(
            StatusCode::NOT_IMPLEMENTED,
            "this deployment configures no decision backends".to_string(),
        );
    };
    let requested = native.model.clone().unwrap_or_default();
    match switchyard_runner::decision_executor::resolve_and_serve(&lane, &native).await {
        Ok(outcome) => {
            let mut payload = switchyard_runner::decision_executor::compatibility_response(
                &outcome.response,
            );
            // Name the backend that answered, so a caller can see which lane position
            // served it without inspecting logs.
            payload["backend"] = json!(outcome.served_by);
            if !requested.is_empty() {
                payload["requested"] = json!(requested);
            }
            (StatusCode::OK, Json(payload)).into_response()
        }
        Err(failure) => {
            let status = if failure.try_next() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_REQUEST
            };
            error_body(status, format!("decision request failed: {failure:?}"))
        }
    }
}

fn error_body(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({"error": message, "type": "invalid_request_error"})),
    )
        .into_response()
}
