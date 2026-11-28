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

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use switchyard_protocol::{ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest};

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

impl CompatibilityDecisionRequest {
    /// Parses a compatibility body from raw JSON, so a caller may build the native
    /// request through the same conversion the serving path uses.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        serde_json::from_value(value.clone()).map_err(|error| error.to_string())
    }
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
                            description: if description.is_null() {
                                None
                            } else {
                                Some(description)
                            },
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
    body: std::result::Result<
        Json<CompatibilityDecisionRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> Response {
    let Json(request) = match body {
        Ok(parsed) => parsed,
        Err(error) => return error_body(StatusCode::BAD_REQUEST, error.body_text()),
    };

    let native = match to_native(request) {
        Ok(native) => native,
        Err(message) => return error_body(StatusCode::BAD_REQUEST, message),
    };

    let requested = native.model.clone().unwrap_or_default();

    // An EXPLICIT lane name is authoritative and is resolved before any question-kind
    // selection. Requesting a family or a tier runs exactly that lane's policy; kind
    // eligibility is then applied WITHIN it, so two policy lanes that both support a
    // kind can never silently take the request from one another.
    let explicit_lane = state.runner.named_decision_lane(&requested);

    // An EXPLICIT lane name is served by `serve_named`, the same function the internal
    // `Call::Decision` handler uses. One mechanism, so the public surface and the
    // internal path cannot disagree about what a caller named.
    let handles: BTreeMap<String, switchyard_runner::decision_executor::LaneResolverHandle> =
        state.runner.decision_lane_handles().into_iter().collect();
    let composition = state.runner.decision_lane_composition_map().clone();
    match switchyard_runner::decision_executor::serve_named(
        &composition,
        &handles,
        &requested,
        &native,
    )
    .await
    {
        Ok(Some(outcome)) => return finish(Ok(outcome), &requested),
        Ok(None) => {}
        Err(failure) => return refusal(&failure),
    }

    // The lane serving this request's question kinds. Candidate order may differ per
    // type, so a noul request and a choice request can legitimately resolve differently.
    //
    // An UNADDRESSED request is served by the DECLARED default lane when it serves this
    // request's kinds. Selecting by question kind alone would instead pick whichever
    // unscoped lane came first in map order, so the declared default would be honoured
    // only by accident of spelling.
    let declared_default = if explicit_lane.is_none() {
        state
            .runner
            .decision_lane_name()
            .and_then(|name| state.runner.named_decision_lane(name))
            .filter(|lane| switchyard_runner::decision_transport::lane_serves(&lane.types, &native))
    } else {
        None
    };
    let Some(lane) = explicit_lane
        .or(declared_default)
        .or_else(|| state.runner.lane_for_request(&native))
    else {
        return error_body(
            StatusCode::NOT_IMPLEMENTED,
            "this deployment configures no decision lane for this request".to_string(),
        );
    };
    let lane = Arc::clone(&lane.resolver);
    // The lane is addressed by its configured name, or a configured target is served
    // alone; an unknown identity fails closed rather than reaching some other backend.
    // An UNADDRESSED request is served by the DECLARED default lane. `lane_name_for`
    // supplies that name as the identity being resolved, so it must be resolved against
    // the lane it names rather than against the caller's (absent) name; otherwise an
    // unaddressed request could never match its own default and would fail closed.
    let explicit = explicit_lane.is_some();
    let lane_name = lane_name_for(&state, &native, explicit);
    let identity_to_resolve = lane_name.clone().unwrap_or_else(|| requested.to_string());
    let resolved = switchyard_runner::decision_executor::resolve_identity(
        &lane,
        lane_name.as_deref(),
        &identity_to_resolve,
    );
    let outcome = match resolved {
        Err(failure) => return refusal(&failure),
        Ok(switchyard_runner::decision_executor::DecisionIdentity::Target(name)) => {
            let Some(candidate) = lane
                .candidates
                .iter()
                .find(|candidate| candidate.name == name || candidate.model == name)
            else {
                return error_body(StatusCode::NOT_FOUND, format!("no decision target {name}"));
            };
            switchyard_runner::decision_executor::serve_target(candidate, &native).await
        }
        Ok(switchyard_runner::decision_executor::DecisionIdentity::Lane) => {
            switchyard_runner::decision_executor::resolve_and_serve(&lane, &native).await
        }
    };
    finish(outcome, &requested)
}

/// Renders a decision outcome, naming both the lane that was asked for and the child
/// policy lane that served it, so a tier trace is not reduced to one backend name.
fn finish(
    outcome: Result<
        switchyard_runner::decision_executor::DecisionOutcome,
        switchyard_runner::decision_executor::DecisionFailure,
    >,
    requested: &str,
) -> Response {
    match outcome {
        Ok(outcome) => {
            let mut payload =
                switchyard_runner::decision_executor::compatibility_response(&outcome.response);
            // Name the backend that answered, so a caller can see which lane position
            // served it without inspecting logs.
            payload["backend"] = json!(outcome.served_by);
            if let Some(child) = outcome.served_by_lane {
                payload["child_lane"] = json!(child);
            }
            if !requested.is_empty() {
                payload["requested"] = json!(requested);
            }
            (StatusCode::OK, Json(payload)).into_response()
        }
        Err(failure) => refusal(&failure),
    }
}

/// The explicit name a caller may use to address the lane serving this request.
///
/// When the caller named a lane explicitly, that name IS the addressed identity and no
/// default lane is substituted for it. Otherwise the deployment's default lane name is
/// offered, so an unaddressed request still resolves.
fn lane_name_for(
    state: &ServerState,
    request: &switchyard_protocol::DecisionRequest,
    explicit: bool,
) -> Option<String> {
    if explicit {
        return request.model.as_ref().map(|model| model.to_string());
    }
    state.runner.decision_lane_name().map(str::to_string)
}

/// Maps a classified decision failure to the status a caller should see.
///
/// The three classes are kept apart deliberately:
///
/// * a refusal of the REQUEST (state shape, question kind, wire bounds, unknown identity)
///   is the caller's to fix and is never retried;
/// * a refusal of our own BUDGET is a rate limit: the lane is working as configured and the
///   same request may succeed after the window refills or the day resets;
/// * anything else that was actually sent and failed is an availability answer.
///
/// A deployment fault - no credential, an unusable counter - is ours, and is reported as a
/// server error so it cannot be mistaken for caller misuse.
fn status_for(failure: &switchyard_runner::decision_executor::DecisionFailure) -> StatusCode {
    use switchyard_runner::decision_executor::{DecisionFailure, DecisionSkip};
    match failure {
        // `Refused` is the lane-scoped form of a candidate eligibility refusal: nothing was
        // dispatched, so it is never an availability answer.
        DecisionFailure::NotEligible(DecisionSkip::TargetNotFound)
        | DecisionFailure::Refused(DecisionSkip::TargetNotFound) => StatusCode::NOT_FOUND,
        DecisionFailure::NotEligible(DecisionSkip::Unauthenticated)
        | DecisionFailure::Refused(DecisionSkip::Unauthenticated) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        DecisionFailure::NotEligible(_) | DecisionFailure::Refused(_) => StatusCode::BAD_REQUEST,
        // Budget refusals. `daily_allowance_unavailable` is excluded on purpose: a counter
        // that cannot be read is a deployment fault, not a quota the caller spent.
        DecisionFailure::Serving { reason, .. }
            if matches!(
                *reason,
                "minute_allowance_throttled"
                    | "input_exceeds_minute_window"
                    | "daily_allowance_exhausted"
            ) =>
        {
            StatusCode::TOO_MANY_REQUESTS
        }
        DecisionFailure::Serving { .. } => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Renders a decision failure as the response a caller should act on.
///
/// One place decides both the status and the words, so a caller mistake can never be
/// reported as an outage (or the reverse) by one call site drifting from another.
fn refusal(failure: &switchyard_runner::decision_executor::DecisionFailure) -> Response {
    error_body(status_for(failure), failure.message())
}

/// The error class a caller reads in the body's `type`.
fn error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::NOT_FOUND => "model_not_found",
        status if status.is_server_error() => "api_error",
        _ => "invalid_request_error",
    }
}

fn error_body(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({"error": message, "type": error_type(status)})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_runner::decision_executor::{DecisionFailure, DecisionSkip};

    /// The status a caller sees is decided by the CLASS of the failure, never by which call
    /// site reported it.
    ///
    /// These are the four answers a decision caller can act on differently: fix the request
    /// (400), fix the name (404), wait (429), or treat it as ours (500/503). A regression
    /// here is a caller retrying a request that can never succeed, or a monitoring system
    /// treating a spent quota as an outage.
    #[test]
    fn every_failure_class_maps_to_the_answer_the_caller_can_act_on() {
        use DecisionFailure as F;
        use DecisionSkip as S;

        for (failure, status, class) in [
            (
                F::Refused(S::UnsupportedState),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                F::Refused(S::UnsupportedType),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                F::NotEligible(S::Contract),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                F::Refused(S::TargetNotFound),
                StatusCode::NOT_FOUND,
                "model_not_found",
            ),
            (
                F::NotEligible(S::Unauthenticated),
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
            ),
            (
                F::Serving {
                    reason: "minute_allowance_throttled",
                    retryable: true,
                },
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
            (
                F::Serving {
                    reason: "input_exceeds_minute_window",
                    retryable: true,
                },
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
            (
                F::Serving {
                    reason: "daily_allowance_exhausted",
                    retryable: true,
                },
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
            // Ours, not the caller's: a counter that cannot be read is not a quota spent.
            (
                F::Serving {
                    reason: "daily_allowance_unavailable",
                    retryable: false,
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "api_error",
            ),
            (
                F::Serving {
                    reason: "connect",
                    retryable: true,
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "api_error",
            ),
            (
                F::Serving {
                    reason: "decodable_response",
                    retryable: false,
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "api_error",
            ),
        ] {
            assert_eq!(status_for(&failure), status, "{failure:?}");
            assert_eq!(error_type(status), class, "{failure:?}");
        }
    }

    /// A quota refusal must not read like a caller error, and a caller error must not read
    /// like a quota refusal: the two bodies are the two remedies.
    #[test]
    fn the_body_carries_the_remedy_not_our_vocabulary() {
        let refused = refusal(
            &switchyard_runner::decision_executor::DecisionFailure::Refused(
                DecisionSkip::UnsupportedState,
            ),
        );
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);

        let throttled = refusal(
            &switchyard_runner::decision_executor::DecisionFailure::Serving {
                reason: "daily_allowance_exhausted",
                retryable: true,
            },
        );
        assert_eq!(throttled.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
