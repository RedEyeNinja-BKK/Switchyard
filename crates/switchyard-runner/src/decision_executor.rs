// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fulfils a native `Call::Decision` against a configured decision backend.
//!
//! Upstream's driver emits a decision call with one selected model, and the hosting
//! client answered it with "not supported". This supplies that answer: resolve the
//! logical decision identity to its eligible ordered targets, then try them in order.
//!
//! Two things are deliberately kept apart.
//!
//! Eligibility is decided before dispatch. A backend that cannot answer one of the
//! question kinds, or cannot accept the request's state shape, is skipped without a
//! request being sent, so an unsupported type never looks like a serving failure.
//!
//! Failure classification decides fallback, and it is not completion fallback by
//! assumption. A transport error, timeout, rate limit or eligible server error means
//! the next backend is worth trying. A malformed request, an unsupported kind, a
//! contract violation, a credential problem, or a rejection of our own encoding is
//! evidence about this request or this configuration, not evidence that another
//! backend should serve it, so it stops immediately.

use std::time::Duration;

use serde_json::{Value, json};
use switchyard_protocol::{DecisionRequest, DecisionResponse};

use crate::decision_transport::{
    DecisionTransport, StateForms, SupportedTypes, parse_response, validate_systemone,
};

/// Why a decision call could not be served by one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionSkip {
    /// The backend cannot answer one of the question kinds in the request.
    UnsupportedType,
    /// The backend cannot accept the request's state shape.
    UnsupportedState,
    /// The request does not fit the backend's wire contract.
    Contract,
    /// No credential is available for this backend.
    Unauthenticated,
}

impl DecisionSkip {
    /// Whether another backend could still serve the request.
    ///
    /// Only the properties of *this backend* are worth trying elsewhere: an unsupported
    /// question kind or an unsupported state shape is a fact about the backend, and the
    /// next one may well accept what this one refused.
    ///
    /// A contract violation is different. An out-of-bounds request is invalid for every
    /// backend on this transport, so trying the next one would send a request already
    /// known to be malformed. A credential failure is a deployment fault, and moving on
    /// would hide it. Both are terminal.
    pub fn try_next(self) -> bool {
        matches!(self, Self::UnsupportedType | Self::UnsupportedState)
    }
}

/// A backend refused or failed to serve the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionFailure {
    /// Eligible backend that could not be reached or refused transiently.
    Serving { reason: &'static str, retryable: bool },
    /// The request or configuration is at fault; another backend would not help.
    NotEligible(DecisionSkip),
}

impl DecisionFailure {
    /// Whether the resolver should try the next candidate.
    pub fn try_next(&self) -> bool {
        match self {
            DecisionFailure::Serving { retryable, .. } => *retryable,
            DecisionFailure::NotEligible(skip) => skip.try_next(),
        }
    }
}

/// One configured decision backend, resolved for serving.
///
/// `Debug` is written by hand because the candidate holds a credential: a derived
/// implementation would print the key into logs, tracing, receipts and test failures.
/// Only presence and the logical identity are ever shown.
#[derive(Clone)]
pub struct DecisionCandidate {
    /// Configuration name of the target.
    pub name: String,
    /// Provider-side model or engine id sent on the wire.
    pub model: String,
    /// Wire family.
    pub transport: DecisionTransport,
    /// Full request URL.
    pub url: String,
    /// Credential for this backend. Never rendered by `Debug`.
    pub api_key: Option<String>,
    /// Name of the environment variable this credential was read from, for diagnostics.
    pub api_key_env: Option<String>,
    /// Question kinds this backend currently answers.
    pub supported_types: SupportedTypes,
    /// State shapes this backend accepts.
    pub state_forms: StateForms,
}

impl std::fmt::Debug for DecisionCandidate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecisionCandidate")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("transport", &self.transport)
            .field("url", &self.url)
            .field("supported_types", &self.supported_types)
            .field("state_forms", &self.state_forms)
            .field("credential_present", &self.api_key.is_some())
            .field("credential_env", &self.api_key_env)
            .finish()
    }
}

impl DecisionCandidate {
    /// Whether this backend may serve the request, without sending anything.
    ///
    /// Order matters: contract bounds are checked first, because a request outside the
    /// wire bounds is invalid for every backend on that transport and must not be
    /// reported as one backend being unavailable.
    pub fn eligibility(&self, request: &DecisionRequest) -> Result<(), DecisionSkip> {
        if self.transport == DecisionTransport::SystemOne {
            validate_systemone(request).map_err(|_| DecisionSkip::Contract)?;
        }
        if !self.supported_types.accepts_request(request) {
            return Err(DecisionSkip::UnsupportedType);
        }
        if !self.state_forms.accepts(&request.context) {
            return Err(DecisionSkip::UnsupportedState);
        }
        Ok(())
    }
}

/// Selects the eligible ordered candidates for one logical decision identity.
///
/// The smallest resolver that answers one question: which of the configured ordered
/// targets may serve this request. Ordering is the configured order; there is no
/// scoring, no ranking and no second routing system.
#[derive(Default)]
pub struct DecisionResolver {
    /// Configured targets in preference order.
    pub candidates: Vec<DecisionCandidate>,
}

impl DecisionResolver {
    /// Candidates eligible for this request, in configured order.
    pub fn eligible(&self, request: &DecisionRequest) -> Vec<&DecisionCandidate> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.eligibility(request).is_ok())
            .collect()
    }
}

/// Outcome of resolving and serving one decision request.
#[derive(Debug)]
pub struct DecisionOutcome {
    /// The response in upstream's native types.
    pub response: DecisionResponse,
    /// Name of the backend that answered.
    pub served_by: String,
    /// Candidates that were tried and failed, with the reason.
    pub attempts: Vec<(String, DecisionFailure)>,
}

/// Default deadline for one decision call.
const DEFAULT_DECISION_TIMEOUT_SECONDS: u64 = 30;

async fn send(
    candidate: &DecisionCandidate,
    body: Value,
) -> Result<Value, DecisionFailure> {
    let Some(api_key) = candidate.api_key.as_deref().filter(|key| !key.trim().is_empty()) else {
        return Err(DecisionFailure::NotEligible(DecisionSkip::Unauthenticated));
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_DECISION_TIMEOUT_SECONDS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| DecisionFailure::Serving {
            reason: "client_unavailable",
            retryable: false,
        })?;
    let response = client
        .post(&candidate.url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|error| {
            // A transport failure or deadline is exactly the evidence that the next
            // backend should be tried.
            let retryable = error.is_timeout() || error.is_connect() || error.is_request();
            DecisionFailure::Serving {
                reason: if error.is_timeout() {
                    "timeout"
                } else if error.is_connect() {
                    "connect"
                } else {
                    "transport"
                },
                retryable,
            }
        })?;
    let status = response.status();
    let payload = response
        .json::<Value>()
        .await
        .map_err(|_| DecisionFailure::Serving {
            reason: "unreadable_response",
            retryable: false,
        })?;
    if status.is_success() {
        return Ok(payload);
    }
    // A rejection of our own encoding or a credential problem is not a reason to try a
    // different backend, so only the transient classes are retryable.
    let retryable = status.as_u16() == 408
        || status.as_u16() == 429
        || (status.is_server_error() && status.as_u16() != 501);
    let reason: &'static str = if status.as_u16() == 401 || status.as_u16() == 403 {
        "auth"
    } else if status.as_u16() == 400 || status.as_u16() == 422 {
        "rejected"
    } else if status.as_u16() == 429 {
        "rate_limited"
    } else if status.is_server_error() {
        "server_error"
    } else {
        "http"
    };
    Err(DecisionFailure::Serving { reason, retryable })
}

/// Serves one named target, and nothing else.
///
/// A concrete decision call names its model, so it means that backend. Eligibility
/// still applies, but a backend that cannot serve the request produces that result
/// rather than an answer from a different provider: silently substituting one would
/// make a pinned diagnostic call report another backend's opinion.
pub async fn serve_target(
    candidate: &DecisionCandidate,
    request: &DecisionRequest,
) -> Result<DecisionOutcome, DecisionFailure> {
    if let Err(skip) = candidate.eligibility(request) {
        return Err(DecisionFailure::NotEligible(skip));
    }
    let body = candidate
        .transport
        .request_body(&candidate.model, request)
        .map_err(|_| DecisionFailure::NotEligible(DecisionSkip::Contract))?;
    match send(candidate, body).await {
        Ok(payload) => {
            let response = parse_response(&payload).map_err(|_| DecisionFailure::Serving {
                reason: "undecodable_response",
                retryable: false,
            })?;
            Ok(DecisionOutcome {
                response,
                served_by: candidate.name.clone(),
                attempts: Vec::new(),
            })
        }
        Err(failure) => Err(failure),
    }
}

/// Resolves eligible candidates and serves the request through the first that answers.
///
/// The same resolver and executor back both the internal decision call path and any
/// public typed surface, so there is one implementation of Decision semantics.
pub async fn resolve_and_serve(
    resolver: &DecisionResolver,
    request: &DecisionRequest,
) -> Result<DecisionOutcome, DecisionFailure> {
    let mut attempts = Vec::new();
    for candidate in &resolver.candidates {
        if let Err(skip) = candidate.eligibility(request) {
            let failure = DecisionFailure::NotEligible(skip);
            // Terminal before any request is sent: an out-of-bounds request must never
            // reach a backend, and a credential fault must not be hidden behind a
            // different backend answering.
            if !failure.try_next() {
                return Err(failure);
            }
            attempts.push((candidate.name.clone(), failure));
            continue;
        }
        let body = match candidate.transport.request_body(&candidate.model, request) {
            Ok(body) => body,
            Err(_) => {
                let failure = DecisionFailure::NotEligible(DecisionSkip::Contract);
                if !failure.try_next() {
                    return Err(failure);
                }
                attempts.push((candidate.name.clone(), failure));
                continue;
            }
        };
        match send(candidate, body).await {
            Ok(payload) => {
                let response = parse_response(&payload).map_err(|_| DecisionFailure::Serving {
                    // A schema mismatch we caused must not trigger a different backend:
                    // the next one would receive the same request and answer the same way.
                    reason: "undecodable_response",
                    retryable: false,
                })?;
                return Ok(DecisionOutcome {
                    response,
                    served_by: candidate.name.clone(),
                    attempts,
                });
            }
            Err(failure) => {
                if !failure.try_next() {
                    return Err(failure);
                }
                attempts.push((candidate.name.clone(), failure));
            }
        }
    }
    Err(DecisionFailure::Serving {
        reason: "no_eligible_backend",
        retryable: false,
    })
}

/// Serializes a native decision response in the retained compatibility shape.
///
/// The shape is the canonical one: keyed `answers`, each tagged with its kind and
/// carrying the provider's own value plus probabilities and confidence where they
/// were reported. A caller reading either this or a direct provider response sees the
/// same fields, because they are the same answer.
pub fn compatibility_response(response: &switchyard_protocol::DecisionResponse) -> Value {
    let mut answers = serde_json::Map::new();
    for (id, answer) in &response.answers {
        let mut entry = serde_json::Map::new();
        match &answer.value {
            switchyard_protocol::DecisionValue::Boolean(estimate) => {
                let probability = match estimate {
                    switchyard_protocol::BooleanEstimate::ProbabilityTrue(probability) => {
                        probability.0
                    }
                    switchyard_protocol::BooleanEstimate::Value(value) => {
                        if *value { 1.0 } else { 0.0 }
                    }
                };
                entry.insert("type".to_string(), json!("noul"));
                entry.insert("noul".to_string(), json!(probability));
            }
            switchyard_protocol::DecisionValue::Choice {
                selected,
                probabilities,
            } => {
                entry.insert("type".to_string(), json!("choice"));
                entry.insert("choice".to_string(), json!(selected));
                if let Some(probabilities) = probabilities {
                    let map: serde_json::Map<String, Value> = probabilities
                        .iter()
                        .map(|(name, probability)| (name.clone(), json!(probability.0)))
                        .collect();
                    entry.insert("probabilities".to_string(), Value::Object(map));
                }
            }
            switchyard_protocol::DecisionValue::Score {
                value,
                probabilities,
            } => {
                entry.insert("type".to_string(), json!("score"));
                entry.insert("score".to_string(), json!(value.0));
                if let Some(probabilities) = probabilities {
                    let ordered: Vec<Value> =
                        probabilities.iter().map(|value| json!(value.0)).collect();
                    entry.insert("probabilities".to_string(), Value::Array(ordered));
                }
            }
        }
        if let Some(confidence) = answer.provider_confidence {
            entry.insert("confidence".to_string(), json!(confidence.0));
        }
        answers.insert(id.clone(), Value::Object(entry));
    }
    let mut body = json!({ "answers": Value::Object(answers) });
    if let Some(id) = &response.id {
        body["id"] = json!(id);
    }
    if let Some(model) = &response.model {
        body["model"] = json!(model.as_str());
    }
    let usage = &response.usage;
    if usage.input_tokens.is_some() || usage.output_tokens.is_some() || usage.total_tokens.is_some() {
        body["usage"] = json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        });
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{DecisionKind, DecisionQuestion};
    use std::collections::BTreeMap;

    pub(crate) fn candidate(
        name: &str,
        transport: DecisionTransport,
        types: SupportedTypes,
        state_forms: StateForms,
    ) -> DecisionCandidate {
        DecisionCandidate {
            name: name.to_string(),
            model: "backend-model".to_string(),
            transport,
            url: format!("http://{name}.test{}", transport.path()),
            api_key: Some("SUPER_SECRET_DECISION_TEST_KEY".to_string()),
            api_key_env: Some("TEST_DECISION_KEY".to_string()),
            supported_types: types,
            state_forms,
        }
    }

    pub(crate) fn noul_request(context: Value) -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "risk".to_string(),
            DecisionQuestion {
                instructions: json!("Is this risky?"),
                kind: DecisionKind::Boolean {
                    true_description: None,
                    false_description: None,
                },
            },
        );
        DecisionRequest {
            model: None,
            context,
            questions,
        }
    }

    pub(crate) fn choice_request() -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "team".to_string(),
            DecisionQuestion {
                instructions: json!("Pick"),
                kind: DecisionKind::Choice {
                    options: vec![switchyard_protocol::ChoiceOption {
                        id: "a".into(),
                        description: None,
                    }],
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!("state"),
            questions,
        }
    }

    pub(crate) fn local() -> DecisionCandidate {
        candidate(
            "laya",
            DecisionTransport::SystemOne,
            SupportedTypes::ALL,
            StateForms::ALL,
        )
    }

    /// A local backend serves every kind with either state form, so it is the default
    /// answer for anything a caller can express.
    #[test]
    fn a_local_backend_is_eligible_for_every_supported_shape() {
        let backend = local();
        assert!(backend.eligibility(&noul_request(json!("plain"))).is_ok());
        assert!(backend.eligibility(&noul_request(json!({"task": "x"}))).is_ok());
        assert!(backend.eligibility(&choice_request()).is_ok());
    }

    /// SPAN is noul-only and, as observed on 2026-10-02, takes a plain string state.
    /// Both dimensions must be checked before dispatch.
    #[test]
    fn span_exclusion_is_multi_dimensional() {
        let span = candidate(
            "span",
            DecisionTransport::OpenRouterAlpha,
            SupportedTypes::NOUL_ONLY,
            StateForms::SPAN_OBSERVED,
        );
        assert!(
            span.eligibility(&noul_request(json!("plain text"))).is_ok(),
            "noul with a plain state is eligible"
        );
        assert_eq!(
            span.eligibility(&choice_request()).unwrap_err(),
            DecisionSkip::UnsupportedType
        );
        assert_eq!(
            span.eligibility(&noul_request(json!({"task": "x"})))
                .unwrap_err(),
            DecisionSkip::UnsupportedState,
            "an unsupported state shape excludes the backend before dispatch"
        );
    }

    /// A request outside the wire bounds is a contract error, not a backend being
    /// unavailable, so it is not treated as a reason to try the next one.
    #[test]
    fn a_contract_violation_is_not_an_availability_signal() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "wide".to_string(),
            DecisionQuestion {
                instructions: json!("Pick"),
                kind: DecisionKind::Choice {
                    options: (0..256)
                        .map(|index| switchyard_protocol::ChoiceOption {
                            id: format!("o{index}"),
                            description: None,
                        })
                        .collect(),
                },
            },
        );
        let request = DecisionRequest {
            model: None,
            context: json!("state"),
            questions,
        };
        let failure = local().eligibility(&request).unwrap_err();
        assert_eq!(failure, DecisionSkip::Contract);
    }

    /// The resolver keeps configured order and drops only ineligible candidates.
    #[test]
    fn resolution_preserves_configured_order() {
        let resolver = DecisionResolver {
            candidates: vec![
                candidate(
                    "laya",
                    DecisionTransport::SystemOne,
                    SupportedTypes::ALL,
                    StateForms::ALL,
                ),
                candidate(
                    "span",
                    DecisionTransport::OpenRouterAlpha,
                    SupportedTypes::NOUL_ONLY,
                    StateForms::SPAN_OBSERVED,
                ),
            ],
        };
        let both = resolver.eligible(&noul_request(json!("plain")));
        assert_eq!(
            both.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["laya", "span"],
            "a plain noul is eligible for both, in configured order"
        );
        let choice = resolver.eligible(&choice_request());
        assert_eq!(
            choice.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["laya"],
            "a choice request excludes the noul-only backend entirely"
        );
    }

    /// Classification is the contract: transient classes try the next backend, and
    /// anything about the request or our configuration stops immediately.
    #[test]
    fn failure_classification_decides_whether_to_try_another_backend() {
        for retryable in [
            DecisionFailure::Serving {
                reason: "timeout",
                retryable: true,
            },
            DecisionFailure::Serving {
                reason: "connect",
                retryable: true,
            },
            DecisionFailure::Serving {
                reason: "rate_limited",
                retryable: true,
            },
            DecisionFailure::Serving {
                reason: "server_error",
                retryable: true,
            },
            DecisionFailure::NotEligible(DecisionSkip::UnsupportedType),
            DecisionFailure::NotEligible(DecisionSkip::UnsupportedState),
        ] {
            assert!(retryable.try_next(), "{retryable:?} should try the next backend");
        }

        for terminal in [
            // A contract violation is terminal: the request is invalid for every backend
            // on this transport, so dispatching another would send a malformed request.
            DecisionFailure::NotEligible(DecisionSkip::Contract),
            DecisionFailure::Serving {
                reason: "auth",
                retryable: false,
            },
            DecisionFailure::Serving {
                reason: "rejected",
                retryable: false,
            },
            DecisionFailure::Serving {
                reason: "undecodable_response",
                retryable: false,
            },
            DecisionFailure::NotEligible(DecisionSkip::Unauthenticated),
        ] {
            assert!(
                !terminal.try_next(),
                "{terminal:?} must not silently move to another backend"
            );
        }
    }
}
