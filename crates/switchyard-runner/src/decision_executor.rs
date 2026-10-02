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

use serde_json::Value;
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
    /// Only eligibility problems are worth trying elsewhere: they are properties of the
    /// backend, and the next one may well accept what this one refused.
    pub fn try_next(self) -> bool {
        // A credential failure is a deployment problem. Silently moving to the next
        // backend would hide it, so it is reported instead of absorbed.
        !matches!(self, Self::Unauthenticated)
    }
}

/// A backend refused or failed to serve the request.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct DecisionCandidate {
    /// Configuration name of the target.
    pub name: String,
    /// Provider-side model or engine id sent on the wire.
    pub model: String,
    /// Wire family.
    pub transport: DecisionTransport,
    /// Full request URL.
    pub url: String,
    /// Credential for this backend.
    pub api_key: Option<String>,
    /// Question kinds this backend currently answers.
    pub supported_types: SupportedTypes,
    /// State shapes this backend accepts.
    pub state_forms: StateForms,
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
            api_key: Some("key".to_string()),
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
            DecisionFailure::NotEligible(DecisionSkip::Contract),
        ] {
            assert!(retryable.try_next(), "{retryable:?} should try the next backend");
        }

        for terminal in [
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
