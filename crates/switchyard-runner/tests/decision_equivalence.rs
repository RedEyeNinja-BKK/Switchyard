// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Switchyard Decision path must mean the same thing as a direct canonical call.
//!
//! These tests drive the real resolver and executor against a mock decision backend, so
//! what is compared is the Switchyard-encoded request and the natively decoded answer
//! rather than two copies of one function's output.

use serde_json::{Value, json};
use switchyard_protocol::{ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, DecisionValue};
use switchyard_runner::decision_executor::{
    DecisionCandidate, DecisionResolver, resolve_and_serve,
};
use switchyard_runner::decision_transport::{
    DecisionTransport, StateForms, SupportedTypes, parse_response,
};
use std::collections::BTreeMap;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn noul_request(state: Value) -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "risk".to_string(),
        DecisionQuestion {
            instructions: json!("Does this need deep reasoning?"),
            kind: DecisionKind::Boolean {
                true_description: Some(json!("yes")),
                false_description: Some(json!("no")),
            },
        },
    );
    DecisionRequest {
        model: None,
        context: state,
        questions,
    }
}

fn choice_request() -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "team".to_string(),
        DecisionQuestion {
            instructions: json!("Which team owns this?"),
            kind: DecisionKind::Choice {
                options: vec![
                    ChoiceOption {
                        id: "billing".into(),
                        description: None,
                    },
                    ChoiceOption {
                        id: "technical".into(),
                        description: None,
                    },
                ],
            },
        },
    );
    DecisionRequest {
        model: None,
        context: json!("a plain state"),
        questions,
    }
}

fn score_request() -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "severity".to_string(),
        DecisionQuestion {
            instructions: json!("Rate it"),
            kind: DecisionKind::Score {
                levels: vec![json!("low"), json!("medium"), json!("high")],
            },
        },
    );
    DecisionRequest {
        model: None,
        context: json!("a plain state"),
        questions,
    }
}

fn local_backend(url: String) -> DecisionCandidate {
    DecisionCandidate {
        name: "laya".into(),
        model: "laya-rl-agent".into(),
        transport: DecisionTransport::SystemOne,
        url: format!("{url}/v1/systemone"),
        api_key: Some("key".into()),
        supported_types: SupportedTypes::ALL,
        state_forms: StateForms::ALL,
    }
}

/// A SystemOne backend must receive the canonical shape: `state`, a `questions` object
/// keyed by id, and `type` as the discriminator. Asserted on the outbound body, because
/// a decoder that is correct against a wrongly encoded request still proves nothing.
#[tokio::test]
async fn switchyard_sends_the_canonical_request_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(body_partial_json(json!({
            "model": "laya-rl-agent",
            "state": "a plain state",
            "questions": {
                "team": {
                    "type": "choice",
                    "criteria": {"billing": null, "technical": null}
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6, "technical": 0.4}}}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let resolver = DecisionResolver {
        candidates: vec![local_backend(server.uri())],
    };
    resolve_and_serve(&resolver, &choice_request())
        .await
        .expect("request is served");
}

/// The decoded answer must equal the provider's own answer for the same request,
/// including the full probability distribution and provider confidence.
#[tokio::test]
async fn the_decoded_choice_equals_the_provider_answer() {
    let provider_answer = json!({
        "answers": {
            "team": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.3781819865330762,
                "probabilities": {"billing": 0.3109090067334619, "technical": 0.6890909932665381}
            }
        }
    });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(provider_answer.clone()))
        .mount(&server)
        .await;

    let outcome = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![local_backend(server.uri())],
        },
        &choice_request(),
    )
    .await
    .expect("request is served");

    // The direct reading of the same provider body.
    let direct = parse_response(&provider_answer).expect("provider body decodes");

    let DecisionValue::Choice {
        selected,
        probabilities,
    } = &outcome.response.answers["team"].value
    else {
        panic!("choice decodes as a choice");
    };
    let direct_choice = match &direct.answers["team"].value {
        DecisionValue::Choice { selected, probabilities } => (selected, probabilities),
        other => panic!("direct decode disagrees: {other:?}"),
    };
    assert_eq!(selected.as_str(), direct_choice.0.as_str(), "same selected option");
    assert_eq!(
        probabilities.as_ref().map(|p| p["technical"].0),
        direct_choice.1.as_ref().map(|p| p["technical"].0),
        "same probabilities"
    );
    assert_eq!(
        outcome.response.answers["team"].provider_confidence.map(|c| c.0),
        direct.answers["team"].provider_confidence.map(|c| c.0),
        "same provider confidence"
    );
}

#[tokio::test]
async fn a_noul_decodes_to_a_probability_rather_than_a_threshold() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.517510208202746}}
        })))
        .mount(&server)
        .await;

    let outcome = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![local_backend(server.uri())],
        },
        &noul_request(json!("state")),
    )
    .await
    .expect("request is served");
    match &outcome.response.answers["risk"].value {
        DecisionValue::Boolean(estimate) => {
            let switchyard_protocol::BooleanEstimate::ProbabilityTrue(probability) = estimate
            else {
                panic!("noul carries a probability, not a boolean");
            };
            assert!((probability.0 - 0.517510208202746).abs() < 1e-12);
        }
        other => panic!("noul decodes as a boolean: {other:?}"),
    }
}

#[tokio::test]
async fn a_score_keeps_its_estimate_and_ordered_distribution() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {
                "severity": {
                    "type": "score",
                    "score": 1.25,
                    "confidence": 0.6,
                    "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
                    "legend": {"0": "low", "1": "medium", "2": "high"}
                }
            }
        })))
        .mount(&server)
        .await;

    let outcome = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![local_backend(server.uri())],
        },
        &score_request(),
    )
    .await
    .expect("request is served");
    let DecisionValue::Score {
        value,
        probabilities,
    } = &outcome.response.answers["severity"].value
    else {
        panic!("score decodes as a score");
    };
    assert!((value.0 - 1.25).abs() < f64::EPSILON);
    assert_eq!(probabilities.as_ref().map(|p| p.len()), Some(3));
}

/// A serving failure moves to the next eligible backend; a rejection of our own request
/// does not. Both are observable end to end.
#[tokio::test]
async fn a_transport_failure_falls_through_but_a_rejection_does_not() {
    // First backend is unreachable, second answers.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.4}}
        })))
        .mount(&server)
        .await;

    let unreachable = DecisionCandidate {
        name: "down".into(),
        url: "http://127.0.0.1:1/v1/systemone".into(),
        ..local_backend("unused".into())
    };
    let healthy = local_backend(server.uri());
    let outcome = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![unreachable, healthy],
        },
        &noul_request(json!("state")),
    )
    .await
    .expect("the healthy backend answers");
    assert_eq!(outcome.served_by, "laya");
    assert_eq!(outcome.attempts.len(), 1, "the unreachable backend was tried first");
    assert!(
        outcome.attempts[0].1.try_next(),
        "an unreachable backend is a reason to try the next one"
    );

    // A backend that rejects the request must not be retried elsewhere.
    let rejecting = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": "nope"})))
        .mount(&rejecting)
        .await;
    let strict = DecisionCandidate {
        name: "strict".into(),
        url: format!("{}/v1/systemone", rejecting.uri()),
        ..local_backend("unused".into())
    };
    let failure = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![strict, local_backend(server.uri())],
        },
        &noul_request(json!("state")),
    )
    .await
    .expect_err("a rejected request is not served by another backend");
    assert!(
        !failure.try_next(),
        "our request being rejected is not evidence about the next backend"
    );
}

/// SPAN's state limitation is enforced before dispatch, so the request never leaves.
#[tokio::test]
async fn an_ineligible_state_shape_is_never_sent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(0)
        .mount(&server)
        .await;

    let span = DecisionCandidate {
        name: "span".into(),
        transport: DecisionTransport::OpenRouterAlpha,
        url: format!("{}/api/alpha/decisions", server.uri()),
        supported_types: SupportedTypes::NOUL_ONLY,
        state_forms: StateForms::SPAN_OBSERVED,
        ..local_backend("unused".into())
    };
    let resolver = DecisionResolver {
        candidates: vec![span],
    };
    assert_eq!(
        resolver.eligible(&noul_request(json!("plain"))).len(),
        1,
        "a plain state is eligible"
    );
    assert_eq!(
        resolver.eligible(&noul_request(json!({"task": "x"}))).len(),
        0,
        "a structured state is not"
    );
    let failure = resolve_and_serve(&resolver, &noul_request(json!({"task": "x"})))
        .await
        .expect_err("nothing eligible can serve it");
    assert!(!failure.try_next());
}