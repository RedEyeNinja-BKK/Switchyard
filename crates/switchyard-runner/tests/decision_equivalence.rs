// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Switchyard Decision path must mean the same thing as a direct canonical call.
//!
//! These tests drive the real resolver and executor against a mock decision backend, so
//! what is compared is the Switchyard-encoded request and the natively decoded answer
//! rather than two copies of one function's output.

use serde_json::{Value, json};
use switchyard_protocol::{ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, DecisionValue};
use switchyard_runner::Runner;
use switchyard_runner::decision_executor::{
    DecisionCandidate, DecisionFailure, DecisionIdentity, DecisionResolver, DecisionSkip,
    resolve_and_serve, resolve_identity,
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
        api_key: Some("SUPER_SECRET_DECISION_TEST_KEY".into()),
        api_key_env: Some("TEST_DECISION_KEY".into()),
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
/// A request outside the wire bounds is invalid for every backend on that transport.
/// The executor must report it without dispatching anything, rather than sending a
/// malformed request to the next candidate and hoping it behaves differently.
#[tokio::test]
async fn an_out_of_bounds_request_dispatches_no_backend() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(0)
        .mount(&server)
        .await;

    let backend = |name: &str| DecisionCandidate {
        name: name.into(),
        url: format!("{}/v1/systemone", server.uri()),
        ..local_backend("unused".into())
    };
    let resolver = DecisionResolver {
        candidates: vec![backend("a"), backend("b")],
    };

    // Asserted on the failure kind itself, not only that it fails: a contract violation
    // that merely "fails eventually" would still have dispatched the other backend.
    let too_many_options = wide_choice_request(256);
    let failure = resolve_and_serve(&resolver, &too_many_options)
        .await
        .expect_err("256 options is a contract violation");
    assert_eq!(
        failure,
        DecisionFailure::NotEligible(DecisionSkip::Contract),
        "256 options is a contract violation, not an availability signal"
    );

    let too_many_levels = wide_score_request(11);
    let failure = resolve_and_serve(&resolver, &too_many_levels)
        .await
        .expect_err("11 levels is a contract violation");
    assert_eq!(
        failure,
        DecisionFailure::NotEligible(DecisionSkip::Contract)
    );

    // A request that is merely ineligible for one backend still reaches the next.
    let answerer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "technical": 0.0}}}
        })))
        .mount(&answerer)
        .await;
    let eligible = DecisionResolver {
        candidates: vec![
            DecisionCandidate {
                name: "span".into(),
                supported_types: SupportedTypes::NOUL_ONLY,
                ..local_backend(server.uri())
            },
            local_backend(answerer.uri()),
        ],
    };
    let outcome = resolve_and_serve(&eligible, &choice_request())
        .await
        .expect("the full-type backend answers the choice request");
    assert_eq!(outcome.served_by, "laya");
    assert_eq!(outcome.attempts.len(), 1, "only the noul-only backend was skipped");
}

fn wide_choice_request(options: usize) -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "wide".to_string(),
        DecisionQuestion {
            instructions: json!("Pick one"),
            kind: DecisionKind::Choice {
                options: (0..options)
                    .map(|index| ChoiceOption {
                        id: format!("o{index}"),
                        description: None,
                    })
                    .collect(),
            },
        },
    );
    DecisionRequest {
        model: None,
        context: json!("a plain state"),
        questions,
    }
}

fn wide_score_request(levels: usize) -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "deep".to_string(),
        DecisionQuestion {
            instructions: json!("Rate it"),
            kind: DecisionKind::Score {
                levels: (0..levels).map(|index| json!(format!("level-{index}"))).collect(),
            },
        },
    );
    DecisionRequest {
        model: None,
        context: json!("a plain state"),
        questions,
    }
}

/// No representation of a candidate may reveal its credential.
#[test]
fn a_candidate_never_renders_its_credential() {
    let backend = local_backend("https://example.test".into());
    let rendered = format!("{backend:?}");
    assert!(
        !rendered.contains("SUPER_SECRET_DECISION_TEST_KEY"),
        "Debug leaked the credential: {rendered}"
    );
    assert!(rendered.contains("credential_present: true"), "{rendered}");
    assert!(rendered.contains("TEST_DECISION_KEY"), "{rendered}");
}

/// A missing credential is reported, not hidden behind another backend answering, and
/// the failure text must not carry the value.
#[tokio::test]
async fn a_credential_failure_is_terminal_and_silent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.4}}
        })))
        .mount(&server)
        .await;
    let healthy = local_backend(server.uri());
    let nameless = DecisionCandidate {
        name: "no-credential".into(),
        api_key: None,
        ..local_backend("unused".into())
    };
    let failure = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![nameless, healthy],
        },
        &noul_request(json!("state")),
    )
    .await
    .expect_err("a missing credential is a deployment fault, not a reason to move on");
    assert!(!failure.try_next());
    let rendered = format!("{failure:?}");
    assert!(!rendered.contains("SUPER_SECRET_DECISION_TEST_KEY"), "{rendered}");
}

/// An upstream HTTP error may echo the request; whatever it returns, the rendered
/// failure must not carry our credential.
#[tokio::test]
async fn an_upstream_error_does_not_expose_the_credential() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": "upstream failed"
        })))
        .mount(&server)
        .await;
    let backend = local_backend(server.uri());
    let failure = resolve_and_serve(
        &DecisionResolver {
            candidates: vec![backend],
        },
        &noul_request(json!("state")),
    )
    .await
    .expect_err("a 500 with no healthy backend is a failure");
    let rendered = format!("{failure:?}");
    assert!(!rendered.contains("SUPER_SECRET_DECISION_TEST_KEY"), "{rendered}");
}

/// An unknown identity must fail closed: nothing is dispatched, and the failure is
/// terminal rather than falling through to whichever backend exists.
#[tokio::test]
async fn an_unknown_decision_identity_dispatches_nothing() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(0)
        .mount(&server)
        .await;

    let resolver = DecisionResolver {
        candidates: vec![local_backend(server.uri())],
    };
    let failure = resolve_identity(&resolver, Some("default"), "decision-does-not-exist")
        .expect_err("an unknown identity is not a lane");
    assert_eq!(
        failure,
        DecisionFailure::NotEligible(DecisionSkip::TargetNotFound)
    );
    assert!(!failure.try_next());

    // The lane is reachable only by its explicit name.
    assert_eq!(
        resolve_identity(&resolver, Some("default"), "default"),
        Ok(DecisionIdentity::Lane)
    );
    assert_eq!(
        resolve_identity(&resolver, Some("default"), "laya"),
        Ok(DecisionIdentity::Target("laya"))
    );
}

/// Explicit lane order must survive parsing and reach the resolver unchanged. A lane
/// whose order merely happened to match sorted target names would still pass a test
/// that only checked membership, so the sequence itself is asserted.
#[tokio::test]
async fn lane_order_is_explicit_and_preserved() {
    let local = MockServer::start().await;
    let remote = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/alpha/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.2}}
        })))
        .expect(1)
        .mount(&remote)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.9}}
        })))
        .expect(0)
        .mount(&local)
        .await;

    // Declared remote-first, even though "a-local" sorts before "z-span".
    let config = format!(
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

[targets.a-local]
id = "laya-rl-agent"
llm_client = "responses"
decision_transport = "system_one"
decision_path = "/v1/systemone"
decision_base_url = "{local}"
decision_api_key_env = "TEST_DECISION_KEY"

[targets.z-span]
id = "respan/span-01-lite"
llm_client = "responses"
decision_transport = "open_router_alpha"
decision_path = "/api/alpha/decisions"
decision_base_url = "{remote}"
decision_api_key_env = "TEST_DECISION_KEY"
supported_types = {{ noul = true, choice = false, score = false }}
state_forms = {{ plain_string = true, structured = false }}

[decision_lanes.default]
targets = ["z-span", "a-local"]
"#,
        local = local.uri(),
        remote = remote.uri()
    );
    unsafe { std::env::set_var("TEST_DECISION_KEY", "test-value") };
    let runner = Runner::from_toml(&config).expect("deployment builds");
    let request = noul_request(json!("plain"));
    let lane = runner
        .lane_for_request(&request)
        .expect("a configured lane serves a plain noul request");
    assert_eq!(
        lane.resolver.candidates
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect::<Vec<_>>(),
        vec!["z-span", "a-local"],
        "declared order is the candidate order, not sorted target names"
    );

    // And that order decides which backend answers.
    let outcome = resolve_and_serve(&lane.resolver, &request).await.expect("served");
    assert_eq!(outcome.served_by, "z-span");
}
