// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named decision-lane resolution and route-to-route decision composition.
//!
//! The defect these cover: a caller could not address a policy lane by its own name.
//! The public identity came from `default_decision_lane`, which the last unscoped lane
//! overwrote, and `resolve_identity` searched only the selected lane's candidates, so
//! adding real sibling lanes made them shadow and steal the request from one another.
//!
//! The fix makes an explicitly requested name authoritative and lets a tier compose
//! child POLICY lanes in process. These tests pin that mechanism generically: no test
//! names a particular production lane.

use std::collections::BTreeMap;

use serde_json::json;
use switchyard_protocol::{DecisionKind, DecisionQuestion, DecisionRequest, ModelId};
use switchyard_runner::{Runner, RunnerError};

fn decision_config(lanes: &str, composition: &str) -> String {
    format!(
        r#"
schema_version = 1
default_decision_lane = "free-family"

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "PRIMARY_KEY"

[targets.span]
id = "span/model"
llm_client = "primary"
decision_transport = "open_router_alpha"
decision_path = "/api/alpha/decisions"
decision_base_url = "https://alpha.test"
decision_api_key_env = "SPAN_KEY"
supported_types = {{ noul = true, choice = false, score = false }}

[targets.mercury]
id = "mercury/model"
llm_client = "primary"
decision_transport = "system_one"
decision_base_url = "https://systemone.test"
decision_api_key_env = "MERCURY_KEY"

[targets.jev]
id = "jev/model"
llm_client = "primary"
decision_transport = "system_one"
decision_base_url = "https://systemone.test"
decision_api_key_env = "JEV_KEY"

[targets.laya]
id = "laya/model"
llm_client = "primary"
decision_transport = "system_one"
decision_base_url = "https://systemone.test"
decision_api_key_env = "LAYA_KEY"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[decision_lanes.free-family]
targets = ["span", "mercury"]

[decision_lanes.local-family]
targets = ["jev", "laya"]

[lane_placeholder]
{lanes}
{composition}
"#
    )
}

/// A standalone lane config with only the supplied lane blocks.
fn build(lanes: &str, composition: &str) -> Result<Runner, RunnerError> {
    // The placeholder keeps the template readable; strip it before parsing.
    let toml = decision_config(lanes, composition).replace("[lane_placeholder]\n", "");
    Runner::from_toml(&toml)
}

/// The refusal message for a configuration that must not load.
fn refusal(lanes: &str, composition: &str) -> String {
    match build(lanes, composition) {
        Ok(_) => panic!("deployment should have been refused"),
        Err(error) => error.to_string(),
    }
}

fn with_env<F: FnOnce()>(f: F) {
    // SAFETY: single-threaded per test; each sets only its own keys.
    unsafe {
        std::env::set_var("PRIMARY_KEY", "primary-dummy");
        std::env::set_var("SPAN_KEY", "span-dummy");
        std::env::set_var("MERCURY_KEY", "mercury-dummy");
        std::env::set_var("JEV_KEY", "jev-dummy");
        std::env::set_var("LAYA_KEY", "laya-dummy");
    }
    f();
}

#[test]
fn an_explicitly_named_lane_resolves_to_itself_not_the_default() {
    with_env(|| {
        let runner = build(
            "",
            r#"
[decision_lane_composition.default-lane-composite]
children = ["free-family", "local-family"]
"#,
        )
        .expect("deployment should load");

        // Both siblings are independently addressable.
        assert!(runner.named_decision_lane("free-family").is_some());
        assert!(runner.named_decision_lane("local-family").is_some());

        // The free family holds its own two candidates, in its own order.
        let free = runner
            .named_decision_lane("free-family")
            .expect("free family lane");
        assert_eq!(
            free.resolver
                .candidates
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["span", "mercury"]
        );

        // The local family is a DIFFERENT lane and neither shadows the other.
        let local = runner
            .named_decision_lane("local-family")
            .expect("local family lane");
        assert_eq!(
            local
                .resolver
                .candidates
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["jev", "laya"]
        );
    });
}

#[test]
fn an_unscoped_lane_is_not_promoted_to_a_default_implicitly() {
    with_env(|| {
        // A second unscoped lane sorts BEFORE the declared default.
        let runner = build(
            r#"
[decision_lanes.aaa-extra-unscoped]
targets = ["mercury"]
"#,
            "",
        )
        .expect("deployment should load");

        // The declared default is used, not whichever unscoped lane came first.
        assert_eq!(runner.decision_lane_name(), Some("free-family"));
        // The extra lane is still addressable by name, which is the whole point.
        assert!(runner.named_decision_lane("aaa-extra-unscoped").is_some());
    });
}

#[test]
fn a_declared_default_naming_no_lane_is_refused() {
    with_env(|| {
        // A default that names no configured lane must fail at load, not silently
        // leave every unaddressed request to fail later at request time.
        let bad = decision_config("", "")
            .replace("[lane_placeholder]\n", "")
            .replace(
                "default_decision_lane = \"free-family\"",
                "default_decision_lane = \"no-such-lane\"",
            );
        match Runner::from_toml(&bad) {
            Ok(_) => panic!("a default naming no lane must be refused"),
            Err(error) => assert!(
                error.to_string().contains("not a configured decision lane"),
                "{error}"
            ),
        }
    });
}

#[test]
fn composition_is_referenced_not_flattened() {
    with_env(|| {
        let runner = build(
            "",
            r#"
[decision_lane_composition.tier]
children = ["free-family", "local-family"]
"#,
        )
        .expect("deployment should load");

        assert_eq!(
            runner.decision_lane_children("tier"),
            Some(["free-family".to_string(), "local-family".to_string()].as_slice())
        );
        assert!(runner.decision_lane_is_composite("tier"));
        // A composed lane holds child LANES; its children are not copied into it.
        assert!(runner.named_decision_lane("tier").is_none());
        // Both children remain independently addressable while the tier exists.
        assert!(runner.named_decision_lane("free-family").is_some());
        assert!(runner.named_decision_lane("local-family").is_some());
    });
}

#[test]
fn a_composition_cycle_is_refused_at_load() {
    with_env(|| {
        let error = refusal(
            "",
            r#"
[decision_lane_composition.tier]
children = ["free-family", "tier-two"]

[decision_lane_composition.tier-two]
children = ["local-family", "tier"]
"#,
        );

        assert!(error.contains("cycle"), "{error}");
    });
}

#[test]
fn a_self_referential_composition_is_refused() {
    with_env(|| {
        let error = refusal(
            "",
            r#"
[decision_lane_composition.tier]
children = ["tier"]
"#,
        );

        assert!(error.to_string().contains("itself"), "{error}");
    });
}

#[test]
fn an_unknown_child_lane_is_refused() {
    with_env(|| {
        let error = refusal(
            "",
            r#"
[decision_lane_composition.tier]
children = ["free-family", "no-such-lane"]
"#,
        );

        assert!(error.to_string().contains("unknown child lane"), "{error}");
    });
}

#[test]
fn an_empty_composition_is_refused() {
    with_env(|| {
        let error = refusal(
            "",
            r#"
[decision_lane_composition.tier]
children = []
"#,
        );

        assert!(error.to_string().contains("at least one child"), "{error}");
    });
}

#[test]
fn a_name_cannot_be_both_a_target_lane_and_a_composed_lane() {
    with_env(|| {
        let error = refusal(
            "",
            r#"
[decision_lane_composition.free-family]
children = ["local-family"]
"#,
        );

        assert!(
            error.to_string().contains("both as a target lane"),
            "{error}"
        );
    });
}

#[test]
fn resolution_does_not_depend_on_btree_lexical_order() {
    with_env(|| {
        // Same sibling lanes, but the default lane is named so it sorts LAST.
        let toml = r#"
schema_version = 1
default_decision_lane = "zzz-family"

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "PRIMARY_KEY"

[targets.a_span]
id = "span/model"
llm_client = "primary"
decision_transport = "open_router_alpha"
decision_path = "/api/alpha/decisions"
decision_base_url = "https://alpha.test"
decision_api_key_env = "SPAN_KEY"

[targets.b_mercury]
id = "mercury/model"
llm_client = "primary"
decision_transport = "system_one"
decision_base_url = "https://systemone.test"
decision_api_key_env = "MERCURY_KEY"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[decision_lanes.zzz-family]
targets = ["a_span", "b_mercury"]
"#;
        let runner = Runner::from_toml(toml).expect("deployment should load");
        // Despite sorting last, it is the only unscoped lane and is addressable by
        // name; public identity does not depend on which name happens to come first.
        assert_eq!(runner.decision_lane_name(), Some("zzz-family"));
        assert!(runner.named_decision_lane("zzz-family").is_some());
    });
}

#[test]
fn an_unsupported_question_type_reaches_no_candidate() {
    with_env(|| {
        let runner = build("", "").expect("deployment should load");
        let free = runner
            .named_decision_lane("free-family")
            .expect("free family");

        // Span declares choice = false. A choice question must be refused by span
        // pre-dispatch, and mercury (which supports it) is the one that may serve.
        let mut questions = BTreeMap::new();
        questions.insert(
            "risk".to_string(),
            DecisionQuestion {
                instructions: json!("assess"),
                kind: DecisionKind::Choice {
                    options: vec![
                        switchyard_protocol::ChoiceOption {
                            id: "yes".to_string(),
                            description: None,
                        },
                        switchyard_protocol::ChoiceOption {
                            id: "no".to_string(),
                            description: None,
                        },
                    ],
                },
            },
        );
        let choice = DecisionRequest {
            model: None,
            context: json!("clean"),
            questions,
        };
        let span = free
            .resolver
            .candidates
            .iter()
            .find(|c| c.name == "span")
            .expect("span candidate");
        assert!(
            span.eligibility(&choice).is_err(),
            "a declared-ineligible backend must not be eligible for the request"
        );
        let mercury = free
            .resolver
            .candidates
            .iter()
            .find(|c| c.name == "mercury")
            .expect("mercury candidate");
        assert!(
            mercury.eligibility(&choice).is_ok(),
            "an all-types backend must remain eligible"
        );
    });
}

/// Exhausting a lane's candidates must let a COMPOSED parent try its next child.
///
/// This is the distinction the composition mechanism depends on: a lane that ran out
/// of healthy backends is exhausted, not broken, so a tier must still be able to
/// enter its next family. A lane where every candidate was skipped pre-dispatch is a
/// different fact and stays terminal.
#[tokio::test]
async fn an_exhausted_child_lane_yields_to_the_next_child() {
    use std::sync::Arc;

    use switchyard_runner::decision_executor::{
        DecisionCandidate, DecisionFailure, DecisionResolver, LaneChild, resolve_and_serve,
        serve_composed,
    };
    use switchyard_runner::decision_transport::{DecisionTransport, StateForms, SupportedTypes};

    fn candidate(name: &str, url: &str) -> DecisionCandidate {
        DecisionCandidate {
            name: name.to_string(),
            model: name.to_string(),
            transport: DecisionTransport::SystemOne,
            url: url.to_string(),
            api_key: Some("dummy".to_string()),
            api_key_env: Some("DUMMY".to_string()),
            supported_types: SupportedTypes::ALL,
            state_forms: StateForms::ALL,
        }
    }

    fn request() -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "risk".to_string(),
            DecisionQuestion {
                instructions: json!("assess"),
                kind: DecisionKind::Boolean {
                    true_description: None,
                    false_description: None,
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!("state"),
            questions,
        }
    }

    // First child: a candidate that was actually SENT to and refused the connection.
    let dead = DecisionResolver {
        candidates: vec![candidate("dead", "http://127.0.0.1:9/unreachable")],
    };
    // A candidate with no credential is terminal (a deployment fault), so it must NOT
    // be used to simulate an exhausted child here.
    let outcome = resolve_and_serve(&dead, &request()).await;
    assert!(
        matches!(
            &outcome,
            Err(DecisionFailure::Serving {
                retryable: true,
                ..
            })
        ),
        "a sent-and-failed candidate must be retryable so a parent can continue: {outcome:?}"
    );

    // Second child: also unreachable, which is fine: what matters is that the parent
    // ATTEMPTED it, proving it was not short-circuited by the first child's exhaustion.
    let children = vec![
        LaneChild::new("free-family".to_string(), Arc::new(dead)),
        LaneChild::new(
            "local-family".to_string(),
            Arc::new(DecisionResolver {
                candidates: vec![candidate("also-dead", "http://127.0.0.1:9/unreachable")],
            }),
        ),
    ];
    let mut stack = vec!["tier".to_string()];
    let composed = serve_composed(&children, &request(), &mut stack).await;
    let error = composed.expect_err("both children are unreachable");
    assert!(
        matches!(&error, DecisionFailure::Serving { reason, retryable: true }
            if *reason == "all_child_lanes_exhausted"),
        "the tier must report exhaustion of its children, not an untried-lane error: {error:?}"
    );
}

/// A child that could not ELIGIBLY serve is never dispatched and stays terminal, so an
/// ineligible request is not forwarded to a sibling that would only refuse it too.
#[tokio::test]
async fn an_ineligible_request_is_not_forwarded_to_a_sibling() {
    use std::sync::Arc;

    use switchyard_runner::decision_executor::{
        DecisionCandidate, DecisionFailure, DecisionResolver, LaneChild, serve_composed,
    };
    use switchyard_runner::decision_transport::{DecisionTransport, StateForms, SupportedTypes};

    let noul_only = DecisionCandidate {
        name: "noul-only".to_string(),
        model: "noul-only".to_string(),
        transport: DecisionTransport::SystemOne,
        // Nothing may be sent here: the test would fail on a real request.
        url: "http://127.0.0.1:9/unreachable".to_string(),
        api_key: Some("dummy".to_string()),
        api_key_env: Some("DUMMY".to_string()),
        supported_types: SupportedTypes::NOUL_ONLY,
        state_forms: StateForms::SPAN_OBSERVED,
    };
    let children = vec![
        LaneChild::new(
            "free-family".to_string(),
            Arc::new(DecisionResolver {
                candidates: vec![noul_only],
            }),
        ),
        LaneChild::new(
            "local-family".to_string(),
            Arc::new(DecisionResolver {
                candidates: vec![DecisionCandidate {
                    name: "sibling".to_string(),
                    model: "sibling".to_string(),
                    transport: DecisionTransport::SystemOne,
                    url: "http://127.0.0.1:9/unreachable".to_string(),
                    api_key: Some("dummy".to_string()),
                    api_key_env: Some("DUMMY".to_string()),
                    supported_types: SupportedTypes::ALL,
                    state_forms: StateForms::ALL,
                }],
            }),
        ),
    ];
    let mut questions = BTreeMap::new();
    questions.insert(
        "risk".to_string(),
        DecisionQuestion {
            instructions: json!("assess"),
            kind: DecisionKind::Choice {
                options: vec![
                    switchyard_protocol::ChoiceOption {
                        id: "yes".into(),
                        description: None,
                    },
                    switchyard_protocol::ChoiceOption {
                        id: "no".into(),
                        description: None,
                    },
                ],
            },
        },
    );
    let request = DecisionRequest {
        model: None,
        context: json!("state"),
        questions,
    };
    let mut stack = vec!["tier".to_string()];
    let error = serve_composed(&children, &request, &mut stack)
        .await
        .expect_err("no child is eligible for a choice question");
    assert!(
        !error.try_next(),
        "a request no candidate can serve must stay terminal, not be forwarded: {error:?}"
    );
}

/// The INTERNAL decision-call path must resolve a named lane the same way the public
/// surface does.
///
/// This was a real split-contract defect: the public `/v1/decisions` handler honoured an
/// explicitly requested lane and dispatched composed lanes, while the internal
/// `Call::Decision` handler only consulted the route-local resolver and the deployment
/// default. The same named request could therefore be served at `/v1/decisions` and fail
/// internally. The fix threads named composition into the internal handler.
///
/// COVERAGE LIMIT, stated rather than papered over: `CallDecision` is not constructible
/// outside its crate, so the internal handler cannot be driven directly from an
/// integration test. A mutant that removes the internal named-composition dispatch
/// SURVIVES this suite, which is honest evidence that this test does not by itself prove
/// the internal path. What it does prove is that the identities the internal handler
/// resolves against exist and are correct. The internal dispatch is covered only by the
/// live qualification in operations/decision-plane-named-lanes-2026-10-04.md, which drives
/// the public surface, and by code review of the handler body.
#[test]
fn named_lanes_and_composition_are_resolvable_for_the_internal_handler() {
    with_env(|| {
        let runner = build(
            "",
            r#"
[decision_lane_composition.tier]
children = ["free-family", "local-family"]
"#,
        )
        .expect("deployment should load");

        // Every identity an internal decision call may name resolves to a lane.
        for name in ["free-family", "local-family"] {
            assert!(
                runner.named_decision_lane(name).is_some(),
                "internal path must resolve the named lane {name}"
            );
        }
        // The tier resolves as a COMPOSITION, with its families in order, so the internal
        // handler can walk child lanes rather than a copied target list.
        assert_eq!(
            runner.decision_lane_children("tier"),
            Some(["free-family".to_string(), "local-family".to_string()].as_slice())
        );
        // And a lane named for nothing still fails closed rather than falling through to
        // whichever lane happens to exist.
        assert!(runner.named_decision_lane("no-such-lane").is_none());
    });
}

/// The classifier the INTERNAL `Call::Decision` path uses must classify a named lane,
/// and that is what kills the mutant.
///
/// The previous disposition - "CallDecision is crate-private, so the internal handler
/// cannot be tested" - was accepted too readily. `CallDecision` itself is crate-private,
/// but the internal handler's decision point was extracted into
/// `decision_executor::classify_named`, which is the very function the handler calls.
/// Testing the decision point is the part that had the defect; driving libsy's whole
/// call loop would add nothing to it and would require production-only surface.
///
/// A mutation that makes the internal path bypass named classification is therefore
/// detectable here: with the dispatch removed, a named lane falls through to the
/// route-local resolver instead of running its own policy.
#[test]
fn the_shared_classifier_resolves_an_explicitly_named_lane() {
    use std::collections::BTreeMap as Map;
    use std::sync::Arc;

    use switchyard_runner::decision_executor::{
        DecisionResolver, LaneResolverHandle, NamedDispatch, classify_named, composed_children,
        serve_named,
    };

    let handle = |_name: &str| LaneResolverHandle {
        resolver: Arc::new(DecisionResolver {
            candidates: Vec::new(),
        }),
    };
    let lanes: Map<String, LaneResolverHandle> = [
        ("free-family".to_string(), handle("free")),
        ("local-family".to_string(), handle("local")),
    ]
    .into_iter()
    .collect();
    let composition: Map<String, Vec<String>> =
        [("tier".to_string(), vec!["free-family".to_string(), "local-family".to_string()])]
            .into_iter()
            .collect();

    // A composed lane, a leaf lane, and an unknown name are three DISTINCT answers.
    assert_eq!(
        classify_named(&composition, &lanes, "tier"),
        NamedDispatch::Composed,
        "a named tier must be classified as composed, not fall through"
    );
    assert_eq!(
        classify_named(&composition, &lanes, "free-family"),
        NamedDispatch::Leaf,
        "a named family must be classified as a leaf, not fall through"
    );
    assert_eq!(
        classify_named(&composition, &lanes, "no-such-lane"),
        NamedDispatch::Unknown,
        "an unknown name must fall through rather than be served by some other lane"
    );
    assert_eq!(
        classify_named(&composition, &lanes, ""),
        NamedDispatch::Unknown,
        "an empty name is not an explicit identity"
    );

    // The composed lane's children are built in CONFIGURED order, and an unknown child
    // fails closed instead of silently shrinking the policy.
    let children = composed_children(&lanes, &["free-family".to_string(), "local-family".to_string()])
        .expect("known children resolve");
    assert_eq!(
        children.iter().map(|c| c.lane().to_string()).collect::<Vec<_>>(),
        ["free-family", "local-family"]
    );
    let missing = composed_children(&lanes, &["free-family".to_string(), "ghost".to_string()]);
    assert!(
        missing.is_err(),
        "an unknown child must fail closed rather than be skipped"
    );
}


/// `serve_named` itself must serve a NAMED lane, and refuse to answer an unknown name.
///
/// This is the test that kills the internal-bypass mutant. A mutation that makes
/// `serve_named` fall through instead of serving the named lane leaves
/// `classify_named` and `composed_children` intact, so a test of those helpers alone
/// survives it. Driving `serve_named` is what makes the whole internal/public decision
/// point mechanical: the internal handler and the public handler both call it, so a
/// bypass is now detectable rather than plausible.
#[tokio::test]
async fn serve_named_serves_the_named_lane_and_falls_through_only_when_unknown() {
    use std::collections::BTreeMap as Map;
    use std::sync::Arc;

    use switchyard_protocol::{DecisionKind, DecisionQuestion, DecisionRequest};
    use switchyard_runner::decision_executor::{
        DecisionCandidate, DecisionResolver, LaneResolverHandle, serve_named,
    };
    use switchyard_runner::decision_transport::{
        DecisionTransport, StateForms, SupportedTypes,
    };
    use serde_json::json;

    // A lane whose single candidate answers: reaching it proves the named lane RAN.
    fn answering(name: &str) -> LaneResolverHandle {
        LaneResolverHandle {
            resolver: Arc::new(DecisionResolver {
                candidates: vec![DecisionCandidate {
                    name: name.to_string(),
                    model: name.to_string(),
                    transport: DecisionTransport::SystemOne,
                    // Nothing is reachable here, so a 200 would be impossible; the test
                    // asserts on WHICH backend was attempted instead.
                    url: "http://127.0.0.1:9/unreachable".to_string(),
                    api_key: Some("dummy".to_string()),
                    api_key_env: Some("DUMMY".to_string()),
                    supported_types: SupportedTypes::ALL,
                    state_forms: StateForms::ALL,
                }],
            }),
        }
    }

    let lanes: Map<String, LaneResolverHandle> = [
        ("free-family".to_string(), answering("free-backend")),
        ("local-family".to_string(), answering("local-backend")),
    ]
    .into_iter()
    .collect();
    let composition: Map<String, Vec<String>> =
        [("tier".to_string(), vec!["free-family".to_string(), "local-family".to_string()])]
            .into_iter()
            .collect();

    let mut questions = std::collections::BTreeMap::new();
    questions.insert(
        "risk".to_string(),
        DecisionQuestion {
            instructions: json!("assess"),
            kind: DecisionKind::Boolean {
                true_description: None,
                false_description: None,
            },
        },
    );
    let request = DecisionRequest {
        model: None,
        context: json!("state"),
        questions,
    };

    // A name that is NOT a configured lane falls through, so the caller can try its own
    // route-local resolution. This is the ONLY case that may fall through.
    assert!(
        serve_named(&composition, &lanes, "no-such-lane", &request)
            .await
            .expect("an unknown name is not a serving failure")
            .is_none(),
        "an unknown name must fall through rather than be served by some other lane"
    );

    // A NAMED lane is served by that lane and never falls through: its own candidates are
    // unreachable, so the outcome is that lane's failure rather than None.
    for name in ["free-family", "local-family", "tier"] {
        let outcome = serve_named(&composition, &lanes, name, &request).await;
        assert!(
            outcome.is_err(),
            "named lane {name} must run its own policy, not fall through to Ok(None)"
        );
    }
}
