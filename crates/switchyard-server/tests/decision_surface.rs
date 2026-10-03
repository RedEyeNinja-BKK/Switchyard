// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The retained typed decision compatibility surface, and the routing preview it must not
//! disturb.
//!
//! `/v1/decisions` exists because current consumers call it. `/v1/decision` is upstream's
//! routing preview and keeps its own contract; the singular and plural spellings are
//! different endpoints on purpose, and both are exercised here so neither can drift into
//! the other.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use switchyard_runner::Runner;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn serve(config: &str) -> axum::Router {
    let runner = Runner::from_toml(config).expect("runner builds");
    switchyard_server::build_switchyard_router(
        switchyard_server::ServerState::from_runner(runner).expect("state builds"),
    )
}

/// A deployment with one completion route and two decision backends: a local
/// full-type SystemOne backend and a noul-only alpha backend that also accepts only a
/// plain string state.
fn config(systemone_url: &str, alpha_url: &str) -> String {
    format!(
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

[targets.decision-laya]
id = "laya-rl-agent"
llm_client = "responses"
decision_transport = "system_one"
decision_path = "/v1/systemone"
decision_base_url = "{systemone_url}"
decision_api_key_env = "TEST_KEY"

[targets.decision-jev-style]
id = "jev-style-0.8b-decision-v3"
llm_client = "responses"
decision_transport = "system_one"
decision_path = "/v1/systemone"
decision_base_url = "{systemone_url}"
decision_api_key_env = "TEST_KEY"

[targets.decision-span]
id = "respan/span-01-lite"
llm_client = "responses"
decision_transport = "open_router_alpha"
decision_path = "/api/alpha/decisions"
decision_base_url = "{alpha_url}"
decision_api_key_env = "TEST_KEY"
supported_types = {{ noul = true, choice = false, score = false }}
state_forms = {{ plain_string = true, structured = false }}

[decision_lanes.default]
targets = ["decision-laya", "decision-jev-style", "decision-span"]

[decision_lanes.noul]
types = ["noul"]
targets = ["decision-laya", "decision-jev-style", "decision-span"]

[decision_lanes.choice]
types = ["choice"]
targets = ["decision-span", "decision-laya"]

[decision_lanes.score]
types = ["score"]
targets = ["decision-laya", "decision-span"]
"#
    )
}

async fn post(router: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body reads");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

/// A Turnstone-shaped noul request reaches the lane and comes back in the shape the
/// consumer parses: keyed answers, each tagged with its kind.
#[tokio::test]
async fn a_typed_noul_request_is_served_through_the_lane() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "laya-rl-agent",
            "answers": {"risk": {"type": "noul", "noul": 0.82}},
            "usage": {"input_tokens": 52, "output_tokens": 0, "total_tokens": 52}
        })))
        .mount(&backend)
        .await;

    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&backend.uri(), &backend.uri())).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "reviewing a tool output",
            "questions": {
                "risk": {
                    "type": "noul",
                    "instructions": "Flag any error in this output.",
                    "criteria": {"true": "a problem is present", "false": "the output is clean"}
                }
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    let answers = body["answers"].as_object().expect("keyed answers");
    assert_eq!(answers["risk"]["type"], "noul");
    assert_eq!(answers["risk"]["noul"], 0.82);
    assert_eq!(body["usage"]["input_tokens"], 52, "provider usage is carried");
    assert_eq!(body["model"], "laya-rl-agent");
    assert_eq!(body["backend"], "decision-laya", "the answering backend is named");
}

#[tokio::test]
async fn a_choice_request_reaches_the_full_type_backend() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {
                "team": {
                    "type": "choice",
                    "choice": "technical",
                    "confidence": 0.78,
                    "probabilities": {"billing": 0.22, "technical": 0.78}
                }
            }
        })))
        .expect(1)
        .mount(&backend)
        .await;

    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&backend.uri(), &backend.uri())).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "a ticket needs triage",
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": null, "technical": null}}
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"]["team"]["type"], "choice");
    assert_eq!(body["answers"]["team"]["choice"], "technical");
    assert_eq!(body["answers"]["team"]["probabilities"]["technical"], 0.78);
    assert_eq!(body["answers"]["team"]["confidence"], 0.78);
    assert_eq!(
        body["backend"], "decision-laya",
        "the noul-only backend must be excluded for a choice request"
    );
}

#[tokio::test]
async fn a_score_request_keeps_its_estimate_and_distribution() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {
                "severity": {
                    "type": "score",
                    "score": 1.25,
                    "confidence": 0.6,
                    "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}
                }
            }
        })))
        .mount(&backend)
        .await;

    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&backend.uri(), &backend.uri())).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "a ticket needs triage",
            "questions": {
                "severity": {"type": "score", "instructions": "Rate it", "criteria": ["low", "medium", "high"]}
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"]["severity"]["type"], "score");
    assert_eq!(body["answers"]["severity"]["score"], 1.25);
    assert_eq!(
        body["answers"]["severity"]["probabilities"],
        json!([0.1, 0.5, 0.4]),
        "the ordered distribution survives"
    );
}

/// An out-of-bounds request is a contract error and must reach no backend at all.
#[tokio::test]
async fn an_out_of_bounds_request_reaches_no_backend() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(0)
        .mount(&backend)
        .await;

    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&backend.uri(), &backend.uri())).await;
    let criteria: serde_json::Map<String, Value> = (0..256)
        .map(|index| (format!("o{index}"), Value::Null))
        .collect();
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "state",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": criteria
                }
            }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"].as_str().is_some_and(|error| error.contains("Contract")),
        "the failure must name the contract violation: {body}"
    );
}

#[tokio::test]
async fn a_malformed_compatibility_body_is_refused() {
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let backend = MockServer::start().await;
    let router = serve(&config(&backend.uri(), &backend.uri())).await;

    let (status, body) = post(
        router.clone(),
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "state",
            "questions": {"q": {"type": "vibe", "instructions": "?"}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().is_some_and(|e| e.contains("vibe")), "{body}");

    let (status, _) = post(
        router,
        "/v1/decisions",
        json!({"model": "x", "state": "s", "questions": {}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Upstream's routing preview keeps its own contract: it takes `input_format` and
/// `request`, and answers with routing information rather than typed answers. Adding the
/// plural surface must not change any of that.
#[tokio::test]
async fn the_upstream_routing_preview_is_unchanged() {
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let backend = MockServer::start().await;
    let router = serve(&config(&backend.uri(), &backend.uri())).await;

    // Its documented body is rejected if it has been repointed at typed questions.
    let (status, body) = post(
        router.clone(),
        "/v1/decision",
        json!({"state": "x", "questions": {"risk": {"type": "noul"}}}),
    )
    .await;
    assert_ne!(
        status,
        StatusCode::OK,
        "the routing preview must not accept a typed decision body"
    );
    assert!(
        !body.get("answers").is_some(),
        "the preview must not answer with typed decisions: {body}"
    );

    // And it still answers a routing request in its own shape.
    let (status, body) = post(
        router,
        "/v1/decision",
        json!({
            "input_format": "openai_responses",
            "request": {
                "model": "probe/only",
                "input": [{"role": "user", "content": "hello"}]
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body.get("selected").is_some() || body.get("model").is_some(),
        "the preview answers with routing information: {body}"
    );
}
/// A provider outage must not be reported to the caller as a malformed request. These
/// assert the externally visible status, since that is what a consumer reacts to.
#[tokio::test]
async fn a_backend_outage_is_service_unavailable_not_bad_request() {
    // Every configured backend times out at the transport: nothing is reachable.
    let unreachable = "http://127.0.0.1:1";
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(unreachable, unreachable)).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "a plain state",
            "questions": {"risk": {"type": "noul", "instructions": "Risky?"}}
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "an exhausted lane is unavailable, not a caller error: {body}"
    );
}

/// A 503 from the provider is likewise an availability problem.
#[tokio::test]
async fn a_provider_503_is_service_unavailable() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({"error": "down"})))
        .mount(&backend)
        .await;
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&backend.uri(), &backend.uri())).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "default",
            "state": "a plain state",
            "questions": {"risk": {"type": "noul", "instructions": "Risky?"}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
}

/// An unknown identity is a missing model, not an outage and not a bad request body.
#[tokio::test]
async fn an_unknown_identity_is_not_found_and_dispatches_nothing() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(0)
        .mount(&backend)
        .await;
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&backend.uri(), &backend.uri())).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "decision-does-not-exist",
            "state": "a plain state",
            "questions": {"risk": {"type": "noul", "instructions": "Risky?"}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|error| error.contains("TargetNotFound")),
        "{body}"
    );
}

/// Naming a configured backend directly serves that backend alone.
#[tokio::test]
async fn a_concrete_target_is_served_without_touching_the_rest_of_the_lane() {
    let laya = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.7}}
        })))
        .expect(0)
        .mount(&laya)
        .await;
    let span = MockServer::start().await;
    // SPAN answers noul, so naming it serves SPAN and never consults the rest of the
    // lane, even though Laya is first in the configured order.
    Mock::given(method("POST"))
        .and(path("/api/alpha/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.31}}
        })))
        .expect(1)
        .mount(&span)
        .await;

    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let router = serve(&config(&laya.uri(), &span.uri())).await;
    let (status, body) = post(
        router,
        "/v1/decisions",
        json!({
            "model": "decision-span",
            "state": "a plain state",
            "questions": {"risk": {"type": "noul", "instructions": "Risky?"}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["backend"], "decision-span", "the named backend answered");
    assert_eq!(body["answers"]["risk"]["noul"], 0.31);
}

/// A lane listing a target twice, an unknown target, or a non-decision target is a
/// configuration error, caught at load rather than at the first request.
#[tokio::test]
async fn an_invalid_lane_is_rejected_at_load() {
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let base = |lane: &str| {
        format!(
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

[targets.decision-laya]
id = "laya-rl-agent"
llm_client = "responses"
decision_transport = "system_one"
decision_api_key_env = "TEST_KEY"

{lane}
"#
        )
    };
    let message = |source: &str| {
        Runner::from_toml(source)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    };

    let duplicate = message(&base(
        "[decision_lanes.default]\ntargets = [\"decision-laya\", \"decision-laya\"]",
    ));
    assert!(duplicate.contains("more than once"), "{duplicate}");

    let unknown = message(&base(
        "[decision_lanes.default]\ntargets = [\"decision-nope\"]",
    ));
    assert!(unknown.contains("unknown target"), "{unknown}");

    let not_a_decision = message(&base("[decision_lanes.default]\ntargets = [\"only\"]"));
    assert!(
        not_a_decision.contains("not a decision target"),
        "{not_a_decision}"
    );

    let empty = message(&base("[decision_lanes.default]\ntargets = []"));
    assert!(empty.contains("at least one target"), "{empty}");
}

/// The bake-off produced different orders per question type, so lane selection must be
/// per type rather than one ranking applied to every request. A type-scoped lane must win
/// over the general lane, and a request no lane covers must be refused rather than
/// falling through to whichever lane is registered first.
#[tokio::test]
async fn a_type_scoped_lane_is_selected_for_its_own_kinds() {
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {
                "q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "technical": 0.0}}
            }
        })))
        .mount(&backend)
        .await;
    let runner = Runner::from_toml(&config(&backend.uri(), &backend.uri())).expect("deployment builds");

    let choice = json!({
        "model": "switchyard/choice",
        "state": {"structured": true},
        "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"billing": null, "technical": null}}}
    });
    let request = switchyard_server::decision_surface::CompatibilityDecisionRequest::from_value(&choice).and_then(switchyard_server::decision_surface::to_native).expect("native request");
    let lane = runner
        .lane_for_request(&request)
        .expect("a choice request selects a lane");
    assert_eq!(
        lane.resolver.candidates.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        vec!["decision-span", "decision-laya"],
        "the choice lane's own order is used, not the general lane's"
    );

    let noul = json!({
        "model": "switchyard/noul",
        "state": "plain",
        "questions": {"q": {"type": "noul", "instructions": "Risky?"}}
    });
    let noul_request = switchyard_server::decision_surface::CompatibilityDecisionRequest::from_value(&noul).and_then(switchyard_server::decision_surface::to_native).expect("native request");
    let noul_lane = runner
        .lane_for_request(&noul_request)
        .expect("a noul request selects a lane");
    assert_eq!(
        noul_lane.resolver.candidates.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        vec!["decision-laya", "decision-jev-style", "decision-span"]
    );
}

/// Two scoped lanes claiming the same kind would make selection ambiguous, so the
/// deployment is refused at load rather than producing no lane at request time.
#[tokio::test]
async fn an_ambiguous_lane_scope_is_rejected_at_load() {
    unsafe { std::env::set_var("TEST_KEY", "test-value") };
    let source = r#"
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

[targets.decision-a]
id = "backend-a"
llm_client = "responses"
decision_transport = "system_one"
decision_api_key_env = "TEST_KEY"

[targets.decision-b]
id = "backend-b"
llm_client = "responses"
decision_transport = "system_one"
decision_api_key_env = "TEST_KEY"

[decision_lanes.first]
types = ["noul"]
targets = ["decision-a"]

[decision_lanes.second]
types = ["noul"]
targets = ["decision-b"]
"#;
    let error = Runner::from_toml(source)
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        error.contains("same specificity"),
        "duplicate scopes must be refused at load: {error}"
    );

    // Distinct kinds are not ambiguous, and the unscoped lane may coexist with both.
    let distinct = source
        .replace(
            "[decision_lanes.second]\ntypes = [\"noul\"]",
            "[decision_lanes.second]\ntypes = [\"choice\"]",
        )
        .replace(
            "[decision_lanes.first]",
            "[decision_lanes.general]\ntargets = [\"decision-a\"]\n\n[decision_lanes.first]",
        );
    Runner::from_toml(&distinct).expect("distinct kinds and an unscoped lane are valid");
}
