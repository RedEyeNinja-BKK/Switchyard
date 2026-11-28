// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability-first, provider-neutral selection for a composed decision lane.
//!
//! The behaviour under test is a policy, so the tests are about *who is asked*, not about
//! the answer. Each physical backend is a real mock HTTP server, so "SPAN was not called"
//! is measured on the wire rather than inferred from a code path.
//!
//! Two properties are load-bearing and are tested separately:
//!
//! * **Fit before dispatch.** Every candidate in the pool is tested against the whole
//!   request before anything is sent, so a backend that cannot serve it is never contacted
//!   and never decides the lane by being declared first.
//! * **Neutral entry point.** The pool is entered at a rotating position, so no declared
//!   position - and therefore no provider - is systematically offered the request first.
//!
//! A third property is the negative one: ordered composition, which the historic tiers
//! use, must be untouched.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use switchyard_llm_client::DailyRequestAllowance;
use switchyard_protocol::{ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, ModelId};
use switchyard_runner::Runner;
use switchyard_runner::decision_executor::{
    ComposedLane, DecisionBudget, DecisionCandidate, DecisionFailure, DecisionResolver,
    DecisionSkip, LaneChild, LaneResolverHandle, LaneSelection, capability_pool, rotation_order,
    serve_capability_first, serve_named,
};
use switchyard_runner::decision_transport::{
    DecisionTransport, MAX_SCORE_LEVELS, StateForms, SupportedTypes,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// --- fixtures ---------------------------------------------------------------

fn noul_request(context: serde_json::Value) -> DecisionRequest {
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
        context,
        questions,
    }
}

fn choice_request(context: serde_json::Value) -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "team".to_string(),
        DecisionQuestion {
            instructions: json!("pick"),
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
        context,
        questions,
    }
}

/// A `score` request one level beyond the SystemOne wire bound, so a local backend
/// refuses it on its contract while the request stays parseable.
fn oversize_score_request() -> DecisionRequest {
    let mut questions = BTreeMap::new();
    questions.insert(
        "quality".to_string(),
        DecisionQuestion {
            instructions: json!("rate"),
            kind: DecisionKind::Score {
                levels: (0..=MAX_SCORE_LEVELS).map(|level| json!(level)).collect(),
            },
        },
    );
    DecisionRequest {
        model: None,
        context: json!("state"),
        questions,
    }
}

fn backend(
    name: &str,
    model: &str,
    transport: DecisionTransport,
    types: SupportedTypes,
    state_forms: StateForms,
    server: &MockServer,
) -> DecisionCandidate {
    DecisionCandidate {
        name: name.to_string(),
        model: model.to_string(),
        transport,
        url: format!("{}{}", server.uri(), transport.path()),
        api_key: Some("SUPER_SECRET_DECISION_TEST_KEY".to_string()),
        api_key_env: Some("TEST_DECISION_KEY".to_string()),
        supported_types: types,
        state_forms,
        budget: None,
    }
}

fn local(name: &str, server: &MockServer) -> DecisionCandidate {
    backend(
        name,
        &format!("{name}-model"),
        DecisionTransport::SystemOne,
        SupportedTypes::ALL,
        StateForms::ALL,
        server,
    )
}

/// The one OpenRouter decision leaf: noul only, plain-string state only.
fn span(server: &MockServer) -> DecisionCandidate {
    backend(
        "span",
        "respan/span-01-lite",
        DecisionTransport::OpenRouterAlpha,
        SupportedTypes::NOUL_ONLY,
        StateForms::SPAN_OBSERVED,
        server,
    )
}

/// Google's typed decision surface: every kind, structured input only.
fn google(server: &MockServer) -> DecisionCandidate {
    backend(
        "gemini",
        "gemini-decision-1",
        DecisionTransport::GoogleDecisions,
        SupportedTypes::ALL,
        StateForms {
            plain_string: false,
            structured: true,
        },
        server,
    )
}

/// One configured child lane, addressable both as a `LaneChild` for serving and as a
/// `LaneResolverHandle` for `serve_named`. Both share one resolver, so what is served is
/// exactly what is described.
struct Lane {
    name: String,
    resolver: Arc<DecisionResolver>,
}

impl Lane {
    fn new(name: &str, candidates: Vec<DecisionCandidate>) -> Self {
        Self {
            name: name.to_string(),
            resolver: Arc::new(DecisionResolver { candidates }),
        }
    }

    fn child(&self) -> LaneChild {
        LaneChild::new(self.name.clone(), Arc::clone(&self.resolver))
    }

    fn handle(&self) -> (String, LaneResolverHandle) {
        (
            self.name.clone(),
            LaneResolverHandle {
                resolver: Arc::clone(&self.resolver),
            },
        )
    }
}

/// Number of requests a mock server received.
async fn hits(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .map(|requests| requests.len())
        .unwrap_or(0)
}

async fn serve_noul(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.5}}
        })))
        .mount(server)
        .await;
}

async fn serve_noul_alpha(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/api/alpha/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.5}}
        })))
        .mount(server)
        .await;
}

async fn serve_noul_google(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1beta/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"risk": {"type": "noul", "noul": 0.5}}
        })))
        .mount(server)
        .await;
}

async fn serve_choice(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "answers": {"team": {"type": "choice", "choice": "billing"}}
        })))
        .mount(server)
        .await;
}

// --- the pure pieces --------------------------------------------------------

#[test]
fn rotation_visits_every_position_before_repeating() {
    assert_eq!(rotation_order(0, 7), Vec::<usize>::new());
    assert_eq!(rotation_order(1, 3), vec![0]);
    assert_eq!(rotation_order(3, 0), vec![0, 1, 2]);
    assert_eq!(rotation_order(3, 1), vec![1, 2, 0]);
    assert_eq!(rotation_order(3, 2), vec![2, 0, 1]);
    assert_eq!(rotation_order(3, 3), vec![0, 1, 2]);
}

#[tokio::test]
async fn the_pool_is_split_by_fit_before_anything_is_dispatched() {
    let span_server = MockServer::start().await;
    let local_server = MockServer::start().await;
    let google_server = MockServer::start().await;
    let children = vec![
        Lane::new("local-family", vec![local("jev", &local_server)]).child(),
        Lane::new("free-family", vec![span(&span_server)]).child(),
        Lane::new("google-family", vec![google(&google_server)]).child(),
    ];

    // A `choice` request: SPAN cannot answer the kind, Google cannot take a bare string,
    // the local backend fits both. Fit is decided with no request sent.
    let (eligible, refused) = capability_pool(&children, &choice_request(json!("state")));
    assert_eq!(
        eligible
            .iter()
            .map(|entry| entry.candidate.name.as_str())
            .collect::<Vec<_>>(),
        vec!["jev"]
    );
    assert_eq!(
        refused
            .iter()
            .map(|(name, skip)| (name.as_str(), *skip))
            .collect::<Vec<_>>(),
        vec![
            ("free-family/span", DecisionSkip::UnsupportedType),
            ("google-family/gemini", DecisionSkip::UnsupportedState),
        ]
    );
    assert_eq!(hits(&span_server).await, 0);
    assert_eq!(hits(&local_server).await, 0);
    assert_eq!(hits(&google_server).await, 0);
}

// --- dispatch ---------------------------------------------------------------

/// A kind SPAN cannot answer must reach a backend that can, and SPAN must not be called
/// to find that out. This is the regression the ordered composition had: the free family
/// was asked first and its type refusal decided the lane.
#[tokio::test]
async fn a_choice_request_never_spends_a_span_call() {
    let span_server = MockServer::start().await;
    let local_server = MockServer::start().await;
    serve_noul_alpha(&span_server).await;
    serve_choice(&local_server).await;

    let children = vec![
        Lane::new("free-family", vec![span(&span_server)]).child(),
        Lane::new("local-family", vec![local("jev", &local_server)]).child(),
    ];
    let stack = vec!["smart".to_string()];
    let outcome = serve_capability_first(&children, &choice_request(json!("state")), &stack, 0)
        .await
        .expect("the local family answers a choice request");

    assert_eq!(outcome.served_by, "jev");
    assert_eq!(outcome.served_by_lane.as_deref(), Some("local-family"));
    assert_eq!(
        hits(&span_server).await,
        0,
        "SPAN must not be called at all"
    );
    assert_eq!(hits(&local_server).await, 1);
    assert!(
        outcome
            .attempts
            .iter()
            .any(|(name, failure)| name == "free-family/span"
                && *failure == DecisionFailure::NotEligible(DecisionSkip::UnsupportedType)),
        "the trace must say why SPAN was not used: {:?}",
        outcome.attempts
    );
}

/// A structured state is refused by SPAN and accepted by Google, so the rotation can put
/// Google first and the answer comes from there.
#[tokio::test]
async fn a_structured_request_can_be_served_by_google() {
    let span_server = MockServer::start().await;
    let local_server = MockServer::start().await;
    let google_server = MockServer::start().await;
    serve_noul_alpha(&span_server).await;
    serve_noul(&local_server).await;
    serve_noul_google(&google_server).await;

    let children = vec![
        Lane::new("local-family", vec![local("jev", &local_server)]).child(),
        Lane::new("free-family", vec![span(&span_server)]).child(),
        Lane::new("google-family", vec![google(&google_server)]).child(),
    ];
    let request = noul_request(json!({"task": "review"}));
    let stack = vec!["smart".to_string()];
    // Eligible, in declaration order: jev, gemini. Position 1 enters at gemini.
    let outcome = serve_capability_first(&children, &request, &stack, 1)
        .await
        .expect("Google answers a structured noul");

    assert_eq!(outcome.served_by, "gemini");
    assert_eq!(outcome.served_by_lane.as_deref(), Some("google-family"));
    assert_eq!(hits(&span_server).await, 0, "SPAN refuses structured state");
    assert_eq!(
        hits(&local_server).await,
        0,
        "the local peer was not needed"
    );
    assert_eq!(hits(&google_server).await, 1);
}

/// Neutrality is a distribution over requests, not a single ordering: over one full cycle
/// each eligible backend is offered the request first exactly once.
#[tokio::test]
async fn the_entry_point_rotates_over_every_eligible_backend() {
    let span_server = MockServer::start().await;
    let jev_server = MockServer::start().await;
    let laya_server = MockServer::start().await;
    serve_noul_alpha(&span_server).await;
    serve_noul(&jev_server).await;
    serve_noul(&laya_server).await;

    let children = vec![
        Lane::new(
            "local-family",
            vec![local("jev", &jev_server), local("laya", &laya_server)],
        )
        .child(),
        Lane::new("free-family", vec![span(&span_server)]).child(),
    ];
    let request = noul_request(json!("plain state"));

    let mut first_seen = Vec::new();
    for rotation in 0..6 {
        let stack = vec!["smart".to_string()];
        let outcome = serve_capability_first(&children, &request, &stack, rotation)
            .await
            .expect("every rotation has an eligible backend");
        first_seen.push(outcome.served_by);
    }

    // Three eligible backends, six requests: each is offered first twice, and no backend
    // is offered it more often than another.
    for backend in ["jev", "laya", "span"] {
        assert_eq!(
            first_seen.iter().filter(|name| **name == backend).count(),
            2,
            "rotation over 6 requests must offer each backend first twice: {first_seen:?}"
        );
    }
    assert_eq!(hits(&span_server).await, 2);
    assert_eq!(hits(&jev_server).await, 2);
    assert_eq!(hits(&laya_server).await, 2);
}

/// A backend declared by two children is one physical model. Calling it twice for one
/// request would be two provider requests for one answer.
#[tokio::test]
async fn a_backend_declared_by_two_children_is_called_once() {
    let server = MockServer::start().await;
    serve_noul(&server).await;
    let children = vec![
        Lane::new("first", vec![local("laya", &server)]).child(),
        Lane::new("second", vec![local("laya", &server)]).child(),
    ];
    let stack = vec!["smart".to_string()];
    let outcome = serve_capability_first(&children, &noul_request(json!("state")), &stack, 0)
        .await
        .expect("the shared backend answers");

    assert_eq!(outcome.served_by, "laya");
    assert_eq!(outcome.served_by_lane.as_deref(), Some("first"));
    assert_eq!(hits(&server).await, 1, "one physical call for one answer");
    assert_eq!(
        capability_pool(&children, &noul_request(json!("state")))
            .0
            .len(),
        1,
        "the pool must not list the same physical model twice"
    );
}

/// A request that no candidate fits is refused before dispatch, and the refusal names the
/// most specific reason rather than an outage.
#[tokio::test]
async fn a_request_that_fits_nothing_dispatches_nothing() {
    let local_server = MockServer::start().await;
    let span_server = MockServer::start().await;
    serve_noul(&local_server).await;
    serve_noul_alpha(&span_server).await;

    let children = vec![
        Lane::new("local-family", vec![local("jev", &local_server)]).child(),
        Lane::new("free-family", vec![span(&span_server)]).child(),
    ];
    let stack = vec!["smart".to_string()];
    let failure = serve_capability_first(&children, &oversize_score_request(), &stack, 0)
        .await
        .expect_err("nothing can serve an out-of-bounds score");

    assert_eq!(failure, DecisionFailure::Refused(DecisionSkip::Contract));
    assert_eq!(hits(&local_server).await, 0);
    assert_eq!(hits(&span_server).await, 0);
}

/// A spent daily allowance is a physical limit: it refuses before dispatch, does not spend
/// the peer's budget, and yields to a peer that is still authorised.
#[tokio::test]
async fn an_exhausted_daily_allowance_yields_to_an_authorised_peer() {
    let exhausted_server = MockServer::start().await;
    let peer_server = MockServer::start().await;
    serve_noul(&exhausted_server).await;
    serve_noul(&peer_server).await;

    let directory = scratch("smart-decision-allowance");
    std::fs::create_dir_all(&directory).expect("scratch dir");
    let allowance = DailyRequestAllowance::new(
        "local-free",
        ModelId::new("jev-style-model"),
        1,
        directory.join("local-free.json"),
    );
    allowance
        .try_reserve()
        .expect("the single unit is spent here");
    assert_eq!(allowance.remaining(), 0);

    let mut depleted = local("jev", &exhausted_server);
    depleted.budget = Some(Arc::new(DecisionBudget {
        allowance: Some(Arc::new(allowance)),
        pacing: None,
    }));

    let children = vec![
        Lane::new("local-family", vec![depleted]).child(),
        Lane::new("free-family", vec![local("peer", &peer_server)]).child(),
    ];
    let stack = vec!["smart".to_string()];
    // Rotation 0 enters at the depleted backend, which must refuse rather than answer.
    let outcome = serve_capability_first(&children, &noul_request(json!("state")), &stack, 0)
        .await
        .expect("an authorised peer is still eligible");

    assert_eq!(outcome.served_by, "peer");
    assert_eq!(
        hits(&exhausted_server).await,
        0,
        "an exhausted allowance must refuse before the call is sent"
    );
    assert_eq!(hits(&peer_server).await, 1);
    assert!(
        outcome
            .attempts
            .iter()
            .any(|(name, failure)| name == "local-family/jev"
                && *failure
                    == DecisionFailure::Serving {
                        reason: "daily_allowance_exhausted",
                        retryable: true,
                    }),
        "the trace must name the budget refusal: {:?}",
        outcome.attempts
    );
}

/// Ordered composition is a different policy and must keep its meaning: the first child
/// that can answer wins, whatever the rotation would have done.
#[tokio::test]
async fn ordered_composition_is_unchanged() {
    let first_server = MockServer::start().await;
    let second_server = MockServer::start().await;
    serve_noul(&first_server).await;
    serve_noul(&second_server).await;

    let first = Lane::new("first", vec![local("jev", &first_server)]);
    let second = Lane::new("second", vec![local("laya", &second_server)]);
    let lanes = BTreeMap::from([first.handle(), second.handle()]);
    let composition = BTreeMap::from([(
        "tier".to_string(),
        ComposedLane {
            children: vec!["first".to_string(), "second".to_string()],
            selection: LaneSelection::Ordered,
        },
    )]);

    for _ in 0..4 {
        let outcome = serve_named(&composition, &lanes, "tier", &noul_request(json!("state")))
            .await
            .expect("the first child answers")
            .expect("the tier is a composed lane");
        assert_eq!(outcome.served_by, "jev");
        assert_eq!(outcome.served_by_lane.as_deref(), Some("first"));
    }
    assert_eq!(hits(&first_server).await, 4);
    assert_eq!(hits(&second_server).await, 0);
}

/// The same tier declared capability-first reaches the second child as well, which is the
/// only difference between the two policies.
#[tokio::test]
async fn a_capability_first_tier_is_told_apart_from_an_ordered_one() {
    let span_server = MockServer::start().await;
    let local_server = MockServer::start().await;
    serve_noul_alpha(&span_server).await;
    serve_choice(&local_server).await;

    let span_child = Lane::new("free-family", vec![span(&span_server)]);
    let local_child = Lane::new("local-family", vec![local("jev", &local_server)]);
    let lanes = BTreeMap::from([span_child.handle(), local_child.handle()]);
    let composition = BTreeMap::from([(
        "smart".to_string(),
        ComposedLane {
            children: vec!["free-family".to_string(), "local-family".to_string()],
            selection: LaneSelection::CapabilityFirst,
        },
    )]);

    let outcome = serve_named(
        &composition,
        &lanes,
        "smart",
        &choice_request(json!("state")),
    )
    .await
    .expect("the local child answers")
    .expect("smart is a composed lane");

    assert_eq!(outcome.served_by, "jev");
    assert_eq!(outcome.served_by_lane.as_deref(), Some("local-family"));
    assert_eq!(
        hits(&span_server).await,
        0,
        "ordered selection would have asked SPAN first and failed the request"
    );
}

// --- configuration ----------------------------------------------------------

fn config(composition: &str) -> String {
    format!(
        r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "SMART_DECISION_CONFIG_KEY"

[targets.span]
id = "span/model"
llm_client = "primary"
decision_transport = "open_router_alpha"
decision_path = "/api/alpha/decisions"
decision_base_url = "https://alpha.test"
decision_api_key_env = "SMART_DECISION_CONFIG_KEY"
supported_types = {{ noul = true, choice = false, score = false }}

[targets.jev]
id = "jev/model"
llm_client = "primary"
decision_transport = "system_one"
decision_base_url = "https://systemone.test"
decision_api_key_env = "SMART_DECISION_CONFIG_KEY"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[decision_lanes.free-family]
targets = ["span"]

[decision_lanes.local-family]
targets = ["jev"]

{composition}
"#
    )
}

#[test]
fn a_composed_lane_defaults_to_ordered_selection() {
    unsafe { std::env::set_var("SMART_DECISION_CONFIG_KEY", "config-dummy") };
    let runner = Runner::from_toml(&config(
        "[decision_lane_composition.tier]\nchildren = [\"free-family\", \"local-family\"]\n",
    ))
    .expect("the historic composition still loads");
    assert_eq!(
        runner.decision_lane_selection("tier"),
        Some(LaneSelection::Ordered),
        "a lane that names no policy keeps the ordered meaning"
    );
}

#[test]
fn a_composed_lane_can_declare_capability_first_selection() {
    unsafe { std::env::set_var("SMART_DECISION_CONFIG_KEY", "config-dummy") };
    let runner = Runner::from_toml(&config(
        "[decision_lane_composition.smart]\nchildren = [\"free-family\", \"local-family\"]\nselection = \"capability_first\"\n",
    ))
    .expect("the capability-first composition loads");
    assert_eq!(
        runner.decision_lane_selection("smart"),
        Some(LaneSelection::CapabilityFirst)
    );
    assert_eq!(
        runner.decision_lane_children("smart"),
        Some(["free-family".to_string(), "local-family".to_string()].as_slice())
    );
}

#[test]
fn an_unknown_selection_policy_is_refused_at_load() {
    unsafe { std::env::set_var("SMART_DECISION_CONFIG_KEY", "config-dummy") };
    let message = match Runner::from_toml(&config(
        "[decision_lane_composition.smart]\nchildren = [\"free-family\"]\nselection = \"cheapest_first\"\n",
    )) {
        Ok(_) => panic!("an invented policy must not silently become the default"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("selection") || message.contains("unknown variant"),
        "the refusal must name the policy: {message}"
    );
}

// --- helpers ----------------------------------------------------------------

fn scratch(label: &str) -> PathBuf {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/var/tmp".to_string());
    let unique = format!(
        "{label}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0)
    );
    let path = PathBuf::from(base).join(unique);
    let _ = std::fs::remove_dir_all(&path);
    path
}
