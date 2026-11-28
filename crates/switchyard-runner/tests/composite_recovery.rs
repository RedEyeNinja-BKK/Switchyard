// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Composite recovery, exercised through the real composite executor.
//!
//! A composite is only resilient if a failure inside one child is read for what it says
//! about the next one. These tests drive `Runner::execute_route` over registered
//! composites, so the crossing decision (`composition.rs`, `crossable`) runs for real
//! instead of being re-modelled in the assertion.
//!
//! Every stub counts its own calls: "the leg was skipped" and "the leg answered and was
//! ignored" are different claims, and only the first one is a guarantee about cost.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use libsy::RuntimeModels;
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_request, text_response,
};
use switchyard_runner::{
    AlgorithmSpec, CompositeCrossPolicy, ModelCapabilities, Route, RouteReasoningPolicy, Runner,
    RunnerError,
};

const REPLAY_REFUSAL: &str = r#"{"error":{"message":"The `reasoning_text` in the thinking mode must be passed back to the API.","type":"upstream_error","code":"invalid_request_error"}}"#;
const GENERIC_REFUSAL: &str =
    r#"{"error":{"message":"Invalid request body","type":"invalid_request_error"}}"#;

/// What a stub answers on every call.
#[derive(Clone)]
enum Behaviour {
    Serve,
    /// Transport-level unavailability: another child may still answer.
    Unavailable,
    /// The reconciled provider reading refused the send. `group` is the authoritative
    /// pool identity; the runner declares the same string per target.
    ProviderAllowance {
        model: &'static str,
        group: Option<&'static str>,
    },
    /// The local daily counter refused the send.
    DailyAllowance {
        model: &'static str,
        group: Option<&'static str>,
    },
    /// A provider 400 that names a missing replay artifact: another candidate may serve.
    ReplayRefusal,
    /// A provider 400 the caller must fix: terminal.
    Malformed,
    /// A plain 429: no credential block, so only a class policy may cross.
    RateLimited,
}

struct Stub {
    model: &'static str,
    calls: Arc<AtomicUsize>,
    behaviour: Behaviour,
}

#[async_trait]
impl RoutedLlmClient for Stub {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match &self.behaviour {
            Behaviour::Serve => Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    request.llm_request.model.clone(),
                    "served",
                )),
                metadata: None,
                upstream_headers: Default::default(),
            }),
            Behaviour::Unavailable => Err(LlmClientError::TemporarilyUnavailable),
            Behaviour::ProviderAllowance { model, group } => {
                Err(LlmClientError::ProviderAllowanceExhausted {
                    model: ModelId::from(*model),
                    limit: 1_000,
                    group: group.map(ToString::to_string),
                })
            }
            Behaviour::DailyAllowance { model, group } => {
                Err(LlmClientError::DailyRequestAllowanceExhausted {
                    model: ModelId::from(*model),
                    limit: 1_000,
                    group: group.map(ToString::to_string),
                })
            }
            Behaviour::ReplayRefusal => Err(LlmClientError::UpstreamHttp {
                status: http::StatusCode::BAD_REQUEST,
                body: REPLAY_REFUSAL.to_string(),
            }),
            Behaviour::Malformed => Err(LlmClientError::UpstreamHttp {
                status: http::StatusCode::BAD_REQUEST,
                body: GENERIC_REFUSAL.to_string(),
            }),
            Behaviour::RateLimited => Err(LlmClientError::UpstreamHttp {
                status: http::StatusCode::TOO_MANY_REQUESTS,
                body: r#"{"error":{"message":"rate limit exceeded"}}"#.to_string(),
            }),
        }
    }
}

/// One leaf route over `target` plus `candidates`, all resolving to stubs.
fn leaf(
    name: &str,
    target: &'static str,
    candidates: &[&'static str],
    stubs: Vec<Arc<Stub>>,
) -> (ModelId, Route) {
    let mut names = vec![target];
    names.extend_from_slice(candidates);
    let algorithm = AlgorithmSpec::Passthrough {
        target: target.to_string(),
        candidates: candidates.iter().map(|c| (*c).to_string()).collect(),
        subagents: None,
    }
    .build(
        name,
        &names
            .iter()
            .map(|n| ((*n).to_string(), ModelId::from(*n)))
            .collect::<BTreeMap<_, _>>(),
    )
    .expect("leaf route builds");

    let clients: HashMap<ModelId, Arc<dyn RoutedLlmClient>> = stubs
        .into_iter()
        .map(|stub| (ModelId::from(stub.model), stub as Arc<dyn RoutedLlmClient>))
        .collect();
    (
        ModelId::from(name),
        Route::new(
            algorithm,
            ClientRouter::new(clients),
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new(
                [(
                    Category::Any,
                    names.iter().map(|n| ModelId::from(*n)).collect(),
                )]
                .into(),
            ),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        ),
    )
}

/// One leaf route whose candidates share an account-scoped allowance declaration.
fn leaf_with_groups(
    name: &str,
    target: &'static str,
    candidates: &[&'static str],
    stubs: Vec<Arc<Stub>>,
    groups: &[(&'static str, &'static str)],
) -> (ModelId, Route) {
    let mut names = vec![target];
    names.extend_from_slice(candidates);
    let algorithm = AlgorithmSpec::Passthrough {
        target: target.to_string(),
        candidates: candidates.iter().map(|c| (*c).to_string()).collect(),
        subagents: None,
    }
    .build(
        name,
        &names
            .iter()
            .map(|n| ((*n).to_string(), ModelId::from(*n)))
            .collect::<BTreeMap<_, _>>(),
    )
    .expect("leaf route builds");

    let clients: HashMap<ModelId, Arc<dyn RoutedLlmClient>> = stubs
        .into_iter()
        .map(|stub| (ModelId::from(stub.model), stub as Arc<dyn RoutedLlmClient>))
        .collect();
    let router = ClientRouter::new(clients).with_target_allowance_groups(
        groups
            .iter()
            .map(|(model, group)| (ModelId::from(*model), (*group).to_string()))
            .collect(),
    );
    (
        ModelId::from(name),
        Route::new(
            algorithm,
            router,
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new(
                [(
                    Category::Any,
                    names.iter().map(|n| ModelId::from(*n)).collect(),
                )]
                .into(),
            ),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        ),
    )
}

fn stub(model: &'static str, behaviour: Behaviour) -> (Arc<Stub>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        Arc::new(Stub {
            model,
            calls: Arc::clone(&calls),
            behaviour,
        }),
        calls,
    )
}

/// The target that actually answered, as the response envelope records it.
///
/// `RunOutput::selected_model` is the algorithm's first pick, which is not the same claim:
/// after a candidate fails the run falls through, so a test about recovery must read the
/// served target or it cannot tell "the sibling answered" from "the first pick was
/// reported".
fn served(output: &switchyard_runner::composition::ExecutionOutput) -> &str {
    output
        .output
        .response
        .served_model()
        .expect("a served response names its target")
        .as_str()
}

/// Records the credential configuration resolves for a route's target.
///
/// A credential-level block is only actionable when configuration names the credential
/// serving the failing slug, so a composite test that expects a crossing must state it.
fn with_credential(
    (id, route): (ModelId, Route),
    model: &str,
    credential: &str,
) -> (ModelId, Route) {
    (
        id,
        route.with_target_credentials(HashMap::from([(
            ModelId::from(model),
            credential.to_string(),
        )])),
    )
}

fn request() -> Request {
    let raw = serde_json::json!({
        "messages": [{"role": "user", "content": "ping"}],
    });
    Request {
        llm_request: text_request(Some("composite".to_string()), "ping"),
        raw_request: Some(raw),
        ..Request::default()
    }
}

/// A composite descends when a child cannot serve at all.
///
/// The failing child is contacted once and the sibling answers; without the call counts,
/// "the answer came from the second child" would also be satisfied by never trying the
/// first one.
#[tokio::test]
async fn a_failed_child_hands_the_request_to_the_next_child() {
    let (google, google_calls) = stub("google/dead", Behaviour::Unavailable);
    let (deepseek, deepseek_calls) = stub("deepseek/ok", Behaviour::Serve);
    let runner = Runner::new(vec![
        leaf("google", "google/dead", &[], vec![google]),
        leaf("deepseek", "deepseek/ok", &[], vec![deepseek]),
    ])
    .with_composite(
        "ladder".to_string(),
        vec!["google".into(), "deepseek".into()],
    );

    let output = runner
        .execute_route("ladder", request(), None)
        .await
        .expect("the second child serves");

    assert_eq!(
        google_calls.load(Ordering::SeqCst),
        1,
        "the first child is tried once"
    );
    assert_eq!(
        deepseek_calls.load(Ordering::SeqCst),
        1,
        "the second child answers"
    );
    assert_eq!(output.output.selected_model.as_str(), "deepseek/ok");
}

/// Free allowance spent, Google down: the metered child still serves.
///
/// This is the measured live shape (`smart-route`): every earlier child refuses for a
/// reason that says nothing about the next one's allowance.
#[tokio::test]
async fn a_free_pool_that_is_spent_does_not_end_the_ladder() {
    let (google, google_calls) = stub("google/dead", Behaviour::Unavailable);
    let (free, free_calls) = stub(
        "vendor/free-a:free",
        Behaviour::ProviderAllowance {
            model: "vendor/free-a:free",
            group: Some("free-owner"),
        },
    );
    let (deepseek, deepseek_calls) = stub("deepseek/ok", Behaviour::Serve);
    let runner = Runner::new(vec![
        leaf("google", "google/dead", &[], vec![google]),
        leaf("free", "vendor/free-a:free", &[], vec![free]),
        leaf("deepseek", "deepseek/ok", &[], vec![deepseek]),
    ])
    .with_composite(
        "ladder".to_string(),
        vec!["google".into(), "free".into(), "deepseek".into()],
    );

    let output = runner
        .execute_route("ladder", request(), None)
        .await
        .expect("a child on another allowance serves");

    assert_eq!(google_calls.load(Ordering::SeqCst), 1);
    assert_eq!(free_calls.load(Ordering::SeqCst), 1);
    assert_eq!(deepseek_calls.load(Ordering::SeqCst), 1);
    assert_eq!(output.output.selected_model.as_str(), "deepseek/ok");
}

/// Inside one child, the second candidate answers and the first is tried exactly once.
#[tokio::test]
async fn a_failing_candidate_advances_to_its_sibling_once() {
    let (first, first_calls) = stub("vendor/a", Behaviour::Unavailable);
    let (second, second_calls) = stub("vendor/b", Behaviour::Serve);
    let runner = Runner::new(vec![leaf(
        "solo",
        "vendor/a",
        &["vendor/b"],
        vec![first, second],
    )]);

    let output = runner
        .execute_route("solo", request(), None)
        .await
        .expect("the sibling serves");
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    // The route reports the algorithm's first pick; the envelope names the target that
    // actually answered, which is the claim this test makes.
    assert_eq!(served(&output), "vendor/b");
}

/// A spent account is one fact: its second member must not be asked.
///
/// Without the group declaration the walk advances one leg at a time — the live
/// `smart-route` defect, where one account produced three refusals at 0 ms — and the
/// call counts here would read (1, 1, 1).
#[tokio::test]
async fn a_spent_account_skips_its_own_members() {
    let (first, first_calls) = stub(
        "vendor/free-a",
        Behaviour::DailyAllowance {
            model: "vendor/free-a",
            group: Some("owner-pool"),
        },
    );
    let (second, second_calls) = stub("vendor/free-b", Behaviour::Serve);
    let (third, third_calls) = stub("paid/ok", Behaviour::Serve);
    let runner = Runner::new(vec![leaf_with_groups(
        "solo",
        "vendor/free-a",
        &["vendor/free-b", "paid/ok"],
        vec![first, second, third],
        &[
            ("vendor/free-a", "owner-pool"),
            ("vendor/free-b", "owner-pool"),
            ("paid/ok", "paid-pool"),
        ],
    )]);

    let output = runner
        .execute_route("solo", request(), None)
        .await
        .expect("the leg on another pool serves");
    assert_eq!(
        first_calls.load(Ordering::SeqCst),
        1,
        "the spent member is asked once"
    );
    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        0,
        "the sibling on the same spent account must not be dispatched"
    );
    assert_eq!(third_calls.load(Ordering::SeqCst), 1);
    assert_eq!(served(&output), "paid/ok");
}

/// The `crossable` path with no test before this one: a free-scope block crosses.
///
/// Paired control: under the same `credential_block` policy a plain 429 carries no block,
/// so it must NOT spend the paid child.
#[tokio::test]
async fn a_free_scope_block_crosses_to_the_paid_child_and_a_throttle_does_not() {
    let (blocked, blocked_calls) = stub(
        "vendor/free-x:free",
        Behaviour::ProviderAllowance {
            model: "vendor/free-x:free",
            group: Some("free-owner"),
        },
    );
    let (paid, paid_calls) = stub("paid/ok", Behaviour::Serve);
    let runner = Runner::new(vec![
        with_credential(
            leaf("free", "vendor/free-x:free", &[], vec![blocked]),
            "vendor/free-x:free",
            "free-account",
        ),
        with_credential(
            leaf("paid", "paid/ok", &[], vec![paid]),
            "paid/ok",
            "paid-account",
        ),
    ])
    .with_composite("route".to_string(), vec!["free".into(), "paid".into()])
    .with_composite_cross_policy("route".to_string(), CompositeCrossPolicy::CredentialBlock);

    let output = runner
        .execute_route("route", request(), None)
        .await
        .expect("a free-scope block crosses to the paid child");
    assert_eq!(paid_calls.load(Ordering::SeqCst), 1);
    assert_eq!(blocked_calls.load(Ordering::SeqCst), 1);
    assert_eq!(output.output.selected_model.as_str(), "paid/ok");
    let composition = output
        .composition
        .expect("a composite execution carries its context");
    assert!(composition.crossed_child, "the crossing is recorded");

    // Control: the same policy, a plain 429 instead of a credential block.
    let (throttled, throttled_calls) = stub("vendor/free-x:free", Behaviour::RateLimited);
    let (paid, paid_calls) = stub("paid/ok", Behaviour::Serve);
    let runner = Runner::new(vec![
        with_credential(
            leaf("free", "vendor/free-x:free", &[], vec![throttled]),
            "vendor/free-x:free",
            "free-account",
        ),
        with_credential(
            leaf("paid", "paid/ok", &[], vec![paid]),
            "paid/ok",
            "paid-account",
        ),
    ])
    .with_composite("route".to_string(), vec!["free".into(), "paid".into()])
    .with_composite_cross_policy("route".to_string(), CompositeCrossPolicy::CredentialBlock);

    let error = runner
        .execute_route("route", request(), None)
        .await
        .err()
        .expect("a bare throttle is not a credential block");
    assert!(
        error.to_string().contains("rate limit"),
        "the child's own refusal surfaces: {error}"
    );
    assert_eq!(throttled_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        paid_calls.load(Ordering::SeqCst),
        0,
        "a throttle must not spend the paid child"
    );
}

/// A provider 400 that names a missing replay artifact is routable; a malformed one is not.
#[tokio::test]
async fn a_replay_requirement_refusal_advances_but_a_malformed_request_does_not() {
    let (refusing, refusing_calls) = stub("deepseek/flash", Behaviour::ReplayRefusal);
    let (sibling, sibling_calls) = stub("paid/ok", Behaviour::Serve);
    let runner = Runner::new(vec![leaf(
        "solo",
        "deepseek/flash",
        &["paid/ok"],
        vec![refusing, sibling],
    )]);

    let output = runner
        .execute_route("solo", request(), None)
        .await
        .expect("the candidate with another continuation contract serves");
    assert_eq!(refusing_calls.load(Ordering::SeqCst), 1);
    assert_eq!(sibling_calls.load(Ordering::SeqCst), 1);
    assert_eq!(served(&output), "paid/ok");

    // Control: a generic 400 is the caller's to fix and must not be handed on.
    let (malformed, malformed_calls) = stub("deepseek/flash", Behaviour::Malformed);
    let (sibling, sibling_calls) = stub("paid/ok", Behaviour::Serve);
    let runner = Runner::new(vec![leaf(
        "solo",
        "deepseek/flash",
        &["paid/ok"],
        vec![malformed, sibling],
    )]);
    let error = runner
        .execute_route("solo", request(), None)
        .await
        .err()
        .expect("a malformed request stays terminal");
    assert_eq!(malformed_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        sibling_calls.load(Ordering::SeqCst),
        0,
        "a malformed request must not be replayed elsewhere"
    );
    assert!(
        error.to_string().to_lowercase().contains("invalid request"),
        "the provider's own diagnosis surfaces: {error}"
    );
}

/// When every child is out of allowance, the composite reports the allowance, not a 400.
#[tokio::test]
async fn an_all_exhausted_composite_reports_the_allowance_it_hit() {
    let (only, calls) = stub(
        "vendor/free-a:free",
        Behaviour::ProviderAllowance {
            model: "vendor/free-a:free",
            group: Some("free-owner"),
        },
    );
    let runner = Runner::new(vec![leaf("free", "vendor/free-a:free", &[], vec![only])])
        .with_composite("route".to_string(), vec!["free".into()]);
    assert!(
        runner.composites().contains_key("route"),
        "the composite is registered"
    );

    let error = runner
        .execute_route("route", request(), None)
        .await
        .err()
        .expect("no child can serve");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        matches!(error, RunnerError::Algorithm(_) | RunnerError::Client(_)),
        "the exhausted allowance is reported as itself, not as a request fault: {error}"
    );
    let text = error.to_string().to_lowercase();
    assert!(
        text.contains("allowance") || text.contains("spent") || text.contains("unavailable"),
        "the terminal names what was hit: {error}"
    );
}
