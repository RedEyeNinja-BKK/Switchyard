// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adaptive bounded behaviour a Gemma/Lite release depends on.
//!
//! Four properties are exercised here, each with a control that fails without it:
//!
//! * a private or unknown payload skips a Google leg inside a composite and reaches the
//!   strict non-training leg, while a directly addressed Google leaf still refuses;
//! * a request too large for a target's physical window or per-minute input budget is not
//!   sent to it, so an aggregate context claim stays truthful;
//! * measured selection reorders candidates only after it has evidence, and only among the
//!   candidates the hard gates already admitted;
//! * a route may not advertise capacity no declared target can serve.
//!
//! Every stub here counts its calls, because "the leg was skipped" and "the leg answered and
//! was ignored" are different claims and only the first one is a privacy guarantee.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

use async_trait::async_trait;
use libsy::RuntimeModels;
use switchyard_llm_client::{CapabilityFacts, ClientRouter};
use switchyard_protocol::{
    Category, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_request, text_response,
};
use switchyard_runner::{
    AlgorithmSpec, MeasuredSelector, ModelCapabilities, Route, RouteReasoningPolicy, Runner,
    RunnerError,
};

/// Records that it was asked and how big the request was.
struct CountingClient {
    calls: Arc<AtomicUsize>,
    /// Set when the stub should behave like an unavailable backend.
    failing: bool,
}

#[async_trait]
impl RoutedLlmClient for CountingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.failing {
            return Err(LlmClientError::TemporarilyUnavailable);
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "served",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

const TEST_KEY: &[u8] = b"switchyard-adaptive-bounded-test-key";

fn clearance_key() -> Arc<switchyard_runner::egress::ClearanceKey> {
    Arc::new(switchyard_runner::egress::ClearanceKey::new(TEST_KEY))
}

/// Signs a clearance prefix the way a caller holding the key would.
fn sign(prefix: &str) -> String {
    use hmac::Mac;
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(TEST_KEY).expect("any key length");
    mac.update(prefix.as_bytes());
    let signature: String = mac
        .finalize()
        .into_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{prefix};mac={signature}")
}

fn today_header() -> String {
    let day = switchyard_llm_client::allowance::provider_day(SystemTime::now());
    let label = switchyard_llm_client::allowance::provider_day_label(day);
    sign(&format!("v1;class=public;scope=assembled;day={label}"))
}

fn request(clearance: Option<&str>, text: &str) -> Request {
    // The raw body travels with the decoded request, because the token estimate and the
    // media checks read the body a caller actually sent rather than the decoded form.
    let raw = serde_json::json!({
        "messages": [{"role": "user", "content": text}],
    });
    let mut request = Request {
        llm_request: text_request(Some("routed".to_string()), text),
        raw_request: Some(raw),
        ..Request::default()
    };
    if let Some(clearance) = clearance {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            switchyard_runner::CLEARANCE_HEADER,
            http::HeaderValue::from_str(clearance).expect("valid test header"),
        );
        request.metadata = Some(switchyard_protocol::Metadata {
            http_headers: Some(headers),
            ..Default::default()
        });
    }
    request
}

/// A one-target leaf route.
fn leaf(
    name: &str,
    calls: &Arc<AtomicUsize>,
    facts: CapabilityFacts,
    capabilities: ModelCapabilities,
    failing: bool,
) -> (ModelId, Route) {
    let target = format!("{name}-target");
    let algorithm = AlgorithmSpec::Passthrough {
        target: target.clone(),
        candidates: Vec::new(),
        subagents: None,
    }
    .build(
        name,
        &BTreeMap::from([(target.clone(), ModelId::from(target.as_str()))]),
    )
    .expect("single target builds");
    let clients = ClientRouter::new(
        BTreeMap::from([(
            ModelId::from(target.as_str()),
            Arc::new(CountingClient {
                calls: Arc::clone(calls),
                failing,
            }) as Arc<dyn RoutedLlmClient>,
        )])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([(ModelId::from(target.as_str()), facts)]));
    (
        ModelId::from(name),
        Route::new(
            algorithm,
            clients,
            None,
            capabilities,
            None,
            None,
            Vec::new(),
            RuntimeModels::new([(Category::Any, vec![ModelId::from(target.as_str())])].into()),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        )
        .with_clearance_key(clearance_key()),
    )
}

fn gated() -> CapabilityFacts {
    CapabilityFacts {
        requires_public_payload: true,
        ..CapabilityFacts::default()
    }
}

fn open() -> CapabilityFacts {
    CapabilityFacts::default()
}

// ---------------------------------------------------------------------------
// Privacy: a composite skips the leg it may not reach
// ---------------------------------------------------------------------------

/// The owner's requested branch: an uncleared payload starts after Google, not at it.
///
/// This is the behaviour the deployed `smart-nt-bounded-trial` lacks — it terminates with
/// the child's refusal instead — so the assertion is on both legs' call counters: Google
/// must be at zero and the strict leg must be exactly one.
#[tokio::test]
async fn an_uncleared_payload_skips_a_gated_child_and_serves_the_next_leg() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    let output = runner
        .execute_route("bounded", request(None, "private material"), None)
        .await
        .expect("the strict leg can serve a private payload");

    assert_eq!(
        google_calls.load(Ordering::SeqCst),
        0,
        "the gated leg was contacted"
    );
    assert_eq!(
        strict_calls.load(Ordering::SeqCst),
        1,
        "the strict leg was not used"
    );
    assert!(
        output.output.response.served_model().is_some(),
        "the response must name what served it"
    );
}

/// The deployed `smart-nt-bounded-trial` shape, asserted rather than assumed.
///
/// Four children in the order the production manifest states: two Google legs, then the
/// strict non-training OpenAI leg, then the strict non-training DeepSeek leg. An uncleared
/// payload must start at leg three, and a cleared one must start at leg one -- without the
/// second assertion the first could be satisfied by never using Google at all.
#[tokio::test]
async fn the_deployed_bounded_trial_shape_starts_after_google_for_an_uncleared_payload() {
    async fn trial() -> ([Arc<AtomicUsize>; 4], Runner) {
        let counters: [Arc<AtomicUsize>; 4] =
            std::array::from_fn(|_| Arc::new(AtomicUsize::new(0)));
        let runner = Runner::new(vec![
            leaf(
                "google-a",
                &counters[0],
                gated(),
                ModelCapabilities::default(),
                false,
            ),
            leaf(
                "google-b",
                &counters[1],
                gated(),
                ModelCapabilities::default(),
                false,
            ),
            leaf(
                "luna-nt",
                &counters[2],
                open(),
                ModelCapabilities::default(),
                false,
            ),
            leaf(
                "deepseek-nt",
                &counters[3],
                open(),
                ModelCapabilities::default(),
                false,
            ),
        ])
        .with_composite(
            "smart-nt-bounded-trial".to_string(),
            vec![
                "google-a".to_string(),
                "google-b".to_string(),
                "luna-nt".to_string(),
                "deepseek-nt".to_string(),
            ],
        );
        (counters, runner)
    }

    let (counters, runner) = trial().await;
    runner
        .execute_route(
            "smart-nt-bounded-trial",
            request(None, "private material"),
            None,
        )
        .await
        .expect("the strict leg can serve a private payload");
    assert_eq!(
        (
            counters[0].load(Ordering::SeqCst),
            counters[1].load(Ordering::SeqCst)
        ),
        (0, 0),
        "an uncleared payload reached a Google leg"
    );
    assert_eq!(
        counters[2].load(Ordering::SeqCst),
        1,
        "the first strict leg did not serve it"
    );
    assert_eq!(
        counters[3].load(Ordering::SeqCst),
        0,
        "the second strict leg was spent"
    );

    // Control: with a valid clearance the same ladder starts at Google, so the skip above
    // is conditional on the payload rather than a lane that never uses the free legs.
    let (counters, runner) = trial().await;
    runner
        .execute_route(
            "smart-nt-bounded-trial",
            request(Some(&today_header()), "publishable material"),
            None,
        )
        .await
        .expect("a cleared payload may use the first leg");
    assert_eq!(
        counters[0].load(Ordering::SeqCst),
        1,
        "the cleared payload skipped Google"
    );
    assert_eq!(
        counters[2].load(Ordering::SeqCst),
        0,
        "a paid-tier leg was spent unnecessarily"
    );
}

/// The same request addressed straight at the gated leaf is still refused, with no call.
///
/// Without this control the skip above could be implemented by weakening the gate, which is
/// the failure mode that would matter most.
#[tokio::test]
async fn a_directly_addressed_gated_leaf_still_refuses_and_is_never_called() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "google",
        &google_calls,
        gated(),
        ModelCapabilities::default(),
        false,
    )]);

    let error = runner
        .execute_route("google", request(None, "private material"), None)
        .await
        .err()
        .expect("a direct address to a gated leaf must not leave");

    match error {
        RunnerError::PrivacyRefused { refusals, .. } => {
            let codes: Vec<&str> = refusals.iter().map(|refusal| refusal.as_str()).collect();
            assert_eq!(codes, vec!["no_clearance"]);
        }
        other => panic!("expected PrivacyRefused, got {other:?}"),
    }
    assert_eq!(google_calls.load(Ordering::SeqCst), 0);
}

/// A cleared payload is served by the gated leg, so the skip is about clearance, not about
/// the leg being permanently unusable.
#[tokio::test]
async fn a_cleared_payload_is_served_by_the_gated_child() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    runner
        .execute_route(
            "bounded",
            request(Some(&today_header()), "public material"),
            None,
        )
        .await
        .expect("a cleared payload may use the gated leg");

    assert_eq!(google_calls.load(Ordering::SeqCst), 1);
    assert_eq!(strict_calls.load(Ordering::SeqCst), 0);
}

/// A gated child that is contacted for a cleared payload and then fails still crosses, so the
/// skip did not replace the existing failure handling.
#[tokio::test]
async fn a_failing_gated_child_still_crosses_on_a_cleared_payload() {
    let google_calls = Arc::new(AtomicUsize::new(0));
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "google",
            &google_calls,
            gated(),
            ModelCapabilities::default(),
            true,
        ),
        leaf(
            "strict",
            &strict_calls,
            open(),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["google".to_string(), "strict".to_string()],
    );

    runner
        .execute_route(
            "bounded",
            request(Some(&today_header()), "public material"),
            None,
        )
        .await
        .expect("an unavailable gated leg crosses to the next child");

    assert_eq!(google_calls.load(Ordering::SeqCst), 1);
    assert_eq!(strict_calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Physical fit
// ---------------------------------------------------------------------------

fn physical(context_window: u64, max_input_tokens: Option<u64>) -> CapabilityFacts {
    CapabilityFacts {
        context_window: Some(context_window),
        max_input_tokens,
        ..CapabilityFacts::default()
    }
}

fn long_request(words: usize) -> Request {
    // Four ASCII characters per estimated token, so the estimate is close to `words / 4`.
    request(Some(&today_header()), &"alpha ".repeat(words))
}

/// A request larger than one target's physical window is not dispatched to it, and the
/// wider target serves it.
#[tokio::test]
async fn a_request_beyond_a_targets_window_goes_to_the_wider_target() {
    let narrow_calls = Arc::new(AtomicUsize::new(0));
    let wide_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "narrow",
            &narrow_calls,
            physical(1_024, None),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "wide",
            &wide_calls,
            physical(262_144, None),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["narrow".to_string(), "wide".to_string()],
    );

    runner
        .execute_route("bounded", long_request(4_000), None)
        .await
        .expect("the wider target can hold the request");

    assert_eq!(
        narrow_calls.load(Ordering::SeqCst),
        0,
        "a request that cannot fit was sent"
    );
    assert_eq!(wide_calls.load(Ordering::SeqCst), 1);
}

/// When nothing in the route can hold the request the refusal must be immediate and local.
///
/// This is what a 16,000-token bounded lane does with a 40,000-token prompt: it must not
/// send it to Gemma and let the provider reject it, because the call would still be counted
/// against the day's budget while returning nothing.
#[tokio::test]
async fn a_request_no_candidate_can_hold_is_refused_without_a_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "gemma",
        &calls,
        physical(262_144, Some(16_000)),
        ModelCapabilities::default(),
        false,
    )]);

    // ~120k estimated tokens: inside the 262,144-token window, far above the 16,000-token
    // per-minute allowance, so the input ceiling is the limit that binds.
    let error = runner
        .execute_route("gemma", long_request(80_000), None)
        .await
        .err()
        .expect("a request above every candidate's ceiling must be refused");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the provider was contacted for a request it had already been ruled unable to serve"
    );
    assert!(
        matches!(
            error,
            RunnerError::UnsupportedCapability {
                capability: "input_tokens_per_minute",
                ..
            }
        ),
        "the refusal must name the limit that bound it: {error}"
    );
}

/// A per-request input ceiling smaller than the window — the Gemma case, where a
/// 262,144-token model is allocated 16,000 input tokens per minute — excludes the target
/// even though the window would fit.
#[tokio::test]
async fn a_request_beyond_the_input_budget_is_not_sent_to_that_target() {
    let gemma_calls = Arc::new(AtomicUsize::new(0));
    let lite_calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![
        leaf(
            "gemma",
            &gemma_calls,
            physical(262_144, Some(16_000)),
            ModelCapabilities::default(),
            false,
        ),
        leaf(
            "lite",
            &lite_calls,
            physical(1_048_576, None),
            ModelCapabilities::default(),
            false,
        ),
    ])
    .with_composite(
        "bounded".to_string(),
        vec!["gemma".to_string(), "lite".to_string()],
    );

    // ~50k tokens: fits the 262,144 window, far above the 16,000-token minute.
    runner
        .execute_route("bounded", long_request(200_000), None)
        .await
        .expect("the wider budget serves it");

    assert_eq!(gemma_calls.load(Ordering::SeqCst), 0);
    assert_eq!(lite_calls.load(Ordering::SeqCst), 1);
}

/// The caller's output reservation counts against the target's fit, because a window that
/// holds the input but not the answer rejects the request upstream after dispatch.
#[tokio::test]
async fn an_output_reservation_counts_against_the_physical_window() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "small",
        &calls,
        physical(1_000, None),
        ModelCapabilities::default(),
        false,
    )]);

    // ~400 estimated input tokens, which the window holds comfortably; the 900-token
    // reservation is what does not fit. Without that separation the test would pass on the
    // input alone and prove nothing about the reservation.
    let mut sized = request(Some(&today_header()), &"alpha ".repeat(260));
    sized.llm_request.output.max_output_tokens = Some(900);

    let error = runner
        .execute_route("small", sized, None)
        .await
        .err()
        .expect("input plus reservation exceeds the window");

    assert!(
        matches!(error, RunnerError::UnsupportedCapability { .. }),
        "{error:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// A target that declares nothing is unconstrained, so a manifest written before the field
/// existed keeps its behaviour.
#[tokio::test]
async fn an_undeclared_window_admits_every_request() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = Runner::new(vec![leaf(
        "legacy",
        &calls,
        open(),
        ModelCapabilities::default(),
        false,
    )]);

    runner
        .execute_route("legacy", long_request(400_000), None)
        .await
        .expect("an undeclared window is not a refusal");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------------------
// Measured selection
// ---------------------------------------------------------------------------

/// Builds a two-target leaf route whose candidates are both served, scored in the given
/// declared order, with the second leg shaped to answer faster.
fn measured_route(
    first_calls: &Arc<AtomicUsize>,
    second_calls: &Arc<AtomicUsize>,
    failing_first: bool,
) -> (ModelId, Route) {
    let algorithm = AlgorithmSpec::Passthrough {
        target: "first-target".to_string(),
        candidates: vec!["second-target".to_string()],
        subagents: None,
    }
    .build(
        "measured",
        &BTreeMap::from([
            ("first-target".to_string(), ModelId::from("first-target")),
            ("second-target".to_string(), ModelId::from("second-target")),
        ]),
    )
    .expect("two targets build");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (
                ModelId::from("first-target"),
                Arc::new(CountingClient {
                    calls: Arc::clone(first_calls),
                    failing: failing_first,
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("second-target"),
                Arc::new(CountingClient {
                    calls: Arc::clone(second_calls),
                    failing: false,
                }) as Arc<dyn RoutedLlmClient>,
            ),
        ])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([
        (ModelId::from("first-target"), open()),
        (ModelId::from("second-target"), open()),
    ]));
    (
        ModelId::from("measured"),
        Route::new(
            algorithm,
            clients,
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new(
                [(
                    Category::Any,
                    vec![
                        ModelId::from("first-target"),
                        ModelId::from("second-target"),
                    ],
                )]
                .into(),
            ),
            RouteReasoningPolicy::Inherit,
            None,
            None,
        )
        .with_clearance_key(clearance_key())
        .with_measured_selection(MeasuredSelector::new()),
    )
}

/// A cold route follows the declared order: measurement has nothing to say yet.
#[tokio::test]
async fn a_cold_measured_route_still_uses_the_declared_order() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let (model, route) = measured_route(&first, &second, false);
    let runner = Runner::new(vec![(model, route)]);
    runner
        .execute_route("measured", request(Some(&today_header()), "hello"), None)
        .await
        .expect("served");
    assert_eq!(
        first.load(Ordering::SeqCst),
        1,
        "the declared first leg must be tried"
    );
    assert_eq!(second.load(Ordering::SeqCst), 0);
}

/// Once the declared first leg has failed enough to be measured as unreliable, the route
/// prefers the second leg — and reaches it without a provider failure on this request.
#[tokio::test]
async fn a_measured_route_prefers_the_leg_that_has_been_completing() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let (_, route) = measured_route(&first, &second, true);

    // Three requests where the declared first leg fails and the second serves it: that is
    // the evidence the selector needs, and it is produced by requests, not by setup.
    for _ in 0..3 {
        route
            .execute(request(Some(&today_header()), "warm"), None)
            .await
            .expect("the second leg serves");
    }
    let first_calls_before = first.load(Ordering::SeqCst);
    let second_calls_before = second.load(Ordering::SeqCst);
    assert_eq!(
        first_calls_before, 3,
        "the failing leg was tried while cold"
    );

    route
        .execute(request(Some(&today_header()), "warm"), None)
        .await
        .expect("served");

    assert_eq!(
        first.load(Ordering::SeqCst),
        first_calls_before,
        "the measured-unreliable leg must not be tried first any more"
    );
    assert_eq!(second.load(Ordering::SeqCst), second_calls_before + 1);
}

// ---------------------------------------------------------------------------
// Truthful aggregate context
// ---------------------------------------------------------------------------

/// A route may not advertise more context than its widest declared target can serve.
///
/// The claim is what a caller reads before choosing the route, and nothing at request time
/// compares it to a model's real window, so the check has to happen at load.
#[test]
fn a_route_cannot_advertise_more_context_than_its_widest_target() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.narrow]
id = "narrow"
llm_client = "stub"
context_window = 32768

[targets.wide]
id = "wide"
llm_client = "stub"
context_window = 262144

[routes.overclaimed]
id = "overclaimed"
type = "passthrough"
target = "narrow"
context_window = 1048576
candidates = ["narrow", "wide"]
"#;
    // SAFETY: test-only, read exclusively by the configuration build in this test.
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    let error = switchyard_runner::Runner::from_toml(configuration)
        .err()
        .expect("the over-claim must be refused");
    let message = error.to_string();
    assert!(
        message.contains("context_window 1048576"),
        "the refusal must name the claim: {message}"
    );
    assert!(
        message.contains("262144"),
        "the refusal must name the widest real window: {message}"
    );
}

/// A composite is resolved before its children and reports its own guarantee, so the same
/// over-claim has to be refused one level up. This is the shape a bounded lane takes: a
/// composite over two Google leaves, where the promise a caller reads comes from the
/// composite rather than from either child.
#[test]
fn a_composite_cannot_advertise_more_context_than_its_widest_child() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.small]
id = "small"
llm_client = "stub"
context_window = 262144
max_input_tokens = 16000

[routes.leaf-small]
id = "leaf-small"
type = "passthrough"
target = "small"
context_window = 262144
candidates = []

[composites.overclaimed-bounded]
children = ["leaf-small"]
context_window = 1048576
"#;
    // SAFETY: test-only, read exclusively by the configuration build in this test.
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    let error = switchyard_runner::Runner::from_toml(configuration)
        .err()
        .expect("the composite over-claim must be refused");
    let message = error.to_string();
    assert!(
        message.contains("overclaimed-bounded"),
        "the refusal must name the composite: {message}"
    );
    assert!(
        message.contains("262144"),
        "the refusal must name the widest window below it: {message}"
    );
}

/// The same configuration with an honest claim loads, and the narrow target is skipped for a
/// request only the wide one can hold.
#[test]
fn an_honest_aggregate_claim_loads() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.narrow]
id = "narrow"
llm_client = "stub"
context_window = 32768

[targets.wide]
id = "wide"
llm_client = "stub"
context_window = 262144

[routes.bounded]
id = "bounded"
type = "passthrough"
target = "narrow"
context_window = 262144
candidates = ["wide"]
selection = "measured"
"#;
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    switchyard_runner::Runner::from_toml(configuration).expect("an honest claim loads");
}

/// A per-request ceiling above the declared window is a configuration error, because the
/// larger of the two can never bind.
#[test]
fn a_ceiling_above_the_window_is_refused() {
    let configuration = r#"
schema_version = 1

[llm_clients.stub]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "SWITCHYARD_ADAPTIVE_TEST_KEY"

[targets.gemma]
id = "gemma-4-31b-it"
llm_client = "stub"
context_window = 16000
max_input_tokens = 262144

[routes.bounded]
id = "bounded"
type = "passthrough"
target = "gemma"
"#;
    unsafe {
        std::env::set_var("SWITCHYARD_ADAPTIVE_TEST_KEY", "test");
    }
    let error = switchyard_runner::Runner::from_toml(configuration)
        .err()
        .expect("the inconsistent pair must be refused");
    assert!(error.to_string().contains("max_input_tokens"), "{error}");
}

// ---------------------------------------------------------------------------
// Adaptive free-pool selection
// ---------------------------------------------------------------------------
//
// These exercise the route, not the selector in isolation: a request is admitted,
// candidates are filtered, the shared evidence and the deployment facts order the
// survivors, and the called leg is counted. The point of each test is a decision that
// cannot be reached by reading the manifest's order.

/// Records what it was asked, including the body the route handed it.
struct RecordingClient {
    calls: Arc<AtomicUsize>,
    bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

#[async_trait]
impl RoutedLlmClient for RecordingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.bodies
            .lock()
            .expect("test mutex is never poisoned")
            .push(
                request
                    .raw_request
                    .clone()
                    .unwrap_or(serde_json::Value::Null),
            );
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "served",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

type CallCounters = HashMap<&'static str, Arc<AtomicUsize>>;
type RecordedBodies = HashMap<&'static str, Arc<std::sync::Mutex<Vec<serde_json::Value>>>>;

/// The deployed free Google pair, in the order the manifest lists them.
const LITE_DAILY: u64 = 500;
const GEMMA_DAILY: u64 = 14_400;
const LITE_INPUT_CEILING: u64 = 250_000;
const GEMMA_INPUT_CEILING: u64 = 16_000;

fn free_lite() -> CapabilityFacts {
    CapabilityFacts {
        free_tier: true,
        context_window: Some(1_048_576),
        max_input_tokens: Some(LITE_INPUT_CEILING),
        ..CapabilityFacts::default()
    }
}

fn free_gemma(accepts_reasoning_control: bool) -> CapabilityFacts {
    CapabilityFacts {
        free_tier: true,
        context_window: Some(262_144),
        max_input_tokens: Some(GEMMA_INPUT_CEILING),
        no_reasoning_control: !accepts_reasoning_control,
        ..CapabilityFacts::default()
    }
}

fn a_paid_leg() -> CapabilityFacts {
    CapabilityFacts {
        context_window: Some(262_144),
        ..CapabilityFacts::default()
    }
}

/// One test's private counter directory, since the allowance reads a real file.
fn allowance_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "switchyard-adaptive-{label}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("test allowance directory");
    dir
}

/// A measured passthrough route over `legs`, declared in the order given.
///
/// `legs` carries each target's facts and daily budget, so a test states the deployment
/// rather than the order it expects.
fn recording_pool(
    dir: &std::path::Path,
    legs: &[(&'static str, CapabilityFacts, Option<u64>)],
) -> (Route, CallCounters, RecordedBodies) {
    let targets: Vec<(String, ModelId)> = legs
        .iter()
        .map(|(name, _, _)| {
            (
                format!("{name}-target"),
                ModelId::from(format!("{name}-target").as_str()),
            )
        })
        .collect();
    let names: Vec<String> = targets.iter().map(|(name, _)| name.clone()).collect();
    let algorithm = AlgorithmSpec::Passthrough {
        target: names[0].clone(),
        candidates: names[1..].to_vec(),
        subagents: None,
    }
    .build("adaptive-pool", &BTreeMap::from_iter(targets.clone()))
    .expect("the pool builds");

    let mut calls = CallCounters::new();
    let mut bodies = RecordedBodies::new();
    let mut clients: Vec<(ModelId, Arc<dyn RoutedLlmClient>)> = Vec::new();
    for ((_, model), (name, _, _)) in targets.iter().zip(legs.iter()) {
        let calls_seen = Arc::new(AtomicUsize::new(0));
        let bodies_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        calls.insert(name, Arc::clone(&calls_seen));
        bodies.insert(name, Arc::clone(&bodies_seen));
        clients.push((
            model.clone(),
            Arc::new(RecordingClient {
                calls: calls_seen,
                bodies: bodies_seen,
            }) as Arc<dyn RoutedLlmClient>,
        ));
    }

    let facts: HashMap<ModelId, CapabilityFacts> = targets
        .iter()
        .zip(legs.iter())
        .map(|((_, model), (_, facts, _))| (model.clone(), *facts))
        .collect();
    let allowances: HashMap<ModelId, Arc<switchyard_llm_client::DailyRequestAllowance>> = targets
        .iter()
        .zip(legs.iter())
        .filter_map(|((_, model), (name, _, limit))| {
            limit.map(|limit| {
                (
                    model.clone(),
                    Arc::new(switchyard_llm_client::DailyRequestAllowance::new(
                        "google-test",
                        model.clone(),
                        limit,
                        dir.join(format!("{name}.json")),
                    )),
                )
            })
        })
        .collect();

    let clients = ClientRouter::new(clients.into_iter().collect()).with_target_capabilities(facts);
    let route = Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(
            [(
                Category::Any,
                targets.iter().map(|(_, model)| model.clone()).collect(),
            )]
            .into(),
        ),
        RouteReasoningPolicy::Inherit,
        None,
        None,
    )
    .with_clearance_key(clearance_key())
    .with_measured_selection(MeasuredSelector::new())
    .with_daily_allowances(Arc::new(allowances));
    (route, calls, bodies)
}

fn calls_of(calls: &CallCounters, leg: &str) -> usize {
    calls[leg].load(Ordering::SeqCst)
}

/// A small request lands on the leg whose daily budget can afford it, with no prior
/// failure needed to discover that.
///
/// This is the correction the release exists for: the deployed chain spends the 500-request
/// model first and reaches the 14,400-request model only when the first one fails.
#[tokio::test]
async fn a_small_bounded_task_reaches_the_daily_rich_leg_before_the_scarce_one() {
    let dir = allowance_dir("small");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("lite", free_lite(), Some(LITE_DAILY)),
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route(
            "adaptive-pool",
            request(Some(&today_header()), "hello"),
            None,
        )
        .await
        .expect("the pool serves a small request");

    assert_eq!(
        calls_of(&calls, "gemma"),
        1,
        "the 14,400-request leg is the economical first choice for small work"
    );
    assert_eq!(
        calls_of(&calls, "lite"),
        0,
        "the scarce leg was spent on work the daily-rich leg could serve"
    );
}

/// A request that would claim most of the small leg's minute goes to the leg with
/// headroom, even though its daily budget is smaller.
#[tokio::test]
async fn a_request_that_fills_the_small_legs_minute_goes_to_the_wide_leg() {
    let dir = allowance_dir("large");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("lite", free_lite(), Some(LITE_DAILY)),
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    // ~12k estimated tokens: inside Gemma's 16,000-token ceiling but three quarters of
    // its minute, so the Lite is the appropriate leg for it.
    runner
        .execute_route("adaptive-pool", long_request(8_000), None)
        .await
        .expect("the pool serves a large request");

    assert_eq!(calls_of(&calls, "lite"), 1);
    assert_eq!(calls_of(&calls, "gemma"), 0);
}

/// A tool exchange is budgeted as the conversation it becomes, so the first turn alone
/// does not make the small leg look appropriate.
#[tokio::test]
async fn a_tool_exchange_is_not_placed_on_the_leg_that_cannot_hold_its_answer() {
    let dir = allowance_dir("tools");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
            ("lite", free_lite(), Some(LITE_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    // ~9k estimated input tokens before the tool growth, comfortably inside Gemma's
    // 16,000-token ceiling on its own but not once the answer turn is budgeted.
    let prompt = "alpha ".repeat(6_000);
    let mut tool_request = request(Some(&today_header()), &prompt);
    tool_request.raw_request = Some(serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }));

    runner
        .execute_route("adaptive-pool", tool_request, None)
        .await
        .expect("the pool serves a tool request");

    assert_eq!(
        calls_of(&calls, "lite"),
        1,
        "a tool conversation does not fit the leg that declares a 16,000-token minute"
    );
    assert_eq!(calls_of(&calls, "gemma"), 0);
}

/// Cost tier is structural: an eligible free leg is offered before a metered one whatever
/// the manifest order says.
#[tokio::test]
async fn a_paid_leg_never_overtakes_an_eligible_free_leg() {
    let dir = allowance_dir("tier");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("paid", a_paid_leg(), None),
            ("free", free_gemma(true), Some(GEMMA_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route(
            "adaptive-pool",
            request(Some(&today_header()), "hello"),
            None,
        )
        .await
        .expect("the free leg serves it");

    assert_eq!(calls_of(&calls, "free"), 1);
    assert_eq!(
        calls_of(&calls, "paid"),
        0,
        "a metered leg was chosen ahead of an eligible free one"
    );
}

/// A request carrying reasoning control skips a leg that cannot accept it, instead of
/// being sent a parameter the provider rejects.
#[tokio::test]
async fn a_reasoning_control_skips_the_leg_that_cannot_accept_it() {
    let dir = allowance_dir("control");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("gemma", free_gemma(false), Some(GEMMA_DAILY)),
            ("lite", free_lite(), Some(LITE_DAILY)),
        ],
    );
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    let mut controlled = request(Some(&today_header()), "hello");
    controlled.llm_request.reasoning.effort = Some("none".to_string());

    runner
        .execute_route("adaptive-pool", controlled, None)
        .await
        .expect("the leg that accepts control serves it");

    assert_eq!(
        calls_of(&calls, "gemma"),
        0,
        "an unsupported reasoning control was forwarded to a target that states it cannot carry one"
    );
    assert_eq!(calls_of(&calls, "lite"), 1);
}

/// The deployment never invents a reasoning control for a Google leg, in either direction:
/// a request that carries none reaches the target with none, so native thinking and the
/// provider default stand.
#[tokio::test]
async fn a_google_leg_is_never_handed_a_reasoning_control_it_was_not_given() {
    let dir = allowance_dir("native");
    let (route, calls, bodies) =
        recording_pool(&dir, &[("gemma", free_gemma(false), Some(GEMMA_DAILY))]);
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route(
            "adaptive-pool",
            request(Some(&today_header()), "hello"),
            None,
        )
        .await
        .expect("served");

    assert_eq!(calls_of(&calls, "gemma"), 1);
    let seen = bodies["gemma"]
        .lock()
        .expect("test mutex is never poisoned")
        .clone();
    let body = seen.first().expect("the body reached the target");
    for key in ["reasoning", "reasoning_effort", "thinking"] {
        assert!(
            body.get(key).is_none_or(serde_json::Value::is_null),
            "the route invented a {key} field the caller never sent: {body}"
        );
    }
    assert!(
        body.pointer("/output_config/effort").is_none(),
        "the route invented an output_config.effort the caller never sent: {body}"
    );
}

/// An exhausted daily budget demotes a leg within its tier without making it ineligible:
/// the hard quota gate stays the authority on whether a request may be sent at all.
#[tokio::test]
async fn a_spent_allowance_is_offered_after_a_leg_that_still_has_budget() {
    let dir = allowance_dir("spent");
    let (route, calls, _) = recording_pool(
        &dir,
        &[
            ("gemma", free_gemma(true), Some(GEMMA_DAILY)),
            ("lite", free_lite(), Some(1)),
        ],
    );
    // Spend the Lite's single request before the measured order is computed.
    let lite_path = dir.join("lite.json");
    std::fs::write(
        &lite_path,
        format!(
            "{{\"day\":{},\"spent\":1,\"limit\":1,\"group\":\"google-test\",\"model\":\"lite-target\",\"basis\":\"local_budget\",\"provider_remaining\":null}}\n",
            switchyard_llm_client::allowance::provider_day(SystemTime::now())
        ),
    )
    .expect("the counter is writable");
    let runner = Runner::new(vec![(ModelId::from("adaptive-pool"), route)]);

    runner
        .execute_route(
            "adaptive-pool",
            request(Some(&today_header()), "hello"),
            None,
        )
        .await
        .expect("the leg with budget serves it");

    assert_eq!(calls_of(&calls, "gemma"), 1);
    assert_eq!(calls_of(&calls, "lite"), 0);
}
