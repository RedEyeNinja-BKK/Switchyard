// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use libsy::RuntimeModels;
use switchyard_llm_client::{
    ClientRouter, DailyRequestAllowance, DayBoundary, ProviderReading, ReserveError, RunObservation,
};
use switchyard_protocol::{
    Category, ContentBlock, LlmClientError, LlmRequest, LlmResponse, Message, ModelId, Request,
    Response, Role, RoutedLlmClient, text_request, text_response,
};
use switchyard_runner::{
    AlgorithmSpec, ModelCapabilities, Route, request_fit::estimate_input_tokens,
};

struct StubClient;

#[async_trait]
impl RoutedLlmClient for StubClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(
                request.llm_request.model.clone(),
                "plugin response",
            )),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

fn plugin_route(client: Arc<dyn RoutedLlmClient>) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "semantic-target".to_string(),
        candidates: Vec::new(),
        subagents: None,
    };
    let targets = BTreeMap::from([(
        "semantic-target".to_string(),
        ModelId::from("semantic-target"),
    )]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("identity target map should build");
    let clients = ClientRouter::new(
        BTreeMap::from([(ModelId::from("semantic-target"), client)])
            .into_iter()
            .collect(),
    );
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new([(Category::Any, vec![ModelId::from("semantic-target")])].into()),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        None,
        None,
    )
}

#[tokio::test]
async fn plugin_shaped_route_executes_without_runner_model_or_toml() {
    let route = plugin_route(Arc::new(StubClient));
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observer = {
        let observations = Arc::clone(&observations);
        Arc::new(move |observation| observations.lock().unwrap().push(observation))
    };
    let request = Request {
        llm_request: text_request(Some("arbitrary-upstream-model".to_string()), "hello"),
        ..Request::default()
    };
    let output = route
        .execute(request, Some(observer))
        .await
        .expect("route should execute");

    assert_eq!(output.selected_model, "semantic-target");
    assert_eq!(
        output
            .response
            .llm_response
            .as_agg()
            .unwrap()
            .model
            .as_deref(),
        Some("semantic-target")
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::AnswerCall(_)) })
    );
    assert!(
        observations
            .lock()
            .unwrap()
            .iter()
            .any(|observation| { matches!(observation, RunObservation::RoutingOverhead(_)) })
    );
}

struct LazyStreamClient {
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl RoutedLlmClient for LazyStreamClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        let polls = Arc::clone(&self.polls);
        let stream = futures_util::stream::poll_fn(move |_| {
            polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::Ready(None)
        })
        .boxed();
        Ok(Response {
            llm_response: LlmResponse::Stream(stream),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

#[tokio::test]
async fn route_returns_stream_without_polling_it() {
    let polls = Arc::new(AtomicUsize::new(0));
    let route = plugin_route(Arc::new(LazyStreamClient {
        polls: Arc::clone(&polls),
    }));
    let request = Request {
        llm_request: text_request(None, "hello"),
        ..Request::default()
    };

    let output = route
        .execute(request, None)
        .await
        .expect("stream handle should be returned");

    assert!(matches!(
        output.response.llm_response,
        LlmResponse::Stream(_)
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
}

/// Two distinct clients so the selected target is observable from which one answered,
/// not merely from a stored field.
struct EchoClient {
    model: &'static str,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl RoutedLlmClient for EchoClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.lock().unwrap().push(self.model);
        if self.model == "refused" {
            // Classified as "another candidate may serve this", which is what makes the
            // walk continue to the legs behind it.
            return Err(LlmClientError::TemporarilyUnavailable);
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(request.llm_request.model.clone(), "ok")),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

/// A local route whose narrow leg is primary and whose wide leg is the escalation target.
///
/// The wide leg is declared vision-incapable, matching the real deployment where the
/// escalation leg is the VL model and the primary is not. Without that fact the capability
/// filter removes nothing and any test about ineligibility is vacuous.
fn escalating_route(
    threshold: Option<u64>,
    escalation_target: Option<&str>,
    calls: &Arc<Mutex<Vec<&'static str>>>,
) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "narrow".to_string(),
        candidates: vec!["wide".to_string(), "refused".to_string()],
        subagents: None,
    };
    let targets = BTreeMap::from([
        ("narrow".to_string(), ModelId::from("narrow")),
        ("wide".to_string(), ModelId::from("wide")),
        ("refused".to_string(), ModelId::from("refused")),
    ]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("algorithm builds");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (
                ModelId::from("narrow"),
                Arc::new(EchoClient {
                    model: "narrow",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("wide"),
                Arc::new(EchoClient {
                    model: "wide",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("refused"),
                Arc::new(EchoClient {
                    model: "refused",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
        ])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([(
        ModelId::from("wide"),
        switchyard_llm_client::CapabilityFacts {
            no_vision: true,
            no_video: true,
            // This fixture is the "cannot see anything" leg; it serves no carrier, so the
            // carrier facts agree with the video fact rather than widening it.
            no_remote_video: true,
            no_inline_video: true,
            no_remote_image: true,
            no_inline_image: true,
            no_tool_calling: false,
            context_window: None,
            max_input_tokens: None,
            ..switchyard_llm_client::CapabilityFacts::default()
        },
    )]));
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(HashMap::from([(
            Category::Any,
            vec![
                ModelId::from("narrow"),
                ModelId::from("wide"),
                ModelId::from("refused"),
            ],
        )])),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        threshold,
        escalation_target.map(ModelId::from),
    )
}

/// A body whose recovered estimate is exactly `tokens`.
///
/// The estimator charges ASCII `div_ceil(4)` per string plus one per string, plus 8
/// structural tokens. Two strings are fixed here ("m" and "user"), contributing 2 + 2, so
/// the repeated text must contribute `tokens - 12` and each of its 4-char groups is worth
/// one token: n = (tokens - 13) * 4. Asserting the estimate keeps the fixture and the
/// comparator in step, so a fixture cannot silently drift off the boundary.
fn body_of_estimated_tokens(tokens: u64) -> serde_json::Value {
    let body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "q".repeat(((tokens - 13) * 4) as usize)}],
    });
    assert_eq!(
        estimate_input_tokens(&body),
        tokens,
        "fixture must land on the exact boundary"
    );
    body
}

/// The historical comparator is strictly greater-than: 61,440 stays on the narrow primary
/// and only 61,441 moves to the declared escalation leg before any provider is called.
#[tokio::test]
async fn context_pressure_reorders_the_escalation_target_before_dispatch() {
    let at_threshold = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("wide"), &at_threshold);
    let output = route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body_of_estimated_tokens(61_440)),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("request at the threshold is served");
    assert_eq!(output.selected_model, "narrow", "61,440 must not escalate");
    assert_eq!(*at_threshold.lock().unwrap(), ["narrow"]);

    let above = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("wide"), &above);
    let output = route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body_of_estimated_tokens(61_441)),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("request above the threshold is served");
    assert_eq!(output.selected_model, "wide", "61,441 must escalate");
    assert_eq!(
        *above.lock().unwrap(),
        ["wide"],
        "the narrow leg must receive no provider call"
    );
}

/// A route that declares no threshold, or a threshold with no escalation target, keeps its
/// declared order. This is the property that stops the reorder leaking into other routes.
#[tokio::test]
async fn no_escalation_contract_leaves_the_declared_order_intact() {
    for (threshold, escalation_target) in [(None, Some("wide")), (Some(61_440), None), (None, None)]
    {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let route = escalating_route(threshold, escalation_target, &calls);
        let output = route
            .execute(
                Request {
                    llm_request: text_request(None, "hello"),
                    raw_request: Some(body_of_estimated_tokens(400_000)),
                    ..Request::default()
                },
                None,
            )
            .await
            .expect("request is served");
        assert_eq!(
            output.selected_model, "narrow",
            "threshold {threshold:?} with target {escalation_target:?} must not reorder"
        );
        assert_eq!(*calls.lock().unwrap(), ["narrow"]);
    }
}

/// The promotion must be observable from outside the route, or the behaviour is invisible
/// in production: nothing else distinguishes it from an ordinary failure fallback.
#[tokio::test]
async fn a_promotion_is_observable_by_the_run_observer() {
    for (tokens, expect_promoted) in [(61_440, false), (61_441, true)] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let route = escalating_route(Some(61_440), Some("wide"), &calls);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observer = {
            let observed = Arc::clone(&observed);
            Arc::new(move |observation| observed.lock().unwrap().push(observation))
        };
        route
            .execute(
                Request {
                    llm_request: text_request(None, "hello"),
                    raw_request: Some(body_of_estimated_tokens(tokens)),
                    ..Request::default()
                },
                Some(observer),
            )
            .await
            .expect("request is served");

        let promotions: Vec<_> = observed
            .lock()
            .unwrap()
            .iter()
            .filter_map(|observation| match observation {
                RunObservation::ContextPressurePromotion {
                    promoted,
                    estimated_input_tokens,
                } => Some((promoted.clone(), *estimated_input_tokens)),
                _ => None,
            })
            .collect();
        if expect_promoted {
            assert_eq!(
                promotions,
                vec![(ModelId::from("wide"), tokens)],
                "a request above the threshold must report the leg it promoted"
            );
        } else {
            assert!(
                promotions.is_empty(),
                "a request at the threshold must not report a promotion, got {promotions:?}"
            );
        }
    }
}

/// The promotion report must describe a reorder that actually happened. A leg that
/// capability filtering already removed cannot be promoted, so claiming a promotion would
/// tell an operator a wider leg was preferred when the narrow one served the request.
#[tokio::test]
async fn an_ineligible_escalation_target_is_not_reported_as_promoted() {
    // The escalation leg is excluded for this request because it cannot see an image, while
    // the route's primary can. Under pressure the leg must neither be reordered nor reported.
    let calls = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("wide"), &calls);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer = {
        let observed = Arc::clone(&observed);
        Arc::new(move |observation| observed.lock().unwrap().push(observation))
    };
    // The image is added after sizing, so the text is sized to leave room for the few tokens
    // the image wrapper adds. The request must END UP over the threshold, otherwise the promotion
    // is never reached and this test would pass for the wrong reason.
    let sized = body_of_estimated_tokens(61_430);
    // `body_of_estimated_tokens` returns the chat shape, so the text is read from where it
    // actually put it rather than from the Responses `input` path.
    let text = sized["messages"][0]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let body = serde_json::json!({
        "model": "smartlocal",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": "https://example.test/i.png"}
            ]
        }]
    });
    let estimate = switchyard_runner::request_fit::estimate_input_tokens(&body);
    assert!(
        estimate > 61_440,
        "the fixture must be under context pressure for this test to mean anything, got {estimate}"
    );

    route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("the request is served by the eligible primary");

    let promotion_count: usize = {
        let observations = observed.lock().unwrap();
        observations
            .iter()
            .filter(|o| matches!(o, RunObservation::ContextPressurePromotion { .. }))
            .count()
    };
    assert_eq!(
        promotion_count, 0,
        "an escalation leg that cannot serve this request must not be reported as promoted"
    );
}

/// Promotion reorders; it must not shorten the list. With the promoted leg forced to fail,
/// the candidates behind it are still reached in their declared relative order.
#[tokio::test]
async fn promotion_keeps_the_rest_of_the_candidate_list() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let route = escalating_route(Some(61_440), Some("refused"), &calls);
    let output = route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                raw_request: Some(body_of_estimated_tokens(61_441)),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("the request is served by a later leg");
    // `selected_model` is the first model attempted, so the promoted leg is expected there
    // even though it failed. The client log is the authority on which legs were actually
    // walked: the response model only echoes the rewritten request, so it cannot show this.
    assert_eq!(
        output.selected_model, "refused",
        "the promoted leg is attempted first"
    );
    assert_eq!(
        *calls.lock().unwrap(),
        ["refused", "narrow"],
        "the promoted leg is tried first, then the remaining legs in declared order until \
         one serves the request"
    );
}

// Preserve checkpoint target order and explicit TOML overrides.
#[test]
fn prefill_router_config_preserves_target_order_and_overrides() {
    let spec: AlgorithmSpec = toml::from_str(
        r#"
type = "prefill_router"
targets = ["fast", "strong"]
checkpoint = "/models/router.pt"
device = "cuda:1"
max_length = 4096
batch_size = 8
"#,
    )
    .expect("prefill router config should parse");

    let AlgorithmSpec::PrefillRouter {
        targets,
        checkpoint,
        device,
        cache_dir,
        max_length,
        batch_size,
    } = spec
    else {
        panic!("expected prefill router config");
    };
    assert_eq!(targets, ["fast", "strong"]);
    assert_eq!(checkpoint.to_string_lossy(), "/models/router.pt");
    assert_eq!(device.as_deref(), Some("cuda:1"));
    assert_eq!(cache_dir, None);
    assert_eq!(max_length, Some(4096));
    assert_eq!(batch_size, Some(8));
}

/// A route whose candidates are all free-scope slugs is skipped, not entered, once the
/// credential's free allowance is gone.
///
/// The provider's dynamic Free Models Router spends that same account-wide allowance
/// without carrying a `:free` suffix. Reading the suffix alone classified a free-only
/// route that contains it as having a paid leg, so the composite entered the child and
/// spent one guaranteed-rejected call per request for the rest of the day.
#[test]
fn a_dynamic_free_router_does_not_make_a_free_route_look_paid() {
    fn passthrough(target: &str, candidates: &[&str]) -> Route {
        let models: Vec<ModelId> = std::iter::once(target)
            .chain(candidates.iter().copied())
            .map(ModelId::from)
            .collect();
        let targets: BTreeMap<String, ModelId> = models
            .iter()
            .map(|model| (model.as_str().to_owned(), model.clone()))
            .collect();
        let algorithm = AlgorithmSpec::Passthrough {
            target: target.to_string(),
            candidates: candidates.iter().map(|c| (*c).to_string()).collect(),
            subagents: None,
        }
        .build("switchyard", &targets)
        .expect("passthrough target map should build");
        Route::new(
            algorithm,
            ClientRouter::new(HashMap::new()),
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new([(Category::Any, models)].into()),
            switchyard_runner::RouteReasoningPolicy::Inherit,
            None,
            None,
        )
    }

    // Freeness is the declared tariff, so the deployment has to declare it. A route whose
    // every target states `free_tier = true` is free end to end, including one whose legs
    // carry no `:free` suffix at all - the case the slug convention can never see.
    fn passthrough_with_tariff(slugs: &[&str], free: &[&str], metered: &[&str]) -> Route {
        let models: Vec<ModelId> = slugs.iter().map(|slug| ModelId::from(*slug)).collect();
        let targets: BTreeMap<String, ModelId> = models
            .iter()
            .map(|model| (model.as_str().to_owned(), model.clone()))
            .collect();
        let mut facts = HashMap::new();
        for slug in slugs {
            facts.insert(
                ModelId::from(*slug),
                switchyard_llm_client::CapabilityFacts {
                    tariff: if free.contains(slug) {
                        switchyard_llm_client::Tariff::Free
                    } else if metered.contains(slug) {
                        switchyard_llm_client::Tariff::Metered
                    } else {
                        switchyard_llm_client::Tariff::Unknown
                    },
                    ..switchyard_llm_client::CapabilityFacts::default()
                },
            );
        }
        let algorithm = AlgorithmSpec::Passthrough {
            target: slugs[0].to_string(),
            candidates: slugs[1..].iter().map(|slug| (*slug).to_string()).collect(),
            subagents: None,
        }
        .build("switchyard", &targets)
        .expect("passthrough target map should build");
        Route::new(
            algorithm,
            ClientRouter::new(HashMap::new()).with_target_capabilities(facts),
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::new([(Category::Any, models)].into()),
            switchyard_runner::RouteReasoningPolicy::Inherit,
            None,
            None,
        )
    }

    assert!(
        passthrough_with_tariff(
            &[
                "baidu/ernie-5.5-flash-preview:free",
                "google/gemma-4-26b-a4b-it"
            ],
            &[
                "baidu/ernie-5.5-flash-preview:free",
                "google/gemma-4-26b-a4b-it",
            ],
            &[],
        )
        .all_targets_free(),
        "every leg declares itself free, so the route is free end to end regardless of slug"
    );
    assert!(
        !passthrough_with_tariff(
            &["baidu/ernie-5.5-flash-preview:free", "z-ai/glm-5.3-flash"],
            &["baidu/ernie-5.5-flash-preview:free"],
            &["z-ai/glm-5.3-flash"],
        )
        .all_targets_free(),
        "one metered candidate keeps a servable leg, so the route is not free"
    );
    // An undeclared tariff still answers the *allowance* question through the provider's own
    // scoping, which is the fact this method exists to supply. The dynamic router and `:free`
    // slugs draw on the same allowance, so a route made only of them is free end to end and a
    // quota block may skip it instead of spending a guaranteed-rejected call.
    assert!(
        passthrough("openrouter/free", &[]).all_targets_free(),
        "an undeclared target falls back to the provider's allowance scope for this question"
    );
    assert!(
        passthrough_with_tariff(
            &["vendor/model:free", "vendor/other:free"],
            &["vendor/model:free"],
            &[],
        )
        .all_targets_free(),
        "the undeclared leg is provider-free-scope, so the route is still all free allowance"
    );
    // A declared tariff beats the slug in both directions: a `:free`-suffixed target the
    // deployment declares metered is metered, because the declaration is the authority.
    assert!(
        !passthrough_with_tariff(&["vendor/model:free"], &[], &["vendor/model:free"])
            .all_targets_free(),
        "a declared metered tariff overrides the `:free` slug"
    );
    // And a target with no `:free` suffix that is declared free still joins the allowance.
    assert!(
        passthrough_with_tariff(
            &["google/gemma-4-26b-a4b-it"],
            &["google/gemma-4-26b-a4b-it"],
            &[]
        )
        .all_targets_free(),
        "a declared free tariff is sufficient without any slug convention"
    );
}

/// A route whose primary leg declares it cannot replay a tool call.
///
/// The declaration is a wire fact about one provider, so the route must keep serving the
/// definitional call there while sending the continuation to a sibling that accepts it.
fn replay_aware_route(calls: &Arc<Mutex<Vec<&'static str>>>) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "no-replay".to_string(),
        candidates: vec!["replays".to_string()],
        subagents: None,
    };
    let targets = BTreeMap::from([
        ("no-replay".to_string(), ModelId::from("no-replay")),
        ("replays".to_string(), ModelId::from("replays")),
    ]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("algorithm builds");
    let clients = ClientRouter::new(
        BTreeMap::from([
            (
                ModelId::from("no-replay"),
                Arc::new(EchoClient {
                    model: "no-replay",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
            (
                ModelId::from("replays"),
                Arc::new(EchoClient {
                    model: "replays",
                    calls: Arc::clone(calls),
                }) as Arc<dyn RoutedLlmClient>,
            ),
        ])
        .into_iter()
        .collect(),
    )
    .with_target_capabilities(HashMap::from([(
        ModelId::from("no-replay"),
        switchyard_llm_client::CapabilityFacts {
            no_tool_call_replay: true,
            ..switchyard_llm_client::CapabilityFacts::default()
        },
    )]));
    Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(HashMap::from([(
            Category::Any,
            vec![ModelId::from("no-replay"), ModelId::from("replays")],
        )])),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        None,
        None,
    )
}

fn request_with_tool_lineage() -> LlmRequest {
    LlmRequest {
        model: None,
        messages: vec![
            Message::text(Role::User, "weather in Paris?"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(switchyard_protocol::ToolCall {
                    provider_state: None,
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city": "Paris"}),
                })],
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult(switchyard_protocol::ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "18C and raining".to_string(),
                    }],
                    is_error: None,
                })],
            },
        ],
        ..LlmRequest::default()
    }
}

/// The declaration reaches candidate selection: a replay skips the leg that cannot
/// continue it, while the same leg still answers the call that merely offers tools.
#[tokio::test]
async fn a_leg_that_cannot_replay_a_tool_call_is_skipped_only_for_the_continuation() {
    let replayed = Arc::new(Mutex::new(Vec::new()));
    let route = replay_aware_route(&replayed);
    route
        .execute(
            Request {
                llm_request: request_with_tool_lineage(),
                ..Request::default()
            },
            None,
        )
        .await
        .expect("a sibling can serve the continuation");
    assert_eq!(
        *replayed.lock().unwrap(),
        vec!["replays"],
        "the replay must not reach the leg that cannot accept one"
    );

    let definitional = Arc::new(Mutex::new(Vec::new()));
    let route = replay_aware_route(&definitional);
    let mut request = text_request(None, "weather in Paris?");
    request.tools = vec![switchyard_protocol::ToolDefinition {
        name: "get_weather".to_string(),
        description: None,
        parameters: serde_json::json!({"type": "object"}),
        strict: None,
    }];
    route
        .execute(
            Request {
                llm_request: request,
                ..Request::default()
            },
            None,
        )
        .await
        .expect("the primary leg serves a definitional call");
    assert_eq!(
        *definitional.lock().unwrap(),
        vec!["no-replay"],
        "offering tools must stay eligible on the leg that cannot continue them"
    );
}

// ---------------------------------------------------------------------------------------
// Capability escalation: a pool whose wires cannot carry a request may look at the next
// declared group, and nothing else changes.
// ---------------------------------------------------------------------------------------

/// A route whose entire primary pool cannot replay a tool call, with one declared later group.
///
/// `terminal_replays` decides whether that group can carry the request: false is the case where
/// no declared group can, which must refuse rather than spend a call proving it again.
fn wire_incompatible_pool(
    primary_calls: &Arc<Mutex<Vec<&'static str>>>,
    terminal_calls: &Arc<Mutex<Vec<&'static str>>>,
    tiers: &[(&'static str, bool)],
    escalation: bool,
) -> Route {
    let spec = AlgorithmSpec::Passthrough {
        target: "google-a".to_string(),
        candidates: vec!["google-b".to_string()],
        subagents: None,
    };
    let targets = BTreeMap::from([
        ("google-a".to_string(), ModelId::from("google-a")),
        ("google-b".to_string(), ModelId::from("google-b")),
    ]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("algorithm builds");
    let mut facts = HashMap::from([
        (
            ModelId::from("google-a"),
            switchyard_llm_client::CapabilityFacts {
                no_tool_call_replay: true,
                ..switchyard_llm_client::CapabilityFacts::default()
            },
        ),
        (
            ModelId::from("google-b"),
            switchyard_llm_client::CapabilityFacts {
                no_tool_call_replay: true,
                ..switchyard_llm_client::CapabilityFacts::default()
            },
        ),
    ]);
    for (name, replays) in tiers {
        if !replays {
            facts.insert(
                ModelId::from(*name),
                switchyard_llm_client::CapabilityFacts {
                    no_tool_call_replay: true,
                    ..switchyard_llm_client::CapabilityFacts::default()
                },
            );
        }
    }
    let mut clients: HashMap<ModelId, Arc<dyn RoutedLlmClient>> = HashMap::from([
        (
            ModelId::from("google-a"),
            Arc::new(EchoClient {
                model: "google-a",
                calls: Arc::clone(primary_calls),
            }) as Arc<dyn RoutedLlmClient>,
        ),
        (
            ModelId::from("google-b"),
            Arc::new(EchoClient {
                model: "google-b",
                calls: Arc::clone(primary_calls),
            }) as Arc<dyn RoutedLlmClient>,
        ),
    ]);
    for (name, _) in tiers {
        clients.insert(
            ModelId::from(*name),
            Arc::new(EchoClient {
                model: name,
                calls: Arc::clone(terminal_calls),
            }) as Arc<dyn RoutedLlmClient>,
        );
    }
    let clients = ClientRouter::new(clients).with_target_capabilities(facts);
    let route = Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(HashMap::from([(
            Category::Any,
            vec![ModelId::from("google-a"), ModelId::from("google-b")],
        )])),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        None,
        None,
    )
    .with_terminal_fallback(
        tiers
            .iter()
            .map(|(name, _)| {
                RuntimeModels::new(HashMap::from([(Category::Any, vec![ModelId::from(*name)])]))
            })
            .collect(),
    );
    if escalation {
        route.with_capability_escalation(true)
    } else {
        route
    }
}

/// Collects every selection record the route emits.
fn trace_sink() -> (
    Arc<Mutex<Vec<switchyard_llm_client::SelectionTrace>>>,
    switchyard_llm_client::RunObserver,
) {
    let sink: Arc<Mutex<Vec<switchyard_llm_client::SelectionTrace>>> =
        Arc::new(Mutex::new(Vec::new()));
    let written = Arc::clone(&sink);
    let observer: switchyard_llm_client::RunObserver =
        Arc::new(move |observation: RunObservation| {
            if let RunObservation::Selection(trace) = observation {
                written.lock().expect("trace sink").push(*trace);
            }
        });
    (sink, observer)
}

/// The measured live case: five wire-incompatible legs in the preferred group, and a later
/// group that accepts the same transcript. Escalation must reach the group and record a reason
/// distinct from exhaustion, naming the target that actually answered.
#[tokio::test]
async fn a_wire_incompatible_pool_escalates_to_the_next_declared_group() {
    let primary = Arc::new(Mutex::new(Vec::new()));
    let terminal = Arc::new(Mutex::new(Vec::new()));
    let route = wire_incompatible_pool(&primary, &terminal, &[("portable", true)], true);
    let (sink, observer) = trace_sink();

    let output = route
        .execute(
            Request {
                llm_request: request_with_tool_lineage(),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("the later group can carry the replay");

    assert_eq!(output.selected_model.as_str(), "portable");
    assert!(
        primary.lock().unwrap().is_empty(),
        "no leg of the incompatible pool may be called"
    );
    assert_eq!(*terminal.lock().unwrap(), vec!["portable"]);

    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert_eq!(trace.outcome, "served");
    assert_eq!(trace.served.as_deref(), Some("portable"));
    assert!(trace.terminal_crossed);
    assert_eq!(
        trace.terminal_reason,
        Some("primary_pool_capability_incompatible"),
        "the crossing names the wire, not exhaustion"
    );
    assert_eq!(trace.terminal_tier, Some(1));
    assert!(
        trace
            .attempts
            .iter()
            .all(|attempt| attempt.model == "portable"),
        "the record must name the leg that actually answered"
    );
}

/// Off is the historical behaviour, and it must not touch the later group: the option adds
/// reach, never a default.
#[tokio::test]
async fn capability_escalation_off_refuses_without_contacting_the_group() {
    let primary = Arc::new(Mutex::new(Vec::new()));
    let terminal = Arc::new(Mutex::new(Vec::new()));
    let route = wire_incompatible_pool(&primary, &terminal, &[("portable", true)], false);
    let (sink, observer) = trace_sink();

    let error = route
        .execute(
            Request {
                llm_request: request_with_tool_lineage(),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .err()
        .expect("the pool cannot carry the replay and nothing authorizes a look further");

    assert!(format!("{error}").contains("tool_call_replay"), "{error}");
    assert!(primary.lock().unwrap().is_empty());
    assert!(
        terminal.lock().unwrap().is_empty(),
        "the default must not consult a later group"
    );

    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert_eq!(trace.outcome, "no_eligible_candidate");
    assert!(!trace.terminal_crossed);
    assert_eq!(trace.terminal_reason, None);
    assert_eq!(trace.terminal_tier, None);
    assert!(
        trace.terminal_skipped.is_empty(),
        "a tier the walk never reached is absent, not refused"
    );
}

/// Escalation that finds no capable group still refuses, and records the tier it reached: the
/// walk must not spend a call to discover what its own declarations already say.
#[tokio::test]
async fn an_escalation_that_finds_no_capable_group_still_refuses() {
    let primary = Arc::new(Mutex::new(Vec::new()));
    let terminal = Arc::new(Mutex::new(Vec::new()));
    let route = wire_incompatible_pool(&primary, &terminal, &[("portable", false)], true);
    let (sink, observer) = trace_sink();

    let error = route
        .execute(
            Request {
                llm_request: request_with_tool_lineage(),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .err()
        .expect("no declared group can carry the replay");

    assert!(format!("{error}").contains("tool_call_replay"), "{error}");
    assert!(primary.lock().unwrap().is_empty());
    assert!(
        terminal.lock().unwrap().is_empty(),
        "an incompatible tier must not be called"
    );

    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert_eq!(trace.outcome, "no_eligible_candidate");
    assert!(!trace.terminal_crossed);
    assert_eq!(
        trace.terminal_skipped,
        vec![("portable".to_string(), "tool_call_replay")],
        "the reached tier is recorded with the rule that excluded it"
    );
}

/// A group the walk reaches and finds incompatible must not end the walk: the next declared
/// group is asked the same question. This is the difference between "evaluate the next
/// declared group" and "evaluate the next group only if the first one had a capable leg".
#[tokio::test]
async fn an_incompatible_group_does_not_end_the_ordered_walk() {
    let primary = Arc::new(Mutex::new(Vec::new()));
    let terminal = Arc::new(Mutex::new(Vec::new()));
    // Group 1 is Google again -- same wire, same refusal. Group 2 can carry the replay.
    let route = wire_incompatible_pool(
        &primary,
        &terminal,
        &[("google-c", false), ("portable", true)],
        true,
    );
    let (sink, observer) = trace_sink();

    let output = route
        .execute(
            Request {
                llm_request: request_with_tool_lineage(),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("the second declared group can carry the replay");

    assert_eq!(output.selected_model.as_str(), "portable");
    assert!(primary.lock().unwrap().is_empty());
    assert_eq!(
        *terminal.lock().unwrap(),
        vec!["portable"],
        "the incompatible group is skipped by its declaration, not called to find out"
    );

    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert_eq!(
        trace.terminal_reason,
        Some("primary_pool_capability_incompatible")
    );
    assert_eq!(
        trace.terminal_tier,
        Some(2),
        "tier 2 is also a record that tier 1 was proved out first"
    );
    assert_eq!(
        trace.terminal_skipped,
        vec![("google-c".to_string(), "tool_call_replay")],
        "the reached group is recorded with the rule that excluded it"
    );
}

// ---------------------------------------------------------------------------------------
// Strict allowance boundary, end to end: two declared primary domains and one local tier.
//
// The claim under test is about the deployment's allowance domains, so the fixtures below
// drive it through a real route: one domain whose members refuse on their own daily budget,
// one domain that either skips the request (fit), answers (recovery) or is proved spent by
// its own counter while remaining ineligible. The local tier records every call, so "no paid
// spill" and "no dispatch to an ineligible leaf" are observable rather than inferred.
// ---------------------------------------------------------------------------------------

/// Answers with the local daily-budget refusal, before any request leaves the process.
struct SpentClient {
    model: &'static str,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl RoutedLlmClient for SpentClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        self.calls.lock().unwrap().push(self.model);
        Err(LlmClientError::DailyRequestAllowanceExhausted {
            model: ModelId::from(self.model),
            limit: 20,
        })
    }
}

struct StrictPool {
    route: Route,
    primary: Arc<Mutex<Vec<&'static str>>>,
    local: Arc<Mutex<Vec<&'static str>>>,
}

/// Answers the way a client does when its pre-send reservation refuses: the send never
/// leaves, and the error says the provider's pool is empty. The reading behind that refusal
/// may be current or retained, and the error cannot say which -- which is the whole point.
struct ProviderExhaustedClient {
    model: &'static str,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl RoutedLlmClient for ProviderExhaustedClient {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        self.calls.lock().unwrap().push(self.model);
        Err(LlmClientError::ProviderAllowanceExhausted {
            model: ModelId::from(self.model),
            limit: 1_000,
        })
    }
}

/// How the other domain's only member answers this request.
#[derive(Clone, Copy)]
enum O1Behavior {
    Serves,
    SpentLocally,
    ProviderRefused,
}

/// Two primary groups over one pool -- `google` {g1, g2} and `openai` {o1} -- and a local
/// tier that is only reachable through the strict allowance boundary.
///
/// `o1_window` is the physical window declared for the other domain's only member, which is
/// how a request is made to skip that domain on fit alone. `allowances` is the authoritative
/// counter state the route reads for members it never dispatches.
fn strict_pool(
    o1_window: Option<u64>,
    o1_refuses: bool,
    allowances: HashMap<ModelId, Arc<DailyRequestAllowance>>,
) -> StrictPool {
    strict_pool_with(
        o1_window,
        if o1_refuses {
            O1Behavior::SpentLocally
        } else {
            O1Behavior::Serves
        },
        allowances,
    )
}

fn strict_pool_with(
    o1_window: Option<u64>,
    o1: O1Behavior,
    allowances: HashMap<ModelId, Arc<DailyRequestAllowance>>,
) -> StrictPool {
    let primary = Arc::new(Mutex::new(Vec::new()));
    let local = Arc::new(Mutex::new(Vec::new()));
    let spec = AlgorithmSpec::Passthrough {
        target: "g1".to_string(),
        candidates: vec!["g2".to_string(), "o1".to_string()],
        subagents: None,
    };
    let targets = BTreeMap::from([
        ("g1".to_string(), ModelId::from("g1")),
        ("g2".to_string(), ModelId::from("g2")),
        ("o1".to_string(), ModelId::from("o1")),
    ]);
    let algorithm = spec
        .build("switchyard", &targets)
        .expect("algorithm builds");
    let wide = switchyard_llm_client::CapabilityFacts {
        context_window: Some(200_000),
        ..switchyard_llm_client::CapabilityFacts::default()
    };
    let facts = HashMap::from([
        (ModelId::from("g1"), wide.clone()),
        (ModelId::from("g2"), wide),
        (
            ModelId::from("o1"),
            switchyard_llm_client::CapabilityFacts {
                context_window: o1_window,
                ..switchyard_llm_client::CapabilityFacts::default()
            },
        ),
    ]);
    let clients: HashMap<ModelId, Arc<dyn RoutedLlmClient>> = HashMap::from([
        (
            ModelId::from("g1"),
            Arc::new(SpentClient {
                model: "g1",
                calls: Arc::clone(&primary),
            }) as Arc<dyn RoutedLlmClient>,
        ),
        (
            ModelId::from("g2"),
            Arc::new(SpentClient {
                model: "g2",
                calls: Arc::clone(&primary),
            }) as Arc<dyn RoutedLlmClient>,
        ),
        (
            ModelId::from("o1"),
            match o1 {
                O1Behavior::SpentLocally => Arc::new(SpentClient {
                    model: "o1",
                    calls: Arc::clone(&primary),
                }) as Arc<dyn RoutedLlmClient>,
                O1Behavior::ProviderRefused => Arc::new(ProviderExhaustedClient {
                    model: "o1",
                    calls: Arc::clone(&primary),
                }) as Arc<dyn RoutedLlmClient>,
                O1Behavior::Serves => Arc::new(EchoClient {
                    model: "o1",
                    calls: Arc::clone(&primary),
                }) as Arc<dyn RoutedLlmClient>,
            },
        ),
        (
            ModelId::from("local"),
            Arc::new(EchoClient {
                model: "local",
                calls: Arc::clone(&local),
            }) as Arc<dyn RoutedLlmClient>,
        ),
    ]);
    let clients = ClientRouter::new(clients)
        .with_target_capabilities(facts)
        .with_target_credentials(HashMap::from([
            (ModelId::from("g1"), "GOOGLE".to_string()),
            (ModelId::from("g2"), "GOOGLE".to_string()),
            (ModelId::from("o1"), "OPENAI".to_string()),
            (ModelId::from("local"), "HTPC".to_string()),
        ]));
    let route = Route::new(
        algorithm,
        clients,
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        RuntimeModels::new(HashMap::from([(
            Category::Any,
            vec![
                ModelId::from("g1"),
                ModelId::from("g2"),
                ModelId::from("o1"),
            ],
        )])),
        switchyard_runner::RouteReasoningPolicy::Inherit,
        None,
        None,
    )
    .with_terminal_fallback(vec![RuntimeModels::new(HashMap::from([(
        Category::Any,
        vec![ModelId::from("local")],
    )]))])
    .with_terminal_policy(switchyard_runner::TerminalPolicy::AllowanceOnly)
    .with_group_plan(switchyard_runner::selection::GroupPlan {
        groups: vec![
            switchyard_runner::selection::GroupMembers {
                name: "google".to_string(),
                members: vec![ModelId::from("g1"), ModelId::from("g2")],
            },
            switchyard_runner::selection::GroupMembers {
                name: "openai".to_string(),
                members: vec![ModelId::from("o1")],
            },
        ],
    })
    .with_daily_allowances(Arc::new(allowances));
    StrictPool {
        route,
        primary,
        local,
    }
}

/// A counter that has spent its whole day, as the deployment's own accounting would leave it.
fn spent_allowance(group: &str, model: &str, tag: &str) -> (ModelId, Arc<DailyRequestAllowance>) {
    let path = std::env::temp_dir().join(format!(
        "sy-strict-{}-{tag}-{}.json",
        std::process::id(),
        model
    ));
    let _ = std::fs::remove_file(&path);
    let allowance = DailyRequestAllowance::new(group, ModelId::from(model), 1, path);
    allowance.try_reserve().expect("the day's only request");
    assert!(
        allowance.try_reserve().is_err(),
        "the counter must be provably spent"
    );
    assert_eq!(allowance.remaining(), 0);
    (ModelId::from(model), Arc::new(allowance))
}

/// A counter that exists on disk but cannot be parsed: the state is unknown, and the
/// two-state reading reports it as a full day's spend.
fn corrupt_allowance(group: &str, model: &str, tag: &str) -> (ModelId, Arc<DailyRequestAllowance>) {
    let path = std::env::temp_dir().join(format!(
        "sy-strict-{}-{tag}-{}.json",
        std::process::id(),
        model
    ));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("a state directory");
    }
    std::fs::write(&path, b"{not a counter").expect("a broken counter");
    let allowance = DailyRequestAllowance::new(group, ModelId::from(model), 1, path);
    assert_eq!(allowance.remaining(), 0, "the trap this fixture exists for");
    (ModelId::from(model), Arc::new(allowance))
}

fn big_request(tokens: u64) -> Request {
    Request {
        llm_request: text_request(None, "hello"),
        raw_request: Some(body_of_estimated_tokens(tokens)),
        ..Request::default()
    }
}

/// Both primary domains refuse on their own budget, so both allowances really are spent and
/// the local tier is the authorized answer. The record names the exhaustion boundary rather
/// than the fit or capability ones.
#[tokio::test]
async fn both_primary_domains_refusing_locally_opens_the_local_tier() {
    let pool = strict_pool(Some(200_000), true, HashMap::new());
    let (sink, observer) = trace_sink();
    let output = pool
        .route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("the local tier serves");
    assert_eq!(output.selected_model, "local");
    assert_eq!(*pool.local.lock().unwrap(), vec!["local"]);
    assert_eq!(
        *pool.primary.lock().unwrap(),
        vec!["g1", "g2", "o1"],
        "every primary leg is offered exactly once"
    );
    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert!(trace.terminal_crossed);
    assert_eq!(trace.terminal_reason, Some("primary_pool_exhausted"));
}

/// The reviewed counterexample at route level: the other domain is absent because the request
/// does not fit it, and nothing has established that its allowance is spent. The local tier
/// must stay closed even though every leg this request did try has spent its budget.
#[tokio::test]
async fn a_domain_that_only_skipped_the_request_keeps_the_local_tier_closed() {
    let pool = strict_pool(Some(1_000), false, HashMap::new());
    let (sink, observer) = trace_sink();
    let result = pool.route.execute(big_request(5_000), Some(observer)).await;
    assert!(
        result.is_err(),
        "an unproven allowance is not a spent one, so the request is refused"
    );
    assert!(
        pool.local.lock().unwrap().is_empty(),
        "the local tier must not be spent on an unproven domain"
    );
    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert!(!trace.terminal_crossed);
    assert_eq!(trace.terminal_reason, None);
}

/// The same shape, with the excluded domain's own counter read instead of inferred: it is
/// spent, so the boundary opens -- and the ineligible leg is never dispatched to learn that.
#[tokio::test]
async fn an_excluded_domain_proved_spent_by_its_counter_opens_the_tier_without_dispatch() {
    let (o1, allowance) = spent_allowance("openai", "o1", "excluded");
    let pool = strict_pool(Some(1_000), false, HashMap::from([(o1, allowance)]));
    let (sink, observer) = trace_sink();
    let output = pool
        .route
        .execute(big_request(5_000), Some(observer))
        .await
        .expect("the local tier serves");
    assert_eq!(output.selected_model, "local");
    assert_eq!(*pool.local.lock().unwrap(), vec!["local"]);
    assert_eq!(
        *pool.primary.lock().unwrap(),
        vec!["g1", "g2"],
        "o1 cannot hold this request and must not be called to find out"
    );
    let traces = sink.lock().unwrap();
    assert_eq!(
        traces.last().expect("a selection record").terminal_reason,
        Some("primary_pool_exhausted")
    );
}

/// Recovery: the first domain is spent, the second has headroom and can serve, so the local
/// tier is never reached. A boundary that opened on the first domain alone would spend it.
#[tokio::test]
async fn a_domain_with_headroom_serves_and_the_local_tier_is_not_spent() {
    let pool = strict_pool(Some(200_000), false, HashMap::new());
    let (sink, observer) = trace_sink();
    let output = pool
        .route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("the second domain serves");
    // The record names the leg that answered; the algorithm's own label for a passthrough
    // is its declared target, so the responder is read from the trace.
    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert_eq!(trace.served.as_deref(), Some("o1"));
    assert!(pool.local.lock().unwrap().is_empty());
    assert!(!trace.terminal_crossed);
    drop(traces);
    assert_eq!(output.selected_model, "g1");
}

/// A target the deployment keeps no counter for is unknown, not spent: the same route, the
/// same exhausted request, and the local tier stays closed because one domain's state was
/// never established.
#[tokio::test]
async fn a_target_without_a_counter_is_not_exhaustion_at_route_level() {
    let pool = strict_pool(Some(1_000), false, HashMap::new());
    // `o1` has no registered allowance, so `allowance_spent` answers false and the excluded
    // domain stays unknown -- which is the point: nothing established its state.
    let (sink, observer) = trace_sink();
    assert!(
        pool.route
            .execute(big_request(5_000), Some(observer))
            .await
            .is_err()
    );
    assert!(pool.local.lock().unwrap().is_empty());
    assert!(
        !sink
            .lock()
            .unwrap()
            .last()
            .expect("a selection record")
            .terminal_crossed
    );
}

/// A counter that exists but cannot be parsed is unknown too, and this is the case a
/// two-state reading gets wrong: `remaining()` reports `0` for a broken file, so reading the
/// figure alone would mark the domain spent and buy the local tier with a corrupt ledger.
#[tokio::test]
async fn a_corrupt_counter_does_not_open_the_local_tier() {
    let (o1, allowance) = corrupt_allowance("openai", "o1", "corrupt");
    let pool = strict_pool(Some(1_000), false, HashMap::from([(o1, allowance)]));
    let (sink, observer) = trace_sink();
    assert!(
        pool.route
            .execute(big_request(5_000), Some(observer))
            .await
            .is_err(),
        "an unparseable ledger is not an exhausted one, so the request is refused"
    );
    assert!(
        pool.local.lock().unwrap().is_empty(),
        "the local tier must not be bought with a broken file"
    );
    assert!(
        !sink
            .lock()
            .unwrap()
            .last()
            .expect("a selection record")
            .terminal_crossed
    );
}

// ---------------------------------------------------------------------------------------
// Retained provider exhaustion: the refusal is real, the statement behind it is about the
// past. The two facts are different, and only the first one belongs to admission.
// ---------------------------------------------------------------------------------------

/// A counter whose provider reading reported an empty pool, refused the send, and whose
/// evidence is whatever `observed` and `tolerance` make it.
fn provider_empty(
    group: &str,
    model: &str,
    tag: &str,
    observed: std::time::SystemTime,
    tolerance: std::time::Duration,
) -> (ModelId, Arc<DailyRequestAllowance>) {
    let path = std::env::temp_dir().join(format!(
        "sy-provider-{}-{tag}-{}.json",
        std::process::id(),
        model
    ));
    let _ = std::fs::remove_file(&path);
    let allowance = DailyRequestAllowance::new_on(
        group,
        ModelId::from(model),
        1_000,
        path,
        DayBoundary::Utc,
    )
    .with_provider_tolerance(tolerance);
    allowance
        .reconcile(ProviderReading {
            used: 1_000,
            limit: 1_000,
            remaining: 0,
            observed_at: observed,
            reset_at: None,
        })
        .expect("the fixture's reading belongs to the current provider day");
    assert!(
        matches!(
            allowance.try_reserve(),
            Err(ReserveError::ProviderExhausted { limit: 1_000 })
        ),
        "fixture: an empty pool refuses the send, whatever its basis"
    );
    (ModelId::from(model), Arc::new(allowance))
}

/// An empty pool whose reading is current: affirmative evidence that the domain is spent.
fn fresh_provider_allowance(
    group: &str,
    model: &str,
    tag: &str,
) -> (ModelId, Arc<DailyRequestAllowance>) {
    let now = std::time::SystemTime::now();
    let fixture = provider_empty(group, model, tag, now, std::time::Duration::from_secs(60));
    assert_eq!(
        fixture.1.spent_today(),
        Some(true),
        "fixture: a current zero reading is spent evidence"
    );
    fixture
}

/// An empty pool whose reading is only being retained: the send is still refused, but the
/// statement describes the past and must not be quoted as "spent now".
///
/// The observation is placed inside the current provider day and at least a second old, so
/// the retention is entered by construction rather than by waiting. The first instant of a
/// provider day has no older same-day instant, so in that one-in-86,400 case the test waits
/// for a second to pass instead of asserting on an impossible state.
fn retained_provider_allowance(
    group: &str,
    model: &str,
    tag: &str,
) -> (ModelId, Arc<DailyRequestAllowance>) {
    let now = std::time::SystemTime::now();
    let day_start =
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(DayBoundary::Utc.day(now) * 86_400);
    let observed = if now
        .duration_since(day_start)
        .expect("now is after the epoch")
        >= std::time::Duration::from_secs(60)
    {
        now - std::time::Duration::from_secs(60)
    } else {
        day_start
    };
    if observed >= now {
        std::thread::sleep(std::time::Duration::from_millis(1_100));
    }
    let fixture = provider_empty(group, model, tag, observed, std::time::Duration::ZERO);
    assert_eq!(
        fixture.1.spent_today(),
        None,
        "fixture: a retained reading is neither spent nor shown to have room"
    );
    fixture
}

/// The retained refusal on the dispatch path. `o1` is offered and its send is refused by the
/// counter before anything leaves, so an attempt exists and its outcome says the provider's
/// pool is empty. That outcome inherits the basis of the reading behind it, and a statement
/// about the past cannot prove the domain is spent now -- so the local tier stays closed.
#[tokio::test]
async fn a_retained_provider_refusal_does_not_open_the_local_tier() {
    let (g1, google_a) = spent_allowance("google", "g1", "retained-a");
    let (g2, google_b) = spent_allowance("google", "g2", "retained-b");
    let (o1, openai) = retained_provider_allowance("openai", "o1", "retained");
    let pool = strict_pool_with(
        Some(200_000),
        O1Behavior::ProviderRefused,
        HashMap::from([(g1, google_a), (g2, google_b), (o1, openai)]),
    );
    let (sink, observer) = trace_sink();
    assert!(
        pool.route
            .execute(
                Request {
                    llm_request: text_request(None, "hello"),
                    ..Request::default()
                },
                Some(observer),
            )
            .await
            .is_err(),
        "one domain is spent and the other is only retained, so the pool's exhaustion is not provable"
    );
    assert_eq!(
        *pool.primary.lock().unwrap(),
        vec!["g1", "g2", "o1"],
        "the retained leg is offered and its client refuses the send"
    );
    assert!(
        pool.local.lock().unwrap().is_empty(),
        "a retained statement about the past must not buy the local tier"
    );
    assert!(
        !sink
            .lock()
            .unwrap()
            .last()
            .expect("a selection record")
            .terminal_crossed
    );
}

/// The control for the test above: the same shape with a current reading. The counter can
/// affirm the domain is spent *now*, so the boundary opens -- the correction narrows the
/// retained case and must not narrow this one.
#[tokio::test]
async fn a_current_provider_exhaustion_still_opens_the_local_tier() {
    let (g1, google_a) = spent_allowance("google", "g1", "fresh-a");
    let (g2, google_b) = spent_allowance("google", "g2", "fresh-b");
    let (o1, openai) = fresh_provider_allowance("openai", "o1", "fresh");
    let pool = strict_pool_with(
        Some(200_000),
        O1Behavior::ProviderRefused,
        HashMap::from([(g1, google_a), (g2, google_b), (o1, openai)]),
    );
    let (sink, observer) = trace_sink();
    let output = pool
        .route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("both domains are provably spent, so the local tier serves");
    assert_eq!(output.selected_model, "local");
    assert_eq!(*pool.local.lock().unwrap(), vec!["local"]);
    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert!(trace.terminal_crossed);
    assert_eq!(trace.terminal_reason, Some("primary_pool_exhausted"));
}

/// The same retained state when the leg is skipped on fit rather than dispatched. Both paths
/// have to agree: the evidence is the counter's, and it does not become stronger because the
/// request happened to reach the leg.
#[tokio::test]
async fn a_retained_reading_does_not_prove_an_undispatched_domain_spent() {
    let (g1, google_a) = spent_allowance("google", "g1", "retained-skip-a");
    let (g2, google_b) = spent_allowance("google", "g2", "retained-skip-b");
    let (o1, openai) = retained_provider_allowance("openai", "o1", "retained-skip");
    let pool = strict_pool_with(
        Some(1_000),
        O1Behavior::Serves,
        HashMap::from([(g1, google_a), (g2, google_b), (o1, openai)]),
    );
    let (sink, observer) = trace_sink();
    assert!(
        pool.route
            .execute(big_request(5_000), Some(observer))
            .await
            .is_err(),
        "a domain that skipped the request and can only be described from the past is not spent"
    );
    assert_eq!(
        *pool.primary.lock().unwrap(),
        vec!["g1", "g2"],
        "the leg cannot hold the request, so it is never called"
    );
    assert!(pool.local.lock().unwrap().is_empty());
    assert!(
        !sink
            .lock()
            .unwrap()
            .last()
            .expect("a selection record")
            .terminal_crossed
    );
}

/// The control for the skipped path: a current reading does prove the undispatched domain
/// spent, and the tier opens without the ineligible leg being called to find out.
#[tokio::test]
async fn a_current_reading_proves_an_undispatched_domain_spent() {
    let (g1, google_a) = spent_allowance("google", "g1", "fresh-skip-a");
    let (g2, google_b) = spent_allowance("google", "g2", "fresh-skip-b");
    let (o1, openai) = fresh_provider_allowance("openai", "o1", "fresh-skip");
    let pool = strict_pool_with(
        Some(1_000),
        O1Behavior::Serves,
        HashMap::from([(g1, google_a), (g2, google_b), (o1, openai)]),
    );
    let (sink, observer) = trace_sink();
    let output = pool
        .route
        .execute(big_request(5_000), Some(observer))
        .await
        .expect("the local tier serves");
    assert_eq!(output.selected_model, "local");
    assert_eq!(
        *pool.primary.lock().unwrap(),
        vec!["g1", "g2"],
        "the ineligible leg is never dispatched to learn its state"
    );
    assert_eq!(*pool.local.lock().unwrap(), vec!["local"]);
    let traces = sink.lock().unwrap();
    assert_eq!(
        traces.last().expect("a selection record").terminal_reason,
        Some("primary_pool_exhausted")
    );
}

/// An empty pool that has recovered: the provider reports headroom again, so the counter
/// admits the send and there is nothing left to retain.
fn recovering_provider_allowance(
    group: &str,
    model: &str,
    tag: &str,
) -> (ModelId, Arc<DailyRequestAllowance>) {
    let path = std::env::temp_dir().join(format!(
        "sy-provider-{}-{tag}-{}.json",
        std::process::id(),
        model
    ));
    let _ = std::fs::remove_file(&path);
    let allowance = DailyRequestAllowance::new_on(
        group,
        ModelId::from(model),
        1_000,
        path,
        DayBoundary::Utc,
    )
    .with_provider_tolerance(std::time::Duration::from_secs(60));
    allowance
        .reconcile(ProviderReading {
            used: 900,
            limit: 1_000,
            remaining: 100,
            observed_at: std::time::SystemTime::now(),
            reset_at: None,
        })
        .expect("the fixture's reading belongs to the current provider day");
    allowance
        .try_reserve()
        .expect("fixture: a recovered pool admits the send");
    assert_eq!(
        allowance.spent_today(),
        Some(false),
        "fixture: headroom is a readable counter with room, not an unknown"
    );
    (ModelId::from(model), Arc::new(allowance))
}

/// The recovery control: the same pool once the provider reports headroom again. The leg
/// serves, nothing is retained, and the local tier is never reached -- the correction withholds
/// a claim about the past, it does not make an exhausted pool sticky.
#[tokio::test]
async fn a_recovered_provider_reading_serves_without_spending_the_local_tier() {
    let (g1, google_a) = spent_allowance("google", "g1", "recovered-a");
    let (g2, google_b) = spent_allowance("google", "g2", "recovered-b");
    let (o1, openai) = recovering_provider_allowance("openai", "o1", "recovered");
    let pool = strict_pool_with(
        Some(200_000),
        O1Behavior::Serves,
        HashMap::from([(g1, google_a), (g2, google_b), (o1, openai)]),
    );
    let (sink, observer) = trace_sink();
    let output = pool
        .route
        .execute(
            Request {
                llm_request: text_request(None, "hello"),
                ..Request::default()
            },
            Some(observer),
        )
        .await
        .expect("the recovered leg serves");
    // The record names the leg that answered; the algorithm's own label for a passthrough is
    // its declared target, so the responder is read from the trace.
    assert_eq!(output.selected_model, "g1");
    assert!(
        pool.local.lock().unwrap().is_empty(),
        "a serving leg must not spend the local tier"
    );
    let traces = sink.lock().unwrap();
    let trace = traces.last().expect("a selection record");
    assert!(!trace.terminal_crossed);
    assert_eq!(trace.served.as_deref(), Some("o1"));
}
