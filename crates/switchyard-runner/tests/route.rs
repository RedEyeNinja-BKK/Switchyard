// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use libsy::RuntimeModels;
use switchyard_llm_client::{ClientRouter, RunObservation};
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
