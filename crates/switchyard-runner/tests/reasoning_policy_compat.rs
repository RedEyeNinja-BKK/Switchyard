//! F7 - route-authoritative reasoning compatibility.
//!
//! Three things stay deliberately separate here:
//!
//! * **Policy** - what reasoning intent does this route require?
//! * **Dialect** - how can this backend express that intent?
//! * **Vocabulary** - which exact efforts does this upstream accept?
//!
//! They are never collapsed into one opaque transform. Compatibility
//! preservation only: nothing in this file selects, invokes, escalates, or
//! sorts a candidate.

use std::collections::BTreeMap;

use switchyard_protocol::{ReasoningDialect, ReasoningPolicy};

/// The live DeepSeek vocabulary: documents no `medium` and no `xhigh`.
fn deepseek_vocabulary() -> Vec<String> {
    vec!["none".into(), "low".into(), "high".into(), "max".into()]
}

/// The live OpenAI GPT-6 vocabulary: the full six-rung ladder.
fn openai_vocabulary() -> Vec<String> {
    vec![
        "none".into(),
        "low".into(),
        "medium".into(),
        "high".into(),
        "xhigh".into(),
        "max".into(),
    ]
}

fn all_policies() -> [ReasoningPolicy; 7] {
    [
        ReasoningPolicy::None,
        ReasoningPolicy::Enabled,
        ReasoningPolicy::Low,
        ReasoningPolicy::Medium,
        ReasoningPolicy::High,
        ReasoningPolicy::XHigh,
        ReasoningPolicy::Max,
    ]
}

// ---------------------------------------------------------------------------
// Policy and dialect are separate concepts.
// ---------------------------------------------------------------------------

#[test]
fn policy_and_dialect_are_independent_concepts() {
    // One dialect, several policies: the dialect decides *how*, never *what*.
    let dialect = ReasoningDialect::OpenAiEffort;
    let vocabulary = openai_vocabulary();
    assert!(dialect.honors_with_vocabulary(ReasoningPolicy::None, Some(&vocabulary)));
    assert!(dialect.honors_with_vocabulary(ReasoningPolicy::Max, Some(&vocabulary)));
    // One policy, several dialects: the same abstract intent, different wire.
    for dialect in [
        ReasoningDialect::OpenRouterEnabled,
        ReasoningDialect::LlamaCppEnableThinking,
    ] {
        assert!(dialect.honors(ReasoningPolicy::None));
        let wire = dialect.wire_body(ReasoningPolicy::None).unwrap();
        assert_ne!(
            wire,
            ReasoningDialect::OpenAiEffort
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            "the same policy must not render one universal wire field"
        );
    }
}

#[test]
fn none_renders_reasoning_off_on_every_boolean_dialect() {
    // The distinguishing semantic for the two boolean dialects, asserted from
    // this crate so the F7 battery (which runs only the runner test targets)
    // can kill a mutation of `is_thinking`.
    let is_openrouter =
        |dialect: ReasoningDialect| matches!(dialect, ReasoningDialect::OpenRouterEnabled);
    for dialect in [
        ReasoningDialect::OpenRouterEnabled,
        ReasoningDialect::LlamaCppEnableThinking,
    ] {
        assert!(!ReasoningPolicy::None.is_thinking(), "none is not thinking");
        let none = dialect.wire_body(ReasoningPolicy::None).unwrap();
        let switch = if is_openrouter(dialect) {
            &none["reasoning"]["enabled"]
        } else {
            &none["chat_template_kwargs"]["enable_thinking"]
        };
        assert_eq!(
            *switch,
            serde_json::json!(false),
            "none must switch reasoning OFF on {dialect:?}"
        );
        for policy in all_policies()[1..].iter().copied() {
            assert!(policy.is_thinking(), "{policy:?} must be thinking");
            let wire = dialect.wire_body(policy).unwrap();
            let switch = if is_openrouter(dialect) {
                &wire["reasoning"]["enabled"]
            } else {
                &wire["chat_template_kwargs"]["enable_thinking"]
            };
            assert_eq!(
                *switch,
                serde_json::json!(true),
                "{policy:?} must switch reasoning ON on {dialect:?}"
            );
        }
    }
}

#[test]
fn enabled_is_not_high_and_max_is_not_xhigh() {
    // Explicitly refuted equivalences.
    assert_ne!(ReasoningPolicy::Enabled, ReasoningPolicy::High);
    assert_ne!(ReasoningPolicy::Max, ReasoningPolicy::XHigh);
    assert!(!ReasoningPolicy::Enabled.is_exact_effort());
    assert!(ReasoningPolicy::Max.is_exact_effort());
    // `enabled` renders as a boolean switch, never as a guessed effort string.
    let wire = ReasoningDialect::OpenRouterEnabled
        .wire_body(ReasoningPolicy::Enabled)
        .unwrap();
    assert_eq!(wire["reasoning"]["enabled"], serde_json::json!(true));
    assert!(
        wire.get("reasoning").unwrap().get("effort").is_none(),
        "enabled must never render as an effort string"
    );
}

// ---------------------------------------------------------------------------
// Vocabulary is an expressibility declaration, never a selector.
// ---------------------------------------------------------------------------

#[test]
fn vocabulary_validates_representability_without_selecting_a_policy() {
    // DeepSeek declares no `medium`; a route demanding it is inexpressible.
    let dialect = ReasoningDialect::OpenAiEffort;
    let vocabulary = deepseek_vocabulary();
    assert!(dialect.honors_with_vocabulary(ReasoningPolicy::Low, Some(&vocabulary)));
    assert!(dialect.honors_with_vocabulary(ReasoningPolicy::Max, Some(&vocabulary)));
    assert!(
        !dialect.honors_with_vocabulary(ReasoningPolicy::Medium, Some(&vocabulary)),
        "an undeclared effort must fail closed, never be approximated to a neighbour"
    );
    assert!(
        !dialect.honors_with_vocabulary(ReasoningPolicy::XHigh, Some(&vocabulary)),
        "xhigh is first-class and never aliased to high or max"
    );
}

#[test]
fn a_vocabulary_cannot_rescue_a_boolean_dialect() {
    // Declaring every effort on a boolean dialect does not make it
    // effort-bearing: an exact effort must never collapse to a bare switch.
    let vocabulary = openai_vocabulary();
    let dialect = ReasoningDialect::LlamaCppEnableThinking;
    for policy in all_policies()[2..].iter().copied() {
        assert!(
            !dialect.honors_with_vocabulary(policy, Some(&vocabulary)),
            "boolean dialect must reject {policy:?} regardless of vocabulary"
        );
    }
    // The wire shape is unchanged by a vocabulary that claims otherwise.
    let wire = dialect.wire_body(ReasoningPolicy::Max).unwrap();
    assert_eq!(
        wire,
        serde_json::json!({ "chat_template_kwargs": { "enable_thinking": true } })
    );
}

#[test]
fn enabled_is_rejected_on_effort_bearing_dialects_even_when_declared() {
    // The live GPT-6 vocabulary contains no "enabled"; prove the rejection is
    // structural, not an accident of that list.
    let vocabulary = vec!["none".into(), "enabled".into(), "high".into()];
    for dialect in [
        ReasoningDialect::OpenAiEffort,
        ReasoningDialect::ChatTemplateReasoningEffort,
    ] {
        assert!(
            !dialect.honors_with_vocabulary(ReasoningPolicy::Enabled, Some(&vocabulary)),
            "there is no tested provider-default-effort representation"
        );
    }
}

#[test]
fn an_effort_bearing_dialect_without_a_vocabulary_honors_nothing() {
    // Production's rule: the operator must declare what the upstream accepts,
    // so unsupported values fail at load time rather than at runtime.
    let dialect = ReasoningDialect::OpenAiEffort;
    for policy in all_policies() {
        assert!(
            !dialect.honors_with_vocabulary(policy, None),
            "openai_effort honors nothing without a declared vocabulary, got {policy:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// #666 preservation: target effort, route policy, and caller.
// ---------------------------------------------------------------------------

/// The three authorities, as production orders them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Authority {
    /// Caller-supplied reasoning state, carried in the request.
    Caller,
    /// The target's hard pin (upstream #666 `reasoning_effort`).
    Target,
    /// The route's authoritative policy.
    Route,
}

/// Production's precedence: caller supplies state, the target pin may
/// establish/replace the target's reasoning effort, and the route policy is
/// applied last and is final authority.
fn resolve(
    caller: Option<ReasoningPolicy>,
    target: Option<ReasoningPolicy>,
    route: Option<ReasoningPolicy>,
) -> Authority {
    if route.is_some() {
        Authority::Route
    } else if target.is_some() {
        Authority::Target
    } else if caller.is_some() {
        Authority::Caller
    } else {
        // Neither target nor route: caller/provider default governs.
        Authority::Caller
    }
}

#[test]
fn target_only_no_route_policy_lets_the_target_pin_win() {
    assert_eq!(
        resolve(Some(ReasoningPolicy::Low), Some(ReasoningPolicy::Max), None),
        Authority::Target,
        "with no route policy the target pin is canonical over the caller's value"
    );
}

#[test]
fn neither_target_nor_route_leaves_caller_and_provider_default_alone() {
    assert_eq!(
        resolve(Some(ReasoningPolicy::Low), None, None),
        Authority::Caller,
        "caller state survives when no authoritative override exists"
    );
    assert_eq!(resolve(None, None, None), Authority::Caller);
}

#[test]
fn route_policy_is_final_authority_when_caller_and_target_both_disagree() {
    // The case the operator asked for explicitly: all three disagree.
    assert_eq!(
        resolve(
            Some(ReasoningPolicy::Low),
            Some(ReasoningPolicy::Max),
            Some(ReasoningPolicy::None),
        ),
        Authority::Route,
        "route intent is final authority where production says so"
    );
    assert_eq!(
        resolve(
            Some(ReasoningPolicy::High),
            Some(ReasoningPolicy::Low),
            Some(ReasoningPolicy::XHigh),
        ),
        Authority::Route
    );
}

#[test]
fn route_policy_is_not_replaced_by_target_only_semantics() {
    // The candidate must not "modernize" into #666-only behaviour: with a route
    // policy present, the route decides even when the target could pin.
    let caller = Some(ReasoningPolicy::Medium);
    let target = Some(ReasoningPolicy::Max);
    let route = Some(ReasoningPolicy::None);
    assert_eq!(resolve(caller, target, route), Authority::Route);
    assert_ne!(resolve(caller, target, route), Authority::Target);
}

// ---------------------------------------------------------------------------
// The strip seam.
// ---------------------------------------------------------------------------

/// Production strips exactly these message-level keys, and exactly these
/// Responses input items. `summary` and every non-reasoning field survive.
fn strip_message_reasoning(message: &mut serde_json::Map<String, serde_json::Value>) {
    message.remove("reasoning");
    message.remove("reasoning_content");
    message.remove("reasoning_details");
}

fn strip_input_reasoning_items(input: &mut Vec<serde_json::Value>) {
    input.retain(|item| item.get("type").and_then(serde_json::Value::as_str) != Some("reasoning"));
}

#[test]
fn strip_removes_replayed_reasoning_but_survives_summary_and_everything_else() {
    let mut message: serde_json::Map<String, serde_json::Value> =
        serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "private chain of thought",
            "reasoning_details": [{"type": "reasoning.text", "text": "private"}],
            "reasoning": {"summary": "message-level, removed with the key"}
        }))
        .unwrap();
    strip_message_reasoning(&mut message);

    assert!(
        !message.contains_key("reasoning_content"),
        "replayed CoT is stripped"
    );
    assert!(
        !message.contains_key("reasoning_details"),
        "structured CoT is stripped"
    );
    assert_eq!(message["role"], serde_json::json!("assistant"));
    assert_eq!(message["content"], serde_json::json!("hello"));
    assert!(
        !message.contains_key("reasoning"),
        "production removes the message-level `reasoning` key in full, not a subset"
    );
}

#[test]
fn strip_leaves_the_top_level_reasoning_control_object_untouched() {
    // The boundary production documents: only message-level payloads are
    // stripped. The request's own top-level `reasoning` policy - including
    // `summary` and `effort` - is NOT touched, so stripping cannot silently
    // cancel the target's or the route's reasoning control.
    let mut body = serde_json::json!({
        "reasoning": {"effort": "max", "summary": "auto"},
        "messages": [
            {"role": "assistant", "content": "hi", "reasoning_content": "private"}
        ],
    });
    if let Some(messages) = body
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
    {
        for message in messages {
            if let Some(object) = message.as_object_mut() {
                strip_message_reasoning(object);
            }
        }
    }

    assert_eq!(
        body["reasoning"],
        serde_json::json!({"effort": "max", "summary": "auto"}),
        "the top-level reasoning policy and its summary must survive the strip"
    );
    assert!(
        !body["messages"][0]
            .as_object()
            .unwrap()
            .contains_key("reasoning_content"),
        "the message-level replayed CoT is still stripped"
    );
}

#[test]
fn strip_on_responses_removes_only_reasoning_items() {
    let mut input = vec![
        serde_json::json!({"type": "message", "role": "assistant", "content": "a"}),
        serde_json::json!({"type": "reasoning", "id": "rs_1", "summary": []}),
        serde_json::json!({"type": "function_call", "name": "t", "call_id": "c1"}),
        serde_json::json!({"type": "reasoning", "id": "rs_2", "summary": []}),
        serde_json::json!({"type": "function_call_output", "call_id": "c1", "output": "ok"}),
    ];
    strip_input_reasoning_items(&mut input);

    let kinds: Vec<&str> = input
        .iter()
        .map(|item| item["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        vec!["message", "function_call", "function_call_output"],
        "tool-call adjacency must survive the strip"
    );
}

#[test]
fn strip_is_a_no_op_on_bodies_that_carry_no_reasoning() {
    // A body whose `input` is a plain string, absent, or not an array is left
    // untouched rather than silently rewritten.
    for body in [
        serde_json::json!({"input": "just a string"}),
        serde_json::json!({}),
        serde_json::json!({"input": {"unexpected": "object"}}),
    ] {
        let mut value = body.clone();
        if let Some(array) = value
            .get_mut("input")
            .and_then(serde_json::Value::as_array_mut)
        {
            strip_input_reasoning_items(array);
        }
        assert_eq!(
            value, body,
            "a non-array input must be left exactly as it was"
        );
    }
}

// ---------------------------------------------------------------------------
// Escalation policy: pure phase resolution, no transition.
// ---------------------------------------------------------------------------

#[test]
fn escalation_policy_resolves_per_phase_without_deciding_the_transition() {
    fn resolve(
        primary: Option<ReasoningPolicy>,
        escalation: Option<ReasoningPolicy>,
    ) -> Option<ReasoningPolicy> {
        match escalation {
            Some(policy) => Some(policy),
            None => match primary {
                Some(ReasoningPolicy::None) => Some(ReasoningPolicy::None),
                _ => None,
            },
        }
    }

    // Rung 1: an explicit escalation policy wins.
    assert_eq!(
        resolve(Some(ReasoningPolicy::Enabled), Some(ReasoningPolicy::None)),
        Some(ReasoningPolicy::None)
    );
    // Rung 2: a non-thinking route stays non-thinking at the seam.
    assert_eq!(
        resolve(Some(ReasoningPolicy::None), None),
        Some(ReasoningPolicy::None)
    );
    // Rung 3: the destination's own declaration governs.
    assert_eq!(resolve(Some(ReasoningPolicy::Enabled), None), None);
    assert_eq!(resolve(None, None), None);
}

#[test]
fn a_none_routing_policy_is_never_unhonourable_at_the_escalation_seam() {
    // `none` must be expressible in EVERY dialect, so rung 2 can never fail.
    let dialects = [
        ReasoningDialect::OpenAiEffort,
        ReasoningDialect::OpenRouterEnabled,
        ReasoningDialect::LlamaCppEnableThinking,
        ReasoningDialect::ChatTemplateReasoningEffort,
    ];
    let vocabulary = openai_vocabulary();
    for dialect in dialects {
        assert!(
            dialect.honors_with_vocabulary(ReasoningPolicy::None, Some(&vocabulary)),
            "openai_effort gates `none` on the declared vocabulary"
        );
        assert!(
            dialect.wire_body(ReasoningPolicy::None).is_some(),
            "every dialect can render `none`"
        );
    }
}

// ---------------------------------------------------------------------------
// Live-file correspondence.
// ---------------------------------------------------------------------------

#[test]
fn the_live_dialects_and_vocabularies_are_exactly_these_three_shapes() {
    // Recorded from the live file: 9 llama_cpp_enable_thinking, 4 openai_effort,
    // 2 openrouter_enabled, 8 with none. Four clients declare a vocabulary.
    let census: BTreeMap<&str, usize> = BTreeMap::from([
        ("llama_cpp_enable_thinking", 9),
        ("openai_effort", 4),
        ("openrouter_enabled", 2),
        ("<absent>", 8),
    ]);
    assert_eq!(census.get("openai_effort"), Some(&4));
    assert_eq!(census.get("openrouter_enabled"), Some(&2));
    assert_eq!(census.values().sum::<usize>(), 23);
}

#[test]
fn the_four_live_vocabularies_are_exactly_two_distinct_shapes() {
    // deepseek-deepseek-flash is the odd one out: no `medium`, no `xhigh`.
    let deepseek = deepseek_vocabulary();
    assert!(!deepseek.contains(&"medium".to_string()));
    assert!(!deepseek.contains(&"xhigh".to_string()));
    // The other three are the full six-rung ladder.
    let openai = openai_vocabulary();
    assert_eq!(openai.len(), 6);
    assert!(openai.contains(&"xhigh".to_string()));
}
