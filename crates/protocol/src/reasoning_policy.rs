// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Route-authoritative reasoning policy: an abstract route-level intent.
//!
//! `Responses is the common transport, not the reasoning dialect.` A route's
//! [`ReasoningPolicy`] states **what** execution mode is forced on every
//! candidate the route can reach; each target's backend translates that intent
//! into the wire control its endpoint actually understands (OpenAI-family
//! `reasoning.effort`, OpenRouter `reasoning.enabled`, llama.cpp
//! `chat_template_kwargs.*`). Routes never carry provider syntax.

use std::fmt;

/// Reasoning execution mode a route forces onto every reachable target.
///
/// Declared per route as `reasoning_policy = "none" | "enabled" | "low" |
/// "medium" | "high" | "xhigh" | "max"`. Absent means the route carries no
/// policy and routing behaves exactly as before (callers and target pins
/// decide).
///
/// The contract has exactly three tiers:
///
/// * [`ReasoningPolicy::None`] — reasoning must be OFF.
/// * [`ReasoningPolicy::Enabled`] — reasoning must be ON, effort unspecified
///   (the only policy boolean-only backends can faithfully express besides
///   `none`).
/// * `low|medium|high|xhigh|max` — an EXACT effort assertion. Never silently
///   treated as generic "thinking on": a dialect that cannot express the
///   exact level must reject the route at load time (see
///   [`ReasoningDialect::honors`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningPolicy {
    /// Forbid reasoning: no reasoning may be produced or billed.
    None,
    /// Force reasoning ON without asserting a concrete effort level (the
    /// provider's default effort applies). Boolean-only backends express this
    /// exactly; effort-bearing backends need an explicit native
    /// "enabled, default effort" representation to honor it.
    Enabled,
    /// Permit reasoning with a bounded effort budget.
    Low,
    /// Permit reasoning with a bounded effort budget.
    Medium,
    /// Permit reasoning with a bounded effort budget.
    High,
    /// Permit reasoning at an extra-high effort budget. First-class: never
    /// aliased or silently mapped to `high` or `max` — a backend either
    /// declares `xhigh` in its accepted vocabulary or the route is rejected.
    XHigh,
    /// Permit reasoning at the provider's maximum effort.
    Max,
}

impl ReasoningPolicy {
    /// Canonical wire-neutral name of the policy (also the config spelling).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Enabled => "enabled",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// Whether this policy demands that reasoning is enabled at all.
    pub const fn is_thinking(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether this policy asserts an EXACT effort level (`enabled` does not:
    /// it turns reasoning on without naming an effort).
    pub const fn is_exact_effort(self) -> bool {
        matches!(
            self,
            Self::Low | Self::Medium | Self::High | Self::XHigh | Self::Max
        )
    }

    /// Parses the config spelling; unknown values are a configuration error.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "none" => Some(Self::None),
            "enabled" => Some(Self::Enabled),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

impl fmt::Display for ReasoningPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The reasoning-control dialect an upstream speaks: how an abstract
/// [`ReasoningPolicy`] is expressed on that provider's wire.
///
/// `Responses is the common transport, not the reasoning dialect`: two
/// clients can share the `openai_responses` wire format yet express reasoning
/// control with entirely different keys. Declared per llm client
/// (`reasoning_dialect = ...`); the send seam merges
/// [`ReasoningDialect::wire_body`] into the outbound body, authoritative over
/// caller values.
///
/// Expressibility invariant: `honors(policy) == true` only when the dialect
/// can faithfully express the EXACT abstract policy — an exact effort is
/// never collapsed to a generic boolean "on".
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReasoningDialect {
    /// `reasoning.effort = "<policy>"` (DeepSeek, OpenAI, and
    /// OpenAI-compatible Responses endpoints). Exact-effort dialect: honors
    /// only policies present in the client's declared `reasoning_efforts`
    /// vocabulary. `enabled` is NOT honored by default: expressing
    /// "reasoning on, provider-default effort" requires a deliberate native
    /// representation that must be explicitly modeled and tested first.
    #[serde(rename = "openai_effort")]
    OpenAiEffort,
    /// `reasoning = { enabled = <bool> }` (OpenRouter). Boolean switch only:
    /// honors exactly `none` and `enabled`; rejects every exact effort.
    #[serde(rename = "openrouter_enabled")]
    OpenRouterEnabled,
    /// `chat_template_kwargs.enable_thinking = <bool>` (llama.cpp-derived Qwen
    /// template endpoints). Boolean switch only: honors exactly `none` and
    /// `enabled`; rejects every exact effort. This is the HTPC contract
    /// (both shapes observed live in the Gate 0 decision snapshot).
    #[serde(rename = "llama_cpp_enable_thinking")]
    LlamaCppEnableThinking,
    /// Effort-bearing chat-template control (llama.cpp Qwen3 template
    /// endpoints whose thinking side takes `chat_template_kwargs.reasoning_effort`):
    /// `none` → `enable_thinking = false` (the observed NT shape);
    /// an exact effort → `reasoning_effort = <policy>`, honored only when the
    /// policy is present in the client's declared `reasoning_efforts`
    /// vocabulary. `enabled` is NOT honored: the endpoint has no tested
    /// "enabled, default effort" representation. This is the ComfyNinja Q3/Q4
    /// contract observed live in the Gate 0 decision snapshot.
    #[serde(rename = "chat_template_reasoning_effort")]
    ChatTemplateReasoningEffort,
}

impl ReasoningDialect {
    /// The JSON object this dialect merges into the outbound body to express
    /// `policy` (e.g. `{"reasoning": {"effort": "none"}}`).
    ///
    /// Callers must check [`Self::honors`] first — load-time validation does.
    /// Returns `None` for a policy the dialect cannot faithfully express; the
    /// send seam treats that as fail-closed, never as a silent downgrade.
    pub fn wire_body(self, policy: ReasoningPolicy) -> Option<serde_json::Value> {
        match self {
            Self::OpenAiEffort => {
                Some(serde_json::json!({ "reasoning": { "effort": policy.as_str() } }))
            }
            Self::OpenRouterEnabled => Some(serde_json::json!(
                { "reasoning": { "enabled": policy.is_thinking() } }
            )),
            Self::LlamaCppEnableThinking => Some(serde_json::json!(
                { "chat_template_kwargs": { "enable_thinking": policy.is_thinking() } }
            )),
            Self::ChatTemplateReasoningEffort => {
                if policy.is_exact_effort() {
                    Some(serde_json::json!(
                        { "chat_template_kwargs": { "reasoning_effort": policy.as_str() } }
                    ))
                } else {
                    // `none` uses the observed boolean NT shape; `enabled`
                    // never reaches here (not honored — checked by
                    // `honors`/load-time validation).
                    Some(serde_json::json!(
                        { "chat_template_kwargs": { "enable_thinking": false } }
                    ))
                }
            }
        }
    }

    /// Whether this dialect can faithfully express the EXACT `policy`.
    ///
    /// `openai_effort` is fully vocabulary-gated: EVERY policy it expresses —
    /// `none` included — must be present in the client's declared
    /// `reasoning_efforts` list, because the endpoint's accepted values are
    /// exactly what the operator declared. `chat_template_reasoning_effort`
    /// has a fixed, observed `none` shape (`enable_thinking = false`) and
    /// gates only its exact-effort policies on the vocabulary. The boolean
    /// dialects honor exactly `none` and `enabled` and reject every exact
    /// effort: collapsing an exact effort to a bare switch would silently
    /// change what the route author asserted. `enabled` is never honored on
    /// effort-bearing dialects this cycle (no tested provider-default-effort
    /// representation).
    pub fn honors_with_vocabulary(
        self,
        policy: ReasoningPolicy,
        vocabulary: Option<&[String]>,
    ) -> bool {
        match policy {
            ReasoningPolicy::None => match self {
                Self::OpenAiEffort => {
                    vocabulary.is_some_and(|efforts| efforts.iter().any(|effort| effort == "none"))
                }
                Self::OpenRouterEnabled
                | Self::LlamaCppEnableThinking
                | Self::ChatTemplateReasoningEffort => true,
            },
            ReasoningPolicy::Enabled => {
                matches!(self, Self::OpenRouterEnabled | Self::LlamaCppEnableThinking)
            }
            ReasoningPolicy::Low
            | ReasoningPolicy::Medium
            | ReasoningPolicy::High
            | ReasoningPolicy::XHigh
            | ReasoningPolicy::Max => match self {
                Self::OpenRouterEnabled | Self::LlamaCppEnableThinking => false,
                Self::OpenAiEffort | Self::ChatTemplateReasoningEffort => vocabulary
                    .is_some_and(|efforts| efforts.iter().any(|effort| effort == policy.as_str())),
            },
        }
    }

    /// Vocabulary-free expressibility check (unit-level; config validation
    /// always uses [`Self::honors_with_vocabulary`]).
    pub fn honors(self, policy: ReasoningPolicy) -> bool {
        self.honors_with_vocabulary(policy, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    #[test]
    fn parse_round_trips_every_value() {
        for policy in all_policies() {
            assert_eq!(ReasoningPolicy::parse(policy.as_str()), Some(policy));
        }
    }

    #[test]
    fn xhigh_is_first_class_and_never_aliased() {
        let xhigh = ReasoningPolicy::parse("xhigh").unwrap();
        assert_eq!(xhigh, ReasoningPolicy::XHigh);
        assert_ne!(xhigh, ReasoningPolicy::High);
        assert_ne!(xhigh, ReasoningPolicy::Max);
        assert_eq!(xhigh.as_str(), "xhigh");
    }

    #[test]
    fn parse_rejects_unknown_and_empty() {
        assert_eq!(ReasoningPolicy::parse(""), None);
        assert_eq!(ReasoningPolicy::parse("true"), None);
        assert_eq!(ReasoningPolicy::parse("None"), None, "config is lowercase");
        assert_eq!(ReasoningPolicy::parse("on"), None);
        assert_eq!(ReasoningPolicy::parse("default"), None);
    }

    #[test]
    fn is_thinking_partitions_none_from_the_rest() {
        assert!(!ReasoningPolicy::None.is_thinking());
        for policy in all_policies()[1..].iter().copied() {
            assert!(policy.is_thinking(), "{policy:?} must be thinking");
        }
    }

    #[test]
    fn exact_effort_excludes_none_and_enabled() {
        assert!(!ReasoningPolicy::None.is_exact_effort());
        assert!(!ReasoningPolicy::Enabled.is_exact_effort());
        for policy in all_policies()[2..].iter().copied() {
            assert!(policy.is_exact_effort(), "{policy:?} must be exact effort");
        }
    }

    #[test]
    fn wire_bodies_match_the_gate0_observed_shapes() {
        // Shapes recorded in the 2026-09-17 Gate 0 decision snapshot — the
        // existing per-target extra_body pins ARE each dialect's wire oracle.
        assert_eq!(
            ReasoningDialect::OpenAiEffort
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "reasoning": { "effort": "none" } })
        );
        assert_eq!(
            ReasoningDialect::OpenAiEffort
                .wire_body(ReasoningPolicy::Medium)
                .unwrap(),
            json!({ "reasoning": { "effort": "medium" } })
        );
        assert_eq!(
            ReasoningDialect::OpenAiEffort
                .wire_body(ReasoningPolicy::XHigh)
                .unwrap(),
            json!({ "reasoning": { "effort": "xhigh" } })
        );
        assert_eq!(
            ReasoningDialect::OpenRouterEnabled
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "reasoning": { "enabled": false } })
        );
        assert_eq!(
            ReasoningDialect::OpenRouterEnabled
                .wire_body(ReasoningPolicy::Enabled)
                .unwrap(),
            json!({ "reasoning": { "enabled": true } })
        );
        assert_eq!(
            ReasoningDialect::LlamaCppEnableThinking
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "chat_template_kwargs": { "enable_thinking": false } })
        );
        assert_eq!(
            ReasoningDialect::LlamaCppEnableThinking
                .wire_body(ReasoningPolicy::Enabled)
                .unwrap(),
            json!({ "chat_template_kwargs": { "enable_thinking": true } })
        );
        // ComfyNinja-shaped dialect: effort-bearing thinking side, boolean NT side.
        assert_eq!(
            ReasoningDialect::ChatTemplateReasoningEffort
                .wire_body(ReasoningPolicy::Medium)
                .unwrap(),
            json!({ "chat_template_kwargs": { "reasoning_effort": "medium" } })
        );
        assert_eq!(
            ReasoningDialect::ChatTemplateReasoningEffort
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "chat_template_kwargs": { "enable_thinking": false } })
        );
    }

    #[test]
    fn dialect_deserializes_config_spellings() {
        assert_eq!(
            serde_json::from_value::<ReasoningDialect>(json!("openai_effort")).unwrap(),
            ReasoningDialect::OpenAiEffort
        );
        assert_eq!(
            serde_json::from_value::<ReasoningDialect>(json!("openrouter_enabled")).unwrap(),
            ReasoningDialect::OpenRouterEnabled
        );
        assert_eq!(
            serde_json::from_value::<ReasoningDialect>(json!("llama_cpp_enable_thinking")).unwrap(),
            ReasoningDialect::LlamaCppEnableThinking
        );
        assert_eq!(
            serde_json::from_value::<ReasoningDialect>(json!("chat_template_reasoning_effort"))
                .unwrap(),
            ReasoningDialect::ChatTemplateReasoningEffort
        );
        assert!(serde_json::from_value::<ReasoningDialect>(json!("chat_template")).is_err());
    }

    // ---- Steering §3 focused semantic battery (protocol-level rows) ---------

    #[test]
    fn s03_row1_2_xhigh_and_enabled_parse_and_round_trip() {
        assert_eq!(
            ReasoningPolicy::parse("xhigh"),
            Some(ReasoningPolicy::XHigh)
        );
        assert_eq!(
            ReasoningPolicy::parse("enabled"),
            Some(ReasoningPolicy::Enabled)
        );
        assert_eq!(ReasoningPolicy::XHigh.as_str(), "xhigh");
        assert_eq!(ReasoningPolicy::Enabled.as_str(), "enabled");
    }

    #[test]
    fn s03_row4_5_6_openrouter_boolean_semantics() {
        let d = ReasoningDialect::OpenRouterEnabled;
        assert!(d.honors(ReasoningPolicy::None));
        assert!(d.honors(ReasoningPolicy::Enabled));
        assert_eq!(
            d.wire_body(ReasoningPolicy::None).unwrap(),
            json!({ "reasoning": { "enabled": false } })
        );
        assert_eq!(
            d.wire_body(ReasoningPolicy::Enabled).unwrap(),
            json!({ "reasoning": { "enabled": true } })
        );
        for policy in [
            ReasoningPolicy::Low,
            ReasoningPolicy::Medium,
            ReasoningPolicy::High,
            ReasoningPolicy::XHigh,
            ReasoningPolicy::Max,
        ] {
            assert!(
                !d.honors(policy),
                "openrouter must reject exact effort {policy:?}"
            );
        }
    }

    #[test]
    fn s03_row7_8_9_llamacpp_boolean_semantics() {
        let d = ReasoningDialect::LlamaCppEnableThinking;
        assert!(d.honors(ReasoningPolicy::None));
        assert!(d.honors(ReasoningPolicy::Enabled));
        assert_eq!(
            d.wire_body(ReasoningPolicy::Enabled).unwrap(),
            json!({ "chat_template_kwargs": { "enable_thinking": true } })
        );
        for policy in [
            ReasoningPolicy::Low,
            ReasoningPolicy::Medium,
            ReasoningPolicy::High,
            ReasoningPolicy::XHigh,
            ReasoningPolicy::Max,
        ] {
            assert!(
                !d.honors(policy),
                "llama.cpp boolean must reject exact effort {policy:?}"
            );
        }
    }

    #[test]
    fn s03_row13_14_comfyninja_wire_shapes_unchanged() {
        let d = ReasoningDialect::ChatTemplateReasoningEffort;
        let vocab: Vec<String> = vec!["none".into(), "medium".into()];
        assert_eq!(
            d.wire_body(ReasoningPolicy::Medium).unwrap(),
            json!({ "chat_template_kwargs": { "reasoning_effort": "medium" } })
        );
        assert_eq!(
            d.wire_body(ReasoningPolicy::None).unwrap(),
            json!({ "chat_template_kwargs": { "enable_thinking": false } })
        );
        assert!(d.honors_with_vocabulary(ReasoningPolicy::Medium, Some(&vocab)));
        assert!(d.honors_with_vocabulary(ReasoningPolicy::None, Some(&vocab)));
    }

    #[test]
    fn s03_row15_16_qwen_declared_xhigh_and_family_rejection() {
        // A Qwen/Unsloth-style family MAY declare xhigh where operator-validated...
        let qwen: Vec<String> = vec![
            "none".into(),
            "enabled".into(),
            "medium".into(),
            "xhigh".into(),
        ];
        let comfy = ReasoningDialect::ChatTemplateReasoningEffort;
        assert!(comfy.honors_with_vocabulary(ReasoningPolicy::XHigh, Some(&qwen)));
        // ...and a family for which high is invalid must reject high.
        assert!(!comfy.honors_with_vocabulary(ReasoningPolicy::High, Some(&qwen)));
        let oa = ReasoningDialect::OpenAiEffort;
        assert!(oa.honors_with_vocabulary(ReasoningPolicy::XHigh, Some(&qwen)));
        assert!(!oa.honors_with_vocabulary(ReasoningPolicy::High, Some(&qwen)));
    }

    #[test]
    fn s03_enabled_never_honored_on_effort_bearing_dialects() {
        let vocab: Vec<String> = vec!["none".into(), "enabled".into(), "high".into()];
        for dialect in [
            ReasoningDialect::OpenAiEffort,
            ReasoningDialect::ChatTemplateReasoningEffort,
        ] {
            assert!(
                !dialect.honors_with_vocabulary(ReasoningPolicy::Enabled, Some(&vocab)),
                "enabled must be rejected on {dialect:?} this cycle even if present in the vocabulary"
            );
        }
    }

    #[test]
    fn honors_rejects_exact_efforts_without_vocabulary() {
        // Without a declared vocabulary: openai_effort honors NOTHING (even
        // `none` is vocabulary-gated); chat_template_reasoning_effort honors
        // its fixed `none` shape; boolean dialects honor none+enabled.
        let oa = ReasoningDialect::OpenAiEffort;
        assert!(
            !oa.honors(ReasoningPolicy::None),
            "openai_effort `none` is vocabulary-gated"
        );
        for policy in all_policies()[1..].iter().copied() {
            assert!(
                !oa.honors(policy),
                "openai_effort must not honor {policy:?} without a vocabulary"
            );
        }
        let comfy = ReasoningDialect::ChatTemplateReasoningEffort;
        assert!(
            comfy.honors(ReasoningPolicy::None),
            "comfy `none` has a fixed observed shape"
        );
        for policy in all_policies()[1..].iter().copied() {
            assert!(
                !comfy.honors(policy),
                "comfy must not honor {policy:?} without a vocabulary"
            );
        }
        for dialect in [
            ReasoningDialect::OpenRouterEnabled,
            ReasoningDialect::LlamaCppEnableThinking,
        ] {
            assert!(dialect.honors(ReasoningPolicy::None));
            assert!(dialect.honors(ReasoningPolicy::Enabled));
            for policy in all_policies()[2..].iter().copied() {
                assert!(
                    !dialect.honors(policy),
                    "boolean {dialect:?} must never honor exact effort {policy:?}"
                );
            }
        }
    }

    #[test]
    fn honors_with_vocabulary_is_exact() {
        let deepseek: Vec<String> = vec!["none".into(), "low".into(), "high".into(), "max".into()];
        let dialect = ReasoningDialect::OpenAiEffort;
        // In vocabulary: honored.
        assert!(dialect.honors_with_vocabulary(ReasoningPolicy::None, Some(&deepseek)));
        assert!(dialect.honors_with_vocabulary(ReasoningPolicy::Low, Some(&deepseek)));
        assert!(dialect.honors_with_vocabulary(ReasoningPolicy::High, Some(&deepseek)));
        // DeepSeek documents no `medium`: must be rejected, never approximated.
        assert!(!dialect.honors_with_vocabulary(ReasoningPolicy::Medium, Some(&deepseek)));
        // `xhigh` honored ONLY when explicitly declared (first-class, per family).
        assert!(!dialect.honors_with_vocabulary(ReasoningPolicy::XHigh, Some(&deepseek)));
        let qwen: Vec<String> = vec![
            "none".into(),
            "enabled".into(),
            "medium".into(),
            "xhigh".into(),
        ];
        let comfy = ReasoningDialect::ChatTemplateReasoningEffort;
        assert!(comfy.honors_with_vocabulary(ReasoningPolicy::XHigh, Some(&qwen)));
        // `enabled` is never honored on effort-bearing dialects in this cycle.
        assert!(!comfy.honors_with_vocabulary(ReasoningPolicy::Enabled, Some(&qwen)));
        assert!(!dialect.honors_with_vocabulary(ReasoningPolicy::Enabled, Some(&qwen)));
    }
}
