// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Route-authoritative reasoning policy: an abstract route-level intent.
//!
//! `Responses is the common transport, not the reasoning dialect.` A route's
//! [`ReasoningPolicy`] states **what** execution mode is forced on every
//! candidate the route can reach; each target's backend translates that intent
//! into the wire control its endpoint actually understands. Routes never carry
//! provider syntax.

use std::fmt;

/// Reasoning execution mode a route forces onto every reachable target.
///
/// The contract has exactly three tiers:
///
/// * [`ReasoningPolicy::None`] — reasoning must be OFF.
/// * [`ReasoningPolicy::Enabled`] — reasoning must be ON, effort unspecified.
/// * `low|medium|high|xhigh|max` — an EXACT effort assertion, never silently
///   treated as generic "thinking on": a dialect that cannot express the exact
///   level must reject the route at load time (see
///   [`ReasoningDialect::honors_with_vocabulary`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningPolicy {
    /// Forbid reasoning: no reasoning may be produced or billed.
    None,
    /// Force reasoning ON without asserting a concrete effort level.
    Enabled,
    /// Permit reasoning with a bounded effort budget.
    Low,
    /// Permit reasoning with a bounded effort budget.
    Medium,
    /// Permit reasoning with a bounded effort budget.
    High,
    /// Permit reasoning at an extra-high effort budget. First-class: never
    /// aliased or silently mapped to `high` or `max`.
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

    /// Whether this policy asserts an EXACT effort level (`enabled` does not).
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
/// Policy answers *what reasoning intent does this route require?*; dialect
/// answers *how can this backend express that intent?* They stay separately
/// testable and are never collapsed into one opaque transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReasoningDialect {
    /// `reasoning.effort = "<policy>"`. Exact-effort dialect: honors only
    /// policies present in the client's declared `reasoning_efforts`
    /// vocabulary. `enabled` is NOT honored by default.
    #[serde(rename = "openai_effort")]
    OpenAiEffort,
    /// `reasoning = { enabled = <bool> }` (OpenRouter). Boolean switch only:
    /// honors exactly `none` and `enabled`; rejects every exact effort.
    #[serde(rename = "openrouter_enabled")]
    OpenRouterEnabled,
    /// `chat_template_kwargs.enable_thinking = <bool>` (llama.cpp-derived Qwen
    /// template endpoints). Boolean switch only.
    #[serde(rename = "llama_cpp_enable_thinking")]
    LlamaCppEnableThinking,
    /// Effort-bearing chat-template control. `none` → `enable_thinking = false`;
    /// an exact effort → `reasoning_effort = <policy>`, honored only when the
    /// policy is present in the declared vocabulary.
    #[serde(rename = "chat_template_reasoning_effort")]
    ChatTemplateReasoningEffort,
}

impl ReasoningDialect {
    /// The JSON object this dialect merges into the outbound body to express
    /// `policy`. Returns `None` for a policy the dialect cannot faithfully
    /// express; callers must check [`Self::honors_with_vocabulary`] first, and
    /// a `None` there is fail-closed, never a silent downgrade.
    pub fn wire_body(self, policy: ReasoningPolicy) -> Option<serde_json::Value> {
        match self {
            Self::OpenAiEffort => {
                Some(serde_json::json!({ "reasoning": { "effort": policy.as_str() } }))
            }
            Self::OpenRouterEnabled => {
                Some(serde_json::json!({ "reasoning": { "enabled": policy.is_thinking() } }))
            }
            Self::LlamaCppEnableThinking => Some(
                serde_json::json!({ "chat_template_kwargs": { "enable_thinking": policy.is_thinking() } }),
            ),
            Self::ChatTemplateReasoningEffort => {
                if policy.is_exact_effort() {
                    Some(
                        serde_json::json!({ "chat_template_kwargs": { "reasoning_effort": policy.as_str() } }),
                    )
                } else {
                    Some(
                        serde_json::json!({ "chat_template_kwargs": { "enable_thinking": false } }),
                    )
                }
            }
        }
    }

    /// Whether this dialect can faithfully express the EXACT `policy`.
    ///
    /// `openai_effort` is fully vocabulary-gated: EVERY policy it expresses —
    /// `none` included — must be present in the declared `reasoning_efforts`
    /// list. `chat_template_reasoning_effort` has a fixed, observed `none`
    /// shape and gates only its exact-effort policies. The boolean dialects
    /// honor exactly `none` and `enabled` and reject every exact effort.
    /// `enabled` is never honored on effort-bearing dialects.
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

    /// Whether this dialect requires an operator-declared `reasoning_efforts`
    /// vocabulary before any policy may be honored.
    pub const fn is_effort_bearing(self) -> bool {
        matches!(self, Self::OpenAiEffort | Self::ChatTemplateReasoningEffort)
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
    fn parse_rejects_unknown_and_empty() {
        assert_eq!(ReasoningPolicy::parse(""), None);
        assert_eq!(ReasoningPolicy::parse("true"), None);
        assert_eq!(ReasoningPolicy::parse("None"), None);
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
    fn is_thinking_is_what_the_boolean_wire_bodies_actually_render() {
        // The distinguishing semantic for the two boolean dialects: `none`
        // renders a FALSE switch, every other policy a TRUE switch. This is
        // exactly what "none leaves reasoning enabled" would break, and it is
        // the property every live `none` route depends on.
        for dialect in [
            ReasoningDialect::OpenRouterEnabled,
            ReasoningDialect::LlamaCppEnableThinking,
        ] {
            let is_openrouter = matches!(dialect, ReasoningDialect::OpenRouterEnabled);
            let none = dialect.wire_body(ReasoningPolicy::None).unwrap();
            let none_switch = if is_openrouter {
                &none["reasoning"]["enabled"]
            } else {
                &none["chat_template_kwargs"]["enable_thinking"]
            };
            assert_eq!(
                *none_switch,
                serde_json::json!(false),
                "none must switch reasoning OFF on {dialect:?}"
            );

            for policy in all_policies()[1..].iter().copied() {
                let wire = dialect.wire_body(policy).unwrap();
                let switch = if is_openrouter {
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
    fn exact_effort_excludes_none_and_enabled() {
        assert!(!ReasoningPolicy::None.is_exact_effort());
        assert!(!ReasoningPolicy::Enabled.is_exact_effort());
        for policy in all_policies()[2..].iter().copied() {
            assert!(policy.is_exact_effort(), "{policy:?} must be exact effort");
        }
    }

    #[test]
    fn xhigh_is_first_class_and_never_aliased() {
        let xhigh = ReasoningPolicy::parse("xhigh").unwrap();
        assert_ne!(xhigh, ReasoningPolicy::High);
        assert_ne!(xhigh, ReasoningPolicy::Max);
        assert_eq!(xhigh.as_str(), "xhigh");
    }

    #[test]
    fn wire_bodies_match_the_gate0_observed_shapes() {
        assert_eq!(
            ReasoningDialect::OpenAiEffort
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "reasoning": { "effort": "none" } })
        );
        assert_eq!(
            ReasoningDialect::OpenAiEffort
                .wire_body(ReasoningPolicy::Max)
                .unwrap(),
            json!({ "reasoning": { "effort": "max" } })
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
        assert_eq!(
            ReasoningDialect::ChatTemplateReasoningEffort
                .wire_body(ReasoningPolicy::Max)
                .unwrap(),
            json!({ "chat_template_kwargs": { "reasoning_effort": "max" } })
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
        assert!(serde_json::from_value::<ReasoningDialect>(json!("llama_cpp")).is_err());
    }

    #[test]
    fn boolean_dialects_honor_none_and_enabled_and_reject_every_effort() {
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
    fn enabled_is_never_honored_on_effort_bearing_dialects() {
        let vocab: Vec<String> = vec!["none".into(), "enabled".into(), "high".into()];
        for dialect in [
            ReasoningDialect::OpenAiEffort,
            ReasoningDialect::ChatTemplateReasoningEffort,
        ] {
            assert!(
                !dialect.honors_with_vocabulary(ReasoningPolicy::Enabled, Some(&vocab)),
                "enabled must be rejected on {dialect:?} even if present in the vocabulary"
            );
        }
    }

    #[test]
    fn openai_effort_honors_nothing_without_a_vocabulary() {
        let oa = ReasoningDialect::OpenAiEffort;
        for policy in all_policies() {
            assert!(
                !oa.honors(policy),
                "openai_effort `none` is vocabulary-gated too; got {policy:?}"
            );
        }
    }

    #[test]
    fn honors_with_vocabulary_is_exact() {
        let deepseek: Vec<String> = vec!["none".into(), "low".into(), "high".into(), "max".into()];
        let dialect = ReasoningDialect::OpenAiEffort;
        assert!(dialect.honors_with_vocabulary(ReasoningPolicy::None, Some(&deepseek)));
        assert!(dialect.honors_with_vocabulary(ReasoningPolicy::Low, Some(&deepseek)));
        assert!(dialect.honors_with_vocabulary(ReasoningPolicy::Max, Some(&deepseek)));
        // DeepSeek documents no `medium`: rejected, never approximated.
        assert!(!dialect.honors_with_vocabulary(ReasoningPolicy::Medium, Some(&deepseek)));
        // `xhigh` honored ONLY when explicitly declared.
        assert!(!dialect.honors_with_vocabulary(ReasoningPolicy::XHigh, Some(&deepseek)));
    }

    #[test]
    fn effort_bearing_classification_matches_the_validation_branch() {
        assert!(ReasoningDialect::OpenAiEffort.is_effort_bearing());
        assert!(ReasoningDialect::ChatTemplateReasoningEffort.is_effort_bearing());
        assert!(!ReasoningDialect::OpenRouterEnabled.is_effort_bearing());
        assert!(!ReasoningDialect::LlamaCppEnableThinking.is_effort_bearing());
    }
}
