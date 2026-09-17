// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Route-authoritative reasoning policy: an abstract route-level intent.
//!
//! `Responses is the common transport, not the reasoning dialect.` A route's
//! [`ReasoningPolicy`] states **what** execution mode is forced on every
//! candidate the route can reach; each target's backend translates that intent
//! into the wire control its endpoint actually understands (OpenAI-family
//! `reasoning.effort`, OpenRouter `reasoning.enabled`, llama.cpp
//! `chat_template_kwargs.enable_thinking`). Routes never carry provider syntax.

use std::fmt;

/// Reasoning execution mode a route forces onto every reachable target.
///
/// Declared per route as `reasoning_policy = "none" | "low" | "medium" |
/// "high" | "max"`. Absent means the route carries no policy and routing
/// behaves exactly as before (callers and target pins decide).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningPolicy {
    /// Forbid reasoning: no reasoning may be produced or billed.
    None,
    /// Permit reasoning with a bounded effort budget.
    Low,
    /// Permit reasoning with a bounded effort budget.
    Medium,
    /// Permit reasoning with a bounded effort budget.
    High,
    /// Permit reasoning at the provider's maximum effort.
    Max,
}

impl ReasoningPolicy {
    /// Canonical wire-neutral name of the policy (also the config spelling).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    /// Whether this policy demands that reasoning is enabled at all.
    ///
    /// Dialects that cannot express effort gradations (only enabled/disabled)
    /// accept exactly [`Self::None`] and the [`Self::is_thinking`] complement.
    pub const fn is_thinking(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Parses the config spelling; unknown values are a configuration error.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "none" => Some(Self::None),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
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
/// (`reasoning_dialect = "openai_effort" | "openrouter_enabled" |
/// "llama_cpp_enable_thinking"`); the send seam merges
/// [`ReasoningDialect::wire_body`] into the outbound body, authoritative over
/// caller values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReasoningDialect {
    /// `reasoning.effort = "<policy>"` with effort `none` disabling reasoning
    /// (DeepSeek, OpenAI, and OpenAI-compatible Responses endpoints). Preserves
    /// effort gradations verbatim.
    #[serde(rename = "openai_effort")]
    OpenAiEffort,
    /// `reasoning = { enabled = <bool> }` (OpenRouter). Boolean switch only.
    #[serde(rename = "openrouter_enabled")]
    OpenRouterEnabled,
    /// `chat_template_kwargs.enable_thinking = <bool>` (llama.cpp-derived Qwen
    /// template endpoints). Boolean switch only. This is the HTPC contract:
    /// thinking = `enable_thinking = true`, none = `enable_thinking = false`
    /// (both shapes observed live in the Gate 0 decision snapshot).
    #[serde(rename = "llama_cpp_enable_thinking")]
    LlamaCppEnableThinking,
    /// Effort-bearing chat-template control (llama.cpp Qwen3 template
    /// endpoints whose thinking side takes `chat_template_kwargs.reasoning_effort`):
    /// none = `enable_thinking = false`, any thinking policy =
    /// `reasoning_effort = <policy>` (the ComfyNinja Q3/Q4 contract observed
    /// live in the Gate 0 decision snapshot: NT twins pin
    /// `enable_thinking = false` while the thinking twin pins
    /// `reasoning_effort = "medium"`). Distinct from
    /// [`ReasoningDialect::LlamaCppEnableThinking`] precisely because that
    /// dialect cannot express the thinking-side effort.
    #[serde(rename = "chat_template_reasoning_effort")]
    ChatTemplateReasoningEffort,
}

impl ReasoningDialect {
    /// The JSON object this dialect merges into the outbound body to express
    /// `policy` (e.g. `{"reasoning": {"effort": "none"}}`).
    ///
    /// Every current dialect expresses both execution modes — `none` disables
    /// reasoning exactly; any thinking policy enables it. On the boolean
    /// dialects an effort gradation collapses to the native "thinking on"
    /// semantic (the adapter owns translation; the endpoint cannot express
    /// amounts). Returns `None` only if a future dialect gains an
    /// inexpressible policy — the send seam treats that as fail-closed, never
    /// as a silent downgrade.
    pub fn wire_body(self, policy: ReasoningPolicy) -> Option<serde_json::Value> {
        match self {
            Self::OpenAiEffort => {
                Some(serde_json::json!({ "reasoning": { "effort": policy.as_str() } }))
            }
            Self::OpenRouterEnabled => {
                Some(serde_json::json!({ "reasoning": { "enabled": policy.is_thinking() } }))
            }
            Self::LlamaCppEnableThinking => Some(serde_json::json!(
                { "chat_template_kwargs": { "enable_thinking": policy.is_thinking() } }
            )),
            Self::ChatTemplateReasoningEffort => {
                if policy.is_thinking() {
                    Some(serde_json::json!(
                        { "chat_template_kwargs": { "reasoning_effort": policy.as_str() } }
                    ))
                } else {
                    // The observed NT contract is the boolean switch, not
                    // `reasoning_effort = "none"`.
                    Some(serde_json::json!(
                        { "chat_template_kwargs": { "enable_thinking": false } }
                    ))
                }
            }
        }
    }

    /// Whether `policy` is expressible in this dialect.
    pub fn honors(self, policy: ReasoningPolicy) -> bool {
        self.wire_body(policy).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_round_trips_every_value() {
        for policy in [
            ReasoningPolicy::None,
            ReasoningPolicy::Low,
            ReasoningPolicy::Medium,
            ReasoningPolicy::High,
            ReasoningPolicy::Max,
        ] {
            assert_eq!(ReasoningPolicy::parse(policy.as_str()), Some(policy));
        }
    }

    #[test]
    fn parse_rejects_unknown_and_empty() {
        assert_eq!(ReasoningPolicy::parse(""), None);
        assert_eq!(ReasoningPolicy::parse("xhigh"), None);
        assert_eq!(ReasoningPolicy::parse("true"), None);
        assert_eq!(ReasoningPolicy::parse("None"), None, "config is lowercase");
    }

    #[test]
    fn is_thinking_partitions_none_from_the_rest() {
        assert!(!ReasoningPolicy::None.is_thinking());
        assert!(ReasoningPolicy::Low.is_thinking());
        assert!(ReasoningPolicy::Max.is_thinking());
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
            ReasoningDialect::OpenRouterEnabled
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "reasoning": { "enabled": false } })
        );
        assert_eq!(
            ReasoningDialect::LlamaCppEnableThinking
                .wire_body(ReasoningPolicy::None)
                .unwrap(),
            json!({ "chat_template_kwargs": { "enable_thinking": false } })
        );
        assert_eq!(
            ReasoningDialect::LlamaCppEnableThinking
                .wire_body(ReasoningPolicy::High)
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

    #[test]
    fn honors_tracks_expressibility() {
        for dialect in [
            ReasoningDialect::OpenAiEffort,
            ReasoningDialect::OpenRouterEnabled,
            ReasoningDialect::LlamaCppEnableThinking,
            ReasoningDialect::ChatTemplateReasoningEffort,
        ] {
            for policy in [
                ReasoningPolicy::None,
                ReasoningPolicy::Low,
                ReasoningPolicy::Medium,
                ReasoningPolicy::High,
                ReasoningPolicy::Max,
            ] {
                assert!(dialect.honors(policy), "{dialect:?} must honor {policy:?}");
            }
        }
    }
}
