use serde::{Deserialize, Serialize};

/// Whether a target's model is asked to reason, and how that reaches the provider.
///
/// `reasoning_effort` covers the OpenAI-shaped dialects, where the knob is a value
/// (`none`, `low`, ... `max`). Providers that accept reasoning but expose it as a
/// boolean switch have no effort value, so they carry `Enabled`/`Disabled` instead.
/// The distinction matters because a strict non-reasoning target must be able to say so
/// even when the caller asked for reasoning: this is applied after the caller's fields
/// and after any `extra_body` default, so the target's choice wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningPolicy {
    /// Leave reasoning to the provider's own default.
    Unspecified,
    /// Ask the provider not to reason (`reasoning.enabled = false`, or
    /// `chat_template_kwargs.enable_thinking = false` where that is the dialect).
    Disabled,
    /// Ask the provider to reason.
    Enabled,
}

impl ReasoningPolicy {
    /// Whether this policy names a concrete choice. `Unspecified` defers to the
    /// provider and therefore must not clear fields the caller or target already set.
    pub fn is_specified(self) -> bool {
        !matches!(self, Self::Unspecified)
    }

    /// The provider-shaped body fragment for a dialect, or `None` when the dialect has
    /// no representation and the policy must fail closed rather than send a request
    /// that silently reasons.
    pub fn wire_body(self, dialect: ReasoningDialect) -> Option<serde_json::Value> {
        use serde_json::json;
        match dialect {
            // No dialect was declared, so there is nothing to encode. Reached only when a
            // caller skips `validate_for`; configuration loading never admits this pair.
            ReasoningDialect::Unspecified => None,
            ReasoningDialect::OpenAiEffort => match self {
                Self::Unspecified | Self::Enabled => None,
                Self::Disabled => Some(json!({"reasoning": {"effort": "none"}})),
            },
            ReasoningDialect::OpenRouterEnabled => match self {
                Self::Unspecified => None,
                Self::Disabled => Some(json!({"reasoning": {"enabled": false}})),
                Self::Enabled => Some(json!({"reasoning": {"enabled": true}})),
            },
            ReasoningDialect::LlamaCppEnableThinking => match self {
                Self::Unspecified => None,
                Self::Disabled => {
                    Some(json!({"chat_template_kwargs": {"enable_thinking": false}}))
                }
                Self::Enabled => Some(json!({"chat_template_kwargs": {"enable_thinking": true}})),
            },
        }
    }

    /// Checks that this policy can actually be honoured by the declared dialect.
    ///
    /// An authoritative policy with no dialect would otherwise acquire a default dialect
    /// and emit a provider control nobody declared, which is the failure mode this
    /// rejects: a strict target silently sending the wrong knob.
    pub fn validate_for(
        self,
        dialect: ReasoningDialect,
    ) -> Result<(), ReasoningPolicyError> {
        if !self.is_specified() {
            // An unspecified policy changes nothing, so any dialect is acceptable and an
            // absent one is the normal case.
            return Ok(());
        }
        if matches!(dialect, ReasoningDialect::Unspecified) {
            return Err(ReasoningPolicyError::MissingDialect);
        }
        if matches!(self, Self::Enabled) && matches!(dialect, ReasoningDialect::OpenAiEffort) {
            // The OpenAI dialect only grades effort, and `none` is its sole off value, so
            // there is no fragment meaning "think". A thinking OpenAI target sets
            // `reasoning_effort` instead, which is authoritative on its own.
            return Err(ReasoningPolicyError::CannotExpressEnabled);
        }
        Ok(())
    }
}

/// Why a declared reasoning policy cannot be honoured by the declared dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningPolicyError {
    /// The policy is authoritative but no dialect was declared.
    MissingDialect,
    /// The policy asks for reasoning on a dialect that cannot express it.
    CannotExpressEnabled,
}

impl std::fmt::Display for ReasoningPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDialect => write!(
                formatter,
                "a reasoning_policy needs an explicit reasoning_dialect, because a policy \
                 cannot be applied without knowing how the provider carries it"
            ),
            Self::CannotExpressEnabled => write!(
                formatter,
                "reasoning_policy = \"enabled\" cannot be expressed by reasoning_dialect \
                 \"openai_effort\", which only grades effort and has no on/off switch; set \
                 reasoning_effort on this target instead"
            ),
        }
    }
}

impl std::error::Error for ReasoningPolicyError {}

/// How a provider expects reasoning control on the wire.
///
/// This is a property of the provider, not of the request, so it belongs on the target
/// rather than on the transport: two targets behind one transport routinely use
/// different dialects.
///
/// The default is [`Unspecified`](Self::Unspecified) rather than a real dialect on
/// purpose: a target that never declares one must not silently acquire one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningDialect {
    /// No dialect was declared. Carries no policy, and fails closed when a policy needs one.
    #[default]
    #[serde(rename = "unspecified")]
    Unspecified,
    /// `reasoning.effort` with graded values, including `none`.
    #[serde(rename = "openai_effort")]
    OpenAiEffort,
    /// `reasoning.enabled`, a boolean switch with no graded values.
    #[serde(rename = "openrouter_enabled")]
    OpenRouterEnabled,
    /// `chat_template_kwargs.enable_thinking`, as llama.cpp servers accept it.
    #[serde(rename = "llama_cpp_enable_thinking")]
    LlamaCppEnableThinking,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unspecified_defers_and_never_clears() {
        assert!(!ReasoningPolicy::Unspecified.is_specified());
        assert!(ReasoningPolicy::Disabled.is_specified());
        assert!(ReasoningPolicy::Enabled.is_specified());
        assert_eq!(
            ReasoningPolicy::Unspecified.wire_body(ReasoningDialect::OpenRouterEnabled),
            None
        );
    }

    #[test]
    fn openai_effort_cannot_express_enabled() {
        assert_eq!(
            ReasoningPolicy::Enabled.wire_body(ReasoningDialect::OpenAiEffort),
            None
        );
        assert_eq!(
            ReasoningPolicy::Disabled.wire_body(ReasoningDialect::OpenAiEffort),
            Some(json!({"reasoning": {"effort": "none"}}))
        );
    }

    #[test]
    fn a_declared_dialect_is_required_before_a_policy_can_be_applied() {
        // The unsafe case: a strict target that omitted its dialect. It must not silently
        // acquire a default dialect and emit a control nobody declared.
        assert_eq!(
            ReasoningPolicy::Disabled
                .validate_for(ReasoningDialect::Unspecified)
                .unwrap_err(),
            ReasoningPolicyError::MissingDialect
        );
        assert_eq!(
            ReasoningPolicy::Enabled
                .validate_for(ReasoningDialect::Unspecified)
                .unwrap_err(),
            ReasoningPolicyError::MissingDialect
        );
        // An unspecified policy changes nothing, so no dialect is needed.
        assert_eq!(
            ReasoningPolicy::Unspecified.validate_for(ReasoningDialect::Unspecified),
            Ok(())
        );
    }

    #[test]
    fn every_dialect_can_express_a_disabled_policy() {
        for dialect in [
            ReasoningDialect::OpenAiEffort,
            ReasoningDialect::OpenRouterEnabled,
            ReasoningDialect::LlamaCppEnableThinking,
        ] {
            assert_eq!(
                ReasoningPolicy::Disabled.validate_for(dialect),
                Ok(()),
                "{dialect:?} must be able to turn reasoning off"
            );
            assert!(
                ReasoningPolicy::Disabled
                    .wire_body(dialect)
                    .is_some_and(|body| body.to_string().contains("false")
                        || body.to_string().contains("none")),
                "{dialect:?} must emit a concrete disable control"
            );
        }
    }

    #[test]
    fn an_undeclared_dialect_emits_nothing() {
        assert_eq!(
            ReasoningPolicy::Disabled.wire_body(ReasoningDialect::Unspecified),
            None
        );
    }

    #[test]
    fn openai_thinking_targets_use_the_effort_mechanism_not_the_policy() {
        // `enabled` on the OpenAI dialect is a configuration mistake rather than a silent
        // no-op, so it is rejected and the error points at the authoritative alternative.
        assert_eq!(
            ReasoningPolicy::Enabled
                .validate_for(ReasoningDialect::OpenAiEffort)
                .unwrap_err(),
            ReasoningPolicyError::CannotExpressEnabled
        );
        assert!(
            ReasoningPolicyError::CannotExpressEnabled
                .to_string()
                .contains("reasoning_effort")
        );
        // The boolean dialects do express it, so the rejection is dialect-specific.
        assert_eq!(
            ReasoningPolicy::Enabled.validate_for(ReasoningDialect::OpenRouterEnabled),
            Ok(())
        );
    }

    #[test]
    fn boolean_dialects_round_trip_both_directions() {
        for dialect in [
            ReasoningDialect::OpenRouterEnabled,
            ReasoningDialect::LlamaCppEnableThinking,
        ] {
            assert!(ReasoningPolicy::Disabled
                .wire_body(dialect)
                .is_some_and(|body| body.to_string().contains("false")));
            assert!(ReasoningPolicy::Enabled
                .wire_body(dialect)
                .is_some_and(|body| body.to_string().contains("true")));
        }
    }

    #[test]
    fn policy_round_trips_through_the_config_schema() {
        for (json_text, expected) in [
            ("\"disabled\"", ReasoningPolicy::Disabled),
            ("\"enabled\"", ReasoningPolicy::Enabled),
            ("\"unspecified\"", ReasoningPolicy::Unspecified),
        ] {
            #[derive(Deserialize)]
            struct Holder {
                reasoning_policy: ReasoningPolicy,
            }
            let holder: Holder =
                serde_json::from_str(&format!("{{\"reasoning_policy\": {json_text}}}"))
                    .expect("policy parses");
            assert_eq!(holder.reasoning_policy, expected);
        }
    }

    #[test]
    fn an_unknown_policy_name_is_rejected_rather_than_defaulted() {
        // A typo must fail closed at load time, not silently leave reasoning unspecified.
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Holder {
            reasoning_policy: ReasoningPolicy,
        }
        assert!(
            serde_json::from_str::<Holder>(r#"{"reasoning_policy": "disbled"}"#).is_err(),
            "unknown policy name must not parse"
        );
    }
}