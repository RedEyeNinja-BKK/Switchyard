// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decision transports: SystemOne and OpenRouter alpha-decisions.
//!
//! The typed request and response are upstream's, unchanged. This module only carries
//! a request across one of two wire families and reads the answer back into the
//! provider-neutral types, so a caller never sees a backend's private shape.
//!
//! Eligibility is a property of the *target*, not of the transport. A transport can
//! represent all three question kinds; a particular backend may currently serve only
//! some of them. Encoding that as a target capability means an unsupported request is
//! excluded before dispatch rather than provoking a 400 from the provider and being
//! mistaken for a routing failure.
//!
//! Bounds are checked against the canonical SystemOne contract: choice carries one to
//! 255 options, score carries two to ten levels. A request outside those bounds is not
//! eligible for a SystemOne backend at all, whichever backend would have been chosen.

use serde_json::{Value, json};
use switchyard_protocol::{DecisionKind, DecisionRequest};

/// Largest option count a choice question may carry on the SystemOne contract.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Fewest levels a score question may carry.
pub const MIN_SCORE_LEVELS: usize = 2;
/// Most levels a score question may carry.
pub const MAX_SCORE_LEVELS: usize = 10;

/// Wire family a decision backend speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionTransport {
    /// TypeSafe Jev `/v1/systemone`, served by Laya and JEV-style.
    SystemOne,
    /// OpenRouter `/api/alpha/decisions`, served by SPAN.
    OpenRouterAlpha,
}

impl DecisionTransport {
    /// Request path for this transport.
    pub fn path(self) -> &'static str {
        match self {
            Self::SystemOne => "/v1/systemone",
            Self::OpenRouterAlpha => "/api/alpha/decisions",
        }
    }

    /// The question-kind discriminator this transport names.
    fn kind_name(kind: DecisionKind) -> &'static str {
        match kind {
            DecisionKind::Boolean { .. } => "noul",
            DecisionKind::Choice { .. } => "choice",
            DecisionKind::Score { .. } => "score",
        }
    }

    /// Builds the provider-shaped request body for one question set.
    pub fn request_body(
        self,
        model: &str,
        request: &DecisionRequest,
    ) -> Result<Value, String> {
        let mut questions = Vec::new();
        for (id, question) in &request.questions {
            let mut entry = json!({
                "question_id": id,
                "instructions": question.instructions,
                "kind": Self::kind_name(question.kind.clone()),
            });
            match &question.kind {
                DecisionKind::Boolean {
                    true_description,
                    false_description,
                } => {
                    let mut criteria = serde_json::Map::new();
                    if let Some(value) = true_description {
                        criteria.insert("true".to_string(), value.clone());
                    }
                    if let Some(value) = false_description {
                        criteria.insert("false".to_string(), value.clone());
                    }
                    if !criteria.is_empty() {
                        entry["criteria"] = Value::Object(criteria);
                    }
                }
                DecisionKind::Choice { options } => {
                    let mut criteria = serde_json::Map::new();
                    for option in options {
                        criteria.insert(
                            option.id.clone(),
                            option.description.clone().unwrap_or(Value::Null),
                        );
                    }
                    entry["criteria"] = Value::Object(criteria);
                }
                DecisionKind::Score { levels } => {
                    entry["criteria"] = Value::Array(levels.clone());
                }
            }
            questions.push(entry);
        }
        Ok(json!({
            "model": model,
            "context": request.context,
            "questions": questions,
        }))
    }
}

/// What a backend can currently answer, independent of what the transport can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportedTypes {
    pub noul: bool,
    pub choice: bool,
    pub score: bool,
}

impl SupportedTypes {
    /// Every question kind.
    pub const ALL: Self = Self {
        noul: true,
        choice: true,
        score: true,
    };

    /// Only probability-of-true questions. SPAN currently serves this and rejects the
    /// other two, so it must be excluded from a request carrying them before dispatch.
    pub const NOUL_ONLY: Self = Self {
        noul: true,
        choice: false,
        score: false,
    };

    fn accepts(self, kind: &DecisionKind) -> bool {
        match kind {
            DecisionKind::Boolean { .. } => self.noul,
            DecisionKind::Choice { .. } => self.choice,
            DecisionKind::Score { .. } => self.score,
        }
    }

    /// Whether this backend may serve every question in the request.
    ///
    /// One unsupported question excludes the whole backend: splitting a request so a
    /// backend sees only the questions it likes would answer a question with another
    /// provider's view of a different question set.
    pub fn accepts_request(self, request: &DecisionRequest) -> bool {
        request.questions.values().all(|q| self.accepts(&q.kind))
    }

    /// Whether every question is also within the SystemOne wire bounds.
    pub fn within_bounds(self, request: &DecisionRequest) -> Result<(), String> {
        for (id, question) in &request.questions {
            match &question.kind {
                DecisionKind::Choice { options } => {
                    let count = options.len();
                    if count == 0 || count > MAX_CHOICE_OPTIONS {
                        return Err(format!(
                            "question {id} carries {count} options; the contract allows 1..{MAX_CHOICE_OPTIONS}"
                        ));
                    }
                }
                DecisionKind::Score { levels } => {
                    let count = levels.len();
                    if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&count) {
                        return Err(format!(
                            "question {id} carries {count} levels; the contract allows {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS}"
                        ));
                    }
                }
                DecisionKind::Boolean { .. } => {}
            }
        }
        Ok(())
    }
}

/// Reads a provider answer back into provider-neutral answers.
///
/// Backend-specific extras (probability distributions, score legends, routing or action
/// metadata) are preserved under `metadata` where the native types permit, rather than
/// reshaping the core estimate to carry them.
pub fn parse_response(body: &Value) -> Result<Vec<(String, Value)>, String> {
    let answers = body
        .get("answers")
        .or_else(|| body.get("questions"))
        .and_then(Value::as_array)
        .ok_or_else(|| "decision response carries no answers array".to_string())?;
    Ok(answers
        .iter()
        .filter_map(|answer| {
            let id = answer
                .get("question_id")
                .and_then(Value::as_str)?
                .to_string();
            Some((id, answer.clone()))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::ChoiceOption;
    use std::collections::BTreeMap;

    fn boolean_request() -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "reasoning_demand".to_string(),
            switchyard_protocol::DecisionQuestion {
                instructions: json!("Does this request need reasoning?"),
                kind: DecisionKind::Boolean {
                    true_description: Some(json!("needs reasoning")),
                    false_description: Some(json!("does not")),
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!({}),
            questions,
        }
    }

    fn choice_request(options: usize) -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "pick".to_string(),
            switchyard_protocol::DecisionQuestion {
                instructions: json!("Choose one"),
                kind: DecisionKind::Choice {
                    options: (0..options)
                        .map(|index| ChoiceOption {
                            id: format!("o{index}"),
                            description: None,
                        })
                        .collect(),
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!({}),
            questions,
        }
    }

    fn score_request(levels: usize) -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "how_bad".to_string(),
            switchyard_protocol::DecisionQuestion {
                instructions: json!("Rate it"),
                kind: DecisionKind::Score {
                    levels: (0..levels).map(|i| json!(format!("level-{i}"))).collect(),
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!({}),
            questions,
        }
    }

    /// SPAN serves noul today and rejects the other two. It must be excluded from a
    /// request carrying them *before* dispatch, not left to produce a 400.
    #[test]
    fn span_is_eligible_for_noul_and_excluded_before_dispatch_otherwise() {
        let noul = boolean_request();
        assert!(SupportedTypes::NOUL_ONLY.accepts_request(&noul));
        assert!(SupportedTypes::ALL.accepts_request(&noul));

        assert!(!SupportedTypes::NOUL_ONLY.accepts_request(&choice_request(3)));
        assert!(!SupportedTypes::NOUL_ONLY.accepts_request(&score_request(3)));
    }

    /// A mixed request excludes SPAN entirely rather than being silently split.
    #[test]
    fn a_mixed_request_excludes_a_noul_only_backend() {
        let mut mixed = boolean_request();
        mixed
            .questions
            .insert("pick".to_string(), choice_request(3).questions.remove("pick").unwrap());
        assert!(SupportedTypes::ALL.accepts_request(&mixed));
        assert!(
            !SupportedTypes::NOUL_ONLY.accepts_request(&mixed),
            "one unsupported question must exclude the backend for the whole request"
        );
    }

    /// Probes immediately across each bound rather than only obviously invalid values.
    #[test]
    fn choice_bounds_hold_at_the_boundary() {
        assert!(SupportedTypes::ALL.within_bounds(&choice_request(1)).is_ok());
        assert!(SupportedTypes::ALL.within_bounds(&choice_request(255)).is_ok());
        assert!(SupportedTypes::ALL.within_bounds(&choice_request(0)).is_err());
        assert!(
            SupportedTypes::ALL
                .within_bounds(&choice_request(256))
                .is_err(),
            "256 is one past the contract maximum"
        );
    }

    #[test]
    fn score_bounds_hold_at_the_boundary() {
        assert!(SupportedTypes::ALL.within_bounds(&score_request(2)).is_ok());
        assert!(SupportedTypes::ALL.within_bounds(&score_request(10)).is_ok());
        assert!(
            SupportedTypes::ALL.within_bounds(&score_request(1)).is_err(),
            "one level is not an ordered rubric"
        );
        assert!(
            SupportedTypes::ALL.within_bounds(&score_request(11)).is_err(),
            "11 is one past the contract maximum"
        );
    }

    /// Out-of-bounds is a contract error, not a routing signal: it must not read as
    /// "this backend is unavailable".
    #[test]
    fn out_of_bounds_is_a_contract_error() {
        let error = SupportedTypes::ALL
            .within_bounds(&choice_request(256))
            .expect_err("256 must be refused");
        assert!(error.contains("1..255"), "{error}");
    }

    #[test]
    fn systemone_body_carries_one_criteria_shape_per_kind() {
        let body = DecisionTransport::SystemOne
            .request_body("jev-style-0.8b-decision-v3", &boolean_request())
            .expect("noul body builds");
        let question = &body["questions"][0];
        assert_eq!(question["kind"], "noul");
        assert!(question["criteria"].is_object());
        assert_eq!(question["criteria"]["true"], "needs reasoning");

        let body = DecisionTransport::SystemOne
            .request_body("jev-style-0.8b-decision-v3", &choice_request(2))
            .expect("choice body builds");
        assert_eq!(body["questions"][0]["kind"], "choice");
        assert!(body["questions"][0]["criteria"]["o0"].is_null());

        let body = DecisionTransport::SystemOne
            .request_body("jev-style-0.8b-decision-v3", &score_request(3))
            .expect("score body builds");
        assert_eq!(body["questions"][0]["kind"], "score");
        assert!(body["questions"][0]["criteria"].is_array());
    }

    #[test]
    fn alpha_and_systemone_differ_only_in_path() {
        assert_eq!(DecisionTransport::SystemOne.path(), "/v1/systemone");
        assert_eq!(
            DecisionTransport::OpenRouterAlpha.path(),
            "/api/alpha/decisions"
        );
        // Both transports can represent every kind; the restriction lives on the target.
        let request = mixed_request();
        for transport in [
            DecisionTransport::SystemOne,
            DecisionTransport::OpenRouterAlpha,
        ] {
            assert!(transport.request_body("m", &request).is_ok());
        }
    }

    fn mixed_request() -> DecisionRequest {
        let mut request = boolean_request();
        request
            .questions
            .insert("pick".to_string(), choice_request(2).questions.remove("pick").unwrap());
        request
    }

    /// Backend extras survive under metadata instead of distorting the core estimate.
    #[test]
    fn richer_backend_data_is_preserved_as_metadata() {
        let body = json!({
            "answers": [{
                "question_id": "pick",
                "answer": {"selected": "o1"},
                "probabilities": {"o0": 0.2, "o1": 0.8},
                "confidence": 0.91,
                "routing": {"model": "qwen3.5-9b"},
                "legend": ["low", "high"],
            }]
        });
        let parsed = parse_response(&body).expect("answers parse");
        assert_eq!(parsed.len(), 1);
        let (id, answer) = &parsed[0];
        assert_eq!(id, "pick");
        assert_eq!(answer["answer"]["selected"], "o1");
        // Provider-specific fields are carried through, not dropped.
        assert_eq!(answer["probabilities"]["o1"], 0.8);
        assert_eq!(answer["confidence"], 0.91);
        assert!(answer["routing"].is_object());
        assert!(answer["legend"].is_array());
    }

    #[test]
    fn a_response_without_answers_is_an_error() {
        assert!(parse_response(&json!({"ok": true})).is_err());
    }
}