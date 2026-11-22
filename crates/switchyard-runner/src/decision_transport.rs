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

/// A question kind, named in configuration so a lane can be scoped to one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionKind {
    /// Probability of true.
    Noul,
    /// One of a declared option set.
    Choice,
    /// A position on an ordered rubric.
    Score,
}

impl QuestionKind {
    fn from_kind(kind: &DecisionKind) -> Self {
        match kind {
            DecisionKind::Boolean { .. } => Self::Noul,
            DecisionKind::Choice { .. } => Self::Choice,
            DecisionKind::Score { .. } => Self::Score,
        }
    }
}

/// The question kinds a request carries, in a stable order.
pub fn question_kinds(request: &DecisionRequest) -> Vec<QuestionKind> {
    let mut kinds: Vec<QuestionKind> = request
        .questions
        .values()
        .map(|question| QuestionKind::from_kind(&question.kind))
        .collect();
    kinds.sort_by_key(|kind| match kind {
        QuestionKind::Noul => 0,
        QuestionKind::Choice => 1,
        QuestionKind::Score => 2,
    });
    kinds.dedup();
    kinds
}

/// Whether a lane scoped to `types` serves a request carrying `kinds`.
///
/// An unscoped lane serves every kind. A scoped lane serves a request only when it
/// covers every kind present, so a mixed request never lands on a lane tuned for one of
/// its halves.
pub fn lane_serves(lane_types: &[QuestionKind], request: &DecisionRequest) -> bool {
    if lane_types.is_empty() {
        return true;
    }
    question_kinds(request)
        .iter()
        .all(|kind| lane_types.contains(kind))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionTransport {
    /// TypeSafe Jev `/v1/systemone`, served by Laya and JEV-style.
    SystemOne,
    /// OpenRouter `/api/alpha/decisions`, served by SPAN.
    OpenRouterAlpha,
    /// Google Gemini Decisions API `/v1beta/decisions`, served by `gemini-decision-1`.
    ///
    /// The question map is the same semantic encoding, but three details are not: Google
    /// names the shared input `input` rather than `state`, spells the yes/no criteria keys
    /// out (`true_description` / `false_description`), and wants a string description for
    /// every choice option where the other transports pass `null`. Authentication is
    /// `x-goog-api-key`, not a bearer token.
    GoogleDecisions,
}

impl DecisionTransport {
    /// Request path for this transport.
    pub fn path(self) -> &'static str {
        match self {
            Self::SystemOne => "/v1/systemone",
            Self::OpenRouterAlpha => "/api/alpha/decisions",
            Self::GoogleDecisions => "/v1beta/decisions",
        }
    }

    /// Whether this transport authenticates with an `x-goog-api-key` header.
    pub fn uses_google_api_key(self) -> bool {
        matches!(self, Self::GoogleDecisions)
    }

    /// Whether this transport speaks Google's field spellings.
    ///
    /// A separate question from authentication: both are true of today's one Google
    /// transport, and keeping them apart means a transport that authenticates one way
    /// and encodes another does not silently inherit the wrong answer.
    fn google_field_names(self) -> bool {
        matches!(self, Self::GoogleDecisions)
    }

    /// The field that carries the shared input.
    fn context_field(self) -> &'static str {
        match self {
            Self::GoogleDecisions => "input",
            _ => "state",
        }
    }

    /// The question-kind discriminator this transport names.
    fn kind_name(kind: &DecisionKind) -> &'static str {
        match kind {
            DecisionKind::Boolean { .. } => "noul",
            DecisionKind::Choice { .. } => "choice",
            DecisionKind::Score { .. } => "score",
        }
    }

    /// Builds the canonical request body shared by the transport families.
    ///
    /// The external contract carries `questions` as an object keyed by question id and
    /// names the kind `type`. SystemOne and OpenRouter alpha-decisions also name the
    /// context `state`, so the difference between them is endpoint, authentication and
    /// response handling rather than a second semantic encoding. Google is the one family
    /// whose field NAMES differ, and those differences are stated here rather than in a
    /// second encoder, so a question can never mean two things on two backends.
    pub fn request_body(self, model: &str, request: &DecisionRequest) -> Result<Value, String> {
        let google = self.google_field_names();
        let mut questions = serde_json::Map::new();
        for (id, question) in &request.questions {
            let mut entry = serde_json::Map::new();
            entry.insert("type".to_string(), json!(Self::kind_name(&question.kind)));
            entry.insert("instructions".to_string(), question.instructions.clone());
            match &question.kind {
                DecisionKind::Boolean {
                    true_description,
                    false_description,
                } => {
                    // Optional: a noul question may omit criteria entirely.
                    let (true_key, false_key) = if google {
                        ("true_description", "false_description")
                    } else {
                        ("true", "false")
                    };
                    let mut criteria = serde_json::Map::new();
                    if let Some(value) = true_description {
                        criteria.insert(true_key.to_string(), value.clone());
                    }
                    if let Some(value) = false_description {
                        criteria.insert(false_key.to_string(), value.clone());
                    }
                    if !criteria.is_empty() {
                        entry.insert("criteria".to_string(), Value::Object(criteria));
                    }
                }
                DecisionKind::Choice { options } => {
                    // A required object mapping option id to a description. Google refuses
                    // a null description, so an option that names none is described as the
                    // empty string, which is what its own documentation prescribes for an
                    // option id that speaks for itself.
                    let mut criteria = serde_json::Map::new();
                    for option in options {
                        criteria.insert(
                            option.id.clone(),
                            option
                                .description
                                .clone()
                                .unwrap_or_else(|| if google { json!("") } else { Value::Null }),
                        );
                    }
                    entry.insert("criteria".to_string(), Value::Object(criteria));
                }
                DecisionKind::Score { levels } => {
                    // A required ordered array of levels.
                    entry.insert("criteria".to_string(), Value::Array(levels.clone()));
                }
            }
            questions.insert(id.clone(), Value::Object(entry));
        }
        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), json!(model));
        body.insert(self.context_field().to_string(), request.context.clone());
        body.insert("questions".to_string(), Value::Object(questions));
        Ok(Value::Object(body))
    }
}

/// What a backend can currently answer, independent of what the transport can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct SupportedTypes {
    #[serde(default = "default_true")]
    pub noul: bool,
    #[serde(default = "default_true")]
    pub choice: bool,
    #[serde(default = "default_true")]
    pub score: bool,
}

const fn default_true() -> bool {
    true
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
}

/// State shapes a decision backend accepts.
///
/// This is a property of the backend, not of the transport. SPAN is an OpenRouter
/// alpha backend whose live service currently accepts a string state or an approved
/// message form and rejects other structured objects; recording that here keeps it a
/// dated, per-target capability rather than a permanent limitation of the OpenRouter
/// alpha family, which the provider documents more generally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(from = "StateFormsAcceptance")]
pub struct StateForms {
    /// A plain string state.
    pub plain_string: bool,
    /// An arbitrary structured object state.
    pub structured: bool,
}

/// Absent state-form configuration means no restriction is declared.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
pub struct StateFormsAcceptance {
    plain_string: Option<bool>,
    structured: Option<bool>,
}

impl From<StateFormsAcceptance> for StateForms {
    fn from(acceptance: StateFormsAcceptance) -> Self {
        Self {
            plain_string: acceptance.plain_string.unwrap_or(true),
            structured: acceptance.structured.unwrap_or(true),
        }
    }
}

impl Default for StateForms {
    /// No declared restriction means every state shape is accepted.
    fn default() -> Self {
        Self::ALL
    }
}

impl StateForms {
    /// Every shape: the default for a backend that documents no restriction.
    pub const ALL: Self = Self {
        plain_string: true,
        structured: true,
    };

    /// What SPAN's live service was observed to accept on 2026-10-02.
    pub const SPAN_OBSERVED: Self = Self {
        plain_string: true,
        structured: false,
    };

    /// Whether this backend accepts the state carried by the request.
    pub fn accepts(self, context: &Value) -> bool {
        match context {
            Value::String(_) => self.plain_string,
            // An array or a scalar is structured content too; only an object is the
            // form SPAN was seen rejecting, but the distinction is kept simple here.
            _ => self.structured,
        }
    }
}

/// Validates a request against the SystemOne wire contract.
///
/// This is a transport contract check, deliberately not a target capability: the option
/// and level bounds belong to this wire family, so they must not become permanent
/// attributes of every Decision target if another backend later supports the same kinds
/// under different limits. An out-of-bounds request is a contract error and is never an
/// availability or fallback signal.
pub fn validate_systemone(request: &DecisionRequest) -> Result<(), String> {
    for (id, question) in &request.questions {
        match &question.kind {
            DecisionKind::Choice { options } => {
                let count = options.len();
                if count == 0 || count > MAX_CHOICE_OPTIONS {
                    return Err(format!(
                        "question {id} carries {count} options; the SystemOne contract allows 1..{MAX_CHOICE_OPTIONS}"
                    ));
                }
            }
            DecisionKind::Score { levels } => {
                let count = levels.len();
                if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&count) {
                    return Err(format!(
                        "question {id} carries {count} levels; the SystemOne contract allows {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS}"
                    ));
                }
            }
            DecisionKind::Boolean { .. } => {}
        }
    }
    Ok(())
}

/// Reads a provider response into upstream's native decision types.
///
/// The provider answers are an object keyed by question id, each tagged with its kind.
/// Values are carried into `DecisionResponse` unchanged in meaning: a noul becomes a
/// probability of true rather than a threshold, a choice keeps its full probability
/// distribution alongside the selection, and a score keeps its ordered distribution
/// alongside the estimate.
///
/// The ordered rubric needed to read a score is already present in the request the caller
/// made, so a provider's score legend is not duplicated into the response, and a
/// backend's own routing or action metadata is not added here: it is diagnostic, and no
/// portable caller field is manufactured to carry it.
pub fn parse_response(body: &Value) -> Result<switchyard_protocol::DecisionResponse, String> {
    let answers = body
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| "decision response carries no answers object".to_string())?;
    let mut parsed = std::collections::BTreeMap::new();
    for (id, answer) in answers {
        let kind = answer
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("answer {id} carries no type"))?;
        let confidence = answer
            .get("confidence")
            .and_then(Value::as_f64)
            .map(switchyard_protocol::ProviderConfidence);
        let probabilities = || {
            answer
                .get("probabilities")
                .and_then(Value::as_object)
                .map(|map| {
                    map.iter()
                        .filter_map(|(name, value)| {
                            value.as_f64().map(|value| {
                                (name.clone(), switchyard_protocol::Probability(value))
                            })
                        })
                        .collect()
                })
        };
        let value = match kind {
            "noul" => {
                let probability = answer
                    .get("noul")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("noul answer {id} carries no probability"))?;
                switchyard_protocol::DecisionValue::Boolean(
                    switchyard_protocol::BooleanEstimate::ProbabilityTrue(
                        switchyard_protocol::Probability(probability),
                    ),
                )
            }
            "choice" => {
                let selected = answer
                    .get("choice")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("choice answer {id} names no option"))?
                    .to_string();
                switchyard_protocol::DecisionValue::Choice {
                    selected,
                    probabilities: probabilities(),
                }
            }
            "score" => {
                let estimate = answer
                    .get("score")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("score answer {id} carries no estimate"))?;
                let ordered = answer
                    .get("probabilities")
                    .and_then(Value::as_object)
                    .map(|map| {
                        map.iter()
                            .filter_map(|(_, value)| value.as_f64())
                            .map(switchyard_protocol::Probability)
                            .collect()
                    });
                switchyard_protocol::DecisionValue::Score {
                    value: switchyard_protocol::ScoreValue(estimate),
                    probabilities: ordered,
                }
            }
            other => return Err(format!("answer {id} has unsupported type {other}")),
        };
        parsed.insert(
            id.clone(),
            switchyard_protocol::DecisionAnswer {
                value,
                provider_confidence: confidence,
            },
        );
    }
    // Provider-reported identity and usage are carried when supplied; a provider that
    // does not report them leaves them absent rather than invented.
    let id = body.get("id").and_then(Value::as_str).map(str::to_string);
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(switchyard_protocol::ModelId::from);
    let usage = match body.get("usage") {
        Some(usage) => switchyard_protocol::Usage {
            input_tokens: usage.get("input_tokens").and_then(Value::as_u64),
            cache: None,
            output_tokens: usage.get("output_tokens").and_then(Value::as_u64),
            total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
            reasoning_tokens: None,
            cost: None,
            cost_currency: None,
        },
        None => switchyard_protocol::Usage::default(),
    };
    Ok(switchyard_protocol::DecisionResponse {
        id,
        model,
        answers: parsed,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use switchyard_protocol::{ChoiceOption, DecisionQuestion, DecisionValue};

    fn boolean_request() -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "risk".to_string(),
            DecisionQuestion {
                instructions: json!("Does this request need reasoning?"),
                kind: DecisionKind::Boolean {
                    true_description: Some(json!("needs reasoning")),
                    false_description: Some(json!("does not")),
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!({"session": "abc"}),
            questions,
        }
    }

    fn choice_request(options: usize) -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "team".to_string(),
            DecisionQuestion {
                instructions: json!("Pick a team"),
                kind: DecisionKind::Choice {
                    options: (0..options)
                        .map(|index| ChoiceOption {
                            id: if index == 0 {
                                "billing".into()
                            } else {
                                format!("o{index}")
                            },
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
            "severity".to_string(),
            DecisionQuestion {
                instructions: json!("Rate severity"),
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
        assert!(SupportedTypes::NOUL_ONLY.accepts_request(&boolean_request()));
        assert!(SupportedTypes::ALL.accepts_request(&boolean_request()));
        assert!(!SupportedTypes::NOUL_ONLY.accepts_request(&choice_request(3)));
        assert!(!SupportedTypes::NOUL_ONLY.accepts_request(&score_request(3)));
    }

    /// A mixed request excludes SPAN entirely rather than being silently split.
    #[test]
    fn a_mixed_request_excludes_a_noul_only_backend() {
        let mut mixed = boolean_request();
        mixed.questions.extend(choice_request(2).questions);
        assert!(SupportedTypes::ALL.accepts_request(&mixed));
        assert!(!SupportedTypes::NOUL_ONLY.accepts_request(&mixed));
    }

    /// Target capability and wire bounds are separate questions: a target may support
    /// every kind and still be unable to carry a request outside this contract.
    #[test]
    fn capability_and_wire_bounds_are_independent() {
        let out_of_bounds = choice_request(256);
        assert!(
            SupportedTypes::ALL.accepts_request(&out_of_bounds),
            "capability asks which kinds the target supports, not how many options fit"
        );
        assert!(
            validate_systemone(&out_of_bounds).is_err(),
            "the SystemOne contract still refuses 256 options"
        );
    }

    /// Probes immediately across each bound rather than only obviously invalid values.
    #[test]
    fn choice_bounds_hold_at_the_boundary() {
        assert!(validate_systemone(&choice_request(1)).is_ok());
        assert!(validate_systemone(&choice_request(255)).is_ok());
        assert!(validate_systemone(&choice_request(0)).is_err());
        assert!(validate_systemone(&choice_request(256)).is_err());
    }

    #[test]
    fn score_bounds_hold_at_the_boundary() {
        assert!(validate_systemone(&score_request(2)).is_ok());
        assert!(validate_systemone(&score_request(10)).is_ok());
        assert!(validate_systemone(&score_request(1)).is_err());
        assert!(validate_systemone(&score_request(11)).is_err());
    }

    #[test]
    fn out_of_bounds_is_a_contract_error() {
        let error = validate_systemone(&choice_request(256)).expect_err("256 must be refused");
        assert!(error.contains("1..255"), "{error}");
    }

    /// Pins the canonical external request shape: `state`, a `questions` object keyed by
    /// id, and `type` as the discriminator. These field names are the provider contract,
    /// so a change here is a wire break, not a refactor.
    #[test]
    fn the_request_matches_the_canonical_external_contract() {
        let body = DecisionTransport::SystemOne
            .request_body("jev-style-0.8b-decision-v3", &boolean_request())
            .expect("body builds");
        assert!(body.get("state").is_some(), "context is named state");
        assert!(
            body.get("context").is_none(),
            "context is not the field name"
        );
        let questions = body["questions"]
            .as_object()
            .expect("questions is an object");
        assert_eq!(questions.len(), 1);
        let risk = &questions["risk"];
        assert_eq!(risk["type"], "noul");
        assert!(
            risk.get("kind").is_none(),
            "the discriminator is type, not kind"
        );
        assert!(risk.get("question_id").is_none(), "the id is the map key");
        assert!(risk.get("instructions").is_some());
        assert_eq!(risk["criteria"]["true"], "needs reasoning");
        assert_eq!(risk["criteria"]["false"], "does not");

        let choice = DecisionTransport::SystemOne
            .request_body("m", &choice_request(2))
            .expect("choice body builds");
        assert_eq!(choice["questions"]["team"]["type"], "choice");
        assert!(choice["questions"]["team"]["criteria"].is_object());

        let score = DecisionTransport::SystemOne
            .request_body("m", &score_request(3))
            .expect("score body builds");
        assert_eq!(score["questions"]["severity"]["type"], "score");
        assert!(score["questions"]["severity"]["criteria"].is_array());
    }

    /// A noul question may legitimately carry no criteria, and that must stay absent
    /// rather than becoming an empty object.
    #[test]
    fn noul_criteria_are_omitted_when_absent() {
        let mut request = boolean_request();
        let question = request.questions.get_mut("risk").expect("question");
        question.kind = DecisionKind::Boolean {
            true_description: None,
            false_description: None,
        };
        let body = DecisionTransport::SystemOne
            .request_body("m", &request)
            .expect("body builds");
        assert!(body["questions"]["risk"].get("criteria").is_none());
    }

    /// Both transport families emit the same semantic encoding; they differ in endpoint
    /// and authentication, not in how a question is expressed.
    #[test]
    fn both_families_share_one_encoding() {
        let request = boolean_request();
        let systemone = DecisionTransport::SystemOne
            .request_body("m", &request)
            .expect("systemone body");
        let alpha = DecisionTransport::OpenRouterAlpha
            .request_body("m", &request)
            .expect("alpha body");
        assert_eq!(systemone, alpha);
        assert_ne!(
            DecisionTransport::SystemOne.path(),
            DecisionTransport::OpenRouterAlpha.path()
        );
    }

    /// Google is the one family whose field NAMES differ: the shared input is `input`, the
    /// yes/no criteria keys are spelled out, and every choice option needs a string
    /// description rather than `null`. The question map itself stays the same encoding.
    #[test]
    fn google_decisions_speaks_googles_field_names() {
        let body = DecisionTransport::GoogleDecisions
            .request_body("gemini-decision-1", &boolean_request())
            .expect("body builds");
        assert_eq!(body["model"], "gemini-decision-1");
        assert!(body.get("input").is_some(), "Google names the input input");
        assert!(
            body.get("state").is_none(),
            "Google does not accept state as the input field"
        );
        let risk = &body["questions"]["risk"];
        assert_eq!(risk["type"], "noul");
        assert_eq!(risk["criteria"]["true_description"], "needs reasoning");
        assert_eq!(risk["criteria"]["false_description"], "does not");
        assert!(
            risk["criteria"].get("true").is_none(),
            "the short key belongs to the other transports"
        );

        let choice = DecisionTransport::GoogleDecisions
            .request_body("m", &choice_request(2))
            .expect("choice body builds");
        assert_eq!(
            choice["questions"]["team"]["criteria"]["billing"], "",
            "Google wants a string description where the others pass null"
        );

        let score = DecisionTransport::GoogleDecisions
            .request_body("m", &score_request(3))
            .expect("score body builds");
        assert!(
            score["questions"]["severity"]["criteria"].is_array(),
            "a score rubric is an ordered array on every transport"
        );
    }

    /// The one Google transport has its own path and its own authentication, and no other
    /// transport may claim either.
    #[test]
    fn google_decisions_has_its_own_path_and_authentication() {
        assert_eq!(
            DecisionTransport::GoogleDecisions.path(),
            "/v1beta/decisions"
        );
        assert!(DecisionTransport::GoogleDecisions.uses_google_api_key());
        assert!(!DecisionTransport::SystemOne.uses_google_api_key());
        assert!(!DecisionTransport::OpenRouterAlpha.uses_google_api_key());
    }

    /// Answers arrive as an object keyed by question id, each tagged with its kind.
    #[test]
    fn keyed_answers_decode_into_native_types() {
        let body = json!({
            "model": "jev-style-0.8b-decision-v3",
            "answers": {
                "risk": {"type": "noul", "noul": 0.67},
                "team": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.78,
                    "probabilities": {"billing": 0.78, "technical": 0.22}
                }
            }
        });
        let parsed = parse_response(&body).expect("answers decode");
        assert_eq!(parsed.answers.len(), 2);

        let risk = &parsed.answers["risk"];
        assert!(
            matches!(
                risk.value,
                DecisionValue::Boolean(switchyard_protocol::BooleanEstimate::ProbabilityTrue(_))
            ),
            "noul becomes a probability of true, not a threshold"
        );

        let team = &parsed.answers["team"];
        let DecisionValue::Choice {
            selected,
            probabilities,
        } = &team.value
        else {
            panic!("choice answer must decode as a choice");
        };
        assert_eq!(selected, "billing");
        assert_eq!(probabilities.as_ref().map(|p| p.len()), Some(2));
        assert_eq!(
            team.provider_confidence.map(|c| c.0),
            Some(0.78),
            "provider confidence is carried natively"
        );
    }

    #[test]
    fn a_score_answer_keeps_its_ordered_distribution() {
        let body = json!({
            "answers": {
                "severity": {
                    "type": "score",
                    "score": 1.25,
                    "confidence": 0.6,
                    "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
                    "legend": {"0": "low", "1": "medium", "2": "high"}
                }
            }
        });
        let parsed = parse_response(&body).expect("score decodes");
        let DecisionValue::Score {
            value,
            probabilities,
        } = &parsed.answers["severity"].value
        else {
            panic!("score answer must decode as a score");
        };
        assert!((value.0 - 1.25).abs() < f64::EPSILON);
        assert_eq!(probabilities.as_ref().map(|p| p.len()), Some(3));
    }

    #[test]
    fn a_response_without_answers_is_an_error() {
        assert!(parse_response(&json!({"ok": true})).is_err());
        assert!(parse_response(&json!({"answers": []})).is_err());
    }

    #[test]
    fn an_unknown_answer_type_is_refused() {
        let body = json!({"answers": {"q": {"type": "vibe", "noul": 0.5}}});
        assert!(parse_response(&body).is_err());
    }
}
