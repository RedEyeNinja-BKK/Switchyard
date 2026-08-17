// SPDX-FileCopyrightText: Copyright (c) 2026 LocalClaw integration fork.
// SPDX-License-Identifier: Apache-2.0

//! Resource-aware candidate routing.
//!
//! This algorithm implements the LocalClaw "smart routing" layer: Turnstone
//! selects a *work class* (bounded / agentic / reasoning) and this router
//! selects the *eligible model* for that class using live resource state.
//!
//! Eligibility (hard constraints) is evaluated before preference:
//!
//! - credential-pool allowance (OpenAI weekly / DeepSeek balance);
//! - provider availability / limit-reached / spend-control state;
//! - request context size against a candidate's economic context cap;
//! - request modality (text / image) against candidate modality support;
//! - reasoning contract (non-thinking vs thinking target).
//!
//! Candidates are consulted in configured preference order; the first
//! eligible candidate is selected. If none is eligible the request fails
//! closed with coarse eligibility reasons (detailed resource values are
//! kept in server-side logs, never in client-visible errors).
//!
//! Resource state is fetched on demand by an injected [`ResourceFetcher`]
//! (production: HTTP against the :8645 sanitized surface + DeepSeek balance)
//! and cached with a TTL so provider usage endpoints are not hit on every
//! request. A single in-flight refresh is coalesced with an async mutex so a
//! TTL expiry does not stampede the upstream.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result};
use switchyard_protocol::{Decision, ModelId, Request, Response};

/// A sanitized snapshot of the Turnstone OpenAI OAuth pool.
#[derive(Clone, Debug, Default)]
pub struct OpenAiResourceState {
    pub available: bool,
    pub limit_reached: bool,
    pub spend_control_reached: bool,
    pub weekly_used_percent: Option<f64>,
    pub weekly_reset_at: Option<i64>,
    pub credits_balance: Option<String>,
    /// Unix seconds of the last successful read (None when never/unknown).
    pub last_success_at: Option<i64>,
    /// Coarse error text when the most recent read failed (no secrets).
    pub error: Option<String>,
}

impl OpenAiResourceState {
    /// Remaining weekly allowance percentage (derived; no stored invariant).
    pub fn weekly_remaining_percent(&self) -> Option<f64> {
        self.weekly_used_percent.map(|used| 100.0 - used)
    }

    /// Hard eligibility for consuming the *included weekly allowance*.
    ///
    /// Purchased credits are intentionally NOT spent automatically once the
    /// weekly allowance is exhausted (owner policy). `credits_balance` is
    /// telemetry only. Unknown/error state is INELIGIBLE (fail closed).
    pub fn weekly_eligible(&self) -> bool {
        self.available
            && !self.limit_reached
            && !self.spend_control_reached
            && self
                .weekly_used_percent
                .map(|used| used < 100.0)
                .unwrap_or(true)
    }

    /// Error-marked state for pool-local failure isolation: this pool is
    /// unknown, so candidates that require it become ineligible, while
    /// independent healthy pools remain usable.
    pub fn unavailable_error(error: String) -> Self {
        Self {
            available: false,
            limit_reached: false,
            spend_control_reached: false,
            weekly_used_percent: None,
            weekly_reset_at: None,
            credits_balance: None,
            last_success_at: None,
            error: Some(error),
        }
    }
}

/// A sanitized snapshot of the Turnstone DeepSeek pool.
#[derive(Clone, Debug, Default)]
pub struct DeepSeekResourceState {
    pub is_available: bool,
    pub currency: String,
    pub total_balance: String,
    pub granted_balance: String,
    pub topped_up_balance: String,
    /// Unix seconds of the last successful read (None when never/unknown).
    pub last_success_at: Option<i64>,
    /// Coarse error text when the most recent read failed (no secrets).
    pub error: Option<String>,
}

impl DeepSeekResourceState {
    /// Hard eligibility: available positive balance in the configured
    /// currency. No arbitrary CONSERVE cutoff yet (owner policy). Unknown/
    /// error state is INELIGIBLE (fail closed).
    pub fn eligible(&self) -> bool {
        if !self.is_available {
            return false;
        }
        self.total_balance
            .trim()
            .parse::<f64>()
            .map(|balance| balance > 0.0)
            .unwrap_or(false)
    }

    /// Error-marked state for pool-local failure isolation.
    pub fn unavailable_error(error: String) -> Self {
        Self {
            is_available: false,
            currency: String::new(),
            total_balance: String::new(),
            granted_balance: String::new(),
            topped_up_balance: String::new(),
            last_success_at: None,
            error: Some(error),
        }
    }
}

/// One fetch of every configured pool's state.
#[derive(Clone, Debug, Default)]
pub struct ResourceSnapshot {
    pub openai: Option<OpenAiResourceState>,
    pub deepseek: Option<DeepSeekResourceState>,
}

/// Injectable resource fetcher. Production uses HTTP; tests inject mocks.
#[async_trait]
pub trait ResourceFetcher: Send + Sync {
    async fn fetch(&self) -> Result<ResourceSnapshot>;
}

/// Shared, TTL-cached resource state for one pool definition.
///
/// The cache is guarded by a tokio async mutex and the lock is held across
/// the fetch, so only one task refreshes at a time; concurrent callers wait
/// on the lock and then observe the fresh snapshot. Cache hits return a
/// cheap `Arc` clone.
pub struct ResourceState {
    fetcher: Arc<dyn ResourceFetcher>,
    ttl: Duration,
    cache: Mutex<Option<(std::time::Instant, Arc<ResourceSnapshot>)>>,
}

impl ResourceState {
    pub fn new(fetcher: Arc<dyn ResourceFetcher>, ttl: Duration) -> Self {
        Self {
            fetcher,
            ttl,
            cache: Mutex::new(None),
        }
    }

    pub async fn snapshot(&self) -> Result<Arc<ResourceSnapshot>> {
        let mut guard = self.cache.lock().await;
        if let Some((observed_at, snapshot)) = guard.as_ref() {
            if observed_at.elapsed() < self.ttl {
                return Ok(Arc::clone(snapshot));
            }
        }
        // Hold the lock across the fetch: one refresher, others wait and
        // read the fresh snapshot (no stampede, no guard across await on a
        // blocking mutex).
        let snapshot = self.fetcher.fetch().await?;
        let snapshot = Arc::new(snapshot);
        *guard = Some((std::time::Instant::now(), Arc::clone(&snapshot)));
        Ok(snapshot)
    }
}

/// Which resource pool a candidate requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    None,
    OpenAi,
    DeepSeek,
}

/// Reasoning contract a candidate enforces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningPolicy {
    Any,
    NonThinking,
    Thinking,
}

/// Semantic work class carried structurally by the request (never guessed
/// from prompt text). LEGACY taxonomy (Phase 1) kept for backward
/// compatibility: `bounded` | `agentic` | `reasoning`.
///
/// Phase 2 correction: "reasoning" was not a work shape. Work shape and
/// reasoning intent are ORTHOGONAL dimensions. Legacy values translate:
///   bounded   -> (bounded,   none)
///   agentic   -> (agentic,   none)
///   reasoning -> (bounded,   deliberate)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkClass {
    Bounded,
    Agentic,
    Reasoning,
}

impl std::str::FromStr for WorkClass {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bounded" => Ok(WorkClass::Bounded),
            "agentic" => Ok(WorkClass::Agentic),
            "reasoning" => Ok(WorkClass::Reasoning),
            other => Err(format!(
                "invalid work_class {other:?} (expected bounded | agentic | reasoning)"
            )),
        }
    }
}

/// WORK SHAPE: describes the shape of the work (bounded task vs long-lived
/// agentic session). Independent of reasoning intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkShape {
    Bounded,
    Agentic,
}

impl std::str::FromStr for WorkShape {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bounded" => Ok(WorkShape::Bounded),
            "agentic" => Ok(WorkShape::Agentic),
            other => Err(format!(
                "invalid work_shape {other:?} (expected bounded | agentic)"
            )),
        }
    }
}

/// REASONING INTENT: whether the caller wants deliberate reasoning behavior.
/// Independent of work shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningIntent {
    None,
    Deliberate,
}

impl std::str::FromStr for ReasoningIntent {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(ReasoningIntent::None),
            "deliberate" => Ok(ReasoningIntent::Deliberate),
            other => Err(format!(
                "invalid reasoning_intent {other:?} (expected none | deliberate)"
            )),
        }
    }
}

/// The orthogonal semantic contract: work shape × reasoning intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticContract {
    pub shape: WorkShape,
    pub intent: ReasoningIntent,
}

impl SemanticContract {
    /// Translate a LEGACY `work_class` value into the orthogonal contract.
    pub fn from_work_class(class: WorkClass) -> Self {
        match class {
            WorkClass::Bounded => SemanticContract {
                shape: WorkShape::Bounded,
                intent: ReasoningIntent::None,
            },
            WorkClass::Agentic => SemanticContract {
                shape: WorkShape::Agentic,
                intent: ReasoningIntent::None,
            },
            WorkClass::Reasoning => SemanticContract {
                shape: WorkShape::Bounded,
                intent: ReasoningIntent::Deliberate,
            },
        }
    }
}

/// How a resource router learns the request's semantic contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractFilter {
    /// No contract filtering (candidate `shapes`/`intents` lists ignored).
    Any,
    /// A fixed contract configured on the route.
    Fixed(SemanticContract),
    /// The contract is read from structured request metadata:
    /// prefer `work_shape` + `reasoning_intent`, fall back to legacy
    /// `work_class` (translated). Missing => error, never guessed.
    Request,
}

/// Request modality derived from message + instruction content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
}

/// One ordered candidate target with its eligibility contract.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub target: ModelId,
    pub pool: Pool,
    pub context_cap_tokens: Option<u64>,
    pub modalities: Vec<Modality>,
    pub reasoning: ReasoningPolicy,
    /// Work shapes this candidate serves. Empty = all shapes.
    pub shapes: Vec<WorkShape>,
    /// Reasoning intents this candidate serves. Empty = all intents.
    pub intents: Vec<ReasoningIntent>,
}

/// Conservative per-image token estimate for context-cap eligibility.
/// Images contribute far more than a single text token; 1024 is a defensible
/// floor for a single image input. Text uses characters / 4 (the same class
/// of pre-calibration estimate Turnstone itself uses).
const IMAGE_TOKEN_ESTIMATE: u64 = 1024;

/// Estimate request size for context-cap eligibility.
fn estimate_request_tokens(request: &Request) -> u64 {
    let mut tokens: u64 = 0;
    for message in &request.llm_request.messages {
        for block in &message.content {
            match block {
                switchyard_protocol::ContentBlock::Text { text } => {
                    tokens += text.chars().count() as u64 / 4;
                }
                switchyard_protocol::ContentBlock::Image { .. } => {
                    tokens += IMAGE_TOKEN_ESTIMATE;
                }
                _ => {}
            }
        }
    }
    for block in &request.llm_request.instructions {
        for content in &block.content {
            match content {
                switchyard_protocol::ContentBlock::Text { text } => {
                    tokens += text.chars().count() as u64 / 4;
                }
                switchyard_protocol::ContentBlock::Image { .. } => {
                    tokens += IMAGE_TOKEN_ESTIMATE;
                }
                _ => {}
            }
        }
    }
    tokens
}

fn request_modality(request: &Request) -> Modality {
    let mut has_image = false;
    for message in &request.llm_request.messages {
        for block in &message.content {
            if matches!(block, switchyard_protocol::ContentBlock::Image { .. }) {
                has_image = true;
            }
        }
    }
    for block in &request.llm_request.instructions {
        for content in &block.content {
            if matches!(content, switchyard_protocol::ContentBlock::Image { .. }) {
                has_image = true;
            }
        }
    }
    if has_image {
        Modality::Image
    } else {
        Modality::Text
    }
}

fn request_reasoning_effort(request: &Request) -> Option<String> {
    request.llm_request.reasoning.effort.clone()
}

/// Resource-aware candidate router. See module docs.
pub struct ResourceRouter {
    name: &'static str,
    candidates: Vec<Candidate>,
    state: Arc<ResourceState>,
    contract_filter: ContractFilter,
}

impl ResourceRouter {
    pub fn new(name: &'static str, candidates: Vec<Candidate>, state: Arc<ResourceState>) -> Self {
        Self {
            name,
            candidates,
            state,
            contract_filter: ContractFilter::Any,
        }
    }

    pub fn with_contract_filter(mut self, contract_filter: ContractFilter) -> Self {
        self.contract_filter = contract_filter;
        self
    }

    /// Resolve the effective semantic contract for this request.
    ///
    /// `ContractFilter::Request` reads STRUCTURAL metadata from the request
    /// extensions. Prefer `work_shape` + `reasoning_intent` (orthogonal);
    /// fall back to the LEGACY `work_class` (translated). Missing or invalid
    /// metadata is a hard error — the router never guesses critical semantics
    /// from prompt text.
    fn effective_contract(&self, request: &Request) -> Result<Option<SemanticContract>> {
        match self.contract_filter {
            ContractFilter::Any => Ok(None),
            ContractFilter::Fixed(contract) => Ok(Some(contract)),
            ContractFilter::Request => {
                let fields = &request.llm_request.extensions.fields;

                // Preferred: orthogonal dimensions.
                let shape_raw = fields.get("work_shape").and_then(serde_json::Value::as_str);
                let intent_raw = fields
                    .get("reasoning_intent")
                    .and_then(serde_json::Value::as_str);
                if let (Some(shape_raw), Some(intent_raw)) = (shape_raw, intent_raw) {
                    let shape = shape_raw.parse::<WorkShape>().map_err(|message| {
                        LibsyError::AlgorithmError {
                            message: format!("{}: {message}", self.name()),
                        }
                    })?;
                    let intent = intent_raw.parse::<ReasoningIntent>().map_err(|message| {
                        LibsyError::AlgorithmError {
                            message: format!("{}: {message}", self.name()),
                        }
                    })?;
                    return Ok(Some(SemanticContract { shape, intent }));
                }

                // Legacy fallback: single work_class value.
                if let Some(raw) = fields.get("work_class").and_then(serde_json::Value::as_str) {
                    let class = raw.parse::<WorkClass>().map_err(|message| {
                        LibsyError::AlgorithmError {
                            message: format!("{}: {message}", self.name()),
                        }
                    })?;
                    return Ok(Some(SemanticContract::from_work_class(class)));
                }

                Err(LibsyError::AlgorithmError {
                    message: format!(
                        "{}: work_shape + reasoning_intent metadata required (or legacy work_class); refusing to guess from prompt text",
                        self.name()
                    ),
                })
            }
        }
    }
}

/// Coarse reason shown to the caller when a candidate is ineligible.
/// Detailed resource values stay in the server-side log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IneligibleKind {
    OpenAiPool,
    DeepSeekPool,
    ContextCap,
    Modality,
    ReasoningContract,
    WorkShape,
    ReasoningIntent,
}

fn candidate_eligible(
    candidate: &Candidate,
    snapshot: &ResourceSnapshot,
    request_tokens: u64,
    modality: Modality,
    reasoning_effort: &Option<String>,
    contract: Option<SemanticContract>,
    reasons: &mut Vec<(String, IneligibleKind)>,
) -> bool {
    // Semantic-contract eligibility: a contract-filtered route only considers
    // candidates that serve the resolved shape and intent (empty = all).
    if let Some(contract) = contract {
        if !candidate.shapes.is_empty() && !candidate.shapes.contains(&contract.shape) {
            reasons.push((candidate.target.to_string(), IneligibleKind::WorkShape));
            return false;
        }
        if !candidate.intents.is_empty() && !candidate.intents.contains(&contract.intent) {
            reasons.push((candidate.target.to_string(), IneligibleKind::ReasoningIntent));
            return false;
        }
    }

    match candidate.pool {
        Pool::OpenAi => {
            let Some(openai) = snapshot.openai.as_ref() else {
                reasons.push((candidate.target.to_string(), IneligibleKind::OpenAiPool));
                return false;
            };
            if !openai.weekly_eligible() {
                tracing::info!(
                    target = %candidate.target,
                    available = openai.available,
                    limit_reached = openai.limit_reached,
                    spend_control_reached = openai.spend_control_reached,
                    weekly_used_percent = openai.weekly_used_percent,
                    "openai pool ineligible for candidate"
                );
                reasons.push((candidate.target.to_string(), IneligibleKind::OpenAiPool));
                return false;
            }
        }
        Pool::DeepSeek => {
            let Some(deepseek) = snapshot.deepseek.as_ref() else {
                reasons.push((candidate.target.to_string(), IneligibleKind::DeepSeekPool));
                return false;
            };
            if !deepseek.eligible() {
                tracing::info!(
                    target = %candidate.target,
                    is_available = deepseek.is_available,
                    total_balance = deepseek.total_balance,
                    currency = deepseek.currency,
                    "deepseek pool ineligible for candidate"
                );
                reasons.push((candidate.target.to_string(), IneligibleKind::DeepSeekPool));
                return false;
            }
        }
        Pool::None => {}
    }

    if let Some(cap) = candidate.context_cap_tokens {
        if request_tokens > cap {
            reasons.push((candidate.target.to_string(), IneligibleKind::ContextCap));
            return false;
        }
    }

    if !candidate.modalities.contains(&modality) {
        reasons.push((candidate.target.to_string(), IneligibleKind::Modality));
        return false;
    }

    match candidate.reasoning {
        ReasoningPolicy::NonThinking => {
            if let Some(effort) = reasoning_effort {
                if !effort.is_empty() && effort != "none" {
                    reasons.push((
                        candidate.target.to_string(),
                        IneligibleKind::ReasoningContract,
                    ));
                    return false;
                }
            }
        }
        // Symmetric guard: a thinking-only candidate must not serve a request
        // that explicitly disables reasoning.
        ReasoningPolicy::Thinking => {
            if reasoning_effort.as_deref() == Some("none") {
                reasons.push((
                    candidate.target.to_string(),
                    IneligibleKind::ReasoningContract,
                ));
                return false;
            }
        }
        ReasoningPolicy::Any => {}
    }

    true
}

fn reason_label(kind: IneligibleKind) -> &'static str {
    match kind {
        IneligibleKind::OpenAiPool => "openai pool ineligible",
        IneligibleKind::DeepSeekPool => "deepseek pool unavailable",
        IneligibleKind::ContextCap => "request exceeds candidate context cap",
        IneligibleKind::Modality => "request modality not supported by candidate",
        IneligibleKind::ReasoningContract => "request reasoning contract conflicts with candidate",
        IneligibleKind::WorkShape => "candidate does not serve the request work shape",
        IneligibleKind::ReasoningIntent => "candidate does not serve the request reasoning intent",
    }
}

#[async_trait]
impl Algorithm for ResourceRouter {
    fn name(&self) -> &str {
        self.name
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        let snapshot = self.state.snapshot().await.map_err(|error| {
            LibsyError::AlgorithmError {
                message: format!(
                    "{}: resource state fetch failed ({}); see server logs",
                    self.name(),
                    error
                ),
            }
        })?;

        let contract = self.effective_contract(&request)?;
        let request_tokens = estimate_request_tokens(&request);
        let modality = request_modality(&request);
        let reasoning_effort = request_reasoning_effort(&request);

        let mut reasons = Vec::new();
        for candidate in &self.candidates {
            if candidate_eligible(
                candidate,
                &snapshot,
                request_tokens,
                modality,
                &reasoning_effort,
                contract,
                &mut reasons,
            ) {
                tracing::info!(
                    target = %candidate.target,
                    pool = ?candidate.pool,
                    contract = ?contract,
                    "{} selected eligible target",
                    self.name()
                );
                let decision = Decision::new(candidate.target.clone(), true);
                driver.decide(decision.clone()).await?;
                return driver
                    .call_model(request, vec![candidate.target.clone()], true)
                    .await;
            }
        }

        // Coarse client-visible reasons only; detailed values stay in logs.
        let coarse: Vec<&'static str> = reasons
            .iter()
            .map(|(_, kind)| reason_label(*kind))
            .collect();
        Err(LibsyError::AlgorithmError {
            message: format!(
                "{}: no eligible target ({}); reasons: {}",
                self.name(),
                self.candidates.len(),
                coarse.join("; ")
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::core::algorithm::Algorithm;
    use crate::core::testing::{echo, test_drive};
    use switchyard_protocol::{Request, text_request};

    /// Stub fetcher returning a fixed snapshot.
    struct StubFetcher(ResourceSnapshot);

    #[async_trait]
    impl ResourceFetcher for StubFetcher {
        async fn fetch(&self) -> Result<ResourceSnapshot> {
            Ok(self.0.clone())
        }
    }

    fn snapshot(openai: OpenAiResourceState, deepseek: DeepSeekResourceState) -> ResourceSnapshot {
        ResourceSnapshot {
            openai: Some(openai),
            deepseek: Some(deepseek),
        }
    }

    fn openai_healthy() -> OpenAiResourceState {
        OpenAiResourceState {
            available: true,
            limit_reached: false,
            spend_control_reached: false,
            weekly_used_percent: Some(72.0),
            weekly_reset_at: Some(1787197109),
            credits_balance: Some("0".into()),
            last_success_at: Some(1786959960),
            error: None,
        }
    }

    fn openai_exhausted() -> OpenAiResourceState {
        OpenAiResourceState {
            available: true,
            limit_reached: true,
            spend_control_reached: false,
            weekly_used_percent: Some(100.0),
            weekly_reset_at: Some(1787197109),
            credits_balance: Some("500".into()),
            last_success_at: Some(1786959960),
            error: None,
        }
    }

    fn openai_spend_control() -> OpenAiResourceState {
        OpenAiResourceState {
            available: true,
            limit_reached: false,
            spend_control_reached: true,
            weekly_used_percent: Some(60.0),
            weekly_reset_at: Some(1787197109),
            credits_balance: Some("0".into()),
            last_success_at: Some(1786959960),
            error: None,
        }
    }

    fn openai_unknown() -> OpenAiResourceState {
        OpenAiResourceState::unavailable_error("openai-resource fetch failed (HTTP 500)".into())
    }

    fn deepseek_available() -> DeepSeekResourceState {
        DeepSeekResourceState {
            is_available: true,
            currency: "CNY".into(),
            total_balance: "23.98".into(),
            granted_balance: "0.00".into(),
            topped_up_balance: "23.98".into(),
            last_success_at: Some(1786959960),
            error: None,
        }
    }

    fn deepseek_empty() -> DeepSeekResourceState {
        DeepSeekResourceState {
            is_available: true,
            currency: "CNY".into(),
            total_balance: "0.00".into(),
            granted_balance: "0.00".into(),
            topped_up_balance: "0.00".into(),
            last_success_at: Some(1786959960),
            error: None,
        }
    }

    fn deepseek_unknown() -> DeepSeekResourceState {
        DeepSeekResourceState::unavailable_error("deepseek-resource fetch failed (HTTP 500)".into())
    }

    fn text_request_hi() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        }
    }

    fn make_router(candidates: Vec<Candidate>, snapshot: ResourceSnapshot) -> Arc<dyn Algorithm> {
        let state = Arc::new(ResourceState::new(
            Arc::new(StubFetcher(snapshot)),
            Duration::from_secs(30),
        ));
        Arc::new(ResourceRouter::new("test_resource", candidates, state))
    }

    fn candidate(name: &str, pool: Pool) -> Candidate {
        Candidate {
            target: name.into(),
            pool,
            context_cap_tokens: Some(266_000),
            modalities: vec![Modality::Text],
            reasoning: ReasoningPolicy::NonThinking,
            shapes: vec![],
            intents: vec![],
        }
    }

    fn candidate_for(name: &str, pool: Pool, shape: WorkShape) -> Candidate {
        Candidate {
            shapes: vec![shape],
            ..candidate(name, pool)
        }
    }

    fn candidate_for_contract(
        name: &str,
        pool: Pool,
        shape: WorkShape,
        intent: ReasoningIntent,
    ) -> Candidate {
        Candidate {
            shapes: vec![shape],
            intents: vec![intent],
            ..candidate(name, pool)
        }
    }

    fn request_with_work_class(class: &str, prompt: &str) -> Request {
        let mut llm = text_request(Some("auto".to_string()), prompt);
        llm.extensions
            .fields
            .insert("work_class".to_string(), serde_json::Value::String(class.to_string()));
        Request {
            llm_request: llm,
            raw_request: None,
            metadata: None,
        }
    }

    fn request_with_contract(shape: &str, intent: &str, prompt: &str) -> Request {
        let mut llm = text_request(Some("auto".to_string()), prompt);
        llm.extensions.fields.insert(
            "work_shape".to_string(),
            serde_json::Value::String(shape.to_string()),
        );
        llm.extensions.fields.insert(
            "reasoning_intent".to_string(),
            serde_json::Value::String(intent.to_string()),
        );
        Request {
            llm_request: llm,
            raw_request: None,
            metadata: None,
        }
    }

    fn make_request_router(
        candidates: Vec<Candidate>,
        snapshot: ResourceSnapshot,
    ) -> Arc<dyn Algorithm> {
        let state = Arc::new(ResourceState::new(
            Arc::new(StubFetcher(snapshot)),
            Duration::from_secs(30),
        ));
        Arc::new(
            ResourceRouter::new("test_request_resource", candidates, state)
                .with_contract_filter(ContractFilter::Request),
        )
    }

    /// The universal candidate set (orthogonal semantics):
    ///   bounded+none       -> Luna NT (OpenAI) preferred, Flash NT fallback
    ///   agentic+none       -> Flash NT (Luna excluded)
    ///   bounded+deliberate -> Flash thinking (legacy reasoning)
    ///   agentic+deliberate -> Flash thinking (newly surfaced Hermes case)
    fn universal_candidates() -> Vec<Candidate> {
        let mut luna = candidate_for_contract(
            "testing/luna",
            Pool::OpenAi,
            WorkShape::Bounded,
            ReasoningIntent::None,
        );
        luna.context_cap_tokens = Some(266_000);
        let mut flash_nt = candidate_for_contract(
            "testing/flash-nt",
            Pool::DeepSeek,
            WorkShape::Agentic,
            ReasoningIntent::None,
        );
        flash_nt.shapes = vec![WorkShape::Bounded, WorkShape::Agentic];
        flash_nt.context_cap_tokens = Some(750_000);
        let mut flash_think = candidate_for_contract(
            "testing/flash-think",
            Pool::DeepSeek,
            WorkShape::Bounded,
            ReasoningIntent::Deliberate,
        );
        flash_think.shapes = vec![WorkShape::Bounded, WorkShape::Agentic];
        flash_think.reasoning = ReasoningPolicy::Thinking;
        flash_think.context_cap_tokens = Some(1_048_576);
        vec![luna, flash_nt, flash_think]
    }

    #[tokio::test]
    async fn openai_healthy_bounded_text_selects_luna() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn openai_weekly_exhausted_bounded_text_falls_to_flash() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_exhausted(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn credits_positive_but_weekly_exhausted_still_selects_flash() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_exhausted(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn spend_control_reached_excludes_luna() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_spend_control(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn deepseek_empty_excludes_flash() -> crate::Result<()> {
        let router = make_router(
            vec![candidate("testing/flash", Pool::DeepSeek)],
            snapshot(openai_healthy(), deepseek_empty()),
        );
        let result = test_drive(router, text_request_hi(), echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("no eligible target"));
        // Coarse reason only — no balance/percentage values in the error.
        assert!(err.to_string().contains("deepseek pool unavailable"));
        assert!(!err.to_string().contains("0.00"));
        Ok(())
    }

    #[tokio::test]
    async fn context_over_cap_excludes_luna() -> crate::Result<()> {
        let mut cand = candidate("testing/luna", Pool::OpenAi);
        cand.context_cap_tokens = Some(8); // tiny cap
        let request = Request {
            llm_request: text_request(
                Some("auto".to_string()),
                "this is a substantially longer request body used to exceed the tiny configured context cap for the luna candidate so that eligibility falls through to flash",
            ),
            raw_request: None,
            metadata: None,
        };
        let router = make_router(
            vec![cand, candidate("testing/flash", Pool::DeepSeek)],
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request, echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn thinking_candidate_rejects_explicit_none() -> crate::Result<()> {
        let mut cand = candidate("testing/thinking", Pool::DeepSeek);
        cand.reasoning = ReasoningPolicy::Thinking;
        let router = make_router(
            vec![cand, candidate("testing/flash", Pool::DeepSeek)],
            snapshot(openai_healthy(), deepseek_available()),
        );
        let mut request = text_request_hi();
        request.llm_request.reasoning.effort = Some("none".to_string());
        let (trace, _) = test_drive(router, request, echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn modality_incompatible_candidate_excluded() -> crate::Result<()> {
        // §29.14: multimodal request must not select a text-only candidate
        // merely because its pool is healthier.
        let mut text_only = candidate("testing/text-only", Pool::DeepSeek);
        text_only.modalities = vec![Modality::Text];
        let mut image_ok = candidate("testing/image-ok", Pool::OpenAi);
        image_ok.modalities = vec![Modality::Text, Modality::Image];
        let router = make_router(
            vec![text_only, image_ok],
            snapshot(openai_healthy(), deepseek_available()),
        );
        let mut llm = text_request(Some("auto".to_string()), "describe this image");
        llm.messages.push(switchyard_protocol::Message {
            role: switchyard_protocol::Role::User,
            content: vec![switchyard_protocol::ContentBlock::Image {
                source: switchyard_protocol::ImageSource::Base64 {
                    media_type: Some("image/png".to_string()),
                    data: "iVBORw0KGgo=".to_string(),
                },
            }],
        });
        let request = Request {
            llm_request: llm,
            raw_request: None,
            metadata: None,
        };
        let (trace, _) = test_drive(router, request, echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/image-ok");
        Ok(())
    }

    // ── Provider-pool-local failure isolation (§3) ─────────────────────
    // One pool's telemetry unknown must not disable every smart route.

    #[tokio::test]
    async fn openai_unknown_deepseek_healthy_bounded_selects_flash() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_unknown(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn deepseek_unknown_openai_healthy_bounded_selects_luna() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_healthy(), deepseek_unknown()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn both_pools_unknown_controlled_no_eligible_target() -> crate::Result<()> {
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_unknown(), deepseek_unknown()),
        );
        let result = test_drive(router, text_request_hi(), echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("no eligible target"));
        assert!(err.to_string().contains("openai pool ineligible"));
        assert!(err.to_string().contains("deepseek pool unavailable"));
        Ok(())
    }

    // ── Universal single-endpoint (orthogonal semantic contract) ────────
    // work_shape (bounded|agentic) × reasoning_intent (none|deliberate),
    // plus LEGACY work_class translation. One route id `localclaw/smart`.

    #[tokio::test]
    async fn universal_bounded_metadata_selects_luna_when_healthy() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request_with_work_class("bounded", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn universal_bounded_openai_unknown_falls_to_flash_nt() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_unknown(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request_with_work_class("bounded", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-nt");
        Ok(())
    }

    #[tokio::test]
    async fn universal_agentic_metadata_selects_flash_nt_luna_excluded() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request_with_work_class("agentic", "hi"), echo()).await?;
        // Luna is bounded-shape only; must never be chosen for agentic even
        // though OpenAI is healthy.
        assert_eq!(trace[0].selected_model_id(), "testing/flash-nt");
        Ok(())
    }

    #[tokio::test]
    async fn universal_agentic_large_context_luna_never_considered() -> crate::Result<()> {
        // ~320K chars -> ~80K estimated tokens; Luna is shape-excluded for
        // agentic regardless of its 266K cap.
        let prompt = "x".repeat(320_000);
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request_with_work_class("agentic", &prompt), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-nt");
        Ok(())
    }

    #[tokio::test]
    async fn universal_reasoning_metadata_selects_flash_thinking() -> crate::Result<()> {
        // Legacy work_class=reasoning translates to (bounded, deliberate).
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) =
            test_drive(router, request_with_work_class("reasoning", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-think");
        Ok(())
    }

    // ── §7 matrix: work_shape × reasoning_intent (orthogonal) ─────────

    #[tokio::test]
    async fn contract_bounded_none_selects_luna_when_healthy() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) =
            test_drive(router, request_with_contract("bounded", "none", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn contract_bounded_none_openai_exhausted_selects_flash_nt() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_exhausted(), deepseek_available()),
        );
        let (trace, _) =
            test_drive(router, request_with_contract("bounded", "none", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-nt");
        Ok(())
    }

    #[tokio::test]
    async fn contract_agentic_none_selects_flash_nt() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) =
            test_drive(router, request_with_contract("agentic", "none", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-nt");
        Ok(())
    }

    #[tokio::test]
    async fn contract_bounded_deliberate_selects_flash_thinking() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(
            router,
            request_with_contract("bounded", "deliberate", "hi"),
            echo(),
        )
        .await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-think");
        Ok(())
    }

    #[tokio::test]
    async fn contract_agentic_deliberate_selects_flash_thinking() -> crate::Result<()> {
        // THE newly surfaced Hermes case: agentic work CAN require deliberate
        // reasoning. It must NOT be rejected as "non-thinking only".
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(
            router,
            request_with_contract("agentic", "deliberate", "hi"),
            echo(),
        )
        .await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash-think");
        Ok(())
    }

    #[tokio::test]
    async fn contract_agentic_deliberate_deepseek_unknown_no_luna_downgrade()
    -> crate::Result<()> {
        // Agentic + deliberate + DeepSeek unknown: flash-think ineligible.
        // DO NOT silently downgrade to Luna NT (shape+intent incompatible) ->
        // controlled no-eligible-target.
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_unknown()),
        );
        let result = test_drive(
            router,
            request_with_contract("agentic", "deliberate", "hi"),
            echo(),
        )
        .await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("no eligible target"));
        assert!(!err.to_string().contains("testing/luna"));
        Ok(())
    }

    #[tokio::test]
    async fn contract_agentic_none_deepseek_unknown_no_luna_downgrade() -> crate::Result<()> {
        // Agentic + none + DeepSeek unknown: Flash NT ineligible; Luna is
        // shape-excluded even though OpenAI healthy -> controlled error.
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_unknown()),
        );
        let result = test_drive(
            router,
            request_with_contract("agentic", "none", "hi"),
            echo(),
        )
        .await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("no eligible target"));
        assert!(!err.to_string().contains("testing/luna"));
        Ok(())
    }

    #[tokio::test]
    async fn contract_agentic_none_deepseek_empty_excludes_flash() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_empty()),
        );
        let result = test_drive(
            router,
            request_with_contract("agentic", "none", "hi"),
            echo(),
        )
        .await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn universal_missing_metadata_errors_no_guessing() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let result = test_drive(router, text_request_hi(), echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("work_shape + reasoning_intent metadata required"));
        Ok(())
    }

    #[tokio::test]
    async fn universal_invalid_metadata_errors() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let result = test_drive(router, request_with_work_class("chatty", "hi"), echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("invalid work_class"));
        Ok(())
    }

    #[tokio::test]
    async fn universal_invalid_orthogonal_metadata_errors() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let result =
            test_drive(router, request_with_contract("chatty", "none", "hi"), echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("invalid work_shape"));
        Ok(())
    }

    #[tokio::test]
    async fn universal_reasoning_deepseek_unknown_no_luna_substitution() -> crate::Result<()> {
        // Legacy reasoning -> (bounded, deliberate): thinking candidate
        // ineligible and Luna NT is NOT a semantic substitute -> controlled
        // no-eligible-target.
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_unknown()),
        );
        let result = test_drive(router, request_with_work_class("reasoning", "hi"), echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("no eligible target"));
        assert!(!err.to_string().contains("testing/luna"));
        Ok(())
    }
}
