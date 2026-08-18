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
    /// telemetry only.
    ///
    /// NOTE: this predicate is NOT the router's gate. The resource router
    /// uses [`Self::confirmed_exhausted`] so that unknown/error telemetry
    /// fails TOWARD Luna instead of spilling (owner policy). Keep this
    /// method for allowance math; do not re-introduce it as a routing gate.
    pub fn weekly_eligible(&self) -> bool {
        self.available
            && !self.limit_reached
            && !self.spend_control_reached
            && self
                .weekly_used_percent
                .map(|used| used < 100.0)
                .unwrap_or(true)
    }

    /// CONFIRMED included-allowance exhaustion ONLY: the provider-reported
    /// limit flag (`rate_limit.limit_reached` from the OpenAI/Codex /usage
    /// surface) or used >= 100% of the included weekly allowance
    /// (`rate_limit.primary_window.used_percent`).
    ///
    /// Deliberately EXCLUDED: `spend_control_reached` (spend-control / paid
    /// overage policy state, NOT included-allowance exhaustion — must not
    /// unlock a fallback), `available=false` and unknown/stale/error telemetry
    /// (bounded lanes fail toward Luna instead of spilling).
    pub fn confirmed_exhausted(&self) -> bool {
        self.limit_reached
            || self
                .weekly_used_percent
                .map(|used| used >= 100.0)
                .unwrap_or(false)
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

/// Shared, last-known sanitized resource snapshot for observability surfaces
/// (e.g. the read-only `/v1/resource/deepseek` endpoint).
///
/// The server attaches one telemetry slot to resource states that carry a
/// DeepSeek fetcher; after each successful TTL refresh the fresh snapshot is
/// published here. Never holds secrets — it is the same sanitized state the
/// router already uses. A plain `std::sync::Mutex` suffices: `get`/`set` are
/// never held across an await, so the routing hot path cannot deadlock.
#[derive(Clone, Debug, Default)]
pub struct SharedResourceTelemetry(
    std::sync::Arc<std::sync::Mutex<Option<Arc<ResourceSnapshot>>>>,
);

impl SharedResourceTelemetry {
    pub fn new() -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(None)))
    }

    /// Latest published snapshot, if any.
    pub fn get(&self) -> Option<Arc<ResourceSnapshot>> {
        self.0.lock().expect("resource telemetry mutex poisoned").clone()
    }

    /// Publish a snapshot (used by ResourceState after refresh; public so
    /// tests can seed a deterministic state).
    pub fn set(&self, snapshot: Arc<ResourceSnapshot>) {
        *self.0.lock().expect("resource telemetry mutex poisoned") = Some(snapshot);
    }
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
    telemetry: std::sync::OnceLock<SharedResourceTelemetry>,
}

impl ResourceState {
    pub fn new(fetcher: Arc<dyn ResourceFetcher>, ttl: Duration) -> Self {
        Self {
            fetcher,
            ttl,
            cache: Mutex::new(None),
            telemetry: std::sync::OnceLock::new(),
        }
    }

    /// Attach a shared observability slot. The last successful snapshot is
    /// published to it after every refresh. Idempotent: the first attach wins.
    pub fn attach_telemetry(&self, telemetry: SharedResourceTelemetry) {
        let _ = self.telemetry.set(telemetry);
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
        // Publish after the cache store. The telemetry mutex is a plain sync
        // mutex (never held across await), so this cannot deadlock and does
        // not extend an async critical section.
        if let Some(telemetry) = self.telemetry.get() {
            telemetry.set(Arc::clone(&snapshot));
        }
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
    ///
    /// Operator correction (2026-08-18): `bounded` work may NOT carry
    /// deliberate reasoning intent (that would be an inference-model escape
    /// hatch). Legacy `reasoning` is therefore classified as NON-bounded
    /// deliberate work: (agentic, deliberate) → Flash thinking.
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
                shape: WorkShape::Agentic,
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
    ///
    /// Operator correction (2026-08-18): `bounded` work MUST carry
    /// `reasoning_intent=none`. A contradictory bounded+deliberate contract
    /// is REJECTED — reasoning requirements belong to work classification,
    /// not an inference-model escape hatch.
    fn effective_contract(&self, request: &Request) -> Result<Option<SemanticContract>> {
        fn reject_contradiction(
            name: &str,
            contract: SemanticContract,
        ) -> Result<SemanticContract> {
            if contract.shape == WorkShape::Bounded && contract.intent == ReasoningIntent::Deliberate
            {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "{}: bounded work_shape with reasoning_intent=deliberate is contradictory; reclassify the task as deliberate/non-bounded before routing",
                        name
                    ),
                });
            }
            Ok(contract)
        }

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
                    return reject_contradiction(self.name(), SemanticContract { shape, intent })
                        .map(Some);
                }

                // Legacy fallback: single work_class value.
                if let Some(raw) = fields.get("work_class").and_then(serde_json::Value::as_str) {
                    let class = raw.parse::<WorkClass>().map_err(|message| {
                        LibsyError::AlgorithmError {
                            message: format!("{}: {message}", self.name()),
                        }
                    })?;
                    return reject_contradiction(
                        self.name(),
                        SemanticContract::from_work_class(class),
                    )
                    .map(Some);
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

/// Semantic-only fit (no resource state, no context cap): does this candidate
/// serve the request's shape/intent/modality/reasoning contract? Mirrors the
/// semantic block of [`candidate_eligible`]; used by [`ResourceRouter::route`]
/// to decide whether a DeepSeek fallback is permitted in a route that also
/// carries an OpenAI candidate.
fn candidate_semantic_eligible(
    candidate: &Candidate,
    modality: Modality,
    reasoning_effort: &Option<String>,
    contract: Option<SemanticContract>,
) -> bool {
    if let Some(contract) = contract {
        if !candidate.shapes.is_empty() && !candidate.shapes.contains(&contract.shape) {
            return false;
        }
        if !candidate.intents.is_empty() && !candidate.intents.contains(&contract.intent) {
            return false;
        }
    }
    if !candidate.modalities.contains(&modality) {
        return false;
    }
    match candidate.reasoning {
        ReasoningPolicy::NonThinking => {
            if let Some(effort) = reasoning_effort {
                if !effort.is_empty() && effort != "none" {
                    return false;
                }
            }
        }
        ReasoningPolicy::Thinking => {
            if reasoning_effort.as_deref() == Some("none") {
                return false;
            }
        }
        ReasoningPolicy::Any => {}
    }
    true
}

/// Coarse reason shown to the caller when a candidate is ineligible.
/// Detailed resource values stay in the server-side log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IneligibleKind {
    OpenAiPoolExhausted,
    DeepSeekPool,
    DeepSeekFallbackBlocked,
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
    deepseek_fallback_allowed: bool,
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
            // Exhaustion-aware gate: a candidate is resource-ineligible ONLY on
            // CONFIRMED exhaustion (limit/spend-control flags or used >= 100%).
            // Missing/unknown/stale/error telemetry keeps the candidate
            // eligible — bounded lanes fail toward Luna rather than spill to a
            // fallback pool on unconfirmed signals (owner policy).
            if let Some(openai) = snapshot.openai.as_ref() {
                if openai.confirmed_exhausted() {
                    tracing::info!(
                        target = %candidate.target,
                        limit_reached = openai.limit_reached,
                        spend_control_reached = openai.spend_control_reached,
                        weekly_used_percent = openai.weekly_used_percent,
                        "openai pool CONFIRMED exhausted for candidate (fallback permitted)"
                    );
                    reasons.push((
                        candidate.target.to_string(),
                        IneligibleKind::OpenAiPoolExhausted,
                    ));
                    return false;
                }
                if !openai.available {
                    tracing::warn!(
                        target = %candidate.target,
                        error = openai.error.as_deref().unwrap_or(""),
                        "openai pool unavailable but NOT confirmed exhausted; candidate remains eligible (fail toward Luna)"
                    );
                }
            } else {
                tracing::warn!(
                    target = %candidate.target,
                    "openai pool state missing; candidate remains eligible (fail toward Luna)"
                );
            }
        }
        Pool::DeepSeek => {
            // Evaluate both failure conditions so the client-visible reasons
            // are not masked when both apply (observability: a fail-closed
            // bounded lane should report WHY it blocked, and whether the
            // fallback pool itself was unavailable).
            let deepseek_unavailable = match snapshot.deepseek.as_ref() {
                Some(deepseek) => !deepseek.eligible(),
                None => true,
            };
            if deepseek_unavailable {
                tracing::info!(
                    target = %candidate.target,
                    is_available = snapshot.deepseek.as_ref().map_or(false, |d| d.is_available),
                    total_balance = snapshot.deepseek.as_ref().map_or("", |d| d.total_balance.as_str()),
                    currency = snapshot.deepseek.as_ref().map_or("", |d| d.currency.as_str()),
                    "deepseek pool ineligible for candidate"
                );
                reasons.push((candidate.target.to_string(), IneligibleKind::DeepSeekPool));
            }
            // Fallback gate: DeepSeek may only serve this request when no
            // OpenAI candidate semantically serves it, or the OpenAI pool is
            // CONFIRMED exhausted. Unknown/over-cap/error are NOT spill
            // grounds — the bounded lane fails closed instead.
            if !deepseek_fallback_allowed {
                tracing::info!(
                    target = %candidate.target,
                    "deepseek fallback blocked: openai candidate serves request but is not confirmed exhausted"
                );
                reasons.push((
                    candidate.target.to_string(),
                    IneligibleKind::DeepSeekFallbackBlocked,
                ));
            }
            if deepseek_unavailable || !deepseek_fallback_allowed {
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
        IneligibleKind::OpenAiPoolExhausted => "openai pool exhausted (confirmed)",
        IneligibleKind::DeepSeekPool => "deepseek pool unavailable",
        IneligibleKind::DeepSeekFallbackBlocked => "deepseek fallback blocked (openai not confirmed exhausted)",
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
        // Bounded lanes are non-thinking by policy (operator correction
        // 2026-08-18): a client-attached reasoning effort must NOT exclude the
        // Luna NT candidate and silently open the DeepSeek fallback. Normalize
        // effort to none for bounded work; deliberate work is classified via
        // reasoning_intent/work_class, never via an effort flag on a bounded
        // task.
        let reasoning_effort = match contract {
            Some(SemanticContract {
                shape: WorkShape::Bounded,
                ..
            }) => None,
            _ => reasoning_effort,
        };

        let mut reasons = Vec::new();
        // Fallback gate (owner policy): DeepSeek may be selected only when no
        // OpenAI candidate in this route semantically serves the request, or
        // when the OpenAI pool is CONFIRMED exhausted. Unknown/stale/error
        // telemetry and context-cap overflow are NOT spill grounds — bounded
        // lanes fail toward Luna or fail closed.
        let route_has_openai = self
            .candidates
            .iter()
            .any(|candidate| matches!(candidate.pool, Pool::OpenAi));
        let openai_confirmed_exhausted = snapshot
            .openai
            .as_ref()
            .map(OpenAiResourceState::confirmed_exhausted);
        let openai_semantically_serves = route_has_openai
            && self.candidates.iter().any(|candidate| {
                matches!(candidate.pool, Pool::OpenAi)
                    && candidate_semantic_eligible(
                        candidate,
                        modality,
                        &reasoning_effort,
                        contract,
                    )
            });
        let deepseek_fallback_allowed =
            !openai_semantically_serves || openai_confirmed_exhausted == Some(true);
        // Observability: label a selection when DeepSeek is chosen as the
        // sanctioned fallback (route carries an OpenAI candidate that is
        // CONFIRMED exhausted). DeepSeek selections in routes without a
        // serving OpenAI candidate are designated lanes, not fallbacks.
        let confirmed_openai_exhausted = route_has_openai && openai_confirmed_exhausted == Some(true);
        for candidate in &self.candidates {
            if candidate_eligible(
                candidate,
                &snapshot,
                request_tokens,
                modality,
                &reasoning_effort,
                contract,
                deepseek_fallback_allowed,
                &mut reasons,
            ) {
                tracing::info!(
                    target = %candidate.target,
                    pool = ?candidate.pool,
                    contract = ?contract,
                    fallback_reason = if confirmed_openai_exhausted
                        && matches!(candidate.pool, Pool::DeepSeek)
                    {
                        "openai_confirmed_exhausted"
                    } else {
                        ""
                    },
                    "{} selected eligible target",
                    self.name()
                );
                // Enforce the candidate's reasoning policy on the request IR
                // BEFORE the offloaded call. The client-side wire body for a
                // same-format hop is the PRESERVED original request, so a
                // target's extra_body (or the caller's own fields) may not
                // reflect the selected candidate's policy; the candidate's
                // policy must WIN (operator invariant: no request-level field
                // may silently turn an agentic-NT contract into thinking, or
                // deliberately turn a thinking contract off).
                //
                // `reasoning_effort` is the provider-neutral knob: DeepSeek
                // honors "none" (thinking disabled) and "high" (thinking
                // enabled); the OpenAI-shaped gateway ignores the field for
                // its NT targets. The llm-client applies it to the preserved
                // wire body in send_encoded.
                let mut request = request;
                request.llm_request.reasoning.effort = match candidate.reasoning {
                    ReasoningPolicy::NonThinking => Some("none".to_string()),
                    ReasoningPolicy::Thinking => Some("high".to_string()),
                    ReasoningPolicy::Any => request.llm_request.reasoning.effort,
                };
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
    ///   bounded+deliberate -> REJECTED (contradictory; reclassify as
    ///                          deliberate/non-bounded — operator correction)
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
    async fn spend_control_reached_does_not_unlock_fallback() -> crate::Result<()> {
        // spend_control.reached is spend-control/paid-overage policy state,
        // NOT included-allowance exhaustion (operator audit 2026-08-18): it
        // must NOT unlock the DeepSeek fallback. Bounded stays on Luna.
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_spend_control(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn bounded_with_effort_high_stays_on_luna() -> crate::Result<()> {
        // A client-attached reasoning effort on a bounded contract must NOT
        // exclude Luna NT and open the DeepSeek fallback (operator correction
        // 2026-08-18): bounded work is non-thinking by policy; effort is
        // normalized to none for eligibility.
        let mut request = request_with_contract("bounded", "none", "hi");
        request.llm_request.reasoning.effort = Some("high".to_string());
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request, echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn deepseek_only_route_selects_flash_when_available() -> crate::Result<()> {
        // A route with no OpenAI candidate is a designated DeepSeek lane; the
        // fallback gate must not block it.
        let router = make_router(
            vec![candidate("testing/flash", Pool::DeepSeek)],
            snapshot(openai_healthy(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn used_percent_100_confirmed_exhaustion_selects_flash() -> crate::Result<()> {
        // used >= 100% alone (no limit/spend-control flags) is still CONFIRMED
        // exhaustion -> the single sanctioned spill path.
        let mut openai = openai_healthy();
        openai.weekly_used_percent = Some(100.0);
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai, deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/flash");
        Ok(())
    }

    #[tokio::test]
    async fn context_over_cap_fails_closed_no_deepseek_escape_hatch() -> crate::Result<()> {
        // No context-length escape hatch (owner policy): a bounded request that
        // exceeds the Luna context cap must FAIL, not spill to DeepSeek.
        let mut cand = candidate("testing/luna", Pool::OpenAi);
        cand.context_cap_tokens = Some(8); // tiny cap
        let request = Request {
            llm_request: text_request(
                Some("auto".to_string()),
                "this is a substantially longer request body used to exceed the tiny configured context cap for the luna candidate",
            ),
            raw_request: None,
            metadata: None,
        };
        let router = make_router(
            vec![cand, candidate("testing/flash", Pool::DeepSeek)],
            snapshot(openai_healthy(), deepseek_available()),
        );
        let result = test_drive(router, request, echo()).await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("no eligible target"));
        assert!(err
            .to_string()
            .contains("deepseek fallback blocked (openai not confirmed exhausted)"));
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
    async fn openai_unknown_deepseek_healthy_bounded_stays_on_luna() -> crate::Result<()> {
        // Unknown/error telemetry is NOT confirmed exhaustion: the bounded lane
        // must fail toward Luna and must NOT spill to DeepSeek (owner policy).
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_unknown(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn openai_available_false_not_exhausted_bounded_stays_on_luna() -> crate::Result<()> {
        // available=false WITHOUT limit/spend-control/used>=100 is a transient
        // or stale signal, not exhaustion: the bounded lane stays on Luna.
        let mut openai = openai_healthy();
        openai.available = false;
        openai.error = Some("transient read failure".into());
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai, deepseek_available()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
        Ok(())
    }

    #[tokio::test]
    async fn openai_state_missing_bounded_stays_on_luna() -> crate::Result<()> {
        // Missing OpenAI slot is NOT confirmed exhaustion: keep Luna eligible
        // and block the fallback (fail toward Luna, no spill).
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            ResourceSnapshot {
                openai: None,
                deepseek: Some(deepseek_available()),
            },
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
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
    async fn both_pools_unknown_bounded_stays_on_luna() -> crate::Result<()> {
        // OpenAI unknown -> not confirmed exhausted -> Luna stays eligible and
        // the fallback is blocked; DeepSeek unknown is irrelevant because Luna
        // is selected first. No spill, no spurious error.
        let router = make_router(
            vec![
                candidate("testing/luna", Pool::OpenAi),
                candidate("testing/flash", Pool::DeepSeek),
            ],
            snapshot(openai_unknown(), deepseek_unknown()),
        );
        let (trace, _) = test_drive(router, text_request_hi(), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
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
    async fn universal_bounded_openai_unknown_stays_on_luna() -> crate::Result<()> {
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_unknown(), deepseek_available()),
        );
        let (trace, _) = test_drive(router, request_with_work_class("bounded", "hi"), echo()).await?;
        assert_eq!(trace[0].selected_model_id(), "testing/luna");
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
        // Legacy work_class=reasoning translates to (agentic, deliberate) —
        // non-bounded deliberate work -> Flash thinking (operator correction
        // 2026-08-18; bounded may never carry deliberate intent).
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
    async fn contract_bounded_deliberate_rejected() -> crate::Result<()> {
        // Operator correction (2026-08-18): bounded work MUST be non-thinking.
        // A contradictory bounded+deliberate contract is rejected, never
        // auto-selected to DeepSeek thinking.
        let router = make_request_router(
            universal_candidates(),
            snapshot(openai_healthy(), deepseek_available()),
        );
        let result = test_drive(
            router,
            request_with_contract("bounded", "deliberate", "hi"),
            echo(),
        )
        .await;
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(error) => error,
        };
        assert!(err
            .to_string()
            .contains("bounded work_shape with reasoning_intent=deliberate is contradictory"));
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
