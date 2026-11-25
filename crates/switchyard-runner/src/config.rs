// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-1 TOML deployment loading for the shared runner.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use libsy::LibsyError;
use libsy::RuntimeModels;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_llm_client::{
    AuxiliaryOperation, Backend, ClientRouter, DEFAULT_MAX_RETRIES, HttpBackendConfig, ModelConfig,
    TranslatingLlmClient,
};
use switchyard_protocol::{
    Category, ModelId, ReasoningDialect, ReasoningPolicy, RoutedLlmClient, WireFormat,
};
use switchyard_translation::HostedReasoningMarker;

use crate::{
    AlgorithmSpec, AuxiliaryTarget, CallerAuthKind, DecisionTarget, HeldRoute, MediaCarrier,
    ModelCapabilities, Route, Runner, RunnerError, capability, decision_executor,
    decision_transport,
};

const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Carries a classified decision failure out of the host so the reason is not lost.
#[derive(Debug)]
struct DecisionCallError {
    requested: String,
    failure: decision_executor::DecisionFailure,
}

impl std::fmt::Display for DecisionCallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "decision call to {} failed: {:?}",
            self.requested, self.failure
        )
    }
}

impl std::error::Error for DecisionCallError {}
const MAX_CONFIGURED_RETRIES: u32 = 10;

type RunnerResult<T> = Result<T, RunnerError>;

pub(crate) fn load_runner(path: impl AsRef<Path>) -> RunnerResult<Runner> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|error| {
        RunnerError::configuration_source(
            format!("failed to read server config {}: {error}", path.display()),
            error,
        )
    })?;
    runner_from_toml(&source).map_err(|error| {
        RunnerError::configuration_source(
            format!("invalid server config {}: {error}", path.display()),
            error,
        )
    })
}

pub(crate) fn runner_from_toml(source: &str) -> RunnerResult<Runner> {
    let config: DeploymentConfig = toml::from_str(source).map_err(|error| {
        RunnerError::configuration_source(format!("failed to parse TOML: {error}"), error)
    })?;
    config.build()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeploymentConfig {
    schema_version: u32,
    fallback_client: Option<String>,
    #[serde(default)]
    llm_clients: BTreeMap<String, LlmClientConfig>,
    targets: BTreeMap<String, TargetConfig>,
    routes: BTreeMap<String, RouteConfig>,
    /// Specialized capability endpoints, by configuration name.
    #[serde(default)]
    pub(crate) capability_clients: BTreeMap<String, capability::CapabilityClientConfig>,
    /// Caller-visible capabilities, by configuration name.
    #[serde(default)]
    pub(crate) capabilities: BTreeMap<String, capability::CapabilityConfig>,
    /// Explicitly ordered decision lanes, by lane name.
    #[serde(default)]
    pub(crate) decision_lanes: BTreeMap<String, DecisionLaneConfig>,
    /// The lane that serves a request naming no lane. Declared explicitly: promoting
    /// whichever unscoped lane happened to be registered first would make a
    /// deployment's public identity depend on map ordering.
    #[serde(default)]
    pub(crate) default_decision_lane: Option<String>,
    /// Decision lanes that compose child POLICY lanes rather than physical targets.
    /// A tier over families references its families, so each keeps its own identity,
    /// stays independently addressable, and its policy change is inherited.
    #[serde(default)]
    pub(crate) decision_lane_composition: BTreeMap<String, DecisionLaneCompositionConfig>,
    /// Ordered composites over complete child routes, by route name.
    #[serde(default)]
    pub(crate) composites: BTreeMap<String, CompositeRouteConfig>,
    /// Where per-target daily request allowances keep their counters.
    ///
    /// Optional as a whole: a deployment with no `daily_request_limit` has no budget to
    /// account for. Required as soon as one target declares a limit, because an allowance
    /// whose counter cannot be written is a cap nobody is enforcing.
    #[serde(default)]
    quota: Option<QuotaConfig>,
}

/// Accounting for per-target daily request allowances.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QuotaConfig {
    /// Directory holding one counter file per (llm client, model).
    ///
    /// Created if it does not exist. Verified writable at load, so a budget that cannot be
    /// counted is a startup error rather than a silently unenforced cap.
    pub(crate) state_dir: std::path::PathBuf,
}

/// A route composed of ordered whole child routes.
///
/// Children are route names, not targets. Each child runs as a complete route with its own
/// algorithm, preparation, reasoning policy and fallback, and the parent adds only
/// ordered selection between them.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompositeRouteConfig {
    /// Child route names in preference order.
    pub(crate) children: Vec<String>,
    /// When this composite may try its next child.
    ///
    /// Defaults to the historical failure-class rule, which is right for a composite whose
    /// children sit on different credentials. A composite whose children share one
    /// credential has a different question to answer: whether the *scope* the failed child
    /// used is exhausted while the next child's scope is not. `credential_block` says that
    /// and nothing more, so an ordinary rate limit, a timeout or a provider fault cannot
    /// spend a paid call.
    #[serde(default)]
    pub(crate) cross_on: CompositeCrossPolicy,
    /// Certified context this composite guarantees, as an explicit policy promise.
    ///
    /// Stated rather than derived: a composite whose first child accepts a million tokens
    /// still cannot guarantee that size when a later child does not.
    #[serde(default)]
    pub(crate) context_window: Option<u32>,
    #[serde(default)]
    pub(crate) tool_calling: Option<bool>,
    #[serde(default)]
    pub(crate) reasoning: Option<bool>,
    #[serde(default)]
    pub(crate) vision: Option<bool>,
    /// Whether any child can accept video input. Stated, never inferred from `vision`.
    #[serde(default)]
    pub(crate) video: Option<bool>,
    /// Which video carriers any child can accept, when the composite claims video.
    ///
    /// Same reasoning as the route field: a composite resolves before its children, so a
    /// carrier it cannot back would advertise a medium and then skip every child.
    #[serde(default)]
    pub(crate) video_carrier: Option<MediaCarrier>,
    /// Which image carriers any child can accept, when the composite claims vision.
    #[serde(default)]
    pub(crate) image_carrier: Option<MediaCarrier>,
}

/// When a whole-route composite may try its next child.
///
/// The two answers differ in what they treat as evidence. `failure_class` asks whether the
/// failure looks recoverable somewhere else, which is the right question when children sit
/// on different credentials. `credential_block` asks whether the failure named a
/// credential-scoped resource as out, which is the right question when children share one
/// credential and the later child's *scope* is what differs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CompositeCrossPolicy {
    /// Cross on any failure class that says another complete route might answer.
    ///
    /// This is the historical rule and stays the default, so an existing composite's
    /// behaviour does not change because a new policy became available.
    #[default]
    FailureClass,
    /// Cross only when the failed child reported a credential-level block AND a later
    /// child on that credential still has a servable scope.
    ///
    /// A block is what the provider said about the credential, not what the request
    /// looked like: the free-model daily allowance reported exhausted is a block, while a
    /// 20-per-minute throttle, a timeout, a transport fault or a per-model refusal is not.
    /// A later child on a different credential is unaffected by any block and is judged on
    /// its own failure. A whole-credential exhaustion or a rejected key blocks every child
    /// on that credential, so no later child is called and the child's own error surfaces.
    CredentialBlock,
}

/// One explicitly ordered decision lane.
///
/// Candidate order is a routing decision, so it is stated rather than inferred. An
/// ordered list that merely happened to be sorted by name would make a deployment's
/// behaviour depend on target spelling.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionLaneConfig {
    /// Decision targets in preference order. Each must be a decision target, appear
    /// once, and exist.
    pub(crate) targets: Vec<String>,
    /// When set, this lane serves only requests of these question kinds. Different kinds
    /// can genuinely have different best orders, so the split is per type rather than one
    /// ranking applied to everything.
    #[serde(default)]
    pub(crate) types: Vec<decision_transport::QuestionKind>,
}

/// A decision lane composed of child POLICY lanes.
///
/// Children are lane names, not targets. Each child keeps its own question-kind
/// eligibility and failure policy in process, and the parent adds selection between them.
/// Keeping the reference rather than flattening it is what lets a family stay
/// independently addressable and its policy stay single-sourced.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionLaneCompositionConfig {
    /// Child decision lane names in preference order.
    pub(crate) children: Vec<String>,
    /// How this lane selects between its children.
    ///
    /// Absent means [`decision_executor::LaneSelection::Ordered`], the historic
    /// behaviour, so a manifest that does not name a policy keeps its meaning. The policy
    /// is stated here rather than inferred from the lane's name: renaming a lane must not
    /// change how it routes.
    #[serde(default)]
    pub(crate) selection: decision_executor::LaneSelection,
}

#[derive(Debug)]
struct RouteConfig {
    id: ModelId,
    context_window: Option<u32>,
    tool_calling: Option<bool>,
    reasoning: Option<bool>,
    vision: Option<bool>,
    video: Option<bool>,
    /// Which video carriers this route admits, when it claims video.
    ///
    /// A route promise, not a target fact: the target says what one endpoint can serve and
    /// this says what the route lets through. They are intersected at load, so a route that
    /// promises both carriers over an inline-only target is a configuration error rather
    /// than a request that fails at the provider.
    video_carrier: Option<MediaCarrier>,
    /// Which image carriers this route admits, when it claims vision.
    image_carrier: Option<MediaCarrier>,
    /// Reasoning policy for this route. Applies on top of the target's factual policy.
    route_reasoning_policy: Option<RouteReasoningPolicy>,
    /// Whether this route's reasoning policy is applied on a best-effort basis.
    ///
    /// Absent or `false` keeps the strict behaviour: a route policy the target cannot
    /// express is a configuration error, because a route that reads as authoritative and
    /// silently does nothing is a misconfiguration.
    ///
    /// `true` is for routes whose fallbacks are interchangeable models rather than a
    /// designed lineup. Such a fallback is left to its own native reasoning behaviour
    /// instead of being rejected, so replacing it stays a configuration-only change. The
    /// policy is still applied to every target that can carry it, so opting in never
    /// alters a target that could honour it — and it is scoped to this route, so a target
    /// shared with another route keeps that route's strictness.
    route_reasoning_policy_best_effort: bool,
    /// Estimated input tokens above which this route escalates to a wider-context leg.
    escalation_max_input_tokens: Option<u64>,
    /// Groups whose members form this route's candidate pool, when it declares one.
    pool_groups: Option<Vec<String>>,
    /// Targets this route may reach only after its entire eligible primary pool is
    /// verifiably unavailable, in declared order.
    ///
    /// Kept separate from the candidate list on purpose. A terminal leg that was merely a
    /// candidate would be ranked with its peers and would serve whenever the pool was busy,
    /// which is the paid escalation this field exists to prevent; naming it here means the
    /// exhaustion rule is the only way to reach it. A terminal target therefore must not also
    /// appear among the candidates.
    ///
    /// A flat list of names is ONE leg, which is the historical shape and still means exactly
    /// what it meant: its members are peers of each other and the route crosses onto them as a
    /// single set. A list of lists declares ordered TIERS instead, and only the nested form
    /// orders them. The distinction is not cosmetic. Peerage inside one set is decided by the
    /// route's selection policy, so two providers' last resorts written as `[a, b]` would
    /// compete and the cheaper or faster one would win regardless of the order they were
    /// typed in. Tier `n + 1` is reachable only once tier `n` has itself been proved out, and
    /// each boundary is authorized separately.
    terminal_fallback: Option<TerminalFallback>,
    /// Which evidence may spend `terminal_fallback`.
    ///
    /// Absent keeps the historical rule, so no route changes behaviour because this option
    /// exists. Declared, it is a statement about money: a lane whose terminal leg exists for
    /// one kind of failure must not be able to spend it on another. Declaring a policy with
    /// no terminal leg is refused rather than ignored, because a policy that governs nothing
    /// reads as a guarantee it cannot provide.
    terminal_policy: Option<TerminalPolicy>,
    /// Whether a request no primary candidate can hold may be escalated to the declared
    /// tiers.
    ///
    /// Absent is the historical behaviour and stays the default: the pool admitted nothing,
    /// the caller is told what did not fit, and no further leg is contacted. Declared, it is
    /// a second and narrower claim beside `terminal_policy` -- not a wider reading of it.
    ///
    /// The two are deliberately separate axes. `terminal_policy` decides what may spend the
    /// leg when the pool *failed*: an allowance gone, a throttle, an outage. This decides
    /// whether a pool that never dispatched anything may hand the request to the next tier
    /// because the request is larger than the pool can hold. Reading a size refusal as
    /// capacity exhaustion would let a metered tier absorb a request the deployment never
    /// promised to serve and would hide from the caller that their prompt was the problem;
    /// refusing to escalate at all strands a request an authorized wider tier can serve.
    ///
    /// It cannot widen what any tier accepts: each tier is judged on its own gates, and a
    /// request that fails them ends the walk with the refusal the primary pool gave.
    fit_escalation: Option<bool>,
    /// Whether a request no primary candidate's *wire* can carry may be escalated to the
    /// declared tiers.
    ///
    /// Absent is the historical behaviour and stays the default: a pool whose targets cannot
    /// carry the request refuses, and the caller is told what the deployment lacks rather than
    /// having a later group asked the same question. Declared, it is a third and narrower claim
    /// beside `terminal_policy` and `fit_escalation`, and the three are deliberately separate
    /// axes: capacity, size, and the wire.
    ///
    /// It authorizes consideration of the next declared group, not expenditure. Every group is
    /// judged on its own credential, allowance, fit, privacy and continuation gates, so a
    /// metered group reached this way still has to admit the request on its own terms, or it is
    /// skipped exactly as before.
    ///
    /// Only a closed vocabulary of measured per-target wire facts may open this boundary
    /// (`tool_call_replay` today). A malformed or unsafe continuation, an authentication,
    /// billing or policy refusal, and a request the route never promised to serve all fail
    /// outside it and end the walk with the pool's own refusal.
    capability_escalation: Option<bool>,
    /// Target this route prefers once it is under context pressure.
    ///
    /// Named rather than inferred: a target's context window is not a fact this config
    /// carries. `context_window` is a route-level promise, and upstream rejects it as an
    /// unknown field on a target, so which leg is wider has to be stated.
    escalation_target: Option<String>,
    /// How this route orders the candidates that survived eligibility filtering.
    ///
    /// Declared order is the historical behaviour and stays the default, so no existing
    /// route reorders because a new option became available. `measured` lets a route whose
    /// candidates are interchangeable free models pick by what this deployment has actually
    /// observed — completion, then latency — instead of by configuration order, which for
    /// interchangeable models is an accident of spelling rather than a decision.
    ///
    /// Read by the manual `Deserialize` below, which is why it carries no attribute: this
    /// struct removes its own fields from the table before the algorithm spec sees them.
    selection: SelectionPolicy,
    algorithm: AlgorithmSpec,
}

/// How a route orders its eligible candidates.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    /// Walk candidates in the order configuration states. Unchanged behaviour.
    #[default]
    Declared,
    /// Order by measured completion and latency, falling back to declared order.
    ///
    /// Only reorders among candidates that already survived every hard gate — privacy,
    /// capability, physical fit, quota — so it can never spend a leg the deployment would
    /// have refused. A target with too few observations keeps its declared position, so a
    /// cold deployment behaves exactly like a declared-order one.
    Measured,
    /// Treat every candidate as a peer, regardless of which provider or credential serves it.
    ///
    /// `Measured` still breaks ties with the manifest's own order, which is the right answer
    /// for a lineup an operator wrote down deliberately. It is the wrong answer for a pool
    /// assembled from several providers' free tiers, because the manifest order is then an
    /// accident of how the file was written and would silently become a provider priority:
    /// the leg typed first would win every tie, every time.
    ///
    /// `Peer` removes that priority in two places. Ties are broken by canonical route id, so
    /// reordering the manifest cannot change the chosen model; and a candidate whose own
    /// remaining allowance is unknown ranks behind one whose headroom is verifiable, instead
    /// of ahead of every declared budget as an absent number used to. Free eligibility is
    /// still decided by declared tariff, never by position: a `Peer` pool excludes every
    /// candidate that is not stated free before it ranks anything.
    Peer,
}

/// One declared terminal leg: either a single target or one tier of peers.
///
/// Untagged so the manifest stays readable. This is the INNER shape — what one tier is —
/// and it is only ever reached through [`TerminalFallback::Tiered`], so seeing `One` here
/// never means "a separate tier", only "a tier written without brackets".
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TerminalTierSpec {
    /// A tier of one target.
    One(String),
    /// One tier whose members are peers of each other.
    Many(Vec<String>),
}

impl TerminalTierSpec {
    /// The target names in this tier, in declared order.
    pub fn names(&self) -> Vec<&str> {
        match self {
            Self::One(name) => vec![name.as_str()],
            Self::Many(names) => names.iter().map(String::as_str).collect(),
        }
    }
}

/// A route's declared terminal legs, and whether their order means anything.
///
/// Untagged so that every manifest written before tiers existed keeps its meaning exactly.
/// The distinction is not cosmetic and it is not inferable from the inner type: if this
/// enum did not exist, `["a", "b"]` would deserialise as *two* tiers of one target each, and
/// a route that declares two peers would silently become an escalation from `a` to `b`.
/// Peerage inside one set is decided by the route's own selection policy; only the nested
/// form orders legs, and only the nested form makes a leg wait for its predecessor.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TerminalFallback {
    /// The historical shape: one leg, whose members are peers of one another.
    Flat(Vec<String>),
    /// Declared escalation order: a tier is reachable only once its predecessor is proved out.
    Tiered(Vec<TerminalTierSpec>),
}

impl TerminalFallback {
    /// The declaration as ordered tiers, flattening the historical one-leg shape into a
    /// single tier so every reader sees one representation.
    fn tiers(&self) -> Vec<TerminalTierSpec> {
        match self {
            Self::Flat(names) => vec![TerminalTierSpec::Many(names.clone())],
            Self::Tiered(tiers) => tiers.clone(),
        }
    }

    /// Every declared target name, in declaration order, borrowed from the declaration.
    ///
    /// Separate from [`Self::tiers`] because the target lists here are handed to builders
    /// that store `&str`: those borrows have to point into the configuration, not into a
    /// temporary that `tiers` allocated.
    fn name_refs(&self) -> Vec<&str> {
        match self {
            Self::Flat(names) => names.iter().map(String::as_str).collect(),
            Self::Tiered(tiers) => tiers.iter().flat_map(TerminalTierSpec::names).collect(),
        }
    }
}

/// Which evidence may spend a route's declared terminal fallback.
///
/// A terminal leg exists so that one specific kind of failure does not become the caller's
/// problem. Which kind is a policy decision, and it is not the same decision for every lane:
/// a provider-neutral pool escalates on "nothing in this pool can answer right now", while a
/// lane built to work inside a free allowance escalates only when that allowance is spent.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalPolicy {
    /// Any capacity or availability failure of the whole pool, or a pool wholly covered by a
    /// credential block. The historical rule; every route that declares no policy keeps it.
    #[default]
    Capacity,
    /// Quota evidence only: the terminal leg is spent when the primary pool is demonstrably
    /// out of allowance or throttled, and never on an outage, a timeout, a transport fault or
    /// a credential/permission failure. Those say the deployment cannot reach the pool, not
    /// that the pool's free allowance ran out, and the terminal leg is reserved for the
    /// second case.
    QuotaOnly,
    /// Capacity or availability failure of the whole pool, but never a credential refusal and
    /// never a fault that is about this deployment's accounting or this request's size.
    ///
    /// The difference from `Capacity` is the two authentication outcomes, plus
    /// `daily_allowance_unavailable` and `minute_window_too_small`. A rejected key or a
    /// refused balance says the deployment is misconfigured or unpaid, and spending a
    /// further leg on it converts a configuration fault into cost without making the request
    /// any more likely to be answered. An unreadable allowance ledger is an accounting fault
    /// rather than a measured exhaustion -- the allowance may be untouched -- and a request
    /// larger than a model's whole minute window can never be served by that model. Every
    /// other capacity class is accepted -- allowance exhaustion, a minute window, a retryable
    /// 5xx or 429, a transport fault, a leg that went unavailable, or an attempt that timed
    /// out -- because those are the ways a healthy deployment legitimately cannot serve right
    /// now.
    Availability,
}

/// Route reasoning policy: what this ROUTE asks of the provider, independent of what the
/// target is capable of.
///
/// Two routes may deliberately share one physical model while one forbids reasoning, so
/// this cannot live on the target. Target reasoning stays the factual capability; this is
/// the policy applied on top of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteReasoningPolicy {
    /// Use the target's own declared policy.
    Inherit,
    /// Thinking on. Where the provider exposes no graded effort axis this means thinking
    /// enabled, not a fabricated intermediate level.
    Medium,
    /// The strongest level the provider genuinely supports.
    ///
    /// Distinct from `Medium` because a route policy is applied last and therefore wins over
    /// the target default. An unbounded primitive that must reach a stronger level than a
    /// direct address of the same physical model needs to say so; inheriting would hand it
    /// the target's level, which is the point of the two being separate decisions.
    High,
    /// Thinking off.
    None,
}

/// A route policy's effect on the wire, which is not always expressible as on/off.
///
/// A dialect that only grades effort (`openai_effort`) has no fragment meaning "think", so an
/// on/off policy is inert there and the level has to arrive as an effort string instead. This
/// records the effort a route asks for when its own policy would otherwise be unrepresentable,
/// so the client can write it rather than silently sending the target's weaker default.
pub(crate) fn route_policy_effort(policy: RouteReasoningPolicy) -> Option<&'static str> {
    match policy {
        RouteReasoningPolicy::High => Some("high"),
        _ => None,
    }
}

struct TargetPromptPolicy {
    prompts: HashMap<ModelId, String>,
    routing_answer_target: Option<ModelId>,
}

impl<'de> Deserialize<'de> for RouteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut table = toml::Table::deserialize(deserializer)?;
        let id = take_required(&mut table, "id")?;
        let context_window = take_optional(&mut table, "context_window")?;
        let tool_calling = take_optional(&mut table, "tool_calling")?;
        let reasoning = take_optional(&mut table, "reasoning")?;
        let vision = take_optional(&mut table, "vision")?;
        let video = take_optional(&mut table, "video")?;
        let video_carrier = take_optional::<MediaCarrier, _>(&mut table, "video_carrier")?;
        let image_carrier = take_optional::<MediaCarrier, _>(&mut table, "image_carrier")?;
        let route_reasoning_policy = take_optional(&mut table, "route_reasoning_policy")?;
        let route_reasoning_policy_best_effort: Option<bool> =
            take_optional(&mut table, "route_reasoning_policy_best_effort")?;
        let escalation_max_input_tokens = take_optional(&mut table, "escalation_max_input_tokens")?;
        let pool_groups = take_optional(&mut table, "pool_groups")?;
        let terminal_fallback = take_optional(&mut table, "terminal_fallback")?;
        let terminal_policy = take_optional(&mut table, "terminal_policy")?;
        let fit_escalation = take_optional(&mut table, "fit_escalation")?;
        let capability_escalation = take_optional(&mut table, "capability_escalation")?;
        let escalation_target = take_optional(&mut table, "escalation_target")?;
        let selection: Option<SelectionPolicy> = take_optional(&mut table, "selection")?;
        let algorithm = AlgorithmSpec::deserialize(toml::Value::Table(table))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            id,
            context_window,
            tool_calling,
            reasoning,
            vision,
            video,
            video_carrier,
            image_carrier,
            route_reasoning_policy,
            route_reasoning_policy_best_effort: route_reasoning_policy_best_effort.unwrap_or(false),
            escalation_max_input_tokens,
            pool_groups,
            terminal_fallback,
            terminal_policy,
            fit_escalation,
            capability_escalation,
            escalation_target,
            selection: selection.unwrap_or_default(),
            algorithm,
        })
    }
}

fn take_required<T, E>(table: &mut toml::Table, name: &'static str) -> Result<T, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    let value = table.remove(name).ok_or_else(|| E::missing_field(name))?;
    T::deserialize(value).map_err(E::custom)
}

fn take_optional<T, E>(table: &mut toml::Table, name: &'static str) -> Result<Option<T>, E>
where
    T: DeserializeOwned,
    E: serde::de::Error,
{
    table
        .remove(name)
        .map(|value| T::deserialize(value).map_err(E::custom))
        .transpose()
}

impl RouteConfig {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            context_window: self.context_window,
            tool_calling: self.tool_calling,
            reasoning: self.reasoning,
            vision: self.vision,
            video: self.video,
            // A route that claims a medium without naming carriers means every carrier, so
            // a caller reading `/v1/models` and a request carrying either form agree.
            video_carrier: (self.video == Some(true))
                .then(|| self.video_carrier.unwrap_or(MediaCarrier::DEFAULT)),
            image_carrier: (self.vision == Some(true))
                .then(|| self.image_carrier.unwrap_or(MediaCarrier::DEFAULT)),
            // Deliberately undeclared at route level: whether a replayed tool call can be
            // accepted is a property of one provider's wire, not a promise a route can make
            // across children with different wires. Candidate filtering reads each target's
            // fact instead, so this stays `None` rather than becoming a knob that does nothing.
            tool_call_replay: None,
        }
    }

    fn routing_target_names(&self) -> Vec<&str> {
        self.algorithm.routing_target_names()
    }

    fn callable_target_names(&self) -> Vec<&str> {
        self.algorithm.callable_target_names()
    }

    /// Every target this route may contact: the ranked candidates plus the terminal last
    /// resort. Facts, credentials, daily ledgers and prompt policy have to describe a
    /// target the route can actually call, or the terminal leg is reached without the
    /// admission and block machinery that protects every ranked leg.
    fn served_target_names(&self) -> Vec<&str> {
        let mut names = self.routing_target_names();
        if let Some(terminal) = &self.terminal_fallback {
            for name in terminal.name_refs() {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
        names
    }
}

/// The two ledgers a resolved deployment shares, owned together.
///
/// Kept as one value because the pair is not optional in either direction: a resolver
/// built with a governor but no counters would pace its calls without counting them,
/// which is the defect this whole path exists to prevent. Passing them as one value
/// makes that omission unrepresentable rather than merely discouraged.
struct Accounting<'a> {
    governor: &'a Arc<switchyard_llm_client::QuotaGovernor>,
    allowances: &'a HashMap<(String, ModelId), Arc<switchyard_llm_client::DailyRequestAllowance>>,
}

impl DeploymentConfig {
    /// The media capabilities a composite actually guarantees.
    ///
    /// A composite can serve a request only through one of its children, so what it can
    /// promise is what the children between them accept. When the parent states a medium or
    /// a carrier it keeps its word (an explicit `false` is a policy cap, not an omission),
    /// and when it states nothing the answer is derived from the children:
    ///
    /// * a medium is claimed only when some child declares it -- an undeclared child is not
    ///   evidence that it reads images, so derivation never invents a claim;
    /// * a carrier is the union over the children that admit the medium, so a composite over
    ///   inline-only children advertises inline and a mixed set advertises both.
    ///
    /// Deriving rather than hand-editing every parent is what keeps the declaration honest
    /// when a child's carrier narrows: the parent follows, and
    /// `validate_composite_media_claims` still refuses an explicit claim no child keeps.
    fn composite_capabilities(&self, composite: &CompositeRouteConfig) -> crate::ModelCapabilities {
        let children: Vec<crate::ModelCapabilities> = composite
            .children
            .iter()
            .filter_map(|child| self.routes.get(child))
            .map(RouteConfig::capabilities)
            .collect();
        let serves = |medium: fn(&crate::ModelCapabilities) -> Option<bool>| {
            children.iter().any(|caps| medium(caps) == Some(true))
        };
        let carrier_union =
            |medium: fn(&crate::ModelCapabilities) -> Option<bool>,
             carrier: fn(&crate::ModelCapabilities) -> Option<MediaCarrier>| {
                children
                    .iter()
                    .filter(|caps| medium(caps) == Some(true))
                    .map(|caps| carrier(caps).unwrap_or(MediaCarrier::DEFAULT))
                    .reduce(MediaCarrier::union)
            };
        let video = composite.video.unwrap_or_else(|| serves(|caps| caps.video));
        let vision = composite
            .vision
            .unwrap_or_else(|| serves(|caps| caps.vision));
        crate::ModelCapabilities {
            context_window: composite.context_window,
            tool_calling: composite.tool_calling,
            reasoning: composite.reasoning,
            vision: Some(vision),
            video: Some(video),
            // A stated carrier is a policy narrowing; otherwise the union of the children's
            // carriers, never the permissive default, so the parent cannot advertise a
            // carrier nothing below it serves.
            video_carrier: video.then(|| {
                composite
                    .video_carrier
                    .or_else(|| carrier_union(|caps| caps.video, |caps| caps.video_carrier))
                    .unwrap_or(MediaCarrier::DEFAULT)
            }),
            image_carrier: vision.then(|| {
                composite
                    .image_carrier
                    .or_else(|| carrier_union(|caps| caps.vision, |caps| caps.image_carrier))
                    .unwrap_or(MediaCarrier::DEFAULT)
            }),
            // A composite resolves before its children and each child filters its own
            // candidates on this axis, so the parent states nothing and cannot exclude a
            // child that could still serve the continuation.
            tool_call_replay: None,
        }
    }

    fn decision_target(&self, name: &str) -> Option<DecisionTarget> {
        let target = self.targets.get(name)?;
        let client = self.llm_clients.get(&target.llm_client)?;
        Some(DecisionTarget {
            target: name.to_string(),
            model: target.id.clone(),
            format: client.format.wire_format(),
            base_url: client.base_url.as_str().to_string(),
            extra_body: target.extra_body.clone(),
            decision_transport: target.decision_transport,
            supported_types: target.supported_types,
            state_forms: target.state_forms,
            decision_path: target.decision_path.clone(),
            decision_base_url: target.decision_base_url.clone(),
            decision_api_key_env: target.decision_api_key_env.clone(),
            target_capabilities: crate::ModelCapabilities {
                vision: target.vision,
                tool_calling: target.tool_calling,
                ..crate::ModelCapabilities::default()
            },
        })
    }

    /// Extracts the deployment facts an economics surface needs.
    ///
    /// These are facts, not inferences: a client's base URL, a target's model id and
    /// client binding, and a route's primary target and candidate list. The economics
    /// layer derives the provider, route family and cost class from them, so a
    /// dashboard's attribution can be traced back to the configuration.
    fn deployment_facts(&self) -> crate::facts::DeploymentFacts {
        let mut facts = crate::facts::DeploymentFacts::default();
        for (name, client) in &self.llm_clients {
            facts.clients.insert(
                name.clone(),
                crate::facts::ClientFacts {
                    name: name.clone(),
                    base_url: client.base_url.as_str().to_string(),
                    api_key_env: client.api_key_env.clone(),
                    billing: client.billing,
                    quota_group: Some(client.quota_group.clone().unwrap_or_else(|| name.clone())),
                    quota_provider: client.quota_provider,
                },
            );
        }
        for (name, target) in &self.targets {
            facts.targets.insert(
                name.clone(),
                crate::facts::TargetFacts {
                    name: name.clone(),
                    model: target.id.as_str().to_string(),
                    llm_client: target.llm_client.clone(),
                },
            );
        }
        for (name, config) in &self.routes {
            let (target, has_candidates) = match &config.algorithm {
                AlgorithmSpec::Passthrough {
                    target, candidates, ..
                } => (Some(target.clone()), !candidates.is_empty()),
                _ => (None, false),
            };
            facts.routes.insert(
                name.clone(),
                crate::facts::RouteFacts {
                    id: config.id.as_str().to_string(),
                    target,
                    has_candidates,
                },
            );
        }
        for (name, composite) in &self.composites {
            facts
                .composites
                .insert(name.clone(), composite.children.clone());
        }
        facts
    }

    fn build(self) -> RunnerResult<Runner> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(RunnerError::configuration(format!(
                "unsupported schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
                self.schema_version
            )));
        }

        let facts = self.deployment_facts();
        // Resolved once, before any route is built, so a gated deployment that cannot

        let mut route_names_by_id = HashMap::new();
        for (route_name, config) in &self.routes {
            validate_value("route name", route_name)?;
            validate_value(&format!("route {route_name} id"), &config.id)?;
            if let Some(first_route_name) =
                route_names_by_id.insert(config.id.as_str(), route_name.as_str())
            {
                return Err(RunnerError::configuration(format!(
                    "routes {first_route_name} and {route_name} both use id {}; route ids must be unique",
                    config.id
                )));
            }
        }

        // The LLM client keeps one backend per model id, so two targets naming the same model on
        // the same client share it. That is harmless when their request settings agree (an alias
        // for a different system prompt, say) and silently wrong when they do not: the second
        // target's reasoning_effort or extra_body would never reach the wire.
        let mut seen_client_model_ids: HashMap<(&str, &str), (&String, &TargetConfig)> =
            HashMap::new();
        for (target_name, target) in &self.targets {
            validate_value("target name", target_name)?;
            validate_value(&format!("target {target_name} id"), &target.id)?;
            match seen_client_model_ids.entry((target.llm_client.as_str(), target.id.as_str())) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert((target_name, target));
                }
                std::collections::hash_map::Entry::Occupied(slot) => {
                    let (first_name, first) = slot.get();
                    if first.reasoning_effort != target.reasoning_effort
                        || first.extra_body != target.extra_body
                        || first.omit_body_fields != target.omit_body_fields
                        || first.unsupported_reasoning_effort != target.unsupported_reasoning_effort
                        || first.reasoning_policy != target.reasoning_policy
                        || first.reasoning_dialect != target.reasoning_dialect
                    {
                        return Err(RunnerError::configuration(format!(
                            "targets {first_name} and {target_name} both name model {} on llm client {} but with different reasoning_effort, reasoning_policy, reasoning_dialect, extra_body, omit_body_fields, or unsupported_reasoning_effort; one target per model id is kept, so give each its own model id or llm client",
                            target.id, target.llm_client
                        )));
                    }
                    tracing::warn!(
                        "target {target_name} reuses model id {} on llm client {}; only one target per id is kept and the other is dropped. Give each target a unique model id, or point both routes at one target.",
                        target.id,
                        target.llm_client
                    );
                }
            }
        }

        let mut provider_api_keys = Vec::new();
        // Built once and shared: the dispatch gate reserves against these counters and
        // selection reads the same remainder, so one budget has one source of truth.
        let allowances = self.build_allowances()?;
        self.validate_minute_windows()?;
        self.validate_quota_providers()?;
        // One governor for the whole deployment. Its key is (quota group, physical model),
        // so every alias of one model draws on one window — which is what the provider
        // counts, and the only way `smart-nt`, `smart_gemini` and a direct leaf cannot each
        // spend the same minute's allowance.
        let governor = Arc::new(switchyard_llm_client::QuotaGovernor::new());
        let accounting = Accounting {
            governor: &governor,
            allowances: &allowances,
        };
        let clients = self.build_clients(&mut provider_api_keys, &allowances, &governor)?;
        capability::CapabilityDeployment {
            clients: self.capability_clients.clone(),
            capabilities: self.capabilities.clone(),
        }
        .validate()
        .map_err(RunnerError::configuration)?;
        let targets = self.build_targets();
        let fallback_base_url = self.fallback_base_url()?;
        self.validate_composites()?;
        self.validate_route_capability_claims()?;
        self.validate_composite_media_claims()?;
        self.validate_physical_context_claims()?;
        self.validate_video_transport()?;
        self.validate_responses_wire_targets()?;
        self.validate_decision_lanes()?;
        self.validate_default_decision_lane()?;
        self.validate_decision_lane_composition()?;
        // Lane name -> resolver, with the kinds each lane serves. Built before the routes
        // so a route's decision handler can select by the request's question kinds.
        let mut lane_index: Vec<(
            String,
            Arc<decision_executor::DecisionResolver>,
            Vec<decision_transport::QuestionKind>,
        )> = Vec::new();
        let mut lane_handles: BTreeMap<String, decision_executor::LaneResolverHandle> =
            BTreeMap::new();
        for (name, lane_config) in &self.decision_lanes {
            let names: Vec<&str> = lane_config.targets.iter().map(String::as_str).collect();
            let Some(resolver) = self.build_decision_resolver(&names, &accounting) else {
                continue;
            };
            let handle = decision_executor::LaneResolverHandle {
                resolver: Arc::new(resolver),
            };
            lane_handles.insert(name.clone(), handle.clone());
            lane_index.push((name.clone(), handle.resolver, lane_config.types.clone()));
        }
        let lane_composition = self.resolve_decision_lane_composition(&lane_handles)?;
        let lane_identity = self.default_decision_lane.clone();
        // One evidence set for the whole deployment, shared by every measured route. A
        // physical model behaves the same whichever alias named it, so a request served by
        // one route is evidence for the others — and a subsidiary route cannot accumulate
        // observations nobody reads.
        let measured_selector = crate::selection::MeasuredSelector::new();
        let mut routes = Vec::with_capacity(self.routes.len());
        for (route_name, config) in &self.routes {
            // Before the unknown-target sweep below, so a malformed candidate list is
            // reported as itself rather than as a target that does not exist.
            config
                .algorithm
                .validate_candidates(route_name)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            for target_name in config.callable_target_names() {
                self.targets.get(target_name).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "route references unknown target {target_name}"
                    ))
                })?;
            }
            let capabilities = config.capabilities();
            let mut escalation_model: Option<ModelId> = None;
            if capabilities.context_window == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} context_window must be greater than zero"
                )));
            }
            // A route policy the target cannot carry is a route that reads as authoritative and
            // is not. Only the graded `reasoning_effort` axis and the target's own dialect
            // can carry it: a target that declares `reasoning_effort` expresses "medium" on
            // its own, so requiring a dialect there would reject the shipped manifest. What
            // must not pass is an intent with neither carrier, or "thinking on" on a dialect
            // whose only off value is `none`.
            if let Some(policy) = config.route_reasoning_policy {
                let requested = match policy {
                    crate::config::RouteReasoningPolicy::Inherit => None,
                    crate::config::RouteReasoningPolicy::Medium => {
                        Some(switchyard_protocol::ReasoningPolicy::Enabled)
                    }
                    // `High` also carries a graded effort, so it is expressible on every
                    // dialect rather than only the boolean ones.
                    crate::config::RouteReasoningPolicy::High => {
                        Some(switchyard_protocol::ReasoningPolicy::Enabled)
                    }
                    crate::config::RouteReasoningPolicy::None => {
                        Some(switchyard_protocol::ReasoningPolicy::Disabled)
                    }
                };
                if let Some(requested) = requested {
                    for target_name in config.callable_target_names() {
                        let Some(target) = self.targets.get(target_name) else {
                            continue;
                        };
                        if target.reasoning_effort.is_some() {
                            continue;
                        }
                        if let Err(error) = requested.validate_for(target.reasoning_dialect) {
                            if config.route_reasoning_policy_best_effort {
                                // This route declared its policy best-effort, so a target
                                // that cannot carry it keeps its own native reasoning
                                // behaviour instead of being rejected. The policy is still
                                // applied to every target that can express it, and the
                                // request-time path already omits the fragment for a
                                // dialect with no representation, so nothing unsupported is
                                // sent to this one.
                                tracing::debug!(
                                    route = %route_name,
                                    target = %target_name,
                                    "route reasoning policy not expressible by target; \
                                     leaving it to its native behaviour"
                                );
                                continue;
                            }
                            return Err(RunnerError::configuration(format!(
                                "route {route_name} reasoning policy cannot be applied \
                                 to target {target_name}: {error}"
                            )));
                        }
                    }
                }
            }
            // An escalation leg that does not exist, or a threshold with no leg to move to,
            // would both silently serve the request on the primary leg instead. Reject both.
            if config.escalation_target.is_some() != config.escalation_max_input_tokens.is_some() {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} must declare escalation_max_input_tokens and \
                     escalation_target together"
                )));
            }
            // Zero would mean "every request", since the estimator floors at the structural
            // overhead. That silently removes the declared primary, so it is refused rather
            // than interpreted.
            if config.escalation_max_input_tokens == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} escalation_max_input_tokens must be greater than zero"
                )));
            }
            if let Some(escalation_target) = &config.escalation_target {
                let target = self.targets.get(escalation_target).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "route {route_name} escalation_target references unknown target \
                         {escalation_target}"
                    ))
                })?;
                // The leg must be one this route can actually serve. `promoting` is a no-op
                // for an id the route does not offer, so accepting it would leave a route
                // that reads as escalating, reports promotions, and never reorders.
                if !config
                    .callable_target_names()
                    .contains(&escalation_target.as_str())
                {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} escalation_target {escalation_target} is not a \
                         candidate of this route"
                    )));
                }
                escalation_model = Some(target.id.clone());
            }
            // Ordered-candidate validation runs before anything resolves targets, so a malformed
            // list is reported as itself rather than as an unknown target.
            config
                .algorithm
                .validate_candidates(route_name)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            // A pool route states its membership by group label, so the two records of the
            // pool - the labels on the targets and the candidate list on the route - have to
            // agree. Disagreeing would either silently narrow the pool (a member forgotten in
            // the list) or advertise a member the route cannot call (a stale entry). Both are
            // startup errors, because a pool whose membership is unclear is not a policy.
            if let Some(groups) = &config.pool_groups {
                self.validate_route_pool(route_name, config, groups)?;
            }
            // A terminal leg is a last resort, so its declaration has to be unambiguous: it
            // must name real targets, none of them a candidate of this route, and none of
            // them held. A terminal target that was also a candidate would be ranked as a
            // peer and could serve without exhaustion; one that does not exist would make
            // the route silently unwilling to use its last resort at startup.
            let terminal_ids = self.validate_terminal_fallback(route_name, config)?;
            let algorithm = config
                .algorithm
                .build(route_name, &targets)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let lane_composition: BTreeMap<String, decision_executor::ComposedLane> = self
                .decision_lane_composition
                .iter()
                .map(|(name, composition)| {
                    (
                        name.clone(),
                        decision_executor::ComposedLane {
                            children: composition.children.clone(),
                            selection: composition.selection,
                        },
                    )
                })
                .collect();
            let (route_clients, caller_auth) = self.build_route_clients(
                route_name,
                config,
                &clients,
                lane_identity.clone(),
                &lane_index,
                &lane_composition,
                &accounting,
            )?;
            let anthropic_auxiliary_target =
                self.build_anthropic_auxiliary_target(config, &clients);
            let responses_auxiliary_target =
                self.build_responses_auxiliary_target(config, &clients);
            let decision_targets = config
                .served_target_names()
                .into_iter()
                .filter_map(|name| self.decision_target(name))
                .collect();
            let names = config
                .algorithm
                .runtime_model_names(route_name)
                .map_err(|error| RunnerError::configuration_source(error.to_string(), error))?;
            let mut models = RuntimeModels::new(resolve_category_models(names.parent, &targets)?);
            if let Some(subagent) = names.subagent {
                models = models.with_subagent(resolve_category_models(subagent, &targets)?);
            }
            // Held targets keep their declaration, credential and facts, but they leave the
            // candidate set here — before any algorithm is handed it — so no selection rule
            // and no fallback walk can reach one, and no request spends a provider call on a
            // model the provider has withdrawn. Filtering the model set rather than each
            // algorithm's configuration is what makes this true for every algorithm: the
            // dispatch path reads candidates from here and nowhere else.
            let held_models: BTreeSet<ModelId> = config
                .callable_target_names()
                .into_iter()
                .filter_map(|name| self.targets.get(name))
                .filter(|target| target.held.is_some())
                .map(|target| target.id.clone())
                .collect();
            if !held_models.is_empty() {
                models = models.filtered(&|model| !held_models.contains(model));
            }
            let held = self.route_held(config, &held_models);
            // The terminal leg is handed to the route as its own model sets, one per declared
            // tier, never merged with the candidates, so the ranking machinery cannot see it at
            // all and tier order cannot be re-decided by a cost or latency score.
            let terminal = (!terminal_ids.is_empty()).then(|| {
                terminal_ids
                    .iter()
                    .map(|tier| {
                        RuntimeModels::new(HashMap::from([(
                            switchyard_protocol::Category::Any,
                            tier.clone(),
                        )]))
                    })
                    .collect::<Vec<_>>()
            });
            let route = Route::new(
                algorithm,
                route_clients,
                caller_auth,
                capabilities,
                anthropic_auxiliary_target,
                responses_auxiliary_target,
                decision_targets,
                models,
                config
                    .route_reasoning_policy
                    .unwrap_or(crate::config::RouteReasoningPolicy::Inherit),
                config.escalation_max_input_tokens,
                escalation_model,
            )
            .with_target_credentials(target_credentials(config, &self.targets))
            .with_locality(route_locality(config, &self.targets, &self.llm_clients))
            .with_daily_allowances(Arc::new(
                config
                    .served_target_names()
                    .into_iter()
                    .filter_map(|name| self.targets.get(name))
                    .filter_map(|target| {
                        let group = self
                            .llm_clients
                            .get(&target.llm_client)
                            .and_then(|client| client.quota_group.clone())
                            .unwrap_or_else(|| target.llm_client.clone());
                        // Look the counter up under the scope it was built with: account
                        // scope has one counter for the whole group, so every model in the
                        // group resolves to that one allowance rather than to a per-model
                        // counter that was never created.
                        let counter_model = match self
                            .llm_clients
                            .get(&target.llm_client)
                            .and_then(|client| client.quota_scope)
                            .unwrap_or_default()
                        {
                            QuotaScope::Account => {
                                ModelId::from(switchyard_llm_client::ACCOUNT_SCOPE_MODEL)
                            }
                            QuotaScope::Model => target.id.clone(),
                        };
                        allowances
                            .get(&(group, counter_model))
                            .map(|allowance| (target.id.clone(), Arc::clone(allowance)))
                    })
                    .collect::<HashMap<ModelId, Arc<switchyard_llm_client::DailyRequestAllowance>>>(
                    ),
            ));
            let route = match held {
                Some(held) => route.with_held(held),
                None => route,
            };
            let route = match terminal {
                Some(terminal) => route
                    .with_terminal_fallback(terminal)
                    .with_terminal_policy(config.terminal_policy.unwrap_or_default())
                    .with_fit_escalation(config.fit_escalation.unwrap_or(false))
                    .with_capability_escalation(config.capability_escalation.unwrap_or(false)),
                None => route,
            };
            // A measured route ranks candidates only after it has evidence, and shares that
            // evidence with every other measured route so real work is not isolated to one
            // alias.
            let route = match config.selection {
                crate::config::SelectionPolicy::Measured => {
                    route.with_measured_selection(Arc::clone(&measured_selector))
                }
                // A peer pool needs the evidence of a measured route: the ranking is
                // computed from facts and observation, so without the selector it would
                // fall back to declared order and quietly become the waterfall this policy
                // exists to remove.
                crate::config::SelectionPolicy::Peer => route
                    .with_measured_selection(Arc::clone(&measured_selector))
                    .with_pool_groups(config.pool_groups.clone().unwrap_or_default()),
                crate::config::SelectionPolicy::Declared => route,
            };
            routes.push((config.id.clone(), route));
        }
        // The default logical decision lane. Every configured decision target in
        // configuration order, so a caller asking for a decision without naming a
        // backend gets per-request eligibility and ordered fallback instead of having
        // to choose a provider for resilience. Shared by every route and by the
        // compatibility surface, so Decision has one implementation.

        let mut runner = Runner::new(routes)
            .with_daily_allowance_registry(allowances.clone())
            .with_fallback_url(fallback_base_url)
            .with_provider_api_keys(provider_api_keys)
            .with_deployment_facts(facts)
            .with_capabilities(
                capability::resolve_with_budget(
                    &capability::CapabilityDeployment {
                        clients: self.capability_clients.clone(),
                        capabilities: self.capabilities.clone(),
                    },
                    Some(&governor),
                    self.quota.as_ref().map(|quota| quota.state_dir.as_path()),
                )
                .map_err(RunnerError::configuration)?,
            );
        self.validate_decision_lanes()?;
        // Composites are registered as named routes, so lookup is uniform: a caller
        // resolving a route cannot tell whether it is a leaf or a composite.
        for (name, composite) in &self.composites {
            runner = runner.with_composite(name.clone(), composite.children.clone());
            runner = runner.with_composite_cross_policy(name.clone(), composite.cross_on);
            runner = runner
                .with_composite_capabilities(name.clone(), self.composite_capabilities(composite));
        }

        // The same lanes are registered on the runner, so the public surface and the
        // internal call path resolve a logical identity identically.
        for (name, resolver, types) in lane_index {
            runner = runner.with_decision_lane(name, resolver, types);
        }
        if let Some(name) = &self.default_decision_lane {
            runner = runner.with_default_decision_lane(name.clone());
        }
        for (name, composition) in lane_composition {
            runner = runner.with_composed_decision_lane(name, composition);
        }
        Ok(runner)
    }

    /// Builds one daily request allowance per (quota group, model) that declares a limit.
    ///
    /// Keyed by the client's declared quota group rather than by the client's name, because
    /// the provider caps requests per project: two clients that name the same project are
    /// one budget, and keying them apart would hand the deployment twice the allowance it
    /// actually has. A client that declares no group is its own group, which leaves a
    /// single-client deployment unchanged.
    /// The counter a target's declared daily limit is counted on, and the clock it resets on.
    ///
    /// Model scope — the default — keys the counter by the target's model id, which is
    /// correct for a provider that budgets each model separately. Account scope collapses
    /// every model in the group onto one counter and one sentinel, because the provider
    /// states one allowance for the whole account; under model scope each free model would
    /// be granted the full budget and the fleet could spend several times the real cap
    /// before any counter reported exhaustion.
    fn quota_counter(
        &self,
        target: &TargetConfig,
        group: &str,
        state_dir: &std::path::Path,
    ) -> (
        ModelId,
        std::path::PathBuf,
        switchyard_llm_client::DayBoundary,
    ) {
        let client = self.llm_clients.get(&target.llm_client);
        let boundary = match client
            .and_then(|client| client.quota_reset)
            .unwrap_or_default()
        {
            QuotaReset::Pacific => switchyard_llm_client::DayBoundary::Pacific,
            QuotaReset::Utc => switchyard_llm_client::DayBoundary::Utc,
        };
        match client
            .and_then(|client| client.quota_scope)
            .unwrap_or_default()
        {
            QuotaScope::Account => (
                ModelId::from(switchyard_llm_client::ACCOUNT_SCOPE_MODEL),
                switchyard_llm_client::account_allowance_path(state_dir, group),
                boundary,
            ),
            QuotaScope::Model => (
                target.id.clone(),
                switchyard_llm_client::allowance_path(state_dir, group, target.id.as_str()),
                boundary,
            ),
        }
    }

    fn build_allowances(
        &self,
    ) -> RunnerResult<HashMap<(String, ModelId), Arc<switchyard_llm_client::DailyRequestAllowance>>>
    {
        let mut allowances = HashMap::new();
        let mut limits: HashMap<(String, ModelId), (u64, &str)> = HashMap::new();
        for (target_name, target) in &self.targets {
            let Some(limit) = target.daily_request_limit else {
                continue;
            };
            if limit == 0 {
                return Err(RunnerError::configuration(format!(
                    "target {target_name} daily_request_limit must be at least 1"
                )));
            }
            let group = self
                .llm_clients
                .get(&target.llm_client)
                .and_then(|client| client.quota_group.clone())
                .unwrap_or_else(|| target.llm_client.clone());
            let quota = self.quota.as_ref().ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} declares daily_request_limit but the deployment configures no [quota] state_dir; a daily budget has to be counted to be enforced"
                ))
            })?;
            let (counter_model, path, boundary) =
                self.quota_counter(target, &group, &quota.state_dir);
            let key = (group.clone(), counter_model.clone());
            match limits.entry(key.clone()) {
                std::collections::hash_map::Entry::Occupied(existing) => {
                    let (first_limit, first_name) = existing.get();
                    if *first_limit != limit {
                        return Err(RunnerError::configuration(format!(
                            "targets {first_name} and {target_name} draw on the same daily counter ({}) in quota group {group} but declare different daily_request_limit values ({first_limit} and {limit}); one counter serves one budget, so state it once",
                            counter_model.as_str()
                        )));
                    }
                    continue;
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert((limit, target_name.as_str()));
                }
            }
            allowances.insert(
                key,
                Arc::new(switchyard_llm_client::DailyRequestAllowance::new_on(
                    group.clone(),
                    counter_model,
                    limit,
                    path,
                    boundary,
                )),
            );
        }
        if !allowances.is_empty() {
            self.validate_quota_state_dir()?;
            // The directory probe above is not enough. A counter file can be left behind by
            // a different account - the deployment path runs as root while the unit runs as
            // its own user - and then the DIRECTORY is writable while that model's counter
            // and lock are not. The first caller would see a bare `Permission denied`
            // reported as an unavailable allowance.
            //
            // Checked per counter, at load, and refused with the file and its owner named.
            // Absent counters are not created here on purpose: the first reservation creates
            // them as the account that spends, so a root-run preflight cannot leave files
            // the service user cannot write.
            for allowance in allowances.values() {
                allowance.check_usable().map_err(|reason| {
                    RunnerError::configuration(format!(
                        "the daily request counter for model {} in quota group {} cannot be used \
                         by this process: {reason}",
                        allowance.model(),
                        allowance.group()
                    ))
                })?;
            }
        }
        Ok(allowances)
    }

    /// Proves the allowance counter directory is usable now, not at the first request.
    ///
    /// Without this, a wrong path or ownership would leave the budget unenforced until a
    /// request arrived - and then surface as a refused leg rather than as the configuration
    /// error it is.
    /// Checks that a pool route's group labels and candidate list describe the same leaves.
    ///
    /// Every member must exist, be callable by this route, and positively declare itself
    /// free. A group with no members is refused too: a pool that names an empty group has
    /// silently become a different pool, and that is exactly the drift the label exists to
    /// prevent.
    fn validate_terminal_fallback(
        &self,
        route_name: &str,
        config: &RouteConfig,
    ) -> RunnerResult<Vec<Vec<ModelId>>> {
        let Some(declaration) = &config.terminal_fallback else {
            if config.terminal_policy.is_some() {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} declares terminal_policy but no terminal_fallback; a \
                     policy that governs no leg would read as a guarantee it cannot provide"
                )));
            }
            if config.fit_escalation == Some(true) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} declares fit_escalation but no terminal_fallback; a \
                     size refusal would have no wider tier to escalate to, so the option would \
                     read as a guarantee it cannot provide"
                )));
            }
            if config.capability_escalation == Some(true) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} declares capability_escalation but no terminal_fallback; a \
                     wire refusal would have no later group to consider, so the option would \
                     read as a guarantee it cannot provide"
                )));
            }
            return Ok(Vec::new());
        };
        let tiers = declaration.tiers();
        if tiers.is_empty() {
            return Err(RunnerError::configuration(format!(
                "route {route_name} declares terminal_fallback but names no target"
            )));
        }
        let candidates = config.callable_target_names();
        let mut resolved: Vec<Vec<ModelId>> = Vec::new();
        // Duplicates are refused across the whole declaration, not just inside one tier: a
        // target named twice would be tried twice under two different exhaustion claims, and
        // the second attempt would repeat a call the first one already made.
        let mut seen: Vec<ModelId> = Vec::new();
        for (index, tier) in tiers.iter().enumerate() {
            let names = tier.names();
            if names.is_empty() {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} terminal_fallback tier {} names no target; an empty \
                     tier would escalate past a leg that was never consulted",
                    index + 1
                )));
            }
            let mut tier_ids: Vec<ModelId> = Vec::new();
            for name in names {
                let Some(target) = self.targets.get(name) else {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} terminal_fallback names unknown target {name}"
                    )));
                };
                if candidates.contains(&name) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} terminal_fallback names {name}, which is also a \
                         candidate; a terminal leg must not be ranked with the primary pool, or \
                         it would serve whenever that pool was merely busy"
                    )));
                }
                if target.held.is_some() {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} terminal_fallback names {name}, which is held; a \
                         withdrawn target cannot be a last resort"
                    )));
                }
                if seen.contains(&target.id) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} terminal_fallback names {name} more than once"
                    )));
                }
                seen.push(target.id.clone());
                tier_ids.push(target.id.clone());
            }
            resolved.push(tier_ids);
        }
        Ok(resolved)
    }

    fn validate_route_pool(
        &self,
        route_name: &str,
        config: &RouteConfig,
        groups: &[String],
    ) -> RunnerResult<()> {
        if groups.is_empty() {
            return Err(RunnerError::configuration(format!(
                "route {route_name} declares pool_groups but names no group"
            )));
        }
        let callable = config.callable_target_names();
        let mut members: Vec<&str> = Vec::new();
        for group in groups {
            let mut names: Vec<&str> = self
                .targets
                .iter()
                .filter(|(_name, target)| target.groups.iter().any(|label| label == group))
                .map(|(name, _target)| name.as_str())
                .collect();
            names.sort_unstable();
            if names.is_empty() {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} pool_groups names group {group}, which no target belongs to"
                )));
            }
            for name in names {
                let target = self
                    .targets
                    .get(name)
                    .expect("group membership was read from this map");
                // The pool is a free pool by definition, so membership implies a positive
                // free declaration. `Unknown` is refused for the same reason it is excluded
                // at request time.
                // One authority for the question, so the load gate and the request-time gate
                // cannot disagree about whether a tariff is free.
                if !switchyard_llm_client::Tariff::from_declaration(target.free_tier).is_free() {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} pool_groups names group {group}, whose member                          {name} does not declare free_tier = true; a free peer pool cannot                          contain a target whose tariff is not positively free"
                    )));
                }
                if !callable.contains(&name) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} pool_groups names group {group}, whose member                          {name} is not a candidate of this route; the pool and the candidate                          list must state the same membership"
                    )));
                }
                if !members.contains(&name) {
                    members.push(name);
                }
            }
        }
        // The union of the groups must be exactly the route's candidates: a member missing
        // from the list would never be called, and a listed candidate outside every group
        // would be a leg with no stated membership.
        let listed: Vec<&str> = callable.clone();
        let mut extra: Vec<&str> = listed
            .iter()
            .copied()
            .filter(|name| !members.contains(name))
            .collect();
        extra.sort_unstable();
        if !extra.is_empty() {
            return Err(RunnerError::configuration(format!(
                "route {route_name} lists candidates {extra:?} that belong to none of its                  pool_groups {groups:?}"
            )));
        }
        let mut missing: Vec<&str> = members
            .iter()
            .copied()
            .filter(|name| !listed.contains(name))
            .collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(RunnerError::configuration(format!(
                "route {route_name} pool_groups include {missing:?}, which are not in its                  candidate list"
            )));
        }
        Ok(())
    }

    fn validate_quota_state_dir(&self) -> RunnerResult<()> {
        let Some(quota) = self.quota.as_ref() else {
            return Ok(());
        };
        // Only an EXISTING directory is probed. Creating it here would be a load-time side
        // effect performed by whoever validated - including a root-run preflight - and a
        // root-owned state directory under a unit that runs as its own user is exactly the
        // failure this check exists to prevent. A missing directory is created by the first
        // reservation, as the account that will own it.
        if !quota.state_dir.exists() {
            return Ok(());
        }
        let probe = quota.state_dir.join(".write-probe");
        std::fs::write(&probe, b"switchyard quota state\n").map_err(|error| {
            RunnerError::configuration(format!(
                "[quota] state_dir {} is not writable: {error}",
                quota.state_dir.display()
            ))
        })?;
        std::fs::remove_file(&probe).map_err(|error| {
            RunnerError::configuration(format!(
                "[quota] state_dir {} probe could not be removed: {error}",
                quota.state_dir.display()
            ))
        })?;
        Ok(())
    }

    /// Rejects a provider-pool mapping that could not be applied honestly.
    ///
    /// Three rules, each of which prevents a specific wrong admission rather than a
    /// stylistic preference:
    ///
    /// * The pool is account-wide, so the counter it reconciles must be account-scoped. A
    ///   per-model counter fed an account total would refuse model B on the account's
    ///   spend and then refuse model A again on the same number -- one account's figure
    ///   spent as though it were several budgets.
    /// * The boundary must be the pool's own. A reading is placed in a day on the
    ///   counter's clock; an OpenRouter reading counted on the Pacific clock lands in the
    ///   wrong day for eight hours a night, so exhaustion from yesterday could refuse the
    ///   new UTC day or, worse, a fresh reading could be ignored as belonging to another
    ///   day.
    /// * The client must actually have a counter. Naming a pool for a client whose targets
    ///   declare no `daily_request_limit` would authorize a reconciliation with nowhere to
    ///   write, so the mapping would read as configured while doing nothing.
    fn validate_quota_providers(&self) -> RunnerResult<()> {
        for (name, client) in &self.llm_clients {
            let Some(provider) = client.quota_provider else {
                continue;
            };
            let scope = client.quota_scope.unwrap_or_default();
            if scope != QuotaScope::Account {
                return Err(RunnerError::configuration(format!(
                    "llm client {name} names quota_provider = \"{}\" but quota_scope is \
                     {:?}; {} reports one allowance for the whole account, so the counter it \
                     reconciles must be account-scoped (quota_scope = \"account\") or several \
                     model counters would each spend the one account figure",
                    provider.as_str(),
                    scope,
                    provider.as_str()
                )));
            }
            let reset = client.quota_reset.unwrap_or_default();
            if reset != provider.reset() {
                return Err(RunnerError::configuration(format!(
                    "llm client {name} names quota_provider = \"{}\" but quota_reset is {:?}; \
                     {} turns over at midnight {} ({}), so the counter must use the same \
                     clock or a reading lands in a day the provider does not recognise",
                    provider.as_str(),
                    reset,
                    provider.as_str(),
                    provider.reset().as_str(),
                    provider.pool()
                )));
            }
            let holds_counter = self.targets.values().any(|target| {
                target.llm_client == *name
                    && target.daily_request_limit.is_some_and(|limit| limit > 0)
            });
            if !holds_counter {
                return Err(RunnerError::configuration(format!(
                    "llm client {name} names quota_provider = \"{}\" but none of its targets \
                     declares daily_request_limit, so there is no counter to reconcile and \
                     the mapping would read as configured while admitting on nothing",
                    provider.as_str()
                )));
            }
        }
        Ok(())
    }

    fn build_clients(
        &self,
        provider_api_keys: &mut Vec<String>,
        allowances: &HashMap<(String, ModelId), Arc<switchyard_llm_client::DailyRequestAllowance>>,
        governor: &Arc<switchyard_llm_client::QuotaGovernor>,
    ) -> RunnerResult<BTreeMap<String, Arc<TranslatingLlmClient>>> {
        // Built before the per-target backends so each one can carry its own budget.
        let mut models_by_client = self
            .llm_clients
            .keys()
            .map(|name| (name.clone(), Vec::new()))
            .collect::<BTreeMap<String, Vec<ModelConfig>>>();

        for (name, client_config) in &self.llm_clients {
            validate_value("llm client name", name)?;
            let backend = build_backend(
                name,
                client_config,
                &BTreeMap::new(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                None,
                None,
                ReasoningDialect::default(),
                None,
                None,
                None,
                None,
            )?;
            let (Backend::OpenAiChat(config)
            | Backend::OpenAiResponses(config)
            | Backend::Anthropic(config)) = backend;
            if let Some(key) = config.api_key {
                provider_api_keys.push(key);
            }
        }
        for (target_name, target) in &self.targets {
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            let model_configs = models_by_client
                .get_mut(&target.llm_client)
                .ok_or_else(|| {
                    RunnerError::configuration("validated llm client was not initialized")
                })?;
            if let Some(policy) = target.reasoning_policy {
                policy
                    .validate_for(target.reasoning_dialect)
                    .map_err(|error| {
                        RunnerError::configuration(format!("target {target_name} {error}"))
                    })?;
            }
            if let Some(effort) = &target.reasoning_effort {
                if effort.trim().is_empty() {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort must not be empty"
                    )));
                }
                if matches!(client_config.format, ClientFormat::AnthropicMessages) {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} reasoning_effort is only supported on openai_chat and openai_responses clients"
                    )));
                }
            }
            let hosted_reasoning = match target.hosted_reasoning.as_deref() {
                None => None,
                Some(value) => Some(HostedReasoningMarker::parse(value).ok_or_else(|| {
                    RunnerError::configuration(format!(
                        "target {target_name} hosted_reasoning {value:?} is not a known marker; expected thought_tag"
                    ))
                })?),
            };
            model_configs.push(ModelConfig::new(
                target.id.clone(),
                build_backend(
                    &target.llm_client,
                    client_config,
                    &target.extra_body,
                    &target.omit_body_fields,
                    &target.unsupported_reasoning_effort,
                    target.reasoning_effort.clone(),
                    target.reasoning_policy,
                    target.reasoning_dialect,
                    target.cooldown_group.clone(),
                    allowances
                        .get(&(
                            client_config
                                .quota_group
                                .clone()
                                .unwrap_or_else(|| target.llm_client.clone()),
                            target.id.clone(),
                        ))
                        .cloned(),
                    pacing_for(target, client_config, governor),
                    hosted_reasoning,
                )?,
                None,
            ));
        }

        let mut clients = BTreeMap::new();
        for (name, model_configs) in models_by_client {
            let client = Arc::new(
                TranslatingLlmClient::new(&model_configs)
                    .map_err(|error| RunnerError::configuration(error.to_string()))?,
            );
            clients.insert(name, client);
        }
        Ok(clients)
    }

    fn build_targets(&self) -> BTreeMap<String, ModelId> {
        self.targets
            .iter()
            .map(|(name, config)| (name.clone(), config.id.clone()))
            .collect()
    }

    /// The holding record for a route whose every completion target is held.
    ///
    /// A route with at least one live candidate is not held: the live ones serve, and the
    /// held ones were already removed from its candidate set. Only a route with nothing left
    /// to call is registered as held, which is what keeps the refusal specific instead of
    /// letting the caller's alias fall through to the unknown-model path.
    fn route_held(
        &self,
        config: &RouteConfig,
        held_models: &BTreeSet<ModelId>,
    ) -> Option<HeldRoute> {
        if held_models.is_empty() {
            return None;
        }
        let names = config.routing_target_names();
        if names.is_empty() {
            return None;
        }
        let all_held = names.iter().all(|name| {
            self.targets
                .get(*name)
                .is_some_and(|target| target.held.is_some())
        });
        if !all_held {
            return None;
        }
        let first = self.targets.get(names[0])?;
        Some(HeldRoute {
            // Named by configuration, not by the model id: it is the alias the manifest
            // declares and the caller reads, and it survives in the message even though the
            // dispatch set no longer contains it.
            model: names[0].to_string(),
            reason: first.held.clone()?,
        })
    }

    /// Builds the decision resolver for a route, from its configured targets in order.
    ///
    /// Returns `None` when the route configures no decision backend, leaving the host
    /// to report decision calls as unsupported. Targets are taken in the route's own
    /// configured order, which is the candidate order; there is no scoring here.
    /// Validates every configured lane.
    ///
    /// Each lane's identity is explicit and its order comes from configuration. When no
    /// lane is configured there is none: a caller must then name a target, and an unknown
    /// name fails rather than quietly reaching whatever happens to exist.
    /// Validates every composite: children exist, appear once, are leaves, and cannot
    /// form a cycle.
    ///
    /// Nested composites are refused rather than supported: V6 needs ordered whole-route
    /// composition, and accepting recursion would mean either a cycle class of bugs or a
    /// general workflow engine.
    fn validate_composites(&self) -> RunnerResult<()> {
        for (parent, composite) in &self.composites {
            if composite.children.is_empty() {
                return Err(RunnerError::configuration(format!(
                    "composite route {parent} must list at least one child route"
                )));
            }
            let mut seen: Vec<&str> = Vec::new();
            for child in &composite.children {
                if child == parent {
                    return Err(RunnerError::configuration(format!(
                        "composite route {parent} lists itself as a child"
                    )));
                }
                if seen.contains(&child.as_str()) {
                    return Err(RunnerError::configuration(format!(
                        "composite route {parent} lists child {child} more than once"
                    )));
                }
                if self.composites.contains_key(child) {
                    return Err(RunnerError::configuration(format!(
                        "composite route {parent} lists composite {child} as a child; nested \
                         composites are not supported, so child routes must be leaves"
                    )));
                }
                if !self.routes.contains_key(child) {
                    return Err(RunnerError::configuration(format!(
                        "composite route {parent} references unknown route {child}"
                    )));
                }
                seen.push(child);
            }
        }
        Ok(())
    }

    /// Checks that a route advertising a capability has at least one completion target that
    /// actually supports it.
    ///
    /// The route declaration stays a policy promise; this only refuses a promise nothing in
    /// the route could keep. Capabilities are not derived from a union or intersection of
    /// targets: a route may deliberately be narrower than its widest target.
    fn validate_route_capability_claims(&self) -> RunnerResult<()> {
        for (route_name, route) in &self.routes {
            // A carrier without its medium would be a silently inert field: `capabilities()`
            // only reports a carrier for a medium the route actually claims, so a route
            // writing `video_carrier` without `video = true` would read as configured and do
            // nothing. Refuse it rather than accept a no-op.
            if route.video_carrier.is_some() && route.video != Some(true) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} declares video_carrier without video = true; a carrier \
                     is a narrowing of a medium the route must first claim"
                )));
            }
            if route.image_carrier.is_some() && route.vision != Some(true) {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} declares image_carrier without vision = true; a carrier \
                     is a narrowing of a medium the route must first claim"
                )));
            }
            let capabilities = route.capabilities();
            let mut names: Vec<&str> = route.routing_target_names().into_iter().collect();
            names.sort_unstable();
            for (label, claimed) in [
                ("vision", capabilities.vision),
                ("video", capabilities.video),
                ("tool_calling", capabilities.tool_calling),
            ] {
                if claimed != Some(true) {
                    continue;
                }
                // A target that says nothing is not a refusal: absence of a fact must not
                // invalidate a pre-existing deployment that never declared one.
                let supported = names.iter().any(|name| {
                    self.targets.get(*name).is_some_and(|target| match label {
                        "vision" => target.vision != Some(false),
                        // Video differs from the others: a target that says nothing is
                        // not a video target, it is one that has not claimed video.
                        "video" => target.video == Some(true),
                        _ => target.tool_calling != Some(false),
                    })
                });
                if !supported {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} declares {label} = true but none of its targets \
                         supports it; declare a target that does, or stop claiming it"
                    )));
                }
                // A carrier claim is a promise about a specific transport, and every carrier
                // the route advertises must be backed by some target below it. "Both" is two
                // promises: a target that only fetches URLs does not keep the inline half,
                // so a route advertising both over inline-only targets would send a caller
                // into a refusal the advertisement said could not happen.
                let claimed_carrier = match label {
                    "vision" => capabilities.image_carrier,
                    "video" => capabilities.video_carrier,
                    _ => None,
                };
                if let Some(claimed) = claimed_carrier {
                    let missing: Vec<MediaCarrier> = claimed
                        .carriers()
                        .iter()
                        .copied()
                        .filter(|need| {
                            !names.iter().any(|name| {
                                let Some(target) = self.targets.get(*name) else {
                                    return false;
                                };
                                // A target that refuses the medium serves none of its
                                // carriers, whatever its carrier field says.
                                let admits_medium = match label {
                                    "vision" => target.vision != Some(false),
                                    _ => target.video == Some(true),
                                };
                                // An undeclared carrier on an admitted medium means both,
                                // exactly as it does on a route.
                                let served = match label {
                                    "vision" => target.image_carrier,
                                    _ => target.video_carrier,
                                }
                                .unwrap_or(MediaCarrier::Both);
                                admits_medium && served.includes(*need)
                            })
                        })
                        .collect();
                    if !missing.is_empty() {
                        return Err(RunnerError::configuration(format!(
                            "route {route_name} declares {label} carrier {claimed:?} but no \
                             target serves {missing:?}; declare a target that does, or narrow \
                             the route's carrier"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Checks the per-minute window declarations before any backend carries one.
    ///
    /// Two rules, both of which fail closed at load rather than as a 429 under traffic: a
    /// window whose per-request ceiling exceeds it can never admit that request however long
    /// it waits, and two targets naming one physical model in one quota group share one
    /// window, so stating different values would make whichever loaded first the real limit.
    fn validate_minute_windows(&self) -> RunnerResult<()> {
        // Request rate, token rate, admission pad in per mille, and the target that stated
        // them: one declared window, as compared across aliases of one physical model.
        type DeclaredMinuteWindow<'a> = (Option<u64>, Option<u64>, u32, &'a str);
        let mut declared: HashMap<(String, ModelId), DeclaredMinuteWindow<'_>> = HashMap::new();
        for (target_name, target) in &self.targets {
            // Validate the pad for every target, not only paced ones: a margin naming no
            // token window still means the author expected padding to happen, and skipping
            // the check would let that mistake load and surface only as absent protection.
            let margin = admission_margin(target, target_name)?;
            if target.requests_per_minute.is_none() && target.input_tokens_per_minute.is_none() {
                continue;
            }
            if target.requests_per_minute == Some(0) || target.input_tokens_per_minute == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "target {target_name} declares a per-minute limit of zero; a window that \
                     admits nothing would refuse every request, so omit the field instead"
                )));
            }
            if let Some(limit) = target.input_tokens_per_minute {
                // The ceiling is judged at the margin, because that is the reservation the
                // send path will compute. Checking the raw ceiling instead would admit a
                // manifest whose stated ceiling the governor then refuses on every request —
                // a window that looks configured and serves nothing.
                let safe = switchyard_llm_client::WindowLimits {
                    requests_per_minute: target.requests_per_minute,
                    input_tokens_per_minute: target.input_tokens_per_minute,
                }
                .safe_input_ceiling(margin)
                .unwrap_or(limit);
                let margin_per_mille = margin.permille();
                match target.max_input_tokens {
                    None => {
                        return Err(RunnerError::configuration(format!(
                            "target {target_name} declares input_tokens_per_minute {limit} \
                             without max_input_tokens; a per-request ceiling at or below the \
                             minute budget is what keeps a request from being unservable no \
                             matter how long it waits"
                        )));
                    }
                    Some(max_input) if max_input > limit => {
                        return Err(RunnerError::configuration(format!(
                            "target {target_name} declares max_input_tokens {max_input} above \
                             its input_tokens_per_minute {limit}; a single request larger than \
                             a whole window can never be admitted"
                        )));
                    }
                    // Above the margin-adjusted ceiling but not above the raw window: this
                    // would admit nothing the governor allows, so it is a configuration that
                    // only pretends to be open past `safe`.
                    Some(max_input) if max_input > safe => {
                        return Err(RunnerError::configuration(format!(
                            "target {target_name} declares max_input_tokens {max_input}, but \
                             at an admission_margin_permille of {margin_per_mille} per mille against \
                             input_tokens_per_minute {limit} the largest estimate it can admit \
                             is {safe}; lower max_input_tokens to {safe} or below, or lower the \
                             margin"
                        )));
                    }
                    Some(_) => {}
                }
            }
            let group = self
                .llm_clients
                .get(&target.llm_client)
                .and_then(|client| client.quota_group.clone())
                .unwrap_or_else(|| target.llm_client.clone());
            match declared.entry((group.clone(), target.id.clone())) {
                std::collections::hash_map::Entry::Occupied(existing) => {
                    let (first_rpm, first_tpm, first_margin, first_name) = *existing.get();
                    if first_rpm != target.requests_per_minute
                        || first_tpm != target.input_tokens_per_minute
                        || first_margin != margin.permille()
                    {
                        return Err(RunnerError::configuration(format!(
                            "targets {first_name} and {target_name} share model {} in quota \
                             group {group} but declare different per-minute limits or \
                             admission margins; one window serves them both, so state it once",
                            target.id
                        )));
                    }
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert((
                        target.requests_per_minute,
                        target.input_tokens_per_minute,
                        margin.permille(),
                        target_name.as_str(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Checks that a route's advertised context is capacity some target can actually serve.
    ///
    /// `context_window` on a route is what a caller reads before choosing it, and it is
    /// plain metadata — nothing at request time compares it to a model's real window. A
    /// route aggregating several physical models may therefore advertise the widest of
    /// them, which is truthful: some eligible target can hold such a request. Advertising
    /// more than the widest target is not, so a route that states a window while its
    /// targets state physical windows must fit inside the largest of them. A target that
    /// states no physical window is treated as undeclared rather than as zero, so a
    /// manifest written before this field existed keeps loading unchanged.
    fn validate_physical_context_claims(&self) -> RunnerResult<()> {
        for (target_name, target) in &self.targets {
            if let Some(max_input) = target.max_input_tokens {
                if max_input == 0 {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} max_input_tokens must be at least 1"
                    )));
                }
                if let Some(window) = target.context_window
                    && max_input > window
                {
                    return Err(RunnerError::configuration(format!(
                        "target {target_name} declares max_input_tokens {max_input} above its \
                         context_window {window}; a per-request ceiling larger than the window \
                         is never the binding limit, so one of the two is wrong"
                    )));
                }
            }
            if target.context_window == Some(0) {
                return Err(RunnerError::configuration(format!(
                    "target {target_name} context_window must be at least 1"
                )));
            }
        }

        for (route_name, route) in &self.routes {
            let Some(claimed) = route.context_window else {
                continue;
            };
            let declared: Vec<u64> = route
                .routing_target_names()
                .into_iter()
                .filter_map(|name| self.targets.get(name))
                .filter_map(|target| target.context_window)
                .collect();
            let Some(widest) = declared.iter().copied().max() else {
                // No target states a physical window, so there is nothing to check against
                // and the historical manifest keeps its behaviour.
                continue;
            };
            if u64::from(claimed) > widest {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} advertises context_window {claimed} but its widest \
                     declared target window is {widest}; the route would promise capacity no \
                     candidate can serve"
                )));
            }
        }

        // A composite is resolved before its children, so the number a caller reads comes
        // from the composite itself. Leaving it unchecked would let the one surface a
        // bounded lane publishes be the only unverified promise in the file. A child
        // route's own stated window is the tighter and therefore the authoritative value;
        // only a child that states nothing falls back to its targets.
        for (name, composite) in &self.composites {
            let Some(claimed) = composite.context_window else {
                continue;
            };
            let widest = composite
                .children
                .iter()
                .filter_map(|child| self.routes.get(child))
                .filter_map(|route| {
                    route
                        .context_window
                        .map(u64::from)
                        .or_else(|| self.widest_target_window(route))
                })
                .max();
            let Some(widest) = widest else {
                // Nothing below states a window, so there is no claim to contradict.
                continue;
            };
            if u64::from(claimed) > widest {
                return Err(RunnerError::configuration(format!(
                    "composite {name} advertises context_window {claimed} but its widest \
                     declared child window is {widest}; the composite would promise capacity \
                     no child route can serve"
                )));
            }
        }
        Ok(())
    }

    /// The widest physical window any target behind `route` declares, if any does.
    fn widest_target_window(&self, route: &RouteConfig) -> Option<u64> {
        route
            .routing_target_names()
            .into_iter()
            .filter_map(|name| self.targets.get(name))
            .filter_map(|target| target.context_window)
            .max()
    }

    /// Checks a composite's effective media claim against the children that must keep it.
    ///
    /// A composite is resolved before its children and reports its own guarantee, so a claim
    /// nothing below it can keep would advertise a medium and then skip every child at
    /// request time. The claim checked here is the *effective* one -- a stated value, or the
    /// value derived from the children when the parent states nothing -- so a derived claim
    /// is held to the same standard as a hand-written one.
    ///
    /// Two failures are refused and one is only reported:
    ///
    /// * a carrier stated without the medium it narrows, which would otherwise be silently
    ///   inert;
    /// * an effective claim no child can keep (an explicit over-claim);
    /// * a parent that states *less* than its children serve (an explicit under-claim). That
    ///   is safe -- nothing false is advertised -- so it is warned about rather than refused,
    ///   and naming it at load is what stops a narrowed child from leaving a stale parent
    ///   promise behind unnoticed.
    fn validate_composite_media_claims(&self) -> RunnerResult<()> {
        for (name, composite) in &self.composites {
            let children: Vec<crate::ModelCapabilities> = composite
                .children
                .iter()
                .filter_map(|child| self.routes.get(child))
                .map(RouteConfig::capabilities)
                .collect();
            let effective = self.composite_capabilities(composite);
            let claims = [
                (
                    "vision",
                    composite.vision,
                    composite.image_carrier,
                    effective.vision,
                    effective.image_carrier,
                ),
                (
                    "video",
                    composite.video,
                    composite.video_carrier,
                    effective.video,
                    effective.video_carrier,
                ),
            ];
            for (label, stated_medium, stated_carrier, effective_medium, effective_carrier) in
                claims
            {
                // A stated carrier narrows a medium that is *served*, whether it was stated
                // or derived. Reading the raw statement here would refuse a narrowing over a
                // derived medium, which is a normal thing to write once children are the
                // source of the capability.
                if stated_carrier.is_some() && effective_medium != Some(true) {
                    return Err(RunnerError::configuration(format!(
                        "composite route {name} states a {label} carrier but does not serve \
                         {label}; a carrier narrows a medium, so state {label} = true beside a \
                         child that serves it, or drop the carrier"
                    )));
                }
                let serving = |caps: &crate::ModelCapabilities| match label {
                    "vision" => caps.vision == Some(true),
                    _ => caps.video == Some(true),
                };
                let child_carrier = |caps: &crate::ModelCapabilities| match label {
                    "vision" => caps.image_carrier.unwrap_or(MediaCarrier::DEFAULT),
                    _ => caps.video_carrier.unwrap_or(MediaCarrier::DEFAULT),
                };
                let Some(claimed) = effective_medium
                    .filter(|medium| *medium)
                    .and(effective_carrier)
                else {
                    continue;
                };
                if !children.iter().any(serving) {
                    return Err(RunnerError::configuration(format!(
                        "composite route {name} declares {label} = true but none of its \
                         children does; declare a child that does, or stop claiming it"
                    )));
                }
                let missing: Vec<MediaCarrier> = claimed
                    .carriers()
                    .iter()
                    .copied()
                    .filter(|need| {
                        !children
                            .iter()
                            .any(|caps| serving(caps) && child_carrier(caps).includes(*need))
                    })
                    .collect();
                if !missing.is_empty() {
                    return Err(RunnerError::configuration(format!(
                        "composite route {name} declares {label} carrier {claimed:?} but no \
                         child serves {missing:?}; declare a child that does, or narrow the \
                         composite's carrier"
                    )));
                }
                // Report, do not refuse: an explicit promise narrower than what the children
                // serve is safe, but it is the shape that goes stale when a child changes.
                if stated_medium == Some(false)
                    || stated_carrier.is_some_and(|stated| stated != claimed)
                {
                    let available = children
                        .iter()
                        .filter(|caps| serving(caps))
                        .map(child_carrier)
                        .reduce(MediaCarrier::union);
                    if let Some(available) = available.filter(|available| *available != claimed) {
                        tracing::warn!(
                            "composite route {name} states {label} = {stated_medium:?} carrier \
                             {stated_carrier:?} but its children serve {available:?}; the parent \
                             refuses media a child could serve. Omit the statement to derive it \
                             from the children."
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Checks that every declared carrier is backed by the medium it narrows.
    ///
    /// A video declaration without a carrier is the ambiguity this change removes: the
    /// provider may answer an inline video and 404 a URL, and nothing in the manifest would
    /// say which. Requiring the carrier turns that into a load error instead of a
    /// caller-visible 404 at request time. An image carrier is the mirror case and needs the
    /// vision claim it narrows.
    fn validate_target_carriers(&self) -> RunnerResult<()> {
        for (target_name, target) in &self.targets {
            if target.video == Some(true) && target.video_carrier.is_none() {
                return Err(RunnerError::configuration(format!(
                    "target {target_name} declares video = true but no video_carrier; state \
                     which of \"inline\" / \"remote\" / \"both\" it serves"
                )));
            }
            if target.image_carrier.is_some() && target.vision != Some(true) {
                return Err(RunnerError::configuration(format!(
                    "target {target_name} declares image_carrier without vision = true; \
                     a carrier is a narrowing of a medium it must first claim"
                )));
            }
        }
        Ok(())
    }

    /// Checks that a target claiming video sits on a transport that can carry it.
    ///
    /// Video has one faithful wire representation, the Responses `input_video` part. The
    /// OpenAI Chat codec has no stable video mapping and encodes the block as text, so a
    /// video declaration there would admit a request and then deliver a paragraph instead
    /// of the video. Refusing at load makes that a configuration error rather than a
    /// silent conversion at request time.
    fn validate_video_transport(&self) -> RunnerResult<()> {
        // Carrier declarations are checked for EVERY target, not only the video ones: an
        // image carrier is a real narrowing on a target that may declare no video at all.
        self.validate_target_carriers()?;
        for (target_name, target) in &self.targets {
            if target.video != Some(true) {
                continue;
            }
            let client = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {target_name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            if !matches!(client.format, ClientFormat::OpenAiResponses) {
                let format = match client.format {
                    ClientFormat::OpenAiChat => "openai_chat",
                    ClientFormat::AnthropicMessages => "anthropic_messages",
                    ClientFormat::OpenAiResponses => "openai_responses",
                };
                return Err(RunnerError::configuration(format!(
                    "target {target_name} declares video = true but its client uses {format}; \
                     video requires an openai_responses client"
                )));
            }
        }
        Ok(())
    }

    /// Refuses a Responses backend pointed at a provider that serves no such path.
    ///
    /// Google documents `/v1beta/openai/chat/completions` and no `/openai/responses`
    /// equivalent. A target configured as `openai_responses` against that base URL would build
    /// a path the provider has never offered, and the resulting 404 is a missing endpoint
    /// rather than a missing model - which is exactly the confusion that gets a healthy model
    /// recorded as retired. Checked at load so the misconfiguration cannot reach a request,
    /// and the message names the fix.
    fn validate_responses_wire_targets(&self) -> RunnerResult<()> {
        for (name, client) in &self.llm_clients {
            if !matches!(client.format, ClientFormat::OpenAiResponses) {
                continue;
            }
            if client
                .base_url
                .as_str()
                .to_ascii_lowercase()
                .contains("generativelanguage.googleapis.com")
            {
                return Err(RunnerError::configuration(format!(
                    "llm client {name} uses openai_responses against {}; Google's \
                     OpenAI-compatible surface documents /v1beta/openai/chat/completions only \
                     and serves no Responses path. Use format = \"openai_chat\" for this client",
                    client.base_url.as_str()
                )));
            }
        }
        Ok(())
    }

    fn validate_decision_lanes(&self) -> RunnerResult<()> {
        for (name, lane) in &self.decision_lanes {
            if lane.targets.is_empty() {
                return Err(RunnerError::configuration(format!(
                    "decision lane {name} must list at least one target"
                )));
            }
            let mut seen: Vec<&str> = Vec::new();
            for target_name in &lane.targets {
                if seen.contains(&target_name.as_str()) {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} lists target {target_name} more than once; \
                         candidate order is a sequence, so each target may appear once"
                    )));
                }
                let Some(target) = self.targets.get(target_name) else {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} references unknown target {target_name}"
                    )));
                };
                if target.decision_transport.is_none() {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} lists target {target_name}, which is not a decision target"
                    )));
                }
                seen.push(target_name);
            }
        }
        // Two scoped lanes claiming the same kinds would make selection ambiguous, so the
        // deployment is refused here rather than failing at request time.
        let mut claimed: BTreeMap<decision_transport::QuestionKind, String> = BTreeMap::new();
        for (name, lane) in &self.decision_lanes {
            for kind in &lane.types {
                if claimed.insert(*kind, name.clone()).is_some() {
                    return Err(RunnerError::configuration(format!(
                        "decision lanes claim question kind {kind:?} at the same specificity; \
                         each kind may be scoped by at most one lane"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validates the declared default decision lane, which must name a lane this
    /// deployment actually configures. A default that names nothing would make every
    /// unaddressed request fail late, at request time, instead of at load.
    fn validate_default_decision_lane(&self) -> RunnerResult<()> {
        let Some(name) = self.default_decision_lane.as_deref() else {
            return Ok(());
        };
        if name.is_empty() {
            return Err(RunnerError::configuration(
                "default_decision_lane must not be empty; omit it instead",
            ));
        }
        if !self.decision_lanes.contains_key(name) {
            return Err(RunnerError::configuration(format!(
                "default_decision_lane names {name}, which is not a configured decision lane"
            )));
        }
        Ok(())
    }

    /// A decision lane that composes child POLICY lanes in order.
    ///
    /// Distinct from `DecisionLaneConfig`, which holds physical targets: a composed
    /// lane holds child LANE names, so a tier references its families and inherits
    /// their policy instead of copying their target lists.
    fn validate_decision_lane_composition(&self) -> RunnerResult<()> {
        for (name, composition) in &self.decision_lane_composition {
            let children = &composition.children;
            if children.is_empty() {
                return Err(RunnerError::configuration(format!(
                    "decision lane {name} must list at least one child lane"
                )));
            }
            // A name may not be both a physical lane and a composed lane: the two are
            // different mechanisms, and silently preferring one would make the other's
            // policy unreachable.
            if self.decision_lanes.contains_key(name) {
                return Err(RunnerError::configuration(format!(
                    "decision lane {name} is declared both as a target lane and as a composed lane"
                )));
            }
            let mut seen: Vec<&str> = Vec::new();
            for child in children {
                if child == name {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} lists itself as a child"
                    )));
                }
                if seen.contains(&child.as_str()) {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} lists child lane {child} more than once"
                    )));
                }
                if !self.decision_lanes.contains_key(child.as_str())
                    && !self.decision_lane_composition.contains_key(child.as_str())
                {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} references unknown child lane {child}"
                    )));
                }
                seen.push(child.as_str());
            }
        }
        self.validate_decision_lane_acyclic()
    }

    /// Rejects a composition cycle, so a configuration that would recurse at request
    /// time is refused at load instead.
    fn validate_decision_lane_acyclic(&self) -> RunnerResult<()> {
        /// Depth-first walk with the current path on the stack.
        fn walk(
            composition: &BTreeMap<String, DecisionLaneCompositionConfig>,
            lane: &str,
            path: &mut Vec<String>,
        ) -> Result<(), String> {
            if let Some(start) = path.iter().position(|entry| entry == lane) {
                let mut cycle: Vec<&str> = path[start..].iter().map(String::as_str).collect();
                cycle.push(lane);
                return Err(cycle.join(" -> "));
            }
            let Some(entry) = composition.get(lane) else {
                return Ok(());
            };
            path.push(lane.to_string());
            for child in &entry.children {
                if composition.contains_key(child.as_str()) {
                    walk(composition, child.as_str(), path)?;
                }
            }
            path.pop();
            Ok(())
        }

        let mut lanes: Vec<&String> = self.decision_lane_composition.keys().collect();
        lanes.sort();
        for lane in lanes {
            let mut path = Vec::new();
            if let Err(cycle) = walk(&self.decision_lane_composition, lane.as_str(), &mut path) {
                return Err(RunnerError::configuration(format!(
                    "decision lane composition contains a cycle: {cycle}"
                )));
            }
        }
        Ok(())
    }

    /// Flattens each composed lane into its ordered children once, at load, so a
    /// request walks a precomputed list and a child that could not be built is a load
    /// error rather than a silent skip at request time.
    fn resolve_decision_lane_composition(
        &self,
        lanes: &BTreeMap<String, decision_executor::LaneResolverHandle>,
    ) -> RunnerResult<Vec<(String, decision_executor::ComposedLane)>> {
        let mut resolved = Vec::with_capacity(self.decision_lane_composition.len());
        for (name, composition) in &self.decision_lane_composition {
            let children = &composition.children;
            let mut ordered = Vec::with_capacity(children.len());
            for child in children {
                // A composed child is legal and is resolved recursively at request
                // time; requiring it to hold targets here would forbid a tier of
                // tiers. It is accepted only if it is itself a composition, and
                // cycles were already rejected above.
                let is_leaf = lanes.contains_key(child.as_str());
                let is_composite = self.decision_lane_composition.contains_key(child.as_str());
                if !is_leaf && !is_composite {
                    return Err(RunnerError::configuration(format!(
                        "decision lane {name} composes child lane {child}, which resolved to no \
                         decision targets"
                    )));
                }
                ordered.push(child.clone());
            }
            resolved.push((
                name.clone(),
                decision_executor::ComposedLane {
                    children: ordered,
                    selection: composition.selection,
                },
            ));
        }
        Ok(resolved)
    }

    fn build_decision_resolver(
        &self,
        target_names: &[&str],
        accounting: &Accounting<'_>,
    ) -> Option<decision_executor::DecisionResolver> {
        let mut candidates = Vec::new();
        for name in target_names {
            let target = self.targets.get(*name)?;
            let client = self.llm_clients.get(&target.llm_client)?;
            let transport = target.decision_transport?;
            let credential = target
                .decision_api_key_env
                .as_deref()
                .map(|variable| std::env::var(variable).ok())
                .unwrap_or(None);
            let url = format!(
                "{}{}",
                target
                    .decision_base_url
                    .clone()
                    .unwrap_or_else(|| client.base_url.as_str().to_string())
                    .trim_end_matches('/'),
                target
                    .decision_path
                    .clone()
                    .unwrap_or_else(|| transport.path().to_string())
            );
            candidates.push(decision_executor::DecisionCandidate {
                name: (*name).to_string(),
                model: target.id.to_string(),
                transport,
                url,
                api_key: credential,
                api_key_env: target.decision_api_key_env.clone(),
                supported_types: target
                    .supported_types
                    .unwrap_or(decision_transport::SupportedTypes::ALL),
                state_forms: target.state_forms,
                budget: self.decision_budget(
                    target,
                    client,
                    accounting.governor,
                    accounting.allowances,
                ),
            });
        }
        (!candidates.is_empty()).then_some(decision_executor::DecisionResolver { candidates })
    }

    /// The provider budgets one decision backend's calls are charged to.
    ///
    /// A decision call is one request to one physical model, so it draws on exactly the
    /// budgets the same model draws on when it is used for generation: the same quota
    /// group, the same daily counter, the same minute window. Two counters for one model
    /// would let a deployment hold two allowances the provider never granted.
    fn decision_budget(
        &self,
        target: &TargetConfig,
        client: &LlmClientConfig,
        governor: &Arc<switchyard_llm_client::QuotaGovernor>,
        allowances: &HashMap<(String, ModelId), Arc<switchyard_llm_client::DailyRequestAllowance>>,
    ) -> Option<Arc<decision_executor::DecisionBudget>> {
        let group = client
            .quota_group
            .clone()
            .unwrap_or_else(|| target.llm_client.clone());
        let allowance = allowances.get(&(group.clone(), target.id.clone())).cloned();
        let limits = switchyard_llm_client::WindowLimits {
            requests_per_minute: target.requests_per_minute,
            input_tokens_per_minute: target.input_tokens_per_minute,
        };
        let pacing = if limits.is_unlimited() {
            None
        } else {
            Some(Arc::new(switchyard_llm_client::PacedQuota::new(
                Arc::clone(governor),
                switchyard_llm_client::GovernorKey::new(group, target.id.clone(), "decide"),
                limits,
                admission_margin_opt(target),
            )))
        };
        if allowance.is_none() && pacing.is_none() {
            return None;
        }
        Some(Arc::new(decision_executor::DecisionBudget {
            allowance,
            pacing,
        }))
    }

    // One internal builder with one caller. Grouping these further would introduce a
    // carrier struct purely to satisfy the lint, and would separate the decision wiring
    // from the route it wires.
    #[allow(clippy::too_many_arguments)]
    fn build_route_clients(
        &self,
        route_name: &str,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        lane_identity: Option<String>,
        lanes: &[(
            String,
            Arc<decision_executor::DecisionResolver>,
            Vec<decision_transport::QuestionKind>,
        )],
        // Named composition, threaded to the internal decision handler so an internal
        // `Call::Decision` resolves a named lane by the SAME mechanism as the public
        // surface. Without it, the same request could be served at `/v1/decisions`
        // and fail internally, which is a split contract rather than one mechanism.
        composition: &BTreeMap<String, decision_executor::ComposedLane>,
        // The deployment's one governor and its per-model daily counters, threaded here so
        // a route-local decision resolver is paced by the same ledgers as a named lane:
        // the public surface and an internal Call::Decision must not hold two different
        // views of one model's remaining budget.
        accounting: &Accounting<'_>,
    ) -> RunnerResult<(ClientRouter, Option<CallerAuthKind>)> {
        let TargetPromptPolicy {
            prompts,
            routing_answer_target,
        } = self.build_route_target_prompts(route_name, route)?;
        let mut by_model = HashMap::new();
        let mut targets_by_model: HashMap<&str, (&str, &TargetConfig)> = HashMap::new();
        let mut caller_auth = None;
        // The terminal leg is dispatched through the same router, so its client has to be
        // constructed here even though it is never a ranked candidate.
        let mut contact_names = route.callable_target_names();
        for name in route
            .terminal_fallback
            .iter()
            .flat_map(TerminalFallback::name_refs)
        {
            if !contact_names.contains(&name) {
                contact_names.push(name);
            }
        }
        for name in contact_names {
            let target = self.targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            if let Some((first_name, first)) = targets_by_model.insert(&target.id, (name, target))
                && first.llm_client != target.llm_client
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} targets {first_name} and {name} use model {} on different llm clients; execution is keyed by model id, so use distinct model ids within this route or put these targets in separate routes",
                    target.id
                )));
            }
            let client = clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!("target {name} has no constructed llm client"))
            })?;
            let client_config = self.llm_clients.get(&target.llm_client).ok_or_else(|| {
                RunnerError::configuration(format!(
                    "target {name} references unknown llm client {}",
                    target.llm_client
                ))
            })?;
            if client_config.forward_auth {
                let target_auth = client_config.format.caller_auth_kind();
                if caller_auth.is_some_and(|kind| kind != target_auth) {
                    return Err(RunnerError::configuration(format!(
                        "route {route_name} cannot forward both Anthropic and OpenAI caller credentials"
                    )));
                }
                caller_auth = Some(target_auth);
            }
            let client: Arc<dyn RoutedLlmClient> = client.clone();
            by_model.insert(target.id.clone(), client);
        }
        let completion_targets = route
            .routing_target_names()
            .into_iter()
            .map(|name| self.targets[name].id.clone())
            .collect::<Vec<_>>();
        let router = ClientRouter::new_with_completion_targets(
            by_model,
            prompts,
            routing_answer_target,
            &completion_targets,
        );
        // Decision calls need a host. One resolver is built from this route's configured
        // decision targets, in configured order, and serves both the internal call path
        // and any public typed surface, so Decision has one implementation.
        let route_targets = route.served_target_names();
        // Target capability facts reach the client router so ordered candidates can be
        // filtered before dispatch. A target with no declaration is left in place.
        let mut facts: HashMap<ModelId, switchyard_llm_client::CapabilityFacts> = HashMap::new();
        for name in &route_targets {
            if let Some(target) = self.targets.get(*name) {
                facts.insert(
                    target.id.clone(),
                    switchyard_llm_client::CapabilityFacts {
                        no_vision: target.vision == Some(false),
                        // Video is never inferred: a target that does not state support is
                        // treated as unable to receive it.
                        no_video: target.video != Some(true),
                        // A target that declares video without carriers means both, so an
                        // older manifest keeps the behaviour it had before carriers existed.
                        no_remote_video: !matches!(
                            target.video_carrier.unwrap_or(MediaCarrier::Both),
                            MediaCarrier::Remote | MediaCarrier::Both
                        ),
                        no_inline_video: !matches!(
                            target.video_carrier.unwrap_or(MediaCarrier::Both),
                            MediaCarrier::Inline | MediaCarrier::Both
                        ),
                        // Images carry the same distinction and the same default: a target
                        // that declares vision without carriers means both.
                        no_remote_image: !matches!(
                            target.image_carrier.unwrap_or(MediaCarrier::Both),
                            MediaCarrier::Remote | MediaCarrier::Both
                        ),
                        no_inline_image: !matches!(
                            target.image_carrier.unwrap_or(MediaCarrier::Both),
                            MediaCarrier::Inline | MediaCarrier::Both
                        ),
                        no_tool_calling: target.tool_calling == Some(false),
                        // A target that states its wire cannot accept a replayed tool call
                        // is skipped for a request carrying one, and stays eligible for the
                        // request that merely offers tools. Undeclared keeps permitting,
                        // so a manifest written before the field behaves as it did.
                        no_tool_call_replay: target.tool_call_replay == Some(false),
                        // A target states when it cannot carry a caller's reasoning control;
                        // an undeclared target keeps permitting it.
                        no_reasoning_control: target.accepts_reasoning_control == Some(false),
                        // Declared per target so selection can order a free candidate ahead
                        // of a metered one without inferring cost from a provider name or a
                        // model slug. Absent stays `Unknown`, which is not free.
                        tariff: switchyard_llm_client::Tariff::from_declaration(target.free_tier),
                        // The physical window and the per-request input ceiling travel as
                        // facts so the pre-dispatch filter can drop a target that cannot
                        // hold this request, before any provider is contacted.
                        context_window: target.context_window,
                        max_input_tokens: target.max_input_tokens,
                        // The bound the deployment will actually admit: the declared ceiling
                        // capped by what a margin-padded estimate can fit in the minute
                        // window. Computed from the same arithmetic the send path uses, so
                        // the filter and the governor cannot disagree about where a request
                        // stops fitting.
                        safe_input_tokens: safe_input_tokens(target),
                        // A target that charges its own reasoning against the caller's
                        // output reservation states the smallest reservation it can answer
                        // within, so a smaller one is skipped instead of attempted.
                        min_output_tokens: target.min_output_tokens,
                        // The same declaration the send boundary uses to strip a value the
                        // provider refuses. Carried here so a candidate can be judged
                        // ineligible BEFORE the call when the caller is the one who asked
                        // for that value, instead of being sent and silently altered.
                        unsupported_reasoning_effort: target.unsupported_reasoning_effort.clone(),
                    },
                );
            }
        }
        let router = router.with_target_capabilities(facts);
        // The credential each target draws on travels with the router as well, so a
        // failure is attributed to the target that actually failed rather than to the
        // route's first target. A mixed-provider free pool is exactly the case where
        // those two differ.
        let router = router.with_target_credentials(target_credentials(route, &self.targets));

        // A route-level resolver serves a concrete target; the configured lanes serve a
        // logical identity and may order candidates differently per question type.
        if let Some(resolver) = self.build_decision_resolver(&route_targets, accounting) {
            let resolver = Arc::new(resolver);
            let handler_resolver = Arc::clone(&resolver);
            let handler_lane: Arc<Option<String>> = Arc::new(lane_identity.clone());
            let handler_lanes = Arc::new(lanes.to_vec());
            let handler_composition = Arc::new(composition.clone());
            let handler_lane_resolvers: Arc<
                BTreeMap<String, Arc<decision_executor::DecisionResolver>>,
            > = Arc::new(
                lanes
                    .iter()
                    .map(|(name, resolver, _)| (name.clone(), Arc::clone(resolver)))
                    .collect(),
            );
            let router = router.with_decision_handler(Arc::new(move |request, model| {
                let resolver = Arc::clone(&handler_resolver);
                let lane_name = Arc::clone(&handler_lane);
                let lanes = Arc::clone(&handler_lanes);
                let composition = Arc::clone(&handler_composition);
                let lane_resolvers = Arc::clone(&handler_lane_resolvers);
                Box::pin(async move {
                    // A call that names a concrete model means that backend only: it is
                    // how a caller pins one for diagnostics or comparison. The lane is
                    // reached when the named target is the lane identity itself, or when
                    // no configured candidate matches the name.
                    // An explicit identity is either a configured target or the named
                    // lane. An unknown one fails closed instead of reaching whichever
                    // backend happens to exist.
                    // An EXPLICIT lane name is authoritative and is resolved BEFORE
                    // the route-local resolver and before any question-kind selection, so
                    // a named family or tier runs its own policy internally exactly as it
                    // does on the public surface. Without this the same request could be
                    // served at `/v1/decisions` and fail here, which would be a split
                    // contract rather than one mechanism.
                    let handles: BTreeMap<String, decision_executor::LaneResolverHandle> =
                        lane_resolvers
                            .iter()
                            .map(|(name, resolver)| {
                                (
                                    name.clone(),
                                    decision_executor::LaneResolverHandle {
                                        resolver: Arc::clone(resolver),
                                    },
                                )
                            })
                            .collect();
                    let composition = Arc::clone(&composition);
                    let requested_model = model.clone();
                    // An EXPLICIT lane name is authoritative and is resolved BEFORE the
                    // route-local resolver and before any question-kind selection, so a
                    // named family or tier runs its own policy internally exactly as it
                    // does on the public surface. Without this the same request could be
                    // served at `/v1/decisions` and fail here, which would be a split
                    // contract rather than one mechanism.
                    let served = async move {
                        if let Some(outcome) = decision_executor::serve_named(
                            &composition,
                            &handles,
                            requested_model.as_str(),
                            &request,
                        )
                        .await?
                        {
                            return Ok(outcome);
                        }
                        match decision_executor::resolve_identity(
                            &resolver,
                            lane_name.as_deref(),
                            requested_model.as_str(),
                        )? {
                            decision_executor::DecisionIdentity::Target(name) => {
                                let candidate = resolver
                                    .candidates
                                    .iter()
                                    .find(|candidate| {
                                        candidate.name == name || candidate.model == name
                                    })
                                    .ok_or(decision_executor::DecisionFailure::NotEligible(
                                        decision_executor::DecisionSkip::TargetNotFound,
                                    ))?;
                                decision_executor::serve_target(candidate, &request).await
                            }
                            decision_executor::DecisionIdentity::Lane => {
                                let selected = lanes
                                    .iter()
                                    .find(|(_, _, types)| {
                                        decision_transport::lane_serves(types, &request)
                                    })
                                    .map(|(_, resolver, _)| Arc::clone(resolver));
                                match selected {
                                    Some(lane) => {
                                        decision_executor::resolve_and_serve(&lane, &request).await
                                    }
                                    None => Err(decision_executor::DecisionFailure::NotEligible(
                                        decision_executor::DecisionSkip::TargetNotFound,
                                    )),
                                }
                            }
                        }
                    }
                    .await;
                    served.map(|outcome| outcome.response).map_err(|failure| {
                        // The host contract carries a LibsyError. The classified
                        // reason is kept in the message so a caller can tell an
                        // eligibility or contract refusal from a serving failure.
                        LibsyError::client_call(
                            model.clone(),
                            switchyard_protocol::LlmClientError::General(
                                DecisionCallError {
                                    requested: model.to_string(),
                                    failure,
                                }
                                .to_string(),
                            ),
                        )
                    })
                })
            }));
            return Ok((router, caller_auth));
        }
        Ok((router, caller_auth))
    }

    /// Builds the effective prompt policy for this route's completion targets.
    fn build_route_target_prompts(
        &self,
        route_name: &str,
        route: &RouteConfig,
    ) -> RunnerResult<TargetPromptPolicy> {
        let mut prompts = HashMap::new();
        let mut aliases = HashMap::<&ModelId, Option<&str>>::new();
        for name in route.served_target_names() {
            let target = self.targets.get(name).ok_or_else(|| {
                RunnerError::configuration(format!("route references unknown target {name}"))
            })?;
            let prompt = target.system_prompt.as_deref();
            if aliases
                .insert(&target.id, prompt)
                .is_some_and(|configured| configured != prompt)
            {
                return Err(RunnerError::configuration(format!(
                    "route {route_name} maps completion target aliases to model {} with different system_prompt values",
                    target.id
                )));
            }
            if let Some(prompt) = prompt {
                prompts.insert(target.id.clone(), prompt.to_string());
            }
        }
        let mut policy = TargetPromptPolicy {
            prompts,
            routing_answer_target: None,
        };
        let Some((response_name, dependency_name)) =
            route.algorithm.routing_response_and_dependency()
        else {
            return Ok(policy);
        };
        let response = self.targets.get(response_name).ok_or_else(|| {
            RunnerError::configuration(format!("route references unknown target {response_name}"))
        })?;
        if !policy.prompts.contains_key(&response.id) {
            return Ok(policy);
        }
        let dependency = self.targets.get(dependency_name).ok_or_else(|| {
            RunnerError::configuration(format!("route references unknown target {dependency_name}"))
        })?;
        if response.id == dependency.id {
            return Err(RunnerError::configuration(format!(
                "route {route_name} cannot apply system_prompt to target {response_name}: model {} is also used by routing-only target {dependency_name}",
                response.id,
            )));
        }
        policy.routing_answer_target = Some(response.id.clone());
        Ok(policy)
    }

    fn fallback_base_url(&self) -> RunnerResult<Option<String>> {
        let Some(name) = &self.fallback_client else {
            return Ok(None);
        };
        let config = self.llm_clients.get(name).ok_or_else(|| {
            RunnerError::configuration(format!(
                "fallback_client references unknown llm client {name}"
            ))
        })?;
        Ok(Some(config.base_url.as_str().to_string()))
    }

    fn build_anthropic_auxiliary_target(
        &self,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<AuxiliaryTarget> {
        route
            .routing_target_names()
            .into_iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let target = self.build_auxiliary_target(
                    name,
                    clients,
                    AuxiliaryOperation::AnthropicCountTokens,
                )?;
                Some((count_tokens_priority(name, &target.model), index, target))
            })
            .min_by_key(|(priority, index, _)| (*priority, *index))
            .map(|(_, _, target)| target)
    }

    fn build_responses_auxiliary_target(
        &self,
        route: &RouteConfig,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
    ) -> Option<AuxiliaryTarget> {
        route.routing_target_names().into_iter().find_map(|name| {
            self.build_auxiliary_target(name, clients, AuxiliaryOperation::ResponsesInputTokens)
        })
    }

    fn build_auxiliary_target(
        &self,
        name: &str,
        clients: &BTreeMap<String, Arc<TranslatingLlmClient>>,
        operation: AuxiliaryOperation,
    ) -> Option<AuxiliaryTarget> {
        let target = self.targets.get(name)?;
        let client = clients.get(&target.llm_client)?;
        client
            .supports_auxiliary(&target.id, operation)
            .then(|| AuxiliaryTarget {
                model: target.id.clone(),
                client: client.clone(),
            })
    }
}

fn count_tokens_priority(target_name: &str, model_id: &ModelId) -> usize {
    let target_name = target_name.to_ascii_lowercase();
    let model_id = model_id.to_ascii_lowercase();
    ["opus", "sonnet", "haiku"]
        .iter()
        .position(|hint| target_name.contains(hint) || model_id.contains(hint))
        .unwrap_or(3)
}

/// Resolves a route's locality from the billing its targets' clients declare.
///
/// The answer is the fact the configuration states, not an inference from a name, a slug
/// or a base URL. A single leg whose client states no billing makes the whole answer
/// `Unknown`: one unstated leg is enough that "this never leaves the building" could be
/// false, and calling it external instead would be a guess presented as a fact.
fn route_locality(
    route: &RouteConfig,
    targets: &BTreeMap<String, TargetConfig>,
    clients: &BTreeMap<String, LlmClientConfig>,
) -> crate::route::Locality {
    use crate::route::Locality;

    let mut legs = 0usize;
    let mut saw_local = false;
    let mut saw_external = false;
    for name in route.served_target_names() {
        let Some(target) = targets.get(name) else {
            continue;
        };
        let Some(client) = clients.get(&target.llm_client) else {
            return Locality::Unknown;
        };
        legs += 1;
        // A declared locality is authoritative. Undeclared falls back to a `local`
        // billing, which states the same fact for an operator-hosted client; any other
        // undeclared combination stays unknown rather than being given a cost class it
        // did not claim, because a subscription credential is still an external provider.
        match client.locality {
            Some(crate::config::ClientLocality::Local) => saw_local = true,
            Some(crate::config::ClientLocality::External) => saw_external = true,
            None => match client.billing {
                Some(BillingKind::Local) => saw_local = true,
                Some(BillingKind::Metered | BillingKind::Subscription) | None => {
                    return Locality::Unknown;
                }
            },
        }
    }

    match (legs, saw_local, saw_external) {
        (0, _, _) => Locality::Unknown,
        (_, true, false) => Locality::Local,
        (_, false, true) => Locality::Upstream,
        (_, true, true) => Locality::Mixed,
        (_, false, false) => Locality::Unknown,
    }
}

/// Resolves each target model id to the `llm_client` credential that serves it.
///
/// Whole-route composition uses this identity to decide whether a credential-level
/// failure condemns the next child too: the wire error names a target slug, not the
/// credential, and two routes with different `llm_client` names are different
/// credentials even when they point at the same provider.
fn target_credentials(
    route: &RouteConfig,
    targets: &BTreeMap<String, TargetConfig>,
) -> std::collections::HashMap<ModelId, String> {
    route
        .served_target_names()
        .into_iter()
        .filter_map(|name| {
            targets
                .get(name)
                .map(|target| (target.id.clone(), target.llm_client.clone()))
        })
        .collect()
}

/// A client endpoint, parsed when the config loads rather than checked afterwards.
///
/// Holding a `HttpBaseUrl` is proof the value is an absolute HTTP(S) URL, so no
/// later stage has to re-check it or can forget to.
#[derive(Clone, Debug)]
struct HttpBaseUrl(reqwest::Url);

impl HttpBaseUrl {
    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl<'de> Deserialize<'de> for HttpBaseUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let url = reqwest::Url::parse(raw.trim()).map_err(|error| {
            serde::de::Error::custom(format!("base_url must be an absolute HTTP(S) URL: {error}"))
        })?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(serde::de::Error::custom(
                "base_url must be an absolute HTTP(S) URL",
            ));
        }
        Ok(Self(url))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmClientConfig {
    format: ClientFormat,
    base_url: HttpBaseUrl,
    api_key_env: Option<String>,
    #[serde(default)]
    forward_auth: bool,
    #[serde(default)]
    extra_headers: BTreeMap<String, String>,
    #[serde(default = "default_max_retries")]
    max_retries: u32,
    /// Cooldown after an exhausted transient completion failure. Defaults to 5 seconds; zero disables it.
    #[serde(default = "default_failure_cooldown_ms")]
    failure_cooldown_ms: u64,
    /// Deadline in milliseconds for all attempts and the complete response. Unset is unbounded.
    timeout_ms: Option<u64>,
    /// Deadline in milliseconds for one attempt, from send to the first decoded event.
    ///
    /// Bounds "this candidate is producing nothing": connect, response headers and the
    /// first event. It stops applying once the attempt returns, so it never truncates a
    /// stream that is already delivering. An expiry here is pre-commit and may advance
    /// to another candidate; unset means an attempt is bounded only by `timeout_ms`.
    #[serde(default)]
    attempt_timeout_ms: Option<u64>,
    /// Maximum gap in milliseconds between events once a stream has started.
    ///
    /// A slow but progressing stream is unaffected; a stream that stops producing ends
    /// with an in-band timeout. Unset means the gap is bounded only by `timeout_ms`.
    #[serde(default)]
    stream_idle_timeout_ms: Option<u64>,
    /// This upstream requires the Responses `input` to be a JSON array.
    ///
    /// The Codex backend rejects the compact bare-string form with `Input must be a list`.
    /// Declaring it keeps chat ingress working on that lane instead of narrowing the
    /// advertised inbound formats. Ignored by the Chat and Anthropic formats.
    #[serde(default)]
    requires_list_input: bool,
    /// How this client's usage is billed.
    ///
    /// Stated rather than inferred, because the difference decides whether a per-token
    /// price may be attached to the lane at all. An allowance-backed client is not
    /// billed per token, so pricing it as if it were would manufacture a charge the
    /// provider never made.
    #[serde(default)]
    billing: Option<BillingKind>,
    /// Where requests sent to this client are transported, as the deployment declares it.
    ///
    /// Deliberately separate from `billing`. That field classifies cost for the economics
    /// surface, and a subscription credential is still an external provider; deriving an
    /// egress claim from it would let an economics label decide whether a payload leaves
    /// the operator's infrastructure. Locality answers that second question on its own and
    /// stays undeclared when the deployment does not state it.
    #[serde(default)]
    locality: Option<ClientLocality>,
    /// The provider-side quota group every model served here draws on.
    ///
    /// A provider caps requests per *project*, and several clients can name the same
    /// project - a second base URL, a different wire format, a second model list. Naming
    /// the group here makes the counters meet: two clients declaring the same group share
    /// one counter per model instead of each believing it owns the whole daily budget.
    /// Defaults to the client's own name, which keeps a single-client deployment exactly
    /// as it was.
    #[serde(default)]
    quota_group: Option<String>,
    /// Whether this client's daily cap is stated per model or for the whole account.
    ///
    /// Model scope is the historical behaviour and stays the default, because it is
    /// correct for a provider that budgets each model separately (Google's free tier).
    /// Account scope exists for providers that publish one free-request allowance for the
    /// whole account (OpenRouter): under model scope every free model would be given the
    /// full daily budget as its own, and the fleet would collectively spend several times
    /// the real cap before any counter reported exhaustion.
    #[serde(default)]
    quota_scope: Option<QuotaScope>,
    /// Which midnight this client's daily counter turns over on.
    ///
    /// Pacific by default, which is the historical behaviour and is correct for Google's
    /// AI Studio free tier. OpenRouter's free-model counter turns over at midnight UTC;
    /// counting it on the Pacific clock would either spend the new day's budget hours
    /// early or refuse traffic the provider has already restored.
    #[serde(default)]
    quota_reset: Option<QuotaReset>,
    /// The provider-reported request pool this client's counter reconciles against.
    ///
    /// A local counter sees only this deployment's traffic. On a shared free tier another
    /// client -- or the same person in a browser -- can spend the account's allowance while
    /// our counter still shows room. Naming a pool here authorizes the economics collector
    /// to write the provider's own figure into this client's counter, so admission is
    /// decided on the account rather than on our own slice of it.
    ///
    /// Omitted means the local budget is the only authority, which is the historical
    /// behaviour and is correct for a provider that exposes no such reading.
    #[serde(default)]
    quota_provider: Option<QuotaProvider>,
}

/// A provider-reported request pool a counter may reconcile against.
///
/// A closed set rather than a free string: each variant names a pool whose wire shape and
/// reset semantics this code knows, so a configuration cannot ask for a mapping the
/// parser does not implement and then silently admit on a reading that never arrives.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum QuotaProvider {
    /// OpenRouter's account-wide free-model daily request counter, reported by
    /// `GET /api/v1/key` as `data.free_model_daily_requests`. It resets at midnight UTC.
    ///
    /// The spelling is pinned rather than derived from the variant name: serde's snake_case
    /// rule would render `OpenRouterFreeModels` as `open_router_free_models`, which is not
    /// how the provider's name is written anywhere else in this codebase and would become a
    /// second spelling for one thing.
    #[serde(rename = "openrouter_free_models")]
    OpenRouterFreeModels,
}

impl QuotaProvider {
    /// The pool name the provider itself uses.
    ///
    /// This must match the `pool` an account reader publishes, because reconciliation
    /// selects the reading by that string. The two are joined here rather than written
    /// out twice, so a rename cannot leave the reader and the config disagreeing.
    pub fn pool(self) -> &'static str {
        match self {
            QuotaProvider::OpenRouterFreeModels => "free_model_daily_requests",
        }
    }

    /// The stable spelling used in configuration and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            QuotaProvider::OpenRouterFreeModels => "openrouter_free_models",
        }
    }

    /// The day boundary the pool turns over on.
    ///
    /// Not configurable: the provider states when its own window resets, and counting it
    /// on a different clock would place a reading in a day the provider does not recognise.
    pub fn reset(self) -> QuotaReset {
        match self {
            QuotaProvider::OpenRouterFreeModels => QuotaReset::Utc,
        }
    }
}

/// How a client's daily request cap is scoped.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum QuotaScope {
    /// One counter per model within the quota group.
    #[default]
    Model,
    /// One counter for the whole quota group: every model draws on one account budget.
    Account,
}

/// Which midnight a client's daily counter turns over on.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum QuotaReset {
    /// Midnight Pacific — Google's AI Studio free tier.
    #[default]
    Pacific,
    /// Midnight UTC — OpenRouter's free-model daily counter.
    Utc,
}

impl QuotaReset {
    /// The stable spelling used in configuration and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            QuotaReset::Pacific => "pacific",
            QuotaReset::Utc => "utc",
        }
    }
}

/// Where a client's traffic terminates.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ClientLocality {
    /// The operator's own deployment: requests never leave operator infrastructure.
    Local,
    /// A third-party provider reached over the public network.
    External,
}

/// How a client's usage is billed.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BillingKind {
    /// A metered API key on a paid provider: billed per token.
    Metered,
    /// An allowance-bearing credential: the provider charges a subscription or a
    /// credit allowance, not a per-token amount.
    Subscription,
    /// A deployment the operator hosts: no external provider charge at all.
    Local,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetConfig {
    id: ModelId,
    llm_client: String,
    #[serde(default)]
    extra_body: BTreeMap<String, Value>,
    /// Top-level request fields dropped from the outbound body before `extra_body` is merged.
    /// For providers that reject an otherwise standard field.
    #[serde(default)]
    omit_body_fields: BTreeSet<String>,
    /// Reasoning-effort values this target's provider refuses outright.
    ///
    /// The effort vocabulary is not portable: Google's OpenAI-compatibility surface accepts
    /// `minimal`/`low`/`medium`/`high` and answers HTTP 400 `INVALID_ARGUMENT` to `none`
    /// (measured 2026-11-21 on `gemini-3.5-flash-lite`), and a no-thinking consumer sends
    /// exactly `none`. A value named here is removed at the target instead of being forwarded,
    /// which leaves the provider's own default. Unset forwards every value.
    #[serde(default)]
    unsupported_reasoning_effort: BTreeSet<String>,
    system_prompt: Option<String>,
    /// Reasoning effort forced on every request to this target, replacing the caller's value.
    /// Only meaningful on `openai_chat` and `openai_responses` clients.
    reasoning_effort: Option<String>,
    /// Authoritative reasoning choice for this target. Applied after the caller's fields and
    /// after `extra_body`, so a strict non-reasoning target cannot be re-enabled by a caller.
    reasoning_policy: Option<ReasoningPolicy>,
    /// How this target's provider expects reasoning control on the wire. Two targets behind
    /// one transport routinely differ here, which is why it is not an `llm_client` setting.
    #[serde(default)]
    reasoning_dialect: ReasoningDialect,
    /// The marker this target's upstream uses when it writes reasoning inline in the message.
    ///
    /// A per-target wire fact, like `reasoning_dialect`: Google's hosted Gemma endpoints
    /// answer `<thought>…</thought>` followed by the answer and count only the answer in
    /// `completion_tokens`, while the Gemini chat models on the same credential do not.
    /// Declaring `thought_tag` moves the block into a reasoning block so the monologue is
    /// never returned as output. An unrecognised value is a load error rather than a
    /// silently disabled filter.
    #[serde(default)]
    hosted_reasoning: Option<String>,
    /// Wire family this target speaks when serving a typed decision call.
    #[serde(default)]
    decision_transport: Option<decision_transport::DecisionTransport>,
    /// Question kinds this target currently answers. A request carrying any other kind
    /// excludes the target before dispatch rather than provoking a provider rejection.
    #[serde(default)]
    supported_types: Option<decision_transport::SupportedTypes>,
    /// State shapes this target accepts. Unset accepts every shape.
    #[serde(default)]
    state_forms: decision_transport::StateForms,
    /// Path for a typed decision call, when the backend does not use its default.
    #[serde(default)]
    decision_path: Option<String>,
    /// Base URL for a typed decision call, when the decision backend is not the
    /// completion endpoint this target's client points at.
    #[serde(default)]
    decision_base_url: Option<String>,
    /// Environment variable holding the decision backend's credential.
    #[serde(default)]
    decision_api_key_env: Option<String>,
    /// Name shared by every target that sits behind one upstream provider pool.
    ///
    /// Their failure cooldowns then move together: a pool-wide 429 is one fact
    /// about the pool, and immediately retrying a sibling model would spend a
    /// request the pool has already refused. Unset keeps the cooldown per model.
    #[serde(default)]
    cooldown_group: Option<String>,
    /// Factual capability of the backing deployment behind this target.
    ///
    /// A target fact, not a policy: it says what this model can do, while the route's own
    /// declaration says what the route promises to callers. The two are validated against
    /// each other at load, and the target facts are what ordered-candidate eligibility
    /// filters on before dispatch.
    #[serde(default)]
    vision: Option<bool>,
    /// Whether this deployment accepts video input.
    ///
    /// Separate from `vision` on purpose: an endpoint that accepts images need not accept
    /// video, and on this deployment it does not. Omitted means "cannot".
    #[serde(default)]
    video: Option<bool>,
    /// Which video carriers this deployment accepts, when it accepts video at all.
    ///
    /// Required whenever `video` is `true`. A provider can list video among a model's input
    /// modalities and still refuse to fetch a URL for it: the free Nemotron endpoint answers
    /// an inline `data:` video and 404s `video_url` with "No endpoints found that support
    /// video URLs". Declaring `video = true` without this field leaves the same ambiguity
    /// the boolean has, so the load refuses it rather than guessing a carrier.
    #[serde(default)]
    video_carrier: Option<MediaCarrier>,
    /// Which image carriers this deployment accepts, when it accepts images at all.
    ///
    /// The image counterpart of `video_carrier`, and optional for the same historical reason
    /// `vision` is: a deployment that predates carriers declares images and means both.
    #[serde(default)]
    image_carrier: Option<MediaCarrier>,
    #[serde(default)]
    tool_calling: Option<bool>,
    /// Whether this deployment accepts a request that replays an earlier tool call.
    ///
    /// `false` is a measured wire fact, not a policy: Google's OpenAI-compatibility surface
    /// issues a `thought_signature` with each function call and refuses a replayed
    /// `functionCall` part without it (HTTP 400 `INVALID_ARGUMENT`, measured 2026-11-26 for
    /// both a call that surface had just issued with the signature stripped and a call issued
    /// elsewhere). A caller that only offers tools is still served there, so this is narrower
    /// than `tool_calling` and does not replace it. Omitted permits replayed calls, which is
    /// what every manifest written before the field did.
    #[serde(default)]
    tool_call_replay: Option<bool>,
    /// Requests this target may make per provider day, when the provider caps them.
    ///
    /// Stated on the target rather than the client because the cap is per model within one
    /// project: two targets on one credential routinely have different daily budgets. A
    /// declared limit must be accompanied by a `[quota]` state directory; a spent
    /// allowance removes the target from consideration and the route advances.
    #[serde(default)]
    daily_request_limit: Option<u64>,
    /// Requests permitted in any rolling minute, when the provider publishes a window.
    ///
    /// A different resource from `daily_request_limit` and enforced separately: the provider
    /// counts requests per model per minute, and a daily budget reached by bursting through
    /// the minute is refused long before the day is spent. Declared per target for the same
    /// reason the daily budget is — the cap belongs to the model, not the client — and targets
    /// that name one physical model in one quota group must state the same value, because one
    /// window serves them all.
    #[serde(default)]
    requests_per_minute: Option<u64>,
    /// Input tokens permitted in any rolling minute, when the provider publishes a window.
    ///
    /// Tokens are estimated from the outbound body before the call, so this is pacing rather
    /// than billing: being slightly early is the safe direction, and the request is routed to
    /// a leg with headroom rather than retried. Requires `max_input_tokens` at or below this
    /// value — a request larger than a whole window's budget can never be admitted, and
    /// discovering that at the send path would spend eligibility on a 429.
    #[serde(default)]
    input_tokens_per_minute: Option<u64>,
    /// How much larger than its pre-dispatch estimate a request is assumed to be, in
    /// thousandths, when it is reserved against `input_tokens_per_minute`.
    ///
    /// The estimate is characters over the outbound body divided by four, which measured
    /// against the provider's own count has erred in both directions on real prompts. This
    /// is the protective pad that keeps a near-limit request from being dispatched on a
    /// number the provider will exceed: at 1300, a gemma request estimating 15,202 tokens
    /// reserves 19,763 and is refused against a 16,000 window instead of being sent and
    /// charged 18,054.
    ///
    /// Not a claim of accuracy. A value below 1000 is rejected at load, because a pad that
    /// makes the reservation smaller than the estimate is an under-count by construction.
    /// Omitted means 1000 — the estimate as stated — so a manifest that does not declare it
    /// keeps the behaviour it had.
    #[serde(default)]
    admission_margin_permille: Option<u32>,
    /// The physical model's own request window, when the deployment knows it.
    ///
    /// A route's `context_window` is what callers are promised; this is what one backing
    /// model can actually hold, input plus output. A route whose aggregate claim exceeds
    /// every candidate's physical window would promise capacity no target can serve, so
    /// load refuses the pair rather than discovering it as an upstream 400. Unset leaves
    /// the target unconstrained on this axis, which keeps every manifest written before the
    /// field existed behaving exactly as it did.
    #[serde(default)]
    context_window: Option<u64>,
    /// The most input a single request may carry here, when the provider caps it below the
    /// context window.
    ///
    /// This is the per-minute input budget expressed as a per-request ceiling: a request
    /// that needs more than a whole minute's allowance can never be served, so admitting it
    /// would spend the target's eligibility and return a 429. The free hosted Gemma models
    /// are the case that forced the field — 262,144-token window, 16,000 input tokens per
    /// minute — and a 256K prompt is not a 16K request with a long wait.
    #[serde(default)]
    max_input_tokens: Option<u64>,
    /// Whether the provider charges nothing for a request here inside a declared allowance.
    ///
    /// Whether a request here draws on a free allowance, when the deployment states it.
    ///
    /// Tri-state, because two answers are not enough to make an economic claim. `true` says
    /// the request is free, `false` says it is billed, and absent says nothing at all - which
    /// is [`switchyard_llm_client::Tariff::Unknown`] and never counts as free entitlement.
    /// A boolean could not tell "this is metered" apart from "nobody declared this", and the
    /// second is exactly the case a free-only selection pool must exclude.
    ///
    /// It is also a selection fact: no measured latency moves a paid leg in front of an
    /// eligible free leg, and no free leg is entered ahead of a cheaper tier by accident.
    #[serde(default)]
    free_tier: Option<bool>,
    /// Economic/selection groups this target belongs to, as labels only.
    ///
    /// A group is a membership tag, never an execution stage: nothing dispatches *to* a
    /// group, and no group is preferred over another. It exists so one route can declare a
    /// pool by membership (`pool_groups`) and so a catalogue can describe which leaves belong
    /// to `ST_OR_Free`, `ST_Gemini_Free` or `ST_OR_Paid` without the route's candidate list
    /// being the only record of it.
    #[serde(default)]
    groups: Vec<String>,
    /// Whether the backing model accepts a caller's explicit reasoning control.
    ///
    /// `false` is a measured wire fact, not a policy: Google's hosted Gemma answers HTTP
    /// 400 to `reasoning_effort`, while the Gemini chat models on the same credential accept
    /// it. A target that states `false` is skipped for a request carrying any reasoning
    /// control and stays eligible for one carrying none, so its native reasoning default is
    /// never overridden. Unset keeps the previous behaviour, where control is permitted.
    #[serde(default)]
    accepts_reasoning_control: Option<bool>,
    /// The smallest output reservation this target can produce an answer within.
    ///
    /// Set only for a target whose reasoning is charged against the caller's
    /// `max_output_tokens` — Google's hosted Gemma writes `<thought>` inline and stops when
    /// the budget runs out mid-thought, so at a small reservation it answers with thinking
    /// only (measured 2026-11-21: 61–328 thinking tokens for prompts of 49–438 tokens, and
    /// zero visible text once the budget fell below the thought). A request whose output
    /// reservation is below this floor is not dispatched here: the route keeps the candidate
    /// list honest and spends the provider request on a leg that can answer. Unset means no
    /// floor, which is every target outside this shape.
    #[serde(default)]
    min_output_tokens: Option<u64>,
    /// Why this target is held out of service, stated as the provider's own evidence.
    ///
    /// A held target keeps its place in the inventory — the declaration, its credential and
    /// its facts stay readable — but it is removed from every route's candidate set at load,
    /// so no request is ever dispatched to it and no `/v1/models` entry advertises it. It is
    /// not retired: retiring would delete the record of what was measured and would make the
    /// alias indistinguishable from a typo.
    ///
    /// The value is the reason the caller is shown, so it must be a fact rather than a
    /// judgement: for a model the provider has withdrawn, the provider's own sentence. Held
    /// is deliberately not a fallback: a held target must never be replaced by a live model,
    /// because the caller asked for this one and a silent substitution would answer a
    /// different question.
    #[serde(default)]
    held: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
enum ClientFormat {
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

impl ClientFormat {
    const fn wire_format(self) -> WireFormat {
        match self {
            Self::OpenAiChat => WireFormat::OpenAiChat,
            Self::OpenAiResponses => WireFormat::OpenAiResponses,
            Self::AnthropicMessages => WireFormat::AnthropicMessages,
        }
    }

    const fn caller_auth_kind(self) -> CallerAuthKind {
        match self {
            Self::AnthropicMessages => CallerAuthKind::Anthropic,
            Self::OpenAiChat | Self::OpenAiResponses => CallerAuthKind::OpenAi,
        }
    }
}

/// Resolves one scope's configured target names to the models the driver serves.
fn resolve_category_models(
    names: HashMap<Category, Vec<String>>,
    targets: &BTreeMap<String, ModelId>,
) -> RunnerResult<HashMap<Category, Vec<ModelId>>> {
    names
        .into_iter()
        .map(|(category, names)| {
            let models = names
                .into_iter()
                .map(|name| {
                    targets.get(&name).cloned().ok_or_else(|| {
                        RunnerError::configuration(format!(
                            "route references unknown target {name}"
                        ))
                    })
                })
                .collect::<RunnerResult<Vec<_>>>()?;
            Ok((category, models))
        })
        .collect()
}

// The per-target overrides are all forwarded verbatim from the target being built;
// bundling them into a struct would only move the same fields one level down.
/// The shared minute-window budget one target draws on, when it declares limits.
///
/// The key is the provider's own identity — quota group plus physical model — not the
/// target name or the route alias, because that is what the provider counts. Every alias
/// therefore draws on one ledger, which is the whole point: three names for one model must
/// not each be handed the same minute.
fn pacing_for(
    target: &TargetConfig,
    client_config: &LlmClientConfig,
    governor: &Arc<switchyard_llm_client::QuotaGovernor>,
) -> Option<Arc<switchyard_llm_client::PacedQuota>> {
    let limits = switchyard_llm_client::WindowLimits {
        requests_per_minute: target.requests_per_minute,
        input_tokens_per_minute: target.input_tokens_per_minute,
    };
    if limits.is_unlimited() {
        return None;
    }
    let group = client_config
        .quota_group
        .clone()
        .unwrap_or_else(|| target.llm_client.clone());
    Some(Arc::new(switchyard_llm_client::PacedQuota::new(
        Arc::clone(governor),
        switchyard_llm_client::GovernorKey::new(group, target.id.clone(), "generate"),
        limits,
        // Validation already rejected a pad below 1.0, so the fallback here is the
        // no-pad default rather than an error: this function builds, it does not judge.
        admission_margin_opt(target),
    )))
}

/// The configured admission pad, or no pad when the target declares none.
fn admission_margin_opt(target: &TargetConfig) -> switchyard_llm_client::AdmissionMargin {
    target
        .admission_margin_permille
        .and_then(switchyard_llm_client::AdmissionMargin::from_permille)
        .unwrap_or(switchyard_llm_client::AdmissionMargin::NONE)
}

/// The most estimated input a target will admit, mirroring the send path's arithmetic.
///
/// `None` when the target declares no minute token window. Otherwise the smaller of the
/// declared per-request ceiling and the largest estimate a margin-padded reservation can
/// fit. A target that declares a window but no ceiling is already a load error, so the
/// `max_input_tokens` arm is only for completeness.
fn safe_input_tokens(target: &TargetConfig) -> Option<u64> {
    let limits = switchyard_llm_client::WindowLimits {
        requests_per_minute: target.requests_per_minute,
        input_tokens_per_minute: target.input_tokens_per_minute,
    };
    let window = limits.safe_input_ceiling(admission_margin_opt(target))?;
    Some(match target.max_input_tokens {
        Some(ceiling) => ceiling.min(window),
        None => window,
    })
}

/// The admission pad, with load-time validation of the declared value.
fn admission_margin(
    target: &TargetConfig,
    target_name: &str,
) -> RunnerResult<switchyard_llm_client::AdmissionMargin> {
    let Some(declared) = target.admission_margin_permille else {
        return Ok(switchyard_llm_client::AdmissionMargin::NONE);
    };
    if target.input_tokens_per_minute.is_none() {
        return Err(RunnerError::configuration(format!(
            "target {target_name} declares admission_margin_permille {declared} without \
             input_tokens_per_minute; the margin pads a token reservation, and there is no \
             token window here to pad, so it would protect nothing"
        )));
    }
    switchyard_llm_client::AdmissionMargin::from_permille(declared).ok_or_else(|| {
        RunnerError::configuration(format!(
            "target {target_name} declares admission_margin_permille {declared}; a margin below \
             1000 would reserve less than the estimate, which is an under-count by construction"
        ))
    })
}
#[allow(clippy::too_many_arguments)]
fn build_backend(
    client_name: &str,
    config: &LlmClientConfig,
    extra_body: &BTreeMap<String, Value>,
    omit_body_fields: &BTreeSet<String>,
    unsupported_reasoning_effort: &BTreeSet<String>,
    reasoning_effort: Option<String>,
    reasoning_policy: Option<ReasoningPolicy>,
    reasoning_dialect: ReasoningDialect,
    cooldown_group: Option<String>,
    allowance: Option<Arc<switchyard_llm_client::DailyRequestAllowance>>,
    pacing: Option<Arc<switchyard_llm_client::PacedQuota>>,
    hosted_reasoning: Option<HostedReasoningMarker>,
) -> RunnerResult<Backend> {
    if config.max_retries > MAX_CONFIGURED_RETRIES {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} max_retries must be at most {MAX_CONFIGURED_RETRIES}"
        )));
    }
    if config.timeout_ms == Some(0) {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} timeout_ms must be at least 1"
        )));
    }
    for (label, value) in [
        ("attempt_timeout_ms", config.attempt_timeout_ms),
        ("stream_idle_timeout_ms", config.stream_idle_timeout_ms),
    ] {
        if value == Some(0) {
            return Err(RunnerError::configuration(format!(
                "llm client {client_name} {label} must be at least 1"
            )));
        }
    }
    // Naming both makes the relationship explicit here rather than at the call site:
    // an attempt bound wider than the overall bound can never take effect.
    if let (Some(attempt), Some(overall)) = (config.attempt_timeout_ms, config.timeout_ms)
        && attempt > overall
    {
        tracing::warn!(
            llm_client = client_name,
            attempt_timeout_ms = attempt,
            timeout_ms = overall,
            "attempt_timeout_ms exceeds timeout_ms; the overall deadline wins"
        );
    }
    if config.forward_auth && config.api_key_env.is_some() {
        return Err(RunnerError::configuration(format!(
            "llm client {client_name} cannot set both forward_auth and api_key_env"
        )));
    }
    let api_key = config
        .api_key_env
        .as_deref()
        .map(|variable| {
            if variable.trim().is_empty() {
                return Err(RunnerError::configuration(format!(
                    "llm client {client_name} api_key_env must not be empty"
                )));
            }
            let api_key = std::env::var(variable).map_err(|error| {
                RunnerError::configuration(format!(
                    "llm client {client_name} could not read api_key_env {variable}: {error}"
                ))
            })?;
            if api_key.trim().is_empty() {
                return Err(RunnerError::configuration(format!(
                    "llm client {client_name} api_key_env {variable} is empty"
                )));
            }
            Ok(api_key)
        })
        .transpose()?;
    let http = HttpBackendConfig {
        base_url: config.base_url.as_str().to_string(),
        api_key,
        forward_auth: config.forward_auth,
        extra_headers: config.extra_headers.clone(),
        extra_body: extra_body.clone(),
        omit_body_fields: omit_body_fields.clone(),
        unsupported_reasoning_effort: unsupported_reasoning_effort.clone(),
        reasoning_effort,
        reasoning_policy,
        reasoning_dialect,
        max_retries: config.max_retries,
        failure_cooldown: Duration::from_millis(config.failure_cooldown_ms),
        cooldown_group,
        timeout: config.timeout_ms.map(Duration::from_millis),
        attempt_timeout: config.attempt_timeout_ms.map(Duration::from_millis),
        stream_idle_timeout: config.stream_idle_timeout_ms.map(Duration::from_millis),
        requires_list_input: config.requires_list_input,
        allowance,
        pacing,
        hosted_reasoning,
    };
    let backend = match config.format {
        ClientFormat::OpenAiChat => Backend::OpenAiChat(http),
        ClientFormat::OpenAiResponses => Backend::OpenAiResponses(http),
        ClientFormat::AnthropicMessages => Backend::Anthropic(http),
    };
    Ok(backend)
}

// A function so that serde default can use it.
const fn default_max_retries() -> u32 {
    DEFAULT_MAX_RETRIES
}

const fn default_failure_cooldown_ms() -> u64 {
    5000
}

fn validate_value(label: &str, value: &str) -> RunnerResult<()> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(RunnerError::configuration(format!(
            "{label} must be non-empty and have no surrounding whitespace"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads a deployment that must be refused, returning the refusal text.
    ///
    /// `Runner` has no `Debug` impl, so `expect_err` cannot be used; matching keeps the
    /// failure honest - a successful load panics rather than being reported as a refusal.
    fn refuse(source: &str) -> String {
        match runner_from_toml(source) {
            Ok(_) => panic!("deployment loaded but should have been refused"),
            Err(error) => format!("{error}"),
        }
    }

    /// A target may state that its wire cannot accept a replayed tool call.
    ///
    /// The declaration has to be a consumed field on the target: an unconsumed key would be
    /// rejected by the strict schema, and a key that parses into nothing would be a knob with
    /// no effect. This asserts the load accepts it and that a misspelling does not.
    #[test]
    fn a_target_may_declare_that_it_cannot_replay_a_tool_call() {
        let accepted = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

[targets.gemini]
id = "gemini-lite"
llm_client = "google"
tool_calling = true
tool_call_replay = false

[routes.ladder]
id = "ladder"
type = "passthrough"
target = "gemini"
candidates = []
"#;
        runner_from_toml(accepted).expect("a declared replay refusal loads");

        // The same manifest with the key misspelt must fail, not silently ignore it.
        let misspelt = accepted.replace("tool_call_replay = false", "tool_call_replays = false");
        let message = refuse(&misspelt);
        assert!(
            message.contains("tool_call_replays") || message.to_lowercase().contains("unknown"),
            "a misspelt capability key must be refused by name, got: {message}"
        );
    }

    /// A deployment whose paced lane declares a margin but no token window refuses to load.
    ///
    /// The margin pads a token reservation. A target with no `input_tokens_per_minute` has
    /// nothing to pad, so the declaration states an intent the runtime cannot honour. This
    /// was reachable only through the helper: the paced-target check ran first and skipped
    /// unpaced targets before the margin was ever validated.
    #[test]
    fn a_margin_without_a_token_window_refuses_the_deployment() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.padded]
id = "gemma/padded"
llm_client = "google"
admission_margin_permille = 1300

[routes.padded]
id = "padded"
type = "passthrough"
target = "padded"
candidates = []
"#;
        let message = refuse(config);
        assert!(
            message.contains("without input_tokens_per_minute"),
            "refusal must name the missing window, got: {message}"
        );
        assert!(
            message.contains("protect nothing"),
            "refusal must say the margin protects nothing, got: {message}"
        );
    }

    /// A margin below parity reserves less than the estimate, which is an under-count by
    /// construction. It must be refused rather than silently clamped.
    #[test]
    fn a_margin_below_parity_refuses_the_deployment() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.padded]
id = "gemma/padded"
llm_client = "google"
requests_per_minute = 30
input_tokens_per_minute = 16000
admission_margin_permille = 900

[routes.padded]
id = "padded"
type = "passthrough"
target = "padded"
candidates = []
"#;
        let message = refuse(config);
        assert!(
            message.contains("below") && message.contains("1000"),
            "refusal must name the parity floor, got: {message}"
        );
    }

    /// The measured overrun, expressed as a manifest: a 16,000-token window padded by 1,300
    /// per mille can admit at most 12,307 estimated tokens, so a 16,000-token ceiling is a
    /// promise the governor would refuse on every large request.
    #[test]
    fn a_ceiling_above_the_padded_window_refuses_the_deployment() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.padded]
id = "gemma/padded"
llm_client = "google"
requests_per_minute = 30
input_tokens_per_minute = 16000
max_input_tokens = 16000
admission_margin_permille = 1300

[routes.padded]
id = "padded"
type = "passthrough"
target = "padded"
candidates = []
"#;
        let message = refuse(config);
        assert!(
            message.contains("12307"),
            "refusal must state the derived ceiling 12307, got: {message}"
        );
    }

    /// The honest form of the same lane loads: the ceiling is lowered to what the padded
    /// window can actually admit.
    #[test]
    fn a_padded_lane_loads_at_its_derived_ceiling() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.padded]
id = "gemma/padded"
llm_client = "google"
requests_per_minute = 30
input_tokens_per_minute = 16000
max_input_tokens = 12307
admission_margin_permille = 1300

[routes.padded]
id = "padded"
type = "passthrough"
target = "padded"
candidates = []
"#;
        runner_from_toml(config).expect("the derived ceiling is admittable");
    }

    /// Locality is the client's own declaration, not a cost class read from billing.
    ///
    /// A subscription credential is still an external provider, so `billing` cannot answer
    /// this question. An undeclared client stays unknown rather than being given the
    /// answer its billing field implies.
    #[test]
    fn route_locality_comes_from_the_declared_transport() {
        let base = r#"
schema_version = 1

[llm_clients.ext]
format = "openai_responses"
base_url = "https://provider.test/v1"
locality = "external"

[targets.ext-a]
id = "vendor/ext-a"
llm_client = "ext"

[routes.remote]
id = "remote"
type = "passthrough"
target = "ext-a"
candidates = []

[llm_clients.mine]
format = "openai_responses"
base_url = "http://host.test:8446/v1"
billing = "local"

[targets.mine-a]
id = "vendor/mine-a"
llm_client = "mine"

[routes.here]
id = "here"
type = "passthrough"
target = "mine-a"
candidates = []

[llm_clients.undeclared]
format = "openai_responses"
base_url = "https://other.test/v1"
billing = "subscription"

[targets.undeclared-a]
id = "vendor/undeclared-a"
llm_client = "undeclared"

[routes.unknown]
id = "unknown"
type = "passthrough"
target = "undeclared-a"
candidates = []
"#;
        let runner = crate::Runner::from_toml(base).expect("deployment builds");
        let by_id = runner
            .models()
            .map(|model| (model.id.as_str().to_string(), model.locality))
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(by_id["remote"], crate::route::Locality::Upstream);
        assert_eq!(by_id["here"], crate::route::Locality::Local);
        assert_eq!(
            by_id["unknown"],
            crate::route::Locality::Unknown,
            "a subscription credential alone states no locality"
        );
    }

    /// A target declares the marker its upstream uses, or the deployment does not load.
    ///
    /// Accepting an unknown value would disable the filter silently, which is the defect
    /// the filter exists to prevent, so the refusal must name the field and the accepted
    /// value.
    #[test]
    fn a_target_names_its_hosted_reasoning_marker() {
        let deployment = |marker: &str| {
            format!(
                r#"
schema_version = 1

[llm_clients.gemma]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma-a]
id = "google/gemma"
llm_client = "gemma"
hosted_reasoning = "{marker}"

[routes.chat]
id = "chat"
type = "passthrough"
target = "gemma-a"
candidates = []
"#
            )
        };

        crate::Runner::from_toml(&deployment("thought_tag")).expect("a known marker loads");

        let Err(error) = crate::Runner::from_toml(&deployment("thought")) else {
            panic!("an unknown marker must not load");
        };
        let message = error.to_string();
        assert!(
            message.contains("hosted_reasoning") && message.contains("thought_tag"),
            "the refusal must name the field and the accepted value: {message}"
        );
    }

    #[test]
    fn a_responses_client_cannot_be_pointed_at_google() {
        // Google serves /v1beta/openai/chat/completions and no Responses path. A target built
        // against that base with the Responses format would request an endpoint that does not
        // exist, and a 404 from a missing path is not a retired model. Refused at load.
        let deployment = |format: &str, base_url: &str| {
            format!(
                r#"
schema_version = 1

[llm_clients.google]
format = "{format}"
base_url = "{base_url}"

[targets.gemini]
id = "gemini-3.5-flash-lite"
llm_client = "google"

[routes.chat]
id = "chat"
type = "passthrough"
target = "gemini"
candidates = []
"#
            )
        };

        crate::Runner::from_toml(&deployment(
            "openai_chat",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ))
        .expect("the documented Chat Completions path loads");

        let Err(error) = crate::Runner::from_toml(&deployment(
            "openai_responses",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )) else {
            panic!("a Responses client against Google must not load");
        };
        let message = error.to_string();
        assert!(
            message.contains("openai_chat") && message.contains("generativelanguage"),
            "the refusal must name the fix: {message}"
        );

        // Control: the format itself is still supported for providers that serve it.
        crate::Runner::from_toml(&deployment("openai_responses", "https://api.openai.com/v1"))
            .expect("a Responses client against a Responses provider loads");
    }

    #[test]
    fn route_presentation_fields_are_split_from_the_algorithm() {
        let route: RouteConfig = toml::from_str(
            r#"
type = "random"
id = "switchyard/random"
context_window = 128000
tool_calling = true
reasoning = false
targets = ["fast", "strong"]
weights = [1.0, 2.0]
seed = 7
"#,
        )
        .expect("route should deserialize");

        assert_eq!(route.id, "switchyard/random");
        assert_eq!(route.context_window, Some(128_000));
        assert_eq!(route.tool_calling, Some(true));
        assert_eq!(route.reasoning, Some(false));
        assert_eq!(route.algorithm.routing_target_names(), ["fast", "strong"]);
    }

    #[test]
    fn route_algorithm_unknown_fields_are_rejected() {
        let error = toml::from_str::<RouteConfig>(
            r#"
type = "passthrough"
id = "switchyard/fast"
target = "fast"
bogus = true
"#,
        )
        .expect_err("unknown route keys must be rejected");

        assert!(error.to_string().contains("bogus"), "{error}");
    }

    #[test]
    fn deployment_errors_keep_one_path_context() {
        let path = Path::new("/definitely/missing/switchyard-routes.toml");
        let error = match load_runner(path) {
            Ok(_) => panic!("missing config should fail"),
            Err(error) => error,
        };
        let message = error.to_string();

        assert_eq!(message.matches(&path.display().to_string()).count(), 1);
        assert!(message.starts_with("failed to read server config"));
    }
}
#[cfg(test)]
mod deployment_tests {
    use super::*;
    use serde_json::json;

    const VALID_CONFIG: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.classifier]
id = "classifier/model"
llm_client = "primary"

[targets.strong]
id = "strong/model"
llm_client = "responses"

[targets.weak]
id = "weak/model"
llm_client = "anthropic"

[routes.noop]
id = "switchyard/noop"
type = "noop"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["strong", "weak"]

[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "weak"

[routes.plan_execute]
id = "switchyard/plan-execute"
type = "plan_execute"
capable_target = "strong"
efficient_target = "weak"
"#;

    /// The provider's own sentence, repeated verbatim in the refusal.
    const HELD_REASON: &str = "This model is no longer available to new users.";

    /// A deployment with one held target reached three ways: as a route's whole candidate
    /// set, as an extra candidate, and as the head target with a live fallback behind it.
    ///
    /// The live target is declared once so the three routes differ only in where the held
    /// one sits — that is the variable these tests are about.
    const HELD_CONFIG: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.live]
id = "live/model"
llm_client = "primary"

[targets.retired]
id = "retired/model"
llm_client = "primary"
held = "This model is no longer available to new users."

[routes.held_only]
id = "switchyard/held-only"
type = "passthrough"
target = "retired"

[routes.held_as_candidate]
id = "switchyard/held-as-candidate"
type = "passthrough"
target = "live"
candidates = ["retired"]

[routes.held_as_head]
id = "switchyard/held-as-head"
type = "passthrough"
target = "retired"
candidates = ["live"]

[routes.live_only]
id = "switchyard/live-only"
type = "passthrough"
target = "live"
"#;

    /// A held target must not fail the load.
    ///
    /// Holding is a service decision about a model that exists in the manifest, so the
    /// declaration, its credential and its facts all still have to parse. Refusing the
    /// deployment here would turn a deliberate hold into an outage.
    #[test]
    fn a_held_target_still_loads_and_keeps_its_record() -> RunnerResult<()> {
        let runner = runner_from_toml(HELD_CONFIG)?;
        let route = runner
            .route("switchyard/held-only")
            .expect("a held route keeps its registration");
        assert_eq!(route.held_reason(), Some(HELD_REASON));
        assert!(
            route.models().models_for(&Category::Any).is_empty(),
            "no candidate may remain when every target is held"
        );
        Ok(())
    }

    /// A held target leaves the candidate list without condemning the route that named it.
    ///
    /// Both positions are covered because they are filtered by the same set but reached by
    /// different code: a candidate is one entry in an ordered list, while the head target is
    /// the entry the algorithm would otherwise always offer first.
    #[test]
    fn a_held_candidate_leaves_the_list_without_holding_the_route() -> RunnerResult<()> {
        let runner = runner_from_toml(HELD_CONFIG)?;
        for name in ["switchyard/held-as-candidate", "switchyard/held-as-head"] {
            let route = runner.route(name).expect("route is registered");
            assert_eq!(
                route.held_reason(),
                None,
                "{name} still has a live target and must keep serving"
            );
            assert_eq!(
                route.models().models_for(&Category::Any),
                vec![ModelId::from("live/model")],
                "{name} must offer only the live target"
            );
        }
        Ok(())
    }

    /// The catalogue promises answers; the inventory records what is declared.
    ///
    /// Both are asserted together because either one alone would pass under the wrong
    /// behaviour: dropping the route from both lists would look like a correct retirement,
    /// and leaving it in both would look like an untouched inventory.
    #[test]
    fn the_catalogue_drops_a_held_route_while_the_inventory_keeps_it() -> RunnerResult<()> {
        let runner = runner_from_toml(HELD_CONFIG)?;
        let advertised: Vec<&str> = runner
            .advertised_models()
            .map(|model| model.id.as_str())
            .collect();
        assert!(
            !advertised.contains(&"switchyard/held-only"),
            "a route that can only refuse must not be advertised: {advertised:?}"
        );
        for name in [
            "switchyard/held-as-candidate",
            "switchyard/held-as-head",
            "switchyard/live-only",
        ] {
            assert!(
                advertised.contains(&name),
                "{name} serves and must stay advertised: {advertised:?}"
            );
        }
        let inventory: Vec<&str> = runner.models().map(|model| model.id.as_str()).collect();
        assert!(
            inventory.contains(&"switchyard/held-only"),
            "the declaration stays readable for diagnostics: {inventory:?}"
        );
        Ok(())
    }

    /// The refusal names the held alias and quotes the provider, before any request work.
    ///
    /// Executed with a default request on purpose: a held route has nothing to dispatch, so
    /// the decision must not depend on the request being well formed. A refusal that had to
    /// read the payload could not be trusted to stay free of provider calls.
    #[tokio::test]
    async fn a_held_route_refuses_locally_with_the_providers_evidence() {
        let runner = runner_from_toml(HELD_CONFIG).expect("fixture loads");
        let route = runner
            .route("switchyard/held-only")
            .expect("route is registered");
        match route
            .execute(switchyard_protocol::Request::default(), None)
            .await
        {
            Err(RunnerError::Held { model, reason }) => {
                assert_eq!(model, "retired", "the refusal names the alias asked for");
                assert_eq!(reason, HELD_REASON, "the provider's evidence is forwarded");
            }
            Err(other) => panic!("expected Held, got {other:?}"),
            Ok(_) => panic!("a held route must never produce a response"),
        }
    }

    #[test]
    fn public_runner_from_toml_builds_a_deployment() -> RunnerResult<()> {
        let runner = Runner::from_toml(VALID_CONFIG)?;

        let classifier = runner
            .route("switchyard/classifier")
            .expect("classifier route should exist");
        let models = classifier.models();
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("classifier/model")]
        );
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("weak/model")]
        );
        assert_eq!(
            models.models_for(&Category::Capable),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("weak/model"), ModelId::from("strong/model")]
        );
        let plan_execute = runner
            .route("switchyard/plan-execute")
            .expect("plan-execute route should exist");
        assert_eq!(
            plan_execute.models().models_for(&Category::Capable),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            plan_execute.models().models_for(&Category::Efficient),
            [ModelId::from("weak/model")]
        );
        assert!(runner.route("switchyard/passthrough").is_some());
        Ok(())
    }

    #[test]
    fn duplicate_route_ids_are_rejected() {
        let config = format!(
            r#"{VALID_CONFIG}

[routes.duplicate]
id = "switchyard/passthrough"
type = "passthrough"
target = "strong"
"#
        );

        let error = error_message(&config);

        assert!(
            error.contains(
                "routes duplicate and passthrough both use id switchyard/passthrough; route ids must be unique"
            ),
            "{error}"
        );
    }

    fn error_message(toml: &str) -> String {
        match runner_from_toml(toml) {
            Ok(_) => "configuration unexpectedly succeeded".to_string(),
            Err(error) => error.to_string(),
        }
    }

    /// The parsed `AlgorithmSpec` for one route, so a test can assert what the loader
    /// hands to the runtime rather than only that the file parses.
    fn ordered_spec(toml: &str, route_name: &str) -> AlgorithmSpec {
        let config: DeploymentConfig = toml::from_str(toml).expect("fixture parses");
        config
            .routes
            .get(route_name)
            .expect("fixture declares the route")
            .algorithm
            .clone()
    }

    fn with_subagent_llm_classifier(config: &str, route: &str, extra: &str) -> String {
        let mut configured = config.to_string();
        configured.push_str(&format!("\n[routes.{route}.subagents]\n"));
        configured.push_str(
            r#"type = "llm_classifier"
mode = "custom"
models = { judge = ["classifier"], capable = ["strong"], efficient = ["weak"], any = ["strong", "weak"] }
default_target = "efficient"
prompt = "Select a target for this delegated task."
response_schema = '{"type":"object","properties":{"target":{"type":"string","enum":["capable","efficient"]}},"required":["target"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/target" }
classify_trigger = "new_session""#,
        );
        configured.push_str(extra);
        configured
    }

    #[test]
    fn subagent_models_stay_separate_from_the_parent_tiers() -> RunnerResult<()> {
        // The sub-agent target is also the parent's capable tier. Merged into one group it
        // would be indistinguishable from that tier, and delegated work would follow the
        // parent's ordering instead of its own configured target.
        let runner = runner_from_toml(&with_subagent_passthrough(&stage_config(), "stage"))?;
        let models = runner
            .route("switchyard/stage")
            .expect("stage route should exist")
            .models();

        assert_eq!(
            models.subagent_models_for(&Category::Any),
            [ModelId::from("strong/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("strong/model"), ModelId::from("weak/model")]
        );
        Ok(())
    }

    fn with_subagent_passthrough(config: &str, route: &str) -> String {
        format!("{config}\n[routes.{route}.subagents]\ntype = \"passthrough\"\ntarget = \"strong\"")
    }

    fn stage_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.stage_judge]
id = "stage-judge/model"
llm_client = "primary"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0
capable_hold_turns = 2

[routes.stage.tool_semantics]
observe = ["lookup_customer"]
mutate = ["send_payment"]
plan = ["create_workflow"]
new = ["send_message"]

[routes.stage.classifier]
target = "stage_judge"
base_threshold = 0.5
"#
        )
    }

    fn composite_config() -> String {
        format!(
            r#"{VALID_CONFIG}
[targets.tier_judge]
id = "tier-judge/model"
llm_client = "primary"

[routes.composed]
id = "switchyard/hier"
type = "composite"

[routes.composed.classifier]
target = "tier_judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.composed.stage]
capable_target = "strong"
efficient_target = "weak"
confidence_threshold = 0.5
capable_hold_turns = 2

[routes.composed.stage.tool_semantics]
new = ["send_message"]
"#
        )
    }

    #[test]
    fn composite_route_builds_and_claims_both_tiers_and_its_judge() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_config())?;
        assert!(
            runner
                .models()
                .any(|model| model.id.as_str() == "switchyard/hier")
        );
        Ok(())
    }

    /// Two leaves whose carriers differ, a text-only leaf, and composites over each.
    /// The composites state nothing about media: what they advertise must come from the
    /// children, except where a test deliberately states a policy cap.
    fn composite_media_config(extra: &str) -> String {
        format!(
            r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.inline]
id = "inline/model"
llm_client = "responses"
vision = true
image_carrier = "inline"
video = true
video_carrier = "inline"

[targets.both]
id = "both/model"
llm_client = "responses"
vision = true
video = true
video_carrier = "both"

[targets.text]
id = "text/model"
llm_client = "primary"

[routes.child_inline]
id = "child-inline"
type = "passthrough"
target = "inline"
vision = true
image_carrier = "inline"
video = true
video_carrier = "inline"

[routes.child_both]
id = "child-both"
type = "passthrough"
target = "both"
vision = true
video = true
video_carrier = "both"

[routes.child_text]
id = "child-text"
type = "passthrough"
target = "text"

[composites.tier_inline]
children = ["child_inline"]

[composites.tier_mixed]
children = ["child_inline", "child_both"]

[composites.tier_text]
children = ["child_text"]

[composites.tier_bounded]
children = ["child_inline"]

{extra}
"#
        )
    }

    fn composite_media(runner: &Runner, name: &str) -> crate::ModelCapabilities {
        runner
            .route_capabilities(name)
            .unwrap_or_else(|| panic!("composite {name} is not registered"))
    }

    #[test]
    fn a_composite_derives_video_and_carrier_from_its_only_child() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_media_config(""))?;
        let media = composite_media(&runner, "tier_inline");
        assert_eq!(media.video, Some(true));
        assert_eq!(media.video_carrier, Some(MediaCarrier::Inline));
        assert_eq!(media.vision, Some(true));
        assert_eq!(media.image_carrier, Some(MediaCarrier::Inline));
        Ok(())
    }

    #[test]
    fn a_mixed_composite_derives_the_union_of_child_carriers() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_media_config(""))?;
        let media = composite_media(&runner, "tier_mixed");
        // Inline + Both is Both: a request on either carrier has a child that serves it.
        assert_eq!(media.video_carrier, Some(MediaCarrier::Both));
        assert_eq!(media.image_carrier, Some(MediaCarrier::Both));
        Ok(())
    }

    #[test]
    fn a_composite_over_text_only_children_claims_no_video() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_media_config(""))?;
        let media = composite_media(&runner, "tier_text");
        // Derived, never invented: no child declares video, so the composite refuses it
        // locally instead of offering a medium every child would skip.
        assert_eq!(media.video, Some(false));
        assert_eq!(media.video_carrier, None);
        Ok(())
    }

    #[test]
    fn the_bounded_lane_derives_independently_from_its_own_child() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_media_config(""))?;
        // A bounded composite's children are different routes; its declaration must be read
        // from those children, not inherited from the unbounded twin.
        let media = composite_media(&runner, "tier_bounded");
        assert_eq!(media.video, Some(true));
        assert_eq!(media.video_carrier, Some(MediaCarrier::Inline));
        Ok(())
    }

    #[test]
    fn an_explicit_video_false_stays_a_policy_cap() -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_media_config(
            "[composites.capped]\nchildren = [\"child_both\"]\nvideo = false\n",
        ))?;
        let media = composite_media(&runner, "capped");
        assert_eq!(media.video, Some(false));
        assert_eq!(media.video_carrier, None);
        Ok(())
    }

    #[test]
    fn video_true_without_a_carrier_derives_the_union_not_the_permissive_default()
    -> RunnerResult<()> {
        let runner = runner_from_toml(&composite_media_config(
            "[composites.stated]\nchildren = [\"child_inline\"]\nvideo = true\n",
        ))?;
        let media = composite_media(&runner, "stated");
        // The old default would have been Both, advertising a remote carrier that no child
        // below the composite serves.
        assert_eq!(media.video_carrier, Some(MediaCarrier::Inline));
        Ok(())
    }

    #[test]
    fn a_composite_cannot_advertise_a_carrier_no_child_serves() {
        let error = runner_from_toml(&composite_media_config(
            "[composites.overclaim]\nchildren = [\"child_inline\"]\nvideo = true\n\
             video_carrier = \"both\"\n",
        ))
        .err()
        .expect("an explicit both claim over inline-only children must be refused");
        let text = error.to_string();
        assert!(text.contains("overclaim"), "{text}");
        assert!(text.contains("Remote"), "{text}");
    }

    #[test]
    fn a_composite_cannot_state_a_carrier_without_the_medium_it_narrows() {
        let error = runner_from_toml(&composite_media_config(
            "[composites.inert]\nchildren = [\"child_inline\"]\nvideo = false\n\
             video_carrier = \"inline\"\n",
        ))
        .err()
        .expect("a carrier on a composite that refuses the medium must be refused");
        let text = error.to_string();
        assert!(text.contains("inert"), "{text}");
    }

    #[test]
    fn a_stated_carrier_narrows_a_derived_medium() {
        // The medium is not stated; it is derived from the one inline-only child. Narrowing
        // it to what that child serves is not inert, so it must load.
        runner_from_toml(&composite_media_config(
            "[composites.narrowed]\nchildren = [\"child_inline\"]\n\
             video_carrier = \"inline\"\n",
        ))
        .expect("a carrier matching the derived capability must load");

        // The same shape with a carrier no child serves must still be refused, naming the
        // carrier that is missing rather than the (unstated) medium.
        let error = runner_from_toml(&composite_media_config(
            "[composites.widened]\nchildren = [\"child_inline\"]\n\
             video_carrier = \"both\"\n",
        ))
        .err()
        .expect("a derived medium cannot be widened past its children");
        let text = error.to_string();
        assert!(text.contains("widened"), "{text}");
        assert!(text.contains("Remote"), "{text}");
    }

    #[test]
    fn a_composite_claiming_video_over_text_only_children_is_refused() {
        let error = runner_from_toml(&composite_media_config(
            "[composites.hollow]\nchildren = [\"child_text\"]\nvideo = true\n",
        ))
        .err()
        .expect("a video claim no child can keep must be refused");
        let text = error.to_string();
        assert!(text.contains("hollow"), "{text}");
    }

    #[test]
    fn builds_all_supported_algorithm_types() -> RunnerResult<()> {
        let state = runner_from_toml(VALID_CONFIG)?;
        // The model id array is sorted alphabetically
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            [
                "switchyard/classifier",
                "switchyard/noop",
                "switchyard/passthrough",
                "switchyard/plan-execute",
                "switchyard/random",
            ]
        );
        Ok(())
    }

    #[test]
    fn stage_rejects_ambiguous_or_non_additive_tool_semantics() {
        for (configured, expected) in [
            (
                stage_config().replace(
                    "mutate = [\"send_payment\"]",
                    "mutate = [\"LOOKUP_CUSTOMER\"]",
                ),
                "appears in both tool_semantics.observe and tool_semantics.mutate",
            ),
            (
                stage_config().replace("new = [\"send_message\"]", "new = [\"Read\"]"),
                "already has built-in semantics and cannot be reclassified",
            ),
            (
                stage_config().replace("observe = [\"lookup_customer\"]", "observe = [\" \"]"),
                "contains an empty tool name",
            ),
        ] {
            let message = error_message(&configured);
            assert!(
                message.contains(expected),
                "expected {expected:?} in error, got: {message}"
            );
        }
    }

    #[test]
    fn passthrough_and_stage_accept_subagent_routing() -> RunnerResult<()> {
        let stage = stage_config();
        let stage_with_classifier = with_subagent_llm_classifier(&stage, "stage", "");
        let parsed: DeploymentConfig = toml::from_str(&stage_with_classifier).map_err(|error| {
            RunnerError::configuration(format!("failed to parse stage config: {error}"))
        })?;
        let Some(stage_route) = parsed.routes.get("stage") else {
            return Err(RunnerError::configuration("stage route is missing"));
        };
        let callable_targets = stage_route.callable_target_names();
        for expected in ["strong", "weak", "stage_judge", "classifier"] {
            assert!(callable_targets.contains(&expected));
        }

        for configured in [
            with_subagent_llm_classifier(VALID_CONFIG, "passthrough", ""),
            with_subagent_passthrough(VALID_CONFIG, "passthrough"),
            stage_with_classifier,
            with_subagent_passthrough(&stage, "stage"),
        ] {
            runner_from_toml(&configured)?;
        }
        Ok(())
    }

    #[test]
    fn aliased_completion_targets_reject_prompt_conflicts() {
        let configured = stage_config()
            .replace(
                "id = \"strong/model\"\nllm_client = \"responses\"",
                "id = \"strong/model\"\nllm_client = \"responses\"\nsystem_prompt = \"capable\"",
            )
            .replace(
                "[routes.stage]",
                "[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\"\n\n[routes.stage]",
            )
            .replace("efficient_target = \"weak\"", "efficient_target = \"strong_alias\"");
        let message = error_message(&configured);
        assert!(
            message.contains("completion target aliases to model strong/model with different system_prompt values"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn prompted_routing_response_cannot_share_a_model_with_a_dependency() {
        let configured = VALID_CONFIG
            .replace(
                "id = \"classifier/model\"\nllm_client = \"primary\"",
                "id = \"weak/model\"\nllm_client = \"primary\"",
            )
            .replace(
                "id = \"weak/model\"\nllm_client = \"anthropic\"",
                "id = \"weak/model\"\nllm_client = \"anthropic\"\nsystem_prompt = \"answer prompt\"",
            )
            .replace(
                "base_threshold = 0.5",
                "base_threshold = 0.5\nescalation = { confirmations = 1 }",
            );

        let message = error_message(&configured);

        assert!(
            message.contains("cannot apply system_prompt to target weak: model weak/model is also used by routing-only target classifier"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn rejects_invalid_unreferenced_llm_client() {
        for (settings, expected) in [
            (
                "base_url = \"not a url\"",
                "base_url must be an absolute HTTP(S) URL",
            ),
            (
                "base_url = \"https://example.test\"\ntimeout_ms = 0",
                "timeout_ms must be at least 1",
            ),
        ] {
            let invalid = format!(
                "{VALID_CONFIG}\n[llm_clients.unused]\nformat = \"openai_chat\"\n{settings}"
            );
            let message = error_message(&invalid);
            assert!(message.contains(expected), "unexpected error: {message}");
        }
    }

    #[test]
    fn an_escalation_table_switches_the_classifier_route_to_escalation() -> RunnerResult<()> {
        // Present: the classifier target judges the weak tier's reply each turn instead of
        // picking a tier ahead of it. The route builds either way, so the assertion is that
        // the knob parses and its settings reach the algorithm's validation.
        let escalating = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalating)?;

        let reversible = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 2, deescalation = { strong_min_calls = 3, confirmations = 2, strong_max_calls = 6, weak_cooldown_calls = 8 } }",
        );
        runner_from_toml(&reversible)?;

        // A setting that would starve the judge is rejected here rather than on the first
        // request, the same as any other unusable route configuration.
        let starved = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { confirmations = 0 }",
        );
        assert!(error_message(&starved).contains("confirmations must be at least 1"));

        Ok(())
    }

    #[test]
    fn a_target_reasoning_effort_parses_and_is_rejected_where_unsupported() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let weak = "[targets.weak]\nid = \"weak/model\"\nllm_client = \"anthropic\"";
        assert!(VALID_CONFIG.contains(strong) && VALID_CONFIG.contains(weak));

        let forced = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \"max\""));
        runner_from_toml(&forced)?;

        let blank = VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_effort = \" \""));
        assert!(error_message(&blank).contains("reasoning_effort must not be empty"));

        let anthropic = VALID_CONFIG.replace(weak, &format!("{weak}\nreasoning_effort = \"high\""));
        assert!(
            error_message(&anthropic)
                .contains("only supported on openai_chat and openai_responses")
        );
        Ok(())
    }

    /// The reason dialect lives on the target: two targets behind one transport can pin
    /// different reasoning dialects without duplicating the `llm_client`.
    #[test]
    fn a_target_reasoning_policy_parses_and_carries_its_own_dialect() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        assert!(VALID_CONFIG.contains(strong));

        let strict = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\nreasoning_policy = \"disabled\"\nreasoning_dialect = \"openrouter_enabled\""
            ),
        );
        let runner = runner_from_toml(&strict)?;

        let plain = VALID_CONFIG.replace(
            strong,
            &format!("{strong}\nreasoning_policy = \"disabled\"\nreasoning_dialect = \"llama_cpp_enable_thinking\""),
        );
        runner_from_toml(&plain)?;

        let misspelled =
            VALID_CONFIG.replace(strong, &format!("{strong}\nreasoning_policy = \"disbled\""));
        assert!(error_message(&misspelled).contains("reasoning_policy"));

        drop(runner);
        Ok(())
    }

    /// A strict target that omits its dialect must fail at load time rather than inherit a
    /// default and emit a provider control nobody declared.
    #[test]
    fn an_authoritative_policy_without_a_dialect_is_rejected() {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        for policy in ["disabled", "enabled"] {
            let without_dialect = VALID_CONFIG.replace(
                strong,
                &format!("{strong}\nreasoning_policy = \"{policy}\""),
            );
            assert!(
                error_message(&without_dialect).contains("needs an explicit reasoning_dialect"),
                "policy {policy} without a dialect must be rejected: {}",
                error_message(&without_dialect)
            );
        }

        // An unspecified policy declares nothing, so an absent dialect stays legal.
        let no_policy = VALID_CONFIG.to_string();
        runner_from_toml(&no_policy).expect("a target with no policy needs no dialect");

        // A dialect on its own is also fine; it only becomes meaningful with a policy.
        let dialect_only = VALID_CONFIG.replace(
            strong,
            &format!("{strong}\nreasoning_dialect = \"llama_cpp_enable_thinking\""),
        );
        runner_from_toml(&dialect_only).expect("a bare dialect is accepted");
    }

    #[test]
    fn an_unknown_dialect_is_rejected() {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let unknown = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\nreasoning_policy = \"disabled\"\nreasoning_dialect = \"telepathy\""
            ),
        );
        assert!(error_message(&unknown).contains("reasoning_dialect"));
    }

    /// A llama.cpp target pointed at the OpenAI dialect would send `reasoning.effort`,
    /// which that server does not read. The dialect is a declaration, so a mismatch is
    /// only catchable when the pair is checked; that check is what this pins.
    #[test]
    fn a_llama_target_given_the_openai_dialect_still_loads_but_sends_the_declared_control() {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let mismatched = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\nreasoning_policy = \"disabled\"\nreasoning_dialect = \"openai_effort\""
            ),
        );
        // Loadable: the pair is well-formed, so configuration loading does not reject it.
        runner_from_toml(&mismatched).expect("a valid but semantically odd pair still loads");

        // The emitted control is the declared dialect's, not a guess: `effort = none`.
        // Catching a wrong dialect is a qualification job against the live server, which is
        // why the wire form is asserted here rather than only the load result.
        assert_eq!(
            ReasoningPolicy::Disabled.wire_body(ReasoningDialect::OpenAiEffort),
            Some(serde_json::json!({"reasoning": {"effort": "none"}}))
        );
    }

    /// An OpenAI target that wants reasoning on uses the authoritative effort mechanism;
    /// there is no policy that means "think" on that dialect.
    #[test]
    fn an_openai_thinking_target_uses_reasoning_effort() {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";

        let thinking = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\nreasoning_effort = \"high\"\nreasoning_dialect = \"openai_effort\""
            ),
        );
        runner_from_toml(&thinking).expect("an effort-driven thinking target is valid");

        let impossible = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\nreasoning_policy = \"enabled\"\nreasoning_dialect = \"openai_effort\""
            ),
        );
        assert!(
            error_message(&impossible).contains("reasoning_effort"),
            "{}",
            error_message(&impossible)
        );
    }

    /// The reordering contract must not be declarable in a shape that silently does
    /// nothing: half a contract would leave every request on the primary leg while the
    /// config reads as though escalation were configured.
    #[test]
    fn an_incomplete_escalation_contract_is_rejected_at_load() {
        let passthrough = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"";
        let escalated = format!(
            "{passthrough}\ncandidates = [\"strong\"]\nescalation_max_input_tokens = \
             61440\nescalation_target = \"strong\""
        );

        // Positive control: the complete contract loads, so each rejection below is caused
        // by its specific defect rather than by a fixture that never parsed.
        runner_from_toml(&VALID_CONFIG.replace(passthrough, &escalated))
            .expect("a complete escalation contract must load");

        for (defect, expected) in [
            (
                format!("{passthrough}\nescalation_max_input_tokens = 61440"),
                "route passthrough must declare escalation_max_input_tokens and escalation_target \
                 together",
            ),
            (
                format!("{passthrough}\nescalation_target = \"strong\""),
                "route passthrough must declare escalation_max_input_tokens and escalation_target \
                 together",
            ),
            (
                format!(
                    "{passthrough}\nescalation_max_input_tokens = 61440\nescalation_target = \
                     \"imaginary\""
                ),
                "route passthrough escalation_target references unknown target imaginary",
            ),
        ] {
            assert_eq!(
                error_message(&VALID_CONFIG.replace(passthrough, &defect)),
                expected,
                "config: {defect}"
            );
        }
    }

    /// A complete contract resolves to a real target and leaves the declared order alone:
    /// the promotion is applied at execution time, so loading must not reorder anything.
    #[test]
    fn a_declared_escalation_contract_loads_without_changing_the_primary() {
        let passthrough = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"";
        let escalated = format!(
            "{passthrough}\ncandidates = [\"strong\"]\nescalation_max_input_tokens = \
             61440\nescalation_target = \"strong\""
        );
        let config = VALID_CONFIG.replace(passthrough, &escalated);
        let names = ordered_spec(&config, "passthrough")
            .runtime_model_names("passthrough")
            .expect("declared candidates resolve");
        assert_eq!(
            names.parent.get(&Category::Any),
            Some(&vec!["weak".to_string(), "strong".to_string()]),
            "declaration order still offers the primary first"
        );
    }

    /// A route that configures escalation to a target it does not actually offer is
    /// configured-but-dead: `promoting` is a no-op for an absent id, so every over-threshold
    /// request silently keeps the declared order while the config and the dashboard both claim
    /// escalation happens. The leg must therefore be one of this route's own callable targets.
    #[test]
    fn an_escalation_target_outside_the_route_is_rejected_at_load() {
        let passthrough = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"";
        let configured = format!(
            "{passthrough}\nescalation_max_input_tokens = 61440\nescalation_target = \
         \"strong\""
        );

        // Positive control: `strong` IS a candidate here, so this loads. Without it, the
        // rejection below could be blamed on the fixture rather than on candidate membership.
        runner_from_toml(
        &VALID_CONFIG.replace(
            passthrough,
            &format!("{passthrough}\ncandidates = [\"strong\"]\nescalation_max_input_tokens = 61440\nescalation_target = \"strong\""),
        ),
    )
    .expect("an escalation target that is a candidate must load");

        let message = error_message(&VALID_CONFIG.replace(passthrough, &configured));
        assert_eq!(
            message, "route passthrough escalation_target strong is not a candidate of this route",
            "a real target that this route cannot serve must be rejected, not silently accepted"
        );
    }

    /// A route policy is applied last on the wire, so it must be expressible by every target
    /// the route can reach. A target's own policy is checked against its dialect at load;
    /// without the same check one layer up, a route asking for thinking on an effort-only
    /// dialect would apply nothing and still read as authoritative.
    #[test]
    fn a_route_policy_a_target_cannot_express_is_rejected_at_load() {
        // `responses` is declared as an openai_responses client, so give the target the
        // effort dialect, which grades effort and has no fragment meaning "think".
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        assert!(
            VALID_CONFIG.contains(strong),
            "the strong target fixture must exist verbatim"
        );
        // The route must point AT that target: a policy is checked against every target the
        // route can reach, and the fixture's other targets declare no dialect at all.
        let passthrough = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"";
        let on_effort_target = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"strong\"";
        let configured = VALID_CONFIG
            .replace(
                strong,
                &format!("{strong}\nreasoning_dialect = \"openai_effort\""),
            )
            .replace(
                passthrough,
                &format!("{on_effort_target}\nroute_reasoning_policy = \"medium\""),
            );

        // Positive control: `none` IS expressible on this dialect, so the rejection below
        // is about the policy value rather than the fixture.
        runner_from_toml(
            &VALID_CONFIG
                .replace(
                    strong,
                    &format!("{strong}\nreasoning_dialect = \"openai_effort\""),
                )
                .replace(
                    passthrough,
                    &format!("{on_effort_target}\nroute_reasoning_policy = \"none\""),
                ),
        )
        .expect("a non-thinking policy must load on an effort-only dialect");

        let message = error_message(&configured);
        assert!(
            message.contains("route passthrough reasoning policy cannot be applied"),
            "a medium policy on an effort-only dialect must be rejected, got: {message}"
        );
    }

    /// A fallback that cannot express the route's reasoning policy is rejected by default.
    ///
    /// The primary can carry the policy and the fallback cannot, which is exactly the shape
    /// of a bounded route whose second model has no reasoning switch. Rejecting is the
    /// strict default: a route that reads as authoritative and silently does nothing on one
    /// of its models is a misconfiguration.
    #[test]
    fn an_unexpressible_route_policy_on_a_fallback_is_rejected_by_default() {
        let configured = best_effort_fixture(false);

        let message = error_message(&configured);
        assert!(
            message.contains("route passthrough reasoning policy cannot be applied to target weak"),
            "a fallback with no dialect must be rejected unless the route opts in, got: {message}"
        );
    }

    /// The same route loads when it declares that its policy is best-effort.
    ///
    /// This is what keeps a fallback model a configuration-only concern: the model that
    /// cannot express "no thinking" is still eligible and simply runs its native behaviour.
    #[test]
    fn a_best_effort_route_admits_a_fallback_that_cannot_carry_its_policy() {
        runner_from_toml(&best_effort_fixture(true))
            .expect("a best-effort route must load with an unexpressible fallback");

        // Positive control: the same fixture without the opt-in is still refused, so this
        // test is about the flag rather than about a fixture that never failed.
        assert!(
            runner_from_toml(&best_effort_fixture(false)).is_err(),
            "the strict default must remain strict"
        );
    }

    /// The opt-in is scoped to the route that declares it.
    ///
    /// A target is shared between routes, so the fallback's dialect cannot decide the
    /// policy for every route that happens to reach it: a second route reaching the same
    /// models without the opt-in must still be refused.
    #[test]
    fn best_effort_does_not_weaken_another_route_sharing_the_same_target() {
        let strict_sibling = "[routes.strict-sibling]\nid = \"strict-sibling\"\ntype = \"passthrough\"\ntarget = \"strong\"\ncandidates = [\"weak\"]\nroute_reasoning_policy = \"none\"";
        let optimistic = best_effort_fixture(true);
        assert!(
            !optimistic.contains("[routes.strict-sibling]"),
            "the sibling route must be added here"
        );

        let message = error_message(&(optimistic + "\n" + strict_sibling));
        assert!(
            message.contains("route strict-sibling reasoning policy cannot be applied"),
            "a route without the opt-in must stay strict, got: {message}"
        );
    }

    /// The shared fixture: a route whose primary can express "no thinking" and whose
    /// fallback cannot. `strong` is given a dialect; `weak` deliberately has none.
    fn best_effort_fixture(best_effort: bool) -> String {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        assert!(
            VALID_CONFIG.contains(strong),
            "the strong target fixture must exist verbatim"
        );
        let passthrough = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"";
        let route = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"strong\"\ncandidates = [\"weak\"]\nroute_reasoning_policy = \"none\"";
        let route = if best_effort {
            format!("{route}\nroute_reasoning_policy_best_effort = true")
        } else {
            route.to_string()
        };
        VALID_CONFIG
            .replace(
                strong,
                &format!("{strong}\nreasoning_dialect = \"openai_effort\""),
            )
            .replace(passthrough, &route)
    }

    /// A threshold of zero means "every request", because the estimator floors at the
    /// structural overhead and never returns less. Whether that is intended is stated
    /// nowhere, and it silently removes the declared primary from first position, so it is
    /// rejected for the same reason `context_window = 0` is.
    #[test]
    fn a_zero_escalation_threshold_is_rejected_at_load() {
        let passthrough = "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"";
        assert_eq!(
            error_message(&VALID_CONFIG.replace(
                passthrough,
                &format!(
                    "{passthrough}\ncandidates = [\"strong\"]\nescalation_max_input_tokens = \
                     0\nescalation_target = \"strong\""
                ),
            )),
            "route passthrough escalation_max_input_tokens must be greater than zero",
        );
    }

    /// A deterministic route must be able to state its candidate order in configuration.
    /// The order is the contract: the selected target is offered first and the rest follow
    /// in declared order.
    #[test]
    fn a_passthrough_route_declares_its_ordered_candidates() -> RunnerResult<()> {
        let ordered = VALID_CONFIG.replace(
            "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"weak\"",
            "[routes.passthrough]\nid = \"switchyard/passthrough\"\ntype = \"passthrough\"\ntarget = \"strong\"\ncandidates = [\"weak\", \"classifier\"]",
        );
        assert_ne!(
            ordered, VALID_CONFIG,
            "the passthrough fixture must exist verbatim"
        );
        runner_from_toml(&ordered)?;

        // The order must reach the runtime model pool, not merely parse. This is the
        // property a config-only assertion cannot see: dropping candidates from the pool
        // still loads, but silently serves only the selected target.
        let spec = ordered_spec(&ordered, "passthrough");
        let names = spec
            .runtime_model_names("passthrough")
            .expect("declared candidates resolve");
        assert_eq!(
            names.parent.get(&Category::Any),
            Some(&vec![
                "strong".to_string(),
                "weak".to_string(),
                "classifier".to_string()
            ]),
            "the selected target must be offered first, then candidates in declared order"
        );
        assert_eq!(
            spec.callable_target_names(),
            vec!["strong", "weak", "classifier"],
            "callable targets must carry the same declared order"
        );

        // The same list naming its own selected target is not an order.
        let repeated = VALID_CONFIG.replace(
            "type = \"passthrough\"\ntarget = \"weak\"",
            "type = \"passthrough\"\ntarget = \"weak\"\ncandidates = [\"weak\"]",
        );
        assert!(
            error_message(&repeated).contains("more than once"),
            "{}",
            error_message(&repeated)
        );

        let blank = VALID_CONFIG.replace(
            "type = \"passthrough\"\ntarget = \"weak\"",
            "type = \"passthrough\"\ntarget = \"weak\"\ncandidates = [\" \"]",
        );
        assert!(
            error_message(&blank).contains("empty candidate name"),
            "{}",
            error_message(&blank)
        );

        Ok(())
    }

    /// A policy that is not specified leaves the caller's reasoning choice alone, and one
    /// target per model id is still enforced when two targets disagree on dialect.
    #[test]
    fn reasoning_dialect_differences_are_a_conflict_on_a_shared_model_id() {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        let alias = VALID_CONFIG.replace(
            strong,
            &format!("{strong}\nreasoning_dialect = \"openrouter_enabled\"\n\n[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\"\nreasoning_dialect = \"llama_cpp_enable_thinking\""),
        );
        assert!(
            error_message(&alias).contains("reasoning_dialect"),
            "{}",
            error_message(&alias)
        );
    }

    /// The capability contracts must survive parsing as runtime values, not just load.
    /// A config that parses while dropping the contract id or the rerank limits would
    /// serve vectors callers cannot compare and rerank results callers cannot trust.
    #[test]
    fn capability_contracts_are_carried_as_runtime_values() {
        let source = r#"
schema_version = 1

[llm_clients.responses]
format = "openai_responses"
base_url = "https://example.test/v1"

[targets.only]
id = "vendor/only"
llm_client = "responses"

[routes.only]
id = "probe/only"
type = "passthrough"
target = "only"

[capability_clients.engine_a]
format = "openai_embeddings"
base_url = "https://a.test"
model = "a"

[capability_clients.engine_b]
format = "openai_embeddings"
base_url = "https://b.test"
model = "b"

[capability_clients.reranker]
format = "cohere_jina_rerank"
base_url = "https://r.test"
model = "r"

[capabilities.space_a]
id = "space_a"
target = "engine_a"
contract = "localclaw-embedding-space:v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.space_b]
id = "space_b"
target = "engine_b"
contract = "localclaw-embedding-space:lfm2.5-350m-v1"
dimensions = 1024
normalization = "L2"
max_batch = 64

[capabilities.rerank]
id = "rerank"
target = "reranker"
max_candidates = 64
top_n = 5
max_doc_chars = 1024
max_query_chars = 512
"#;
        let config: DeploymentConfig = toml::from_str(source).expect("fixture parses");
        let a = &config.capabilities["space_a"];
        let b = &config.capabilities["space_b"];
        let r = &config.capabilities["rerank"];
        assert_eq!(a.contract.as_deref(), Some("localclaw-embedding-space:v1"));
        assert_eq!(
            b.contract.as_deref(),
            Some("localclaw-embedding-space:lfm2.5-350m-v1")
        );
        assert_ne!(a.contract, b.contract, "two engines keep two contracts");
        assert_eq!(a.dimensions, Some(1024));
        assert_eq!(r.max_candidates, Some(64));
        assert_eq!(r.top_n, Some(5));
        assert_eq!(r.max_doc_chars, Some(1024));
        assert_eq!(r.max_query_chars, Some(512));
        assert_eq!(
            config.capability_clients["engine_b"].url("/embeddings"),
            "https://b.test/embeddings"
        );
    }

    #[test]
    fn duplicate_targets_with_conflicting_settings_are_rejected() -> RunnerResult<()> {
        let strong = "[targets.strong]\nid = \"strong/model\"\nllm_client = \"responses\"";
        assert!(VALID_CONFIG.contains(strong));
        // Same model, same client, different effort: the second target could never take effect.
        let conflicting = VALID_CONFIG.replace(
            strong,
            &format!(
                "{strong}\n\n[targets.strong_max]\nid = \"strong/model\"\nllm_client = \"responses\"\nreasoning_effort = \"max\""
            ),
        );
        assert!(
            error_message(&conflicting)
                .contains(
                "different reasoning_effort, reasoning_policy, reasoning_dialect, extra_body, omit_body_fields, or unsupported_reasoning_effort",
            ),
            "{}",
            error_message(&conflicting)
        );
        // An alias with identical settings is still allowed (it only warns).
        let alias = VALID_CONFIG.replace(
            strong,
            &format!("{strong}\n\n[targets.strong_alias]\nid = \"strong/model\"\nllm_client = \"responses\""),
        );
        runner_from_toml(&alias)?;
        Ok(())
    }

    #[test]
    fn classifier_judge_completion_caps_are_configurable() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 512",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nmax_output_tokens = 256\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;
        Ok(())
    }

    #[test]
    fn classifier_prompts_are_configurable_in_both_modes() -> RunnerResult<()> {
        let capability = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom capability rubric\"",
        );
        runner_from_toml(&capability)?;

        let escalation = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"custom trajectory rubric\"\nescalation = { confirmations = 2 }",
        );
        runner_from_toml(&escalation)?;

        let empty = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"   \"",
        );
        assert!(error_message(&empty).contains("classifier prompt must not be empty"));

        let schema_placeholder = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nprompt = \"{{RESPONSE_SCHEMA}}\"",
        );
        assert!(
            error_message(&schema_placeholder)
                .contains("Switchyard supplies the schema automatically")
        );
        Ok(())
    }

    #[test]
    fn mode_custom_rejects_capability_fields() {
        let mixed = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "mode = \"custom\"\nbase_threshold = 0.5",
        );

        assert!(
            error_message(&mixed)
                .contains("mode custom cannot use capability or escalation fields")
        );
    }

    #[test]
    fn stage_router_rejects_an_unknown_field() {
        let config = stage_config().replace(
            "picker = \"efficient_first\"",
            "picker = \"efficient_first\"\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn auto_route_builds_a_stage_router_with_no_extra_fields() -> RunnerResult<()> {
        let config = format!(
            r#"{VALID_CONFIG}
[routes.auto]
id = "switchyard/auto"
type = "auto"
capable_target = "strong"
efficient_target = "weak"
"#
        );
        let runner = runner_from_toml(&config)?;
        assert!(runner.route("switchyard/auto").is_some());
        Ok(())
    }

    #[test]
    fn composite_stage_block_rejects_an_unknown_field() {
        let config = composite_config().replace(
            "confidence_threshold = 0.5",
            "confidence_threshold = 0.5\nmagic = true",
        );
        assert!(error_message(&config).contains("unknown field"));
    }

    #[test]
    fn rejects_unknown_fields_and_algorithm_types() {
        let unknown_field =
            VALID_CONFIG.replace("schema_version = 1", "schema_version = 1\nmagic = true");
        assert!(error_message(&unknown_field).contains("unknown field"));

        let nested_completion_cap = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nescalation = { max_output_tokens = 256 }",
        );
        assert!(error_message(&nested_completion_cap).contains("unknown field"));

        let unknown_classifier_field = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.5\nclassifier_magic = true",
        );
        assert!(error_message(&unknown_classifier_field).contains("unknown field"));

        // `context_window`, `max_input_tokens` and `daily_request_limit` are supported
        // target fields, so the control names one that is not: a typo must still be
        // refused rather than silently ignored.
        let target_capability = VALID_CONFIG.replace(
            "llm_client = \"responses\"",
            "llm_client = \"responses\"\nwindow_magic = 1000000",
        );
        assert!(error_message(&target_capability).contains("unknown field `window_magic`"));

        let unknown_algorithm = VALID_CONFIG.replace("type = \"noop\"", "type = \"imaginary\"");
        assert!(error_message(&unknown_algorithm).contains("unknown variant"));
    }

    #[test]
    fn rejects_unknown_stage_classifier_fields() {
        // Nested classifier typos must fail instead of silently using a default.
        let config = format!(
            r#"{VALID_CONFIG}

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
classifier_magic = true
"#
        );

        let error = error_message(&config);
        assert!(
            error.contains("unknown field `classifier_magic`"),
            "{error}"
        );
    }

    #[test]
    fn rejects_invalid_references_and_parameters() {
        let cases = [
            (
                VALID_CONFIG.replace("llm_client = \"primary\"", "llm_client = \"missing\""),
                "unknown llm client missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"missing\"]",
                ),
                "unknown target missing",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [0, 0]",
                ),
                "at least one weight must be positive",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"strong\"]",
                ),
                "targets must be unique, strong is repeated",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nweights = [1]",
                ),
                "expected 2 weights, got 1",
            ),
            (
                VALID_CONFIG.replace("base_threshold = 0.5", "base_threshold = 1.5"),
                "base_threshold must be between 0 and 1",
            ),
            (
                VALID_CONFIG.replace("classifier_target = \"classifier\"\n", ""),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"\"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "classifier_target = \"classifier\"",
                    "classifier_target = \"   \"",
                ),
                "route references unknown target",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nthreshold_step = -0.1",
                ),
                "threshold_step must be finite and greater than or equal to 0",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.8\nthreshold_step = 0.11",
                ),
                "base_threshold + 2 * threshold_step must be at most 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmax_output_tokens = 0\nescalation = { confirmations = 2 }",
                ),
                "max_output_tokens must be at least 1",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "base_threshold = 0.5\nmessage_hash_fallback = true",
                ),
                "message_hash_fallback requires classify_trigger = new_session",
            ),
            (
                with_subagent_llm_classifier(
                    VALID_CONFIG,
                    "passthrough",
                    "\nmessage_hash_fallback = true",
                ),
                "cannot use message_hash_fallback",
            ),
            (
                with_subagent_llm_classifier(VALID_CONFIG, "passthrough", "")
                    .replace("mode = \"custom\"", "mode = \"capability\""),
                "mode capability cannot use custom classifier fields",
            ),
            (
                VALID_CONFIG.replace(
                    "base_threshold = 0.5",
                    "escalation = { confirmations = 2 }\nclassify_trigger = \"user_turn\"",
                ),
                "mode escalation cannot use classify_trigger",
            ),
            (
                VALID_CONFIG.replace("schema_version = 1", "schema_version = 2"),
                "unsupported schema_version 2",
            ),
            (
                VALID_CONFIG.replace("[targets.strong]", "[targets.\" strong \"]"),
                "target name must be non-empty and have no surrounding whitespace",
            ),
            (
                VALID_CONFIG.replace(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\ncontext_window = 0",
                ),
                "route random context_window must be greater than zero",
            ),
        ];

        for (toml, expected) in cases {
            let error = error_message(&toml);
            assert!(
                error.contains(expected),
                "expected error containing {expected}, got {error}"
            );
        }
    }

    #[test]
    fn accepts_duplicate_target_model_ids_on_one_client() -> RunnerResult<()> {
        // Two targets share one model id on one client. The client keeps one and drops the
        // other, so the build warns and still succeeds, and both routes resolve. Serving one
        // model under two route names this way is allowed; pointing both routes at one target
        // is the tidier form.
        const SAME_MODEL_TWO_ROUTES: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.fast]
id = "gpt-4o"
llm_client = "primary"

[targets.smart]
id = "gpt-4o"
llm_client = "primary"

[routes.fast]
id = "switchyard/fast"
type = "passthrough"
target = "fast"

[routes.smart]
id = "switchyard/smart"
type = "passthrough"
target = "smart"
"#;
        let state = runner_from_toml(SAME_MODEL_TWO_ROUTES)?;
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/fast", "switchyard/smart"]
        );
        Ok(())
    }

    #[test]
    fn accepts_same_model_id_on_different_llm_clients() -> RunnerResult<()> {
        // Separate routes may serve the same model through different clients.
        const CROSS_PROVIDER: &str = r#"
schema_version = 1

[llm_clients.openai]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.azure]
format = "openai_chat"
base_url = "https://azure.test/v1"

[targets.openai]
id = "gpt-4o"
llm_client = "openai"

[targets.azure]
id = "gpt-4o"
llm_client = "azure"

[routes.openai]
id = "switchyard/openai-gpt4o"
type = "passthrough"
target = "openai"

[routes.azure]
id = "switchyard/azure-gpt4o"
type = "passthrough"
target = "azure"
"#;
        runner_from_toml(CROSS_PROVIDER)?;
        let shared_route = format!(
            r#"{CROSS_PROVIDER}

[routes.shared]
id = "switchyard/shared"
type = "stage_router"
capable_target = "openai"
efficient_target = "azure"
picker = "efficient_first"
confidence_threshold = 0.5
"#
        );
        let message = error_message(&shared_route);
        assert!(
            message.contains("route shared")
                && message.contains("openai")
                && message.contains("azure")
                && message.contains("gpt-4o")
                && message.contains("different llm clients"),
            "{message}"
        );
        Ok(())
    }

    #[test]
    fn accepts_relative_weights_and_seed() -> RunnerResult<()> {
        let weighted = VALID_CONFIG.replace(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nweights = [1, 3]\nseed = 42",
        );
        runner_from_toml(&weighted)?;
        Ok(())
    }

    #[test]
    fn accepts_new_session_trigger_with_message_hash_fallback() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replace(
            "base_threshold = 0.5",
            "base_threshold = 0.25\nthreshold_step = 0.1\nclassify_trigger = \"new_session\"\nmessage_hash_fallback = true",
        );
        runner_from_toml(&configured)?;
        Ok(())
    }

    /// Video has one faithful transport. Declaring it anywhere else would admit the request
    /// and then deliver its content as text, so the load fails instead.
    #[test]
    fn video_declaration_requires_a_responses_client() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "id = \"classifier/model\"",
            "id = \"classifier/model\"\nvideo = true\nvideo_carrier = \"both\"",
            1,
        );
        let message = match runner_from_toml(&configured) {
            Ok(_) => return Err(RunnerError::configuration("chat client claimed video")),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("video = true") && message.contains("openai_chat"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// A route may not promise a capability no target of that route can keep.
    #[test]
    fn route_claiming_video_needs_a_video_target() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nvideo = true",
            1,
        );
        let message = match runner_from_toml(&configured) {
            Ok(_) => return Err(RunnerError::configuration("route claimed video")),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("video = true") && message.contains("none of its targets"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// A responses target that states video support satisfies the route claim.
    #[test]
    fn video_on_a_responses_target_is_accepted() -> RunnerResult<()> {
        let configured = VALID_CONFIG
            .replacen(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nvideo = true\nvideo_carrier = \"both\"",
                1,
            )
            .replacen(
                "targets = [\"strong\", \"weak\"]",
                "targets = [\"strong\", \"weak\"]\nvideo = true",
                1,
            );
        let runner = runner_from_toml(&configured)?;
        let route = runner
            .route("switchyard/random")
            .ok_or_else(|| RunnerError::configuration("random route is missing"))?;
        let capabilities = route.capabilities();
        assert_eq!(capabilities.video, Some(true));
        Ok(())
    }

    /// A video declaration must say which carrier it serves.
    ///
    /// Without it the manifest cannot express the one distinction that matters: the free
    /// Nemotron endpoint answers an inline video and 404s a URL, and `video = true` alone
    /// would admit the request that then fails at the provider.
    #[test]
    fn a_video_target_must_state_its_carrier() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "id = \"strong/model\"",
            "id = \"strong/model\"\nvideo = true",
            1,
        );
        let message = match runner_from_toml(&configured) {
            Ok(_) => {
                return Err(RunnerError::configuration(
                    "video without a carrier was accepted",
                ));
            }
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("no video_carrier"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// An image carrier is a narrowing of a medium the target must first claim.
    #[test]
    fn an_image_carrier_needs_a_vision_claim() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "id = \"strong/model\"",
            "id = \"strong/model\"\nimage_carrier = \"inline\"",
            1,
        );
        let message = match runner_from_toml(&configured) {
            Ok(_) => {
                return Err(RunnerError::configuration(
                    "carrier without vision was accepted",
                ));
            }
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("image_carrier without vision"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// A carrier on a route that claims no medium is an inert field, so the load refuses it.
    #[test]
    fn a_carrier_without_its_medium_is_refused() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "targets = [\"strong\", \"weak\"]",
            "targets = [\"strong\", \"weak\"]\nvideo_carrier = \"inline\"",
            1,
        );
        let message = match runner_from_toml(&configured) {
            Ok(_) => return Err(RunnerError::configuration("inert carrier was accepted")),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("video_carrier without video = true"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// "Both" is two promises, and a route that advertises it needs targets for each half.
    ///
    /// The previous shape of this check accepted any claim while one target served *a*
    /// carrier, so a route advertising both over an inline-only target still told callers a
    /// remote URL would work. Every half of the claim is now required separately.
    #[test]
    fn a_route_advertising_both_carriers_needs_both_served() -> RunnerResult<()> {
        let configured = VALID_CONFIG
            .replacen(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nvideo = true\nvideo_carrier = \"inline\"",
                1,
            )
            .replacen(
                "targets = [\"strong\", \"weak\"]",
                "targets = [\"strong\", \"weak\"]\nvideo = true",
                1,
            );
        let message = match runner_from_toml(&configured) {
            Ok(_) => return Err(RunnerError::configuration("unserved carrier was accepted")),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("carrier Both") && message.contains("Remote"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// A target that refuses the medium does not serve any of its carriers.
    ///
    /// The carrier field defaults to "both" when absent, so reading it without first asking
    /// whether the target admits the medium at all would let a vision-refusing target keep a
    /// route's vision promise alive.
    #[test]
    fn a_target_refusing_the_medium_serves_no_carrier() -> RunnerResult<()> {
        let configured = VALID_CONFIG
            .replacen(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nvision = true\nimage_carrier = \"inline\"",
                1,
            )
            .replacen(
                "id = \"weak/model\"",
                "id = \"weak/model\"\nvision = false",
                1,
            )
            .replacen(
                "targets = [\"strong\", \"weak\"]",
                "targets = [\"strong\", \"weak\"]\nvision = true",
                1,
            );
        let message = match runner_from_toml(&configured) {
            Ok(_) => return Err(RunnerError::configuration("refusing target was accepted")),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("carrier Both"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// A route may not promise a carrier no target of that route serves.
    ///
    /// This is the carrier counterpart of "video = true with no video target": the promise
    /// would be advertised and then refused at request time. Both targets here are narrowed,
    /// so a route asking for the other carrier has nothing behind it.
    #[test]
    fn a_route_may_not_promise_a_carrier_no_target_serves() -> RunnerResult<()> {
        let configured = VALID_CONFIG
            .replacen(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nvideo = true\nvideo_carrier = \"inline\"",
                1,
            )
            .replacen(
                "id = \"weak/model\"",
                "id = \"weak/model\"\nvideo = true\nvideo_carrier = \"inline\"",
                1,
            )
            .replacen(
                "targets = [\"strong\", \"weak\"]",
                "targets = [\"strong\", \"weak\"]\nvideo = true\nvideo_carrier = \"remote\"",
                1,
            );
        let message = match runner_from_toml(&configured) {
            Ok(_) => {
                return Err(RunnerError::configuration(
                    "unsupported carrier was accepted",
                ));
            }
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("carrier Remote") && message.contains("no target serves"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// The same manifest is accepted once the route asks only for the carrier its targets
    /// serve, which is the case this change exists to make expressible.
    #[test]
    fn an_inline_carrier_is_accepted_when_a_target_serves_it() -> RunnerResult<()> {
        let configured = VALID_CONFIG
            .replacen(
                "id = \"strong/model\"",
                "id = \"strong/model\"\nvideo = true\nvideo_carrier = \"inline\"",
                1,
            )
            .replacen(
                "targets = [\"strong\", \"weak\"]",
                "targets = [\"strong\", \"weak\"]\nvideo = true\nvideo_carrier = \"inline\"",
                1,
            );
        let runner = runner_from_toml(&configured)?;
        let route = runner
            .route("switchyard/random")
            .ok_or_else(|| RunnerError::configuration("random route is missing"))?;
        let capabilities = route.capabilities();
        assert_eq!(capabilities.video, Some(true));
        assert_eq!(capabilities.video_carrier, Some(MediaCarrier::Inline));
        Ok(())
    }

    /// A composite is resolved before its children, so its claim must be backed by one of
    /// them. Otherwise it advertises video and then skips every child.
    #[test]
    fn composite_claiming_video_needs_a_video_child() -> RunnerResult<()> {
        let configured =
            format!("{VALID_CONFIG}\n[composites.tier]\nchildren = [\"random\"]\nvideo = true\n");
        let message = match runner_from_toml(&configured) {
            Ok(_) => return Err(RunnerError::configuration("composite claimed video")),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("composite route tier") && message.contains("none of its children"),
            "unexpected error: {message}"
        );
        Ok(())
    }

    /// The same composite is accepted once a child can serve video.
    #[test]
    fn composite_with_a_video_child_is_accepted() -> RunnerResult<()> {
        let configured = format!(
            "{}\n[composites.tier]\nchildren = [\"random\"]\nvideo = true\n",
            VALID_CONFIG
                .replacen(
                    "id = \"strong/model\"",
                    "id = \"strong/model\"\nvideo = true\nvideo_carrier = \"both\"",
                    1
                )
                .replacen(
                    "targets = [\"strong\", \"weak\"]",
                    "targets = [\"strong\", \"weak\"]\nvideo = true",
                    1
                )
        );
        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn target_extra_body_is_parsed_and_applied_to_its_backend() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             extra_body = { service_tier = \"priority\", \
             chat_template_kwargs = { enable_thinking = false } }",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(RunnerError::configuration("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        let backend = build_backend(
            "primary",
            client,
            &target.extra_body,
            &target.omit_body_fields,
            &target.unsupported_reasoning_effort,
            None,
            None,
            ReasoningDialect::default(),
            target.cooldown_group.clone(),
            None,
            None,
            None,
        )?;

        assert_eq!(
            backend.extra_body().get("service_tier"),
            Some(&json!("priority"))
        );
        assert_eq!(
            backend
                .extra_body()
                .get("chat_template_kwargs")
                .and_then(|value| value.get("enable_thinking")),
            Some(&json!(false))
        );
        Ok(())
    }

    #[test]
    fn target_omit_body_fields_is_parsed_and_applied_to_its_backend() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "llm_client = \"primary\"",
            "llm_client = \"primary\"\n\
             omit_body_fields = [\"max_output_tokens\"]",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&configured).map_err(|error| {
            RunnerError::configuration(format!("failed to parse config: {error}"))
        })?;
        let Some(target) = config.targets.get("classifier") else {
            return Err(RunnerError::configuration("classifier target is missing"));
        };
        let Some(client) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        let backend = build_backend(
            "primary",
            client,
            &target.extra_body,
            &target.omit_body_fields,
            &target.unsupported_reasoning_effort,
            None,
            None,
            ReasoningDialect::default(),
            target.cooldown_group.clone(),
            None,
            None,
            None,
        )?;

        assert!(backend.omit_body_fields().contains("max_output_tokens"));
        Ok(())
    }

    #[test]
    fn client_deadline_defaults_and_rejects_zero() -> RunnerResult<()> {
        for (setting, expected) in [
            ("", None),
            ("timeout_ms = 1500", Some(1500)),
            ("timeout_ms = 0", Some(0)),
        ] {
            let source = format!(
                "format = \"openai_chat\"\nbase_url = \"https://example.test/v1\"\n{setting}"
            );
            let config: LlmClientConfig = toml::from_str(&source).expect("valid deadline config");
            let backend = build_backend(
                "test",
                &config,
                &BTreeMap::new(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                None,
                None,
                ReasoningDialect::default(),
                None,
                None,
                None,
                None,
            );
            if expected == Some(0) {
                assert!(backend.is_err());
            } else {
                assert_eq!(backend?.timeout(), expected.map(Duration::from_millis));
            }
        }
        Ok(())
    }

    #[test]
    fn retry_budget_defaults_and_accepts_an_override() -> RunnerResult<()> {
        let default: DeploymentConfig = toml::from_str(VALID_CONFIG).map_err(|error| {
            RunnerError::configuration(format!("failed to parse default config: {error}"))
        })?;
        let Some(primary) = default.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, DEFAULT_MAX_RETRIES);

        let explicit = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 0",
            1,
        );
        let config: DeploymentConfig = toml::from_str(&explicit).map_err(|error| {
            RunnerError::configuration(format!("failed to parse explicit retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, 0);

        let maximum = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!(
                "base_url = \"https://example.test/v1\"\nmax_retries = {MAX_CONFIGURED_RETRIES}"
            ),
            1,
        );
        let config: DeploymentConfig = toml::from_str(&maximum).map_err(|error| {
            RunnerError::configuration(format!("failed to parse maximum retry config: {error}"))
        })?;
        let Some(primary) = config.llm_clients.get("primary") else {
            return Err(RunnerError::configuration("primary llm client is missing"));
        };
        assert_eq!(primary.max_retries, MAX_CONFIGURED_RETRIES);
        Ok(())
    }

    #[test]
    fn rejects_headers_that_switchyard_sets() {
        let cases = [
            (
                "base_url = \"https://example.test/v1\"",
                "base_url = \"https://example.test/v1\"\n\
                 extra_headers = { AUTHORIZATION = \"Bearer custom-key\" }",
                "AUTHORIZATION",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"X-Api-Key\" = \"custom-key\" }",
                "X-Api-Key",
            ),
            (
                "base_url = \"https://example.test\"",
                "base_url = \"https://example.test\"\n\
                 extra_headers = { \"ANTHROPIC-VERSION\" = \"custom-version\" }",
                "ANTHROPIC-VERSION",
            ),
        ];

        for (original, replacement, header) in cases {
            let configured = VALID_CONFIG.replacen(original, replacement, 1);
            let error = error_message(&configured);
            assert!(
                error.contains(&format!("extra_headers cannot set {header:?}")),
                "expected {header} to be rejected, got: {error}"
            );
        }
    }

    #[test]
    fn accepts_additional_headers() -> RunnerResult<()> {
        let configured = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { X-Inference-Priority = \"batch\" }",
            1,
        );

        runner_from_toml(&configured)?;
        Ok(())
    }

    #[test]
    fn retry_budget_rejects_negative_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = -1",
            1,
        );
        assert!(error_message(&invalid).contains("max_retries"));
    }

    #[test]
    fn retry_budget_rejects_excessive_values() {
        let invalid = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\nmax_retries = 11",
            1,
        );
        assert!(
            error_message(&invalid).contains("llm client primary max_retries must be at most 10")
        );
    }

    #[test]
    fn api_key_environment_reference_is_validated() {
        let missing = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\napi_key_env = \"SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET\"",
            1,
        );
        assert!(error_message(&missing).contains("SWITCHYARD_CONFIG_TEST_KEY_THAT_IS_NOT_SET"));

        const EMPTY_KEY_ENV: &str = "SWITCHYARD_CONFIG_TEST_EMPTY_KEY";
        unsafe {
            // "unsafe" is for concurrent reads and writes, very rare
            std::env::set_var(EMPTY_KEY_ENV, "");
        }
        let empty = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            &format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{EMPTY_KEY_ENV}\""),
            1,
        );
        let message = error_message(&empty);
        unsafe {
            std::env::remove_var(EMPTY_KEY_ENV);
        }
        assert!(message.contains("is empty"));
    }

    #[test]
    fn forward_auth_rejects_conflicting_credentials() {
        let competing_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test/v1\"",
            "base_url = \"https://example.test/v1\"\n\
                 forward_auth = true\n\
                 api_key_env = \"UNUSED_TEST_KEY\"",
            1,
        );
        assert!(
            error_message(&competing_auth).contains("cannot set both forward_auth and api_key_env")
        );

        let static_auth = VALID_CONFIG.replacen(
            "base_url = \"https://example.test\"",
            "base_url = \"https://example.test\"\n\
                 forward_auth = true\n\
                 extra_headers = { Authorization = \"static-value\" }",
            1,
        );
        assert!(error_message(&static_auth).contains("extra_headers cannot set \"Authorization\""));

        let static_beta = static_auth.replace("Authorization", "anthropic-beta");
        assert!(
            error_message(&static_beta).contains("extra_headers cannot set \"anthropic-beta\"")
        );

        for header in ["chatgpt-account-id", "x-openai-fedramp"] {
            let static_context = VALID_CONFIG.replacen(
                "base_url = \"https://example.test/v1\"",
                &format!(
                    "base_url = \"https://example.test/v1\"\n\
                     forward_auth = true\n\
                     extra_headers = {{ \"{header}\" = \"static-value\" }}"
                ),
                1,
            );
            assert!(
                error_message(&static_context)
                    .contains(&format!("extra_headers cannot set \"{header}\""))
            );
        }
    }

    const ADVISOR_CONFIG: &str = r#"
schema_version = 1

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://example.test"

[targets.executor]
id = "executor/model"
llm_client = "anthropic"

[targets.advisor]
id = "advisor/model"
llm_client = "anthropic"

[routes.gated]
id = "switchyard/advisor"
type = "advisor"
executor_target = "executor"
advisor_target = "advisor"
"#;

    #[test]
    fn advisor_route_parses_with_defaults_and_builds() -> RunnerResult<()> {
        let state = runner_from_toml(ADVISOR_CONFIG)?;
        let route = state
            .route("switchyard/advisor")
            .expect("advisor route should exist");
        let models = route.models();
        // The gate calls the executor through `efficient`; `any` keeps it in the
        // route's last-resort pool.
        assert_eq!(
            models.models_for(&Category::Efficient),
            [ModelId::from("executor/model")]
        );
        assert_eq!(
            models.models_for(&Category::Any),
            [ModelId::from("executor/model")]
        );
        assert_eq!(
            models.models_for(&Category::Judge),
            [ModelId::from("advisor/model")]
        );
        assert_eq!(
            state
                .models()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["switchyard/advisor"]
        );
        Ok(())
    }

    #[test]
    fn advisor_route_accepts_every_gate_knob() -> RunnerResult<()> {
        let tuned = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            concat!(
                "advisor_target = \"advisor\"\n",
                "reviewer_system_prompt = \"review it\"\n",
                "redo_feedback_prefix = \"REVIEWER SAYS: \"\n",
                "gate_trigger = \"pattern\"\n",
                "gate_trigger_pattern = 'task_complete[\"\\s>:]*true'\n",
                "max_reviews = 2\n",
                "gate_stall_turns = 40\n",
                "gate_min_tool_results = 1\n",
                "advisor_max_tokens = 1024\n",
                "advisor_temperature = 0.0\n",
                "transcript_max_chars = 100000\n",
                "fail_open = false\n",
                "context_window = 200000\n",
                "tool_calling = true\n",
                "reasoning = true",
            ),
        );
        runner_from_toml(&tuned)?;
        Ok(())
    }

    #[test]
    fn advisor_route_rejects_unknown_keys() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nbogus_field = 1",
        );
        assert!(error_message(&invalid).contains("bogus_field"));
    }

    #[test]
    fn advisor_route_requires_both_targets() {
        let missing = ADVISOR_CONFIG.replace("advisor_target = \"advisor\"\n", "");
        assert!(error_message(&missing).contains("advisor_target"));
    }

    #[test]
    fn advisor_route_rejects_unknown_target() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"missing\"",
        );
        assert!(error_message(&invalid).contains("missing"));
    }

    #[test]
    fn advisor_route_rejects_invalid_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"\ngate_trigger_pattern = \"(unclosed\"",
        );
        assert!(error_message(&invalid).contains("not a valid regex"));
    }

    #[test]
    fn advisor_route_rejects_pattern_without_pattern_trigger() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger_pattern = \"done\"",
        );
        assert!(
            error_message(&invalid)
                .contains("gate_trigger_pattern requires gate_trigger = \"pattern\"")
        );
    }

    #[test]
    fn advisor_route_pattern_trigger_requires_pattern() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\ngate_trigger = \"pattern\"",
        );
        assert!(error_message(&invalid).contains("non-empty gate_trigger_pattern"));
    }

    #[test]
    fn advisor_route_rejects_zero_max_reviews() {
        let invalid = ADVISOR_CONFIG.replace(
            "advisor_target = \"advisor\"",
            "advisor_target = \"advisor\"\nmax_reviews = 0",
        );
        assert!(error_message(&invalid).contains("max_reviews must be at least 1"));
    }

    /// A route may narrow what may spend its terminal leg without changing anything else.
    ///
    /// The policy is a statement about money, so it has to be loadable on exactly the route
    /// that needs it and nowhere else: the default stays the historical capacity rule.
    const JUDGE_CONFIG: &str = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://example.test/v1"

[llm_clients.neutral]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.gemma]
id = "google/gemma-4-26b"
llm_client = "google"
free_tier = true

[targets.luna]
id = "openai/luna-nt"
llm_client = "neutral"

[routes.judge]
id = "smart-judge"
type = "passthrough"
target = "gemma"
candidates = []
terminal_fallback = ["luna"]
terminal_policy = "quota_only"
"#;

    #[test]
    fn a_quota_only_terminal_policy_loads_on_its_own_route() -> RunnerResult<()> {
        let runner = runner_from_toml(JUDGE_CONFIG)?;
        assert!(runner.route("smart-judge").is_some());
        Ok(())
    }

    #[test]
    fn a_terminal_policy_without_a_terminal_leg_is_refused() {
        let invalid = JUDGE_CONFIG.replace(
            "terminal_fallback = [\"luna\"]\nterminal_policy = \"quota_only\"",
            "terminal_policy = \"quota_only\"",
        );
        let message = error_message(&invalid);
        assert!(
            message.contains("terminal_policy but no terminal_fallback"),
            "the refusal names the two declarations: {message}"
        );
    }

    #[test]
    fn an_unknown_terminal_policy_is_refused() {
        let invalid = JUDGE_CONFIG.replace("quota_only", "cheapest");
        let message = error_message(&invalid);
        assert!(
            message.contains("unknown variant `cheapest`"),
            "a policy must be one of the declared ones: {message}"
        );
    }
}

/// Closure controls for the production-path fixes in whole-route composition.
#[cfg(test)]
mod composition_closure_tests {
    use super::*;
    use crate::failure::RouteErrorKind;

    /// A request no eligible route can serve is a caller error, not a deployment fault, so it
    /// is classified as an invalid request rather than a configuration failure.
    #[test]
    fn an_unsupported_capability_is_classified_as_a_request_error() {
        let error = RunnerError::unsupported_capability("tier2", "vision");
        assert!(matches!(
            error,
            RunnerError::UnsupportedCapability {
                capability: "vision",
                ..
            }
        ));
        assert!(error.to_string().contains("vision"));
        assert!(
            matches!(
                error.execution_error_summary().kind,
                RouteErrorKind::InvalidRequest
            ),
            "an unservable request is reported as invalid, not as configuration"
        );
        // A deployment fault stays a configuration failure.
        let broken = RunnerError::configuration("composite names a missing route");
        assert!(matches!(
            broken.execution_error_summary().kind,
            RouteErrorKind::Configuration
        ));
    }

    /// Two children claiming one continuation is a state conflict, never resolved by choosing
    /// the first claimant.
    #[test]
    fn a_state_conflict_is_not_absorbed_as_a_request_error() {
        let conflict = RunnerError::StateConflict("two children claim resp_1".to_string());
        assert!(conflict.to_string().contains("resp_1"));
        assert!(
            matches!(
                conflict.execution_error_summary().kind,
                RouteErrorKind::InvalidRequest
            ),
            "the conflict is reported rather than silently resolved"
        );
    }

    /// Cross-child policy stays narrower than candidate fallback inside a route, but an
    /// exhausted credential allowance is crossable: the next child is only worth trying
    /// when it draws on a different credential, and the composite cannot tell which is
    /// which from the class alone, so the configuration is the gate. A retired model
    /// crosses for the same reason a transient failure does. A credential REJECTION is
    /// crossable through the composite's credential_block path, not this class table —
    /// which is exactly why `Auth` stays terminal HERE.
    #[test]
    fn cross_route_recovery_excludes_policy_and_contract_failures() {
        use switchyard_llm_client::FailureClass;
        for class in [
            FailureClass::Transport,
            FailureClass::Timeout,
            FailureClass::RateLimited,
            FailureClass::TemporarilyUnavailable,
            FailureClass::Server,
            FailureClass::AccountQuota,
            FailureClass::ModelRetired,
        ] {
            assert!(
                class.allows_cross_route(),
                "{class:?} is recoverable across routes"
            );
        }
        for class in [
            FailureClass::Auth,
            FailureClass::Request,
            FailureClass::ProviderPolicy,
            FailureClass::ContextWindow,
            FailureClass::Other,
        ] {
            assert!(
                !class.allows_cross_route(),
                "{class:?} must not cross routes"
            );
        }
    }

    // ---- Per-minute windows -------------------------------------------------

    fn paced_config(extra: &str) -> DeploymentConfig {
        // Self-contained: this module only needs one client for the window checks, and a
        // shared fixture would make these assertions depend on unrelated route wiring.
        toml::from_str(&format!(
            "schema_version = 1\n\
             [llm_clients.primary]\nformat = \"openai_chat\"\n\
             base_url = \"https://example.test/v1\"\n\
             [routes.noop]\nid = \"switchyard/noop\"\ntype = \"noop\"\n\
             {extra}"
        ))
        .expect("the paced fixture parses")
    }

    /// A minute budget with no per-request ceiling, or a ceiling above the whole minute,
    /// can never serve its largest admissible request, so both are refused at load rather
    /// than discovered as a 429 under traffic.
    #[test]
    fn a_minute_window_must_contain_the_per_request_ceiling() {
        let error = paced_config(
            "[targets.paced]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\ninput_tokens_per_minute = 16000\n",
        )
        .validate_minute_windows()
        .expect_err("a token window without a ceiling is unfalsifiable");
        assert!(
            error.to_string().contains("without max_input_tokens"),
            "{error}"
        );

        let error = paced_config(
            "[targets.paced]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\ninput_tokens_per_minute = 16000\n\
             max_input_tokens = 20000\n",
        )
        .validate_minute_windows()
        .expect_err("a ceiling above the window can never be served");
        assert!(error.to_string().contains("above"), "{error}");

        paced_config(
            "[targets.paced]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\ninput_tokens_per_minute = 16000\n\
             max_input_tokens = 16000\n",
        )
        .validate_minute_windows()
        .expect("a ceiling at or below the window is accepted");
    }

    /// A zero window admits nothing, and one window serves every target that names the same
    /// physical model in one quota group, so both mistakes are caught at load.
    #[test]
    fn minute_windows_reject_zero_and_disagreeing_declarations() {
        let error = paced_config(
            "[targets.paced]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 0\n",
        )
        .validate_minute_windows()
        .expect_err("a zero window admits nothing");
        assert!(error.to_string().contains("zero"), "{error}");

        let error = paced_config(
            "[targets.first]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\n\
             [targets.second]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 30\n",
        )
        .validate_minute_windows()
        .expect_err("one window cannot have two limits");
        assert!(
            error.to_string().contains("one window serves them both"),
            "{error}"
        );

        paced_config(
            "[targets.first]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\n\
             [targets.second]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\n",
        )
        .validate_minute_windows()
        .expect("the same value stated twice is one declaration");
    }

    /// A different physical model keeps a separate window even on one credential.
    #[test]
    fn a_different_model_is_a_separate_window() {
        paced_config(
            "[targets.first]\nid = \"google/gemma-4-26b-a4b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 15\n\
             [targets.second]\nid = \"google/gemma-4-31b-it\"\nllm_client = \"primary\"\n\
             requests_per_minute = 30\n",
        )
        .validate_minute_windows()
        .expect("per-model windows are independent");
    }
}

// ---------------------------------------------------------------------------
// Batch A: the remaining Google chat models, and the order of a free lane
// ---------------------------------------------------------------------------

/// A Gemini free-tier deployment carrying every Google chat target, plus the two paid NT
/// legs the live `smart-nt` route actually falls back to.
///
/// The paid targets are present on purpose. A fixture with free legs only cannot say anything
/// about whether a free leg precedes a paid one, which is the property that had to change.
/// No credential is declared: these tests exercise ordering, capability declaration and
/// tariff facts, and a client that reads no environment keeps the suite hermetic.
#[cfg(test)]
mod batch_a_tests {
    use super::*;

    const MANIFEST: &str = r#"
schema_version = 1

[quota]
state_dir = "__STATE_DIR__"

[llm_clients.google]
format = "openai_chat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
locality = "external"
quota_group = "google-ai-studio-free"
max_retries = 0

[llm_clients.openai_nt]
format = "openai_responses"
base_url = "https://api.openai.test/v1"

[llm_clients.deepseek_nt]
format = "openai_responses"
base_url = "https://api.deepseek.test/v1"

[targets."direct-google-gemini-3.5-flash-lite"]
id = "gemini-3.5-flash-lite"
llm_client = "google"
vision = false
video = false
daily_request_limit = 500
context_window = 1048576
max_input_tokens = 250000
free_tier = true
requests_per_minute = 15
input_tokens_per_minute = 250000
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets."direct-google-gemini-3.1-flash-lite"]
id = "gemini-3.1-flash-lite"
llm_client = "google"
vision = false
video = false
daily_request_limit = 500
context_window = 1048576
max_input_tokens = 250000
free_tier = true
requests_per_minute = 15
input_tokens_per_minute = 250000
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets."direct-google-gemini-3.6-flash"]
id = "gemini-3.6-flash"
llm_client = "google"
vision = false
video = false
daily_request_limit = 20
context_window = 1048576
max_input_tokens = 250000
free_tier = true
requests_per_minute = 5
input_tokens_per_minute = 250000
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets."direct-google-gemini-3.8-flash"]
id = "gemini-3.8-flash"
llm_client = "google"
vision = false
video = false
daily_request_limit = 20
context_window = 1048576
max_input_tokens = 250000
free_tier = true
requests_per_minute = 5
input_tokens_per_minute = 250000
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets."direct-google-gemini-4-hydrogen"]
id = "gemini-4-hydrogen"
llm_client = "google"
vision = false
video = false
daily_request_limit = 20
context_window = 1048576
max_input_tokens = 250000
free_tier = true
requests_per_minute = 5
input_tokens_per_minute = 250000
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets."direct-google-gemini-2.5-flash-lite"]
id = "gemini-2.5-flash-lite"
llm_client = "google"
vision = false
video = false
free_tier = true
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]
held = "Google answers HTTP 404: this model is no longer available to new users, and directs callers to models/gemini-3.5-flash-lite (measured 2026-11-21)."

[targets."direct-google-gemini-3-flash"]
id = "gemini-3-flash-preview"
llm_client = "google"
vision = false
video = false
free_tier = true
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]
min_output_tokens = 600

[targets."direct-google-gemma-4-26b-a4b-it"]
id = "gemma-4-26b-a4b-it"
llm_client = "google"
vision = false
video = false
context_window = 262144
max_input_tokens = 12307
daily_request_limit = 14400
free_tier = true
accepts_reasoning_control = false
hosted_reasoning = "thought_tag"
min_output_tokens = 512
requests_per_minute = 30
input_tokens_per_minute = 16000
admission_margin_permille = 1300
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets."direct-google-gemma-4-31b-it"]
id = "gemma-4-31b-it"
llm_client = "google"
vision = false
video = false
context_window = 262144
max_input_tokens = 12307
daily_request_limit = 14400
free_tier = true
accepts_reasoning_control = false
hosted_reasoning = "thought_tag"
min_output_tokens = 512
requests_per_minute = 30
input_tokens_per_minute = 16000
admission_margin_permille = 1300
omit_body_fields = ["store", "metadata", "prompt_cache_key"]
unsupported_reasoning_effort = ["none"]

[targets.openai-gpt-6-luna-nt]
id = "gpt-6-luna"
llm_client = "openai_nt"
vision = true
reasoning_policy = "disabled"
reasoning_dialect = "openai_effort"

[targets.deepseek-deepseek-flash-nt]
id = "deepseek-flash"
llm_client = "deepseek_nt"
vision = true
reasoning_policy = "disabled"
reasoning_dialect = "openai_effort"

[routes."direct-google-gemini-3.5-flash-lite"]
id = "direct-google-gemini-3.5-flash-lite"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = []
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false

[routes."direct-google-gemini-3.1-flash-lite"]
id = "direct-google-gemini-3.1-flash-lite"
type = "passthrough"
target = "direct-google-gemini-3.1-flash-lite"
candidates = []
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false

[routes."direct-google-gemini-3.6-flash"]
id = "direct-google-gemini-3.6-flash"
type = "passthrough"
target = "direct-google-gemini-3.6-flash"
candidates = []
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false

[routes."direct-google-gemini-3.8-flash"]
id = "direct-google-gemini-3.8-flash"
type = "passthrough"
target = "direct-google-gemini-3.8-flash"
candidates = []
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false

[routes."direct-google-gemini-4-hydrogen"]
id = "direct-google-gemini-4-hydrogen"
type = "passthrough"
target = "direct-google-gemini-4-hydrogen"
candidates = []
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false

[routes."direct-google-gemini-2.5-flash-lite"]
id = "direct-google-gemini-2.5-flash-lite"
type = "passthrough"
target = "direct-google-gemini-2.5-flash-lite"
candidates = []
context_window = 1048576
tool_calling = false
reasoning = false
vision = false
video = false

[routes."direct-google-gemini-3-flash"]
id = "direct-google-gemini-3-flash"
type = "passthrough"
target = "direct-google-gemini-3-flash"
candidates = []
context_window = 1048576
tool_calling = false
reasoning = false
vision = false
video = false

[routes."direct-google-gemma-4-26b-a4b-it"]
id = "direct-google-gemma-4-26b-a4b-it"
type = "passthrough"
target = "direct-google-gemma-4-26b-a4b-it"
candidates = []
context_window = 262144
tool_calling = false
reasoning = false
vision = false
video = false

[routes."direct-google-gemma-4-31b-it"]
id = "direct-google-gemma-4-31b-it"
type = "passthrough"
target = "direct-google-gemma-4-31b-it"
candidates = []
context_window = 262144
tool_calling = false
reasoning = false
vision = false
video = false

[routes.smart_gemini]
id = "smart_gemini"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = [
    "direct-google-gemini-3.1-flash-lite",
    "direct-google-gemini-3.6-flash",
    "direct-google-gemini-3.8-flash",
    "direct-google-gemini-4-hydrogen",
    "direct-google-gemma-4-26b-a4b-it",
    "direct-google-gemma-4-31b-it",
]
selection = "measured"
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false

[routes.smart-nt]
id = "smart-nt"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = [
    "direct-google-gemini-3.1-flash-lite",
    "direct-google-gemini-3.6-flash",
    "direct-google-gemini-3.8-flash",
    "direct-google-gemini-4-hydrogen",
    "direct-google-gemma-4-26b-a4b-it",
    "direct-google-gemma-4-31b-it",
    "openai-gpt-6-luna-nt",
    "deepseek-deepseek-flash-nt",
]
selection = "measured"
context_window = 262144
tool_calling = true
reasoning = true
vision = true
video = false

# Mirrors the live pool shape: a held target named as an inner candidate. Nothing about
# this route should change except that the dead leg is never offered.
[routes.pool_with_held]
id = "switchyard/pool-with-held"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = [
    "direct-google-gemini-2.5-flash-lite",
    "direct-google-gemini-3.1-flash-lite",
]
selection = "measured"
"#;

    /// The provider's sentence, repeated from the manifest so the assertions compare the
    /// refusal a caller would actually see rather than a second hand-written copy.
    const HELD_2_5_REASON: &str = "Google answers HTTP 404: this model is no longer available to new users, and directs callers to models/gemini-3.5-flash-lite (measured 2026-11-21).";

    /// The fixture with `dir` as its allowance state directory.
    ///
    /// A load probes an EXISTING state directory by writing and removing a file in it, so
    /// parallel tests must not share a directory that exists: two probes racing is reported
    /// as an unusable quota directory. An absent directory is not created by a load at all.
    fn fixture(dir: &str) -> String {
        MANIFEST.replace("__STATE_DIR__", dir)
    }

    /// Loading a deployment must not create its accounting state.
    ///
    /// A deployment preflight runs this same loader, usually as root. Creating the quota
    /// directory at load left a root-owned directory under a unit that runs as its own
    /// account, and that lane's first call then failed with a permission error reported as an
    /// unavailable budget. Creation belongs to the process that spends.
    #[test]
    fn loading_a_deployment_creates_no_quota_state() {
        let dir = format!(
            "/tmp/switchyard-quota-load-side-effect-{}",
            std::process::id()
        );
        let _ = std::fs::remove_dir_all(&dir);

        let _runner = Runner::from_toml(&fixture(&dir)).expect("a fresh state directory loads");

        assert!(
            !std::path::Path::new(&dir).exists(),
            "a load must leave no accounting state: a root-run validation would own it"
        );
    }

    /// An EXISTING state directory this process cannot write is a configuration error at
    /// load, naming the path, rather than a refused leg at the first request.
    #[test]
    fn loading_refuses_a_state_directory_this_process_cannot_write() {
        use std::os::unix::fs::PermissionsExt;

        let dir = format!("/tmp/switchyard-quota-readonly-{}", std::process::id());
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create the state directory");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500))
            .expect("make it read-only");

        let root = std::fs::write(std::path::Path::new(&dir).join(".root-probe"), b"x").is_ok();
        let loaded = Runner::from_toml(&fixture(&dir));

        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
            .expect("restore so the directory can be removed");
        let _ = std::fs::remove_dir_all(&dir);
        if root {
            assert!(
                loaded.is_ok(),
                "root can write it, so there is nothing to refuse"
            );
            return;
        }
        let error = match loaded {
            Ok(_) => panic!("an unwritable state directory is refused at load"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(&dir),
            "the refusal must name the directory: {error}"
        );
    }

    fn runner() -> Runner {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = format!("/tmp/switchyard-quota-batch-a-{n}");
        Runner::from_toml(&fixture(&dir)).expect("the fixture deployment loads")
    }

    fn parsed() -> DeploymentConfig {
        toml::from_str(&fixture("/tmp/switchyard-quota-batch-a-parse"))
            .expect("the fixture parses as a deployment config")
    }

    fn ordered_ids(route: &Route) -> Vec<String> {
        route
            .models()
            .all_models()
            .iter()
            .map(|model| model.as_str().to_owned())
            .collect()
    }

    /// Every Google chat target the owner's dossier names is loadable in one deployment, and
    /// the two held records are catalogued without claiming a capability they cannot serve.
    #[test]
    fn every_google_chat_target_loads_in_one_deployment() {
        let runner = runner();

        for (name, model) in [
            (
                "direct-google-gemini-3.5-flash-lite",
                "gemini-3.5-flash-lite",
            ),
            (
                "direct-google-gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite",
            ),
            ("direct-google-gemini-3.6-flash", "gemini-3.6-flash"),
            ("direct-google-gemini-3.8-flash", "gemini-3.8-flash"),
            ("direct-google-gemini-4-hydrogen", "gemini-4-hydrogen"),
            // The catalogued id is the preview name; the route name is the stable one.
            ("direct-google-gemini-3-flash", "gemini-3-flash-preview"),
            ("direct-google-gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it"),
            ("direct-google-gemma-4-31b-it", "gemma-4-31b-it"),
        ] {
            let Some(route) = runner.route(name) else {
                panic!("{name} route is missing from the deployment")
            };
            assert_eq!(
                route.first_target_slug(),
                model,
                "{name} must resolve to its model id"
            );
        }

        for serving in [
            "direct-google-gemini-3.6-flash",
            "direct-google-gemini-3.8-flash",
            "direct-google-gemini-4-hydrogen",
        ] {
            let capabilities = runner.route_capabilities(serving).expect("capabilities");
            assert_eq!(
                capabilities.tool_calling,
                Some(true),
                "{serving} must advertise tool_calling"
            );
            assert_eq!(
                capabilities.reasoning,
                Some(true),
                "{serving} must advertise reasoning"
            );
        }

        for held in [
            "direct-google-gemini-2.5-flash-lite",
            "direct-google-gemini-3-flash",
        ] {
            let capabilities = runner.route_capabilities(held).expect("capabilities");
            assert_ne!(
                capabilities.tool_calling,
                Some(true),
                "{held} is a catalogue record and must not claim tool_calling"
            );
            assert_ne!(
                capabilities.reasoning,
                Some(true),
                "{held} is a catalogue record and must not claim reasoning"
            );
        }

        // Only the withdrawn 2.5 record is held out of service. The 3-flash record is a
        // catalogue entry for a model that does answer, so it must stay advertised: holding
        // it would take a working alias out of the catalogue for the wrong reason.
        let withheld = runner
            .route("direct-google-gemini-2.5-flash-lite")
            .expect("the held route keeps its registration");
        assert_eq!(
            withheld.held_reason(),
            Some(HELD_2_5_REASON),
            "the hold must carry the provider's evidence, not a placeholder"
        );
        assert!(
            withheld.models().all_models().is_empty(),
            "a held route must offer no candidate"
        );
        assert_eq!(
            runner
                .route("direct-google-gemini-3-flash")
                .expect("3-flash route exists")
                .held_reason(),
            None,
            "3-flash serves and must not be held"
        );
        let advertised: Vec<&str> = runner
            .advertised_models()
            .map(|model| model.id.as_str())
            .collect();
        assert!(
            !advertised.contains(&"direct-google-gemini-2.5-flash-lite"),
            "a held route must not be advertised: {advertised:?}"
        );
        assert!(
            advertised.contains(&"direct-google-gemini-3-flash"),
            "a serving route must stay advertised: {advertised:?}"
        );
    }

    /// The declared order is the whole point: eight free Gemini legs, then the two paid NT
    /// legs. Selection may demote a free leg on measurement, but the tariff is part of the
    /// sort key, so no paid leg can be ordered ahead of an eligible free one.
    #[test]
    fn smart_nt_declares_eight_free_legs_before_the_paid_fallbacks() {
        let runner = runner();
        let route = runner.route("smart-nt").expect("smart-nt exists");

        assert_eq!(
            ordered_ids(route),
            vec![
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.6-flash",
                "gemini-3.8-flash",
                "gemini-4-hydrogen",
                "gemma-4-26b-a4b-it",
                "gemma-4-31b-it",
                "gpt-6-luna",
                "deepseek-flash",
            ],
            "smart-nt must declare every free Gemini leg before the paid NT legs"
        );

        // The tariff is a per-target fact (`free_tier`), not the `:free` slug convention that
        // `all_targets_free` reads, so the structural claim is asserted on the declarations.
        let config = parsed();
        for free in [
            "direct-google-gemini-3.5-flash-lite",
            "direct-google-gemini-3.1-flash-lite",
            "direct-google-gemini-3.6-flash",
            "direct-google-gemini-3.8-flash",
            "direct-google-gemini-4-hydrogen",
            "direct-google-gemma-4-26b-a4b-it",
            "direct-google-gemma-4-31b-it",
        ] {
            assert!(
                config.targets[free].free_tier == Some(true),
                "{free} must be declared free, or selection may order a paid leg ahead of it"
            );
        }
        for paid in ["openai-gpt-6-luna-nt", "deepseek-deepseek-flash-nt"] {
            assert!(
                config.targets[paid].free_tier != Some(true),
                "{paid} is metered and must stay behind the free tier"
            );
        }
    }

    /// `smart_gemini` is the same pool without the paid fallbacks, so it is genuinely free
    /// end to end and can be offered where spending a paid call is not acceptable.
    #[test]
    fn smart_gemini_carries_every_gemini_leg_and_no_paid_leg() {
        let runner = runner();
        let route = runner.route("smart_gemini").expect("smart_gemini exists");

        assert_eq!(
            ordered_ids(route),
            vec![
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.6-flash",
                "gemini-3.8-flash",
                "gemini-4-hydrogen",
                "gemma-4-26b-a4b-it",
                "gemma-4-31b-it",
            ]
        );
        // Every leg is in the free allowance, so this alias never spends a paid call.
        let config = parsed();
        for free in [
            "direct-google-gemini-3.5-flash-lite",
            "direct-google-gemini-3.1-flash-lite",
            "direct-google-gemini-3.6-flash",
            "direct-google-gemini-3.8-flash",
            "direct-google-gemini-4-hydrogen",
            "direct-google-gemma-4-26b-a4b-it",
            "direct-google-gemma-4-31b-it",
        ] {
            assert!(
                config.targets[free].free_tier == Some(true),
                "{free} must be declared free"
            );
        }
    }

    /// A held leg inside a measured pool is never offered, and the pool keeps serving.
    ///
    /// This is the live shape: `smart_gemini` and `smart-nt` name a target in their candidate
    /// list that the provider has since withdrawn. Removing it from the declared list would
    /// hide the measurement; leaving it in without a hold would spend one dispatch per request
    /// on a 404. Filtering at load is the third option, and the one asserted here.
    #[test]
    fn a_held_leg_leaves_a_measured_pool_without_emptying_it() {
        let runner = runner();
        let route = runner
            .route("switchyard/pool-with-held")
            .expect("the pool exists");

        assert_eq!(
            ordered_ids(route),
            vec![
                "gemini-3.5-flash-lite".to_string(),
                "gemini-3.1-flash-lite".to_string()
            ],
            "the held leg must be absent and the declared order otherwise preserved"
        );
        assert_eq!(
            route.held_reason(),
            None,
            "a pool with live legs is not itself held"
        );
    }

    /// The output floor is declared where it was measured, and nowhere else.
    ///
    /// `gemini-3-flash-preview` answered HTTP 500 with an empty completion at an 8-token
    /// reservation and normally at 600, so 600 is the smallest reservation observed to work
    /// and is the one declared. The Gemma pair keeps its own measured 512. Every other free
    /// leg declares no floor: adding one there would refuse requests that currently succeed,
    /// which no measurement supports.
    #[test]
    fn the_output_floor_is_declared_where_it_was_measured() {
        let config = parsed();
        assert_eq!(
            config.targets["direct-google-gemini-3-flash"].min_output_tokens,
            Some(600),
            "3-flash needs the measured reservation, or an 8-token call becomes a 500"
        );
        for gemma in [
            "direct-google-gemma-4-26b-a4b-it",
            "direct-google-gemma-4-31b-it",
        ] {
            assert_eq!(
                config.targets[gemma].min_output_tokens,
                Some(512),
                "{gemma}'s measured floor must survive untouched"
            );
        }
        for unmeasured in [
            "direct-google-gemini-3.5-flash-lite",
            "direct-google-gemini-3.1-flash-lite",
            "direct-google-gemini-3.6-flash",
            "direct-google-gemini-3.8-flash",
            "direct-google-gemini-4-hydrogen",
        ] {
            assert_eq!(
                config.targets[unmeasured].min_output_tokens, None,
                "{unmeasured} has no measured floor; inventing one would refuse working requests"
            );
        }
    }

    /// A held model serves nothing, so it declares no budget at all: zero is refused at load
    /// ("must be at least 1"), and a fabricated limit would claim an allowance nobody metered.
    #[test]
    fn held_models_declare_no_daily_budget_and_serve_nothing() {
        let config = parsed();

        for held in [
            "direct-google-gemini-2.5-flash-lite",
            "direct-google-gemini-3-flash",
        ] {
            let target = config
                .targets
                .get(held)
                .expect("the held record is catalogued");
            assert_eq!(
                target.daily_request_limit, None,
                "{held} must not declare a daily budget"
            );
        }

        for serving in [
            "direct-google-gemini-3.5-flash-lite",
            "direct-google-gemini-3.1-flash-lite",
            "direct-google-gemini-3.6-flash",
            "direct-google-gemini-3.8-flash",
            "direct-google-gemini-4-hydrogen",
            "direct-google-gemma-4-26b-a4b-it",
            "direct-google-gemma-4-31b-it",
        ] {
            let target = config
                .targets
                .get(serving)
                .expect("the serving target is declared");
            assert!(
                target.daily_request_limit.is_some_and(|limit| limit >= 1),
                "{serving} must declare a real daily budget"
            );
            assert!(
                target.free_tier == Some(true),
                "{serving} is served inside a free allowance"
            );
        }
    }

    /// The measured provider refusal is a declaration, not a code branch, so it has to be on
    /// every Google target: a no-thinking consumer sends `none` and the provider answers 400.
    #[test]
    fn the_refused_effort_value_is_declared_on_every_google_target() {
        let config = parsed();
        let none = BTreeSet::from(["none".to_string()]);

        for name in config
            .targets
            .keys()
            .filter(|name| name.starts_with("direct-google-"))
        {
            assert_eq!(
                config.targets[name].unsupported_reasoning_effort, none,
                "{name} must declare that its provider refuses effort `none`"
            );
            assert_eq!(
                config.targets[name].omit_body_fields,
                BTreeSet::from([
                    "metadata".to_string(),
                    "prompt_cache_key".to_string(),
                    "store".to_string(),
                ]),
                "{name} must drop the Responses-only fields Google rejects"
            );
        }

        // A target on a wire that accepts `none` keeps every value, so the declaration is
        // scoped to the provider that needs it rather than applied fleet-wide.
        assert!(
            config.targets["openai-gpt-6-luna-nt"]
                .unsupported_reasoning_effort
                .is_empty()
        );
    }
}

#[cfg(test)]
mod fit_escalation_config_tests {
    use super::*;

    /// Loads a deployment that must be refused, returning the refusal text.
    ///
    /// Matched rather than unwrapped because `Runner` has no `Debug`; a successful load panics
    /// instead of being reported as a refusal.
    fn refuse(source: &str) -> String {
        match runner_from_toml(source) {
            Ok(_) => panic!("deployment loaded but should have been refused"),
            Err(error) => format!("{error}"),
        }
    }

    /// A deployment whose route opts into fit escalation but declares no wider tier refuses to
    /// load. The option promises a request its pool cannot hold will still be answered, and
    /// without a terminal leg there is nothing that could answer it.
    #[test]
    fn fit_escalation_without_a_terminal_leg_refuses_the_deployment() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma]
id = "gemma/model"
llm_client = "google"

[routes.judge]
id = "judge"
type = "passthrough"
target = "gemma"
candidates = []
fit_escalation = true
"#;
        let message = refuse(config);
        assert!(
            message.contains("fit_escalation"),
            "refusal must name the option, got: {message}"
        );
        assert!(
            message.contains("terminal_fallback"),
            "refusal must name the missing declaration, got: {message}"
        );
    }

    /// The same route with one declared tier loads. This is the control for the refusal above:
    /// the guard must bite on the missing declaration, not on the option itself.
    #[test]
    fn fit_escalation_with_a_terminal_leg_loads() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma]
id = "gemma/model"
llm_client = "google"

[targets.wider]
id = "wider/model"
llm_client = "google"

[routes.judge]
id = "judge"
type = "passthrough"
target = "gemma"
candidates = []
terminal_fallback = [["wider"]]
fit_escalation = true
"#;
        runner_from_toml(config).expect("a route with a wider tier may declare fit_escalation");
    }

    /// Declaring the option off is the pre-change behaviour and must not require a tier.
    #[test]
    fn fit_escalation_off_does_not_require_a_tier() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma]
id = "gemma/model"
llm_client = "google"

[routes.judge]
id = "judge"
type = "passthrough"
target = "gemma"
candidates = []
fit_escalation = false
"#;
        runner_from_toml(config).expect("fit_escalation = false is the default behaviour");
    }

    /// The capability axis refuses the same unfillable promise: a route that says a wire
    /// refusal will still be answered must name the group that would answer it.
    #[test]
    fn capability_escalation_without_a_terminal_leg_refuses_the_deployment() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma]
id = "gemma/model"
llm_client = "google"

[routes.smart-route]
id = "smart-route"
type = "passthrough"
target = "gemma"
candidates = []
capability_escalation = true
"#;
        let message = refuse(config);
        assert!(
            message.contains("capability_escalation"),
            "refusal must name the option, got: {message}"
        );
        assert!(
            message.contains("terminal_fallback"),
            "refusal must name the missing declaration, got: {message}"
        );
    }

    /// Control for the refusal above: the guard bites on the missing declaration, not on the
    /// option.
    #[test]
    fn capability_escalation_with_a_terminal_leg_loads() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma]
id = "gemma/model"
llm_client = "google"

[targets.portable]
id = "portable/model"
llm_client = "google"

[routes.smart-route]
id = "smart-route"
type = "passthrough"
target = "gemma"
candidates = []
terminal_fallback = [["portable"]]
capability_escalation = true
"#;
        runner_from_toml(config).expect("a route with a later group may declare the option");
    }

    /// Off is the default and needs no tier, exactly as the fit axis does. This is the
    /// invariant that keeps the option from widening any existing route at load.
    #[test]
    fn capability_escalation_off_does_not_require_a_tier() {
        let config = r#"
schema_version = 1

[llm_clients.google]
format = "openai_chat"
base_url = "https://provider.test/v1"

[targets.gemma]
id = "gemma/model"
llm_client = "google"

[routes.smart-route]
id = "smart-route"
type = "passthrough"
target = "gemma"
candidates = []
capability_escalation = false
"#;
        runner_from_toml(config).expect("capability_escalation = false is the default behaviour");
    }
}

// ---------------------------------------------------------------------------
// Daily quota scope: one counter per account, or one per model
// ---------------------------------------------------------------------------

/// Two free models drawing on one provider account.
///
/// The account scope exists because the OpenRouter free-model allowance is stated once per
/// account (`free_model_daily_requests: 1000`), so a per-model counter would hand the fleet
/// one 1000-request budget per model and the fleet could spend several times the real cap
/// before any counter refused it.
#[cfg(test)]
mod daily_quota_scope_tests {
    use super::*;

    const SHARED_ACCOUNT: &str = r#"
schema_version = 1

[quota]
state_dir = "/var/tmp/switchyard-quota-scope-fixture"

[llm_clients.openrouter_free]
format = "openai_chat"
base_url = "https://openrouter.test/api/v1"
quota_group = "openrouter-free-owner"
quota_scope = "account"
quota_reset = "utc"

[targets."free-a"]
id = "vendor/free-a"
llm_client = "openrouter_free"
daily_request_limit = 1000

[targets."free-b"]
id = "vendor/free-b"
llm_client = "openrouter_free"
daily_request_limit = 1000

[routes."free"]
id = "free"
type = "passthrough"
target = "free-a"
candidates = ["free-b"]
"#;

    fn shared_account_config() -> DeploymentConfig {
        toml::from_str(SHARED_ACCOUNT).expect("the fixture must parse")
    }

    /// Every model in an account-scoped group resolves to the one sentinel counter, at one
    /// path, on the client's declared clock. A regression that keyed by target id would
    /// give each model its own budget while the manifest still read `quota_scope = "account"`.
    #[test]
    fn account_scope_collapses_every_model_onto_one_counter() {
        let config = shared_account_config();
        let state_dir = std::path::Path::new("/var/tmp/switchyard-quota-scope-fixture");
        let group = "openrouter-free-owner";

        let (model_a, path_a, boundary_a) =
            config.quota_counter(&config.targets["free-a"], group, state_dir);
        let (model_b, path_b, boundary_b) =
            config.quota_counter(&config.targets["free-b"], group, state_dir);

        assert_eq!(
            model_a.as_str(),
            switchyard_llm_client::ACCOUNT_SCOPE_MODEL,
            "an account-scoped target must count against the account sentinel"
        );
        assert_eq!(model_a, model_b, "two models, one account counter");
        assert_eq!(path_a, path_b, "one counter means one file");
        assert_eq!(
            boundary_a,
            switchyard_llm_client::DayBoundary::Utc,
            "the OpenRouter free-model counter turns over at midnight UTC"
        );
        assert_eq!(boundary_a, boundary_b);
    }

    /// One counter cannot serve two budgets: two targets on the same account counter that
    /// declare different limits are refused, and the refusal names both targets and both
    /// numbers so the operator can decide which one is meant.
    #[test]
    fn differing_limits_on_one_account_counter_refuse_the_deployment() {
        let source = SHARED_ACCOUNT.replace(
            "id = \"vendor/free-b\"\nllm_client = \"openrouter_free\"\ndaily_request_limit = 1000",
            "id = \"vendor/free-b\"\nllm_client = \"openrouter_free\"\ndaily_request_limit = 250",
        );
        assert!(
            source.contains("daily_request_limit = 250"),
            "the fixture edit must apply, or this test proves nothing"
        );

        let message = match runner_from_toml(&source) {
            Ok(_) => panic!("a shared counter with two different limits must be refused"),
            Err(error) => format!("{error}"),
        };

        assert!(
            message.contains("free-a") && message.contains("free-b"),
            "the refusal must name both targets, got: {message}"
        );
        assert!(
            message.contains("1000") && message.contains("250"),
            "the refusal must name both limits, got: {message}"
        );
        assert!(
            message.contains("one counter serves one budget"),
            "the refusal must say why it is refused, got: {message}"
        );
    }

    /// The default is per model: a client that names no scope keeps a counter per target,
    /// so declaring a daily limit today cannot silently change meaning.
    #[test]
    fn a_client_naming_no_scope_keeps_one_counter_per_model() {
        let source = SHARED_ACCOUNT
            .replace("quota_scope = \"account\"\n", "")
            .replace("quota_reset = \"utc\"\n", "");
        let config: DeploymentConfig = toml::from_str(&source).expect("the fixture must parse");
        let state_dir = std::path::Path::new("/var/tmp/switchyard-quota-scope-fixture");
        let group = "openrouter-free-owner";

        let (model_a, path_a, boundary_a) =
            config.quota_counter(&config.targets["free-a"], group, state_dir);
        let (model_b, path_b, _) =
            config.quota_counter(&config.targets["free-b"], group, state_dir);

        assert_eq!(model_a.as_str(), "vendor/free-a");
        assert_eq!(model_b.as_str(), "vendor/free-b");
        assert_ne!(path_a, path_b, "per-model counters are separate files");
        assert_eq!(
            boundary_a,
            switchyard_llm_client::DayBoundary::Pacific,
            "Pacific stays the default boundary"
        );
    }
}

#[cfg(test)]
mod quota_provider_tests {
    use super::*;

    const BASE: &str = r#"
schema_version = 1

[quota]
state_dir = "/var/tmp/switchyard-quota-provider-fixture"

[llm_clients.openrouter_free]
format = "openai_chat"
base_url = "https://openrouter.test/api/v1"
quota_group = "openrouter-free-owner"
quota_scope = "account"
quota_reset = "utc"
quota_provider = "openrouter_free_models"

[targets."free-a"]
id = "vendor/free-a"
llm_client = "openrouter_free"
daily_request_limit = 1000

[routes."free"]
id = "free"
type = "passthrough"
target = "free-a"
"#;

    fn config() -> DeploymentConfig {
        toml::from_str(BASE).expect("the fixture must parse")
    }

    /// The declared pool is read into the typed form, so the loader never carries a free
    /// string that a reader might not implement.
    #[test]
    fn a_declared_pool_parses_and_names_the_providers_own_bucket() {
        let config = config();
        let client = &config.llm_clients["openrouter_free"];
        let provider = client.quota_provider.expect("the pool must parse");
        assert_eq!(provider, QuotaProvider::OpenRouterFreeModels);
        assert_eq!(provider.pool(), "free_model_daily_requests");
        assert_eq!(provider.reset(), QuotaReset::Utc);
    }

    /// A pool name the code does not implement is refused at load. Accepting it would
    /// authorize a mapping that never produces a reading, which reads as configured and
    /// admits on the local budget forever.
    #[test]
    fn an_unknown_pool_name_refuses_the_deployment() {
        let source = BASE.replace(
            "quota_provider = \"openrouter_free_models\"",
            "quota_provider = \"some_other_pool\"",
        );
        assert!(source.contains("some_other_pool"), "the edit must apply");
        assert!(
            toml::from_str::<DeploymentConfig>(&source).is_err(),
            "an unimplemented pool must not deserialize"
        );
    }

    /// An account-wide pool cannot feed a per-model counter: one account figure spent as
    /// though it were several budgets would refuse each model on the same number.
    #[test]
    fn a_pool_on_a_model_scoped_counter_refuses_the_deployment() {
        let source = BASE.replace("quota_scope = \"account\"", "quota_scope = \"model\"");
        assert!(
            source.contains("quota_scope = \"model\""),
            "the edit must apply"
        );
        let config: DeploymentConfig = toml::from_str(&source).expect("it parses");
        let refusal = config
            .validate_quota_providers()
            .expect_err("a model-scoped counter must not be reconciled");
        let message = refusal.to_string();
        assert!(
            message.contains("quota_scope") && message.contains("openrouter_free_models"),
            "the refusal must name the pool and the rule: {message}"
        );
    }

    /// The counter must turn over on the pool's own clock. An OpenRouter reading counted on
    /// the Pacific boundary lands in the wrong day for eight hours a night.
    #[test]
    fn a_pool_on_the_wrong_reset_clock_refuses_the_deployment() {
        let source = BASE.replace("quota_reset = \"utc\"", "quota_reset = \"pacific\"");
        assert!(
            source.contains("quota_reset = \"pacific\""),
            "the edit must apply"
        );
        let config: DeploymentConfig = toml::from_str(&source).expect("it parses");
        let refusal = config
            .validate_quota_providers()
            .expect_err("a counter on the wrong clock must not be reconciled");
        let message = refusal.to_string();
        assert!(
            message.contains("quota_reset") && message.contains("utc"),
            "the refusal must name the clock the pool requires: {message}"
        );
    }

    /// A pool named for a client whose targets declare no daily limit has nowhere to write,
    /// so the mapping would read as configured while doing nothing.
    #[test]
    fn a_pool_without_a_counter_refuses_the_deployment() {
        let source = BASE.replace("daily_request_limit = 1000\n", "");
        assert!(
            !source.contains("daily_request_limit"),
            "the edit must apply"
        );
        let config: DeploymentConfig = toml::from_str(&source).expect("it parses");
        let refusal = config
            .validate_quota_providers()
            .expect_err("a pool with no counter must be refused");
        let message = refusal.to_string();
        assert!(
            message.contains("daily_request_limit"),
            "the refusal must say what is missing: {message}"
        );
    }

    /// The valid shape passes, so the three refusals above are about the rule and not about
    /// the fixture being invalid in every direction.
    #[test]
    fn a_valid_pool_mapping_is_accepted() {
        config()
            .validate_quota_providers()
            .expect("the documented shape must be accepted");
    }

    /// The facts carry the resolved group and the pool, so a consumer outside the route path
    /// can find the counter without repeating the loader's defaults.
    #[test]
    fn the_facts_carry_the_group_and_the_pool() {
        let config = config();
        let facts = config.deployment_facts();
        let client = &facts.clients["openrouter_free"];
        assert_eq!(client.quota_group.as_deref(), Some("openrouter-free-owner"));
        assert_eq!(
            client.quota_provider,
            Some(QuotaProvider::OpenRouterFreeModels)
        );

        // A client that names no group is its own group, so a consumer never has to
        // reproduce that default.
        let source = BASE.replace("quota_group = \"openrouter-free-owner\"\n", "")
            + "\n[llm_clients.other]\nformat = \"openai_chat\"\nbase_url = \"https://x.test/v1\"\n";
        let config: DeploymentConfig = toml::from_str(&source).expect("it parses");
        let facts = config.deployment_facts();
        assert_eq!(facts.clients["other"].quota_group.as_deref(), Some("other"));
        assert_eq!(facts.clients["other"].quota_provider, None);
    }
}
