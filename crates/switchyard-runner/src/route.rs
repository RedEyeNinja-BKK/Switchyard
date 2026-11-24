// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One configured algorithm and the clients that serve its targets.

use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, LibsyError, RoutingOutcome, RuntimeModels};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use switchyard_llm_client::{
    AttemptTrace, AuxiliaryOperation, CandidateTrace, ClientRouter, RunObservation, RunObserver,
    SelectionTrace, TranslatingLlmClient,
};
use switchyard_protocol::{LlmClientError, ModelId, Request, Response, WireFormat};
use thiserror::Error;

use crate::DecisionTarget;
use crate::config::{RouteReasoningPolicy, TerminalPolicy};

/// Where a route's work is actually performed.
///
/// Derived from the `billing` declaration of the `llm_client` each target resolves
/// through, never from a model name, a slug or a base URL. A route served by the
/// operator's own deployment and one served by an external provider answer differently
/// to a caller asking "does this leave the building?", and that is a fact configuration
/// already holds once a deployment states it.
///
/// `Unknown` is a real answer, not a placeholder: a deployment that has not declared how a
/// client is billed cannot have its locality inferred, and claiming one would put a guess
/// on a surface where a reader cannot tell it from a fact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    /// Every leg is an operator-hosted deployment: no external provider charge.
    Local,
    /// Every leg is served by an external provider.
    Upstream,
    /// Legs disagree: some operator-hosted, some external.
    Mixed,
    /// At least one leg's billing is undeclared. Never inferred.
    #[default]
    Unknown,
}

/// Capabilities declared for one route.
///
/// `GET /v1/models` includes `context_window`, `tool_calling`, `vision` and `video` in
/// each standard `data` entry, using `null` for unset values. It omits `reasoning`.
/// The server rejects tool inputs, reasoning controls, images or video when the
/// corresponding declaration is explicitly `false` for the selected route.
///
/// `video` is the exception to "absent means permitted": video is admitted only when a
/// route declares it, because no deployment carried video before the field existed.
///
/// `video_carrier` is the second axis of the same fact. An endpoint that can read inline
/// media but cannot fetch a remote URL is video-capable for one carrier only, and a flag
/// that cannot tell them apart either over-claims (admitting a remote URL into a late
/// provider 404) or under-claims (refusing inline media the endpoint does serve). It is
/// `None` on an undeclared route, where the boolean already refuses everything.
#[derive(Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub context_window: Option<u32>,
    pub tool_calling: Option<bool>,
    /// Whether the routed model accepts reasoning controls, as declared in config.
    pub reasoning: Option<bool>,
    /// Whether the routed model accepts image input, as declared in config.
    pub vision: Option<bool>,
    /// Whether the routed model accepts video input, as declared in config.
    ///
    /// Deliberately separate from `vision`: an image-capable target need not be
    /// video-capable, and the providers that serve the two are not the same. Absent means
    /// "cannot accept video".
    pub video: Option<bool>,
    /// Which video carriers the routed model accepts, when video is declared.
    ///
    /// Only meaningful alongside `video = Some(true)`; a target declaring video must state
    /// its carriers, because "can read video" without saying which carrier leaves the same
    /// ambiguity the boolean has.
    pub video_carrier: Option<MediaCarrier>,
    /// Which image carriers the routed model accepts, when vision is declared.
    ///
    /// The image counterpart, and optional for the historical reason `vision` is: a
    /// deployment that predates carriers declares images and means both.
    pub image_carrier: Option<MediaCarrier>,
}

/// The two ways media reaches a provider, which are not the same capability.
///
/// Measured on the live free/paid OpenRouter endpoints: `nvidia/nemotron-3.5-super-vl:free`
/// answers 200 for an inline `data:` image or video and 404 for a remote URL of either,
/// with `No endpoints found that support video URLs`. The model lists video among its input
/// modalities either way, so a modality-only declaration cannot express the difference.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediaCarrier {
    /// Inline bytes in the request: a `data:` URL or base64 payload.
    Inline,
    /// The provider fetches the media itself from a URL.
    Remote,
    /// Both carriers are served.
    #[default]
    Both,
}

impl MediaCarrier {
    /// Whether this carrier set admits a request carrying `remote` media.
    pub const fn admits_remote(self) -> bool {
        matches!(self, Self::Remote | Self::Both)
    }

    /// Whether this carrier set admits a request carrying inline media.
    pub const fn admits_inline(self) -> bool {
        matches!(self, Self::Inline | Self::Both)
    }

    /// The individual carriers this set covers, so a caller can require each one.
    pub const fn carriers(self) -> &'static [Self] {
        match self {
            Self::Inline => &[Self::Inline],
            Self::Remote => &[Self::Remote],
            Self::Both => &[Self::Inline, Self::Remote],
        }
    }

    /// Whether this carrier set covers `carrier`.
    ///
    /// Distinct from equality on purpose: asking whether a set *includes* a carrier is the
    /// question a route's promise asks of its targets, and `Both` is the only set that
    /// includes `Both`.
    pub const fn includes(self, carrier: Self) -> bool {
        match carrier {
            Self::Both => matches!(self, Self::Both),
            Self::Inline => self.admits_inline(),
            Self::Remote => self.admits_remote(),
        }
    }
}

impl MediaCarrier {
    /// The carrier a request is checked against when the config states none.
    ///
    /// A route or target that predates carriers declared a medium and meant both, so the
    /// default must be the permissive end rather than an empty set.
    pub const DEFAULT: Self = Self::Both;

    /// The carriers covered by either set.
    ///
    /// This is the union a composite needs: it can serve a request only through one of its
    /// children, so what it can promise is what the children between them accept. `Inline`
    /// and `Remote` together are `Both`, and `Both` absorbs the other two.
    pub const fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Both, _) | (_, Self::Both) => Self::Both,
            (Self::Inline, Self::Remote) | (Self::Remote, Self::Inline) => Self::Both,
            (carrier, _) => carrier,
        }
    }
}

/// The video carrier a target's capability facts describe, or `None` when it serves neither.
///
/// `None` is not "unstated": it is the case where the target refuses video, expressed either
/// as `no_video` or as both carrier facts at once. Returning a carrier there would name a
/// transport the target cannot use, and the caller's video boolean already refuses it.
/// What one model must be able to hold for this request: the input it carries plus the
/// output the caller reserved.
///
/// A window that fits the input but not the reservation rejects the request upstream after
/// the target was already chosen, so the fit is judged on the whole need. Both the
/// per-candidate filter and the composite's pre-check read this, so the two cannot disagree
/// about whether a request fits.
fn required_tokens(request: &Request) -> u64 {
    let raw = request
        .raw_request
        .as_ref()
        .unwrap_or(&serde_json::Value::Null);
    let input = crate::request_fit::estimate_input_tokens(raw);
    request
        .llm_request
        .output
        .max_output_tokens
        .map_or(input, |reservation| input.saturating_add(reservation))
}

/// The exclusion codes that say a target could not *hold* this request.
///
/// Closed vocabulary. Every one of these is a fact about the size of the request against a
/// limit of one target, so asking the next tier the same question is meaningful. Codes
/// outside this list are not: a capability the request needs, a credential or a balance the
/// deployment lost, an accounting fault, or a policy refusal would be re-learned at the next
/// tier with a bill attached, and would tell the caller nothing they could act on.
fn is_request_fit_exclusion(code: &str) -> bool {
    matches!(
        code,
        "context_window"
            | "max_input_tokens"
            | "minute_window_too_small"
            | "output_budget_below_thinking_floor"
    )
}

/// Whether a pool that admitted nothing did so only because no target could hold the request,
/// and the request is one this route is authorized to serve at all.
///
/// This is deliberately NOT one of [`authorize_terminal`]'s answers. "The pool ran out" is a
/// claim about capacity; "no candidate can hold this request" is a claim about size, and the
/// two lead somewhere different. Reading a size refusal as exhaustion would let a metered tier
/// absorb a request the deployment never declared it could serve and would hide from the
/// caller that their prompt was the problem. Refusing to escalate at all leaves a request that
/// an authorized wider tier can serve unanswered.
///
/// `route_accepts` is the route's own capability declaration read against the request. It is
/// required because the per-target gates test size *before* capabilities, so a request that
/// needs vision can still show nothing but size exclusions here; without this the route would
/// escalate a request it never promised to serve. It also covers the request-level controls
/// the owner named: a malformed payload, a missing capability, a policy refusal and an invalid
/// judgment all fail either this check or the exclusion vocabulary below.
///
/// Every candidate must be excluded, and every exclusion must be a size fact. One candidate
/// the pool simply could hold means the pool is not empty and this function is not reached; one
/// exclusion for another reason means the next tier would be asked to answer a question about
/// the request rather than about capacity.
pub(crate) fn authorize_fit_escalation(
    route_accepts: bool,
    gates: &[Result<(), &'static str>],
) -> Option<&'static str> {
    if !route_accepts || gates.is_empty() {
        return None;
    }
    if !gates.iter().all(|gate| match gate {
        Err(code) => is_request_fit_exclusion(code),
        Ok(()) => false,
    }) {
        return None;
    }
    Some("primary_pool_cannot_hold_request")
}

/// Whether every target in one declared tier was excluded only because it could not hold the
/// request.
///
/// This is the tier-level form of [`authorize_fit_escalation`], and it answers exactly one
/// question: may the ordered walk continue past a tier that admitted nothing? It may, only
/// when the reason is the same one the primary pool gave. A tier whose targets refused for any
/// other reason ends the walk, because the next tier would be consulted on a claim about the
/// request that has already been settled.
fn tier_cannot_hold(gates: &[Result<(), &'static str>]) -> bool {
    !gates.is_empty()
        && gates.iter().all(|gate| match gate {
            Err(code) => is_request_fit_exclusion(code),
            Ok(()) => false,
        })
}

/// Hands the route's decision record to whichever observer the caller supplied.
///
/// Cloning only when someone is listening keeps the cost of the record to the routes that
/// are actually being watched.
fn emit_selection(observer: Option<&RunObserver>, trace: &SelectionTrace) {
    if let Some(observer) = observer {
        observer(RunObservation::Selection(Box::new(trace.clone())));
    }
}

/// Wraps `inner` so every completed call is also appended to `attempts`.
///
/// The attempt list is built from the same observations the metrics are recorded from, so a
/// dispatch that produced no line in the routing log still appears here with its outcome
/// code - which is the difference between "no attempt" and "an attempt that failed".
fn recording_observer(
    attempts: Arc<Mutex<Vec<AttemptTrace>>>,
    inner: Option<RunObserver>,
) -> RunObserver {
    Arc::new(move |observation: RunObservation| {
        let call = match &observation {
            RunObservation::LlmCall(call) => Some(("routing", call)),
            RunObservation::AnswerCall(call) => Some(("answer", call)),
            _ => None,
        };
        if let Some((phase, call)) = call {
            attempts.lock().push(AttemptTrace {
                model: call.selected_model.to_string(),
                phase,
                outcome: call.outcome.unwrap_or("served"),
                duration_ms: u64::try_from(call.duration.as_millis()).unwrap_or(u64::MAX),
                prompt_tokens: call
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.input_tokens)
                    .unwrap_or(0),
                completion_tokens: call
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.output_tokens)
                    .unwrap_or(0),
            });
        }
        if let Some(inner) = inner.as_ref() {
            inner(observation);
        }
    })
}

fn carrier_from_facts(facts: &switchyard_llm_client::CapabilityFacts) -> Option<MediaCarrier> {
    if facts.no_video {
        return None;
    }
    combine_carrier_facts(facts.no_remote_video, facts.no_inline_video)
}

/// The image carrier a target's capability facts describe, or `None` when it serves neither.
///
/// Mirrors `carrier_from_facts` for the other medium. Without this the per-candidate filter
/// would read an image carrier of "both" for every target, and a route that advertises both
/// would send a remote image to an inline-only candidate.
fn image_carrier_from_facts(
    facts: &switchyard_llm_client::CapabilityFacts,
) -> Option<MediaCarrier> {
    if facts.no_vision {
        return None;
    }
    combine_carrier_facts(facts.no_remote_image, facts.no_inline_image)
}

/// Turn the two refusal facts into the carrier set they describe.
fn combine_carrier_facts(no_remote: bool, no_inline: bool) -> Option<MediaCarrier> {
    match (no_remote, no_inline) {
        (false, false) => Some(MediaCarrier::Both),
        (false, true) => Some(MediaCarrier::Remote),
        (true, false) => Some(MediaCarrier::Inline),
        (true, true) => None,
    }
}

/// Decides whether the terminal leg may be spent, and names why.
///
/// This is the whole policy, kept as a free function so its decision table can be
/// exercised without a live route: every branch is a claim about the primary pool, and a
/// wrong claim either strands a request the pool could not serve or spends a paid leg it
/// did not need to.
///
/// * `capable` - targets the deployment says could serve this request at all;
/// * `dispatchable` - of those, the ones no credential block excludes right now;
/// * `attempts` - every upstream call this request actually made;
/// * `policy` - which evidence this route accepts as "the pool cannot serve this";
/// * `credential_of` - the credential that serves a target, when configuration says;
/// * `quota_blocked` - whether a block excludes a target because its allowance is spent,
///   rather than because the credential was rejected.
///
/// Returns `None` when the pool's unavailability is not provable, which is the safe
/// answer: the caller gets the primary error instead of a paid escalation.
pub(crate) fn authorize_terminal(
    capable: &[ModelId],
    dispatchable: &HashSet<ModelId>,
    attempts: &[AttemptTrace],
    policy: TerminalPolicy,
    credential_of: impl Fn(&ModelId) -> Option<String>,
    quota_blocked: impl Fn(&ModelId) -> bool,
) -> Option<&'static str> {
    // Nothing in the pool could serve this request even with unlimited capacity, so
    // "exhausted" would be a different claim from the true one. Refuse.
    if capable.is_empty() {
        return None;
    }
    if dispatchable.is_empty() {
        // Every capable leg is covered by a credential block, so the pool is provably out
        // of capacity without a single request being made. A route that escalates only on
        // quota evidence has to say which block it is trusting: a rejected credential and
        // a spent allowance are both "blocked", but only the second is the claim this
        // route's terminal leg exists for. `Availability` asks the same question for a wider
        // set of reasons, and a rejected credential is still not one of them: it is a
        // deployment fault, and answering it with a further call spends money on a
        // configuration error instead of surfacing it. Only the historical `Capacity` rule
        // reads every block as exhaustion.
        if policy != TerminalPolicy::Capacity && !capable.iter().all(quota_blocked) {
            return None;
        }
        return Some("primary_pool_blocked");
    }
    let primary_attempts: Vec<&AttemptTrace> = attempts
        .iter()
        .filter(|attempt| capable.iter().any(|model| model.as_str() == attempt.model))
        .collect();
    if primary_attempts.is_empty() {
        return None;
    }
    // One leg failing for a reason that is about the request - a policy answer, a
    // malformed tool call, a stream that broke mid-delivery - is enough to keep the
    // terminal leg out: retrying it elsewhere would fail again, with a bill attached.
    if !primary_attempts
        .iter()
        .all(|attempt| is_capacity_outcome(attempt.outcome, policy))
    {
        return None;
    }
    let attempted: HashSet<&str> = primary_attempts
        .iter()
        .map(|attempt| attempt.model.as_str())
        .collect();
    // A capable leg can only be left unattempted if the candidate walk stopped early, and
    // the walk stops early only on a credential-scoped failure. A timeout is not in
    // `is_capacity_outcome`, so it is already excluded above. Any unattempted leg must
    // therefore be explained by a credential this request already proved dead. Anything
    // else is a gap in coverage, not exhaustion. Which failures may prove that follows the
    // same policy as above: `http_429` is both credential-scoped and quota evidence, while a
    // rejected key is credential-scoped but says nothing about the allowance.
    let dead: HashSet<String> = primary_attempts
        .iter()
        .filter(|attempt| {
            is_credential_scope_outcome(attempt.outcome)
                && is_capacity_outcome(attempt.outcome, policy)
        })
        .filter_map(|attempt| credential_of(&ModelId::from(attempt.model.as_str())))
        .collect();
    for model in capable.iter().filter(|model| dispatchable.contains(*model)) {
        if attempted.contains(model.as_str()) {
            continue;
        }
        let Some(credential) = credential_of(model) else {
            return None;
        };
        if !dead.contains(&credential) {
            return None;
        }
    }
    Some("primary_pool_exhausted")
}

/// Whether an attempt ended because a leg ran out of capacity or was unavailable.
///
/// This is the closed list a terminal leg may be authorized by. It deliberately excludes
/// `timeout` (a delivered stream broke, which says nothing about the next leg),
/// `context_window` (a request-fit fact, not a capacity one), `http_403` and `http_400`
/// (answers about the request or the account's permissions), `empty_answer` (a quality
/// fact about one model), and every translation or configuration fault.
fn is_capacity_outcome(outcome: &str, policy: TerminalPolicy) -> bool {
    match policy {
        TerminalPolicy::Capacity => matches!(
            outcome,
            "daily_allowance_exhausted"
                | "daily_allowance_unavailable"
                | "minute_window"
                | "minute_window_too_small"
                | "http_401"
                | "http_402"
                | "http_429"
                | "http_5xx"
                | "transport"
                | "unavailable"
                | "attempt_timeout"
        ),
        // A quota-only route spends its terminal leg on allowance exhaustion or a rate
        // throttle, and on nothing else. `daily_allowance_unavailable` is an accounting
        // fault rather than a measured exhaustion; `minute_window_too_small` is a
        // request-fit fact; 5xx/transport/timeout say the deployment could not reach the
        // pool, and 401/402 say the credential or its balance was refused. None of those
        // is evidence that the free allowance ran out, which is the only claim this
        // policy lets the terminal leg be spent on.
        TerminalPolicy::QuotaOnly => matches!(
            outcome,
            "daily_allowance_exhausted" | "minute_window" | "http_429"
        ),
        // Everything `Capacity` accepts except the two authentication outcomes, the two
        // faults that are about this deployment's accounting or this request's size, and a
        // delivered-stream timeout. A rejected key or a refused balance is not a capacity
        // fact: the deployment is misconfigured or unpaid, and the next leg would be chosen
        // on a claim that is false about this one.
        //
        // `daily_allowance_unavailable` is an unreadable ledger, not a measured exhaustion:
        // the allowance may be entirely unspent, so the honest answer is to fail closed and
        // surface the accounting fault rather than buy a further leg on it.
        //
        // `minute_window_too_small` is a per-request fit failure: the payload is larger than
        // the whole minute budget, so no amount of waiting makes this model able to serve it.
        // Advancing would hand a too-large request to the next tier and silently widen the
        // declaration that admitted it; the caller is told their request does not fit.
        //
        // `attempt_timeout` is accepted here deliberately -- the owner's policy names a timed
        // out attempt as a reason to advance -- and its cost is that a leg which answered
        // slowly may also have been answering; the decision record keeps every attempt, so a
        // duplicated answer is visible rather than silent.
        TerminalPolicy::Availability => matches!(
            outcome,
            "daily_allowance_exhausted"
                | "minute_window"
                | "http_429"
                | "http_5xx"
                | "transport"
                | "unavailable"
                | "attempt_timeout"
        ),
    }
}

/// Whether an attempt's outcome says the CREDENTIAL is unusable, rather than the request.
///
/// Only used to explain why a capable leg was never called. The candidate walk stops early
/// only on these classes, so an unattempted leg here shares a credential the walk
/// already proved dead. A plain per-model 429 never stops the walk — it falls through to the
/// next candidate — so any 429 that leaves a leg unattempted is scope-wide by construction.
///
/// Which of these count as *evidence for this route* is decided by the policy when the set
/// is built: a rejected key stops the walk just as effectively as a spent allowance, but it
/// is not a spent allowance, so it must not explain a skipped leg on a quota-only route.
fn is_credential_scope_outcome(outcome: &str) -> bool {
    matches!(outcome, "http_401" | "http_402" | "http_429")
}

/// Caller credential family required by a forwarded-auth route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerAuthKind {
    Anthropic,
    OpenAi,
}

impl CallerAuthKind {
    /// Stable provider name used by the server compatibility API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }

    const fn accepts(self, wire_format: WireFormat) -> bool {
        matches!(
            (self, wire_format),
            (Self::Anthropic, WireFormat::AnthropicMessages)
                | (
                    Self::OpenAi,
                    WireFormat::OpenAiChat | WireFormat::OpenAiResponses
                )
        )
    }
}

/// Exact upstream model and client used for an auxiliary provider operation.
#[derive(Clone)]
pub struct AuxiliaryTarget {
    pub model: ModelId,
    pub client: Arc<TranslatingLlmClient>,
}

/// Error returned while loading or executing configured routes.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("{message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    #[error("unknown route model {0:?}")]
    UnknownRouteModel(String),
    /// Two children of one composite claim the same provider state for a request.
    ///
    /// Picking by child order would make the outcome depend on configuration order, so the
    /// conflict is reported and no provider is contacted.
    #[error("conflicting provider state: {0}")]
    StateConflict(String),
    /// A request is valid, but no eligible route can satisfy a capability it requires.
    ///
    /// This is the caller's request, not a deployment fault, so it is kept distinct from
    /// [`Self::Configuration`] and reported as a 4xx rather than a server failure.
    #[error("no eligible route for model {model:?}: requires {capability}")]
    UnsupportedCapability {
        model: String,
        capability: &'static str,
    },
    /// A route that exists but is deliberately held out of service.
    ///
    /// Distinct from an unknown model — the alias is declared and documented, so a caller
    /// resolving it is not making a mistake — and distinct from an unsupported capability,
    /// where changing the request could help. Here the provider withdrew the model: the
    /// refusal is terminal, carries the provider's own evidence, and never substitutes a
    /// live model for the one that was asked for.
    #[error("model {model:?} is held and cannot serve: {reason}")]
    Held { model: String, reason: String },
    #[error("caller format is incompatible with {} credentials", .0.as_str())]
    IncompatibleCallerFormat(CallerAuthKind),
    #[error("route has no compatible target for the auxiliary operation")]
    AuxiliaryUnsupported,
    #[error(transparent)]
    Algorithm(#[from] LibsyError),
    #[error(transparent)]
    Client(#[from] LlmClientError),
}

impl RunnerError {
    pub(crate) fn unsupported_capability(
        model: impl Into<String>,
        capability: &'static str,
    ) -> Self {
        Self::UnsupportedCapability {
            model: model.into(),
            capability,
        }
    }

    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn configuration_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

/// A configured algorithm and the per-target clients its calls resolve through.
pub struct Route {
    algorithm: Arc<dyn Algorithm>,
    // Resolves each offloaded call to the client configured for the target the algorithm
    // selected. A route is a synthetic model with no upstream of its own, so this is a
    // per-target lookup, never one client serving the whole route.
    clients: ClientRouter,
    /// Reasoning policy imposed by the route, overriding the target's factual policy.
    route_reasoning_policy: crate::config::RouteReasoningPolicy,
    /// Context-pressure threshold for preferring a wider-context leg.
    escalation_max_input_tokens: Option<u64>,
    /// Target promoted ahead of the declared order under context pressure, resolved
    /// to its model id at load.
    escalation_target: Option<ModelId>,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
    /// Where this route's work is performed, resolved once at load from each target's
    /// `llm_client` billing declaration.
    locality: Locality,
    anthropic_auxiliary_target: Option<AuxiliaryTarget>,
    responses_auxiliary_target: Option<AuxiliaryTarget>,
    decision_targets: Vec<DecisionTarget>,
    models: Arc<RuntimeModels>,
    /// The `llm_client` credential each target model id is served by, resolved at load.
    ///
    /// A whole-route composite needs this to decide whether a child that just reported
    /// an exhausted or rejected credential condemns the NEXT child too: the wire error
    /// names a target slug, not the credential, so the credential identity has to come
    /// from configuration. Empty when the route was built without one (tests).
    target_credentials: std::collections::HashMap<ModelId, String>,
    /// Ordering among candidates that already survived every hard gate.
    ///
    /// `None` keeps declared order, which is what every route did before the option
    /// existed, so a route that does not opt in is untouched by this machinery.
    selector: Option<Arc<crate::selection::MeasuredSelector>>,
    /// The daily request allowance serving each target model id.
    ///
    /// Selection prefer the candidate with the most requests left in its own budget, so it
    /// has to read the same counters the dispatch gate reserves against. Empty when the
    /// route was built without them (tests, or a deployment with no daily limits).
    daily_allowances: Arc<HashMap<ModelId, Arc<switchyard_llm_client::DailyRequestAllowance>>>,
    /// Set when every completion target of this route is held, naming the alias the caller
    /// asked for and the recorded reason. A held route keeps its registration so the refusal
    /// can be specific, but it serves nothing and is not advertised.
    held: Option<HeldRoute>,
    /// The target groups whose members are this route's candidate pool, when it declares one.
    ///
    /// Presence also selects the provider-neutral ranking policy. Keeping the declaration
    /// rather than a boolean means the trace can name the groups the pool was built from,
    /// and a pool route is recognisable at runtime, not only in the manifest that created it.
    pool_groups: Option<Vec<String>>,
    /// Targets this route may reach only when every eligible primary candidate is
    /// verifiably unavailable, grouped into ordered tiers.
    ///
    /// Deliberately not part of `models`: a terminal leg must never be ranked with the
    /// primary candidates, or it would become a peer that merely sorts last and would
    /// serve whenever the pool was busy. It is dispatched as its own list, under its own
    /// exhaustion gate, so "this provider is tired" cannot buy it a request.
    ///
    /// The outer order is the escalation order and is never re-decided by a selector. A
    /// tier's members are peers of each other, so the route's own selection policy orders
    /// them; two tiers are never merged, because "the second tier looks cheaper" is not a
    /// reason to consult it before the first one is proved out.
    terminal: Vec<Arc<RuntimeModels>>,
    /// Which evidence may spend `terminal`. Configuration sets this per route; the default
    /// is the historical capacity rule, so a route that names no policy behaves as before.
    terminal_policy: TerminalPolicy,
    /// Whether a request that no primary candidate can hold may be escalated to the declared
    /// tiers.
    ///
    /// A separate axis from `terminal_policy` on purpose. That one governs a pool that
    /// *failed*; this one governs a pool that never dispatched, because the request is larger
    /// than any of its targets can hold. Escalating on a size refusal is not a redefinition of
    /// capacity exhaustion -- it is a different claim, recorded under its own reason, and the
    /// tier that receives the request is judged on its own gates.
    fit_escalation: bool,
}

/// A route whose targets are all held out of service, with the evidence for holding it.
#[derive(Clone, Debug)]
pub struct HeldRoute {
    /// The first configured completion target, which is the name the caller's alias maps to.
    ///
    /// Stored rather than derived because a held route's candidate set is empty by
    /// construction: there is no first model left to read at request time.
    pub model: String,
    /// The provider's own evidence, forwarded verbatim to the caller.
    pub reason: String,
}

/// The selected model and untouched response produced by a route execution.
pub struct RunOutput {
    pub selected_model: ModelId,
    pub response: Response,
}

impl Route {
    /// Creates a fully configured execution route.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        decision_targets: Vec<DecisionTarget>,
        models: RuntimeModels,
        route_reasoning_policy: crate::config::RouteReasoningPolicy,
        escalation_max_input_tokens: Option<u64>,
        escalation_target: Option<ModelId>,
    ) -> Self {
        Self {
            algorithm,
            clients,
            route_reasoning_policy,
            escalation_max_input_tokens,
            escalation_target,
            caller_auth,
            capabilities,
            locality: Locality::Unknown,
            anthropic_auxiliary_target,
            responses_auxiliary_target,
            decision_targets,
            models: Arc::new(models),
            target_credentials: std::collections::HashMap::new(),
            selector: None,
            daily_allowances: Arc::new(HashMap::new()),
            held: None,
            pool_groups: None,
            terminal: Vec::new(),
            terminal_policy: TerminalPolicy::default(),
            fit_escalation: false,
        }
    }

    /// Declares a leg this route reaches only after its entire eligible primary pool is
    /// verifiably unavailable.
    ///
    /// The terminal models are kept out of the ranked candidate set on purpose: they are a
    /// last resort, not a peer. `smart-nt` is the shape this exists for — Gemini NT leaves
    /// and Luna NT compete as one provider-neutral pool, and a metered DeepSeek NT leg is
    /// reached only when none of them can answer.
    pub fn with_terminal_fallback(mut self, terminal: Vec<RuntimeModels>) -> Self {
        self.terminal = terminal.into_iter().map(Arc::new).collect();
        self
    }

    /// Selects which evidence may spend this route's terminal leg.
    ///
    /// Only meaningful beside `with_terminal_fallback`; configuration refuses the policy
    /// without a leg to govern. Set per route so a lane whose fallback is a metered
    /// last resort can be narrower than a lane whose fallback is an ordinary peer.
    pub fn with_terminal_policy(mut self, policy: TerminalPolicy) -> Self {
        self.terminal_policy = policy;
        self
    }

    /// Declares whether a request no primary candidate can hold may reach the tiers.
    ///
    /// Only meaningful beside `with_terminal_fallback`; configuration refuses the option
    /// without a leg to escalate to. Off is the historical behaviour: the pool admitted
    /// nothing, so the caller is told what did not fit and no further leg is contacted.
    ///
    /// On does not widen any tier. It changes one thing: a pool that dispatched nothing is
    /// asked the size question of each declared tier in order, instead of being refused
    /// before the tiers are looked at. A tier that cannot hold the request either ends the
    /// walk with the same refusal.
    pub fn with_fit_escalation(mut self, enabled: bool) -> Self {
        self.fit_escalation = enabled;
        self
    }

    /// The terminal targets this route declares, in declared order, across every tier.
    pub fn terminal_models(&self) -> Vec<ModelId> {
        self.terminal
            .iter()
            .flat_map(|tier| tier.all_models())
            .collect()
    }

    /// Whether the primary pool is verifiably out of capacity, and which claim says so.
    ///
    /// `None` means the terminal leg is not authorized. The two `Some` answers are the ones
    /// an operator acts on differently:
    ///
    /// * `primary_pool_blocked` — every leg that could serve this request is covered by a
    ///   credential block. Capacity is provably gone and nothing was dispatched.
    /// * `primary_pool_exhausted` — legs were dispatched and every one failed on capacity or
    ///   availability, with no capable leg left unexplained.
    ///
    /// Everything else refuses: no leg could ever serve the request, a leg failed for a
    /// reason that is about the request, or a capable leg's absence is unexplained.
    fn terminal_authorization(
        &self,
        capable: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        blocks: &[switchyard_llm_client::CredentialBlock],
    ) -> Option<&'static str> {
        authorize_terminal(
            capable,
            dispatchable,
            attempts,
            self.terminal_policy,
            |model| self.target_credential(model).map(str::to_string),
            |model| self.quota_blocked(model, blocks),
        )
    }

    /// Whether a quota-scoped block covers `model`.
    ///
    /// A block that says an allowance is spent is exhaustion evidence. A block that says the
    /// credential was rejected is not: the allowance may be untouched, and reading the two as
    /// one is how a wrong key would buy a metered request.
    fn quota_blocked(
        &self,
        model: &ModelId,
        blocks: &[switchyard_llm_client::CredentialBlock],
    ) -> bool {
        let Some(credential) = self.target_credentials.get(model).map(String::as_str) else {
            return false;
        };
        blocks.iter().any(|block| {
            block.blocks(credential, model.as_str())
                && matches!(
                    block.kind,
                    switchyard_llm_client::CredentialBlockKind::FreeQuota
                        | switchyard_llm_client::CredentialBlockKind::ModelQuota
                        | switchyard_llm_client::CredentialBlockKind::WholeQuota
                )
        })
    }

    /// Enables measured ordering among the candidates this route has already admitted.
    ///
    /// Set at load from the route's `selection` policy. Without it the route walks its
    /// declared order, unchanged.
    pub fn with_measured_selection(
        mut self,
        selector: Arc<crate::selection::MeasuredSelector>,
    ) -> Self {
        self.selector = Some(selector);
        self
    }

    /// Declares that this route's candidates come from `groups`, ranked as peers.
    ///
    /// The groups are labels on the targets; this route does not choose between the labels.
    /// It records them so a decision record can say which pool a request was ranked in, and
    /// so the provider-neutral ranking policy is a property of the route rather than of the
    /// spelling of its candidate list.
    pub fn with_pool_groups(mut self, groups: Vec<String>) -> Self {
        self.pool_groups = Some(groups);
        self
    }

    /// The ranking policy this route uses.
    ///
    /// A pool route ranks peers: no group is a stage, and the manifest's order is not a
    /// decision. Every other route keeps the historical policy unchanged.
    fn ranking_policy(&self) -> crate::selection::RankingPolicy {
        if self.pool_groups.is_some() {
            crate::selection::RankingPolicy::PEER
        } else {
            crate::selection::RankingPolicy::LEGACY
        }
    }

    /// Whether this route is a provider-neutral pool, for reports and tests.
    pub fn is_peer_pool(&self) -> bool {
        self.pool_groups.is_some()
    }

    /// The groups this route's pool is built from, when it declares one.
    pub fn pool_groups(&self) -> Option<&[String]> {
        self.pool_groups.as_deref()
    }

    /// Records the daily request allowance behind each target model id.
    ///
    /// Selection and dispatch must read one budget: a candidate is offered later only
    /// because the same counter that will reserve the request says it has room.
    pub fn with_daily_allowances(
        mut self,
        allowances: Arc<HashMap<ModelId, Arc<switchyard_llm_client::DailyRequestAllowance>>>,
    ) -> Self {
        self.daily_allowances = allowances;
        self
    }

    /// Records the `llm_client` credential serving each target model id.
    ///
    /// Resolved once at load from configuration, so a composite never has to infer
    /// credential identity from a wire error.
    pub fn with_target_credentials(
        mut self,
        credentials: std::collections::HashMap<ModelId, String>,
    ) -> Self {
        self.target_credentials = credentials;
        self
    }

    /// The `llm_client` credential serving `model`, when this route carries one.
    pub fn target_credential(&self, model: &ModelId) -> Option<&str> {
        self.target_credentials.get(model).map(String::as_str)
    }

    /// The first target model id this route would call, when it carries any.
    ///
    /// The credential-aware composite uses this as the representative target: the child
    /// would call it first, so a credential-level block is judged against it.
    pub fn first_target_model(&self) -> Option<ModelId> {
        self.models.first_model()
    }

    /// The target models a set of credential blocks still leaves servable.
    ///
    /// Judged per target, on the credential that target actually draws on. A mixed-provider
    /// route has targets on more than one credential, so "the route's credential" is not a
    /// fact: a block on one provider's account must leave every peer on another provider
    /// servable, and a block on one model's own budget must leave its siblings servable.
    ///
    /// A target with no resolved credential is left servable, because a block that cannot be
    /// attributed to it is not evidence against it.
    pub fn servable_after_blocks(
        &self,
        blocks: &[switchyard_llm_client::CredentialBlock],
    ) -> Vec<ModelId> {
        self.models
            .all_models()
            .into_iter()
            .filter(|model| !self.target_is_blocked(model, blocks))
            .collect()
    }

    /// Whether any block covers `model`, on the credential that model draws on.
    pub fn target_is_blocked(
        &self,
        model: &ModelId,
        blocks: &[switchyard_llm_client::CredentialBlock],
    ) -> bool {
        let Some(credential) = self.target_credentials.get(model).map(String::as_str) else {
            return false;
        };
        blocks
            .iter()
            .any(|block| block.blocks(credential, model.as_str()))
    }

    /// The exclusion code for the first block covering `model`, for the decision record.
    fn block_exclusion(
        &self,
        model: &ModelId,
        blocks: &[switchyard_llm_client::CredentialBlock],
    ) -> Option<&'static str> {
        let credential = self.target_credentials.get(model).map(String::as_str)?;
        blocks
            .iter()
            .find(|block| block.blocks(credential, model.as_str()))
            .map(switchyard_llm_client::CredentialBlock::exclusion_code)
    }

    /// The provider slug of the first target model id, when one exists.
    pub fn first_target_slug(&self) -> &str {
        self.first_target_model()
            .map(|model| model.as_str().to_owned())
            .unwrap_or_default()
            .leak()
    }

    /// Whether every servable target on this route draws on the free-model allowance.
    ///
    /// A free-scope quota blocks the route only when this is true; a route with a paid
    /// candidate still has a servable leg on the same credential.
    ///
    /// Judged with the same predicate the failing call used, so the two levels cannot
    /// disagree. Reading only the `:free` suffix made the provider's Free Models Router
    /// (`openrouter/free`) look paid, and a free-only route carrying it was then treated
    /// as having a paid leg: the child was entered after the account-wide allowance was
    /// gone and its first call was guaranteed to be rejected.
    pub fn all_targets_free(&self) -> bool {
        // The question this answers is which allowance the requests are charged against,
        // because that is what the caller (a composite holding a provider quota block) needs
        // to know. Declared tariff decides it; an undeclared target falls back to the
        // provider's own allowance scoping, which is a fact about the provider's billing
        // scope rather than a guess about cost. The economic question - is this request
        // free - is `Tariff::is_free`, asked separately by selection, and it never treats
        // `Unknown` as free.
        let facts = self.clients.target_capabilities();
        self.models.all_models().iter().all(|model| {
            let tariff = facts
                .get(model)
                .map_or(switchyard_llm_client::Tariff::Unknown, |facts| facts.tariff);
            tariff.draws_on_free_allowance(model.as_str())
        })
    }

    /// Returns the configured libsy algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }

    /// The reasoning policy this route imposes, overriding the target's factual policy.
    pub fn route_reasoning_policy(&self) -> crate::config::RouteReasoningPolicy {
        self.route_reasoning_policy
    }

    /// Estimated input tokens above which this route should prefer a wider-context leg.
    pub fn escalation_max_input_tokens(&self) -> Option<u64> {
        self.escalation_max_input_tokens
    }

    /// Whether this request is at or below the configured context-pressure threshold.
    ///
    /// The comparator is strictly greater-than, reproducing the historical
    /// `escalation_max_input_tokens` behaviour rather than approximating it. A request
    /// exactly at the threshold stays on the primary leg.
    pub fn under_context_pressure(&self, estimated_input_tokens: u64) -> bool {
        self.escalation_max_input_tokens
            .is_some_and(|threshold| estimated_input_tokens > threshold)
    }

    /// Returns model-list capability metadata.
    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    /// Records where this route's work is performed, resolved from configuration.
    pub fn with_locality(mut self, locality: Locality) -> Self {
        self.locality = locality;
        self
    }

    /// Marks every completion target of this route as held, with the reason to report.
    ///
    /// The route stays registered so an alias that used to work keeps a specific refusal
    /// instead of becoming an unknown model; it serves nothing, because the candidate set
    /// handed to the algorithm has already had the held targets removed at load.
    pub fn with_held(mut self, held: HeldRoute) -> Self {
        self.held = Some(held);
        self
    }

    /// The holding record, when this route has no servable target left.
    pub fn held(&self) -> Option<&HeldRoute> {
        self.held.as_ref()
    }

    /// Why this route is held, when it is.
    pub fn held_reason(&self) -> Option<&str> {
        self.held.as_ref().map(|held| held.reason.as_str())
    }

    /// Returns where this route's work is performed.
    pub fn locality(&self) -> Locality {
        self.locality
    }

    /// Returns the forwarded caller credential family.
    pub fn caller_auth(&self) -> Option<CallerAuthKind> {
        self.caller_auth
    }

    /// Resolves a selected model to this route's non-secret target metadata.
    pub(crate) fn decision_target(&self, model: &ModelId) -> Option<DecisionTarget> {
        self.decision_targets
            .iter()
            .find(|target| target.model == *model)
            .cloned()
    }

    /// Returns the models grouped for one algorithm execution.
    pub fn models(&self) -> &RuntimeModels {
        &self.models
    }

    /// Rejects a caller format incompatible with forwarded credentials.
    pub fn check_caller_format(&self, input_format: WireFormat) -> Result<(), RunnerError> {
        if let Some(kind) = self.caller_auth
            && !kind.accepts(input_format)
        {
            return Err(RunnerError::IncompatibleCallerFormat(kind));
        }
        Ok(())
    }

    /// Executes the configured route without consuming or proxying streamed responses.
    /// The model this route's client router holds provider state for, when a request
    /// continues a conversation this route previously served.
    pub fn continuation_owner(
        &self,
        request: &switchyard_protocol::Request,
    ) -> Option<switchyard_protocol::ModelId> {
        self.clients.continuation_owner(request)
    }

    /// Whether this route would refuse the request before contacting any provider.
    ///
    /// Always `false` now: the egress gate and its clearance scheme are removed, so no
    /// request is ever refused here. The method stays because a composite asks the question
    /// of every child and the answer is still a routing input - it is simply no longer a
    /// refusal. Keeping the seam means a future caller-facing gate has one place to appear,
    /// and keeps the composite's skip path exercised rather than deleted.
    pub fn payload_refused(&self, _request: &Request) -> bool {
        false
    }

    /// Whether no candidate of this route can hold the request, and why.
    ///
    /// The physical counterpart of [`Self::payload_refused`]: a composite needs to know
    /// before the call, because a child that cannot fit the request is ineligible rather
    /// than failing. Without it the child's own `UnsupportedCapability` error ends the
    /// whole composite — the request never reaches a wider sibling — which is precisely
    /// the case an aggregate route exists to handle.
    ///
    /// Returns the binding limit's name, or `None` when at least one candidate fits or when
    /// a candidate states no limits at all. A target that states nothing is unconstrained,
    /// not zero.
    pub fn physical_ineligibility(&self, request: &Request) -> Option<&'static str> {
        let facts = self.clients.target_capabilities();
        let candidates = self.models.all_models();
        if candidates.is_empty() {
            return None;
        }
        let required = required_tokens(request);
        let mut binding: Option<&'static str> = None;
        for model in &candidates {
            let facts = facts.get(model)?;
            if let Some(window) = facts.context_window
                && required > window
            {
                binding = Some("context_window");
                continue;
            }
            if let Some(ceiling) = facts.max_input_tokens
                && required > ceiling
            {
                binding = binding.or(Some("input_tokens_per_minute"));
                continue;
            }
            // The margin-adjusted bound is the same limit reported under the same name: a
            // caller sees "this is why your request does not fit here", and which side of
            // the margin it fell on is an operator concern, not a caller one.
            if let Some(safe) = facts.safe_input_tokens
                && required > safe
            {
                binding = binding.or(Some("input_tokens_per_minute"));
                continue;
            }
            // The same floor the pre-dispatch filter applies, named here as well so a
            // composite treats it as this child being unable to hold the request rather
            // than as a child that failed.
            if let (Some(floor), Some(reservation)) = (
                facts.min_output_tokens,
                request.llm_request.output.max_output_tokens,
            ) && reservation < floor
            {
                binding = binding.or(Some("output_budget_below_thinking_floor"));
                continue;
            }
            // Mirrors the dispatch filter's reading of a caller-supplied effort the provider
            // refuses, so a composite treats this child as unable to hold the request rather
            // than as a child that failed.
            if let Some(effort) = request.llm_request.reasoning.effort.as_deref()
                && facts.unsupported_reasoning_effort.contains(effort)
            {
                binding = binding.or(Some("unsupported_reasoning_effort"));
                continue;
            }
            return None;
        }
        binding
    }

    pub async fn execute(
        &self,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<RunOutput, RunnerError> {
        self.execute_with_blocks(request, observer, &[]).await
    }

    /// Execute with credential blocks carried in from the surrounding composite.
    ///
    /// A block excludes the candidates it covers and nothing else: the survivors still
    /// compete in the same order, under the same gates. That is what lets one provider's
    /// exhausted account or one model's spent budget leave its peers in the pool instead of
    /// condemning the whole child — which is the mixed-credential defect this parameter
    /// exists to remove.
    pub async fn execute_with_blocks(
        &self,
        request: Request,
        observer: Option<RunObserver>,
        blocks: &[switchyard_llm_client::CredentialBlock],
    ) -> Result<RunOutput, RunnerError> {
        // A held route has no servable target at all, so the refusal is decided before any
        // request shaping: nothing is inspected, planned or dispatched. Reported here rather
        // than as an ineligible candidate so the caller gets the provider's evidence instead
        // of "requires ...", which would imply the request could be changed to fit.
        if let Some(held) = &self.held {
            return Err(RunnerError::Held {
                model: held.model.clone(),
                reason: held.reason.clone(),
            });
        }
        // Ordered candidates are filtered by what each target can actually serve, before any
        // provider is contacted. Declared order is preserved among the survivors, and the
        // native execution, retry, cooldown and fallback machinery still decides among them,
        // so this adds no second fallback engine and issues no request to a target that
        // cannot answer it.
        let facts = self.clients.target_capabilities().clone();
        // The egress gate and its clearance scheme are removed. Every candidate is judged
        // only on whether it can physically serve the request; no payload is inspected and
        // nothing is refused for its content.
        let raw = request
            .raw_request
            .clone()
            .unwrap_or(serde_json::Value::Null);
        let estimated_input_tokens = crate::request_fit::estimate_input_tokens(&raw);
        let required_tokens = required_tokens(&request);
        // Every hard gate returns a stable code rather than a bare `false`, so a candidate
        // that was never contacted can still say why. The codes are a closed vocabulary:
        // they name a rule, never a payload. `unsupported_capability` already answers with
        // one ("vision", "tool_calling", "video", "reasoning", "image_carrier", ...).
        // A provider-neutral pool is a *free* pool: its membership is the eligible free leaves
        // of each provider, so a candidate whose tariff is not positively declared free is
        // excluded before it can be ranked. Without this, a leg listed by mistake would
        // compete one tier below the free candidates and then serve whenever the free pool
        // was busy, which is a paid escalation nobody approved. `unknown` fails closed for
        // the same reason: an undeclared tariff is not evidence of free entitlement.
        let pool_members_only_free = self.is_peer_pool();
        // The block gate and the capability gate are separate questions, and the terminal
        // leg's exhaustion rule needs both answers. `capability_gate` asks "could this
        // target serve this request at all"; `verdict` adds "and is it allowed to right
        // now". A pool whose every capable leg is merely blocked is exhausted; a pool where
        // no leg is capable is a request nobody can serve, and must be refused rather than
        // handed to a paid leg as if capacity had run out.
        let capability_gate = |model: &switchyard_protocol::ModelId| -> Result<(), &'static str> {
            let Some(facts) = facts.get(model) else {
                // Uncharacterised rather than incapable: permitted, except for video,
                // which no target may receive without stating that it can read it.
                return if crate::request_fit::requires_video(&request.llm_request, &raw) {
                    Err("video_requires_declaration")
                } else {
                    Ok(())
                };
            };
            if pool_members_only_free && !facts.tariff.is_free() {
                return Err("not_free_entitled");
            }
            // Physical fit, judged on facts the deployment states about one backing
            // model. A request that cannot fit is not a request the target should be
            // asked to refuse: the route advances instead, which is what makes an
            // aggregate window honest across models of different sizes.
            if let Some(window) = facts.context_window
                && required_tokens > window
            {
                return Err("context_window");
            }
            if let Some(ceiling) = facts.max_input_tokens
                && required_tokens > ceiling
            {
                return Err("max_input_tokens");
            }
            // The deployment's own admission bound: an estimate padded to the
            // admission margin must fit the target's minute window, or the send path
            // will refuse this candidate anyway. Filtering here is what makes the
            // refusal useful — the route advances to a sibling with a wider window
            // instead of discovering the refusal after it has spent eligibility.
            if let Some(safe) = facts.safe_input_tokens
                && required_tokens > safe
            {
                return Err("minute_window_too_small");
            }
            // A target whose own reasoning is charged against the caller's output
            // reservation cannot answer under a smaller reservation: the thinking runs out
            // of budget before the answer starts. Skipping it costs nothing and spares the
            // provider a request whose outcome the deployment can already predict, while
            // the sibling that counts reasoning as output stays eligible.
            if let (Some(floor), Some(reservation)) = (
                facts.min_output_tokens,
                request.llm_request.output.max_output_tokens,
            ) && reservation < floor
            {
                return Err("output_budget_below_thinking_floor");
            }
            // The caller's own reasoning effort is a requirement, not a hint. When the
            // provider refuses that exact value the request cannot be honoured here, and
            // stripping it would leave the provider's native default running while the
            // response implied the caller's choice had been applied. Skipping the candidate
            // lets an aggregate route advance to a leg that accepts the value; a directly
            // addressed leaf then refuses with this name, before any request is sent.
            //
            // Only the caller's normalized field is read. An effort this deployment injected
            // from route or target configuration is not a caller requirement, and is still
            // dropped at the send boundary where it was introduced.
            if let Some(effort) = request.llm_request.reasoning.effort.as_deref()
                && facts.unsupported_reasoning_effort.contains(effort)
            {
                return Err("unsupported_reasoning_effort");
            }
            // An explicit `false` is a refusal; `None` would mean "undeclared",
            // which the predicate treats as permitted, so the flag must be carried
            // as a concrete boolean.
            let capabilities = crate::ModelCapabilities {
                vision: Some(!facts.no_vision),
                tool_calling: Some(!facts.no_tool_calling),
                // A target that cannot carry a caller's reasoning control must not
                // be handed one: the wire would either reject it or silently enforce
                // something the deployment never agreed to. The route advances to a
                // leg that can carry it, and a directly addressed leaf refuses.
                reasoning: Some(!facts.no_reasoning_control),
                // Carried as a concrete boolean like the other facts, so an
                // undeclared target is treated as video-incapable here.
                video: Some(!facts.no_video),
                // A target is video-capable at all only if some carrier survives;
                // otherwise the boolean above already refuses it and naming a
                // carrier would read as a capability it does not have.
                video_carrier: carrier_from_facts(facts),
                image_carrier: image_carrier_from_facts(facts),
                ..crate::ModelCapabilities::default()
            };
            match crate::request_fit::unsupported_capability(
                capabilities,
                &request.llm_request,
                &raw,
            ) {
                Some(capability) => Err(capability),
                None => Ok(()),
            }
        };
        // The block gate wraps the capability gate, so an excluded candidate's reason names
        // the credential when it was the credential, and the request when it was the
        // request. Judged on this target's own credential, so a block on one provider's
        // account or one model's budget never reaches a peer that is still healthy, and
        // never becomes a paid escalation either: a paid leg is only reached by the
        // route's own crossing policy.
        let verdict = |model: &switchyard_protocol::ModelId| -> Result<(), &'static str> {
            match self.block_exclusion(model, blocks) {
                Some(code) => Err(code),
                None => capability_gate(model),
            }
        };
        let considered = self.models.all_models();
        let verdicts: Vec<Result<(), &'static str>> = considered.iter().map(&verdict).collect();
        // Snapshot the two answers the terminal leg's exhaustion rule needs, while `request`
        // is still borrowed read-only. Both gates read request facts, so they cannot be
        // re-evaluated after the request is moved into the reasoning-policy rewrite below.
        let capable_legs: Vec<switchyard_protocol::ModelId> = considered
            .iter()
            .filter(|model| capability_gate(model).is_ok())
            .cloned()
            .collect();
        let dispatchable_legs: HashSet<switchyard_protocol::ModelId> = capable_legs
            .iter()
            .filter(|model| verdict(model).is_ok())
            .cloned()
            .collect();
        // One snapshot per tier: capability is "could this leg serve this request at all",
        // dispatchability adds the block gate. Each tier boundary is judged by both, so both
        // are kept for every tier, not only for the primary pool.
        let terminal_capable: Vec<Vec<switchyard_protocol::ModelId>> = self
            .terminal
            .iter()
            .map(|tier| {
                tier.all_models()
                    .into_iter()
                    .filter(|model| capability_gate(model).is_ok())
                    .collect()
            })
            .collect();
        let terminal_servable: Vec<HashSet<switchyard_protocol::ModelId>> = terminal_capable
            .iter()
            .map(|capable| {
                capable
                    .iter()
                    .filter(|model| verdict(model).is_ok())
                    .cloned()
                    .collect()
            })
            .collect();
        // The same tier boundaries judged by the block gate, keeping each target's refusal
        // code rather than only the survivors. A tier that admitted nothing has to be able to
        // say whether that was a size fact -- in which case the ordered walk continues -- or
        // something the next tier would re-learn at a price.
        let terminal_verdicts: Vec<Vec<Result<(), &'static str>>> = self
            .terminal
            .iter()
            .map(|tier| {
                tier.all_models()
                    .into_iter()
                    .map(|model| verdict(&model))
                    .collect()
            })
            .collect();
        // A request that no primary candidate can hold is a different claim from a pool that
        // ran out, and the route has to say which one it is making. This is computed before
        // the request is moved into the reasoning rewrite, and it is the only place a size
        // escalation is authorized: `route_accepts` is the route's own declaration read
        // against the request, so a request the lane never promised to serve (vision, a
        // reasoning control it refuses) is refused here rather than handed to a metered tier.
        let route_accepts = crate::request_fit::unsupported_capability(
            self.capabilities,
            &request.llm_request,
            &raw,
        )
        .is_none();
        let fit_escalation = self
            .fit_escalation
            .then(|| authorize_fit_escalation(route_accepts, &verdicts))
            .flatten();
        let models = self.models.filtered(&|model| verdict(model).is_ok());
        // Among candidates that survived every hard gate, a route may prefer what this
        // deployment has actually observed. This reorders one already-admitted list and
        // cannot add, drop or resurrect a candidate, so a privacy, capability, fit or
        // quota refusal is still final.
        let demand = crate::selection::Demand {
            required_tokens,
            needs_tools: crate::request_fit::wants_tools(&request.llm_request, &raw),
        };
        // Scores are kept even when no selector is configured, so a report can still name
        // each candidate's tier and remaining budget. Only the reorder depends on the
        // selector being present.
        let mut scores: Vec<crate::selection::CandidateScore> = Vec::new();
        let models = match self.selector.as_ref() {
            Some(selector) => {
                let candidates: Vec<crate::selection::Candidate> = models
                    .all_models()
                    .into_iter()
                    .map(|model| {
                        let facts = facts.get(&model);
                        // The tier is the declared tariff. `Unknown` ranks as metered: it is
                        // not evidence of free eligibility, and a free pool that admitted it
                        // would be spending a request on a cost nobody declared.
                        crate::selection::Candidate {
                            tier: match facts.map(|facts| facts.tariff) {
                                Some(switchyard_llm_client::Tariff::Free) => {
                                    crate::selection::CostTier::Free
                                }
                                _ => crate::selection::CostTier::Paid,
                            },
                            remaining_today: self
                                .daily_allowances
                                .get(&model)
                                .map(|allowance| allowance.remaining()),
                            max_input_tokens: facts.and_then(|facts| facts.max_input_tokens),
                            context_window: facts.and_then(|facts| facts.context_window),
                            model,
                        }
                    })
                    .collect();
                let declared = self.models.all_models();
                scores =
                    selector.explain_with(&candidates, demand, &declared, self.ranking_policy());
                let ordered: Vec<switchyard_protocol::ModelId> =
                    scores.iter().map(|score| score.model.clone()).collect();
                let unchanged = candidates
                    .iter()
                    .zip(ordered.iter())
                    .all(|(candidate, model)| &candidate.model == model);
                if unchanged {
                    models
                } else {
                    models.reordered(&ordered)
                }
            }
            None => models,
        };
        // Under context pressure the declared escalation leg is offered first, before any
        // provider is contacted. This reorders one list; it does not add a second
        // escalation engine, and the native retry, cooldown and fallback machinery still
        // decides among the survivors.
        //
        // The promotion is only real if the leg survived capability filtering *and* was
        // already in the list, so that is what decides whether to reorder and what to
        // report. Reporting on intent alone would over-count: an ineligible or unlisted leg
        // leaves the order untouched while the counter still claimed a wider leg was
        // preferred.
        let promotion = self.escalation_target.as_ref().filter(|escalation_target| {
            self.under_context_pressure(estimated_input_tokens)
                && models.contains(escalation_target)
        });
        let promoted = promotion.cloned();
        let models = match promotion {
            Some(escalation_target) => {
                if let Some(observer) = observer.as_ref() {
                    observer(RunObservation::ContextPressurePromotion {
                        promoted: escalation_target.clone(),
                        estimated_input_tokens,
                    });
                }
                models.promoting(std::slice::from_ref(escalation_target))
            }
            None => models,
        };
        let request =
            match self.route_reasoning_policy {
                RouteReasoningPolicy::Inherit => request,
                RouteReasoningPolicy::Medium => request
                    .with_route_reasoning_policy(switchyard_protocol::ReasoningPolicy::Enabled),
                // `High` needs the graded level as well as the on/off intent: on a dialect that
                // only grades effort the policy alone has no wire form, so the request also
                // carries the effort for the client to write.
                RouteReasoningPolicy::High => request
                    .with_route_reasoning_policy(switchyard_protocol::ReasoningPolicy::Enabled)
                    .with_route_reasoning_effort(
                        crate::config::route_policy_effort(RouteReasoningPolicy::High)
                            .unwrap_or_default(),
                    ),
                RouteReasoningPolicy::None => request
                    .with_route_reasoning_policy(switchyard_protocol::ReasoningPolicy::Disabled),
            };
        // The decision record is built here, before any exit, so a request refused before a
        // single provider was contacted can still say which candidate failed which rule.
        // Every field is a code, a count or a position: no prompt text, no provider id.
        let dispatched = models.all_models();
        let scores: HashMap<switchyard_protocol::ModelId, crate::selection::CandidateScore> =
            scores
                .into_iter()
                .map(|score| (score.model.clone(), score))
                .collect();
        let rank_of: HashMap<switchyard_protocol::ModelId, u32> = dispatched
            .iter()
            .enumerate()
            .map(|(rank, model)| (model.clone(), rank as u32))
            .collect();
        let mut trace = SelectionTrace {
            algorithm: self.algorithm.name().to_string(),
            considered: considered.iter().map(ToString::to_string).collect(),
            candidates: considered
                .iter()
                .zip(verdicts.iter())
                .map(|(model, verdict)| {
                    let facts = facts.get(model);
                    let score = scores.get(model);
                    // The selector's own figures win where it ran; without a selector the
                    // record still names the tier and the pressure the gates judged by,
                    // rather than reporting a zero it never measured.
                    // The tier a report prints is the declared tariff, including the case
                    // where the deployment declared nothing: `unknown` is a distinct answer
                    // from `paid`, and collapsing it into `paid` would hide a cost the
                    // deployment never verified.
                    let tariff =
                        facts.map_or(switchyard_llm_client::Tariff::Unknown, |facts| facts.tariff);
                    CandidateTrace {
                        model: model.to_string(),
                        eligible: verdict.is_ok(),
                        exclusion: verdict.err(),
                        tier: tariff.as_str(),
                        rank: rank_of.get(model).copied(),
                        remaining_today: score.and_then(|score| score.remaining_today),
                        pressure_permille: score.map_or_else(
                            || {
                                crate::selection::pressure_permille(
                                    demand.required_tokens(),
                                    facts.and_then(|facts| {
                                        facts.max_input_tokens.or(facts.context_window)
                                    }),
                                )
                            },
                            |score| score.pressure_permille,
                        ),
                        demotion: score.and_then(|score| score.demotion),
                    }
                })
                .collect(),
            dispatched: dispatched.iter().map(ToString::to_string).collect(),
            attempts: Vec::new(),
            served: None,
            outcome: "no_eligible_candidate",
            demand_tokens: demand.required_tokens(),
            estimated_input_tokens,
            promotion: promoted.map(|model| model.to_string()),
            pool: self.pool_groups.as_ref().map(|groups| groups.join(",")),
            terminal: self
                .terminal_models()
                .iter()
                .map(ToString::to_string)
                .collect(),
            terminal_tiers: self
                .terminal
                .iter()
                .map(|tier| tier.all_models().iter().map(ToString::to_string).collect())
                .collect(),
            terminal_crossed: false,
            terminal_reason: None,
            terminal_tier: None,
            // Filled only as the walk actually reaches a tier and finds it empty. A tier the
            // walk never reached is absent, so `terminal_crossed = false` plus an empty list
            // means "the tiers were never consulted", not "the tiers were consulted and
            // refused". The precomputed `terminal_verdicts` are deliberately NOT recorded here:
            // a gate being evaluated is not the same as a boundary being crossed.
            terminal_skipped: Vec::new(),
        };
        if models.is_empty() && fit_escalation.is_none() {
            // Nothing was dispatched, so the record is the whole story: exclusion codes and
            // an empty attempt list. Emitted before either refusal so the caller and the
            // routing log name the same reason.
            emit_selection(observer.as_ref(), &trace);
            // Nothing in this route can serve the request, which is the caller's request
            // rather than a deployment fault. The skipped targets get no provider call.
            // Name what the request needs, not what the route happens to lack: the route's
            // own declaration may be silent while a specific target was the blocker.
            let capability = self
                .clients
                .target_capabilities()
                .iter()
                .find_map(|(_model, facts)| {
                    // Mirror the eligibility predicate's reading of these facts, or the
                    // refusal would name a different capability than the one that
                    // actually excluded the candidate.
                    let capabilities = crate::ModelCapabilities {
                        vision: Some(!facts.no_vision),
                        tool_calling: Some(!facts.no_tool_calling),
                        video: Some(!facts.no_video),
                        video_carrier: carrier_from_facts(facts),
                        image_carrier: image_carrier_from_facts(facts),
                        ..crate::ModelCapabilities::default()
                    };
                    crate::request_fit::unsupported_capability(
                        capabilities,
                        &request.llm_request,
                        &raw,
                    )
                })
                .or_else(|| {
                    crate::request_fit::unsupported_capability(
                        self.capabilities,
                        &request.llm_request,
                        &raw,
                    )
                })
                // A request that fits every capability and still found nothing was excluded
                // on size. Naming the binding limit is the difference between a caller who
                // knows to send less and one who reads "ineligible" and retries the same
                // payload: the physical gate is the only other reason a candidate drops out,
                // and it is reported only when no candidate fits at all.
                .or_else(|| self.physical_ineligibility(&request))
                .unwrap_or("an eligible route");
            return Err(RunnerError::unsupported_capability(
                self.algorithm.name(),
                capability,
            ));
        }
        // The measurement rides on the observation the run already emits, so no separate
        // timing path exists to disagree with the recorded one.
        //
        // The caller's observer is kept separately: the run consumes its own copy down into
        // the algorithm, and the selection record must be emitted after the terminal attempt
        // has been observed, or it would report a dispatch with no outcomes.
        let report = observer.clone();
        let attempts: Arc<Mutex<Vec<AttemptTrace>>> = Arc::new(Mutex::new(Vec::new()));
        let run_observer = Some(recording_observer(
            Arc::clone(&attempts),
            self.selecting_observer(observer),
        ));
        // A terminal leg is a last resort. Keep a copy of the shaped request only when one
        // exists, so the common route pays nothing for a capability it does not have.
        let terminal_request = (!self.terminal.is_empty()).then(|| request.clone());
        // A pool that admitted nothing has nothing to dispatch, so there is no call to fail.
        // The error exists only to carry the refusal if no authorized tier can hold the
        // request either; `attempts` stays empty, which is what makes the record say
        // `no_eligible_candidate` rather than `failed`.
        let mut result = if models.is_empty() {
            Err(RunnerError::unsupported_capability(
                self.algorithm.name(),
                self.physical_ineligibility(&request)
                    .unwrap_or("an eligible route"),
            ))
        } else {
            switchyard_llm_client::run(
                Arc::clone(&self.algorithm),
                self.clients.clone(),
                request,
                Arc::new(models.clone()),
                run_observer.clone(),
            )
            .await
            .map_err(RunnerError::from)
        };
        if let Some(terminal_request) = terminal_request
            && result.is_err()
        {
            // The primary pool could not answer. Before anything is spent on the terminal
            // leg, the claim has to be true of the same gates the order used and of what the
            // dispatch actually observed. A single rate limit, a policy refusal, or a request
            // that never fit is not that claim -- unless this route declared that a size
            // refusal is exactly what the tiers are for, in which case the claim is
            // "no candidate in the pool can hold it" and it is recorded under its own reason.
            //
            // Each boundary is authorized separately against the set it is leaving, never
            // against the union: tier 2 is only reachable once tier 1 has itself been proved
            // out, which is the entire difference between an ordered declaration and a peer
            // pool. The union would let "the second tier looks cheaper" skip the first.
            let mut authorization = fit_escalation.or_else(|| {
                self.terminal_authorization(
                    &capable_legs,
                    &dispatchable_legs,
                    &attempts.lock().clone(),
                    blocks,
                )
            });
            for (index, tier) in self.terminal.iter().enumerate() {
                let Some(reason) = authorization else { break };
                let servable = tier.filtered(&|model| terminal_servable[index].contains(model));
                if servable.is_empty() {
                    // This tier has nothing to dispatch, so it cannot answer and the request
                    // must not be re-pointed at an empty set. Its own gates still decide
                    // whether the next boundary is reachable: a declared tier that no
                    // request could ever fit is a coverage gap, not exhaustion.
                    //
                    // The walk has reached this tier, so its exclusions are a traversal fact
                    // and are recorded -- a target absent from the record was never consulted.
                    for (model, gate) in tier
                        .all_models()
                        .into_iter()
                        .zip(terminal_verdicts[index].iter())
                    {
                        if let Err(code) = gate {
                            trace.terminal_skipped.push((model.to_string(), *code));
                        }
                    }
                    // A size boundary keeps the walk alive on its own claim. Every target
                    // here being unable to *hold* the request is the same fact the pool
                    // reported, and the next tier is asked the same question; any other
                    // refusal ends the walk, because the next tier would be asked to settle
                    // something about the request rather than about capacity.
                    authorization = fit_escalation
                        .filter(|_| tier_cannot_hold(&terminal_verdicts[index]))
                        .or_else(|| {
                            self.terminal_authorization(
                                &terminal_capable[index],
                                &terminal_servable[index],
                                &attempts.lock().clone(),
                                blocks,
                            )
                        });
                    continue;
                }
                trace.terminal_crossed = true;
                // The entry authorization is what an operator needs: it is the claim that
                // opened the terminal leg at all. A later boundary re-derives its own reason
                // because it is leaving a different set, but overwriting this would lose the
                // fact that the walk started on size rather than on capacity -- both true,
                // and only the first explains why the pool was never called.
                if trace.terminal_reason.is_none() {
                    trace.terminal_reason = Some(reason);
                }
                trace.terminal_tier = Some(index + 1);
                result = switchyard_llm_client::run(
                    Arc::clone(&self.algorithm),
                    self.clients.clone(),
                    terminal_request.clone(),
                    Arc::new(servable),
                    run_observer.clone(),
                )
                .await
                .map_err(RunnerError::from);
                if result.is_ok() {
                    break;
                }
                // This tier answered with a failure. Whether the next one may be consulted is
                // a fact about THIS tier's legs and this request's attempts at them, so the
                // claim is re-derived rather than carried over from the boundary above.
                authorization = self.terminal_authorization(
                    &terminal_capable[index],
                    &terminal_servable[index],
                    &attempts.lock().clone(),
                    blocks,
                );
            }
        }
        {
            trace.attempts = attempts.lock().clone();
            match &result {
                Ok((selected_model, _)) => {
                    // Name the model that answered, not the one the algorithm picked first.
                    // A 5xx or an empty answer on the first candidate makes the run fall
                    // through to the next, and reporting the first would credit a model that
                    // answered nothing -- measured 2026-11-21, six `smart-nt` records named
                    // `gemma-4-31b-it` as served while `gemini-3.x-flash-lite` answered.
                    trace.served = trace
                        .attempts
                        .iter()
                        .rev()
                        .find(|attempt| attempt.outcome == "served")
                        .map(|attempt| attempt.model.clone())
                        .or_else(|| Some(selected_model.to_string()));
                    trace.outcome = "served";
                }
                // Nothing was dispatched at all, so nothing failed: every candidate the route
                // could reach was unable to hold the request. Reporting `failed` here would
                // blame a deployment that was never asked, and would hide the request-fit
                // answer the caller needs.
                Err(_) if trace.attempts.is_empty() => trace.outcome = "no_eligible_candidate",
                // Every candidate was eligible and at least one was dispatched, so an error
                // here is a deployment fault rather than a request the route could not fit.
                Err(_) => trace.outcome = "failed",
            }
        }
        emit_selection(report.as_ref(), &trace);
        let (selected_model, response) = result?;
        Ok(RunOutput {
            selected_model,
            response,
        })
    }

    /// Composes the route's own observer in front of the caller's, when this route measures.
    fn selecting_observer(&self, observer: Option<RunObserver>) -> Option<RunObserver> {
        let Some(selector) = self.selector.as_ref().map(Arc::clone) else {
            return observer;
        };
        Some(match observer {
            Some(observer) => Arc::new(move |observation: RunObservation| {
                selector.observe(&observation);
                observer(observation);
            }),
            None => Arc::new(move |observation: RunObservation| selector.observe(&observation)),
        })
    }

    /// Completes routing-time calls without serving a post-routing completion.
    pub async fn decide(&self, request: Request) -> Result<RoutingOutcome, RunnerError> {
        if let Some(held) = &self.held {
            return Err(RunnerError::Held {
                model: held.model.clone(),
                reason: held.reason.clone(),
            });
        }
        switchyard_llm_client::decide(
            Arc::clone(&self.algorithm),
            self.clients.clone(),
            request,
            Arc::clone(&self.models),
        )
        .await
        .map_err(Into::into)
    }

    /// Executes a model-bearing provider operation through a compatible target.
    pub async fn call_auxiliary(
        &self,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value, RunnerError> {
        let target = match operation {
            AuxiliaryOperation::AnthropicCountTokens => &self.anthropic_auxiliary_target,
            AuxiliaryOperation::ResponsesInputTokens | AuxiliaryOperation::ResponsesCompact => {
                &self.responses_auxiliary_target
            }
        }
        .as_ref()
        .ok_or(RunnerError::AuxiliaryUnsupported)?;
        target
            .client
            .call_auxiliary(&target.model, request, operation)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod context_policy_tests {
    use super::*;
    use crate::RouteReasoningPolicy;

    /// Only the boundary predicate and the stored policy are under test here; both are
    /// plain value logic, so this builds a route with the cheapest algorithm the runner
    /// already uses in its own tests.
    fn build(threshold: Option<u64>, policy: RouteReasoningPolicy) -> Route {
        Route::new(
            Arc::new(libsy::Passthrough),
            ClientRouter::new(std::collections::HashMap::new()),
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
            RuntimeModels::default(),
            policy,
            threshold,
            None,
        )
    }

    /// The recovered historical comparator was strictly greater-than: a request exactly
    /// at the threshold stays on the primary leg and only 61,441 escalates.
    #[test]
    fn the_context_boundary_is_strictly_greater_than() {
        let route = build(Some(61_440), RouteReasoningPolicy::Inherit);
        assert!(
            !route.under_context_pressure(61_440),
            "exactly at the threshold stays on the primary leg"
        );
        assert!(
            route.under_context_pressure(61_441),
            "one token over the threshold escalates"
        );
        assert!(
            !build(None, RouteReasoningPolicy::Inherit).under_context_pressure(1_000_000),
            "a route without a threshold never escalates on context pressure"
        );
    }

    /// The route's policy is what it imposes on a shared physical target.
    #[test]
    fn the_route_policy_is_what_the_route_imposes() {
        for policy in [
            RouteReasoningPolicy::Inherit,
            RouteReasoningPolicy::Medium,
            RouteReasoningPolicy::None,
        ] {
            assert_eq!(
                build(Some(61_440), policy).route_reasoning_policy(),
                policy,
                "the route keeps the policy it was configured with"
            );
        }
    }
}

#[cfg(test)]
mod carrier_tests {
    use super::MediaCarrier;

    /// A composite can only promise what its children between them accept, so the union
    /// states exactly that: `Both` absorbs either carrier and the two singletons meet in
    /// `Both`. Nothing narrower may come out of a union.
    #[test]
    fn the_union_is_the_least_set_covering_both() {
        use MediaCarrier::{Both, Inline, Remote};
        for (left, right, expected) in [
            (Inline, Inline, Inline),
            (Remote, Remote, Remote),
            (Both, Both, Both),
            (Inline, Remote, Both),
            (Remote, Inline, Both),
            (Both, Inline, Both),
            (Inline, Both, Both),
            (Remote, Both, Both),
            (Both, Remote, Both),
        ] {
            assert_eq!(left.union(right), expected, "{left:?} u {right:?}");
            // Union is commutative in effect as well as in claim.
            assert_eq!(right.union(left), expected, "{right:?} u {left:?}");
        }
    }

    /// `includes` is the question a composite's promise asks of a child, and it must be
    /// narrower than equality: `Both` is included only by `Both`, and a singleton is
    /// included by anything that admits it.
    #[test]
    fn includes_is_not_equality() {
        use MediaCarrier::{Both, Inline, Remote};
        assert!(Both.includes(Inline) && Both.includes(Remote) && Both.includes(Both));
        assert!(Inline.includes(Inline) && !Inline.includes(Remote) && !Inline.includes(Both));
        assert!(Remote.includes(Remote) && !Remote.includes(Inline) && !Remote.includes(Both));
    }
}

#[cfg(test)]
mod terminal_authorization_tests {
    use super::{AttemptTrace, HashSet, ModelId, authorize_terminal};
    use crate::config::TerminalPolicy;
    use std::collections::HashMap;

    fn legs(ids: &[(&str, &str)]) -> (Vec<ModelId>, HashMap<ModelId, String>) {
        let mut models = Vec::new();
        let mut credentials = HashMap::new();
        for (model, credential) in ids {
            let id = ModelId::from(*model);
            models.push(id.clone());
            credentials.insert(id, (*credential).to_string());
        }
        (models, credentials)
    }

    fn attempt(model: &str, outcome: &'static str) -> AttemptTrace {
        AttemptTrace {
            model: model.to_string(),
            phase: "routing",
            outcome,
            duration_ms: 1,
            prompt_tokens: 1,
            completion_tokens: 0,
        }
    }

    fn dispatchable(models: &[ModelId]) -> HashSet<ModelId> {
        models.iter().cloned().collect()
    }

    /// The historical route shape: every capacity *or availability* fault may escalate.
    fn authorize(
        models: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        credentials: &HashMap<ModelId, String>,
    ) -> Option<&'static str> {
        authorize_terminal(
            models,
            dispatchable,
            attempts,
            TerminalPolicy::Capacity,
            |model| credentials.get(model).cloned(),
            |_| false,
        )
    }

    /// The narrowed shape: only quota evidence may escalate.
    fn authorize_quota_only(
        models: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        credentials: &HashMap<ModelId, String>,
    ) -> Option<&'static str> {
        authorize_quota_only_with_blocks(models, dispatchable, attempts, credentials, |_| false)
    }

    fn authorize_quota_only_with_blocks(
        models: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        credentials: &HashMap<ModelId, String>,
        quota_blocked: impl Fn(&ModelId) -> bool,
    ) -> Option<&'static str> {
        authorize_terminal(
            models,
            dispatchable,
            attempts,
            TerminalPolicy::QuotaOnly,
            |model| credentials.get(model).cloned(),
            quota_blocked,
        )
    }

    /// The judge's shape: capacity or a real outage may escalate; a fault about the
    /// deployment's accounting or about this request's size may not.
    fn authorize_availability(
        models: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        credentials: &HashMap<ModelId, String>,
    ) -> Option<&'static str> {
        authorize_terminal(
            models,
            dispatchable,
            attempts,
            TerminalPolicy::Availability,
            |model| credentials.get(model).cloned(),
            |_| false,
        )
    }

    #[test]
    fn nothing_capable_never_authorizes() {
        let (models, credentials) = legs(&[("a", "A")]);
        assert_eq!(
            authorize(&[], &dispatchable(&models), &[], &credentials),
            None
        );
    }

    #[test]
    fn every_capable_leg_blocked_is_a_blocked_pool() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "A")]);
        assert_eq!(
            authorize(&models, &HashSet::new(), &[], &credentials),
            Some("primary_pool_blocked")
        );
    }

    #[test]
    fn one_capacity_failure_with_a_dead_credential_is_exhaustion() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "A")]);
        let attempts = vec![attempt("a", "http_429")];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            Some("primary_pool_exhausted")
        );
    }

    #[test]
    fn an_unattempted_leg_on_a_live_credential_is_not_exhaustion() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![attempt("a", "http_429")];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    #[test]
    fn a_request_fault_never_authorizes() {
        for outcome in [
            "http_403",
            "http_400",
            "empty_answer",
            "timeout",
            "context_window",
        ] {
            let (models, credentials) = legs(&[("a", "A")]);
            let attempts = vec![attempt("a", outcome)];
            assert_eq!(
                authorize(&models, &dispatchable(&models), &attempts, &credentials),
                None,
                "outcome {outcome} must not authorize the terminal leg"
            );
        }
    }

    #[test]
    fn a_mixed_run_with_one_request_fault_never_authorizes() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![attempt("a", "http_429"), attempt("b", "http_403")];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    #[test]
    fn an_unattributable_leg_is_never_given_a_dead_credential() {
        // No credential mapping: a skip cannot be explained, so it is not exhaustion.
        let (models, _) = legs(&[("a", "A"), ("b", "A")]);
        let attempts = vec![attempt("a", "http_429")];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &HashMap::new()),
            None
        );
    }

    #[test]
    fn every_leg_attempted_and_capacity_failed_is_exhaustion() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![
            attempt("a", "http_429"),
            attempt("b", "daily_allowance_exhausted"),
        ];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            Some("primary_pool_exhausted")
        );
    }

    #[test]
    fn an_attempt_on_a_non_capable_target_does_not_count() {
        let (models, credentials) = legs(&[("a", "A")]);
        let attempts = vec![attempt("other", "http_429")];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    #[test]
    fn a_quota_only_route_escalates_on_both_legs_being_throttled() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![attempt("a", "http_429"), attempt("b", "http_429")];
        assert_eq!(
            authorize_quota_only(&models, &dispatchable(&models), &attempts, &credentials),
            Some("primary_pool_exhausted")
        );
    }

    #[test]
    fn a_quota_only_route_escalates_on_allowance_exhaustion_and_a_throttle() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![
            attempt("a", "daily_allowance_exhausted"),
            attempt("b", "http_429"),
        ];
        assert_eq!(
            authorize_quota_only(&models, &dispatchable(&models), &attempts, &credentials),
            Some("primary_pool_exhausted")
        );
    }

    #[test]
    fn a_quota_only_route_escalates_on_a_scope_wide_throttle() {
        // One 429 stops the walk only when it is scope-wide, so the sibling it left
        // unattempted is explained by the throttle rather than by a gap in coverage.
        let (models, credentials) = legs(&[("a", "A"), ("b", "A")]);
        let attempts = vec![attempt("a", "http_429")];
        assert_eq!(
            authorize_quota_only(&models, &dispatchable(&models), &attempts, &credentials),
            Some("primary_pool_exhausted")
        );
    }

    /// The discriminator for the whole change: a route that escalates on quota evidence
    /// must not spend its terminal leg on the failures the historical rule would accept.
    #[test]
    fn a_quota_only_route_refuses_the_failures_the_capacity_rule_accepts() {
        for outcome in [
            "http_5xx",
            "transport",
            "unavailable",
            "attempt_timeout",
            "http_401",
            "http_402",
            "daily_allowance_unavailable",
            "minute_window_too_small",
        ] {
            let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
            let attempts = vec![attempt("a", outcome), attempt("b", outcome)];
            assert_eq!(
                authorize(&models, &dispatchable(&models), &attempts, &credentials),
                Some("primary_pool_exhausted"),
                "the capacity rule still escalates on {outcome}"
            );
            assert_eq!(
                authorize_quota_only(&models, &dispatchable(&models), &attempts, &credentials),
                None,
                "outcome {outcome} must not spend a quota-only route's terminal leg"
            );
        }
    }

    /// The judge's escalation taxonomy, stated as a closed list rather than inherited from
    /// `Capacity`. A real outage and a spent allowance are reasons to advance; two faults
    /// that are *about this deployment or this request* are not.
    #[test]
    fn an_availability_route_escalates_on_capacity_and_real_outages() {
        for outcome in [
            "daily_allowance_exhausted",
            "minute_window",
            "http_429",
            "http_5xx",
            "transport",
            "unavailable",
            "attempt_timeout",
        ] {
            let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
            let attempts = vec![attempt("a", outcome), attempt("b", outcome)];
            assert_eq!(
                authorize_availability(&models, &dispatchable(&models), &attempts, &credentials),
                Some("primary_pool_exhausted"),
                "outcome {outcome} is a capacity or availability fact about the pool"
            );
        }
    }

    /// An unreadable allowance ledger is an accounting fault. The allowance may be
    /// untouched, so advancing spends a metered or local leg on a claim nobody measured.
    /// Default fail-closed: the caller gets the primary error.
    #[test]
    fn an_availability_route_refuses_an_accounting_fault() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![
            attempt("a", "daily_allowance_unavailable"),
            attempt("b", "daily_allowance_unavailable"),
        ];
        assert_eq!(
            authorize_availability(&models, &dispatchable(&models), &attempts, &credentials),
            None,
            "an unreadable ledger is not a measured exhaustion"
        );
    }

    /// A request larger than the model's whole minute window can never be served by it,
    /// however long the caller waits. That is a request-fit fact, not a closed window.
    /// Letting it advance would spill a too-large payload onto the next tier, which is
    /// exactly the silent paid/local escalation the policy exists to prevent.
    #[test]
    fn an_availability_route_refuses_a_request_that_cannot_fit() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![
            attempt("a", "minute_window_too_small"),
            attempt("b", "minute_window_too_small"),
        ];
        assert_eq!(
            authorize_availability(&models, &dispatchable(&models), &attempts, &credentials),
            None,
            "a per-request fit failure is not temporary unavailability"
        );
    }

    /// The half of the taxonomy that must never move, asserted against the policy that
    /// actually runs. A request or policy answer is about the request; a rejected key or a
    /// refused balance is about the deployment. Neither is a reason to spend another leg.
    #[test]
    fn an_availability_route_refuses_answers_about_the_request_or_the_credentials() {
        for outcome in [
            "http_400",
            "http_401",
            "http_402",
            "http_403",
            "empty_answer",
            "timeout",
            "context_window",
            "unsupported_capability",
        ] {
            let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
            let attempts = vec![attempt("a", outcome), attempt("b", outcome)];
            assert_eq!(
                authorize_availability(&models, &dispatchable(&models), &attempts, &credentials),
                None,
                "outcome {outcome} must not spend an availability route's terminal leg"
            );
        }
    }

    /// The mutation control for the two exclusions above. Flipping either classification
    /// back to `Capacity`'s answer must change the decision, or the tests above would pass
    /// on a policy that never distinguished them.
    #[test]
    fn the_availability_policy_is_the_discriminator_for_its_two_exclusions() {
        for outcome in ["daily_allowance_unavailable", "minute_window_too_small"] {
            let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
            let attempts = vec![attempt("a", outcome), attempt("b", outcome)];
            assert_eq!(
                authorize(&models, &dispatchable(&models), &attempts, &credentials),
                Some("primary_pool_exhausted"),
                "the historical capacity rule still accepts {outcome}"
            );
            assert_eq!(
                authorize_availability(&models, &dispatchable(&models), &attempts, &credentials),
                None,
                "{outcome} is the one outcome the availability policy must refuse"
            );
        }
    }

    #[test]
    fn a_quota_only_route_refuses_a_mixed_server_error_and_quota() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![attempt("a", "http_5xx"), attempt("b", "http_429")];
        assert_eq!(
            authorize_quota_only(&models, &dispatchable(&models), &attempts, &credentials),
            None,
            "one leg's outage is not evidence that the allowance ran out"
        );
    }

    #[test]
    fn a_quota_only_route_refuses_a_skipped_leg_on_a_live_credential() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![attempt("a", "http_429")];
        assert_eq!(
            authorize_quota_only(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    #[test]
    fn a_quota_only_blocked_pool_needs_a_quota_scoped_block() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "A")]);
        assert_eq!(
            authorize_quota_only_with_blocks(&models, &HashSet::new(), &[], &credentials, |_| true,),
            Some("primary_pool_blocked")
        );
        // One leg covered by a rejected-credential block rather than a spent allowance.
        assert_eq!(
            authorize_quota_only_with_blocks(
                &models,
                &HashSet::new(),
                &[],
                &credentials,
                |model| model.as_str() == "a",
            ),
            None,
            "a rejected credential is not a spent allowance"
        );
    }

    #[test]
    fn the_capacity_rule_does_not_consult_block_kinds() {
        // The default policy's blocked-pool branch is unchanged: it asks whether the pool is
        // blocked, not why.
        let (models, credentials) = legs(&[("a", "A")]);
        assert_eq!(
            authorize_terminal(
                &models,
                &HashSet::new(),
                &[],
                TerminalPolicy::Capacity,
                |model| credentials.get(model).cloned(),
                |_| false,
            ),
            Some("primary_pool_blocked")
        );
    }
}

#[cfg(test)]
mod fit_escalation_tests {
    use super::{authorize_fit_escalation, is_request_fit_exclusion, tier_cannot_hold};

    /// The vocabulary is closed on purpose. Every code listed here is a size fact about one
    /// target, so asking the next tier the same question is meaningful. Anything else is a
    /// statement about the request or the deployment, which a metered tier cannot change.
    #[test]
    fn the_fit_vocabulary_is_exactly_the_size_facts() {
        for code in [
            "context_window",
            "max_input_tokens",
            "minute_window_too_small",
            "output_budget_below_thinking_floor",
        ] {
            assert!(is_request_fit_exclusion(code), "{code} is a size fact");
        }
        for code in [
            "tool_calling",
            "vision",
            "video",
            "reasoning",
            "unsupported_capability",
            "credential_missing",
            "account_balance",
            "no_eligible_candidate",
            "primary_pool_blocked",
            "invalid_request",
            "",
        ] {
            assert!(
                !is_request_fit_exclusion(code),
                "{code} must not escalate on its own"
            );
        }
    }

    #[test]
    fn a_pool_emptied_only_by_size_authorizes_the_tier_walk() {
        assert_eq!(
            authorize_fit_escalation(true, &[Err("max_input_tokens"), Err("max_input_tokens")]),
            Some("primary_pool_cannot_hold_request")
        );
        // Mixed size reasons are still all size reasons.
        assert_eq!(
            authorize_fit_escalation(
                true,
                &[Err("context_window"), Err("minute_window_too_small")]
            ),
            Some("primary_pool_cannot_hold_request")
        );
    }

    #[test]
    fn a_request_the_route_never_promised_is_not_authorized() {
        // The per-target gates test size before capabilities, so a request needing vision can
        // show nothing but size exclusions. The route's own declaration is the only thing that
        // keeps the walk from serving it.
        assert_eq!(
            authorize_fit_escalation(false, &[Err("max_input_tokens")]),
            None
        );
    }

    #[test]
    fn a_reason_other_than_size_is_not_authorized() {
        assert_eq!(
            authorize_fit_escalation(true, &[Err("max_input_tokens"), Err("tool_calling")]),
            None
        );
        assert_eq!(
            authorize_fit_escalation(true, &[Err("credential_missing")]),
            None
        );
    }

    #[test]
    fn a_candidate_that_could_hold_it_means_the_pool_is_not_empty() {
        // Reached only by a caller that asked the wrong question, but answering `Some` here
        // would let a served request be re-sent to a metered tier.
        assert_eq!(
            authorize_fit_escalation(true, &[Err("max_input_tokens"), Ok(())]),
            None
        );
    }

    #[test]
    fn no_candidates_at_all_is_not_a_size_claim() {
        assert_eq!(authorize_fit_escalation(true, &[]), None);
        assert_eq!(authorize_fit_escalation(false, &[]), None);
    }

    #[test]
    fn a_tier_is_passed_over_only_when_every_target_refused_on_size() {
        assert!(tier_cannot_hold(&[
            Err("max_input_tokens"),
            Err("context_window")
        ]));
        assert!(!tier_cannot_hold(&[]));
        assert!(!tier_cannot_hold(&[Err("max_input_tokens"), Ok(())]));
        assert!(!tier_cannot_hold(&[
            Err("max_input_tokens"),
            Err("tool_calling")
        ]));
    }
}
