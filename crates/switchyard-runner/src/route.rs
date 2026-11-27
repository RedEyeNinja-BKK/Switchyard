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
    /// Whether the routed model can accept a request that REPLAYS an earlier tool call.
    ///
    /// A target wire fact, never a route promise: no route declaration feeds it, which is
    /// why [`ModelCapabilities::default`] leaves it `None`. Google's OpenAI-compatibility
    /// surface requires a provider-issued `thought_signature` on every replayed
    /// `functionCall` part and answers HTTP 400 `INVALID_ARGUMENT` without one — measured
    /// 2026-11-26, where a call that surface had just issued was refused once the signature
    /// was stripped, and a call issued elsewhere was refused as well. Switchyard cannot
    /// manufacture the signature and cannot tell whether a caller echoed it, so a target
    /// stating `false` is skipped for any request that carries tool lineage. Its
    /// definitional call stays eligible, because offering tools needs no signature.
    pub tool_call_replay: Option<bool>,
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

/// The exclusion codes that say this *target's* wire cannot carry a request it was given.
///
/// Closed vocabulary, and deliberately narrow. Every code here is a measured property of one
/// provider's wire rather than a judgement about the request: the request is portable, the
/// leg is not, and another leg in a later declared group may be. Codes outside this list are
/// not, and the distinction is the whole safety of capability escalation:
///
/// * a size code is the caller's prompt, and is handled by its own escalation axis;
/// * a credential, allowance or block code is the deployment's capacity;
/// * a malformed, unsafe or unverifiable continuation is a defect in the request, and sending
///   it to a second provider would re-learn the same refusal with a bill attached;
/// * an authentication, billing or policy refusal is a fact about the deployment or the
///   account, and spending another group on it hides the fault the caller needs to see.
///
/// `tool_call_replay` is the only entry because it is the only one measured as a per-wire
/// property with a portable counterpart. Measured 2026-11-26 on the loopback deployment: each
/// Google chat leg accepted a definitional tool call and refused that same call replayed with
/// the signature stripped (HTTP 400 naming `thought_signature`), while a DeepSeek leg served
/// the same replayed transcript. Adding a code here is a new claim about a wire and needs its
/// own measurement and its own test.
fn is_target_capability_exclusion(code: &str) -> bool {
    matches!(code, "tool_call_replay")
}

/// Whether a pool that admitted nothing did so only because every leg's *wire* cannot carry
/// this otherwise valid request, and the route is authorized to look further.
///
/// This is the capability counterpart of [`authorize_fit_escalation`] and answers the same
/// shape of question about a different fact. A size refusal says the pool cannot hold the
/// request; this says the pool cannot carry it, and a later declared group might.
///
/// Three conditions, all required:
///
/// * `route_accepts` -- the lane's own declaration permits the request. A request the route
///   never promised to serve is the caller's problem, not a missing capability in a group.
/// * every candidate is excluded, and
/// * every exclusion is a target wire fact from the closed vocabulary above.
///
/// The answer authorizes *consideration* of the next declared group and nothing else: the
/// target's own credential, allowance, fit, privacy and continuation gates are evaluated
/// exactly as before when that group is reached, so an escalation cannot by itself spend a
/// metered request.
pub(crate) fn authorize_capability_escalation(
    route_accepts: bool,
    gates: &[Result<(), &'static str>],
) -> Option<&'static str> {
    if !route_accepts || gates.is_empty() {
        return None;
    }
    if !gates.iter().all(|gate| match gate {
        Err(code) => is_target_capability_exclusion(code),
        Ok(()) => false,
    }) {
        return None;
    }
    Some("primary_pool_capability_incompatible")
}

/// Whether every target in one declared tier was excluded only by its own wire.
///
/// The tier-level form of [`authorize_capability_escalation`]: a tier that admitted nothing
/// keeps the walk alive only when its own targets are the reason, so the next group is asked
/// about capacity rather than asked to settle something about the request. A tier that has a
/// capable leg blocked by a credential is not this claim, and is judged by
/// [`authorize_terminal`] instead.
fn tier_capability_incompatible(gates: &[Result<(), &'static str>]) -> bool {
    !gates.is_empty()
        && gates.iter().all(|gate| match gate {
            Err(code) => is_target_capability_exclusion(code),
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

/// Whether every declared allowance domain is affirmatively spent.
///
/// The claim is about the deployment's allowance domains, so neither `capable` nor
/// `dispatchable` can bound it: both are what survived gates, and a member excluded by fit,
/// by a rejected key or by an unreadable ledger is precisely a member whose allowance still
/// has to be established. `domains` is the declared membership; a route that declares none
/// is one domain over its capable set, because that is the only pool the deployment named.
///
/// An empty declared domain is a configuration fault and is never read as spent: a claim
/// about a pool with no members is a claim about nothing, and answering it "yes" is how a
/// misconfigured pool would buy the local tier on the first request.
fn declared_domains_exhausted(
    domains: &[Vec<ModelId>],
    ungrouped: &[ModelId],
    exhausted: impl Fn(&ModelId) -> bool,
) -> bool {
    let single: Vec<Vec<ModelId>>;
    let judged: &[Vec<ModelId>] = if domains.is_empty() {
        single = vec![ungrouped.to_vec()];
        &single
    } else {
        domains
    };
    !judged.is_empty()
        && judged
            .iter()
            .all(|domain| !domain.is_empty() && domain.iter().all(&exhausted))
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
/// * `domains` - the declared primary groups: the allowance domains the strict policy
///   makes a claim about, judged over their whole declared membership rather than over the
///   legs this request happened to be able to try. Empty means "one domain, the capable
///   set", which is the strongest statement available when the deployment never declared
///   which targets share a budget;
/// * `credential_of` - the credential that serves a target, when configuration says;
/// * `quota_blocked` - whether a block excludes a target because its allowance is spent,
///   rather than because the credential was rejected;
/// * `allowance_spent` - whether the counter governing a target reports no remainder for
///   the current provider day and can be read at all.
///
/// Returns `None` when the pool's unavailability is not provable, which is the safe
/// answer: the caller gets the primary error instead of a paid escalation.
#[allow(clippy::too_many_arguments)] // Each argument is one piece of evidence, varied independently in tests.
pub(crate) fn authorize_terminal(
    capable: &[ModelId],
    dispatchable: &HashSet<ModelId>,
    attempts: &[AttemptTrace],
    policy: TerminalPolicy,
    domains: &[Vec<ModelId>],
    credential_of: impl Fn(&ModelId) -> Option<String>,
    quota_blocked: impl Fn(&ModelId) -> bool,
    allowance_spent: impl Fn(&ModelId) -> bool,
) -> Option<&'static str> {
    // Nothing in the pool could serve this request even with unlimited capacity, so
    // "exhausted" would be a different claim from the true one. Refuse.
    if capable.is_empty() {
        return None;
    }
    // The strict boundary is decided here, before the block and attempt branches, because
    // it is the one judgement that must not be narrowed by them: a member dropped for fit,
    // for a rejected key or for an unreadable ledger is still a member of its domain, and
    // its allowance still has to be established before the local tier may be spent.
    //
    // Only affirmative current-window evidence counts:
    //   * this request saw the target answer that its daily allowance is spent;
    //   * a quota-scoped credential block covers it -- a block `credential_block` builds
    //     only from a provider quota statement, never from a bare 429, a cooldown or a
    //     rejected key, and scoped so a per-model block covers only that model;
    //   * the counter governing it reports no remainder for the current provider day and
    //     can be read at all.
    // A rejected credential, an unreadable or stale ledger, a leg skipped for fit and a leg
    // nobody looked at leave the domain unknown, and unknown is not spent. A per-model
    // exhaustion is never promoted to its credential: Google allocates that budget per model
    // inside the project, so a spent sibling says nothing about this one.
    let strict = (policy == TerminalPolicy::AllowanceOnly).then(|| {
        let spent_here: HashSet<&str> = attempts
            .iter()
            .filter(|attempt| capable.iter().any(|model| model.as_str() == attempt.model))
            .filter(|attempt| is_capacity_outcome(attempt.outcome, policy))
            // A provider-confirmed refusal is raised by the pre-send reservation, never by
            // this request's own answer: the reservation runs before anything leaves, so the
            // outcome inherits the basis of the reading behind it and the string cannot carry
            // that distinction. Ask the counter again here -- a current zero is evidence the
            // domain is spent, a retained statement about the past is not, and the two arrive
            // as the same outcome. The local refusal is not re-asked: this deployment's own
            // ledger is current by construction.
            .filter(|attempt| {
                attempt.outcome != "provider_allowance_exhausted"
                    || allowance_spent(&ModelId::from(attempt.model.as_str()))
            })
            .map(|attempt| attempt.model.as_str())
            .collect();
        declared_domains_exhausted(domains, capable, |model| {
            spent_here.contains(model.as_str()) || quota_blocked(model) || allowance_spent(model)
        })
    });
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
        //
        // The strict policy keeps its own reading of "blocked": the block kind alone is not
        // proof, so the declared domains are judged as they are on every other path, and a
        // domain whose members are merely rejected stays unknown rather than spent.
        match strict {
            Some(true) => return Some("primary_pool_blocked"),
            Some(false) => return None,
            None => {
                if policy != TerminalPolicy::Capacity && !capable.iter().all(quota_blocked) {
                    return None;
                }
                return Some("primary_pool_blocked");
            }
        }
    }
    let primary_attempts: Vec<&AttemptTrace> = attempts
        .iter()
        .filter(|attempt| capable.iter().any(|model| model.as_str() == attempt.model))
        .collect();
    if primary_attempts.is_empty() && strict.is_none() {
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
    if let Some(spent) = strict {
        // The domain judgement above already accounts for every declared member, including
        // the ones this request never dispatched and the ones it could not fit, so there is
        // no coverage gap left to explain by a dead credential.
        return spent.then_some("primary_pool_exhausted");
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
                | "provider_allowance_exhausted"
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
            "daily_allowance_exhausted"
                | "provider_allowance_exhausted"
                | "minute_window"
                | "http_429"
        ),
        // The strict reading: only an affirmative statement that an allowance domain is at
        // zero for the current window. `QuotaOnly` above also accepts a minute throttle and a
        // bare 429, which is right for a lane whose terminal leg is a local model that costs
        // nothing to try; it is wrong for a lane that promises to cross only once BOTH primary
        // allowances are genuinely spent. A minute window regains room on its own, and an
        // unclassified 429 may be a concurrency limit or a policy answer - neither is a
        // measured exhaustion.
        //
        // `provider_allowance_exhausted` is accepted here only because a *current* provider
        // zero is affirmative evidence. The string is also produced by a refusal that is only
        // being retained past its reading's freshness, and it cannot say which. Acceptance
        // therefore only says the class is eligible: `authorize_terminal`'s strict branch asks
        // the counter again before an attempt of this class may authorise the local tier, and
        // a retained statement about the past fails that second question.
        TerminalPolicy::AllowanceOnly => matches!(
            outcome,
            "daily_allowance_exhausted" | "provider_allowance_exhausted"
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
                | "provider_allowance_exhausted"
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
    /// The declared groups as model sets, when this route orders by group before by leaf.
    ///
    /// Empty for every other route. The membership is resolved at load rather than at request
    /// time so the dispatch path never re-derives it from labels, and so a group the manifest
    /// does not describe cannot be invented mid-request.
    group_plan: crate::selection::GroupPlan,
    /// How the declared groups order the pool: equal peers that adapt and rotate, or a
    /// declared cost sequence in which each group is exhausted before the next.
    ///
    /// Carried beside the plan rather than inside it because the same membership can be
    /// deployed under either reading, and the reading is what a deployment decides.
    group_ordering: crate::selection::Grouping,
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
    /// Per-boundary evidence policy, aligned with `terminal`; empty means every boundary uses
    /// `terminal_policy`.
    ///
    /// The entry boundary is the claim that opens the walk and stays the route's own. Each
    /// later boundary is a claim about the tier being left, and a route may need a different
    /// one there: a local tier is entered on quota evidence, but the metered tier behind it
    /// exists for the local outage, which is not quota evidence.
    terminal_policies: Vec<TerminalPolicy>,
    /// Whether a request that no primary candidate can hold may be escalated to the declared
    /// tiers.
    ///
    /// A separate axis from `terminal_policy` on purpose. That one governs a pool that
    /// *failed*; this one governs a pool that never dispatched, because the request is larger
    /// than any of its targets can hold. Escalating on a size refusal is not a redefinition of
    /// capacity exhaustion -- it is a different claim, recorded under its own reason, and the
    /// tier that receives the request is judged on its own gates.
    fit_escalation: bool,
    /// Whether a request that no primary candidate's *wire* can carry may be escalated to the
    /// declared tiers.
    ///
    /// The third axis, and the narrowest. `terminal_policy` governs a pool that failed;
    /// `fit_escalation` governs a request too large for the pool; this governs a request the
    /// pool's targets cannot carry at the wire level while a later group's can. It is off
    /// unless a route declares it, so a lane that has never been measured for this keeps
    /// refusing rather than quietly spending a downstream group.
    ///
    /// It authorizes consideration, never expenditure: the group that receives the request is
    /// judged on its own credential, allowance, fit, privacy and continuation gates, so an
    /// escalation cannot by itself buy a metered call.
    capability_escalation: bool,
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
            group_plan: crate::selection::GroupPlan::default(),
            group_ordering: crate::selection::Grouping::Adaptive,
            terminal: Vec::new(),
            terminal_policy: TerminalPolicy::default(),
            terminal_policies: Vec::new(),
            fit_escalation: false,
            capability_escalation: false,
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

    /// Declares the evidence policy that governs each declared tier's own boundary.
    ///
    /// Aligned with the tiers and validated at load. An absent or short list is not a silent
    /// downgrade: configuration refuses a count that does not match the tiers, so what the
    /// route stores is either the full declaration or nothing.
    pub fn with_terminal_policies(mut self, policies: Vec<TerminalPolicy>) -> Self {
        self.terminal_policies = policies;
        self
    }

    /// The policy that governs the boundary leaving `tier`.
    fn terminal_policy_leaving(&self, tier: usize) -> TerminalPolicy {
        self.terminal_policies
            .get(tier)
            .copied()
            .unwrap_or(self.terminal_policy)
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

    /// Declares whether a request no primary candidate's wire can carry may reach the tiers.
    ///
    /// Only meaningful beside `with_terminal_fallback`; configuration refuses the option
    /// without a leg to escalate to. Off is the historical behaviour: a pool whose targets
    /// cannot carry the request refuses, and no further group is contacted.
    ///
    /// On does not widen any tier and does not authorize spend. It changes one thing: a pool
    /// that dispatched nothing because every target refused the request *at its own wire* is
    /// allowed to ask the next declared group the same question, instead of being refused
    /// before the groups are looked at. A group that cannot carry it either ends the walk with
    /// the primary pool's refusal.
    pub fn with_capability_escalation(mut self, enabled: bool) -> Self {
        self.capability_escalation = enabled;
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
        // The primary pool's allowance domains are its declared groups, and the strict policy
        // judges them over their whole declared membership -- a member excluded by fit or by
        // a block is exactly a member whose allowance still has to be accounted for.
        //
        // A route that declares no groups is one domain over its capable set, which is the
        // strongest statement its configuration supports. That residual is stated rather than
        // hidden: an ungrouped strict route cannot account for a sibling the request did not
        // fit, because nothing declares that the sibling shares a budget with the legs that
        // were tried. Every strict route in this deployment is grouped.
        let domains: Vec<Vec<ModelId>> = self
            .group_plan
            .groups
            .iter()
            .map(|group| group.members.clone())
            .collect();
        self.terminal_authorization_under(
            capable,
            dispatchable,
            attempts,
            blocks,
            self.terminal_policy,
            &domains,
        )
    }

    /// [`Self::terminal_authorization`] under an explicit boundary policy.
    ///
    /// The policy is a parameter rather than a field read so the boundary that is being
    /// crossed decides it: the entry claim is the route's own, and every later claim is about
    /// the tier being left. A terminal tier is one domain, so callers judging a tier pass
    /// nothing and the tier's capable set is judged whole.
    fn terminal_authorization_under(
        &self,
        capable: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        blocks: &[switchyard_llm_client::CredentialBlock],
        policy: TerminalPolicy,
        domains: &[Vec<ModelId>],
    ) -> Option<&'static str> {
        authorize_terminal(
            capable,
            dispatchable,
            attempts,
            policy,
            domains,
            |model| self.target_credential(model).map(str::to_string),
            |model| self.quota_blocked(model, blocks),
            |model| self.allowance_spent(model),
        )
    }

    /// Whether the counter that governs `model` is knowably out for the current provider day.
    ///
    /// This is the evidence half of the strict boundary, and it exists so a declared domain
    /// member that this request could not dispatch -- excluded by fit, or blocked -- can
    /// still be judged. [`DailyRequestAllowance::spent_today`] answers in three states and
    /// only the affirmative one counts: a readable counter at zero is spent, a readable
    /// counter with room is not, and a counter that cannot be read is unknown. The third must
    /// not collapse into either neighbour -- an unreadable ledger also reports a remainder of
    /// zero, so reading the figure alone would turn "this deployment cannot say" into "the
    /// budget is spent" and buy the local tier on a broken file.
    ///
    /// A *retained* exhaustion is the second way into that third state. Admission keeps
    /// refusing while a past "the pool is empty" is honoured past its freshness, so the
    /// remainder still reads zero, but the statement describes the past: it must not buy the
    /// local tier any more than an unreadable file may.
    ///
    /// A target with no counter is unknown for a different reason: a metered target has no
    /// daily allowance to exhaust, so its exhaustion can only ever come from the provider's
    /// own answer within this request.
    fn allowance_spent(&self, model: &ModelId) -> bool {
        self.daily_allowances
            .get(model)
            .is_some_and(|allowance| allowance.spent_today() == Some(true))
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

    /// Declares the groups' membership, making the group a level of the order.
    ///
    /// Set only for a route whose selection policy compares groups; an empty plan leaves the
    /// pool flat, and the dispatch path decides between the two by whether a plan exists, so
    /// a route can never claim grouped selection while ranking leaf by leaf.
    pub fn with_group_plan(mut self, plan: crate::selection::GroupPlan) -> Self {
        self.group_plan = plan;
        self
    }

    /// Whether this route chooses a group before it chooses a leaf.
    /// Declares how the groups are ordered against each other.
    ///
    /// `Adaptive` is the equal-peer reading: readiness, completion rate, latency and a
    /// rotation floor decide between interchangeable groups. `Ordered` is the cost reading:
    /// the declared sequence is the priority, and a later group is offered only once the
    /// earlier one has no eligible member left. `Flat` is refused for a grouped pool by the
    /// loader, so a route cannot hold a plan and rank leaf by leaf.
    pub fn with_group_ordering(mut self, ordering: crate::selection::Grouping) -> Self {
        self.group_ordering = ordering;
        self
    }

    pub fn is_grouped_pool(&self) -> bool {
        self.group_plan.groups.len() >= 2
    }

    /// Whether this pool's groups are a declared cost sequence rather than equal peers.
    ///
    /// The two grouped shapes differ in one membership rule, not in how they rank: an
    /// equal-peer pool is a free pool, while an ordered pool's later groups are meant to be
    /// metered and would be deleted by a free-only gate.
    pub fn is_ordered_pool(&self) -> bool {
        self.is_grouped_pool() && self.group_ordering == crate::selection::Grouping::Ordered
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

    /// The capability a pool with no eligible leg owed the request, named from the targets that
    /// were excluded rather than from the route's own declaration.
    ///
    /// The route's declaration may be silent while a specific target was the blocker — the live
    /// case is a lane whose own `tool_call_replay` is undeclared (permitted) over a pool of
    /// Google legs that each refuse a replayed call — so naming the route's capabilities alone
    /// would report "an eligible route" for a request that was refused for a concrete reason.
    /// The target facts are read exactly as the eligibility predicate reads them, or the refusal
    /// would name a different capability than the one that excluded the candidate.
    ///
    /// Order is deliberate: a capability a target lacked, then the route's own declaration, then
    /// the binding size limit. The size limit is last because it is the only answer that is not
    /// about a capability, and it is reported only when no candidate fits at all.
    fn missing_capability(&self, request: &Request, raw: &serde_json::Value) -> &'static str {
        self.clients
            .target_capabilities()
            .iter()
            .find_map(|(_model, facts)| {
                let capabilities = crate::ModelCapabilities {
                    vision: Some(!facts.no_vision),
                    tool_calling: Some(!facts.no_tool_calling),
                    tool_call_replay: Some(!facts.no_tool_call_replay),
                    video: Some(!facts.no_video),
                    video_carrier: carrier_from_facts(facts),
                    image_carrier: image_carrier_from_facts(facts),
                    ..crate::ModelCapabilities::default()
                };
                crate::request_fit::unsupported_capability(capabilities, &request.llm_request, raw)
            })
            .or_else(|| {
                crate::request_fit::unsupported_capability(
                    self.capabilities,
                    &request.llm_request,
                    raw,
                )
            })
            .or_else(|| self.physical_ineligibility(request))
            .unwrap_or("an eligible route")
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
        // A ranked pool is a *free* pool. The rule guards the ranking, not the last resort:
        // a declared terminal tier is never ranked, never reached by a busy pool, and may
        // legitimately be metered - a paid NT leg behind an exhausted free allowance is the
        // shape the owner asked for, and refusing it here would leave that tier unreachable.
        // Terminal targets are already forbidden from being candidates at load, so exempting
        // them cannot smuggle a paid leg into the ranking.
        let terminal_models: std::collections::HashSet<switchyard_protocol::ModelId> =
            self.terminal_models().into_iter().collect();
        // A ranked pool is a free pool - unless its groups are a declared cost sequence. An
        // ORDERED pool names its spending boundary in the group order itself: the metered
        // group sits after the free ones and is offered only once they have no eligible member
        // left, which is the protected-spend shape rather than an accident of ranking. Applying
        // the free-only rule there would delete the very groups it describes.
        let pool_members_only_free = self.is_peer_pool() && !self.is_ordered_pool();
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
            if pool_members_only_free && !terminal_models.contains(model) && !facts.tariff.is_free()
            {
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
                // A wire that cannot accept a replayed tool call is skipped for a
                // request that carries one, and stays eligible for the definitional
                // call that merely offers tools. Carried as a concrete boolean like
                // the other facts, so an undeclared target keeps permitting both.
                tool_call_replay: Some(!facts.no_tool_call_replay),
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
        // The same computation over the other closed vocabulary. The two cannot both be
        // `Some` (a gate is excluded for one reason), so a chain is unambiguous, but they are
        // kept as separate values because the reason they are recorded under is different and
        // an operator reads that reason to decide what to fix.
        let capability_escalation = self
            .capability_escalation
            .then(|| authorize_capability_escalation(route_accepts, &verdicts))
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
                scores = if self.is_grouped_pool() {
                    // One request, one rotation step: the group decision is recorded here,
                    // where the dispatch order is actually decided.
                    selector.explain_grouped(
                        &candidates,
                        demand,
                        &declared,
                        &self.group_plan,
                        self.group_ordering,
                    )
                } else {
                    selector.explain_with(&candidates, demand, &declared, self.ranking_policy())
                };
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
        if models.is_empty() && fit_escalation.is_none() && capability_escalation.is_none() {
            // Nothing was dispatched, so the record is the whole story: exclusion codes and
            // an empty attempt list. Emitted before either refusal so the caller and the
            // routing log name the same reason.
            emit_selection(observer.as_ref(), &trace);
            // Nothing in this route can serve the request, which is the caller's request
            // rather than a deployment fault. The skipped targets get no provider call.
            return Err(RunnerError::unsupported_capability(
                self.algorithm.name(),
                self.missing_capability(&request, &raw),
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
            // The walk may still reach a declared group, but this is what the caller gets if
            // no group can carry the request. It names the same capability the direct refusal
            // would have named, so an escalation that finds nothing does not change the answer
            // into a vaguer one.
            Err(RunnerError::unsupported_capability(
                self.algorithm.name(),
                self.missing_capability(&request, &raw),
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
            let mut authorization = fit_escalation.or(capability_escalation).or_else(|| {
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
                    //
                    // The capability boundary does the same for its own vocabulary: a tier
                    // that admitted nothing because every one of its wires refused this
                    // request keeps the walk alive, and a tier that had a capable leg blocked
                    // by a credential falls through to the exhaustion rule below, which is the
                    // only one entitled to call that exhaustion.
                    authorization = fit_escalation
                        .filter(|_| tier_cannot_hold(&terminal_verdicts[index]))
                        .or(capability_escalation
                            .filter(|_| tier_capability_incompatible(&terminal_verdicts[index])))
                        .or_else(|| {
                            self.terminal_authorization_under(
                                &terminal_capable[index],
                                &terminal_servable[index],
                                &attempts.lock().clone(),
                                blocks,
                                self.terminal_policy_leaving(index),
                                &[],
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
                authorization = self.terminal_authorization_under(
                    &terminal_capable[index],
                    &terminal_servable[index],
                    &attempts.lock().clone(),
                    blocks,
                    self.terminal_policy_leaving(index),
                    &[],
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

/// A provider-confirmed allowance refusal must mean exactly what the local one means to
/// every routing decision, and nothing less.
///
/// The two labels are separate so the record can say which authority refused, but the
/// consequence is identical: the leg has no allowance today, so the route may advance and
/// a terminal leg may be spent on it. Keeping them aligned by hand across three policy
/// lists is exactly the change that goes wrong quietly -- a new error class with no policy
/// arm ends the route with a refusal even when a sibling could answer. So the invariant is
/// asserted mechanically, for every policy, rather than trusted to review.
#[cfg(test)]
mod allowance_label_parity_tests {
    use super::{TerminalPolicy, is_capacity_outcome};

    const LOCAL: &str = "daily_allowance_exhausted";
    const PROVIDER: &str = "provider_allowance_exhausted";

    const POLICIES: [TerminalPolicy; 3] = [
        TerminalPolicy::Capacity,
        TerminalPolicy::QuotaOnly,
        TerminalPolicy::Availability,
    ];

    #[test]
    fn the_provider_label_is_accepted_wherever_the_local_one_is() {
        for policy in POLICIES {
            assert_eq!(
                is_capacity_outcome(LOCAL, policy),
                is_capacity_outcome(PROVIDER, policy),
                "policy {policy:?} treats the provider-confirmed refusal differently from the \
                 local one; a route would then refuse instead of advancing, or spend a \
                 terminal leg on a class it was told to ignore"
            );
        }
    }

    /// The parity above would also hold if BOTH labels were wrongly rejected, so the
    /// allowance class is pinned to being accepted where it must be.
    #[test]
    fn the_allowance_refusal_is_accepted_by_every_policy() {
        for policy in POLICIES {
            assert!(
                is_capacity_outcome(LOCAL, policy),
                "policy {policy:?} must accept an allowance refusal"
            );
            assert!(
                is_capacity_outcome(PROVIDER, policy),
                "policy {policy:?} must accept a provider-confirmed allowance refusal"
            );
        }
    }

    /// And the label must stay distinguishable: folding it into the local one would make
    /// the decision record unable to say which authority refused.
    #[test]
    fn the_two_labels_are_different_strings() {
        assert_ne!(LOCAL, PROVIDER);
        assert!(LOCAL.contains("daily_allowance"));
        assert!(PROVIDER.contains("provider_allowance"));
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
            &[],
            |model| credentials.get(model).cloned(),
            |_| false,
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
            &[],
            |model| credentials.get(model).cloned(),
            quota_blocked,
            |_| false,
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
            &[],
            |model| credentials.get(model).cloned(),
            |_| false,
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
                &[],
                |model| credentials.get(model).cloned(),
                |_| false,
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

#[cfg(test)]
mod capability_escalation_tests {
    use super::{
        authorize_capability_escalation, is_target_capability_exclusion,
        tier_capability_incompatible,
    };

    /// Closed on purpose, and deliberately one entry. Each additional code here is a claim that
    /// some *wire* cannot carry a request another wire can, which is a measurement, not a
    /// judgement. Everything a second group would re-learn at a price -- a malformed or unsafe
    /// continuation, a credential, an allowance, a policy or billing refusal, a size fact this
    /// axis does not own, or a request the route never promised -- must stay out.
    #[test]
    fn the_vocabulary_is_exactly_the_measured_wire_facts() {
        assert!(is_target_capability_exclusion("tool_call_replay"));
        for code in [
            "tool_calling",
            "vision",
            "video",
            "video_carrier",
            "image_carrier",
            "reasoning",
            "unsupported_reasoning_effort",
            "context_window",
            "max_input_tokens",
            "minute_window_too_small",
            "output_budget_below_thinking_floor",
            "credential_missing",
            "credential_rejected",
            "daily_allowance_exhausted",
            "provider_allowance_exhausted",
            "account_balance",
            "malformed_tool_call",
            "unsafe_tool_continuation",
            "invalid_request",
            "policy_refused",
            "no_eligible_candidate",
            "primary_pool_blocked",
            "",
        ] {
            assert!(
                !is_target_capability_exclusion(code),
                "{code} must not authorize a capability escalation"
            );
        }
    }

    /// The exact live case the option was added for: five Google chat legs, every one of them
    /// refusing a replayed tool call at its own wire.
    #[test]
    fn a_pool_emptied_only_by_wire_facts_authorizes_the_group_walk() {
        let five_google = [
            Err("tool_call_replay"),
            Err("tool_call_replay"),
            Err("tool_call_replay"),
            Err("tool_call_replay"),
            Err("tool_call_replay"),
        ];
        assert_eq!(
            authorize_capability_escalation(true, &five_google),
            Some("primary_pool_capability_incompatible")
        );
    }

    #[test]
    fn a_request_the_route_never_promised_is_not_authorized() {
        // A lane that declares it does not serve this request has no missing capability to
        // make up for; escalating would hand a request the deployment refused to a metered
        // group.
        assert_eq!(
            authorize_capability_escalation(false, &[Err("tool_call_replay")]),
            None
        );
    }

    #[test]
    fn one_candidate_that_could_carry_it_ends_the_claim() {
        assert_eq!(
            authorize_capability_escalation(true, &[Err("tool_call_replay"), Ok(())]),
            None
        );
    }

    /// Every other class of refusal must not open the boundary, and the list is the owner's
    /// exclusion list rather than an inviting vocabulary.
    #[test]
    fn request_validity_and_deployment_faults_do_not_open_the_walk() {
        for code in [
            "malformed_tool_call",
            "invalid_request",
            "credential_missing",
            "daily_allowance_exhausted",
            "account_balance",
            "vision",
            "context_window",
        ] {
            assert_eq!(
                authorize_capability_escalation(true, &[Err(code)]),
                None,
                "{code} must end the walk"
            );
            // Mixed with a real wire fact, the other reason still ends it: one leg failing
            // for a reason about the request means the next group would settle the same
            // question again.
            assert_eq!(
                authorize_capability_escalation(true, &[Err("tool_call_replay"), Err(code)]),
                None,
                "{code} mixed with the wire fact must end the walk"
            );
        }
    }

    #[test]
    fn no_candidates_at_all_is_not_a_capability_claim() {
        assert_eq!(authorize_capability_escalation(true, &[]), None);
        assert_eq!(authorize_capability_escalation(false, &[]), None);
    }

    /// A tier is passed over on this axis only when every target in it refused at its wire.
    /// A tier with a capable-but-blocked leg is a capacity question and belongs to
    /// `authorize_terminal`, which is a different claim in the record.
    #[test]
    fn a_tier_is_passed_over_only_when_every_target_refused_at_its_wire() {
        assert!(tier_capability_incompatible(&[
            Err("tool_call_replay"),
            Err("tool_call_replay")
        ]));
        assert!(!tier_capability_incompatible(&[]));
        assert!(!tier_capability_incompatible(&[
            Err("tool_call_replay"),
            Ok(())
        ]));
        assert!(!tier_capability_incompatible(&[
            Err("tool_call_replay"),
            Err("daily_allowance_exhausted")
        ]));
    }
}

/// The strict allowance boundary the NT and judge lanes deploy.
///
/// `QuotaOnly` is the historical "the free allowance ran out or we are being throttled"
/// rule, which is right for a lane whose terminal leg is a local model that costs nothing
/// to try. The NT and judge lanes promise something narrower: the local tier opens only once
/// *both* primary allowances are genuinely spent. These tests pin the difference, because the
/// two rules look identical in a decision record and differ by real money and real service.
#[cfg(test)]
mod strict_allowance_boundary_tests {
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
            TerminalPolicy::AllowanceOnly,
            &[],
            |model| credentials.get(model).cloned(),
            |_| false,
            |_| false,
        )
    }

    /// Two primary groups, each on its own credential: the deployed NT and judge shape.
    fn two_domains() -> (Vec<ModelId>, HashMap<ModelId, String>) {
        legs(&[("google", "GOOGLE"), ("openai", "OPENAI")])
    }

    /// The boundary the owner asked for: both allowance domains at zero is the one condition
    /// that may spend the local tier.
    ///
    /// The provider-confirmed refusal is backed by the counter here, because that is where the
    /// refusal came from: the pre-send reservation reads the counter, so "the provider's pool
    /// is empty" is only ever as good as the reading behind it.
    #[test]
    fn both_allowance_domains_exhausted_opens_the_boundary() {
        let (models, credentials) = two_domains();
        let attempts = vec![
            attempt("google", "daily_allowance_exhausted"),
            attempt("openai", "provider_allowance_exhausted"),
        ];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &[],
                &["openai"],
            ),
            Some("primary_pool_exhausted")
        );
    }

    /// A retained refusal reaches this judgement as the same outcome string as a current one,
    /// because the reservation error cannot carry the basis. The counter can, so the attempt
    /// is evidence only while the counter affirms it: a statement about the past may refuse
    /// the send and must not buy the local tier.
    #[test]
    fn a_retained_provider_refusal_is_not_spent_evidence() {
        let (models, credentials) = two_domains();
        let attempts = vec![
            attempt("google", "daily_allowance_exhausted"),
            attempt("openai", "provider_allowance_exhausted"),
        ];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &[],
                &[],
            ),
            None,
            "the counter cannot affirm this domain is spent, so the attempt cannot either"
        );
        // Control: the same attempt opens the boundary once the counter is current.
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &[],
                &["openai"],
            ),
            Some("primary_pool_exhausted")
        );
    }

    /// This deployment's own ledger is always current, so a local refusal is evidence on its
    /// own. Asking the counter about it would be harmless today and wrong the moment the two
    /// questions drift apart again.
    #[test]
    fn a_local_refusal_needs_no_counter_to_be_evidence() {
        let (models, credentials) = two_domains();
        let attempts = vec![
            attempt("google", "daily_allowance_exhausted"),
            attempt("openai", "daily_allowance_exhausted"),
        ];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &[],
                &[],
            ),
            Some("primary_pool_exhausted")
        );
    }

    /// A minute window regains room inside the same minute. Crossing on it buys a local
    /// request to solve a problem that would have solved itself.
    #[test]
    fn a_minute_window_alone_never_opens_the_strict_boundary() {
        let (models, credentials) = two_domains();
        let attempts = vec![
            attempt("google", "daily_allowance_exhausted"),
            attempt("openai", "minute_window"),
        ];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None,
            "one domain throttled is not two domains exhausted"
        );
    }

    /// A bare 429 is unclassified: it may be a concurrency limit or a policy answer, so it
    /// cannot stand as evidence that an allowance is spent.
    #[test]
    fn a_bare_429_never_opens_the_strict_boundary() {
        let (models, credentials) = two_domains();
        let attempts = vec![
            attempt("google", "daily_allowance_exhausted"),
            attempt("openai", "http_429"),
        ];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    /// An unreadable ledger is *unknown*, and unknown is not exhausted. Earlier code folded
    /// this into the same class, which is how a stale reading could have bought a local call.
    #[test]
    fn an_unreadable_ledger_never_opens_the_strict_boundary() {
        let (models, credentials) = two_domains();
        let attempts = vec![
            attempt("google", "daily_allowance_exhausted"),
            attempt("openai", "daily_allowance_unavailable"),
        ];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    /// Everything that is a fact about the deployment or the request stops at the pool. The
    /// list is closed on purpose: `Availability` accepts the outage classes, this rule does
    /// not, because neither says an allowance ran out.
    #[test]
    fn deployment_and_request_faults_never_open_the_strict_boundary() {
        for outcome in [
            "http_5xx",
            "transport",
            "unavailable",
            "attempt_timeout",
            "http_401",
            "http_402",
            "http_400",
            "invalid_request",
            "unsupported_feature",
            "empty_answer",
            "context_window",
            "minute_window_too_small",
        ] {
            let (models, credentials) = two_domains();
            let attempts = vec![
                attempt("google", "daily_allowance_exhausted"),
                attempt("openai", outcome),
            ];
            assert_eq!(
                authorize(&models, &dispatchable(&models), &attempts, &credentials),
                None,
                "{outcome} is not allowance evidence"
            );
        }
    }

    /// One exhausted domain does not explain a sibling on another credential that was never
    /// tried. The other group's allowance is unknown, so the boundary is not provable.
    #[test]
    fn an_unattempted_second_domain_is_not_exhaustion() {
        let (models, credentials) = two_domains();
        let attempts = vec![attempt("google", "daily_allowance_exhausted")];
        assert_eq!(
            authorize(&models, &dispatchable(&models), &attempts, &credentials),
            None
        );
    }

    fn authorize_with_domains(
        models: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        credentials: &HashMap<ModelId, String>,
        domains: &[Vec<ModelId>],
    ) -> Option<&'static str> {
        authorize_terminal(
            models,
            dispatchable,
            attempts,
            TerminalPolicy::AllowanceOnly,
            domains,
            |model| credentials.get(model).cloned(),
            |_| false,
            |_| false,
        )
    }

    /// The refusals, and the positive controls that prove they are refusals and not dead
    /// code. `spent` stands for the authoritative counter state: only a readable counter with
    /// no remainder for the current provider day may appear in it.
    fn authorize_with_state(
        models: &[ModelId],
        dispatchable: &HashSet<ModelId>,
        attempts: &[AttemptTrace],
        credentials: &HashMap<ModelId, String>,
        domains: &[Vec<ModelId>],
        spent: &[&str],
    ) -> Option<&'static str> {
        let spent: HashSet<&str> = spent.iter().copied().collect();
        authorize_terminal(
            models,
            dispatchable,
            attempts,
            TerminalPolicy::AllowanceOnly,
            domains,
            |model| credentials.get(model).cloned(),
            |_| false,
            |model| spent.contains(model.as_str()),
        )
    }

    /// The reviewed counterexample, reversed: a domain whose members did not fit this request
    /// is still part of the claim. `b` is absent from `capable` because the request is larger
    /// than its window, and nothing has established that its allowance is spent -- so the
    /// boundary is not provable and the local tier must not be spent.
    #[test]
    fn a_domain_excluded_by_fit_is_still_part_of_the_claim() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let only_a_capable = vec![ModelId::from("a")];
        let attempts = vec![attempt("a", "daily_allowance_exhausted")];
        let domains = vec![vec![ModelId::from("a")], vec![ModelId::from("b")]];
        assert_eq!(
            authorize_with_state(
                &only_a_capable,
                &dispatchable(&only_a_capable),
                &attempts,
                &credentials,
                &domains,
                &[],
            ),
            None,
            "a domain that only skipped the request has not spent its allowance"
        );
        // Control: the same request shape authorizes once the excluded domain's own counter
        // says its day is spent. Nothing is dispatched to the ineligible leg to learn this.
        assert_eq!(
            authorize_with_state(
                &only_a_capable,
                &dispatchable(&only_a_capable),
                &attempts,
                &credentials,
                &domains,
                &["b"],
            ),
            Some("primary_pool_exhausted")
        );
        assert!(
            models.len() == 2,
            "both domains are declared even when one cannot fit"
        );
    }

    /// The all-blocked branch had the same capable-subset defect: it asked whether every
    /// dispatchable leg was blocked, which cannot see a domain that contributed no
    /// dispatchable leg at all. The declared domains decide it, on the same evidence.
    #[test]
    fn the_all_blocked_branch_judges_the_declared_domains() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let domains = vec![vec![ModelId::from("a")], vec![ModelId::from("b")]];
        let nothing_dispatchable = HashSet::new();
        assert_eq!(
            authorize_with_state(
                &models,
                &nothing_dispatchable,
                &[],
                &credentials,
                &domains,
                &["a"],
            ),
            None,
            "one spent domain and one unknown domain is not a blocked pool"
        );
        assert_eq!(
            authorize_with_state(
                &models,
                &nothing_dispatchable,
                &[],
                &credentials,
                &domains,
                &["a", "b"],
            ),
            Some("primary_pool_blocked")
        );
    }

    /// A mixed pool: every domain affirmatively spent by the provider, none of them
    /// dispatchable, and one leg rejected rather than exhausted. The rejected leg keeps its
    /// domain unknown, so the boundary does not open on a deployment fault.
    #[test]
    fn a_mixed_pool_does_not_open_on_a_rejected_leg() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let domains = vec![vec![ModelId::from("a")], vec![ModelId::from("b")]];
        let attempts = vec![
            attempt("a", "daily_allowance_exhausted"),
            attempt("b", "http_401"),
        ];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &domains,
                &[],
            ),
            None
        );
    }

    /// A group that declares no members exhausts nothing: reading it as spent would let a
    /// misconfigured pool buy the local tier on its first request.
    #[test]
    fn a_declared_group_with_no_members_is_never_spent() {
        let (models, credentials) = legs(&[("a", "A")]);
        let attempts = vec![attempt("a", "daily_allowance_exhausted")];
        let domains = vec![vec![ModelId::from("a")], Vec::new()];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &domains,
                &[],
            ),
            None
        );
    }

    /// The defect this rule was rewritten for: a leg dropped from `dispatchable` by a
    /// rejected credential is still a member of its allowance domain, and nobody looked at
    /// its allowance. The old shape judged only the dispatchable remainder, so one exhausted
    /// domain plus one un-checkable sibling read as "both domains spent" and bought the local
    /// tier.
    #[test]
    fn a_sibling_dropped_for_a_rejected_credential_keeps_its_domain_unknown() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let only_a_dispatchable: HashSet<ModelId> = [ModelId::from("a")].into_iter().collect();
        let attempts = vec![attempt("a", "daily_allowance_exhausted")];
        let domains = vec![vec![ModelId::from("a")], vec![ModelId::from("b")]];
        assert_eq!(
            authorize_with_domains(
                &models,
                &only_a_dispatchable,
                &attempts,
                &credentials,
                &domains
            ),
            None,
            "b's allowance was never established, so the boundary is not provable"
        );
        // The undeclared form is one domain over every capable leg, and it must not be a
        // weaker claim: a route that names no groups is still not allowed to read one
        // exhausted leg as the whole pool being spent.
        assert_eq!(
            authorize(&models, &only_a_dispatchable, &attempts, &credentials),
            None
        );
        // The old reading -- judge only what is still dispatchable -- is what the declared
        // form exists to replace: there the sibling vanishes from the question instead of
        // being answered.
    }

    /// The unproven sibling stops being unproven once the ledger says its allowance is
    /// spent. That is the ordinary shape of a metered pool after the free budget is used
    /// up, and it must still open the boundary.
    #[test]
    fn a_sibling_with_a_spent_ledger_closes_its_domain() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![attempt("a", "daily_allowance_exhausted")];
        let domains = vec![vec![ModelId::from("a")], vec![ModelId::from("b")]];
        // A spent allowance also contributes a credential block, so b is not dispatchable;
        // the ledger, not a request, is what establishes it.
        let only_a_dispatchable: HashSet<ModelId> = [ModelId::from("a")].into_iter().collect();
        assert_eq!(
            authorize_terminal(
                &models,
                &only_a_dispatchable,
                &attempts,
                TerminalPolicy::AllowanceOnly,
                &domains,
                |model| credentials.get(model).cloned(),
                |model| model.as_str() == "b",
                |_| false,
            ),
            Some("primary_pool_exhausted")
        );
    }

    /// The positive control for the same shape: when the second domain is also provably
    /// spent -- here because this request saw it answer with an exhaustion -- the boundary
    /// opens.
    #[test]
    fn every_declared_domain_spent_opens_the_boundary() {
        let (models, credentials) = legs(&[("a", "A"), ("b", "B")]);
        let attempts = vec![
            attempt("a", "daily_allowance_exhausted"),
            attempt("b", "provider_allowance_exhausted"),
        ];
        let domains = vec![vec![ModelId::from("a")], vec![ModelId::from("b")]];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &domains,
                &["b"],
            ),
            Some("primary_pool_exhausted")
        );
    }

    /// Google allocates its free daily requests per model inside one project, so two models
    /// behind the same API key do not share a budget. A spent model therefore says nothing
    /// about its sibling: promoting a per-model exhaustion to the credential would let one
    /// spent model buy the local tier while the second one still had its whole day.
    #[test]
    fn a_per_model_exhaustion_is_not_a_credential_exhaustion() {
        // Both legs draw on the same credential, which is the deployed Google shape.
        let (models, credentials) = legs(&[("g-model-a", "GOOGLE"), ("g-model-b", "GOOGLE")]);
        let attempts = vec![attempt("g-model-a", "daily_allowance_exhausted")];
        let domains = vec![
            vec![ModelId::from("g-model-a")],
            vec![ModelId::from("g-model-b")],
        ];
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &domains,
                &[],
            ),
            None,
            "the second model's own day was never established"
        );
        // Control: the same pool opens once the second model is provably spent itself.
        assert_eq!(
            authorize_with_state(
                &models,
                &dispatchable(&models),
                &attempts,
                &credentials,
                &domains,
                &["g-model-b"],
            ),
            Some("primary_pool_exhausted")
        );
    }

    /// A pool wholly blocked still needs allowance evidence, and only a quota-scoped block
    /// counts as that. This is the "not just a generic credential block" half of the rule.
    #[test]
    fn a_credential_block_only_opens_the_boundary_when_it_is_allowance_scoped() {
        let (models, credentials) = two_domains();
        let blocked = authorize_terminal(
            &models,
            &HashSet::new(),
            &[],
            TerminalPolicy::AllowanceOnly,
            &[],
            |model| credentials.get(model).cloned(),
            |_| false,
            |_| false,
        );
        assert_eq!(
            blocked, None,
            "a block that is not allowance-scoped is not exhaustion"
        );

        let allowed = authorize_terminal(
            &models,
            &HashSet::new(),
            &[],
            TerminalPolicy::AllowanceOnly,
            &[],
            |model| credentials.get(model).cloned(),
            |_| true,
            |_| false,
        );
        assert_eq!(allowed, Some("primary_pool_blocked"));
    }
}
