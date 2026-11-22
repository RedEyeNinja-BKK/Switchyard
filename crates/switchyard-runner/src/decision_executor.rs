// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fulfils a native `Call::Decision` against a configured decision backend.
//!
//! Upstream's driver emits a decision call with one selected model, and the hosting
//! client answered it with "not supported". This supplies that answer: resolve the
//! logical decision identity to its eligible ordered targets, then try them in order.
//!
//! Two things are deliberately kept apart.
//!
//! Eligibility is decided before dispatch. A backend that cannot answer one of the
//! question kinds, or cannot accept the request's state shape, is skipped without a
//! request being sent, so an unsupported type never looks like a serving failure.
//!
//! Failure classification decides fallback, and it is not completion fallback by
//! assumption. A transport error, timeout, rate limit or eligible server error means
//! the next backend is worth trying. A malformed request, an unsupported kind, a
//! contract violation, a credential problem, or a rejection of our own encoding is
//! evidence about this request or this configuration, not evidence that another
//! backend should serve it, so it stops immediately.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use switchyard_protocol::{DecisionRequest, DecisionResponse};

use crate::decision_transport::{
    DecisionTransport, StateForms, SupportedTypes, parse_response, validate_systemone,
};

/// Why a decision call could not be served by one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionSkip {
    /// The caller named an identity this deployment does not configure. Terminal, and
    /// checked before anything is dispatched, so a typo cannot silently reach whichever
    /// backend happened to be configured.
    TargetNotFound,
    /// The backend cannot answer one of the question kinds in the request.
    UnsupportedType,
    /// The backend cannot accept the request's state shape.
    UnsupportedState,
    /// The request does not fit the backend's wire contract.
    Contract,
    /// No credential is available for this backend.
    Unauthenticated,
}

impl DecisionSkip {
    /// Whether another backend could still serve the request.
    ///
    /// Only the properties of *this backend* are worth trying elsewhere: an unsupported
    /// question kind or an unsupported state shape is a fact about the backend, and the
    /// next one may well accept what this one refused.
    ///
    /// A contract violation is different. An out-of-bounds request is invalid for every
    /// backend on this transport, so trying the next one would send a request already
    /// known to be malformed. A credential failure is a deployment fault, and moving on
    /// would hide it. Both are terminal.
    pub fn try_next(self) -> bool {
        matches!(self, Self::UnsupportedType | Self::UnsupportedState)
    }

    /// A caller-facing explanation, for a refusal that is the caller's to fix.
    ///
    /// Written as the reason, not the enum name: `NotEligible(UnsupportedState)` tells a
    /// caller nothing about what to change, and this text is what a 400 body carries.
    pub fn message(self) -> &'static str {
        match self {
            Self::TargetNotFound => "no configured decision backend answers to that model name",
            Self::UnsupportedType => {
                "no configured decision backend can answer this question type; a request carrying \
                 a kind that none supports is refused before any provider is called"
            }
            Self::UnsupportedState => {
                "no configured decision backend accepts this `state` shape; this lane takes a \
                 structured state (an array of typed content blocks), not a bare string"
            }
            Self::Contract => {
                "the request does not fit the decision wire contract: question ids, option and \
                 level bounds, or a required field are out of range"
            }
            Self::Unauthenticated => "the decision backend has no credential configured",
        }
    }

    /// The skip that best explains a lane where every candidate was skipped.
    ///
    /// Precedence runs from the most specific deployment fault to the most general caller
    /// mistake, so the answer names the thing that would actually have to change. A
    /// credential fault is reported rather than masked behind a shape refusal, and a wire
    /// contract violation outranks a capability mismatch because the request is invalid
    /// everywhere, not merely unsuited to one backend.
    fn dominant(skips: &[Self]) -> Self {
        for candidate in [
            Self::Unauthenticated,
            Self::Contract,
            Self::UnsupportedState,
            Self::UnsupportedType,
        ] {
            if skips.contains(&candidate) {
                return candidate;
            }
        }
        Self::TargetNotFound
    }
}

/// A backend refused or failed to serve the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionFailure {
    /// Eligible backend that could not be reached or refused transiently.
    Serving {
        reason: &'static str,
        retryable: bool,
    },
    /// One candidate could not serve this request, but a later candidate might.
    ///
    /// Candidate-scoped: this is what an eligibility refusal looks like while the resolver
    /// still has candidates to try, and its retryability is a property of the skip.
    NotEligible(DecisionSkip),
    /// Nothing was dispatched anywhere in this lane, and nothing here could serve it.
    ///
    /// Lane-scoped and terminal. Distinguished from [`Self::NotEligible`] because the
    /// opposite decision was made at exactly this point once before: a lane where every
    /// candidate was skipped was reported as retryable, so a composed tier forwarded a
    /// request to a sibling family that could only refuse it in the same way, and the
    /// caller saw an outage instead of the shape mismatch it had sent. The authority to
    /// keep trying ends with the lane that has already seen the whole request.
    Refused(DecisionSkip),
}

impl DecisionFailure {
    /// Whether the resolver should try the next candidate.
    pub fn try_next(&self) -> bool {
        match self {
            DecisionFailure::Serving { retryable, .. } => *retryable,
            DecisionFailure::NotEligible(skip) => skip.try_next(),
            // A lane that could not dispatch anything has finished the question: a
            // sibling lane would be asked the same request and answer the same way.
            DecisionFailure::Refused(_) => false,
        }
    }

    /// A caller-facing explanation for a terminal refusal.
    ///
    /// Written as the situation, not the enum name: a caller who reads
    /// `Serving { reason: "daily_allowance_exhausted" }` has to decode our vocabulary, and
    /// the reason strings are also provider-side facts that must not leak as if they were
    /// caller errors.
    pub fn message(&self) -> String {
        match self {
            DecisionFailure::NotEligible(skip) | DecisionFailure::Refused(skip) => {
                skip.message().to_string()
            }
            DecisionFailure::Serving { reason, .. } => match *reason {
                "all_candidates_exhausted" => {
                    "every eligible decision backend failed to answer; the lane is temporarily \
                     exhausted"
                        .to_string()
                }
                "no_eligible_backend" => {
                    "no configured decision backend can serve this request".to_string()
                }
                "minute_allowance_throttled" => {
                    "this model's per-minute request or input-token budget is full; it refills \
                     within a minute"
                        .to_string()
                }
                "input_exceeds_minute_window" => {
                    "this request is larger than the model's whole per-minute input budget, so \
                     waiting will not make room for it"
                        .to_string()
                }
                "daily_allowance_exhausted" => {
                    "this model's daily request allowance is spent; it resets at the next \
                     provider day"
                        .to_string()
                }
                "daily_allowance_unavailable" => {
                    "this model's daily request counter cannot be accounted, so no budget may be \
                     spent"
                        .to_string()
                }
                other => format!("the decision backend could not answer ({other})"),
            },
        }
    }
}

/// The budgets one decision backend's physical calls draw on.
///
/// The same two shapes the capability surfaces use, for the same reason: a provider caps
/// requests and input tokens over a rolling minute, and requests over its civil day. A
/// decision call is one physical request to one physical model, so it is paced exactly like
/// any other physical call: one reservation, released only if nothing was sent.
#[derive(Debug, Default)]
pub struct DecisionBudget {
    /// Persistent per-provider-day request counter, when a daily limit is declared.
    pub allowance: Option<Arc<switchyard_llm_client::DailyRequestAllowance>>,
    /// Rolling per-minute request and input-token window, when either is declared.
    pub pacing: Option<Arc<switchyard_llm_client::PacedQuota>>,
}

impl DecisionBudget {
    /// Whether this budget declares nothing at all.
    pub fn is_unlimited(&self) -> bool {
        self.allowance.is_none() && self.pacing.is_none()
    }
}

/// One configured decision backend, resolved for serving.
///
/// `Debug` is written by hand because the candidate holds a credential: a derived
/// implementation would print the key into logs, tracing, receipts and test failures.
/// Only presence and the logical identity are ever shown.
#[derive(Clone)]
pub struct DecisionCandidate {
    /// Configuration name of the target.
    pub name: String,
    /// Provider-side model or engine id sent on the wire.
    pub model: String,
    /// Wire family.
    pub transport: DecisionTransport,
    /// Full request URL.
    pub url: String,
    /// Credential for this backend. Never rendered by `Debug`.
    pub api_key: Option<String>,
    /// Name of the environment variable this credential was read from, for diagnostics.
    pub api_key_env: Option<String>,
    /// Question kinds this backend currently answers.
    pub supported_types: SupportedTypes,
    /// State shapes this backend accepts.
    pub state_forms: StateForms,
    /// The provider budgets this backend's calls are charged to.
    pub budget: Option<Arc<DecisionBudget>>,
}

impl std::fmt::Debug for DecisionCandidate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecisionCandidate")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("transport", &self.transport)
            .field("url", &self.url)
            .field("supported_types", &self.supported_types)
            .field("state_forms", &self.state_forms)
            .field("credential_present", &self.api_key.is_some())
            .field("credential_env", &self.api_key_env)
            .field("budgeted", &self.budget.is_some())
            .finish()
    }
}

impl DecisionCandidate {
    /// Whether this backend may serve the request, without sending anything.
    ///
    /// Order matters: contract bounds are checked first, because a request outside the
    /// wire bounds is invalid for every backend on that transport and must not be
    /// reported as one backend being unavailable.
    pub fn eligibility(&self, request: &DecisionRequest) -> Result<(), DecisionSkip> {
        if self.transport == DecisionTransport::SystemOne {
            validate_systemone(request).map_err(|_| DecisionSkip::Contract)?;
        }
        if !self.supported_types.accepts_request(request) {
            return Err(DecisionSkip::UnsupportedType);
        }
        if !self.state_forms.accepts(&request.context) {
            return Err(DecisionSkip::UnsupportedState);
        }
        Ok(())
    }
}

/// Selects the eligible ordered candidates for one logical decision identity.
///
/// The smallest resolver that answers one question: which of the configured ordered
/// targets may serve this request. Ordering is the configured order; there is no
/// scoring, no ranking and no second routing system.
#[derive(Default)]
pub struct DecisionResolver {
    /// Configured targets in preference order.
    pub candidates: Vec<DecisionCandidate>,
}

impl DecisionResolver {
    /// Candidates eligible for this request, in configured order.
    pub fn eligible(&self, request: &DecisionRequest) -> Vec<&DecisionCandidate> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.eligibility(request).is_ok())
            .collect()
    }
}

/// Outcome of resolving and serving one decision request.
#[derive(Debug)]
pub struct DecisionOutcome {
    /// The response in upstream's native types.
    pub response: DecisionResponse,
    /// Name of the backend that answered.
    pub served_by: String,
    /// Candidates that were tried and failed, with the reason.
    pub attempts: Vec<(String, DecisionFailure)>,
    /// The child POLICY lane that served the request, when the caller addressed a
    /// composed lane. A tier must not reduce to "some backend answered": which family
    /// policy produced the answer is the operationally meaningful part.
    pub served_by_lane: Option<String>,
}

/// Default deadline for one decision call.
const DEFAULT_DECISION_TIMEOUT_SECONDS: u64 = 30;

/// Reserves one physical decision call against the backend's declared budgets.
///
/// Mirrors the completion path exactly, including the order: the minute window first,
/// because only its ticket is releasable, then the daily counter, because a daily
/// reservation is written before the call and is the provider's own countable unit. The
/// window ticket is released if the daily counter then refuses, since nothing was sent.
///
/// One reservation per physical call. A decision request is one provider request, so this
/// is not a fan-out: a ten-question request still spends one unit.
fn reserve_decision_call(
    candidate: &DecisionCandidate,
    input_tokens: u64,
) -> Result<Option<switchyard_llm_client::PacingTicket>, DecisionFailure> {
    let Some(budget) = &candidate.budget else {
        return Ok(None);
    };
    let ticket = match &budget.pacing {
        None => None,
        Some(pacing) => match pacing.reserve(input_tokens) {
            Ok(ticket) => ticket,
            Err(switchyard_llm_client::PacingError::Throttled(_throttled)) => {
                // This backend has no room in its current window. Another backend may, so
                // this stays recoverable, but a lane of one target must answer 429 rather
                // than pretending to be down.
                return Err(DecisionFailure::Serving {
                    reason: "minute_allowance_throttled",
                    retryable: true,
                });
            }
            Err(switchyard_llm_client::PacingError::ExceedsWindow { .. }) => {
                return Err(DecisionFailure::Serving {
                    reason: "input_exceeds_minute_window",
                    retryable: true,
                });
            }
        },
    };
    if let Some(allowance) = &budget.allowance {
        match allowance.try_reserve() {
            Ok(()) => {}
            Err(switchyard_llm_client::ReserveError::Exhausted) => {
                if let (Some(pacing), Some(ticket)) = (&budget.pacing, &ticket) {
                    pacing.release(ticket);
                }
                return Err(DecisionFailure::Serving {
                    reason: "daily_allowance_exhausted",
                    retryable: true,
                });
            }
            Err(switchyard_llm_client::ReserveError::Unusable(_)) => {
                if let (Some(pacing), Some(ticket)) = (&budget.pacing, &ticket) {
                    pacing.release(ticket);
                }
                // The counter cannot be trusted, so no budget may be spent. This is a
                // deployment fault: trying a sibling would hide it.
                return Err(DecisionFailure::Serving {
                    reason: "daily_allowance_unavailable",
                    retryable: false,
                });
            }
        }
    }
    Ok(ticket)
}

/// Releases a minute slot for a call that never reached the provider.
fn release_decision_call(
    candidate: &DecisionCandidate,
    ticket: &Option<switchyard_llm_client::PacingTicket>,
) {
    if let (Some(budget), Some(ticket)) = (&candidate.budget, ticket)
        && let Some(pacing) = &budget.pacing
    {
        pacing.release(ticket);
    }
}

/// Charges the reservation with the provider's own input-token count, when it reports one.
///
/// Read from the two spellings a decision backend uses - Google's `total_input_tokens` and
/// the OpenAI-shaped `prompt_tokens` - and never invented: a backend that reports no count
/// keeps the estimate it was reserved at, which is the conservative direction already.
fn reconcile_decision_call(
    candidate: &DecisionCandidate,
    ticket: &Option<switchyard_llm_client::PacingTicket>,
    payload: &Value,
) {
    let Some(ticket) = ticket else { return };
    let Some(budget) = &candidate.budget else {
        return;
    };
    let Some(pacing) = &budget.pacing else { return };
    let usage = payload.get("usage");
    let reported = usage
        .and_then(|usage| {
            usage
                .get("total_input_tokens")
                .or_else(|| usage.get("prompt_tokens"))
                .or_else(|| usage.get("input_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if reported > 0 {
        pacing.reconcile(ticket, reported);
    }
}

async fn send(candidate: &DecisionCandidate, body: Value) -> Result<Value, DecisionFailure> {
    let Some(api_key) = candidate
        .api_key
        .as_deref()
        .filter(|key| !key.trim().is_empty())
    else {
        return Err(DecisionFailure::NotEligible(DecisionSkip::Unauthenticated));
    };
    // Budgets are taken before anything is built or sent. A refusal here is a property of
    // this backend's remaining allowance, so it carries no upstream status and nothing was
    // counted by the provider.
    let estimate = switchyard_llm_client::allowance::estimate_input_tokens(&body);
    let ticket = reserve_decision_call(candidate, estimate)?;
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_DECISION_TIMEOUT_SECONDS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(_) => {
            // Nothing left the process, so the window slot goes back.
            release_decision_call(candidate, &ticket);
            return Err(DecisionFailure::Serving {
                reason: "client_unavailable",
                retryable: false,
            });
        }
    };
    // Google reads its key from `x-goog-api-key`; every other decision backend takes a
    // bearer token. Sending the wrong one is an authentication failure that looks like a
    // credential problem, so the header is chosen from the transport rather than guessed.
    let shell = client.post(&candidate.url);
    let shell = if candidate.transport.uses_google_api_key() {
        shell.header("x-goog-api-key", api_key)
    } else {
        shell.bearer_auth(api_key)
    };
    let response = shell.json(&body).send().await.map_err(|error| {
        // A transport failure or deadline is exactly the evidence that the next
        // backend should be tried.
        let retryable = error.is_timeout() || error.is_connect() || error.is_request();
        DecisionFailure::Serving {
            reason: if error.is_timeout() {
                "timeout"
            } else if error.is_connect() {
                "connect"
            } else {
                "transport"
            },
            retryable,
        }
    })?;
    let status = response.status();
    let payload = response
        .json::<Value>()
        .await
        .map_err(|_| DecisionFailure::Serving {
            reason: "unreadable_response",
            retryable: false,
        })?;
    if status.is_success() {
        reconcile_decision_call(candidate, &ticket, &payload);
        return Ok(payload);
    }
    // A rejection of our own encoding or a credential problem is not a reason to try a
    // different backend, so only the transient classes are retryable.
    let retryable = status.as_u16() == 408
        || status.as_u16() == 429
        || (status.is_server_error() && status.as_u16() != 501);
    let reason: &'static str = if status.as_u16() == 401 || status.as_u16() == 403 {
        "auth"
    } else if status.as_u16() == 400 || status.as_u16() == 422 {
        "rejected"
    } else if status.as_u16() == 429 {
        "rate_limited"
    } else if status.is_server_error() {
        "server_error"
    } else {
        "http"
    };
    Err(DecisionFailure::Serving { reason, retryable })
}

/// The single failure that best explains a set of per-candidate attempts.
///
/// Three cases stay distinct, because a caller acts differently on each:
///
/// * something was actually sent and failed transiently: the lane is exhausted, which is
///   an availability answer and may be recovered by retrying later;
/// * candidates were refused by our own budget: 429-class, not an outage - a lane whose
///   daily allowance is spent is working exactly as configured;
/// * every candidate was skipped before dispatch: nothing was tried, so the answer is
///   about the request's shape or the deployment's wiring, never about availability. This
///   is what the surface turns into a 400 with a reason the caller can act on.
fn aggregate_attempts(
    attempts: &[(String, DecisionFailure)],
    exhausted_reason: &'static str,
) -> DecisionFailure {
    if attempts.is_empty() {
        return DecisionFailure::Serving {
            reason: "no_eligible_backend",
            retryable: false,
        };
    }
    let mut sent = false;
    let mut quota: Option<&'static str> = None;
    let mut skips: Vec<DecisionSkip> = Vec::new();
    for (_, failure) in attempts {
        match failure {
            DecisionFailure::Serving { reason, .. } => match *reason {
                "minute_allowance_throttled"
                | "input_exceeds_minute_window"
                | "daily_allowance_exhausted"
                | "daily_allowance_unavailable" => {
                    // Keep the most severe: an unusable counter outranks an exhausted one,
                    // which outranks a minute window that refills shortly.
                    let rank = |reason: &'static str| match reason {
                        "daily_allowance_unavailable" => 3,
                        "daily_allowance_exhausted" => 2,
                        "minute_allowance_throttled" => 1,
                        _ => 0,
                    };
                    if quota.is_none_or(|current| rank(reason) > rank(current)) {
                        quota = Some(*reason);
                    }
                }
                _ => sent = true,
            },
            // A nested aggregate cannot occur through either entry point, but if one is
            // ever passed in it is still a pre-dispatch refusal and must not be counted as
            // something that was sent.
            DecisionFailure::NotEligible(skip) | DecisionFailure::Refused(skip) => {
                skips.push(*skip)
            }
        }
    }
    if sent {
        return DecisionFailure::Serving {
            reason: exhausted_reason,
            retryable: true,
        };
    }
    if let Some(reason) = quota {
        return DecisionFailure::Serving {
            reason,
            retryable: reason != "daily_allowance_unavailable",
        };
    }
    DecisionFailure::Refused(DecisionSkip::dominant(&skips))
}

/// What a caller asked for, resolved against the deployment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionIdentity<'a> {
    /// A concrete decision target or its model id: served by that backend alone.
    Target(&'a str),
    /// The explicitly configured logical lane.
    Lane,
}

/// Resolves the identity a caller named.
///
/// An explicit lane identity is the lane; a configured target or model id is that
/// target; anything else is unknown and terminal. The lane is never inferred from a
/// failed lookup, because that would turn a typo into a different backend's answer.
pub fn resolve_identity<'a>(
    resolver: &DecisionResolver,
    lane_name: Option<&str>,
    requested: &'a str,
) -> Result<DecisionIdentity<'a>, DecisionFailure> {
    if let Some(name) = lane_name
        && requested == name
    {
        return Ok(DecisionIdentity::Lane);
    }
    if resolver
        .candidates
        .iter()
        .any(|candidate| candidate.name == requested || candidate.model == requested)
    {
        return Ok(DecisionIdentity::Target(requested));
    }
    Err(DecisionFailure::NotEligible(DecisionSkip::TargetNotFound))
}

/// Serves one named target, and nothing else.
///
/// A concrete decision call names its model, so it means that backend. Eligibility
/// still applies, but a backend that cannot serve the request produces that result
/// rather than an answer from a different provider: silently substituting one would
/// make a pinned diagnostic call report another backend's opinion.
pub async fn serve_target(
    candidate: &DecisionCandidate,
    request: &DecisionRequest,
) -> Result<DecisionOutcome, DecisionFailure> {
    if let Err(skip) = candidate.eligibility(request) {
        return Err(DecisionFailure::NotEligible(skip));
    }
    let body = candidate
        .transport
        .request_body(&candidate.model, request)
        .map_err(|_| DecisionFailure::NotEligible(DecisionSkip::Contract))?;
    match send(candidate, body).await {
        Ok(payload) => {
            let response = parse_response(&payload).map_err(|_| DecisionFailure::Serving {
                reason: "undecodable_response",
                retryable: false,
            })?;
            Ok(DecisionOutcome {
                response,
                served_by: candidate.name.clone(),
                attempts: Vec::new(),
                served_by_lane: None,
            })
        }
        Err(failure) => Err(failure),
    }
}

/// Resolves eligible candidates and serves the request through the first that answers.
///
/// The same resolver and executor back both the internal decision call path and any
/// public typed surface, so there is one implementation of Decision semantics.
pub async fn resolve_and_serve(
    resolver: &DecisionResolver,
    request: &DecisionRequest,
) -> Result<DecisionOutcome, DecisionFailure> {
    let mut attempts = Vec::new();
    for candidate in &resolver.candidates {
        if let Err(skip) = candidate.eligibility(request) {
            let failure = DecisionFailure::NotEligible(skip);
            // Terminal before any request is sent: an out-of-bounds request must never
            // reach a backend, and a credential fault must not be hidden behind a
            // different backend answering.
            if !failure.try_next() {
                return Err(failure);
            }
            attempts.push((candidate.name.clone(), failure));
            continue;
        }
        let body = match candidate.transport.request_body(&candidate.model, request) {
            Ok(body) => body,
            Err(_) => {
                let failure = DecisionFailure::NotEligible(DecisionSkip::Contract);
                if !failure.try_next() {
                    return Err(failure);
                }
                attempts.push((candidate.name.clone(), failure));
                continue;
            }
        };
        match send(candidate, body).await {
            Ok(payload) => {
                let response = parse_response(&payload).map_err(|_| DecisionFailure::Serving {
                    // A schema mismatch we caused must not trigger a different backend:
                    // the next one would receive the same request and answer the same way.
                    reason: "undecodable_response",
                    retryable: false,
                })?;
                return Ok(DecisionOutcome {
                    response,
                    served_by: candidate.name.clone(),
                    attempts,
                    served_by_lane: None,
                });
            }
            Err(failure) => {
                if !failure.try_next() {
                    return Err(failure);
                }
                attempts.push((candidate.name.clone(), failure));
            }
        }
    }
    // Every candidate was tried and none answered. What that MEANS is not one thing, so
    // it is classified rather than flattened: a transient serving failure is an outage, a
    // budget refusal is a rate limit, and a lane where nothing was ever dispatched is
    // telling the caller its request did not fit - which is a different status and a
    // different remedy.
    Err(aggregate_attempts(&attempts, "all_candidates_exhausted"))
}

/// One child lane's contribution to a composed request.
#[derive(Clone)]
pub struct LaneChild {
    lane: String,
    resolver: std::sync::Arc<DecisionResolver>,
}

impl LaneChild {
    /// The child lane's public name.
    pub fn lane(&self) -> &str {
        &self.lane
    }

    /// Builds one child of a composed lane from its name and resolver.
    pub fn new(lane: impl Into<String>, resolver: std::sync::Arc<DecisionResolver>) -> Self {
        Self {
            lane: lane.into(),
            resolver,
        }
    }
}

/// Serves a request addressed to a lane that composes child POLICY lanes.
///
/// Children are invoked whole and in order, in process, through the same
/// `resolve_and_serve` used by a leaf lane. A tier therefore inherits each child's
/// candidate order, question-kind eligibility, terminal-vs-retry failure policy and
/// telemetry, and a change to a child is inherited rather than copied.
///
/// Three outcomes stay distinct, mirroring the route composition precedent:
///
/// * a child whose own candidates are all ineligible is skipped without a provider
///   call, and the next child is tried (routing, not failure);
/// * a child that fails transiently after exhausting its own candidates may yield, so
///   one child's outage does not decide the tier;
/// * a child that fails for a reason no other lane can fix (contract, credential,
///   undecodable response) stops the tier rather than hiding the defect behind a
///   different family's answer.
///
/// `stack` carries the current recursion path so a cycle fails closed instead of
/// recursing. Cycles are also rejected at configuration load, so this is a backstop.
pub async fn serve_composed(
    children: &[LaneChild],
    request: &DecisionRequest,
    stack: &mut Vec<String>,
) -> Result<DecisionOutcome, DecisionFailure> {
    let mut attempts = Vec::new();
    for child in children {
        if stack.contains(&child.lane) {
            return Err(DecisionFailure::Serving {
                reason: "decision_lane_cycle",
                retryable: false,
            });
        }
        stack.push(child.lane.clone());
        let outcome = resolve_and_serve(&child.resolver, request).await;
        stack.pop();
        match outcome {
            Ok(mut served) => {
                served.served_by_lane = Some(child.lane.clone());
                for (candidate, failure) in served.attempts.drain(..) {
                    attempts.push((format!("{}/{}", child.lane, candidate), failure));
                }
                served.attempts = attempts;
                return Ok(served);
            }
            Err(failure) => {
                if !failure.try_next() {
                    return Err(failure);
                }
                attempts.push((child.lane.clone(), failure.clone()));
            }
        }
    }
    // Every child yielded. The classification is the same question one level up: if any
    // child actually reached a provider and failed transiently the tier is exhausted, and
    // if every child refused before dispatch the request itself did not fit the tier.
    Err(aggregate_attempts(&attempts, "all_child_lanes_exhausted"))
}

/// What an explicitly named identity resolves to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedDispatch {
    /// The name is a COMPOSED lane: it owns child policy lanes.
    Composed,
    /// The name is a leaf lane: it owns ordered physical candidates.
    Leaf,
    /// The name is not a configured decision lane, so the caller falls through to the
    /// route-local resolver. It is never silently served by some other lane.
    Unknown,
}

/// Classifies an explicitly requested decision identity against the deployment's lanes.
///
/// This is the single decision point the public surface and the internal
/// `Call::Decision` handler both use, so the two cannot disagree about what a caller
/// named. An explicit name is authoritative and is classified BEFORE any question-kind
/// selection, which is what keeps two policy lanes that both support a kind from
/// stealing each other's requests.
pub fn classify_named(
    composition: &BTreeMap<String, Vec<String>>,
    lanes: &BTreeMap<String, LaneResolverHandle>,
    requested: &str,
) -> NamedDispatch {
    if requested.is_empty() {
        return NamedDispatch::Unknown;
    }
    if composition.contains_key(requested) {
        return NamedDispatch::Composed;
    }
    if lanes.contains_key(requested) {
        return NamedDispatch::Leaf;
    }
    NamedDispatch::Unknown
}

/// Serves a request whose caller named a decision lane, or reports that it did not.
///
/// Returns `Ok(None)` when the name is not a configured decision lane, which tells the
/// caller to fall through to its own route-local resolution. Returns `Err` when the name
/// IS a configured lane and serving it failed - an explicitly named lane that cannot
/// serve must never be answered by a different lane.
///
/// This is the whole internal/public decision point in one testable function: the
/// previous split, where the public surface honoured a named lane and the internal
/// handler bypassed it, was possible precisely because this logic was inlined in two
/// places.
pub async fn serve_named(
    composition: &BTreeMap<String, Vec<String>>,
    lanes: &BTreeMap<String, LaneResolverHandle>,
    requested: &str,
    request: &DecisionRequest,
) -> Result<Option<DecisionOutcome>, DecisionFailure> {
    match classify_named(composition, lanes, requested) {
        NamedDispatch::Unknown => Ok(None),
        NamedDispatch::Composed => {
            let children = composition.get(requested).cloned().unwrap_or_default();
            // An unknown child fails closed rather than silently shrinking the policy.
            let ordered = composed_children(lanes, &children)?;
            let mut stack = vec![requested.to_string()];
            serve_composed(&ordered, request, &mut stack)
                .await
                .map(Some)
        }
        NamedDispatch::Leaf => {
            let lane = lanes
                .get(requested)
                .map(|handle| Arc::clone(&handle.resolver))
                .ok_or(DecisionFailure::NotEligible(DecisionSkip::TargetNotFound))?;
            // The named lane is served by ITS OWN candidates only; a sibling lane may not
            // answer for it.
            match resolve_identity(&lane, Some(requested), requested)? {
                DecisionIdentity::Target(target) => {
                    let candidate = lane
                        .candidates
                        .iter()
                        .find(|candidate| candidate.name == target || candidate.model == target)
                        .ok_or(DecisionFailure::NotEligible(DecisionSkip::TargetNotFound))?;
                    serve_target(candidate, request).await.map(Some)
                }
                DecisionIdentity::Lane => resolve_and_serve(&lane, request).await.map(Some),
            }
        }
    }
}

/// Builds the ordered child lanes for a composed lane, in configured order.
///
/// An unknown child is reported rather than skipped: a composed lane that silently lost
/// a child would quietly change policy instead of failing closed.
pub fn composed_children(
    lanes: &BTreeMap<String, LaneResolverHandle>,
    children: &[String],
) -> Result<Vec<LaneChild>, DecisionFailure> {
    let mut ordered = Vec::with_capacity(children.len());
    for child in children {
        let handle = lanes
            .get(child.as_str())
            .ok_or(DecisionFailure::NotEligible(DecisionSkip::TargetNotFound))?;
        ordered.push(LaneChild::new(child.clone(), Arc::clone(&handle.resolver)));
    }
    Ok(ordered)
}

/// Resolves the child policy lanes of a composed lane, in configured order.
pub fn lane_children(
    composition: &BTreeMap<String, Vec<String>>,
    lanes: &BTreeMap<String, LaneResolverHandle>,
    lane: &str,
) -> Option<Vec<LaneChild>> {
    let names = composition.get(lane)?;
    let mut children = Vec::with_capacity(names.len());
    for name in names {
        let handle = lanes.get(name)?;
        children.push(LaneChild {
            lane: name.clone(),
            resolver: handle.resolver.clone(),
        });
    }
    Some(children)
}

/// A lane's resolver, without its question-kind scoping, for composition lookup.
#[derive(Clone)]
pub struct LaneResolverHandle {
    /// The lane's ordered candidates.
    pub resolver: std::sync::Arc<DecisionResolver>,
}

/// Builds the ordered child lanes of a composed lane, from the runner's own index.
///
/// Resolution is in process and uses each child's own resolver, so a child keeps its
/// candidate order, eligibility and failure policy. Returns `None` when the composed
/// lane cannot be built, which the caller reports rather than silently degrading to a
/// flat list of physical targets.
pub fn resolve_composed(
    runner: &crate::Runner,
    lane: &str,
    _request: &switchyard_protocol::DecisionRequest,
) -> Option<Vec<LaneChild>> {
    let children = runner.decision_lane_children(lane)?;
    let mut resolved = Vec::with_capacity(children.len());
    for name in children {
        let handle = runner.named_decision_lane(name)?;
        resolved.push(LaneChild::new(
            name.to_string(),
            Arc::clone(&handle.resolver),
        ));
    }
    Some(resolved)
}

/// Serializes a native decision response in the retained compatibility shape.
///
/// The shape is the canonical one: keyed `answers`, each tagged with its kind and
/// carrying the provider's own value plus probabilities and confidence where they
/// were reported. A caller reading either this or a direct provider response sees the
/// same fields, because they are the same answer.
pub fn compatibility_response(response: &switchyard_protocol::DecisionResponse) -> Value {
    let mut answers = serde_json::Map::new();
    for (id, answer) in &response.answers {
        let mut entry = serde_json::Map::new();
        match &answer.value {
            switchyard_protocol::DecisionValue::Boolean(estimate) => {
                let probability = match estimate {
                    switchyard_protocol::BooleanEstimate::ProbabilityTrue(probability) => {
                        probability.0
                    }
                    switchyard_protocol::BooleanEstimate::Value(value) => {
                        if *value {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
                entry.insert("type".to_string(), json!("noul"));
                entry.insert("noul".to_string(), json!(probability));
            }
            switchyard_protocol::DecisionValue::Choice {
                selected,
                probabilities,
            } => {
                entry.insert("type".to_string(), json!("choice"));
                entry.insert("choice".to_string(), json!(selected));
                if let Some(probabilities) = probabilities {
                    let map: serde_json::Map<String, Value> = probabilities
                        .iter()
                        .map(|(name, probability)| (name.clone(), json!(probability.0)))
                        .collect();
                    entry.insert("probabilities".to_string(), Value::Object(map));
                }
            }
            switchyard_protocol::DecisionValue::Score {
                value,
                probabilities,
            } => {
                entry.insert("type".to_string(), json!("score"));
                entry.insert("score".to_string(), json!(value.0));
                if let Some(probabilities) = probabilities {
                    let ordered: Vec<Value> =
                        probabilities.iter().map(|value| json!(value.0)).collect();
                    entry.insert("probabilities".to_string(), Value::Array(ordered));
                }
            }
        }
        if let Some(confidence) = answer.provider_confidence {
            entry.insert("confidence".to_string(), json!(confidence.0));
        }
        answers.insert(id.clone(), Value::Object(entry));
    }
    let mut body = json!({ "answers": Value::Object(answers) });
    if let Some(id) = &response.id {
        body["id"] = json!(id);
    }
    if let Some(model) = &response.model {
        body["model"] = json!(model.as_str());
    }
    let usage = &response.usage;
    if usage.input_tokens.is_some() || usage.output_tokens.is_some() || usage.total_tokens.is_some()
    {
        body["usage"] = json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        });
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;
    use switchyard_protocol::{DecisionKind, DecisionQuestion};

    pub(crate) fn candidate(
        name: &str,
        transport: DecisionTransport,
        types: SupportedTypes,
        state_forms: StateForms,
    ) -> DecisionCandidate {
        DecisionCandidate {
            name: name.to_string(),
            model: "backend-model".to_string(),
            transport,
            url: format!("http://{name}.test{}", transport.path()),
            api_key: Some("SUPER_SECRET_DECISION_TEST_KEY".to_string()),
            api_key_env: Some("TEST_DECISION_KEY".to_string()),
            supported_types: types,
            state_forms,
            budget: None,
        }
    }

    pub(crate) fn noul_request(context: Value) -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "risk".to_string(),
            DecisionQuestion {
                instructions: json!("Is this risky?"),
                kind: DecisionKind::Boolean {
                    true_description: None,
                    false_description: None,
                },
            },
        );
        DecisionRequest {
            model: None,
            context,
            questions,
        }
    }

    pub(crate) fn choice_request() -> DecisionRequest {
        let mut questions = BTreeMap::new();
        questions.insert(
            "team".to_string(),
            DecisionQuestion {
                instructions: json!("Pick"),
                kind: DecisionKind::Choice {
                    options: vec![switchyard_protocol::ChoiceOption {
                        id: "a".into(),
                        description: None,
                    }],
                },
            },
        );
        DecisionRequest {
            model: None,
            context: json!("state"),
            questions,
        }
    }

    pub(crate) fn local() -> DecisionCandidate {
        candidate(
            "laya",
            DecisionTransport::SystemOne,
            SupportedTypes::ALL,
            StateForms::ALL,
        )
    }

    /// A local backend serves every kind with either state form, so it is the default
    /// answer for anything a caller can express.
    #[test]
    fn a_local_backend_is_eligible_for_every_supported_shape() {
        let backend = local();
        assert!(backend.eligibility(&noul_request(json!("plain"))).is_ok());
        assert!(
            backend
                .eligibility(&noul_request(json!({"task": "x"})))
                .is_ok()
        );
        assert!(backend.eligibility(&choice_request()).is_ok());
    }

    /// SPAN is noul-only and, as observed on 2026-10-02, takes a plain string state.
    /// Both dimensions must be checked before dispatch.
    #[test]
    fn span_exclusion_is_multi_dimensional() {
        let span = candidate(
            "span",
            DecisionTransport::OpenRouterAlpha,
            SupportedTypes::NOUL_ONLY,
            StateForms::SPAN_OBSERVED,
        );
        assert!(
            span.eligibility(&noul_request(json!("plain text"))).is_ok(),
            "noul with a plain state is eligible"
        );
        assert_eq!(
            span.eligibility(&choice_request()).unwrap_err(),
            DecisionSkip::UnsupportedType
        );
        assert_eq!(
            span.eligibility(&noul_request(json!({"task": "x"})))
                .unwrap_err(),
            DecisionSkip::UnsupportedState,
            "an unsupported state shape excludes the backend before dispatch"
        );
    }

    /// A request outside the wire bounds is a contract error, not a backend being
    /// unavailable, so it is not treated as a reason to try the next one.
    #[test]
    fn a_contract_violation_is_not_an_availability_signal() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "wide".to_string(),
            DecisionQuestion {
                instructions: json!("Pick"),
                kind: DecisionKind::Choice {
                    options: (0..256)
                        .map(|index| switchyard_protocol::ChoiceOption {
                            id: format!("o{index}"),
                            description: None,
                        })
                        .collect(),
                },
            },
        );
        let request = DecisionRequest {
            model: None,
            context: json!("state"),
            questions,
        };
        let failure = local().eligibility(&request).unwrap_err();
        assert_eq!(failure, DecisionSkip::Contract);
    }

    /// The resolver keeps configured order and drops only ineligible candidates.
    #[test]
    fn resolution_preserves_configured_order() {
        let resolver = DecisionResolver {
            candidates: vec![
                candidate(
                    "laya",
                    DecisionTransport::SystemOne,
                    SupportedTypes::ALL,
                    StateForms::ALL,
                ),
                candidate(
                    "span",
                    DecisionTransport::OpenRouterAlpha,
                    SupportedTypes::NOUL_ONLY,
                    StateForms::SPAN_OBSERVED,
                ),
            ],
        };
        let both = resolver.eligible(&noul_request(json!("plain")));
        assert_eq!(
            both.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["laya", "span"],
            "a plain noul is eligible for both, in configured order"
        );
        let choice = resolver.eligible(&choice_request());
        assert_eq!(
            choice.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["laya"],
            "a choice request excludes the noul-only backend entirely"
        );
    }

    /// Classification is the contract: transient classes try the next backend, and
    /// anything about the request or our configuration stops immediately.
    #[test]
    fn failure_classification_decides_whether_to_try_another_backend() {
        for retryable in [
            DecisionFailure::Serving {
                reason: "timeout",
                retryable: true,
            },
            DecisionFailure::Serving {
                reason: "connect",
                retryable: true,
            },
            DecisionFailure::Serving {
                reason: "rate_limited",
                retryable: true,
            },
            DecisionFailure::Serving {
                reason: "server_error",
                retryable: true,
            },
            DecisionFailure::NotEligible(DecisionSkip::UnsupportedType),
            DecisionFailure::NotEligible(DecisionSkip::UnsupportedState),
        ] {
            assert!(
                retryable.try_next(),
                "{retryable:?} should try the next backend"
            );
        }

        for terminal in [
            // A contract violation is terminal: the request is invalid for every backend
            // on this transport, so dispatching another would send a malformed request.
            DecisionFailure::NotEligible(DecisionSkip::Contract),
            DecisionFailure::Serving {
                reason: "auth",
                retryable: false,
            },
            DecisionFailure::Serving {
                reason: "rejected",
                retryable: false,
            },
            DecisionFailure::Serving {
                reason: "undecodable_response",
                retryable: false,
            },
            DecisionFailure::NotEligible(DecisionSkip::Unauthenticated),
        ] {
            assert!(
                !terminal.try_next(),
                "{terminal:?} must not silently move to another backend"
            );
        }
    }

    // --- one reservation per physical decision call ------------------------------
    //
    // A decision call is one provider request, so it draws on exactly the budgets the same
    // model draws on for generation. These tests assert the two properties that are easy to
    // get wrong and expensive to discover in production: that a refusal leaves no slot
    // consumed, and that a window and a counter are charged once each, not once per child.

    use switchyard_llm_client::{
        AdmissionMargin, DailyRequestAllowance, GovernorKey, PacedQuota, QuotaGovernor,
        WindowLimits, allowance_path,
    };

    fn scratch(label: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = format!(
            "{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            label
        );
        std::env::temp_dir()
            .join("switchyard-decision-budget-tests")
            .join(unique)
    }

    fn budget_key() -> GovernorKey {
        GovernorKey::new(
            "google-free",
            switchyard_protocol::ModelId::new("google/gemini-3.8-flash"),
            "decide",
        )
    }

    fn minute_window(governor: &std::sync::Arc<QuotaGovernor>, requests: u64) -> PacedQuota {
        PacedQuota::new(
            governor.clone(),
            budget_key(),
            WindowLimits {
                requests_per_minute: Some(requests),
                input_tokens_per_minute: None,
            },
            AdmissionMargin::NONE,
        )
    }

    fn budgeted_candidate(
        allowance: Option<DailyRequestAllowance>,
        pacing: Option<PacedQuota>,
    ) -> DecisionCandidate {
        let mut candidate = candidate(
            "gemini",
            DecisionTransport::SystemOne,
            SupportedTypes::ALL,
            StateForms::ALL,
        );
        candidate.budget = Some(Arc::new(DecisionBudget {
            allowance: allowance.map(Arc::new),
            pacing: pacing.map(Arc::new),
        }));
        candidate
    }

    /// A daily refusal must hand the minute slot back.
    ///
    /// The window is reserved first because only its ticket can be released, and a refusal
    /// that kept the slot would make every later call in that minute look throttled by a
    /// budget it never actually spent.
    #[test]
    fn a_daily_refusal_releases_the_minute_window() {
        let directory = scratch("release");
        let governor = std::sync::Arc::new(QuotaGovernor::new());
        let allowance = DailyRequestAllowance::new(
            "google-free",
            switchyard_protocol::ModelId::new("google/gemini-3.8-flash"),
            // Already spent: the very first reservation is refused.
            0,
            allowance_path(&directory, "google-free", "google/gemini-3.8-flash"),
        );
        let candidate = budgeted_candidate(Some(allowance), Some(minute_window(&governor, 5)));

        let failure =
            reserve_decision_call(&candidate, 10).expect_err("an exhausted day refuses the call");
        assert_eq!(
            failure,
            DecisionFailure::Serving {
                reason: "daily_allowance_exhausted",
                retryable: true,
            }
        );
        let (requests, tokens) = governor.used(&budget_key(), std::time::SystemTime::now());
        assert_eq!(requests, 0, "the refused call must keep no minute slot");
        assert_eq!(tokens, 0);
    }

    /// A granted call is charged once to each budget, and the ticket it returns is the one
    /// that later releases or reconciles it.
    #[test]
    fn a_granted_call_is_charged_once_to_each_budget() {
        let directory = scratch("charge");
        let governor = std::sync::Arc::new(QuotaGovernor::new());
        let allowance = DailyRequestAllowance::new(
            "google-free",
            switchyard_protocol::ModelId::new("google/gemini-3.8-flash"),
            2,
            allowance_path(&directory, "google-free", "google/gemini-3.8-flash"),
        );
        let candidate = budgeted_candidate(Some(allowance), Some(minute_window(&governor, 5)));

        let _first = reserve_decision_call(&candidate, 10)
            .expect("within both budgets")
            .expect("a window was declared, so a ticket is returned");
        let (requests, _) = governor.used(&budget_key(), std::time::SystemTime::now());
        assert_eq!(requests, 1, "one physical call, one window reservation");
        assert_eq!(
            candidate
                .budget
                .as_ref()
                .unwrap()
                .allowance
                .as_ref()
                .unwrap()
                .remaining(),
            1,
            "one physical call, one daily unit"
        );

        let _second = reserve_decision_call(&candidate, 10).expect("the last unit");
        assert!(matches!(
            reserve_decision_call(&candidate, 10),
            Err(DecisionFailure::Serving {
                reason: "daily_allowance_exhausted",
                retryable: true,
            })
        ));
    }

    /// An unusable counter is a deployment fault, so it must not be retried elsewhere: a
    /// sibling backend configured against the same counter would fail identically.
    #[test]
    fn an_unusable_counter_is_not_retryable() {
        let directory = scratch("unusable");
        let allowance = DailyRequestAllowance::new(
            "google-free",
            switchyard_protocol::ModelId::new("google/gemini-3.8-flash"),
            20,
            allowance_path(&directory, "google-free", "google/gemini-3.8-flash"),
        );
        // A directory where the counter file should be: the open cannot succeed.
        std::fs::create_dir_all(allowance.path()).expect("plant a directory in the file's place");
        let candidate = budgeted_candidate(Some(allowance), None);
        assert!(matches!(
            reserve_decision_call(&candidate, 10),
            Err(DecisionFailure::Serving {
                reason: "daily_allowance_unavailable",
                retryable: false,
            })
        ));
    }

    /// The two budgets are advisory to nobody: an unlimited candidate reserves nothing.
    #[test]
    fn an_unbudgeted_candidate_reserves_nothing() {
        let candidate = candidate(
            "laya",
            DecisionTransport::SystemOne,
            SupportedTypes::ALL,
            StateForms::ALL,
        );
        assert!(candidate.budget.is_none());
    }

    // --- classifying a lane where nothing answered --------------------------------

    fn served(name: &str, reason: &'static str, retryable: bool) -> (String, DecisionFailure) {
        (
            name.to_string(),
            DecisionFailure::Serving { reason, retryable },
        )
    }

    fn skipped(name: &str, skip: DecisionSkip) -> (String, DecisionFailure) {
        (name.to_string(), DecisionFailure::NotEligible(skip))
    }

    /// Nothing sent, every candidate skipped: the answer is about the request, is terminal,
    /// and names the most specific skip rather than the first one encountered.
    #[test]
    fn a_lane_that_dispatched_nothing_is_refused_not_exhausted() {
        let attempts = vec![
            skipped("a", DecisionSkip::UnsupportedType),
            skipped("b", DecisionSkip::UnsupportedState),
        ];
        let failure = aggregate_attempts(&attempts, "all_candidates_exhausted");
        assert_eq!(
            failure,
            DecisionFailure::Refused(DecisionSkip::UnsupportedState)
        );
        assert!(
            !failure.try_next(),
            "a sibling lane would answer identically"
        );
    }

    /// One backend was sent to and failed: that is an availability answer, and a parent may
    /// try its next child.
    #[test]
    fn a_lane_that_sent_and_failed_is_exhausted() {
        let attempts = vec![served("a", "connect", true)];
        assert_eq!(
            aggregate_attempts(&attempts, "all_candidates_exhausted"),
            DecisionFailure::Serving {
                reason: "all_candidates_exhausted",
                retryable: true,
            }
        );
        assert_eq!(
            aggregate_attempts(&attempts, "all_child_lanes_exhausted"),
            DecisionFailure::Serving {
                reason: "all_child_lanes_exhausted",
                retryable: true,
            },
            "a tier labels its own exhaustion, so an operator can tell which level ran out"
        );
    }

    /// A budget refusal is neither an outage nor a request defect: it keeps its own reason,
    /// and the most severe of several budget refusals is the one reported.
    #[test]
    fn a_budget_refusal_keeps_its_own_reason() {
        let attempts = vec![
            served("a", "minute_allowance_throttled", true),
            served("b", "daily_allowance_exhausted", true),
            skipped("c", DecisionSkip::UnsupportedType),
        ];
        assert_eq!(
            aggregate_attempts(&attempts, "all_candidates_exhausted"),
            DecisionFailure::Serving {
                reason: "daily_allowance_exhausted",
                retryable: true,
            },
            "a spent day outranks a minute window that refills shortly"
        );
        let unusable = vec![
            served("a", "daily_allowance_unavailable", false),
            served("b", "daily_allowance_exhausted", true),
        ];
        assert_eq!(
            aggregate_attempts(&unusable, "all_candidates_exhausted"),
            DecisionFailure::Serving {
                reason: "daily_allowance_unavailable",
                retryable: false,
            }
        );
    }

    /// An empty attempt list cannot claim a cause, so it keeps the old terminal answer
    /// rather than naming a skip nobody observed.
    #[test]
    fn no_attempts_at_all_is_not_a_skip() {
        assert_eq!(
            aggregate_attempts(&[], "all_candidates_exhausted"),
            DecisionFailure::Serving {
                reason: "no_eligible_backend",
                retryable: false,
            }
        );
    }

    /// Every refusal has words a caller can act on, and none of them is an enum name.
    #[test]
    fn refusals_explain_themselves_without_enum_names() {
        for skip in [
            DecisionSkip::TargetNotFound,
            DecisionSkip::UnsupportedType,
            DecisionSkip::UnsupportedState,
            DecisionSkip::Contract,
            DecisionSkip::Unauthenticated,
        ] {
            let message = DecisionFailure::Refused(skip).message();
            assert!(!message.is_empty());
            assert!(
                !message.contains("Unsupported") && !message.contains("NotEligible"),
                "a caller must not be handed our enum name: {message}"
            );
        }
        assert!(
            DecisionFailure::Serving {
                reason: "daily_allowance_exhausted",
                retryable: true
            }
            .message()
            .contains("resets"),
            "a quota refusal must say when it recovers"
        );
    }
}
