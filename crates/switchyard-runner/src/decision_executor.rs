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
}

/// A backend refused or failed to serve the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionFailure {
    /// Eligible backend that could not be reached or refused transiently.
    Serving {
        reason: &'static str,
        retryable: bool,
    },
    /// The request or configuration is at fault; another backend would not help.
    NotEligible(DecisionSkip),
}

impl DecisionFailure {
    /// Whether the resolver should try the next candidate.
    pub fn try_next(&self) -> bool {
        match self {
            DecisionFailure::Serving { retryable, .. } => *retryable,
            DecisionFailure::NotEligible(skip) => skip.try_next(),
        }
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

async fn send(candidate: &DecisionCandidate, body: Value) -> Result<Value, DecisionFailure> {
    let Some(api_key) = candidate
        .api_key
        .as_deref()
        .filter(|key| !key.trim().is_empty())
    else {
        return Err(DecisionFailure::NotEligible(DecisionSkip::Unauthenticated));
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_DECISION_TIMEOUT_SECONDS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| DecisionFailure::Serving {
            reason: "client_unavailable",
            retryable: false,
        })?;
    let response = client
        .post(&candidate.url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|error| {
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
    // Every candidate was tried and every one yielded. The lane itself is exhausted
    // rather than broken, so the outcome stays recoverable: a composed parent is
    // entitled to try its NEXT child lane after this one runs out of candidates.
    //
    // "Yielded" means a candidate was actually SENT to. A candidate skipped
    // pre-dispatch was never tried, so a lane where every candidate was skipped
    // attempted nothing and keeps the terminal no_eligible_backend: forwarding a
    // request that no candidate here can serve would just spread a request that is
    // unserviceable by contract.
    let exhausted = attempts
        .iter()
        .any(|(_, failure)| matches!(failure, DecisionFailure::Serving { .. }));
    Err(DecisionFailure::Serving {
        reason: if exhausted {
            "all_candidates_exhausted"
        } else {
            "no_eligible_backend"
        },
        retryable: exhausted,
    })
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
    let mut yielded = false;
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
                // Only a child that was actually SENT TO has yielded; a child that
                // skipped every candidate pre-dispatch was never tried.
                if matches!(failure, DecisionFailure::Serving { .. }) {
                    yielded = true;
                }
            }
        }
    }
    // Every child yielded after exhausting its own candidates. The composed lane is
    // itself exhausted, so this stays recoverable: an outer tier may still try its
    // own next child rather than treating a sibling's exhaustion as terminal.
    Err(DecisionFailure::Serving {
        reason: if yielded {
            "all_child_lanes_exhausted"
        } else {
            "no_eligible_backend"
        },
        retryable: yielded,
    })
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
            let children = composition
                .get(requested)
                .cloned()
                .unwrap_or_default();
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
        let handle = lanes.get(child.as_str()).ok_or(DecisionFailure::NotEligible(
            DecisionSkip::TargetNotFound,
        ))?;
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
}
