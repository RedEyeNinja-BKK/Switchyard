// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One configured algorithm and the clients that serve its targets.

use std::collections::HashMap;
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
use crate::config::RouteReasoningPolicy;

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
        }
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

    /// The provider slug of the first target model id, when one exists.
    pub fn first_target_slug(&self) -> &str {
        self.first_target_model()
            .map(|model| model.as_str().to_owned())
            .unwrap_or_default()
            .leak()
    }

    /// Whether every servable target on this route is a `:free` slug.
    ///
    /// A free-scope quota blocks the route only when this is true; a route with a paid
    /// candidate still has a servable leg on the same credential.
    pub fn all_targets_free(&self) -> bool {
        self.models
            .all_models()
            .iter()
            .all(|model| model.as_str().ends_with(":free"))
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
            return None;
        }
        binding
    }

    pub async fn execute(
        &self,
        request: Request,
        observer: Option<RunObserver>,
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
        let verdict = |model: &switchyard_protocol::ModelId| -> Result<(), &'static str> {
            let Some(facts) = facts.get(model) else {
                // Uncharacterised rather than incapable: permitted, except for video,
                // which no target may receive without stating that it can read it.
                return if crate::request_fit::requires_video(&request.llm_request, &raw) {
                    Err("video_requires_declaration")
                } else {
                    Ok(())
                };
            };
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
        let considered = self.models.all_models();
        let verdicts: Vec<Result<(), &'static str>> = considered.iter().map(&verdict).collect();
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
                        crate::selection::Candidate {
                            tier: if facts.is_some_and(|facts| facts.free_tier) {
                                crate::selection::CostTier::Free
                            } else {
                                crate::selection::CostTier::Paid
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
                scores = selector.explain(&candidates, demand, &declared);
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
                    let free = score
                        .map(|score| score.tier == crate::selection::CostTier::Free)
                        .or_else(|| facts.map(|facts| facts.free_tier))
                        .unwrap_or(false);
                    CandidateTrace {
                        model: model.to_string(),
                        eligible: verdict.is_ok(),
                        exclusion: verdict.err(),
                        tier: if free { "free" } else { "paid" },
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
        };
        if models.is_empty() {
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
        let observer = Some(recording_observer(
            Arc::clone(&attempts),
            self.selecting_observer(observer),
        ));
        let result = switchyard_llm_client::run(
            Arc::clone(&self.algorithm),
            self.clients.clone(),
            request,
            Arc::new(models),
            observer,
        )
        .await;
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
