// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routed-call server trait and its shared error types.
//!
//! [`RoutedLlmClient`] is the one piece of I/O the protocol does not own: a host
//! implements it to actually perform a model call. It lives here — rather than in
//! libsy's orchestration crate — so a client crate that depends only on the protocol
//! can serve routed calls without pulling in the orchestrator.

use async_trait::async_trait;
use thiserror::Error;

use crate::{ModelId, Request, Response};

/// A boxed client-specific error preserved as the source of a routed call failure.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Failures a routed LLM client can surface to its caller.
///
/// The variants classify failures that routing hosts commonly need to handle,
/// while boxed sources preserve implementation-specific detail. `General` is the
/// escape hatch for failures that do not fit a shared category.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LlmClientError {
    /// The request cannot be served as supplied.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// Human-readable request validation failure.
        message: String,
    },

    /// Decoding the inbound request failed in the translation engine.
    #[error("request translation failed: {0}")]
    RequestTranslation(String),

    /// Encoding the request for the upstream failed in the translation engine.
    #[error("outbound request encoding failed: {0}")]
    RequestEncoding(String),

    /// Decoding or encoding the response failed in the translation engine.
    #[error("response translation failed: {0}")]
    ResponseTranslation(String),

    /// The client is not configured to serve the selected target.
    #[error("client configuration error: {message}")]
    Configuration {
        /// Human-readable configuration failure.
        message: String,
    },

    /// The route cannot record another provider-owned response or conversation ID.
    #[error(
        "Responses state tracking reached its limit of {limit} IDs; no existing records were removed"
    )]
    ResponseStateLimitExceeded {
        /// Maximum number of IDs retained by this route.
        limit: usize,
    },

    /// Two configured models reported the same saved response or conversation ID.
    #[error("Responses state ID is already recorded for another model; its owner was not changed")]
    ResponseStateConflict,

    /// The selected backend is temporarily unavailable. Another candidate may serve the request.
    #[error("backend temporarily unavailable")]
    TemporarilyUnavailable,

    /// The target's own daily request allowance is spent for the current provider day.
    ///
    /// Distinct from an account-wide cap: a per-project, per-model request budget (the
    /// Google free tier publishes one per model) leaves every other candidate on the
    /// deployment servable, so the request may advance to the next leg. It must not be
    /// retried — the allowance resets on the provider's day boundary, not in a moment.
    #[error(
        "daily request allowance for {model} is spent ({limit} per provider day); it resets at the next provider day"
    )]
    DailyRequestAllowanceExhausted {
        /// Target whose own allowance is spent.
        model: ModelId,
        /// The exhausted daily limit.
        limit: u64,
        /// The quota group whose single counter refused, when that counter is
        /// account-scoped.
        ///
        /// `Some` states a shared fact, not a per-model one: every target that draws on
        /// this group is refused by the same counter, so a route may skip its siblings
        /// instead of asking each of them and reading the same zero. `None` is a
        /// model-scoped budget and says nothing about any other model.
        group: Option<String>,
    },

    /// The target's daily request allowance is configured but its accounting state is
    /// unusable, so the request is refused rather than spent unaccounted.
    ///
    /// Fail-closed on purpose: sending anyway would overrun a hard daily cap whose whole
    /// point is to be enforced, and the deployment still has other candidates to serve
    /// the request.
    #[error("daily request allowance for {model} cannot be accounted: {reason}")]
    DailyRequestAllowanceUnavailable {
        /// Target whose accounting failed.
        model: ModelId,
        /// Why the state could not be read or written.
        reason: String,
    },

    /// The provider reported that the account's allowance is spent for the current day.
    ///
    /// Distinct from [`Self::DailyRequestAllowanceExhausted`], which is a statement about
    /// this deployment's own counter. This one is the provider's answer about the whole
    /// account: it counts traffic from every client and every key, so a local budget with
    /// room left cannot make the account servable, and the leg must not be retried. It
    /// also must not be confused with an unusable counter - a stale or unreadable reading
    /// is [`Self::DailyRequestAllowanceUnavailable`], never this.
    ///
    /// The route may advance on it, exactly as for a per-target exhaustion: every other
    /// candidate on a *different* allowance is untouched, which is the whole point of
    /// keeping the account's cap separate from each leg's.
    #[error(
        "provider reports the account allowance for {model} is spent ({limit} per provider day); it resets at the provider's own day boundary"
    )]
    ProviderAllowanceExhausted {
        /// Model whose account allowance is spent.
        model: ModelId,
        /// The account pool's stated daily limit.
        limit: u64,
        /// The quota group the provider's reading governs, when the counter it was
        /// applied to is account-scoped.
        ///
        /// This is the authoritative identity of the exhausted pool: the reading was
        /// reconciled onto one counter for the whole group, so every model in it is out,
        /// not merely the slug the request happened to reach. `None` means the reading
        /// governs one model's allowance and condemns nothing else.
        group: Option<String>,
    },

    /// The target's own per-minute budget has no room for this request yet.
    ///
    /// A different fact from both the daily allowance above and an account-wide cap: the
    /// window refills within a minute, so the leg is still servable and the route may advance
    /// to a candidate with minute headroom. It must not be retried on the same leg inside one
    /// call — that would spend the very window the next request needs, which is how a single
    /// throttle becomes a burst against a provider whose whole point is the minute.
    #[error(
        "per-minute {scope} budget for {model} is full; it clears in about {retry_after_secs}s"
    )]
    MinuteAllowanceThrottled {
        /// Target whose minute window is full.
        model: ModelId,
        /// Which limit refused: `requests_per_minute` or `input_tokens_per_minute`.
        scope: String,
        /// Seconds until the window has room for this request.
        retry_after_secs: u64,
    },

    /// This request can never fit the target's per-minute input budget.
    ///
    /// Distinct from a throttle: waiting cannot help, because the request alone, with the
    /// deployment's uncertainty margin applied, is larger than a whole minute's allowance.
    /// The leg is servable in general and a sibling with a wider window may take the
    /// request now, so the route advances rather than ending. It is not a capability defect
    /// either — this model could read the input if the deployment were not pacing it.
    #[error(
        "an input reserving about {needed} tokens at a {margin_permille}/1000 safety margin \
         exceeds the {limit}-token per-minute input budget for {model}; another candidate \
         with a wider window may serve this request"
    )]
    InputExceedsMinuteWindow {
        /// Target whose minute window the request cannot fit.
        model: ModelId,
        /// The request's reservation after the margin.
        needed: u64,
        /// The whole window's input budget.
        limit: u64,
        /// The margin applied, in permille.
        margin_permille: u32,
    },

    /// The provider answered the request but produced no usable reply.
    ///
    /// `HTTP 200` records that the request was accepted and answered; it does not record that
    /// the caller received anything. Measured 2026-11-21 against hosted Gemma 4 26B: a
    /// 900-token output budget went entirely to inline `<thought>` reasoning, so the reply
    /// carried `completion_tokens: 0` and no content while the wire reported success, and the
    /// caller was handed an empty answer although a sibling candidate could have served it.
    ///
    /// The route advances rather than retrying in place: the same model under the same budget
    /// produces the same non-answer, so a retry spends the window and returns nothing again.
    #[error(
        "{model} returned no usable answer ({reason}); another candidate may serve this request"
    )]
    EmptyAnswer {
        /// Model whose reply carried no content.
        model: ModelId,
        /// Which empty shape was observed.
        reason: &'static str,
    },

    /// The caller asked for a feature the selected target's wire format cannot carry.
    ///
    /// Raised before the request is assembled, so nothing is sent. Terminal for the route
    /// for the same reason a privacy finding is: every candidate on a Chat wire format
    /// would have to answer without the feature too, so advancing would hand the caller a
    /// plausible answer to a request that was never really made.
    #[error("request for {model} needs a feature this target cannot serve: {feature}")]
    UnsupportedFeature {
        /// Target the request would have been sent to.
        model: ModelId,
        /// The feature and why this target cannot serve it.
        feature: String,
    },

    /// The upstream could not be reached or the request could not be sent.
    #[error("upstream transport error: {source}")]
    Transport {
        /// Client-specific transport failure.
        #[source]
        source: BoxError,
    },

    /// The upstream request exceeded its timeout.
    ///
    /// Surfaced once the response is committed (a stream the caller already holds
    /// stopped producing) or when the request's overall lifetime elapsed. A caller
    /// must not route around this: output may already have been delivered.
    #[error("upstream request timed out: {source}")]
    Timeout {
        /// Client-specific timeout failure.
        #[source]
        source: BoxError,
    },

    /// One attempt elapsed its own bound before producing any usable output.
    ///
    /// Pre-commit by construction: the attempt never returned a response to the
    /// caller, so no output can be lost by trying another candidate. Distinct from
    /// [`LlmClientError::Timeout`], which never permits fallback.
    #[error("upstream attempt timed out before producing output: {source}")]
    AttemptTimeout {
        /// Client-specific timeout failure.
        #[source]
        source: BoxError,
    },

    /// The upstream rejected the request because it exceeds the model's context window.
    #[error("context window exceeded for model {model}: {message}")]
    ContextWindowExceeded {
        /// Model whose context window was exceeded.
        model: ModelId,
        /// Upstream error message.
        message: String,
    },

    /// The upstream returned a non-success HTTP response.
    #[error("upstream returned HTTP {status}: {body}")]
    UpstreamHttp {
        /// Upstream HTTP status code.
        status: http::StatusCode,
        /// Raw upstream error body.
        body: String,
    },

    /// The upstream returned a response the client could not decode.
    #[error("invalid upstream response: {source}")]
    InvalidResponse {
        /// Client-specific decoding or validation failure.
        #[source]
        source: BoxError,
    },

    /// A call across a foreign-function boundary (e.g. a Python-implemented client)
    /// failed. The boxed source is the foreign error itself.
    #[error("foreign function interface error: {source}")]
    Ffi {
        /// Foreign-language failure, preserved verbatim.
        #[source]
        source: BoxError,
    },

    /// A string message. Useful in testing, but prefer adding variants over using this.
    #[error("{0}")]
    General(String),
}

/// Why routing replaced a selected target with another eligible target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingFallbackReason {
    /// The selected target rejected the request because its context window was too small.
    ContextWindow,
    /// The selected target was unavailable after its client retries finished.
    Unavailable,
}

impl RoutingFallbackReason {
    /// Stable value used when logging a routing fallback.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextWindow => "context_window",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Performs the actual model call for a target. This is the one piece of I/O the
/// library does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). It serves a call the stream consumer chose not to
/// override, reached as a routed request's `default_client`.
///
/// # Concurrency
///
/// A client may be shared by many targets and concurrent algorithm runs. Calls may
/// overlap, so implementations must synchronize mutable state internally and should
/// not serialize requests unless their transport requires it.
#[async_trait]
pub trait RoutedLlmClient: Send + Sync {
    /// Make a request
    async fn call(&self, request: Request) -> Result<Response, LlmClientError>;
}
