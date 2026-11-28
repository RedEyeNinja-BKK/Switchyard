// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Google's error envelopes, normalized into one condition and one recovery policy.
//!
//! Google reports the same underlying condition in several shapes, and the shape decides
//! nothing while the condition decides everything. A 429 is a short-window throttle or a
//! spent daily quota; a 503 is capacity; a 400 can be an adapter defect, a content block or
//! a malformed tool call. Treating them uniformly is how a quota exhaustion becomes a retry
//! storm, and how a safety refusal becomes a provider outage that gets routed around.
//!
//! This module only *diagnoses*. What may be retried, cooled down, routed around or escalated
//! remains a Switchyard decision taken from [`GoogleRecovery`], so a provider-side hint can
//! never spend money or bypass the privacy gate on its own.
//!
//! Envelopes recognized, all measured or documented for Google's surfaces:
//!
//! * Interactions buffered — `{"error": {"code": "rate_limit_exceeded", …}}`.
//! * Interactions streaming — an SSE `data:` event carrying `event_type: "error"` after an
//!   HTTP 200. The event is a failure even though the status was a success.
//! * Native generative — `{"error": {"code": 503, "status": "UNAVAILABLE", …}}`, where the
//!   numeric code is an HTTP status and `status` is the gRPC-style reason.
//! * OpenAI-compatible — the OpenAI error object, sometimes wrapped in a one-element JSON
//!   list, which is what the hosted Gemma 4 31B 503 actually arrives as.
//!
//! Recognition requires positive Google evidence (an uppercase gRPC status, a
//! `google.rpc.` detail type, one of the documented snake_case codes, or the leaked-key
//! sentence). A body that merely has an `error` field is left to the existing provider
//! parsers, so OpenRouter's and DeepSeek's contracts are unchanged.

use std::time::Duration;

/// A content-policy reason Google names separately from any transport condition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockReason {
    /// The prompt or response was blocked by a safety filter.
    Safety,
    /// The output resembled protected source text.
    Recitation,
    /// The requested language is not supported by this model.
    Language,
    /// Content prohibited by policy.
    ProhibitedContent,
    /// Sensitive personally-identifying information was detected.
    Spii,
    /// A configured blocklist matched.
    Blocklist,
    /// An image block from an image-capable model, relevant to typed contracts only.
    Image,
}

/// How much of the credential a spent quota covers.
///
/// Google's Free tier allocates RPD **per model** (20/day for a Flash, 500/day for a
/// Flash-Lite) alongside separate feature quotas, so a bare `quota_exceeded` proves a quota
/// is spent without proving which one. Treating every such answer as project-wide would let
/// one exhausted model disable the rest of the pool on evidence that never said so.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GoogleQuotaScope {
    /// One exact model: only its own quota identity is spent.
    Model,
    /// The whole project or credential: every candidate behind it shares the quota.
    Project,
    /// One feature or tool quota, separate from the model's own request budget.
    Feature,
    /// Google did not say. Never promoted to `Project` - an unproven scope is not a fact.
    #[default]
    Unknown,
}

/// One condition Google reports, whichever transport carried it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoogleCondition {
    /// Request or token rate for the current window, which clears without intervention.
    ShortWindowThrottle,
    /// The daily quota for this exact scope is spent and resets on the provider's clock.
    DailyQuota,
    /// A **spend-based** rolling ten-minute limit, which is a paid-tier meter.
    ///
    /// Kept apart from [`Self::ShortWindowThrottle`] on purpose. Google publishes spend
    /// limits as Tier 1 $10 / Tier 2 $200 / Tier 3 $200 per ten minutes and states them as
    /// **Free: N/A**, so requests rejected on this condition cannot be a free-tier minute
    /// window and must not be answered with a "clears in a minute" or a same-model retry.
    /// It is only reached from structured evidence (a quota id or metric naming spend,
    /// billing or credits); a bare `429 RESOURCE_EXHAUSTED` stays a short-window throttle,
    /// because inferring a spend meter from the status alone would invent a paid-tier
    /// condition the project may not have.
    SpendLimit,
    /// Capacity: the model or service is briefly unable to serve.
    Capacity,
    /// The exact model id is not served — retired, renamed, or never existed.
    ModelNotFound,
    /// Some other resource (file, asset, session, job) is missing. Not a model failure.
    ResourceNotFound,
    /// The credential was rejected.
    Unauthenticated,
    /// The credential is valid but not permitted for this project, model or asset.
    PermissionDenied,
    /// A malformed or unsupported request: adapter defect or unavailable feature.
    InvalidRequest,
    /// The method or feature is not implemented on this surface.
    UnsupportedOperation,
    /// An idempotency or resource conflict.
    Conflict,
    /// A concurrency conflict Google says to retry at a higher level.
    Aborted,
    /// The caller cancelled; nothing further should be spent.
    ClientCancelled,
    /// A provider-side fault.
    ProviderFault,
    /// The provider's own deadline expired.
    ProviderDeadline,
    /// A content or policy block. Never an outage.
    ContentBlocked(BlockReason),
    /// A generation-structural failure around tool calls or signatures.
    MalformedTool,
    /// The response is structurally incomplete, e.g. an image was requested and not produced.
    IncompleteOutput,
    /// Google reported the credential as publicly exposed and blocked it.
    LeakedCredential,
    /// Recognized as Google, but the subtype is not one this module maps.
    Unknown,
}

/// A decoded Google error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GoogleError {
    /// What happened, independent of how it arrived.
    pub condition: GoogleCondition,
    /// The HTTP status, when the transport had one. A mid-stream SSE error has none.
    pub http: Option<u16>,
    /// The provider's own code, kept for diagnostics: `"rate_limit_exceeded"`, `"503"`,
    /// or a gRPC-style `status`.
    pub code: Option<String>,
    /// How much of the credential a spent quota covers, as far as the body proves.
    ///
    /// Only meaningful for [`GoogleCondition::DailyQuota`]; every other condition leaves it
    /// `Unknown`, because nothing else is scoped.
    pub quota_scope: GoogleQuotaScope,
}

/// What Switchyard may do with a decoded Google error.
///
/// Every field is a permission, not an instruction: `false` means the corresponding action
/// is refused and the existing route, privacy and cost paths decide the rest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoogleRecovery {
    /// Resend the identical request to the same model after a bounded, jittered backoff.
    pub retry_same_model: bool,
    /// Put this exact model into a short cooldown; another eligible model may serve.
    pub cool_down_model: bool,
    /// Do not offer the remaining candidates on this credential at all.
    pub stop_candidate_list: bool,
    /// A credential security incident: stop sending on this key and tell the operator.
    pub credential_incident: bool,
    /// The outcome is about the request or its policy, so routing it to another provider
    /// would be an attempt to evade the answer rather than to serve the caller.
    pub keep_outcome: bool,
}

impl GoogleRecovery {
    /// A recovery that permits nothing and blocks nothing.
    const INERT: Self = Self {
        retry_same_model: false,
        cool_down_model: false,
        stop_candidate_list: false,
        credential_incident: false,
        keep_outcome: false,
    };
}

impl GoogleError {
    /// The actions this error permits.
    pub fn recovery(&self) -> GoogleRecovery {
        use GoogleCondition as C;

        let condition = match self.condition {
            // Unknown is decided by the status: an unrecognised 429, 408 or 5xx stays a
            // bounded retry rather than being promoted to a daily quota we cannot prove,
            // while an unrecognised client error is not resent.
            C::Unknown => {
                return match self.http {
                    Some(408 | 429) | Some(500..=599) => GoogleRecovery {
                        retry_same_model: true,
                        ..GoogleRecovery::INERT
                    },
                    _ => GoogleRecovery::INERT,
                };
            }
            other => other,
        };

        match condition {
            C::ShortWindowThrottle => GoogleRecovery {
                retry_same_model: true,
                ..GoogleRecovery::INERT
            },
            // A paid spend meter does not clear on a backoff and does not promise a minute:
            // its window is ten minutes and it may not apply to this project at all. Cool the
            // model so the route can use a sibling, and never resend the same bytes into a
            // meter that already refused them. The candidate list is deliberately NOT
            // stopped: a spend cap on a paid tier says nothing about whether a free leg on
            // the same project still serves.
            C::SpendLimit => GoogleRecovery {
                cool_down_model: true,
                ..GoogleRecovery::INERT
            },
            // A spent daily quota does not clear on a backoff. Its SCOPE decides how much of
            // the pool it takes with it, and only a scope Google positively identified as
            // project- or credential-wide may stop the candidate list. A per-model quota, or
            // one whose scope the body never named, retires that exact model and leaves the
            // rest of the Free pool available - which is what the owner's per-model RPD
            // allocation actually is.
            C::DailyQuota => GoogleRecovery {
                cool_down_model: true,
                stop_candidate_list: self.quota_scope == GoogleQuotaScope::Project,
                ..GoogleRecovery::INERT
            },
            // Capacity is a fact about one model, and Google's own 31B behaviour shows how
            // quickly it recovers elsewhere: cool this model, cross to a peer, do not storm.
            C::Capacity => GoogleRecovery {
                cool_down_model: true,
                ..GoogleRecovery::INERT
            },
            // The slug is not served, so no request can change the answer; another
            // qualified model may serve it.
            C::ModelNotFound => GoogleRecovery {
                cool_down_model: true,
                ..GoogleRecovery::INERT
            },
            C::ResourceNotFound | C::Conflict => GoogleRecovery::INERT,
            // Retrying or hopping models on one rejected credential only multiplies the
            // rejection, so the credential scope stops here.
            C::Unauthenticated => GoogleRecovery {
                stop_candidate_list: true,
                ..GoogleRecovery::INERT
            },
            C::PermissionDenied => GoogleRecovery {
                stop_candidate_list: true,
                keep_outcome: true,
                ..GoogleRecovery::INERT
            },
            // An adapter defect: the bytes must change before anything is resent.
            C::InvalidRequest | C::UnsupportedOperation => GoogleRecovery {
                keep_outcome: true,
                ..GoogleRecovery::INERT
            },
            // Idempotency conflicts are retried only where the request is known idempotent,
            // which the attempt layer already establishes before it retries anything.
            C::Aborted => GoogleRecovery {
                retry_same_model: true,
                ..GoogleRecovery::INERT
            },
            // The caller withdrew. Another model would be spending on abandoned work.
            C::ClientCancelled => GoogleRecovery {
                keep_outcome: true,
                ..GoogleRecovery::INERT
            },
            C::ProviderFault | C::ProviderDeadline => GoogleRecovery {
                retry_same_model: true,
                ..GoogleRecovery::INERT
            },
            // A block is an answer, not a failure of the provider that produced it.
            C::ContentBlocked(_) => GoogleRecovery {
                keep_outcome: true,
                ..GoogleRecovery::INERT
            },
            // Tool shape and signatures are a protocol question for this model.
            C::MalformedTool | C::IncompleteOutput => GoogleRecovery {
                keep_outcome: true,
                ..GoogleRecovery::INERT
            },
            C::LeakedCredential => GoogleRecovery {
                stop_candidate_list: true,
                credential_incident: true,
                keep_outcome: true,
                ..GoogleRecovery::INERT
            },
            C::Unknown => GoogleRecovery::INERT,
        }
    }
}

/// Decodes a Google error, or `None` when the body carries no Google evidence.
///
/// `http` is the status the transport reported; pass `None` for an in-band event such as a
/// streaming `event_type: "error"`, which arrives after a successful HTTP 200.
pub fn parse(http: Option<u16>, body: &str) -> Option<GoogleError> {
    let error = error_object(body)?;
    let status = error.get("status").and_then(serde_json::Value::as_str);
    let code = error.get("code");
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();

    // A blocked key is an operator incident reported in prose, and the sentence is the only
    // stable marker Google gives it. It arrives with a 403, which on its own would read as
    // an ordinary permission denial.
    if message.to_ascii_lowercase().contains("reported as leaked") {
        return Some(GoogleError {
            condition: GoogleCondition::LeakedCredential,
            http,
            code: status.map(str::to_string),
            quota_scope: GoogleQuotaScope::Unknown,
        });
    }

    if !has_google_evidence(&error, status, code) {
        return None;
    }

    let condition = code
        .and_then(serde_json::Value::as_str)
        .and_then(interactions_condition)
        .or_else(|| status.and_then(grpc_condition))
        .or_else(|| http.and_then(http_condition))
        .unwrap_or(GoogleCondition::Unknown);

    // The status strings are coarser than the details. `RESOURCE_EXHAUSTED` and a bare 429
    // both read as a window that clears, which is why they stay a throttle on their own; but
    // when the body's own quota identifier says `PerDay`, the allowance really is spent until
    // midnight and retrying it is just a slower failure. Only the details may widen the
    // condition, never the status.
    // Structured evidence widens a coarse status; it never narrows one. A daily quota id
    // makes a bare throttle a spent day; a spend/billing id makes it a paid meter. Both are
    // read only from identifiers Google itself supplied, so the status alone can never
    // manufacture either condition.
    let condition = if names_a_daily_quota(&error)
        && matches!(
            condition,
            GoogleCondition::ShortWindowThrottle | GoogleCondition::Unknown
        ) {
        GoogleCondition::DailyQuota
    } else if names_a_spend_limit(&error)
        && matches!(
            condition,
            GoogleCondition::ShortWindowThrottle | GoogleCondition::Unknown
        )
    {
        GoogleCondition::SpendLimit
    } else {
        condition
    };

    // Scope is only read where it can change a decision, and only from structure Google
    // itself supplied. A body that names no scope stays `Unknown`, which retires one model
    // rather than the pool.
    let quota_scope = if condition == GoogleCondition::DailyQuota {
        read_quota_scope(&error)
    } else {
        GoogleQuotaScope::Unknown
    };

    Some(GoogleError {
        condition,
        http,
        code: error_code(code).or_else(|| status.map(str::to_string)),
        quota_scope,
    })
}

/// Reads how much of the credential a `quota_exceeded` covers from Google's structured
/// details.
///
/// The only trustworthy signals are explicit ones. A `quotaDimensions.model` names one
/// model; a `quotaId` that says `PerProjectPerModel` names one model inside one project; a
/// `quotaId`/`quotaMetric` that says `PerProject` without `PerModel` names the project. A
/// feature-named quota is a feature. Anything else is unknown, and unknown is treated as
/// narrow - never as project-wide - because blocking the pool on a guess is the failure this
/// scope exists to prevent.
fn read_quota_scope(error: &serde_json::Value) -> GoogleQuotaScope {
    let Some(details) = error.get("details").and_then(serde_json::Value::as_array) else {
        return GoogleQuotaScope::Unknown;
    };
    let mut saw_project = false;
    let mut saw_feature = false;
    for detail in details {
        // ErrorInfo carries its facts in `metadata`; QuotaFailure in `violations`.
        if let Some(metadata) = detail.get("metadata") {
            if metadata
                .get("model")
                .and_then(serde_json::Value::as_str)
                .is_some()
            {
                return GoogleQuotaScope::Model;
            }
            if let Some(limit) = metadata
                .get("quota_limit")
                .and_then(serde_json::Value::as_str)
            {
                match classify_quota_id(limit) {
                    Some(GoogleQuotaScope::Model) => return GoogleQuotaScope::Model,
                    Some(GoogleQuotaScope::Project) => saw_project = true,
                    Some(GoogleQuotaScope::Feature) => saw_feature = true,
                    _ => {}
                }
            }
        }
        let Some(violations) = detail
            .get("violations")
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        for violation in violations {
            if violation
                .get("quotaDimensions")
                .and_then(|dimensions| dimensions.get("model"))
                .and_then(serde_json::Value::as_str)
                .is_some()
            {
                return GoogleQuotaScope::Model;
            }
            for key in ["quotaId", "quotaMetric"] {
                let Some(text) = violation.get(key).and_then(serde_json::Value::as_str) else {
                    continue;
                };
                match classify_quota_id(text) {
                    Some(GoogleQuotaScope::Model) => return GoogleQuotaScope::Model,
                    Some(GoogleQuotaScope::Project) => saw_project = true,
                    Some(GoogleQuotaScope::Feature) => saw_feature = true,
                    _ => {}
                }
            }
        }
    }
    if saw_project {
        GoogleQuotaScope::Project
    } else if saw_feature {
        GoogleQuotaScope::Feature
    } else {
        GoogleQuotaScope::Unknown
    }
}

/// Reads a quota identifier's spelling, which is how Google scopes it.
/// Whether the body's own quota identifier names a per-day allowance.
///
/// This is the one signal that separates "spent until midnight" from "clears in a minute"
/// when the transport only said `RESOURCE_EXHAUSTED` or 429.
fn names_a_daily_quota(error: &serde_json::Value) -> bool {
    let Some(details) = error.get("details").and_then(serde_json::Value::as_array) else {
        return false;
    };
    details.iter().any(|detail| {
        let in_metadata = detail
            .get("metadata")
            .and_then(|metadata| metadata.get("quota_limit"))
            .and_then(serde_json::Value::as_str);
        let in_violations = detail
            .get("violations")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|violation| {
                ["quotaId", "quotaMetric"]
                    .iter()
                    .find_map(|key| violation.get(key).and_then(serde_json::Value::as_str))
            });
        in_metadata.into_iter().chain(in_violations).any(|text| {
            let lowered = text.to_ascii_lowercase();
            lowered.contains("perday") || lowered.contains("per_day")
        })
    })
}

/// Whether the body's own quota identifiers name a **spend-based** limit.
///
/// Paid tiers meter spend over a rolling ten minutes; the free tier does not. The only
/// evidence that proves a spend meter is an identifier Google wrote: a quota id, quota
/// metric, or the namespaced reason naming spend, billing or credits. Message prose is not
/// used, because a provider sentence can mention cost while reporting a token window.
fn names_a_spend_limit(error: &serde_json::Value) -> bool {
    let in_metadata = error
        .get("metadata")
        .and_then(|metadata| match metadata {
            serde_json::Value::Object(fields) => Some(fields.values().cloned().collect::<Vec<_>>()),
            _ => None,
        })
        .unwrap_or_default();
    let in_details = error
        .get("details")
        .and_then(serde_json::Value::as_array)
        .map(|details| details.to_vec())
        .unwrap_or_default();
    let mut texts: Vec<String> = Vec::new();
    for value in in_metadata {
        if let Some(text) = value.as_str() {
            texts.push(text.to_string());
        }
    }
    for detail in in_details {
        for key in ["reason", "domain", "quotaId", "quotaMetric"] {
            if let Some(text) = detail.get(key).and_then(serde_json::Value::as_str) {
                texts.push(text.to_string());
            }
        }
        if let Some(object) = detail
            .get("metadata")
            .and_then(serde_json::Value::as_object)
        {
            texts.extend(
                object
                    .values()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string),
            );
        }
    }
    texts.iter().any(|text| {
        let lowered = text.to_ascii_lowercase();
        lowered.contains("spend")
            || lowered.contains("billing")
            || lowered.contains("credits")
            || lowered.contains("quota_cost")
    })
}

fn classify_quota_id(text: &str) -> Option<GoogleQuotaScope> {
    let lowered = text.to_ascii_lowercase();
    // Narrowest fact first, because the identifiers nest: `PerProjectPerModel` names a model
    // inside a project, and `PerProjectPerFeature` names a feature inside one. Reading the
    // wider word first would widen every scope it appears in.
    if lowered.contains("permodel") || lowered.contains("per_model") {
        return Some(GoogleQuotaScope::Model);
    }
    if lowered.contains("feature") {
        return Some(GoogleQuotaScope::Feature);
    }
    if lowered.contains("perproject") || lowered.contains("per_project") {
        return Some(GoogleQuotaScope::Project);
    }
    None
}

/// The `error` object from any recognized envelope.
fn error_object(body: &str) -> Option<serde_json::Value> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    // An SSE frame: take the first payload line that carries an error. A provider may send
    // several events, and only one is the failure.
    if trimmed.starts_with("data:") || trimmed.contains("\ndata:") {
        for line in trimmed.lines() {
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            if let Some(error) = error_in_json(payload.trim()) {
                return Some(error);
            }
        }
        return None;
    }
    error_in_json(trimmed)
}

/// Extracts the `error` object from a JSON document or a list-wrapped one.
fn error_in_json(text: &str) -> Option<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    match value {
        serde_json::Value::Object(mut object) => {
            // Interactions streaming: the failure is the event, and `event_type` says so.
            if object.get("event_type").and_then(serde_json::Value::as_str) == Some("error") {
                return object.remove("error").map(into_object);
            }
            object.remove("error").map(into_object)
        }
        // Measured on hosted Gemma 4 31B: the error object arrives inside a one-element list.
        serde_json::Value::Array(values) => values.into_iter().find_map(|value| match value {
            serde_json::Value::Object(mut object) => object.remove("error").map(into_object),
            _ => None,
        }),
        _ => None,
    }
}

/// Keeps the error payload as an object; a string error is kept under `message`.
fn into_object(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(_) => value,
        serde_json::Value::String(message) => serde_json::json!({ "message": message }),
        other => serde_json::json!({ "message": other.to_string() }),
    }
}

/// gRPC-style statuses Google's native surfaces use, in their uppercase spelling.
const GRPC_STATUSES: &[&str] = &[
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
];

/// Positive evidence that this body came from Google rather than another provider.
fn has_google_evidence(
    error: &serde_json::Value,
    status: Option<&str>,
    code: Option<&serde_json::Value>,
) -> bool {
    if status.is_some_and(|status| GRPC_STATUSES.contains(&status)) {
        return true;
    }
    if code
        .and_then(serde_json::Value::as_str)
        .is_some_and(|code| interactions_condition(code).is_some())
    {
        return true;
    }
    // ErrorInfo / QuotaFailure detail types are namespaced and cannot be another provider's.
    error
        .get("details")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|details| {
            details.iter().any(|detail| {
                detail
                    .get("@type")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|kind| {
                        kind.contains("google.rpc.") || kind.contains("googleapis.com")
                    })
            })
        })
}

/// The provider's own code, whichever spelling the transport used.
fn error_code(code: Option<&serde_json::Value>) -> Option<String> {
    match code? {
        serde_json::Value::String(value) => Some(value.clone()),
        serde_json::Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

/// Interactions API snake_case codes, which name a condition exactly.
fn interactions_condition(code: &str) -> Option<GoogleCondition> {
    use BlockReason as B;
    use GoogleCondition as C;
    Some(match code {
        "rate_limit_exceeded" => C::ShortWindowThrottle,
        "quota_exceeded" => C::DailyQuota,
        "invalid_request" | "parameter_unknown" | "failed_precondition" | "out_of_range" => {
            C::InvalidRequest
        }
        "authentication" => C::Unauthenticated,
        "permission_denied" => C::PermissionDenied,
        "model_not_found" => C::ModelNotFound,
        "not_found" => C::ResourceNotFound,
        "already_exists" => C::Conflict,
        "aborted" => C::Aborted,
        "cancelled" => C::ClientCancelled,
        "api_error" => C::ProviderFault,
        "unimplemented" => C::UnsupportedOperation,
        "service_unavailable" => C::Capacity,
        "deadline_exceeded" => C::ProviderDeadline,
        "safety" => C::ContentBlocked(B::Safety),
        "recitation" => C::ContentBlocked(B::Recitation),
        "language" => C::ContentBlocked(B::Language),
        "prohibited_content" => C::ContentBlocked(B::ProhibitedContent),
        "spii" => C::ContentBlocked(B::Spii),
        "blocklist" => C::ContentBlocked(B::Blocklist),
        "content_blocked"
        | "image_safety"
        | "image_prohibited_content"
        | "image_recitation"
        | "image_other" => C::ContentBlocked(B::Image),
        "malformed_function_call"
        | "malformed_tool_call"
        | "unexpected_tool_call"
        | "too_many_tool_calls"
        | "missing_thought_signature" => C::MalformedTool,
        "no_image" => C::IncompleteOutput,
        _ => return None,
    })
}

/// gRPC-style statuses, for the native surface's uppercase spelling.
fn grpc_condition(status: &str) -> Option<GoogleCondition> {
    use GoogleCondition as C;
    Some(match status {
        "CANCELLED" => C::ClientCancelled,
        "UNKNOWN" => C::Unknown,
        "INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE" => C::InvalidRequest,
        "DEADLINE_EXCEEDED" => C::ProviderDeadline,
        "NOT_FOUND" => C::ResourceNotFound,
        "ALREADY_EXISTS" => C::Conflict,
        "PERMISSION_DENIED" => C::PermissionDenied,
        // Deliberately conservative: RESOURCE_EXHAUSTED alone does not prove a *daily*
        // quota is spent, and inferring one would stop every candidate until midnight.
        "RESOURCE_EXHAUSTED" => C::ShortWindowThrottle,
        "ABORTED" => C::Aborted,
        "UNIMPLEMENTED" => C::UnsupportedOperation,
        "INTERNAL" | "DATA_LOSS" => C::ProviderFault,
        "UNAVAILABLE" => C::Capacity,
        "UNAUTHENTICATED" => C::Unauthenticated,
        _ => return None,
    })
}

/// The status, when the body itself names no finer condition.
fn http_condition(http: u16) -> Option<GoogleCondition> {
    use GoogleCondition as C;
    Some(match http {
        400 | 416 => C::InvalidRequest,
        401 => C::Unauthenticated,
        403 => C::PermissionDenied,
        404 => C::ResourceNotFound,
        409 => C::Conflict,
        429 => C::ShortWindowThrottle,
        499 => C::ClientCancelled,
        501 => C::UnsupportedOperation,
        503 => C::Capacity,
        504 => C::ProviderDeadline,
        408 | 500 | 502 => C::ProviderFault,
        _ => return None,
    })
}

/// Whether an answered error may be retried on the same model.
pub fn retry_same_model(http: Option<u16>, body: &str) -> Option<bool> {
    parse(http, body).map(|error| error.recovery().retry_same_model)
}

/// The retry delay a `Retry-After` value names, capped at a bounded ceiling.
///
/// Google recommends exponential backoff with jitter for transient conditions; it also sends
/// `Retry-After` on some. Honouring it is better than guessing, as long as a malformed or
/// absurd value cannot stall a request past its deadline.
pub fn capped_retry_after(value: Option<&str>, ceiling: Duration) -> Option<Duration> {
    let value = value?.trim();
    let seconds = value.parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds).min(ceiling))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn condition(http: Option<u16>, body: &str) -> GoogleCondition {
        parse(http, body)
            .unwrap_or_else(|| panic!("no Google error decoded from {body}"))
            .condition
    }

    #[test]
    fn an_interactions_throttle_is_retried_but_a_daily_quota_is_not() {
        // The two 429 codes Google documents are the whole point of the decoder: one
        // clears in seconds, the other only on the provider's clock.
        let throttle = parse(
            Some(429),
            r#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#,
        )
        .expect("decoded");
        assert_eq!(throttle.condition, GoogleCondition::ShortWindowThrottle);
        assert!(throttle.recovery().retry_same_model);
        assert!(!throttle.recovery().stop_candidate_list);

        let quota = parse(
            Some(429),
            r#"{"error":{"code":"quota_exceeded","message":"daily quota"}}"#,
        )
        .expect("decoded");
        assert_eq!(quota.condition, GoogleCondition::DailyQuota);
        assert!(!quota.recovery().retry_same_model);
        // No details, so no scope: retire this model, do not condemn the pool.
        assert_eq!(quota.quota_scope, GoogleQuotaScope::Unknown);
        assert!(!quota.recovery().stop_candidate_list);
        assert!(quota.recovery().cool_down_model);
    }

    #[test]
    fn a_quota_exhausted_answer_names_its_own_scope_or_none() {
        // The two shapes that differ. A per-model quota must not disable the pool; only a
        // positively project-wide one may, and this Free tier allocates RPD per model.
        let model_scoped = parse(
            Some(429),
            r#"{"error":{"code":"quota_exceeded","message":"daily","details":[
                {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[
                  {"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                   "quotaMetric":"generativelanguage.googleapis.com/generate_content_free_tier_requests",
                   "quotaDimensions":{"location":"global","model":"gemini-3.6-flash"}}]}]}}"#,
        )
        .expect("decoded");
        assert_eq!(model_scoped.quota_scope, GoogleQuotaScope::Model);
        assert!(!model_scoped.recovery().stop_candidate_list);

        let project_scoped = parse(
            Some(429),
            r#"{"error":{"code":"quota_exceeded","message":"daily","details":[
                {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[
                  {"quotaId":"GenerateRequestsPerDayPerProject-FreeTier",
                   "quotaMetric":"generativelanguage.googleapis.com/generate_content_free_tier_requests",
                   "quotaDimensions":{"location":"global"}}]}]}}"#,
        )
        .expect("decoded");
        assert_eq!(project_scoped.quota_scope, GoogleQuotaScope::Project);
        assert!(project_scoped.recovery().stop_candidate_list);

        // A feature quota is narrower than the model's own request budget.
        let feature = parse(
            Some(429),
            r#"{"error":{"code":"quota_exceeded","message":"daily","details":[
                {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[
                  {"quotaId":"GroundingRequestsPerDayPerProjectPerFeature"}]}]}}"#,
        )
        .expect("decoded");
        assert_eq!(feature.quota_scope, GoogleQuotaScope::Feature);
        assert!(!feature.recovery().stop_candidate_list);
    }

    #[test]
    fn an_error_info_model_dimension_is_a_model_scope() {
        // ErrorInfo carries the same fact in `metadata` rather than `violations`.
        let error = parse(
            Some(429),
            r#"{"error":{"code":"quota_exceeded","message":"daily","details":[
                {"@type":"type.googleapis.com/google.rpc.ErrorInfo",
                 "reason":"RATE_LIMIT_EXCEEDED","domain":"googleapis.com",
                 "metadata":{"model":"gemini-3.8-flash"}}]}}"#,
        )
        .expect("decoded");
        assert_eq!(error.quota_scope, GoogleQuotaScope::Model);
        assert!(!error.recovery().stop_candidate_list);
    }

    #[test]
    fn an_error_info_quota_limit_names_one_model_inside_one_project() {
        // The shape generativelanguage actually sends for a spent Free-tier RPD: the limit
        // id says `PerProjectPerModel`, so the project and the model are both named and the
        // narrower one decides. Reading this as project-wide is exactly the bug that let a
        // single exhausted Gemma condemn the whole Free pool.
        let error = parse(
            Some(429),
            r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED","details":[
                {"@type":"type.googleapis.com/google.rpc.ErrorInfo",
                 "reason":"RATE_LIMIT_EXCEEDED","domain":"googleapis.com",
                 "metadata":{"quota_limit":"GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                             "quota_limit_value":"20",
                             "quota_metric":"generativelanguage.googleapis.com/generate_content_free_tier_requests",
                             "service":"generativelanguage.googleapis.com"}}]}}"#,
        )
        .expect("decoded");
        assert_eq!(error.condition, GoogleCondition::DailyQuota);
        assert_eq!(error.quota_scope, GoogleQuotaScope::Model);
        assert!(!error.recovery().stop_candidate_list);
        assert!(error.recovery().cool_down_model);
    }

    #[test]
    fn a_native_status_wins_over_the_bare_http_code() {
        // `{"code":503,"status":"UNAVAILABLE"}` is the native shape: the numeric code is
        // the status, and the reasoned answer is the string. Capacity, not quota.
        let error = parse(
            Some(503),
            r#"{"error":{"code":503,"status":"UNAVAILABLE","message":"high demand"}}"#,
        )
        .expect("decoded");
        assert_eq!(error.condition, GoogleCondition::Capacity);
        let recovery = error.recovery();
        assert!(recovery.cool_down_model);
        assert!(!recovery.retry_same_model);
        assert!(!recovery.stop_candidate_list);
    }

    #[test]
    fn a_list_wrapped_error_is_read_the_way_the_vendor_sends_it() {
        // Measured on hosted Gemma 4 31B: the error object arrives inside a list.
        let body = r#"[{"error":{"code":503,"status":"UNAVAILABLE","message":"high demand"}}]"#;
        assert_eq!(condition(Some(503), body), GoogleCondition::Capacity);
    }

    #[test]
    fn a_streaming_error_event_is_a_failure_even_after_http_200() {
        let body = "data: {\"event_type\":\"error\",\"error\":{\"code\":\"quota_exceeded\",\"message\":\"daily\"}}\n\n";
        let error = parse(None, body).expect("decoded");
        assert_eq!(error.condition, GoogleCondition::DailyQuota);
        // Nothing in the body scoped it, so it retires the model -- not the pool.
        assert!(!error.recovery().stop_candidate_list);
        assert!(!error.recovery().retry_same_model);
    }

    #[test]
    fn a_missing_model_and_a_missing_asset_are_not_the_same_fact() {
        let model = parse(
            Some(404),
            r#"{"error":{"code":"model_not_found","message":"gone"}}"#,
        )
        .expect("decoded");
        assert_eq!(model.condition, GoogleCondition::ModelNotFound);
        assert!(model.recovery().cool_down_model);

        let asset = parse(
            Some(404),
            r#"{"error":{"code":"not_found","message":"file abc missing"}}"#,
        )
        .expect("decoded");
        assert_eq!(asset.condition, GoogleCondition::ResourceNotFound);
        assert!(!asset.recovery().cool_down_model);
    }

    #[test]
    fn a_policy_block_is_an_answer_not_an_outage() {
        for (code, reason) in [
            ("safety", BlockReason::Safety),
            ("recitation", BlockReason::Recitation),
            ("spii", BlockReason::Spii),
            ("prohibited_content", BlockReason::ProhibitedContent),
        ] {
            let body = format!(r#"{{"error":{{"code":"{code}","message":"blocked"}}}}"#);
            let error = parse(Some(400), &body).expect("decoded");
            assert_eq!(error.condition, GoogleCondition::ContentBlocked(reason));
            let recovery = error.recovery();
            assert!(!recovery.retry_same_model, "{code} must not be resent");
            assert!(recovery.keep_outcome, "{code} must not be routed around");
        }
    }

    #[test]
    fn a_malformed_tool_call_is_structural_not_infrastructure() {
        let error = parse(
            Some(400),
            r#"{"error":{"code":"malformed_function_call","message":"bad args"}}"#,
        )
        .expect("decoded");
        assert_eq!(error.condition, GoogleCondition::MalformedTool);
        assert!(!error.recovery().retry_same_model);
        assert!(error.recovery().keep_outcome);
    }

    #[test]
    fn a_reported_leaked_key_is_a_credential_incident() {
        let error = parse(
            Some(403),
            r#"{"error":{"code":403,"status":"PERMISSION_DENIED","message":"Your API key was reported as leaked. Please use another API key."}}"#,
        )
        .expect("decoded");
        assert_eq!(error.condition, GoogleCondition::LeakedCredential);
        let recovery = error.recovery();
        assert!(recovery.credential_incident);
        assert!(recovery.stop_candidate_list);
        assert!(!recovery.retry_same_model);
    }

    #[test]
    fn an_unrecognised_subtype_falls_back_to_the_status_and_stays_conservative() {
        // Google evidence (a namespaced detail) with a code we do not map. The status then
        // attributes the condition: a 429 may be retried briefly and is never promoted to a
        // daily quota we cannot prove; a 400 is not resent at all.
        let body = r#"{"error":{"code":"something_new","message":"?","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"}]}}"#;

        let throttled = parse(Some(429), body).expect("decoded");
        assert_eq!(throttled.condition, GoogleCondition::ShortWindowThrottle);
        assert!(throttled.recovery().retry_same_model);
        assert!(!throttled.recovery().stop_candidate_list);

        let bad = parse(Some(400), body).expect("decoded");
        assert_eq!(bad.condition, GoogleCondition::InvalidRequest);
        assert!(!bad.recovery().retry_same_model);

        // A status this module maps nothing to stays Unknown, and permits no retry.
        let odd = parse(Some(418), body).expect("decoded");
        assert_eq!(odd.condition, GoogleCondition::Unknown);
        assert!(!odd.recovery().retry_same_model);
        assert!(!odd.recovery().stop_candidate_list);
    }

    #[test]
    fn another_providers_error_shape_is_left_alone() {
        // OpenRouter and DeepSeek bodies carry no Google evidence, so the existing parsers
        // keep owning them. This is the regression guard for "preserve other providers".
        for body in [
            r#"{"error":{"code":402,"message":"Insufficient credits","metadata":{"limit_source":"openrouter_free_tier_daily"}}}"#,
            r#"{"error":{"type":"server_error","error_type":"authentication","message":"masked"}}"#,
            r#"{"error":{"message":"rate limited","code":429}}"#,
        ] {
            assert_eq!(parse(Some(429), body), None, "{body}");
        }
    }

    #[test]
    fn retry_after_is_honoured_within_a_ceiling() {
        let ceiling = Duration::from_secs(30);
        assert_eq!(
            capped_retry_after(Some("5"), ceiling),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            capped_retry_after(Some("900"), ceiling),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            capped_retry_after(Some("Wed, 21 Oct 2026 07:28:00 GMT"), ceiling),
            None
        );
        assert_eq!(capped_retry_after(None, ceiling), None);
    }

    #[test]
    fn a_plain_quota_failure_detail_reads_as_a_throttle_until_proven_daily() {
        // RESOURCE_EXHAUSTED is what Google sends for both window limits; refusing to
        // promote it to a daily quota is what keeps a project from being condemned on a
        // condition that clears in a minute.
        let error = parse(
            Some(429),
            r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota"}}"#,
        )
        .expect("decoded");
        assert_eq!(error.condition, GoogleCondition::ShortWindowThrottle);
        assert!(error.recovery().retry_same_model);
        assert!(!error.recovery().stop_candidate_list);
    }
}
