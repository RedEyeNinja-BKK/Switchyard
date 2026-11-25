// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP LLM client that speaks Switchyard's neutral IR directly.
//!
//! [`TranslatingLlmClient`] maps a model name (and the wire format resolved from
//! the request) to a [`Backend`],
//! encodes a [`switchyard_protocol::Request`] to that backend's wire format via
//! `switchyard-translation`, applies auth and forwards caller headers, makes the
//! HTTP call with a shared [`reqwest::Client`], and decodes the wire response
//! back to a [`switchyard_protocol::Response`] — supporting both buffered and
//! streamed responses.
//!
//! [`run()`] pairs the client with a libsy algorithm: it drives
//! [`switchyard_libsy::Algorithm::run_stream`], serves routing-time calls, and makes the terminal
//! answer call from the routing outcome when needed. A host that just wants the answer does not
//! have to drive the step stream itself.

pub mod allowance;
pub mod backend;
pub mod client;
pub mod error;
pub mod google;
pub mod metrics;
mod observability;
mod observation;
pub mod live;
pub mod raw;
pub mod run;

pub use allowance::{
    ACCOUNT_SCOPE_MODEL, AdmissionMargin, DailyRequestAllowance, DayBoundary, GovernorKey,
    PacedQuota, PacingError, PacingScope, PacingTicket, ProviderReading, ProviderState,
    QuotaGovernor, ReserveError, Throttled, WindowLimits, account_allowance_path, allowance_path,
    estimate_input_tokens, reported_input_tokens,
};
pub use backend::{Backend, DEFAULT_MAX_RETRIES, HttpBackendConfig};
pub use client::{AuxiliaryOperation, ModelConfig, TranslatingLlmClient};
pub use error::{LlmClientError, Result};
pub use observation::{
    AttemptTrace, CandidateTrace, LlmCallObservation, RunObservation, RunObserver, SelectionTrace,
    library_outcome_code, outcome_code,
};
pub use raw::RawResponse;
pub use run::{
    CapabilityFacts, ClientRouter, CredentialBlock, CredentialBlockKind, FailureClass, Tariff,
    credential_block, decide, failure_class, is_free_scope, run,
};
pub use switchyard_translation::RawEventStream;

/// Registers process-wide compatibility gauges with the global meter provider.
pub fn initialize_metrics() {
    metrics::initialize();
}
