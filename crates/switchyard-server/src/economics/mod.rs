// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchyard-owned economics telemetry.
//!
//! Everything the economics dashboards consume is produced here: the provider, route
//! and cost classification derived from the deployment configuration, the provider
//! account observations collected with Switchyard's own credentials, and the metrics
//! that publish both.
//!
//! The boundary this module upholds is that Switchyard is the only collector. A
//! provider account API is read with Switchyard's credential authority, and the
//! observation is then exposed through Switchyard's own metric surface. Nothing
//! downstream of Switchyard talks to a provider.

pub mod account_apis;
pub mod calendar;
pub mod classification;
pub mod collector;
pub mod metrics;
pub mod providers;
pub mod runtime;

pub use calendar::CalendarWindow;
pub use classification::{CostClass, DeploymentClassification, ProviderFacts, RouteFamily};
pub use collector::AccountCollector;
