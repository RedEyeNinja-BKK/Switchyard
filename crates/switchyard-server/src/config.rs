// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility wrapper for loading shared runner configuration.

use std::path::Path;

use switchyard_runner::Runner;

use crate::fleet_readiness;
use crate::{ServerError, ServerResult, ServerRuntime, ServerState};

/// Loads a TOML deployment file and constructs the complete server state.
pub fn load_server_state(path: impl AsRef<Path>) -> ServerResult<ServerState> {
    ServerState::from_runner(Runner::load(path).map_err(ServerError::from)?)
}

/// Loads a coherent production runtime bundle from a config path: the server
/// state plus - when the config declares `[fleet_readiness]` - a
/// fleet-readiness monitor built over the SAME `Arc<SharedFleetState>` the
/// `fleet_router` routes read, so the runtime owner never wires the two
/// together manually.
pub fn load_server_runtime(path: impl AsRef<Path>) -> ServerResult<ServerRuntime> {
    let runner = Runner::load(path).map_err(ServerError::from)?;
    // Read the fleet surfaces before `from_runner` consumes the runner.
    let readiness_config = runner.fleet_readiness().cloned();
    let fleet_state = runner.fleet_state().cloned();
    let state = ServerState::from_runner(runner)?;
    // ONE shared sanitized DeepSeek telemetry slot: the fleet-readiness
    // resource client publishes the last-successful factual observation into
    // it, and the server endpoint serves it read-only. No second provider
    // fetch path is created.
    let deepseek_telemetry = fleet_readiness::SharedDeepSeekTelemetry::new();
    let state = state.with_deepseek_telemetry(deepseek_telemetry.clone());
    let monitor = match (readiness_config, fleet_state) {
        (Some(config), Some(fleet_state)) => Some(
            fleet_readiness::build_fleet_readiness_monitor(
                &config,
                fleet_state,
                deepseek_telemetry,
            )
            .map_err(ServerError::new)?,
        ),
        (Some(_), None) => {
            return Err(ServerError::new(
                "[fleet_readiness] declared but the deployment configures no fleet_router \
                 route; the monitor would have nothing to feed",
            ));
        }
        _ => None,
    };
    Ok(ServerRuntime { state, monitor })
}
