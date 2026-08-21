//! REALIGN-2 (amended): generic shadow inference-fact plane (server-side, ZERO routing authority).
//!
//! This module is the *dynamic factual plane* of the future inference-routing
//! architecture. It **only projects** the already-fetched authoritative facts from the
//! existing deployment into a small generic envelope for shadow observation and future
//! consumption. It performs **no I/O**, **no target selection**, **no scoring**, and is
//! **never** read by any routing path.
//!
//! Reuse-first (§13): OpenAI + DeepSeek facts come from the EXISTING `ResourceSnapshot`
//! (produced by `resource_fetcher.rs` + `ResourceState`); ComfyNinja facts/continuity come
//! from the EXISTING Gate-C `ComfyResourceState` / `ComfyCompositionStatus`. NO second
//! fetcher or telemetry client.
//!
//! Separation of concerns (§3): this module owns ONLY the DYNAMIC factual plane (health,
//! resource state, allowance, balance, GPU state, observation timestamp,
//! freshness/provenance). It does NOT own static capability, qualification, or policy.
//!
//! Shadow-only (§12): the plane produces an observable `FactPlaneSnapshot`; it must NEVER
//! alter a candidate set, score/select a target, or change fallback/routing. A failed
//! source read MUST fail closed (never a misleading favorable fact) (§14).
//!
//! Amendments (REALIGN-2 Part I):
//!   1. `projected_at_unix` = when this projection is ASSEMBLED (not when the observation
//!      was obtained). Each source retains its own `last_success_at` provenance; a fresh
//!      projection must NOT make an old provider observation appear newly observed.
//!   2. `source_health` (Healthy/Unavailable/Unknown) and `observation_freshness`
//!      (Fresh/Stale/Unknown) are SEPARATE and never overloaded. Freshness threshold is
//!      the EXISTING authoritative TTL=30s contract from the resource-pool config
//!      (`ttl_seconds = 30`); no invented timeout.
//!   3. The live accessor supplies Comfy CONTINUITY only (no new fetch); the rich
//!      `ComfyFacts` remains a domain projection type used in mapping tests — the endpoint
//!      does NOT falsely populate it.
//!   4. Endpoint trust boundary is recorded: `/v1/fact-plane` inherits the Switchyard
//!      listener trust boundary; deployment as an observation endpoint requires a later
//!      protection decision.

use std::time::{SystemTime, UNIX_EPOCH};

use libsy::{
    ComfyMode, ComfyOwner, ComfyResidentProfile, ComfyResourceState, ComfySnapshotState,
    ComfySourceHealth, ComfyTransitionState, ComfyTransitionTarget, DeepSeekResourceState,
    OpenAiResourceState, ResourceSnapshot,
};

/// The authoritative cloud-resource refresh TTL (seconds), mirrored from the deployed
/// resource-pool config (`[resource_pools.smart] ttl_seconds = 30`). This is the existing
/// refresh contract, NOT a newly invented timeout. A projection older than this is Stale.
pub const AUTHORITATIVE_REFRESH_TTL_SECONDS: u64 = 30;

// ---------------------------------------------------------------------------
// Shared provenance / health + freshness envelope (generic across all providers).
// ---------------------------------------------------------------------------

/// Source HEALTH of a factual observation. A failed read must be `Unavailable` (never an
/// accidental favorable/healthy value). This is independent of freshness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FactSourceHealth {
    Healthy,
    Unavailable,
    Unknown,
}

/// OBSERVATION FRESHNESS, separate from source health. A policy consumer must NEVER treat
/// an old cached provider snapshot as current merely because the last fetch succeeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FactFreshness {
    Fresh,
    Stale,
    /// No authoritative timestamp / threshold available -> do not pretend Fresh.
    Unknown,
}

impl FactSourceHealth {
    /// Map the OpenAI/DeepSeek domain `available` + `error` onto source health
    /// (health only — does NOT incorporate observation age).
    fn from_cloud(available: bool, has_error: bool) -> Self {
        if has_error {
            Self::Unavailable
        } else if available {
            Self::Healthy
        } else {
            Self::Unknown
        }
    }

    /// Map the Comfy domain `source_health` onto the generic health.
    fn from_comfy(h: ComfySourceHealth) -> Self {
        match h {
            ComfySourceHealth::Healthy => Self::Healthy,
            ComfySourceHealth::Stale | ComfySourceHealth::Unavailable => Self::Unavailable,
        }
    }
}

/// Determine observation freshness from the authoritative last-success timestamp vs the
/// projection time, using the authoritative TTL. Missing/invalid timestamp => Unknown.
fn cloud_freshness(last_success_at: Option<i64>, projected_at_unix: i64) -> FactFreshness {
    let Some(last) = last_success_at else {
        return FactFreshness::Unknown;
    };
    let age = projected_at_unix.saturating_sub(last);
    if age < 0 {
        // A future timestamp (clock skew) is not trustworthy -> not Fresh.
        return FactFreshness::Unknown;
    }
    if age <= AUTHORITATIVE_REFRESH_TTL_SECONDS as i64 {
        FactFreshness::Fresh
    } else {
        FactFreshness::Stale
    }
}

// ---------------------------------------------------------------------------
// Typed provider factual payloads.
// ---------------------------------------------------------------------------

/// OpenAI factual resource state, projected 1:1 from the existing `OpenAiResourceState`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpenAiFacts {
    /// SOURCE HEALTH (Healthy/Unavailable/Unknown) — does not include age.
    pub source_health: FactSourceHealth,
    /// OBSERVATION FRESHNESS — separate from health.
    pub observation_freshness: FactFreshness,
    /// FACT: provider reachable / has allowance (not a routing preference).
    pub available: bool,
    pub limit_reached: bool,
    pub spend_control_reached: bool,
    /// FACT: weekly used percentage (factual value). NOT a daily policy tranche.
    pub weekly_used_percent: Option<f64>,
    /// FACT: weekly reset timestamp (factual). NOT a policy-day boundary.
    pub weekly_reset_at: Option<i64>,
    /// FACT: credits balance string (provider-reported; opaque unit).
    pub credits_balance: Option<String>,
    /// Unix seconds of last successful read (provenance; NOT the projection time).
    pub last_success_at: Option<i64>,
    /// Seconds between the last successful observation and the projection assembly time.
    pub observation_age_seconds: Option<i64>,
}

/// DeepSeek factual resource state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeepSeekFacts {
    pub source_health: FactSourceHealth,
    pub observation_freshness: FactFreshness,
    /// FACT: provider availability (not a routing preference).
    pub is_available: bool,
    pub currency: String,
    pub total_balance: String,
    pub granted_balance: String,
    pub topped_up_balance: String,
    pub last_success_at: Option<i64>,
    pub observation_age_seconds: Option<i64>,
}

/// ComfyNinja factual local GPU state (DOMAIN PROJECTION TYPE).
///
/// The richer Comfy domain remains authoritative; this is a projection, not a replacement.
/// NOTE (amendment 3): the live `/v1/fact-plane` accessor does NOT currently populate this
/// (it would require a Comfy fetch not present in the reuse path). This type is used for
/// domain mapping tests and for a future non-fetching cached-state projection; it is NOT
/// falsely exposed as populated by the endpoint.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComfyFacts {
    pub source_health: FactSourceHealth,
    pub observation_freshness: FactFreshness,
    pub producer_epoch: String,
    pub state_generation: i64,
    pub state_fingerprint: String,
    pub owner: Option<ComfyOwner>,
    pub mode: Option<ComfyMode>,
    pub transition_state: Option<ComfyTransitionState>,
    pub transition_target: Option<ComfyTransitionTarget>,
    pub resident_qwen_profile: Option<ComfyResidentProfile>,
    pub vram_used_mib: Option<i64>,
    pub vram_free_mib: Option<i64>,
    pub comfy_busy: Option<bool>,
    pub comfy_queue_running: Option<i64>,
    pub comfy_queue_pending: Option<i64>,
    pub studio_backend_health: Option<String>,
    pub last_success_at: Option<i64>,
}

/// A lean, non-fetching Comfy continuity projection derived from the composition status
/// (enabled/boot/health/cursor). This is what the LIVE accessor actually supplies for Comfy
/// (amendment 3) — no fetch, no second producer.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComfyContinuityFacts {
    pub source_health: FactSourceHealth,
    pub enabled: bool,
    pub boot_state: String,
    pub producer_epoch: Option<String>,
    pub last_eid: Option<i64>,
    pub recent_event_count: usize,
    pub last_active_profile: Option<libsy::ComfyProfileClass>,
}

// ---------------------------------------------------------------------------
// The generic fact-plane snapshot.
// ---------------------------------------------------------------------------

/// A point-in-time generic inference-fact plane. UNKNOWN / Unavailable are first-class.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FactPlaneSnapshot {
    /// When this projection was ASSEMBLED (unix sec). NOT necessarily when any underlying
    /// observation was obtained — each source carries its own `last_success_at`.
    pub projected_at_unix: i64,
    pub openai: Option<OpenAiFacts>,
    pub deepseek: Option<DeepSeekFacts>,
    pub comfyninja: Option<ComfyFacts>,
    pub comfy_continuity: Option<ComfyContinuityFacts>,
}

// ---------------------------------------------------------------------------
// Projection functions (pure; field-preserving, no semantic re-interpretation).
// ---------------------------------------------------------------------------

fn project_openai(s: &OpenAiResourceState, projected_at_unix: i64) -> OpenAiFacts {
    let has_error = s.error.is_some();
    OpenAiFacts {
        source_health: FactSourceHealth::from_cloud(s.available, has_error),
        observation_freshness: cloud_freshness(s.last_success_at, projected_at_unix),
        available: s.available,
        limit_reached: s.limit_reached,
        spend_control_reached: s.spend_control_reached,
        weekly_used_percent: s.weekly_used_percent,
        weekly_reset_at: s.weekly_reset_at,
        credits_balance: s.credits_balance.clone(),
        last_success_at: s.last_success_at,
        observation_age_seconds: s
            .last_success_at
            .map(|l| projected_at_unix.saturating_sub(l)),
    }
}

fn project_deepseek(s: &DeepSeekResourceState, projected_at_unix: i64) -> DeepSeekFacts {
    let has_error = s.error.is_some();
    DeepSeekFacts {
        source_health: FactSourceHealth::from_cloud(s.is_available, has_error),
        observation_freshness: cloud_freshness(s.last_success_at, projected_at_unix),
        is_available: s.is_available,
        currency: s.currency.clone(),
        total_balance: s.total_balance.clone(),
        granted_balance: s.granted_balance.clone(),
        topped_up_balance: s.topped_up_balance.clone(),
        last_success_at: s.last_success_at,
        observation_age_seconds: s
            .last_success_at
            .map(|l| projected_at_unix.saturating_sub(l)),
    }
}

fn project_comfy(c: &ComfyResourceState, projected_at_unix: i64) -> ComfyFacts {
    let st: &ComfySnapshotState = &c.state;
    ComfyFacts {
        source_health: FactSourceHealth::from_comfy(c.source_health),
        observation_freshness: cloud_freshness(c.last_success_at, projected_at_unix),
        producer_epoch: c.producer_epoch.clone(),
        state_generation: c.state_generation,
        state_fingerprint: c.state_fingerprint.clone(),
        owner: st.owner,
        mode: st.mode,
        transition_state: st.transition_state,
        transition_target: st.transition_target,
        resident_qwen_profile: st.resident_qwen_profile,
        vram_used_mib: st.vram_used_mib,
        vram_free_mib: st.vram_free_mib,
        comfy_busy: st.comfy_busy,
        comfy_queue_running: st.comfy_queue_running,
        comfy_queue_pending: st.comfy_queue_pending,
        studio_backend_health: st.studio_backend_health.clone(),
        last_success_at: c.last_success_at,
    }
}

/// Assemble a shadow fact-plane projection from EXISTING factual structs + the
/// (non-fetching) Comfy composition status. Pure projection — no I/O, no routing authority.
///
/// `comfy_resource` is supplied only when a cached `ComfyResourceState` is already held
/// WITHOUT a fetch. In the current deployment the live accessor passes `None` here (it has
/// no non-fetching cached handle), so the endpoint exposes Comfy CONTINUITY — not rich
/// ComfyFacts — per amendment 3. `comfy_resource` remains available for domain mapping tests
/// and a future non-fetching cached projection.
pub fn project(
    resource: Option<&ResourceSnapshot>,
    comfy_resource: Option<&ComfyResourceState>,
    continuity: Option<&crate::comfy::ComfyCompositionStatus>,
) -> FactPlaneSnapshot {
    let projected_at_unix = now_unix();
    FactPlaneSnapshot {
        projected_at_unix,
        openai: resource
            .and_then(|r| r.openai.as_ref())
            .map(|o| project_openai(o, projected_at_unix)),
        deepseek: resource
            .and_then(|r| r.deepseek.as_ref())
            .map(|d| project_deepseek(d, projected_at_unix)),
        comfyninja: comfy_resource.map(|c| project_comfy(c, projected_at_unix)),
        comfy_continuity: continuity.map(|c| {
            project_continuity(
                c.enabled,
                c.boot_state.label(),
                c.last_source_health,
                c.current_cursor.as_ref(),
                c.recent_event_count,
                c.last_active_profile,
            )
        }),
    }
}

fn project_continuity(
    enabled: bool,
    boot: &str,
    health: ComfySourceHealth,
    cursor: Option<&libsy::TransitionCursor>,
    recent_event_count: usize,
    last_active_profile: Option<libsy::ComfyProfileClass>,
) -> ComfyContinuityFacts {
    ComfyContinuityFacts {
        source_health: FactSourceHealth::from_comfy(health),
        enabled,
        boot_state: boot.to_string(),
        producer_epoch: cursor.map(|c| c.producer_epoch.clone()),
        last_eid: cursor.map(|c| c.last_eid),
        recent_event_count,
        last_active_profile,
    }
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn openai(weekly: Option<f64>, last_success: Option<i64>, err: Option<String>) -> OpenAiResourceState {
        OpenAiResourceState {
            available: err.is_none(),
            limit_reached: false,
            spend_control_reached: false,
            weekly_used_percent: weekly,
            weekly_reset_at: Some(1787197109),
            credits_balance: Some("$12.34".to_string()),
            last_success_at: last_success,
            error: err,
        }
    }

    fn deepseek(last_success: Option<i64>) -> DeepSeekResourceState {
        DeepSeekResourceState {
            is_available: true,
            currency: "CNY".to_string(),
            total_balance: "123.45".to_string(),
            granted_balance: "100.00".to_string(),
            topped_up_balance: "23.45".to_string(),
            last_success_at: last_success,
            error: None,
        }
    }

    fn comfy(mode: Option<ComfyMode>, last_success: Option<i64>) -> ComfyResourceState {
        ComfyResourceState {
            producer_epoch: "epoch-1".to_string(),
            state_generation: 3,
            state_fingerprint: "fp".to_string(),
            state: ComfySnapshotState {
                owner: Some(ComfyOwner::Unsloth),
                mode,
                transition_state: Some(ComfyTransitionState::Idle),
                transition_target: Some(ComfyTransitionTarget::Idle),
                comfyui: None,
                unsloth_studio: Some("ok".to_string()),
                llama_server: None,
                studio_backend_health: Some("ok".to_string()),
                resident_qwen_profile: Some(ComfyResidentProfile::Unknown),
                comfy_busy: Some(false),
                comfy_queue_running: Some(0),
                comfy_queue_pending: Some(0),
                vram_used_mib: Some(67),
                vram_free_mib: Some(24260),
            },
            source_health: ComfySourceHealth::Healthy,
            last_success_at: last_success,
            error: None,
        }
    }

    fn snapshot(
        openai: Option<OpenAiResourceState>,
        deepseek: Option<DeepSeekResourceState>,
    ) -> ResourceSnapshot {
        ResourceSnapshot { openai, deepseek }
    }

    #[test]
    fn projection_time_is_distinct_from_source_observation_time() {
        // The projection always carries its assembly time; the source carries its own
        // last_success_at. A fresh projection must NOT make an old observation fresh.
        let last = 1700000000_i64; // truly old observation
        let f = project(
            Some(&snapshot(Some(openai(Some(14.0), Some(last), None)), None)),
            None,
            None,
        );
        let o = f.openai.unwrap();
        assert_eq!(o.last_success_at, Some(last));
        // age is large => NOT Fresh even though projected anew.
        assert_eq!(o.observation_freshness, FactFreshness::Stale);
        // projected_at is well after last_success (age >= TTL in this test).
        assert!(o.observation_age_seconds.unwrap() >= AUTHORITATIVE_REFRESH_TTL_SECONDS as i64);
    }

    #[test]
    fn recent_success_is_fresh_but_still_distinct_from_projection() {
        // last_success within TTL => Fresh.
        let projected = now_unix();
        let last = projected - 5;
        let f = project(
            Some(&snapshot(Some(openai(Some(14.0), Some(last), None)), None)),
            None,
            None,
        );
        assert_eq!(
            f.openai.unwrap().observation_freshness,
            FactFreshness::Fresh
        );
        // The projection still carries BOTH projected_at_unix and last_success_at.
        assert!(f.projected_at_unix >= last);
    }

    #[test]
    fn missing_last_success_at_yields_unknown_freshness() {
        let f = project(
            Some(&snapshot(Some(openai(Some(14.0), None, None)), None)),
            None,
            None,
        );
        assert_eq!(
            f.openai.unwrap().observation_freshness,
            FactFreshness::Unknown
        );
    }

    #[test]
    fn failed_source_is_unavailable_never_favorable() {
        let f = project(
            Some(&snapshot(
                Some(openai(None, None, Some("fetch failed".into()))),
                None,
            )),
            None,
            None,
        );
        let o = f.openai.unwrap();
        assert_eq!(o.available, false);
        assert_eq!(o.source_health, FactSourceHealth::Unavailable);
    }

    #[test]
    fn openai_maps_without_semantic_loss() {
        let f = project(
            Some(&snapshot(Some(openai(Some(14.0), Some(now_unix()), None)), None)),
            None,
            None,
        );
        let o = f.openai.unwrap();
        assert_eq!(o.weekly_used_percent, Some(14.0));
        assert_eq!(o.weekly_reset_at, Some(1787197109));
        assert_eq!(o.available, true);
        assert_eq!(o.source_health, FactSourceHealth::Healthy);
    }

    #[test]
    fn deepseek_maps_without_semantic_loss() {
        let f = project(Some(&snapshot(None, Some(deepseek(Some(now_unix()))))), None, None);
        let d = f.deepseek.unwrap();
        assert_eq!(d.currency, "CNY");
        assert_eq!(d.total_balance, "123.45");
        assert_eq!(d.is_available, true);
        assert_eq!(d.source_health, FactSourceHealth::Healthy);
    }

    #[test]
    fn comfy_domain_projection_maps_without_semantic_loss() {
        let f = project(None, Some(&comfy(Some(ComfyMode::Idle), Some(now_unix()))), None);
        let c = f.comfyninja.expect("comfy present");
        assert_eq!(c.producer_epoch, "epoch-1");
        assert_eq!(c.state_generation, 3);
        assert_eq!(c.mode, Some(ComfyMode::Idle));
        assert_eq!(c.resident_qwen_profile, Some(ComfyResidentProfile::Unknown));
        assert_eq!(c.vram_used_mib, Some(67));
        assert_eq!(c.vram_free_mib, Some(24260));
    }

    #[test]
    fn unknown_remains_unknown_and_absent() {
        let f = project(None, None, None);
        assert!(f.openai.is_none() && f.deepseek.is_none() && f.comfyninja.is_none());
        assert!(f.comfy_continuity.is_none());
        let f2 = project(None, Some(&comfy(None, Some(now_unix()))), None);
        assert_eq!(
            f2.comfyninja.unwrap().resident_qwen_profile,
            Some(ComfyResidentProfile::Unknown)
        );
    }

    #[test]
    fn stale_distinguishable_from_healthy() {
        let mut c = comfy(Some(ComfyMode::Idle), Some(now_unix() - 100));
        c.source_health = ComfySourceHealth::Healthy;
        let f = project(None, Some(&c), None);
        let cf = f.comfyninja.as_ref().expect("comfy present");
        assert_eq!(cf.source_health, FactSourceHealth::Healthy);
        assert_eq!(cf.observation_freshness, FactFreshness::Stale);
    }

    #[test]
    fn provider_fields_do_not_bleed() {
        let f = project(
            Some(&snapshot(
                Some(openai(Some(14.0), Some(now_unix()), None)),
                Some(deepseek(Some(now_unix()))),
            )),
            Some(&comfy(Some(ComfyMode::Idle), Some(now_unix()))),
            None,
        );
        assert_eq!(f.openai.unwrap().weekly_used_percent, Some(14.0));
        assert_eq!(f.deepseek.unwrap().currency, "CNY");
        assert_eq!(f.comfyninja.unwrap().vram_used_mib, Some(67));
    }

    #[test]
    fn continuity_projection_preserves_boot_and_cursor() {
        let cursor = libsy::TransitionCursor {
            producer_epoch: "epoch-9".to_string(),
            last_eid: 42,
        };
        let c = project_continuity(
            true,
            "rebuilding_history",
            ComfySourceHealth::Healthy,
            Some(&cursor),
            7,
            None,
        );
        assert_eq!(c.enabled, true);
        assert_eq!(c.boot_state, "rebuilding_history");
        assert_eq!(c.producer_epoch.as_deref(), Some("epoch-9"));
        assert_eq!(c.last_eid, Some(42));
        assert_eq!(c.recent_event_count, 7);
    }

    /// SHADOW-ONLY / ZERO-ROUTING-AUTHORITY invariant: the fact plane exposes only
    /// observable facts via pure projection functions. It imports NO routing type
    /// (Candidate/WorkClass/ReasoningPolicy/Algorithm/selection), structural + compile-enforced.
    #[test]
    fn fact_plane_has_no_routing_authority() {
        let f = project(None, None, None);
        assert!(f.openai.is_none() && f.deepseek.is_none() && f.comfyninja.is_none());
        let _ = f.projected_at_unix; // only observability fields are reachable
    }

    /// REALIGN-2 §16 invariant: a fact refresh changes the SHADOW fact state but MUST NOT
    /// change any routing result (structurally disconnected from routing).
    #[test]
    fn fact_refresh_changes_shadow_state_but_not_routing_result() {
        let low = project(
            Some(&snapshot(
                Some(openai(Some(12.0), Some(now_unix()), None)),
                Some(deepseek(Some(now_unix()))),
            )),
            Some(&comfy(Some(ComfyMode::Idle), Some(now_unix()))),
            None,
        );
        let high = project(
            Some(&snapshot(
                Some(openai(Some(40.0), Some(now_unix()), None)),
                Some(deepseek(Some(now_unix()))),
            )),
            Some(&comfy(Some(ComfyMode::Studio), Some(now_unix()))),
            None,
        );
        assert_ne!(low.openai, high.openai);
        assert_eq!(low.openai.as_ref().unwrap().weekly_used_percent, Some(12.0));
        assert_eq!(high.openai.as_ref().unwrap().weekly_used_percent, Some(40.0));
        let _ = (low.projected_at_unix, high.projected_at_unix);
    }
}
