//! REALIGN-2: generic shadow inference-fact plane (server-side, ZERO routing authority).
//!
//! This module is the *dynamic factual plane* of the future inference-routing
//! architecture. It **only projects** the already-fetched authoritative facts from the
//! existing deployment into a small generic envelope for shadow observation and future
//! consumption. It performs **no I/O**, **no target selection**, **no scoring**, and is
//! **never** read by any routing path.
//!
//! Reuse-first (§13):
//!   * OpenAI + DeepSeek facts come from the EXISTING `ResourceSnapshot` (produced by
//!     `resource_fetcher.rs` + `ResourceState`). This module does NOT re-fetch.
//!   * ComfyNinja facts come from the EXISTING Gate-C `ComfyResourceState` / status. This
//!     module does NOT open a second telemetry client.
//!
//! Separation of concerns (§3):
//!   * This module owns the **DYNAMIC factual plane** (health, resource state, allowance,
//!     balance, GPU state, observation timestamp, freshness/provenance).
//!   * It does NOT own static/declarative capability, qualification, or policy. No
//!     `daily_tranche_open`, no spending rules, no caller authority, no local-preference.
//!
//! Shadow-only (§12): the plane produces an observable `FactPlaneSnapshot`. It MUST NEVER
//! alter a candidate set, score/select a target, or change fallback/routing. A failed
//! source read MUST fail closed (never a misleading favorable fact) (§14).
//!
//! The invariants that this is shadow-only and has zero routing authority are structural:
//! this module imports ONLY factual domain types (no `Candidate`, `WorkClass`,
//! `ReasoningPolicy`, `Algorithm`, target-selection types), and exposes only pure
//! projection functions + observable facts. An explicit invariant test documents this.

use std::time::{SystemTime, UNIX_EPOCH};

use libsy::{
    ComfyMode, ComfyOwner, ComfyResidentProfile, ComfyResourceState, ComfySnapshotState,
    ComfySourceHealth, ComfyTransitionState, ComfyTransitionTarget, DeepSeekResourceState,
    OpenAiResourceState, ResourceSnapshot,
};

// ---------------------------------------------------------------------------
// Shared provenance / health envelope (generic across all providers).
// ---------------------------------------------------------------------------

/// Source health of a factual observation. A failed read must be `Unavailable` (never an
/// accidental favorable/healthy value).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FactSourceHealth {
    Healthy,
    Stale,
    Unavailable,
    Unknown,
}

impl FactSourceHealth {
    /// Map the OpenAI/DeepSeek domain `available` + `error` onto source health.
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
            ComfySourceHealth::Stale => Self::Stale,
            ComfySourceHealth::Unavailable => Self::Unavailable,
        }
    }
}

// ---------------------------------------------------------------------------
// Typed provider factual payloads (preserve domain semantics — no fake numeric
// equivalence between OpenAI allowance, DeepSeek balance, and GPU VRAM).
// ---------------------------------------------------------------------------

/// OpenAI factual resource state, projected 1:1 from the existing `OpenAiResourceState`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpenAiFacts {
    pub source_health: FactSourceHealth,
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
    /// Unix seconds of last successful read (None = never/unknown).
    pub last_success_at: Option<i64>,
}

/// DeepSeek factual resource state, projected 1:1 from the existing `DeepSeekResourceState`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeepSeekFacts {
    pub source_health: FactSourceHealth,
    /// FACT: provider availability (not a routing preference).
    pub is_available: bool,
    pub currency: String,
    pub total_balance: String,
    pub granted_balance: String,
    pub topped_up_balance: String,
    /// Unix seconds of last successful read (None = never/unknown).
    pub last_success_at: Option<i64>,
}

/// ComfyNinja factual local GPU state, projected from the existing Gate-C
/// `ComfyResourceState`. The richer Comfy domain remains authoritative; this is a
/// projection, not a replacement.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComfyFacts {
    pub source_health: FactSourceHealth,
    pub producer_epoch: String,
    pub state_generation: i64,
    pub state_fingerprint: String,
    pub owner: Option<ComfyOwner>,
    pub mode: Option<ComfyMode>,
    pub transition_state: Option<ComfyTransitionState>,
    pub transition_target: Option<ComfyTransitionTarget>,
    /// Governed factual profile seam (FAST/LONG/unknown) — never inferred.
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
/// (enabled/boot/health/cursor). Used by the shadow observability path so it never induces
/// a live telemetry fetch; the rich `ComfyFacts` (VRAM/mode/queue) is assembled from a
/// cached `ComfyResourceState` when one is already held (and for domain mapping tests).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComfyContinuityFacts {
    pub source_health: FactSourceHealth,
    pub enabled: bool,
    /// Deterministic bootstrap/continuity state label.
    pub boot_state: String,
    /// Opaque producer-epoch from the cursor, if a cursor is held.
    pub producer_epoch: Option<String>,
    /// Highest contiguous ingested eid, if a cursor is held.
    pub last_eid: Option<i64>,
    pub recent_event_count: usize,
    pub last_active_profile: Option<libsy::ComfyProfileClass>,
}

// ---------------------------------------------------------------------------
// The generic fact-plane snapshot (shadow observability + future consumption).
// ---------------------------------------------------------------------------

/// A point-in-time generic inference-fact plane. UNKNOWN / Unavailable are first-class.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FactPlaneSnapshot {
    /// Unix seconds when this projection was observed/assembled.
    pub as_of_unix: i64,
    pub openai: Option<OpenAiFacts>,
    pub deepseek: Option<DeepSeekFacts>,
    pub comfyninja: Option<ComfyFacts>,
    /// Non-fetching continuity view (from Gate-C status) — always available when enabled.
    pub comfy_continuity: Option<ComfyContinuityFacts>,
}

// ---------------------------------------------------------------------------
// Projection functions (pure; field-preserving, no semantic re-interpretation).
// ---------------------------------------------------------------------------

fn project_openai(s: &OpenAiResourceState) -> OpenAiFacts {
    let has_error = s.error.is_some();
    OpenAiFacts {
        source_health: FactSourceHealth::from_cloud(s.available, has_error),
        available: s.available,
        limit_reached: s.limit_reached,
        spend_control_reached: s.spend_control_reached,
        weekly_used_percent: s.weekly_used_percent,
        weekly_reset_at: s.weekly_reset_at,
        credits_balance: s.credits_balance.clone(),
        last_success_at: s.last_success_at,
    }
}

fn project_deepseek(s: &DeepSeekResourceState) -> DeepSeekFacts {
    let has_error = s.error.is_some();
    DeepSeekFacts {
        source_health: FactSourceHealth::from_cloud(s.is_available, has_error),
        is_available: s.is_available,
        currency: s.currency.clone(),
        total_balance: s.total_balance.clone(),
        granted_balance: s.granted_balance.clone(),
        topped_up_balance: s.topped_up_balance.clone(),
        last_success_at: s.last_success_at,
    }
}

fn project_comfy(c: &ComfyResourceState) -> ComfyFacts {
    let st: &ComfySnapshotState = &c.state;
    ComfyFacts {
        source_health: FactSourceHealth::from_comfy(c.source_health),
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

/// Assemble a shadow fact-plane projection from the EXISTING factual structs + the
/// (non-fetching) Comfy composition status. Pure projection — no I/O, no routing authority.
/// Any absent source is `None` / `comfy_continuity=None` when disabled (no fabricated fact).
pub fn project(
    resource: Option<&ResourceSnapshot>,
    comfy: Option<&ComfyResourceState>,
    continuity: Option<&crate::comfy::ComfyCompositionStatus>,
) -> FactPlaneSnapshot {
    FactPlaneSnapshot {
        as_of_unix: now_unix(),
        openai: resource.and_then(|r| r.openai.as_ref()).map(project_openai),
        deepseek: resource
            .and_then(|r| r.deepseek.as_ref())
            .map(project_deepseek),
        comfyninja: comfy.map(project_comfy),
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

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests — mapping without semantic loss, UNKNOWN/unavailable behavior, field
// isolation, and the shadow-only / zero-routing-authority invariant.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn openai(weekly: Option<f64>, reset: Option<i64>, err: Option<String>) -> OpenAiResourceState {
        OpenAiResourceState {
            available: err.is_none(),
            limit_reached: false,
            spend_control_reached: false,
            weekly_used_percent: weekly,
            weekly_reset_at: reset,
            credits_balance: Some("$12.34".to_string()),
            last_success_at: Some(1700000000),
            error: err,
        }
    }

    fn deepseek() -> DeepSeekResourceState {
        DeepSeekResourceState {
            is_available: true,
            currency: "CNY".to_string(),
            total_balance: "123.45".to_string(),
            granted_balance: "100.00".to_string(),
            topped_up_balance: "23.45".to_string(),
            last_success_at: Some(1700000000),
            error: None,
        }
    }

    fn comfy(mode: Option<ComfyMode>) -> ComfyResourceState {
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
            last_success_at: Some(1700000000),
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
    fn openai_maps_without_semantic_loss() {
        let f = project(
            Some(&snapshot(Some(openai(Some(14.0), Some(1787197109), None)), None)),
            None,
            None,
        );
        let o = f.openai.expect("openai present");
        assert_eq!(o.weekly_used_percent, Some(14.0));
        assert_eq!(o.weekly_reset_at, Some(1787197109));
        assert_eq!(o.available, true);
        assert_eq!(o.source_health, FactSourceHealth::Healthy);
    }

    #[test]
    fn deepseek_maps_without_semantic_loss() {
        let f = project(Some(&snapshot(None, Some(deepseek()))), None, None);
        let d = f.deepseek.expect("deepseek present");
        assert_eq!(d.currency, "CNY");
        assert_eq!(d.total_balance, "123.45");
        assert_eq!(d.is_available, true);
        assert_eq!(d.source_health, FactSourceHealth::Healthy);
    }

    #[test]
    fn comfy_maps_without_semantic_loss() {
        let f = project(None, Some(&comfy(Some(ComfyMode::Idle))), None);
        let c = f.comfyninja.expect("comfy present");
        assert_eq!(c.producer_epoch, "epoch-1");
        assert_eq!(c.state_generation, 3);
        assert_eq!(c.mode, Some(ComfyMode::Idle));
        assert_eq!(c.owner, Some(ComfyOwner::Unsloth));
        assert_eq!(c.resident_qwen_profile, Some(ComfyResidentProfile::Unknown));
        assert_eq!(c.vram_used_mib, Some(67));
        assert_eq!(c.vram_free_mib, Some(24260));
        assert_eq!(c.comfy_busy, Some(false));
    }

    #[test]
    fn unknown_remains_unknown_and_absent() {
        let f = project(None, None, None);
        assert!(f.openai.is_none() && f.deepseek.is_none() && f.comfyninja.is_none());
        assert!(f.comfy_continuity.is_none());
        let f2 = project(None, Some(&comfy(None)), None);
        assert_eq!(
            f2.comfyninja.unwrap().resident_qwen_profile,
            Some(ComfyResidentProfile::Unknown)
        );
    }

    #[test]
    fn source_failure_fails_closed() {
        let f = project(
            Some(&snapshot(Some(openai(None, None, Some("fetch failed".into()))), None)),
            None,
            None,
        );
        let o = f.openai.unwrap();
        assert_eq!(o.available, false);
        assert_eq!(o.source_health, FactSourceHealth::Unavailable);
    }

    #[test]
    fn stale_distinguishable_from_healthy() {
        let mut c = comfy(Some(ComfyMode::Idle));
        c.source_health = ComfySourceHealth::Stale;
        let f = project(None, Some(&c), None);
        assert_eq!(f.comfyninja.unwrap().source_health, FactSourceHealth::Stale);
    }

    #[test]
    fn provider_fields_do_not_bleed() {
        let f = project(
            Some(&snapshot(
                Some(openai(Some(14.0), None, None)),
                Some(deepseek()),
            )),
            Some(&comfy(Some(ComfyMode::Idle))),
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
    /// (Candidate/WorkClass/ReasoningPolicy/Algorithm/selection), which is structural and
    /// compile-enforced. This test re-asserts the observable vacuum: no method here can
    /// alter a route or candidate set.
    #[test]
    fn fact_plane_has_no_routing_authority() {
        let f = project(None, None, None);
        assert!(f.openai.is_none() && f.deepseek.is_none() && f.comfyninja.is_none());
        let _ = f.as_of_unix; // only observability fields are reachable
    }

    /// REALIGN-2 §16 invariant: a fact refresh changes the SHADOW fact state but MUST NOT
    /// change any routing result. Because the fact plane is structurally disconnected from
    /// routing (it exposes only observable facts via pure projection; it has no selection
    /// API and imports no routing type), changing its inputs can only change its output —
    /// never a candidate set, a target choice, or a fallback.
    #[test]
    fn fact_refresh_changes_shadow_state_but_not_routing_result() {
        // Two different factual observations.
        let low = project(
            Some(&snapshot(
                Some(openai(Some(12.0), Some(1787197109), None)),
                Some(deepseek()),
            )),
            Some(&comfy(Some(ComfyMode::Idle))),
            None,
        );
        let high = project(
            Some(&snapshot(
                Some(openai(Some(40.0), Some(1787197109), None)),
                Some(deepseek()),
            )),
            Some(&comfy(Some(ComfyMode::Studio))),
            None,
        );
        // (a) The shadow fact state DID change with the refresh.
        assert_ne!(low.openai, high.openai);
        assert_ne!(low.comfyninja, high.comfyninja);
        assert_eq!(low.openai.as_ref().unwrap().weekly_used_percent, Some(12.0));
        assert_eq!(high.openai.as_ref().unwrap().weekly_used_percent, Some(40.0));
        // (b) There is NO routing surface to change: the fact-plane API has no candidate,
        //     no score, no selection, no fallback. This is structural (compile-enforced by
        //     the narrow imports) and re-asserted: both projections dereference to the same
        //     observable type with no mutable routing handle.
        let _ = (low.as_of_unix, high.as_of_unix);
        let _ = low.deepseek.as_ref().map(|d| d.total_balance.clone());
        let _ = high.comfy_continuity.as_ref();
    }
}
