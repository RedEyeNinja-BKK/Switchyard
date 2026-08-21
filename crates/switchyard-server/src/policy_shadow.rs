//! POLICY-CONSUME-SHADOW: OpenAI daily-tranche shadow policy (OBSERVATION ONLY).
//!
//! Evolved from FACT-3A with the POLICY-CONSUME-SHADOW corrections:
//!   1. Provider-window continuity must be POSITIVELY known (Some→Some same); None
//!      transitions are UNKNOWN, never silently continuous.
//!   2. A prior-day observation anchors a new Bangkok policy day ONLY when it sufficiently
//!      brackets the policy boundary under a documented near-boundary rule.
//!   3. Non-monotonic same-window usage (current < prior) is DISCONTINUITY, not clamp-to-zero.
//!   4. A later fresh continuous cumulative observation may recover accounting to Ok within
//!      a positively-known provider week.
//!   5. Policy output describes POLICY PERMISSION/PREFERENCE, not factual provider health.
//!   6. A thin adapter derives observation admissibility from the factual contract
//!      (OpenAiFacts), not an arbitrary caller bool.
//!   7. Repeated projection of an unchanged provider observation does NOT manufacture a new
//!      accounting event.
//!
//! It remains: FACT PLANE ≠ POLICY PLANE (this module consumes facts, never fetches/writes
//! them); POLICY ≠ ROUTING (zero candidate/selection/fallback authority); CAPABILITY is
//! never redefined by budget. This answers ONLY "which ordinary CLOUD lane should
//! participate"; it implies nothing about local inference.

/// Asia/Bangkok constant UTC offset (no DST): +7 hours = 25200 seconds.
const BANGKOK_OFFSET_SECS: i64 = 7 * 3600;
/// The operator daily tranche target: ~14 percentage points of the weekly allowance/day.
pub const OPENAI_DAILY_TRANCHE_PP: f64 = 14.0;
/// Near-boundary bracket (seconds before Bangkok midnight) within which a prior-day
/// observation is a trustworthy day-start anchor. Request-driven cloud refresh may produce
/// no observation near midnight; a sample from hours before midnight is NOT a factual 00:00
/// baseline and must NOT anchor the day. This is a documented SHADOW-phase rule — the
/// observation lifecycle must later guarantee/reconstruct a trustworthy boundary before the
/// policy gains any routing authority.
pub const POLICY_BOUNDARY_BRACKET_SECS: i64 = 3600; // 1 hour before Bangkok midnight

// ---------------------------------------------------------------------------
// Output types.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrancheStatus {
    Open,
    Consumed,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AccountingStatus {
    Ok,
    Uninitialized,
    ContinuityDegraded,
    FreshnessGated,
}

/// The ordinary-CLOUD policy disposition (POLICY PERMISSION/PREFERENCE — NOT factual
/// provider health). These never claim "OpenAI is reachable" / "DeepSeek is reachable";
/// reachability is a fact-plane concern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OrdinaryCloudPhase {
    /// OpenAI daily tranche is open -> ordinary OpenAI is POLICY-PERMITTED.
    OpenaiOrdinaryPermitted,
    /// OpenAI tranche consumed -> ordinary OpenAI conserved; DeepSeek is the normal
    /// ordinary-cloud continuation until the next Bangkok policy day.
    ConserveOpenaiDeepSeekContinuation,
    /// Indeterminate.
    Unknown,
}

/// Provider-week continuity MUST be positively established; absence is never proof of
/// continuity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ProviderWeekContinuity {
    /// previous Some(X) and current Some(X) -> continuity may be established.
    KnownContinuous,
    /// Same day, Some(X) -> Some(Y) with X != Y -> reset/change.
    Changed,
    /// Any None-involving transition (None→None, None→Some, Some→None) or unknown week ->
    /// cannot establish continuity.
    Unknown,
}

/// One structured observation (FACT — supplied by the fact plane, not computed here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenAiObservation {
    pub weekly_used_percent: Option<f64>,
    pub weekly_reset_at: Option<i64>,
    pub last_success_at: Option<i64>,
}

/// Observation IDENTITY (deduplication): a fresh projection/query with an unchanged identity
/// is NOT a new provider observation and must not advance the accounting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservationId {
    pub last_success_at: Option<i64>,
    pub weekly_used_percent: Option<f64>,
    pub weekly_reset_at: Option<i64>,
}

impl ObservationId {
    pub fn of(o: &OpenAiObservation) -> Self {
        Self {
            last_success_at: o.last_success_at,
            weekly_used_percent: o.weekly_used_percent,
            weekly_reset_at: o.weekly_reset_at,
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct OpenAiDailyTranchePolicy {
    pub policy_day_id: i64,
    pub tranche_status: TrancheStatus,
    pub accounting_status: AccountingStatus,
    pub consumed_today_pp: Option<f64>,
    pub day_start_baseline_pp: Option<f64>,
    pub remaining_today_pp: Option<f64>,
    pub day_start_provider_week: Option<i64>,
    pub latest_provider_week: Option<i64>,
    pub provider_week_continuity: ProviderWeekContinuity,
    pub ordinary_cloud_phase: OrdinaryCloudPhase,
    /// true when the observation advanced the accounting (new observation), false when it was
    /// a duplicate of an already-seen observation.
    pub advanced: bool,
    pub reasons: Vec<&'static str>,
}

// ---------------------------------------------------------------------------
// Retained accounting state (volatile — NOT durable in this shadow phase).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OpenAiTrancheState {
    pub current_policy_day: i64,
    pub day_start_baseline_pp: Option<f64>,
    pub day_start_provider_week: Option<i64>,
    pub latest_provider_week: Option<i64>,
    pub last_weekly_observed: Option<f64>,
    pub last_observed_at: Option<i64>,
    pub last_observation_id: Option<ObservationId>,
    pub accounting_status: AccountingStatus,
}

impl Default for OpenAiTrancheState {
    fn default() -> Self {
        Self {
            current_policy_day: 0,
            day_start_baseline_pp: None,
            day_start_provider_week: None,
            latest_provider_week: None,
            last_weekly_observed: None,
            last_observed_at: None,
            last_observation_id: None,
            accounting_status: AccountingStatus::Uninitialized,
        }
    }
}

/// The Asia/Bangkok policy-day id for a unix timestamp (Bangkok = UTC+7, no DST).
pub fn policy_day_id(unix_secs: i64) -> i64 {
    (unix_secs + BANGKOK_OFFSET_SECS).div_euclid(86400)
}

/// Seconds from a given unix time to the NEXT Bangkok midnight boundary (0 at boundary).
fn seconds_to_next_bangkok_midnight(unix_secs: i64) -> i64 {
    let day = policy_day_id(unix_secs);
    let day_start_utc = (day * 86400) - BANGKOK_OFFSET_SECS;
    let next_midnight_utc = day_start_utc + 86400;
    next_midnight_utc.saturating_sub(unix_secs).max(0)
}

// ---------------------------------------------------------------------------
// Observation admissibility adapter (from the factual contract — not a caller bool).
// ---------------------------------------------------------------------------

/// Admissibility summary derived by the policy layer from `OpenAiFacts` (the amended
/// fact-plane contract) — the policy consumer never trusts an arbitrary caller bool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmissibleObservation {
    pub usable: bool,
    pub weekly_used_percent: Option<f64>,
    pub weekly_reset_at: Option<i64>,
    pub last_success_at: Option<i64>,
    pub reason: Option<&'static str>,
}

/// Derive an admissible policy observation from the fact plane's `OpenAiFacts`.
///
/// Admissible only when: source is Healthy AND observation is Fresh AND weekly_used_percent
/// is present AND the percentage is finite + in [0,100]. A failed/stale/unknown source yields
/// `usable=false` (never a confident tranche). This is the REAL shadow composition path — the
/// policy consumer cannot accidentally pass `freshness_ok=true` for stale facts.
pub fn admit_from_openai_facts(o: &crate::fact_plane::OpenAiFacts) -> AdmissibleObservation {
    use crate::fact_plane::{FactFreshness, FactSourceHealth};
    if o.source_health != FactSourceHealth::Healthy {
        return AdmissibleObservation {
            usable: false,
            weekly_used_percent: o.weekly_used_percent,
            weekly_reset_at: o.weekly_reset_at,
            last_success_at: o.last_success_at,
            reason: Some("openai_source_not_healthy"),
        };
    }
    if o.observation_freshness != FactFreshness::Fresh {
        return AdmissibleObservation {
            usable: false,
            weekly_used_percent: o.weekly_used_percent,
            weekly_reset_at: o.weekly_reset_at,
            last_success_at: o.last_success_at,
            reason: Some("openai_observation_not_fresh"),
        };
    }
    match o.weekly_used_percent {
        Some(p) if valid_percent(p) => AdmissibleObservation {
            usable: true,
            weekly_used_percent: Some(p),
            weekly_reset_at: o.weekly_reset_at,
            last_success_at: o.last_success_at,
            reason: None,
        },
        _ => AdmissibleObservation {
            usable: false,
            weekly_used_percent: o.weekly_used_percent,
            weekly_reset_at: o.weekly_reset_at,
            last_success_at: o.last_success_at,
            reason: Some("weekly_used_percent_missing_or_invalid"),
        },
    }
}

fn valid_percent(p: f64) -> bool {
    p.is_finite() && (0.0..=100.0).contains(&p)
}

// ---------------------------------------------------------------------------
// The accounting algorithm.
// ---------------------------------------------------------------------------

/// Determine provider-week continuity. Positively known only when BOTH the previous and the
/// current window id are `Some` and equal.
fn provider_week_continuity(prev: Option<i64>, cur: Option<i64>) -> ProviderWeekContinuity {
    match (prev, cur) {
        (Some(a), Some(b)) if a == b => ProviderWeekContinuity::KnownContinuous,
        (Some(_), Some(_)) => ProviderWeekContinuity::Changed,
        _ => ProviderWeekContinuity::Unknown, // None-involving => cannot establish
    }
}

/// Assess one observation for the current Bangkok policy day, updating retained state.
/// `duplicate` is set when the observation identity matches the last seen (deduplication).
pub fn assess(
    now_unix_secs: i64,
    observation: OpenAiObservation,
    admissible: bool,
    retained: &mut OpenAiTrancheState,
) -> OpenAiDailyTranchePolicy {
    let day = policy_day_id(now_unix_secs);

    // --- Deduplication: an unchanged provider observation is NOT a new accounting event.
    // Observation identity is assigned BEFORE the admissibility gate, so it is INDEPENDENT
    // of whether the observation is usable for policy: polling the same unusable observation
    // repeatedly must not manufacture multiple factual events (POLICY-CONSUME-SHADOW §6).
    let id = ObservationId::of(&observation);
    if retained.last_observation_id == Some(id) {
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: retained.accounting_status,
            consumed_today_pp: consumed_from(retained),
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            provider_week_continuity: ProviderWeekContinuity::Unknown,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            advanced: false,
            reasons: vec!["duplicate_observation_no_new_event"],
        };
    }
    // Record this observation's identity now (before any admissibility/early return) so
    // repeated identical observations — including unusable ones — are correctly deduplicated.
    retained.last_observation_id = Some(id);

    // --- Admissibility gate (fresh + healthy + valid %). Refuse confident tranche otherwise.
    if !admissible {
        retained.accounting_status = AccountingStatus::FreshnessGated;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::FreshnessGated,
            consumed_today_pp: consumed_from(retained),
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            provider_week_continuity: ProviderWeekContinuity::Unknown,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            advanced: true,
            reasons: vec!["observation_not_admissible"],
        };
    }
    let Some(weekly) = observation.weekly_used_percent else {
        retained.accounting_status = retained.accounting_status;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: retained.accounting_status,
            consumed_today_pp: consumed_from(retained),
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            provider_week_continuity: ProviderWeekContinuity::Unknown,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            advanced: true,
            reasons: vec!["weekly_used_percent_missing"],
        };
    };

    let week = observation.weekly_reset_at;
    // Capture continuity between the OLD and NEW provider window (compared against retained
    // prior state before it is updated below).
    let cont = provider_week_continuity(retained.latest_provider_week, week);

    // --- A NEW Bangkok policy day (F): the day's baseline may anchor from a prior-day
    //     observation ONLY if it brackets the policy boundary under the documented rule.
    if retained.current_policy_day != day {
        retained.current_policy_day = day;
        let prior_weekly = retained.last_weekly_observed;
        let prior_obs_at = retained.last_observed_at;
        retained.last_weekly_observed = Some(weekly);
        retained.last_observed_at = observation.last_success_at;
        retained.latest_provider_week = week;

        // Trustworthy boundary anchor requires: provider week positively continuous AND the
        // prior observation was within the near-boundary bracket before Bangkok midnight.
        let brackets_boundary = prior_obs_at.map_or(false, |at| {
            // The prior observation must be on the PRIOR policy day and within the bracket of
            // that day's own midnight boundary -> i.e. close to the start of the new day.
            let prior_day = policy_day_id(at);
            let within_bracket = seconds_to_next_bangkok_midnight(at) <= POLICY_BOUNDARY_BRACKET_SECS;
            prior_day != day && within_bracket
        });
        let anchor_ok = cont == ProviderWeekContinuity::KnownContinuous && brackets_boundary && prior_weekly.is_some();
        if anchor_ok {
            retained.day_start_baseline_pp = prior_weekly;
            retained.day_start_provider_week = week;
            retained.accounting_status = AccountingStatus::Ok;
        } else {
            retained.day_start_baseline_pp = None;
            retained.accounting_status = AccountingStatus::Uninitialized;
        }
        return policy_output(retained, day, cont, true);
    }

    // Same policy day.
    let prior_weekly_same_day = retained.last_weekly_observed;
    retained.last_weekly_observed = Some(weekly);
    retained.last_observed_at = observation.last_success_at;
    retained.latest_provider_week = week;

    // --- Provider week changed INSIDE the day (G/H) -> cannot reconstruct, do NOT auto-grant.
    if cont == ProviderWeekContinuity::Changed {
        retained.accounting_status = AccountingStatus::ContinuityDegraded;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::ContinuityDegraded,
            consumed_today_pp: None,
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            provider_week_continuity: cont,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            advanced: true,
            reasons: vec!["provider_weekly_reset_within_policy_day"],
        };
    }
    // Provider week UNKNOWN (None-involving) -> cannot establish continuity; do not infer.
    if cont == ProviderWeekContinuity::Unknown {
        // We cannot positively know the window is continuous; without another continuity
        // mechanism we must NOT treat it as continuous. Report UNKNOWN (no false baseline).
        retained.accounting_status = AccountingStatus::Uninitialized;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::Uninitialized,
            consumed_today_pp: None,
            day_start_baseline_pp: retained.day_start_baseline_pp,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            provider_week_continuity: cont,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            advanced: true,
            reasons: vec!["provider_week_continuity_unknown"],
        };
    }

    // Positively same provider week: usage must be non-decreasing. A drop is discontinuity
    // (reset/correction/malformed), NOT clamp-to-zero.
    if let (Some(prev), Some(cur)) = (prior_weekly_same_day, Some(weekly)) {
        if cur < prev {
            retained.accounting_status = AccountingStatus::ContinuityDegraded;
            return OpenAiDailyTranchePolicy {
                policy_day_id: day,
                tranche_status: TrancheStatus::Unknown,
                accounting_status: AccountingStatus::ContinuityDegraded,
                consumed_today_pp: None,
                day_start_baseline_pp: retained.day_start_baseline_pp,
                remaining_today_pp: None,
                day_start_provider_week: retained.day_start_provider_week,
                latest_provider_week: retained.latest_provider_week,
                provider_week_continuity: cont,
                ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
                advanced: true,
                reasons: vec!["non_monotonic_same_window_discontinuity"],
            };
        }
    }

    // --- No baseline this day?
    if retained.day_start_baseline_pp.is_none() {
        retained.accounting_status = AccountingStatus::Uninitialized;
        return OpenAiDailyTranchePolicy {
            policy_day_id: day,
            tranche_status: TrancheStatus::Unknown,
            accounting_status: AccountingStatus::Uninitialized,
            consumed_today_pp: None,
            day_start_baseline_pp: None,
            remaining_today_pp: None,
            day_start_provider_week: retained.day_start_provider_week,
            latest_provider_week: retained.latest_provider_week,
            provider_week_continuity: cont,
            ordinary_cloud_phase: OrdinaryCloudPhase::Unknown,
            advanced: true,
            reasons: vec!["no_day_start_baseline"],
        };
    }

    // --- Freshness RECOVERY (correction 5): a later fresh observation in a positively-known
    //     same week + same day + valid baseline + monotonic cumulative usage may restore Ok.
    retained.accounting_status = AccountingStatus::Ok;
    policy_output(retained, day, cont, true)
}

fn consumed_from(r: &OpenAiTrancheState) -> Option<f64> {
    match (r.day_start_baseline_pp, r.last_weekly_observed) {
        (Some(b), Some(n)) => {
            if valid_percent(n) && (n - b) >= 0.0 {
                Some(n - b)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn policy_output(
    retained: &OpenAiTrancheState,
    day: i64,
    cont: ProviderWeekContinuity,
    advanced: bool,
) -> OpenAiDailyTranchePolicy {
    let baseline = retained.day_start_baseline_pp;
    let consumed = consumed_from(retained);
    let (mut tranche, mut phase, mut reasons) = match consumed {
        Some(c) if c < OPENAI_DAILY_TRANCHE_PP => (
            TrancheStatus::Open,
            OrdinaryCloudPhase::OpenaiOrdinaryPermitted,
            vec!["tranche_open_openai_ordinary_permitted"],
        ),
        Some(_) => (
            TrancheStatus::Consumed,
            OrdinaryCloudPhase::ConserveOpenaiDeepSeekContinuation,
            vec!["tranche_consumed_conserve_openai_deepseek_continuation"],
        ),
        None => (
            TrancheStatus::Unknown,
            OrdinaryCloudPhase::Unknown,
            vec!["no_consumed_basis"],
        ),
    };
    if retained.accounting_status != AccountingStatus::Ok {
        tranche = TrancheStatus::Unknown;
        phase = OrdinaryCloudPhase::Unknown;
        reasons.push("accounting_not_ok");
    }
    OpenAiDailyTranchePolicy {
        policy_day_id: day,
        tranche_status: tranche,
        accounting_status: retained.accounting_status,
        consumed_today_pp: consumed,
        day_start_baseline_pp: baseline,
        remaining_today_pp: consumed.map(|c| (OPENAI_DAILY_TRANCHE_PP - c).max(0.0)),
        day_start_provider_week: retained.day_start_provider_week,
        latest_provider_week: retained.latest_provider_week,
        provider_week_continuity: cont,
        ordinary_cloud_phase: phase,
        advanced,
        reasons,
    }
}

// ---------------------------------------------------------------------------
// POLICY-CONSUME-SHADOW tests (corrected contract + §15 mandatory scenarios).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fact_plane::{FactFreshness, FactSourceHealth, OpenAiFacts};
    use std::time::Duration;

    fn obs(weekly: Option<f64>, reset: Option<i64>, last: Option<i64>) -> OpenAiObservation {
        OpenAiObservation {
            weekly_used_percent: weekly,
            weekly_reset_at: reset,
            last_success_at: last,
        }
    }

    /// Build OpenAiFacts directly (admission is derived by the adapter, so we exercise it).
    fn facts(weekly: Option<f64>, reset: Option<i64>, last: Option<i64>) -> OpenAiFacts {
        OpenAiFacts {
            source_health: FactSourceHealth::Healthy,
            observation_freshness: FactFreshness::Fresh,
            available: true,
            limit_reached: false,
            spend_control_reached: false,
            weekly_used_percent: weekly,
            weekly_reset_at: reset,
            credits_balance: None,
            last_success_at: last,
            observation_age_seconds: Some(0),
        }
    }

    /// A fixed Bangkok "now" (well inside a policy day).
    const BANGKOK_NOON_UTC: i64 = 1_700_000_000;
    /// A fixed provider week id.
    const WEEK: i64 = 1000;

    /// A timestamp close to the Bangkok-midnight boundary (within the bracket) on the PRIOR
    /// policy day - so it is a valid day-start anchor for the target day.
    fn prior_boundary_anchor() -> i64 {
        // BANGKOK_NOON_UTC is inside target day D. The prior day's midnight is at the end of
        // day D-1. We want an instant within POLICY_BOUNDARY_BRACKET_SECS before that midnight.
        let day = policy_day_id(BANGKOK_NOON_UTC);
        let this_day_start_utc = (day * 86400) - BANGKOK_OFFSET_SECS;
        // Bangkok midnight that begins the target day == this_day_start_utc in local terms.
        // The anchor is 10 min before that midnight, i.e. 600s before this_day_start_utc.
        this_day_start_utc - 600
    }

    // --- Provider-week continuity (correction 2) --------------------------------

    #[test]
    fn provider_week_some_some_same_is_known_continuous() {
        assert_eq!(provider_week_continuity(Some(1), Some(1)), ProviderWeekContinuity::KnownContinuous);
    }
    #[test]
    fn provider_week_missing_is_unknown() {
        assert_eq!(provider_week_continuity(None, None), ProviderWeekContinuity::Unknown);
        assert_eq!(provider_week_continuity(None, Some(1)), ProviderWeekContinuity::Unknown);
        assert_eq!(provider_week_continuity(Some(1), None), ProviderWeekContinuity::Unknown);
    }
    #[test]
    fn provider_week_change_is_changed() {
        assert_eq!(provider_week_continuity(Some(1), Some(2)), ProviderWeekContinuity::Changed);
    }

    // --- Day-boundary baseline (correction 3) -----------------------------------

    #[test]
    fn near_boundary_prior_observation_anchors_baseline() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor(); // within bracket before target-day midnight
        let p = assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        assert_eq!(p.tranche_status, TrancheStatus::Unknown); // cold start on the prior day
        // Now the target day with a fresh observation.
        let t0 = BANGKOK_NOON_UTC;
        let r = assess(t0, obs(Some(33.0), Some(WEEK), Some(t0)), true, &mut st);
        // Prior observation bracketed the boundary + same week => baseline accepted (31).
        assert_eq!(r.day_start_baseline_pp, Some(31.0));
        assert_eq!(r.tranche_status, TrancheStatus::Open);
    }

    #[test]
    fn old_prior_day_observation_does_not_anchor_baseline() {
        let mut st = OpenAiTrancheState::default();
        // Prior observation is hours before midnight (NOT within the bracket).
        let old = BANGKOK_NOON_UTC - 86400 + 5 * 3600; // prior day ~5h before its midnight
        assess(old, obs(Some(31.0), Some(WEEK), Some(old)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        let r = assess(t0, obs(Some(33.0), Some(WEEK), Some(t0)), true, &mut st);
        // Old observation not near boundary -> NOT accepted as midnight baseline; UNINITIALIZED.
        assert_eq!(r.accounting_status, AccountingStatus::Uninitialized);
        assert_eq!(r.day_start_baseline_pp, None);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
    }

    #[test]
    fn cold_start_mid_day_is_unknown_not_fresh_tranche() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        let r = assess(t0, obs(Some(62.0), Some(WEEK), Some(t0)), true, &mut st);
        assert_eq!(r.accounting_status, AccountingStatus::Uninitialized);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert_eq!(r.consumed_today_pp, None);
    }

    #[test]
    fn bangkok_rollover_without_valid_boundary_is_unknown() {
        let mut st = OpenAiTrancheState::default();
        // Seed a prior day observation NOT near the boundary.
        let far = BANGKOK_NOON_UTC - 86400 + 3 * 3600;
        assess(far, obs(Some(31.0), Some(WEEK), Some(far)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        let r = assess(t0, obs(Some(33.0), Some(WEEK), Some(t0)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert_eq!(r.accounting_status, AccountingStatus::Uninitialized);
    }

    // --- Monotonicity (correction 4) -------------------------------------------

    #[test]
    fn same_week_increasing_usage_is_valid() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        let r = assess(t0 + 500, obs(Some(36.0), Some(WEEK), Some(t0 + 500)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Open); // 31->36 = 5pp
        assert_eq!(r.consumed_today_pp, Some(5.0));
    }

    #[test]
    fn same_week_decreasing_usage_is_continuity_degraded() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        // Decrease same week -> NOT clamp-to-zero; degraded/UNKNOWN.
        let r = assess(t0 + 500, obs(Some(25.0), Some(WEEK), Some(t0 + 500)), true, &mut st);
        assert_eq!(r.accounting_status, AccountingStatus::ContinuityDegraded);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert!(r.reasons.contains(&"non_monotonic_same_window_discontinuity"));
    }

    #[test]
    fn invalid_percent_is_not_admissible() {
        // Adapter rejects out-of-range / non-finite weekly.
        let bad = facts(Some(150.0), Some(WEEK), Some(BANGKOK_NOON_UTC));
        let a = admit_from_openai_facts(&bad);
        assert_eq!(a.usable, false);
        assert_eq!(a.reason, Some("weekly_used_percent_missing_or_invalid"));
        let nan = facts(Some(f64::NAN), Some(WEEK), Some(BANGKOK_NOON_UTC));
        assert_eq!(admit_from_openai_facts(&nan).usable, false);
    }

    // --- Freshness recovery (correction 5) -------------------------------------

    #[test]
    fn stale_then_fresh_recovers_accounting_within_known_week() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        // Stale observation -> FreshnessGated / UNKNOWN.
        let stale = assess(t0 + 500, obs(Some(40.0), Some(WEEK), Some(t0 + 500)), false, &mut st);
        assert_eq!(stale.accounting_status, AccountingStatus::FreshnessGated);
        // Later FRESH observation, same positively-known week, same day, baseline valid,
        // cumulative monotonic (31 baseline, 41 now) -> recover Ok.
        let fresh = assess(t0 + 1000, obs(Some(41.0), Some(WEEK), Some(t0 + 1000)), true, &mut st);
        assert_eq!(fresh.accounting_status, AccountingStatus::Ok);
        assert_eq!(fresh.tranche_status, TrancheStatus::Open); // 41-31=10 <14
        assert_eq!(fresh.consumed_today_pp, Some(10.0));
    }

    // --- Provider reset within a policy day (G/H) ------------------------------

    #[test]
    fn provider_reset_within_day_does_not_auto_grant_tranche() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        // Same Bangkok day (t0+2000 is still within the same day), but week resets X->Y.
        let r = assess(t0 + 2000, obs(Some(5.0), Some(WEEK + 1), Some(t0 + 2000)), true, &mut st);
        assert_eq!(r.accounting_status, AccountingStatus::ContinuityDegraded);
        assert_eq!(r.tranche_status, TrancheStatus::Unknown);
        assert!(r.reasons.contains(&"provider_weekly_reset_within_policy_day"));
    }

    // --- Tranche Open/Consumed (A/B) -------------------------------------------

    #[test]
    fn fresh_under_14_pp_is_open() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        let r = assess(t0 + 1000, obs(Some(40.0), Some(WEEK), Some(t0 + 1000)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Open);
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::OpenaiOrdinaryPermitted);
        assert_eq!(r.consumed_today_pp, Some(9.0));
    }

    #[test]
    fn fresh_over_14_pp_is_consumed() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        let r = assess(t0 + 1200, obs(Some(50.0), Some(WEEK), Some(t0 + 1200)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Consumed);
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::ConserveOpenaiDeepSeekContinuation);
        assert_eq!(r.consumed_today_pp, Some(19.0));
    }

    // --- Deduplication (correction 9) ------------------------------------------

    #[test]
    fn repeated_unchanged_observation_is_not_a_new_event() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        // Same provider observation repeated (same id) -> no new accounting event (advanced=false).
        let r = assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        assert_eq!(r.advanced, false);
        assert!(r.reasons.contains(&"duplicate_observation_no_new_event"));
    }

    // --- Admissible adapter (correction 7) -------------------------------------

    #[test]
    fn adapter_rejects_stale_and_unhealthy_facts() {
        let stale = OpenAiFacts {
            source_health: FactSourceHealth::Healthy,
            observation_freshness: FactFreshness::Stale,
            ..facts(Some(14.0), Some(WEEK), Some(BANGKOK_NOON_UTC))
        };
        assert_eq!(admit_from_openai_facts(&stale).usable, false);
        let unhealthy = OpenAiFacts {
            source_health: FactSourceHealth::Unavailable,
            observation_freshness: FactFreshness::Fresh,
            ..facts(Some(14.0), Some(WEEK), Some(BANGKOK_NOON_UTC))
        };
        assert_eq!(admit_from_openai_facts(&unhealthy).reason, Some("openai_source_not_healthy"));
    }

    // --- No routing authority / separation (I/J/K/L) ---------------------------

    #[test]
    fn i_policy_changes_do_not_affect_routing() {
        let mut a = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut a);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut a);
        let ra = assess(t0 + 1000, obs(Some(40.0), Some(WEEK), Some(t0 + 1000)), true, &mut a);
        // Policy output is pure observation; the module imports no routing type (structural).
        assert_eq!(ra.tranche_status, TrancheStatus::Open);
        let _ = ra;
    }

    #[test]
    fn j_tranche_does_not_imply_local_exclusion() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        let r = assess(t0 + 1200, obs(Some(60.0), Some(WEEK), Some(t0 + 1200)), true, &mut st);
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::ConserveOpenaiDeepSeekContinuation);
        let json = serde_json::to_value(&r).expect("serialize").to_string().to_lowercase();
        assert!(!json.contains("comfyninja") && !json.contains("htpc"), "no local-host mention");
    }

    #[test]
    fn k_thinking_capability_untouched() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        let r = assess(t0 + 1000, obs(Some(50.0), Some(WEEK), Some(t0 + 1000)), true, &mut st);
        assert_eq!(r.tranche_status, TrancheStatus::Consumed);
        let _ = r; // no capability field exists in the output
    }

    #[test]
    fn l_pinned_intent_not_silently_rewritten() {
        let mut st = OpenAiTrancheState::default();
        let anchor = prior_boundary_anchor();
        assess(anchor, obs(Some(31.0), Some(WEEK), Some(anchor)), true, &mut st);
        let t0 = BANGKOK_NOON_UTC;
        assess(t0, obs(Some(31.0), Some(WEEK), Some(t0)), true, &mut st);
        let r = assess(t0 + 1000, obs(Some(50.0), Some(WEEK), Some(t0 + 1000)), true, &mut st);
        let s = serde_json::to_value(&r).expect("serialize").to_string().to_lowercase();
        assert!(!s.contains("substitut") && !s.contains("deepseek-v4"), "policy must not enforce substitution");
        assert_eq!(r.ordinary_cloud_phase, OrdinaryCloudPhase::ConserveOpenaiDeepSeekContinuation);
    }

    /// POLICY-CONSUME-SHADOW §6 regression: repeated identical UNUSABLE observations must
    /// not report advanced=true multiple times (observation identity is independent of
    /// whether the observation is usable for policy).
    #[test]
    fn repeated_unusable_observation_is_deduped() {
        let mut st = OpenAiTrancheState::default();
        let t0 = BANGKOK_NOON_UTC;
        let obs1 = obs(None, Some(WEEK), Some(t0)); // unusable (missing weekly)
        let r1 = assess(t0, obs1, false, &mut st);
        assert_eq!(r1.advanced, true); // first unusable observation is new
        // Same identical unusable observation polled again -> NOT a new event.
        let r2 = assess(t0, obs1, false, &mut st);
        assert_eq!(r2.advanced, false);
        assert!(r2.reasons.contains(&"duplicate_observation_no_new_event"));
        // A DIFFERENT unusable observation IS a new event.
        let r3 = assess(t0, obs(None, Some(WEEK), Some(t0 + 1)), false, &mut st);
        assert_eq!(r3.advanced, true);
    }

}
