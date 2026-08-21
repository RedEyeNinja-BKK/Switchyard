# REALIGN-1B - Authority-Safe Candidate Set + Fallback Contract PROOF (DISPOSABLE)

**Status: ARCHITECTURAL PROOF ONLY. NON-PRODUCTION. DO NOT BUILD/MAKE LIVE.**

## What this branch proves (on PURE stock upstream `c7beccd4`)
1. **CONFIRMED stock gap 1:** `FallThrough::fallbacks(selected)` = **ALL static targets
   minus selected** - so a target rejected by dynamic facts/policy/qualification at
   primary-selection time **CAN appear in the native terminal fallback list** and be tried
   by `libsy-llm-client::call_first_available`. (Test
   `confirms_stock_fallback_escapes_unavailable_target` asserts the leak is present.)
2. **CONFIRMED stock gap 2:** `DefaultTarget` is non-abstaining -> a cascade with all
   fact-admissible targets removed **fails OPEN** to DefaultTarget's target, NOT a
   no-eligible-target result. (Test `confirms_stock_defaulttarget_fails_open`.)
3. **Authority-safe pattern proven:** compute the admissible set from the fact plane, build
   a stock `FallThrough` with `targets = admissible set`, omit `DefaultTarget` for the
   no-eligible case. Terminal fallbacks are then admissible-only; empty -> explicit error.
   All scenarios A-H PASS.

## Scenario coverage (synthetic fixtures, no real provider)
- A pin: exact target, EMPTY fallback (no luna/deepseek/local substitution)
- B thinking envelope: non-thinking never primary/routing-call/fallback
- C non-thinking envelope: thinking never primary/fallback
- D factually-unavailable target never primary OR fallback
- E policy-disallowed target never reappears via fallback
- F local qualified+ready participates; ready-but-unqualified / unqualified-but-ready excluded
      (readiness kept distinct from permanent ineligibility)
- G no eligible target -> explicit no-eligible-target, no DefaultTarget escape
- H (mandatory): primary transport failure -> every attempted terminal fallback stays in
      the admissible candidate set

## Findings
- Stock `FallThrough` is NOT authority-safe for dynamic admissibility BY ITSELF (static
  fallback-set + non-abstaining DefaultTarget). This is a real authority-boundary problem.
- The fix is a SMALL GENERIC pattern (build FallThrough with admissible target set), NOT a
  core rewrite and NOT ResourceRouter v2. No provider names/policy in the extension logic.

## Evidence
- base: `c7beccd4891fa5cfe3a3b94fdd376f5765864507` (pure stock)
- branch: `proof/layer8c-authority-safe-candidates`
- `cargo test -p switchyard-libsy --lib authority_safe_proof` -> **9 passed, 0 failed**
- `cargo test -p switchyard-libsy --lib` -> **268 passed, 0 failed** (no regression)
- Changes are `#[cfg(test)]`-ONLY (single test module appended to fall_through.rs). No
  production code path added or altered.

**DISPOSABLE. Must never be deployed or promoted. Delete branch + prune worktree when done.**
