# Handoff — ST_Route 4/5/3 (execute first), then Particle-1, then retry staging

2026-11-28. Live production state at handoff: release `20261128T040916Z-6a8209f97101563b-fc05b640`,
manifest `fc05b640…52817`, binary `6a8209f9…22e82`, MainPID 880721. Nothing was changed this session.

## What was measured (live /etc/switchyard/routes.toml, read-only)

| Object | Current | Required |
|---|---|---|
| `routes.ST_Free` | passthrough, target `stephen-ernie-5_5-flash-preview-free`, candidates: nemotron, flint, dots, gemma-26b, gemma-31b, google-gemini-3.5-flash-lite (7 legs) | pool = ST_OR_Free (4) + ST_Gemini_Free (5) = 9 legs |
| `routes.ST_Gemini_Free` | passthrough to `stephen-google-gemini-3.5-flash-lite`, no candidates (1 leg) | 5 native Stephen Google leaves (below) |
| `routes.ST_Paid` | target `stephen-paid-deepseek-v4-1-flash-1017` + candidate glm-5.3-flash (2 legs) | 3: deepseek-1017, hy4-flash, glm-5.3-flash |
| `composites.ST_Route` | children `["ST_Free","ST_Paid"]`, cross_on credential_block, ctx 262144 | unchanged (this shape is already correct) |

Owner-specified membership (Issue #35 comment 7234683890):
- **ST_OR_Free (4)**: `direct-openrouter-stephen-ernie-5_5-flash-preview-free`,
  `…-nemotron-3-5-super-vl-free`, `…-flint-v1-0-free`, `…-dots-3-note-preview-free`.
- **ST_Gemini_Free (5, native Stephen Google AI Studio)**: `direct-google-stephen-gemini-4-hydrogen`,
  `…-3.8-flash`, `…-3.6-flash`, `…-3.5-flash-lite`, `…-3.1-flash-lite`.
- **ST_OR_Paid (3)**: `direct-openrouter-stephen-deepseek-v4-1-flash-1017`,
  `…-hy4-flash` (upstream `tencent/hy4-flash`), `…-glm-5-3-flash`.

## Exact change list

1. **Remove** `groups = ["ST_OR_Free"]` membership from `targets.stephen-gemma-4-26b-a4b-it-free` and
   `targets.stephen-gemma-4-31b-it-free`. Keep their direct aliases, routes and any unrelated group.
   Remove their flattened references from `routes.ST_Free`.
2. **Add** `groups = ["ST_Gemini_Free"]` to the five native Stephen Google targets (create the targets
   if absent, on the Stephen Gemini client/credential — not the owner key), and add flattened
   references for all five to `routes.ST_Free` in diagram order (hydrogen, 3.8, 3.6, 3.5-lite, 3.1-lite).
   Reconcile `routes.ST_Gemini_Free` so its own effective expansion is exactly those five.
3. **Add** Stephen `hy4-flash` only if the alias/target is absent: Tencent `tencent/hy4-flash`, on the
   existing Stephen OpenRouter paid scope/client. Verify the exact upstream id from the OpenRouter
   models API (no paid probe). If it cannot be verified, leave it out and record the shortfall.
4. `routes.ST_Free` candidates = 4 OR leaves + 5 Gemini leaves; `routes.ST_Paid` = 3 leaves.
5. Expected direct/advertised census 53/72 → 54/73 only if hy4 is added; measure the real registry
   separately rather than asserting the number.

## Gates before activation (all required by the owner)

- Semantic diff of the exact TOML change, reviewed before applying.
- Candidate provenance for every leaf; strict P9 against the **actual direct inventory**.
- `--dry-run` against the live config with the unit's env files (the validator needs them).
- Protected predecessor: copy the current manifest, record its sha256, and prove the rollback restores
  those exact bytes.
- Account/route checks: every added target resolves to the Stephen credential only; owner
  `smart-route` must remain 3/2/1/2 (8 candidates); Ultra direct-only; no Lite readmission there;
  Particle is not a thirteenth leaf.
- Activate only through the existing managed config-only release mechanism, then verify live:
  `ST_Route` still composite ST_Free → ST_Paid, ctx 262144, and each pool's leg count is 9 / 3.
- Reconcile dashboards (V7 generator + published revision), Turnstone registry and the ledger.

## Then

- **Objective 1**: check Stephen's own OpenRouter allowance (`/api/v1/key` with
  `SWITCHYARD_STEPHEN_OPENROUTER_API_KEY`), then run the pinned Particle-1 fixtures from
  `docs/evidence/p11/qualification/fixtures` with `verify.py`. Stop on real exhaustion and record the
  next window; no paid fallback, no owner-account inference, no admission.
- **Objective 3**: the staging assertions still open, in this order — upstream 429 header forwarding,
  5 s cooldown 503 timing, actual SDK request counts with `max_retries=0`, cancellation during backoff,
  autonomous multi-window completion, build provenance for `6a8209f9` (capsule hint `3d478f5b`).
  Run staging **on the host with access**, in one self-contained script that starts stub + candidate,
  asserts, captures, and tears both down. Do not delegate host-dependent steps to task-agent sandboxes
  (they have no `/etc/switchyard` and no SSH to switchyard-vm).

## SDK scoping check (requested)

`e78ed9d` sets `max_retries=0` at `turnstone/core/providers/__init__.py:265-266`, which is the single
client construction for chat completions; audio already passes `max_retries=0` explicitly and the
registry clients use `with_options(max_retries=0)`. Confirm by `git show e78ed9d` that the diff is that
one construction and that no other provider path lost its retry behaviour.
