# Stephen ST routing — Batch A

Batch A of the owner's [ST routing v1 charter](../../../docs/plans/stephen-st-routing-v1-implementation-charter-20261122.md):
make every Stephen direct leaf tell the truth, and remove the one dependency on a model the
provider chooses at request time.

The production route manifest is not stored here — it names internal hosts. Batch A therefore
ships as a deterministic edit applied to the deployed bytes, plus the evidence needed to judge
the result. Nothing in this directory writes to production.

## What changes

| Edit | Object | From | To | Evidence |
|---|---|---|---|---|
| Context truth | `direct-openrouter-stephen-ernie-5_5-flash-preview-free` | 262144 | 1024000 | OpenRouter catalogue, 2026-11-22 |
| Context truth | `direct-openrouter-stephen-ling-3_2-flash-free` | 262144 | 1048576 | same |
| Context truth | `direct-openrouter-stephen-nemotron-3-5-super-vl-free` | 262144 | 1000000 | same |
| Remove dynamic selector | target `stephen-openrouter-free` + leaf `direct-openrouter-stephen-openrouter-free` | present | absent | 0 routing-log references in 30 days |
| Explicit free pool | `ST_Free` candidates | Ling, Nemotron, `openrouter/free` | Ling, Nemotron, Flint, Dots, Gemma 26B, Gemma 31B | every leg is a `:free` slug |
| Truthful tariff | seven free leaves | `free_tier` unset | `free_tier = true` | provider price 0 on each id |

The three context corrections do **not** raise `ST_Free` (262144) or `ST_Route` (262144): those
are policy promises to callers, and widening them is a separate admission decision. The point
of the correction is that a direct caller of a leaf no longer sees a policy cap presented as a
provider limit.

## Files

| File | Purpose |
|---|---|
| `apply_batch_a.py` | anchored editor; refuses if an anchor is missing or doubled; asserts its own scope |
| `verify_batch_a.py` | 25 structural checks over baseline, candidate, truth table and provider facts |
| `mutants_batch_a.py` | 12 mutants + 3 controls; each must turn the verifier red on a named check |
| `stephen-st-truth-table.json` | per-leaf facts with a grade and a provenance line |
| `stephen-st-truth-table.md` | the same table rendered for review |
| `or-model-facts-20261122.json` | captured public catalogue subset (no credential, no inference call) |

## Run

```sh
python3 apply_batch_a.py --baseline routes.toml --out candidate.toml --diff
python3 verify_batch_a.py --baseline routes.toml --candidate candidate.toml \
    --truth stephen-st-truth-table.json --facts or-model-facts-20261122.json
python3 mutants_batch_a.py  --baseline routes.toml --candidate candidate.toml \
    --truth stephen-st-truth-table.json --facts or-model-facts-20261122.json
```

Exit codes: `0` pass, `1` a check failed, `2` bad input or a refused edit, `3` the verifier
judged nothing. The verifier never reports success on an empty comparison.

## Re-anchoring

The production manifest moves whenever the owner's Gemini suite releases. The applier is
anchored on structure, not on a whole-file hash, so it re-applies to a new baseline — but the
scope assertions run against that new baseline, and any drift outside the six declared objects
aborts the run. Re-run the verifier and the dry-run after every rebase; do not reuse a stored
candidate file.

## Engine change that travels with this batch

`Route::all_targets_free` (`crates/switchyard-runner/src/route.rs`) now asks
`switchyard_llm_client::is_free_scope`, the same predicate a failing call uses, instead of
testing for a `:free` suffix. Before the change a free-only route that carried the provider's
Free Models Router (`openrouter/free`) looked as if it had a paid leg, so a composite entered
it after the account-wide free allowance was gone and spent one guaranteed-rejected call per
request. Batch A removes the router from the Stephen scope, so this matters for the remaining
owner-side route; the predicate change makes the two levels agree rather than fixing one call
site.

## Status

Source, tests, candidate and dry-run pass. No production write, no restart and no reload were
performed. The release remains gated on the Issue #35 release owner's non-overlapping slot.

---

# Batch B - one provider-neutral free pool

Batch B stacks on Batch A. It answers the second half of the charter: `ST_Free` stops being a
hand-ordered list of legs and becomes a pool whose membership is a fact declared once per leaf.

## What changes

| Edit | Object | From | To | Evidence |
|---|---|---|---|---|
| Pool membership | 7 free Stephen leaves | no group | `groups = ["ST_OR_Free"]` | every member is a `:free` slug |
| Paid membership | 2 metered Stephen leaves | no group | `groups = ["ST_OR_Paid"]` | both billed |
| Selection policy | route `ST_Free` | declaration order | `selection = "peer"` | equal fitness breaks ties by canonical id |
| Pool declaration | route `ST_Free` | candidate list only | `pool_groups = ["ST_OR_Free"]` | membership stated where it is checked |

`ST_Gemini_Free` is deliberately **absent**: no qualified Stephen Gemini chat target exists on
this credential, so naming the group would be a phantom pool member that validates yet can never
be served. Adding it later is a two-line manifest change and no code change (check B12 pins this).

## Engine change that travels with Batch B

Three additions, no new selection engine:

- `Tariff { Free, Metered, Unknown }` replaces the boolean `free_tier` fact. The third state is
  what keeps an undeclared target from being counted as free capacity.
- `RankingPolicy` gives the pool a stated tie-break (canonical id) and an honest treatment of
  unknown headroom (demoted, not promoted). `RankingPolicy::LEGACY` preserves declaration order
  for every existing route, so nothing outside the pool changes rank.
- A pool route filters non-free members before ranking (`not_free_entitled`), so a mislabelled
  leg cannot serve as a paid escalation one tier below the pool.

`Tariff::is_free()` and `Tariff::draws_on_free_allowance(target)` are two questions and are kept
separate. The first is economic and strict (`Unknown` is never free); the second is the provider's
own allowance scoping and falls back to `is_free_scope` for `Unknown`. Collapsing them regresses
`a_free_scope_quota_blocks_a_later_free_only_child_on_the_same_credential`.

## Files

| File | Purpose |
|---|---|
| `apply_batch_b.py` | applies Batch A, then the group labels and pool declaration; idempotency-guarded |
| `verify_batch_b.py` | 27 checks; B17-B20 guard structure, B1-B13 guard membership |
| `verify_no_live_regression.py` | anchored on the DEPLOYED manifest; refuses any deployed object the candidate drops or edits outside a named allowlist |
| `mutants_live_regression.py` | 4 regression mutants + a round-trip fidelity gate + an unmutated control |

## Run

```sh
python3 apply_batch_b.py --baseline routes.toml --out batch-b.toml --diff
python3 verify_batch_a.py --baseline routes.toml --candidate <batch-a.toml> \
    --truth stephen-st-truth-table.json --facts or-model-facts-20261122.json
python3 verify_batch_b.py --baseline <batch-a.toml> --candidate batch-b.toml
python3 verify_no_live_regression.py --live routes.toml --candidate batch-b.toml
python3 mutants_live_regression.py routes.toml batch-b.toml
```

`apply_batch_b.py` applies Batch A internally, so its `--baseline` is the **deployed** manifest,
not the Batch A output. Passing a Batch A output is refused by the idempotency guard.

## Rebase onto R6 (2026-11-22)

The batches were first generated against `53fd9a84`'s predecessor. R6 landed the music and
robotics lanes on the same manifest while this work was in progress, so the stack was regenerated
against the deployed bytes and the commits were replayed onto `97768e1` (R6 tip). `94bcd30..97768e1`
touches no crate source, so the rebase cannot change the binary's behaviour. Proof that the base
is production: the R6 receipt names binary `7952200d…` and manifest `53fd9a84…`, and both are the
hashes measured on the host.

A candidate binary built before the rebase could not even parse the deployed manifest
(`unknown field max_session_audio_bytes`). That failure is why `verify_no_live_regression.py`
exists: a batch verifier that only compares its own two files cannot see a section another
workstream added to production after the snapshot was taken.
