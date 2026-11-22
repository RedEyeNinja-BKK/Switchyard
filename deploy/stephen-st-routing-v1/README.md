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
