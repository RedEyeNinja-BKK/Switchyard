# Batch F2 — `smart-nt` authorized terminal leg

Date: 2026-11-22. Branch `feat/st-routing-v1-on-r6-20261122`.

## Contract implemented

The converged charter (`1deca29`) requires `smart-nt` to rank a provider-neutral primary
pool (Gemini + Luna) and reach DeepSeek Flash NT only when the primary pool is *provably*
unavailable. Before this batch, `deepseek-deepseek-flash-nt` was an ordinary candidate:
it could serve whenever it happened to rank first, and a single 429 on a Gemini leg could
hand the request to a paid leg while six healthy free legs were still ranked.

The terminal leg is now a separate model set on the route, never merged into the candidate
list:

* `Route::terminal: Option<Arc<RuntimeModels>>` — set from `terminal_fallback` at load;
* `authorize_terminal(capable, dispatchable, attempts, credential_of)` — the whole policy,
  one free function;
* `SelectionTrace::{terminal, terminal_crossed, terminal_reason}` — the decision record
  names the leg, whether it was crossed and which claim authorized it.

## Authorization decision table

| condition | result |
|---|---|
| no candidate could serve the request (capability/context) | refuse — an unsupported request is not "exhausted" |
| every capable leg excluded by a credential block | `primary_pool_blocked` |
| a dispatched capable leg failed on anything other than capacity (`403`, `400`, `empty_answer`, `timeout`, `context_window`) | refuse |
| a capable leg was never attempted and its credential was not proved dead in this run | refuse |
| every dispatched capable leg failed on capacity, every unattempted capable leg explained by a `401`/`402`/`429` on its credential | `primary_pool_exhausted` |

Two gates make this expressible and are deliberately separate: `capability_gate` asks
"could this target serve at all", `verdict` adds "and is it allowed right now". A pool whose
every *capable* leg is merely blocked is exhausted; a pool where no leg is capable is a
request nobody can serve.

## Defects found while wiring it

1. **Terminal targets were invisible to route plumbing.** `routing_target_names()` delegates
   to the algorithm, which knows only `target` and `candidates`. The terminal target
   therefore had no client constructed, no capability facts, no `target_credentials` entry
   and no daily request allowance — it would have been dispatched outside the admission and
   block machinery that protects every ranked leg. Fixed with
   `RouteConfig::served_target_names()` (routing names + terminal names), now used for
   facts, credentials, allowances, prompt policy, locality and client construction.
2. **The decision record is a second file.** Selection decisions are appended to
   `routing.decisions.jsonl` (`path.with_extension`), not `routing.jsonl`. A reader of the
   attempts file sees `terminal: null` on a request that did cross, which reads as
   "the route has no terminal leg".
3. **A credential-scoped stop did not skip the dead credential's siblings.** A pool lists
   one provider's models together, so stopping at "the next candidate" re-called the same
   spent credential. `call_first_available` now advances to the first later candidate on a
   *different* credential.

## Tests

Unit (`route::terminal_authorization_tests`, 9): nothing capable keys, blocked pool,
capacity failure with a dead credential, unattempted leg on a live credential, five
request-fault outcomes, mixed run with one request fault, unattributable leg, every leg
attempted and capacity-failed, attempt on a non-capable target.

Integration (`crates/switchyard-server/tests/smart_nt_terminal.rs`, 9): exhausted pool
credential serves from the terminal leg and its sibling is not called; policy refusal never
crosses; model-scoped quota served by the sibling; healthy pool never touches the terminal
leg; a request the terminal leg cannot hold is not sent; terminal-also-candidate, unknown
terminal target and empty terminal list are load errors; the decision log records the leg,
the crossing and the reason.

## Manifest (batch F1, staged not deployed)

`work/apply_batch_f1.py` edits the reviewed manifest by line anchor (no TOML
re-serialisation, so the comments recording why each leg is or is not a candidate survive):

* removes `direct-google-gemini-2.5-flash` (held: Google answers 404) from `smart-nt` and
  `smart_gemini`;
* removes `deepseek-deepseek-flash-nt` from `smart-nt` candidates;
* adds `terminal_fallback = ["deepseek-deepseek-flash-nt"]`;
* corrects the `smart_gemini` comment that claimed 3.6/3.8 were absent while the route
  listed them.

`--verify` re-parses with `tomllib`, compares every untouched route and all target records,
asserts the terminal target exists and is not held, checks comment markers, and proves
idempotence by refusing a second application (exit 2).

## Not claimed

Nothing is deployed. The DeepSeek terminal leg has not been exercised against the live
provider; the integration tests use scripted upstreams. Dry-run of the candidate binary
against `routes-batch-f1.toml` with the unit's four EnvironmentFiles is still outstanding,
as is a paired binary+manifest release (the deployed binary rejects the candidate manifest
with `unknown variant 'peer'`). Dashboards are not built.
