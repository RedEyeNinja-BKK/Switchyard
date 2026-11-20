# Gemini free-project capability inventory — 2026-11-21

This directory holds the offline result of the owner instruction of 2026-11-21:
enumerate and preserve every direct capability contract the free AI Studio
project allocates, keep zero-quota rows, never represent a typed operation as a
chat completion, and stage the route graph without activating anything.

Nothing here contacts Google, restarts a service, changes a credential, or
touches the deployed manifest. The deployed release
`20261120T181049Z-a5b9dffc7f198716-f0546288` is unchanged.

## Source and reproduction

Input is the owner-attested crosswalk, copied verbatim:

```
input/owner-free-project-all-47-capability-crosswalk.json
sha256 fd8acaf767c5265c1991f10047e035ac4258734b8609908ac58ce6a612ae7abd
```

```sh
python3 scripts/deploy/switchyard/gemini_capability_inventory.py \
  --crosswalk docs/evidence/gemini-r2/2026-11-21/input/owner-free-project-all-47-capability-crosswalk.json \
  --out-dir   docs/evidence/gemini-r2/2026-11-21

python3 scripts/deploy/switchyard/gemini_capability_verify.py \
  --inventory docs/evidence/gemini-r2/2026-11-21/direct-google-capability-inventory.json \
  --graph     docs/evidence/gemini-r2/2026-11-21/capability-route-graph.json \
  --fragment  docs/evidence/gemini-r2/2026-11-21/gemini-r2-routes.staged.toml \
  --crosswalk docs/evidence/gemini-r2/2026-11-21/input/owner-free-project-all-47-capability-crosswalk.json \
  --json      docs/evidence/gemini-r2/2026-11-21/verify-report.json

python3 scripts/deploy/switchyard/test_gemini_capability_inventory.py
```

Measured 2026-11-21: generator `models=47 routable=2 held=45 features=23`,
verifier `18/18 checks passed -- PASS`, suite `16 passed, 0 failed`.

## Three questions, kept apart

| question | answered by | consequence |
|---|---|---|
| Does the project allocate quota for this row? | the crosswalk, transcribed verbatim | a row is retained, including zero-RPD rows |
| Can Switchyard serve this row's operation? | the crosswalk's `operation_contract` | only `chat.complete` maps to a chat leaf |
| Is the model proven on the wire? | `direct_contract_is_executable` | only a proven model may appear in a route |

A row can be allocated, chat-shaped, and still unqualified. That is the state of
seven rows here, and it is why the inventory has three states for a held chat
model rather than one "unavailable".

## Inventory result

47 models and 23 feature quotas are enumerated. Every displayed peak/limit pair
is carried unchanged (`INV2`), and no native model id is rewritten (`INV3`).

| state | count | meaning |
|---|---:|---|
| `EXECUTABLE_AND_DEPLOYED` | 2 | chat leaf with a proven native id and an allowance |
| `HELD_WIRE_UNQUALIFIED` | 7 | chat leaf, cached native id, never called |
| `HELD_NATIVE_ID_UNRESOLVED` | 4 | chat-shaped, no native id in the evidence |
| `HELD_ZERO_RPD_ALLOWANCE_UNSHOWN` | 4 | chat-shaped, no allowance shown — retained, not deleted |
| `HELD_NON_CHAT_TRANSPORT` | 30 | a typed operation with its own wire shape |

The two routable rows are `direct-google-gemini-3.5-flash-lite` and
`direct-google-gemini-3.1-flash-lite` — the same two leaves the deployed release
already carries. **No new executable contract exists in this evidence set.**

The 30 non-chat rows are not "missing features". They are Live sessions, TTS,
ASR, embeddings, typed decision, image, music, robotics, world and multimodal
generation, and managed agents. Each carries a concrete
`adapter_requirement` string naming what would have to be built, because
"adapter required" on its own reads as a promise.

The 23 feature rows are tool quotas (grounding, search) and are recorded with
`is_model_direct_route: false` and `routing_role:
HELD_TOOL_QUOTA_NOT_A_MODEL_ENDPOINT`. A feature is not an endpoint, and a
grounding allowance cannot be spent as a chat request.

## Route graph

Primitives are filled in order of specificity and each may claim only candidates
no earlier primitive claimed, so one provider-model quota is never reachable
through two legs (`QTA1`). A composite child must be a route and must not itself
be a composite (`GRP2`).

| primitive | executable today | context | status |
|---|---|---|---|
| `smart-gemini-bounded` | none | 16 384 | HELD — no small-budget member is wire-qualified |
| `smart-gemini-chat` | 3.5-flash-lite, 3.1-flash-lite | 1 048 576 | ready, but gated by the composite |
| `smart-gemini-long-context` | none | 1 048 576 | HELD — every member already claimed |
| `smart-gemini-vision` | none | — | HELD — the deployed leaves declare `vision = false` |
| `smart-gemini-antigravity` | none | — | HELD — `agent.run` has no adapter |
| `smart-gemini-live` | none | — | HELD — `session.live` has no adapter |

`smart_gemini` needs two legs before reparenting the live composite buys
anything. Today it would have one, so the fragment is emitted **held**: applying
`smart-gemini-chat` alone would give the two Lite models a second route beside
the live composite and change nothing a caller can observe.

## Gemma 4 and the 16K per-minute budget

Gemma 4 26B and 31B each show 14 400 RPD with a 16 000 TPM input budget. A
256 000-token prompt cannot be served inside a 16 000-token minute, so those
models are qualified for small bounded agent tasks and are excluded from the
long-context primitive by construction: they are claimed by
`smart-gemini-bounded` first, and the long-context primitive can only take what
is left. They are not wire-qualified yet, so both primitives that could hold them
are held.

## The private/unknown branch

The deployed release already drops a Google leg for an uncleared request before
dispatch: `Route::execute` computes admission once per model and
`requires_public_payload` candidates are ineligible when clearance is refused. A
composite whose first leg is a Google leaf therefore falls through to its next
leg rather than sending the payload to Google.

What the graph adds is the requirement that the fall-through legs are strict
non-training targets. That is a configuration property of the composite, not new
machinery, and it can only be expressed once a second Google primitive exists —
which is why no composite change is proposed here.

## Release proposal

**No manifest change is proposed.** The honest delta from this evidence set is
zero routes: ten chat-capable contracts are held on wire qualification or on a
missing native id, and thirty are typed operations. Emitting any of them would
either promise a capability no target can serve or create a second route to a
quota the deployment already reaches.

The fragment in this directory is the reviewable form of that statement. Its
header reads `apply today: no` and it exits with that verdict from the generator,
not from a comment a future edit could drift away from.

First release that changes anything requires one of:

1. a wire qualification run for a second chat leaf (which also releases
   `smart-gemini-bounded` and the composite reparent), or
2. an adapter for a typed operation, delivered as its own change.

## Not done, and why

Two items from the instruction are not satisfied by this directory, and are
recorded rather than implied:

- **Per-target input/context, output reserve, RPM/TPM/RPD on the target.** The
  manifest schema has `daily_request_limit` and route-level `context_window` and
  `escalation_max_input_tokens`; it has no per-target input ceiling, output
  reserve, or per-minute rate. Declaring fields the binary does not read would be
  a configuration that silently does nothing, so the classification here records
  the numbers and the qualification they imply, and the enforcement change is
  left to a change that also tests it.
- **An explicit skip-Gemini composite.** As above, the egress gate already
  produces the skip; what remains is a second Google primitive for the composite
  to sit beside, which does not exist yet.
