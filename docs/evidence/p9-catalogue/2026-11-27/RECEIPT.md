# P9 catalogue — receipt

Purpose: answer three questions about the **deployed** P9 manifest from one place, without
sampling traffic — which targets are shared between routes, which Turnstone aliases actually
consume this router, and where a consumer's context ceiling differs from the route's.

## What was measured, from what

| input | identity | source |
|---|---|---|
| deployed manifest | `e887392c586b86cd4b75b14b6df5fd820ae6bc60a79c8bb7daab63d830c4c4d9`, 131153 B | release store, release `20261126T084503Z-860adbc9af98ae6a-e887392c` |
| release record | binary `860adbc9…b7ad0`, source commit `bd62792`, branch `integrate/p4-on-640a946` | `release.json` + `PROVENANCE.txt` of the same release |
| capability contract | `a8059a1a9f67c6b1059ed1e3f97876ff716b41fa52952def5e34d35232d69097` | same release |
| `/v1/models` | 73 entries | live read from the running process at the same time |
| Turnstone registry | 91 rows, 91 enabled | read-only query of `model_definitions` + `system_settings` on the node; `api_key` never selected |

`routes.canonical.toml` in this directory is the projection of that manifest by
`scripts/canonical_manifest_projection.py` — `api_key_env` → `credential_present`, `base_url`
→ `endpoint_class`, comments dropped, everything else preserved, and the source SHA-256
recorded in `[projection]`. The catalogue was generated twice, once from the deployed bytes
and once from this projection; **every count is identical**, which is what makes the
projection usable as the checked-in copy of the deployed state.

No traffic was sampled. Every cell below comes from one of those documents.

## Measured

| fact | value |
|---|---|
| routes / advertised routes | 73 / 73 |
| direct routes | 54 |
| targets | 59 |
| composites / decision lanes / lane compositions | 3 / 4 / 2 |
| targets shared between owners | **30** |
| targets used by exactly one owner | **29** *(CORRECTED — was published as 36; see below)* |
| targets referenced by nothing | 0 |
| targets reached only by a decision lane | 5 |
| registry rows addressed to this router | 69 |
| registry rows addressed elsewhere | 22 |
| rows addressed here whose model name is declared nowhere | 0 |
| advertised routes with no Turnstone consumer | 15 |
| capabilities whose `target` names no declared capability client | 0 of 21 |

The 69 rows addressed here divide as 58 routes, 2 composites, 4 decision lanes, 2 lane
compositions and 3 capabilities.

### CORRECTION — the single-use count was published as 36; it is 29

The first publication reported 36 single-use entries inside a set of 59 declared targets, which
cannot both be true: 59 was the right total and 36 was the wrong class.

Cause: composite children were folded into the same reverse map the routes use. A composite names
**routes**, so seven route names (`ST_Free`, `ST_Paid`, `direct-deepseek-deepseek-flash-nt`,
`direct-openai-gpt-6-luna-nt`, `direct-openrouter-glm-5_3-flash`,
`direct-openrouter-openrouter-free`, `vc-paid-core`) were recorded as if they were targets and
counted as single-use. Two further composite children (`direct-google-gemini-3.1-flash-lite`,
`direct-google-gemini-3.5-flash-lite`) share a name with a declared target, so they did not invent
an entry — they inflated an existing target's owner count instead. Nine composite-child references
in total: seven phantom entries plus two merged.

Fix: a reference is now resolved against the declared `[targets.*]` set before it is recorded. A
name that does not resolve is kept under the section that made it, in `non_target_references`, and
never becomes a target. The three classes are asserted to be a partition
(`targets_partition_holds`, exit 2 on failure), so `30 + 29 + 0 == 59` is a checked fact rather than
a coincidence of three independent filters.

The corrected split is **30 shared + 29 single-use + 0 unreferenced = 59**. The nine composite
children are listed in the catalogue's *References that are not targets* section, resolved to
`route`, so a reader can see the call path without it entering a cardinality claim.

The 22 rows addressed elsewhere divide as 13 OpenRouter, 4 private engines, 3 OpenAI, 1 DeepSeek
and 1 ThaiLLM — they are registry rows, not consumers of this manifest, and the catalogue does not
credit them to it.

Sixteen rows name an **upstream model id** (`gpt-6-luna`, `qwen/qwen3.8-flash`,
`comfyninja/qwen3.8-27b-q4-mtp`, …) rather than a route id; all sixteen address a provider
directly. That is why a join on `model = route id` alone reports them as "aliases without a
route": they are not missing routes, they are rows that never address this router. The
destination check is what separates the two.

### Consumer ceilings that disagree with the route

Five route/consumer pairs differ, and in all five the consumer is **stricter** — the request
never reaches the router, so the safe direction, but the route's advertised window is not what
that alias can use:

| route | route window | alias ceiling | delta |
|---|---|---|---|
| `smart` | 1000000 | 262144 | −737856 |
| `smart_gemini` | 1048576 | 262144 | −786432 |
| `direct-openrouter-stephen-ling-3_2-flash-free` | 1048576 | 262144 | −786432 |
| `direct-openrouter-stephen-ernie-5_5-flash-preview-free` | 1024000 | 262144 | −761856 |
| `direct-openrouter-stephen-nemotron-3-5-super-vl-free` | 1000000 | 262144 | −737856 |

## Reproduction

```bash
# one command proves the classifier on a fixture whose every cell has one right answer
scripts/canonical_catalogue.py --self-test

scripts/canonical_manifest_projection.py <deployed routes.toml> --out routes.canonical.toml
scripts/canonical_catalogue.py routes.canonical.toml \
    --capabilities capabilities.json --models v1-models.json \
    --registry registry-effective.json --out-dir out
```

The self-test asserts nine independent answers (name kind for route / composite / capability /
upstream id / undeclared, destination split, the addressed-here-names-nothing set, unreferenced
targets, lane-only targets, capability resolution, route-with-consumer). Three mutations —
destination always `this_router`, capability dropped from the name index, capability resolution
always empty — each make it fail, so the fixture is known to bite rather than merely pass.

## Not established here

- Vendor model cards, transport proofs and traffic grades stay on the master direct panel; this
  catalogue makes no claim about them.
- A route's declared capability is an admission gate, not evidence the upstream accepts the
  modality.
- The registry is read at one instant on one node. Another node with its own registry is not
  represented, and the file records the instant rather than claiming liveness.
