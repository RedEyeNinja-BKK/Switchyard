# P9 qualification matrix — direct capability declarations, 2026-11-24

## Owner disposition — four SHARED_MODEL_SPEC rows ACCEPTED as model-level evidence (2026-11-24)

**Explicit owner decision:** ACCEPT the four **`SHARED_MODEL_SPEC`** target `tool_calling=true` qualifications on owner routes `direct-google-gemini-3.1-flash-lite`, `direct-google-gemini-3.5-flash-lite`, `direct-google-gemini-3.6-flash`, and `direct-google-gemini-4-hydrogen`. Their direct routes already advertise `tool_calling=true`; each row in `p9-matrix.json` names the same upstream model's `direct-google-stephen-*` successful completed-tool-call `witness_route` through the `openai_chat` wire. This is **evidence of model-level tool support** sufficient for the proposed *declaration* at P9, not a measured tool call with the owner's `google_gemini` credential.

**Preserve the evidence downgrade conspicuously:** leave `evidence_class=SHARED_MODEL_SPEC` and `resolution=DECLARATION_ONLY_SHARED_MODEL_SPEC`, `witness_route` and `evidence_note` unchanged; never rewrite those four records as `LIVE_TOOL_CALL` or same-credential proof. **This is enforced, not merely asked for:** `test_a_shared_spec_row_relabelled_as_a_live_call_is_refused` requires the verifier to refuse a `SHARED_MODEL_SPEC` row rewritten as `LIVE_TOOL_CALL` with `C3_evidence`, because the verifier re-derives the class from the cited receipt instead of trusting the matrix label. This acceptance grants no owner provider quota, key/project identity, paid agent calls, account-level tool access, fallback policy, runtime deployment or release exception. If a later release candidate widens effective route admission or changes more than the narrowly stated targets' metadata, stop and request a fresh owner decision.

**Report the qualification correctly:** **69 independently supported + 4 owner-accepted shared-model-spec = 73 qualified, 5 `BLOCKED_PROVIDER_PROOF` out of 78**. The five must still be proven separately: owner Gemini 3.8, Stephen Gemini 3.8, OpenAI Luna, Luna NT and Sol `.tool_calling`. The chain's strict P9 gate remains **REFUSED (5)**. No paid/quota-consuming probe is authorized by acceptance of the four; do not call this release-ready. Keep this as a source/evidence-only **draft PR #52** and preserve all matrix verifier and source fixture hashes.


Source-only. No provider call, no credential read, no release-store write, no
service restart. This directory answers one question per gate finding: *what
evidence exists, what declaration does it support, and does the declaration chain
actually carry it?*

## What the gate reports

`direct_capability_gate.py` refuses a direct alias that under- or over-declares a
capability against its own target. On the smart-judge three-tier candidate
(`fff8d8da…`, also the manifest implied by PR #51 at HEAD `44f423d`) it reports:

| Population | Direct routes | Findings | Documented upstream gaps |
|---|---|---|---|
| Candidate `fff8d8da…` | 54 | 78 | 28 |
| Live `5211bdc7…` | 53 | 78 | 28 |
| Chain candidate `ccda1bc6…` | 52 | **5** | 28 |

The candidate adds one route and zero findings: the new
`direct-htpc-qwen3_5-9b-mtp-nt` leaf carries no validation gap of its own.

78 = 56 `TARGET_CAPABILITY_UNQUALIFIED` + 22 `ROUTE_CAPABILITY_NOT_EXPLICIT`, and
by capability 35 tool_calling + 42 video + 1 vision.

## How the 78 become 5

The declaration/retirement chain (`build_p9_candidate.py`, five ordered appliers,
each with its own refusals) resolves **73** of the 78. The tally below is the
chain's own gate result, not a hand count:

| Evidence class | Rows | What it is |
|---|---|---|
| `ADMISSION_NARROWING` | 41 | video/vision declared `false` explicitly; the runner already refuses, so admission does not change |
| `LIVE_TOOL_CALL` | 17 | this route answered a real tool call under its own credential |
| `PRIOR_MEASUREMENT` | 6 | the model answered a tool call upstream under this route's credential |
| `SHARED_MODEL_SPEC` | 4 | the model answered under a **different** credential — see release condition R1 |
| `UPSTREAM_RETIRED_404` | 3 | the provider no longer serves the model; the leaf is removed |
| `PROVIDER_REFUSES_TOOLS` | 2 | ThaiLLM answered 400 to any `tools[]` payload; both routes already refuse |
| `UNQUALIFIED` | 5 | the provider was asked and gave an inconclusive answer |

Every row in `p9-matrix.json` carries the receipt it rests on, the exact
declaration proposed, and the class of that evidence. Nothing is inferred from a
green chat completion: a 200 without a tool call, a 429 and a 503 all count as
nothing.

## The five blocked findings

All five are `TARGET_CAPABILITY_UNQUALIFIED <route> <target>.tool_calling`. In each
case the route already advertises `tool_calling = true` and the target declares
nothing, so the gap is un-auditable metadata rather than a live misroute.

| Route | Target → model | Credential | What the probe returned |
|---|---|---|---|
| `direct-google-gemini-3.8-flash` | `direct-google-gemini-3.8-flash` → `gemini-3.8-flash` | `google_gemini` | 429, `daily request allowance for gemini-3.8-flash is spent (20 per provider day)`; the upstream direct probe returns the same free-tier quota error |
| `direct-google-stephen-gemini-3.8-flash` | `stephen-google-gemini-3.8-flash` → `gemini-3.8-flash` | `google_gemini_stephen` | 503, `This model is currently experiencing high demand` |
| `direct-openai-gpt-6-luna` | `openai-gpt-6-luna` → `gpt-6-luna` | `openai` | 429, `The usage limit has been reached` |
| `direct-openai-gpt-6-luna-nt` | `openai-gpt-6-luna-nt` → `gpt-6-luna` | `openai_nt` | 429, `The usage limit has been reached` |
| `direct-openai-gpt-6-sol` | `openai-gpt-6-sol` → `gpt-6-sol` | `openai` | 429, `The usage limit has been reached` |

Two different reasons, and they matter for any future proposal:

* **`gemini-3.8-flash`** is exhausted on *both* credentials — daily allowance on
  the owner account, capacity on Stephen's. No credential has ever answered a tool
  call for this model, so there is no model-level fact to borrow. A probe would have
  to wait for the provider day boundary (midnight America/Los_Angeles) or the
  503 to clear.
* **`gpt-6-luna` and `gpt-6-sol`** hit the account's usage limit. `-luna` and
  `-luna-nt` are the same model on two credential groups, so one window governs
  both.

None of them can be closed from existing evidence. Closing them needs a bounded
per-provider request proposal, which is a separate owner decision.

## Release conditions

These are not blockers on the matrix. They are properties of it that a release
reader must see.

**R1 — four declarations rest on a sibling credential.** `direct-google-gemini-3.1-flash-lite`,
`…-3.5-flash-lite`, `…-3.6-flash` and `…-4-hydrogen` use the owner credential, which
returned its daily-allowance 429. The tool call was measured for the same model id on
Stephen's credential (`google_gemini_stephen`). Each row names its witness route, and
the verifier requires the witness to be the same model id on a different
`llm_client` with a completed tool call in the cited receipt. The capability is a
property of the model; the *credential path* is not proven. **Resolved 2026-11-24 by
owner decision** — the first option, recorded at the top of this file: the four
declarations are accepted as model-level evidence, the `SHARED_MODEL_SPEC` class is
kept, and the four are counted as 4 accepted of the 73. The alternative (converting
them to explicit refusals) was not taken, so those four targets declare
`tool_calling = true` on shared-model grounds and the owner credential path remains
unproven. `test_verify_p9_qualification` refuses any later rewrite of these four as
`LIVE_TOOL_CALL`.

**R2 — two receipts do not name their own model.** `gemma31-20261121.jsonl` carries
no `model` field. Its rows are attributed to `gemma-4-31b-it` because it is the
continuation of `gemma-tools-20261121.jsonl`, whose rows do name that model and end
with `still 503`. The attribution is recorded in
`evidence/tool-qualification-20261123/receipt-annotations.json` as an annotation,
and the affected rows are marked `receipt_attribution: annotated` so a reader can
weight them separately. `gemma-tools-20261121.jsonl` and `gemma31-20261121.jsonl`
were copied into the evidence directory from `/var/tmp on turnstone-vm`, where they
would not have survived; their hashes are unchanged.

**R3 — 28 documented upstream gaps are untouched.** They record that Google's
upstream supports image/video/audio that Switchyard does not admit. Declaring the
flags `false` states the current admission decision; it says nothing about upstream
inability and nothing here changes admission.

## Reproducing

```bash
# rebuild the chain and the matrix from the two pinned manifests
python3 build_p9_matrix.py \
    --baseline /var/lib/switchyard/releases/CURRENT/routes.toml \
    --candidate candidate.toml \
    --work-dir /var/tmp/p9-matrix --out p9-matrix.json

# re-derive every cell from the evidence and the two projections
python3 verify_p9_qualification.py \
    --matrix p9-matrix.json \
    --candidate-declarations candidate-declarations.json \
    --chain-declarations chain-declarations.json \
    --gate-candidate gate-candidate.json --gate-chain gate-chain.json \
    --evidence-dir ../evidence/tool-qualification-20261123

# mutation controls (21)
python3 -m unittest test_verify_p9_qualification -v
```

The chain is byte-reproducible: rebuilding from `5211bdc7` yields
`ccda1bc6…` at every stage. The verifier exits 0 clean, 3 on any disagreement, and
2 when a check examined nothing — a vacuous pass is treated as a defect.

`candidate-declarations.json` and `chain-declarations.json` are projections of the
two manifests containing only direct routes, their targets, the target model id,
`llm_client` and the three admission flags. No base URL, credential name, limit or
header is included, so the matrix can be reviewed without copying either manifest.

## What this is not

Not a release candidate. Not a P9 waiver. Not a claim that either Gemini 3.8 or the
OpenAI leaves support tools. The five findings stay findings until a provider answers,
and any release still needs the binary and manifest as one guarded pair with
separate approval.
