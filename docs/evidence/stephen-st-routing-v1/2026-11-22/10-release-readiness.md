# ST routing v1 — release readiness

Prepared 2026-11-22. This document is a handoff, not a deployment. Agent Stack #35 is the
sole production writer; nothing here writes to production.

## 1. Candidate and deployed identity

| | deployed (reference) | candidate |
|---|---|---|
| binary sha256 | `7952200d6583170b15a943fde445c9682ce5905d9fe6dfac8a41f29f367f0d5e` | `f2fd9866b1ad597ef3f13bb0b0ea9e03c42706a63bfd842ae3107ca491ee1b50` |
| config sha256 | `53fd9a8443322ccc7a61e51e91d0506bcc4445ef44710260ac6131607bb4ba95` | `f66bbdb2cd28dc853a1a15cd7c885410c6dd89a7de970cfbf6845971603ff3c9` |
| config bytes | 106,754 | 106,506 |
| source | release `20261122T055957Z-7952200d6583170b-53fd9a84` | branch head `bc86c58` |
| MainPID at capture | 458498, `active` | — |

Deployment is accepted on the binary and config digests, not on a source-commit attestation.

## 2. The config requires the new binary

Measured: the deployed binary refuses the candidate config.

```
[routes.ST_Free]
unknown variant `peer`, expected `declared` or `measured`
rc=1
```

The reverse holds — the candidate binary accepts the deployed config (`rr_dry_run`, evidence
`06-dry-run-f2fd9866.txt`, `rc=0` for both). So binary and config must land together; a config
written ahead of the binary would leave the unit unable to boot, and a binary written ahead of
the config is harmless.

Safe ordering inside one release window: install the binary, then the config, then one restart.
Do not restart between the two writes.

## 3. Config delta

`targets` 53 → 52, `routes` 60 → 59, `llm_clients` 11 → 11 (unchanged).

### Removals (intentional, 1 target and 1 route)

| object | why |
|---|---|
| target `stephen-openrouter-free` | `openrouter/openrouter-free` was retired upstream; the target could only ever fail |
| route `direct-openrouter-stephen-openrouter-free` | its only target is the removed target |

Both are deliberate removals, not drift. A verifier run against the wrong baseline reports them
as `changed=[...]` (evidence `07-reverification.txt`).

### Target changes (9)

Seven free members gain a positive free declaration and pool membership; the two paid members
gain paid-group membership only.

| target | provider model id | Δ |
|---|---|---|
| `stephen-dots-3-note-preview-free` | `dots-studio/dots-3-note-preview:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-ernie-5_5-flash-preview-free` | `baidu/ernie-5.5-flash-preview:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-flint-v1-0-free` | `recursive/flint-v1.0:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-gemma-4-26b-a4b-it-free` | `google/gemma-4-26b-a4b-it:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-gemma-4-31b-it-free` | `google/gemma-4-31b-it:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-ling-3_2-flash-free` | `inclusionai/ling-3.2-flash:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-nemotron-3-5-super-vl-free` | `nvidia/nemotron-3.5-super-vl:free` | `+free_tier=true`, `+groups=[ST_OR_Free]` |
| `stephen-paid-deepseek-v4-1-flash-1017` | `deepseek/deepseek-v4.1-flash-1017` | `+groups=[ST_OR_Paid]` |
| `stephen-paid-glm-5-3-flash` | `z-ai/glm-5.3-flash` | `+groups=[ST_OR_Paid]` |

`ST_OR_Free` has 7 members (the six non-primary candidates plus the primary target). The pool
gate requires membership to equal the callable set — candidates plus the primary target — so the
pool and the candidate list cannot drift apart.

### Route changes (4)

`ST_Free` becomes a peer pool:

| field | deployed | candidate |
|---|---|---|
| `selection` | `declared` (implicit, single target preference) | `peer` |
| `pool_groups` | — | `["ST_OR_Free"]` |
| `candidates` | ling, nemotron, `stephen-openrouter-free` | ling, nemotron, flint, dots, gemma-26b, gemma-31b |
| `target` | ernie | ernie (unchanged) |

Three direct leaves correct their advertised window to the provider's published value:
`…stephen-ernie…` 262,144 → 1,024,000; `…stephen-ling…` 262,144 → 1,048,576;
`…stephen-nemotron…` 262,144 → 1,000,000. These routes are pass-through, so the number is an
advertisement the caller plans against.

### Unchanged, and deliberately so

- `smart_gemini` is byte-identical between deployed and candidate (7 candidates, `measured`).
- Every Gemini target, every Gemini client, and the Gemini credentials are untouched.
- No route gains or loses a paid leg; `ST_Paid` is unchanged.

## 4. Behavioural change in the binary

Two source commits since the deployed release.

`8cc75bd` — one provider-neutral free pool ranked by declared facts; truthful direct leaves.
`9c3777c` — a single free-scope predicate, so the load gate and the request gate cannot disagree.

`1fdae7b` — credential exhaustion is scoped to the target that failed:

- `Route::execute_with_blocks` threads credential blocks into the verdict closure, so a refused
  target appears in the selection trace instead of vanishing.
- A composite child is skipped only when *none* of its targets remain servable; a partial survivor
  is executed with the blocks that survive.
- `call_first_available` continues to a candidate on a different credential when the failure is
  credential-scoped (401, account-wide quota, or a Google recovery that stops the candidate list).
  Request and policy refusals still stop.
- Google model-scoped `DailyQuota` maps to `ModelQuota` and retires only that target; project-scoped
  maps to `WholeQuota`.
- `crosses_to_paid()` is false for `ModelQuota`, so `cross_on = "credential_block"` fails closed on
  a per-model budget instead of spending the paid leg.

`bc86c58` — the Gemini Free Suite source (catalogue, two account profiles, renderer, parity and
mutation harness). It changes no served route.

## 5. Verification

| gate | result | evidence |
|---|---|---|
| batch A verifier (deployed → batch A) | 25 checks, 460 object judgements, 0 failed | `07-reverification.txt` |
| batch B verifier (batch A → candidate) | 27 checks, 0 failed | `07-reverification.txt` |
| no-live-regression verifier | 6 checks, 0 failed | `07-reverification.txt`, `no-live-regression.txt` |
| full suite (`llm-client`, `runner`, `server`) | passed 932, failed 0, ignored 0, 31 binaries | `07b-full-suite.txt` |
| `mixed_credential.rs` | 7 passed, 0 failed | `07b-full-suite.txt` |
| mixed-credential mutation controls | 6 mutations, 6 caught; restore control clean | `08-mixed-credential-mutants.json` |
| Gemini parity checks P1–P8 | 8 checks, 0 failures, both directions | `05-gemini-suite-parity.md` |
| owner profile vs deployed config | 0 failures | `07-reverification.txt` |
| parity mutation controls | 6 mutations, 6 caught; restore control clean | `09-gemini-parity-mutants.json` |
| dry run, candidate binary, both configs, four EnvironmentFiles | `rc=0`, `server OK`, production MainPID and digests unchanged | `06-dry-run-f2fd9866.txt` |

The dry run used the unit's own environment files — `providers.env`, `switchyard-openai.env`,
`gemini.env`, `stephen.env` — so a credential that is present for the unit is present for the run.
`--dry-run` validates and exits; it binds no port and does not touch the running service.

## 6. Rollback

Roll back both halves; a half rollback is the one state that cannot boot.

- binary: restore `7952200d…`
- config: restore `53fd9a84…`
- one restart, then assert `MainPID` changed and `systemctl is-active` = `active`

The deployed release `20261122T055957Z-7952200d6583170b-53fd9a84` is the last-known-good anchor.
Rolling the binary back without the config leaves `selection = "peer"` in the file, which the old
binary rejects at load — the same failure as §2.

## 7. Intentional exceptions in the receipt

1. `stephen-openrouter-free` and `direct-openrouter-stephen-openrouter-free` are deleted on
   purpose (§3). A baseline-relative verifier reports them as changes.
2. The `smart_gemini` comment block is stale about `3.6-flash` and `3.8-flash` (it reads as if
   they are unserved). Both are served. The comment is recorded here and left as-is: editing it
   would move a live byte for a cosmetic reason.
3. `ST_Gemini_Free` (Stephen) is staged, not served: the suite renders it, `P4` refuses a route
   for an unqualified member, and the rendered file stays empty until a serving receipt exists
   under that account. Shipping it now would spend the first request after every deploy
   discovering the account cannot answer.
4. Privacy gates for ST* routes remain deferred to after the first live deployment per the
   charter (`fee9e73` §0D) and the owner's decision on Agent Stack #58. `payload_refused()`
   returning false is owner-accepted. Do not reinstate and do not hold the release for it.

## 8. Post-deploy assertions for the writing workstream

Read-only, in this order:

1. `sha256sum /opt/switchyard/bin/switchyard-server` = `f2fd9866…`
2. `sha256sum /etc/switchyard/routes.toml` = `f66bbdb2…`
3. `systemctl is-active switchyard-neutral-green` = `active`, `MainPID` changed
4. startup line lists `ST_Free` and omits `direct-openrouter-stephen-openrouter-free`
5. no credential or provider error in the journal after the restart — the four environment files
   are unchanged by this release, so a missing credential here means the swap, not the config
6. one `ST_Free` request: the response names one of the six pool members rather than always the
   primary target, and the selection trace shows a peer ranking decision

If 1 or 2 disagrees, stop — the release partially applied. Do not reload a mismatched pair.

## 9. Residual risk

- `selection = "peer"` is new to production routing. The ranking is deterministic and
  declaration-driven, but the first live requests are the first time it meets real provider
  behaviour on this fleet. Watch `ST_Free` selection traces for the first few requests.
- The pool now contains six members instead of three. A member that fails a qualification check
  is retired at request time by the credential scoping in §4; that path is covered by tests and
  mutation controls but has never run against these six providers live.
- The removed target was already unusable, so no capability is lost — but any caller pinned to
  `direct-openrouter-stephen-openrouter-free` by name will now get an unknown-route error.
