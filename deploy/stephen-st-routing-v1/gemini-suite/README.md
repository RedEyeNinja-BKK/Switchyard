# Gemini Free Suite — one catalogue, two account profiles

`smart_gemini` (the owner's account) and `ST_Gemini_Free` (Stephen's account) are two
projections of **one** suite. This directory holds the single source of the shared facts and
the two account overlays, and verifies both directions of the relationship.

## Why it exists

The two Gemini lanes were being developed separately: R6 measured the owner's project
(quirk handling, thinking floors, per-model free-tier rows) while Stephen's free pool was
built from OpenRouter leaves only. Any correction learned on one side would have had to be
re-learned on the other, and the second learning is where drift starts.

The split here is the only one that survives contact with a second account:

| belongs to | because | lives in |
|---|---|---|
| the **model** | native id, window, the wire quirks that make the compatibility surface accept an OpenAI-shaped body, the `<thought>` marker, the thinking floor, a provider retirement | `catalogue.json` |
| the **account** | credential, quota group, serving qualification, daily/minute allowance, pool membership and ordering policy | `profile-owner.json`, `profile-stephen.json` |

A model fact in a profile file is the beginning of a second catalogue, so `verify_parity.py`
check **P8** refuses it.

## Files

```
catalogue.json         shared model facts, captured from a named manifest digest
profile-owner.json     smart_gemini: owner credential, measured selection
profile-stephen.json   ST_Gemini_Free: Stephen credential, peer selection over OR+Gemini
project.py             renders both profiles; --check reproduces the deployed owner facts
verify_parity.py       P1–P8, bidirectional; exit 0 only when all pass
mutants_parity.py      six mutations, each must be reported by the check it targets
rendered/              generated fragments (not applied by anything; review artifacts)
```

## Run

```sh
python3 project.py --check --manifest /path/to/deployed-routes.toml
python3 verify_parity.py --manifest /path/to/deployed-routes.toml \
    --expected-capture-hash 53fd9a84
python3 mutants_parity.py
```

`--check` is the owner-regression half: every fact the owner profile renders must equal the
fact the deployed manifest serves, **and** every deployed field must be known to the
catalogue. That second direction is the one that loses capability silently.

## What parity means concretely

* **P1** every chat target deployed for the owner is catalogued, and every catalogue model is
  projected by both profiles.
* **P2** each shared fact renders identically in both profiles, including the provider's own
  retirement reason for the two 404 models.
* **P3** the account facts differ and never cross: no credential env or quota group is
  shared, and neither profile's credential appears in the other's projection.
* **P4** an unqualified member cannot enter a served pool — and qualifying one member adds
  exactly that member. The gate is a flag, not a rewrite.
* **P5** the owner keeps measured selection; Stephen keeps peer selection over OR+Gemini.
  Neither is copied onto the other, because they answer different questions.
* **P6** the owner projection still reproduces the deployed manifest.
* **P7** the catalogue names the bytes it was captured from and they are the bytes supplied.
* **P8** no profile restates a model fact.

## Deliberate differences (not drift)

| | owner `smart_gemini` | Stephen `ST_Gemini_Free` |
|---|---|---|
| credential | `SWITCHYARD_GEMINI_API_KEY` | `SWITCHYARD_GEMINI_STEPHEN_API_KEY` |
| quota group | `google-ai-studio-free` | `google-ai-studio-free-stephen` |
| ordering | `selection = "measured"` over one credential | `selection = "peer"` beside Stephen's OpenRouter free leaves |
| membership | the seven deployed leaves; the two 404s held | same seven, all `unverified` until a serving receipt exists |
| deployment | live since R6 | staged; joins `ST_Free` when qualified |

## What is deliberately NOT done here

* **No new adapters, endpoints or discovery.** The R6 implementation is reused; this records
  its facts rather than re-deriving them.
* **No qualification by deployment.** Stephen's Gemini members are `unverified`: the key is
  a root-only systemd credential, so a serving receipt needs one root-run measurement.
  Deploying them before that would spend the first request after every release learning that
  the leg cannot answer. `render_stephen_staged` therefore emits no route block while no
  member is servable.
* **No privacy gate.** The ST\* privacy policy is deferred by owner decision until after the
  first live deployment; nothing here adds or implies one.
* **No production write.** This directory generates and verifies text. Applying it is a
  release, coordinated by the Agent Stack release workstream, not by this tool.
