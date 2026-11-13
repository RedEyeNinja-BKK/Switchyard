# Production Switchyard manifest (switchyard-vm)

The production Switchyard instance on `switchyard-vm` reads its routing manifest from
`/etc/switchyard/routes.toml`; `activate-release.sh` installs that file from the release store
and the running unit (`switchyard-neutral-green.service`) reads it there.

A routing change used to be reviewable only by reading the host, so the reviewed form was
committed here. The manifest also carries private deployment topology — private-network
hostnames, the addresses and ports of internal services — which does not belong in a public
repository. The resolution is a split, and this directory holds the public half:

| | where it lives | what it is |
|---|---|---|
| canonical manifest | host only, root-owned release store | the bytes that are deployed |
| `routes.redacted.toml` | this directory | its faithful projection, endpoints replaced |
| `manifest_contract.py` | this directory | decides whether a manifest is the reviewed one |
| `sanitize_manifest.py` | this directory | produces the projection, and proves it |

**The redacted copies are not deployable.** Every private endpoint in them is a reserved
placeholder, so deploying one would route to nothing; the file says so in its first line,
because it sits in the same directory as the deploy commands. It exists to be read and
diffed. Everything below the four-line banner is the manifest.

The manifest contains no credentials at all: it names environment variables
(`api_key_env = "SWITCHYARD_…_API_KEY"`), never their values. What is redacted is topology.

## Provenance

Two manifests are published, because activating the routing change safely takes two steps
(see [Why activation is two-phase](#why-activation-is-two-phase)).

| | candidate | recovery |
|---|---|---|
| derived from live manifest | `858590db2522870983d1d814ef891454078cd550f62463c8a359d944930981a6` | live manifest, then the candidate's removals |
| deployed bytes (host only) | `9627a389f8b5abbcb4af99302579e0e3ece4d56e38c5ef92d877da3296719ccc` | `505c8dc632601059f196b281d69054fa4866709735a63c0503a5b21fc8f392d6` |
| redacted projection here | `routes.redacted.toml` `c8785fb763771c246d9faf6c20b15e1a3ca97ab33eb87b81a3cea1f070e21ea7` | `routes.recovery.redacted.toml` `4a556bb2b20e5ccf6e9a8bae843d2bae0643c5229bc72794ad21bc667d9412a0` |
| routes / targets | 42 / 36 | 38 / 34 |
| deployed binary (unchanged) | `eceabe3a6f7bcbd215d1fc9d4b1eae1e4abce659b5f99d1e76abb5dd1a92df6d` | same |
| source commit of that binary | `1a21604fb2b1e1c85c0bc30c8e8905fda01f83c8` | same |
| derived at | 2026-11-13 | 2026-11-13 |

The live manifest was 46 routes and 40 targets. Nothing outside the sections listed under
[The change](#the-change) differs, and no target's own fields change at all.

### Checking that the public copy really is the deployed file

Redaction is deterministic, so the projection can be regenerated and compared. The contract
checker does both layers in one call:

```sh
# On the host, against the canonical file: digest AND projection must both match.
python3 scripts/deploy/production/manifest_contract.py /var/lib/switchyard/staging/routes.candidate.toml \
  --profile candidate \
  --sha256 9627a389f8b5abbcb4af99302579e0e3ece4d56e38c5ef92d877da3296719ccc \
  --redacted scripts/deploy/production/routes.redacted.toml
```

```sh
# Anywhere, against the published copy: every routing fact is still assertable, because
# redaction touches endpoints only.
python3 scripts/deploy/production/manifest_contract.py \
  scripts/deploy/production/routes.redacted.toml --profile candidate
```

`sanitize_manifest.py` enforces four properties and refuses to write if any fails: the output
parses and has exactly the input's table/key set, no private coordinate or credential shape
survives anywhere in the text (including comments), redaction is a fixed point (running it on
its own output changes nothing), and it is deterministic. `--print-mapping` shows what was
rewritten.

## The change

Local inference is served by Unsloth Studio on `comfyninja`, whose profile registry now holds
exactly two profiles, both backed by the `UD-Q4_K_M` trunk:

- `qwen3.8-27b-q4-mtp` — text, verified context 179 712
- `qwen3.8-27b-q4-mtp-vl` — vision with `mmproj-F16`, verified context 152 832

The deployed manifest still names four profiles that no longer exist. That is not cosmetic: a
request naming a retired profile makes Studio re-download the deleted model, so a stale route
is an active hazard rather than dead configuration.

| section | change |
|---|---|
| `targets.comfyninja-qwen3_8-27b-q3` | removed (profile retired) |
| `targets.comfyninja-qwen3_8-27b-q3-mtp` | removed (profile retired) |
| `targets.comfyninja-qwen3_8-27b-q3-mtp-vl` | removed (profile retired) |
| `targets.comfyninja-qwen3_8-27b-q4` | removed (standalone `Q4_0` retired) |
| `routes.direct-comfyninja-{q3,q3-mtp,q3-mtp-vl,q4}` | removed with their targets |
| `routes.direct-comfyninja-…-q4-mtp` | `context_window` 188 416 → **179 712** |
| `routes.direct-comfyninja-…-q4-mtp-vl` | `context_window` 163 840 → **152 832** |
| `routes.smartlocal` | `candidates` and `escalation_target` → `comfyninja-qwen3_8-27b-q4-mtp-vl`; `context_window` 225 280 → **81 920** |
| `routes.smartlocal-bounded` | same retarget and window |

That is the whole delta: 8 route fields changed, 4 targets and 4 routes removed, nothing
added, no target field touched. Deliberately unchanged: the `htpc-qwen3_5-9b-mtp` primary on
both `smartlocal` routes, `escalation_max_input_tokens = 61440`, every reasoning policy,
timeout, admission rule, vision/tool flag, and every other route and target.

The local embedding and rerank services (`embedding-local` on 8448, `reranker-local` on 8449)
are separate processes on the same host, not Studio profiles. They are untouched, and the
contract checker fails if either disappears.

### Where the two context numbers come from

They are the largest contexts that hold under the 97 % VRAM ceiling *while serving*, not at
idle. On a 24 576 MiB card the ceiling is 23 838.72 MiB. Allocation steps up about
20 MiB on the first served request — 40 MiB on the vision lane, which retains image buffers —
and then plateaus:

| profile | idle | served | margin |
|---|---|---|---|
| text 179 712 | 23 806 | 23 826 | +12.72 |
| vision 152 832 | 23 790 | 23 830 | +8.72 (real image prompt) |

Text 180 224 and vision 153 344 / 153 856 read comfortably at idle but exceed the ceiling once
a request has been served, so they are rejected. A context accepted from idle sampling is not
accepted.

## Why activation is two-phase

`activate-release.sh` has two restore paths, and they read different pointers:

| path | target |
|---|---|
| automatic restore when post-activation health fails | the release that was `CURRENT` before the deploy |
| `activate-release.sh rollback` | `LAST-KNOWN-GOOD` |

Both currently resolve to the same release, `20261111T130253Z-eceabe3a6f7bcbd2-858590db`,
whose manifest is the live `858590db…` — the one that still names the four deleted profiles.
Deploying the candidate directly would therefore make the *safety net* the hazard: any failed
activation, and any later manual rollback, would restore a manifest whose first request can
trigger a re-download of a deleted model.

So the pointers are fixed first, with a manifest that contains no local lane at all:

1. deploy `routes.recovery.toml` — every `comfyninja` target, both `direct-comfyninja`
   routes, `smartlocal`/`smartlocal-bounded` and the `comfyninja` client are removed;
   `tier2` and `tier2-bounded` fall back to `smartfree` alone, which is the provider lane and
   needs nothing local. HTPC direct routes stay: they are uninvolved small models;
2. `promote` that release to `LAST-KNOWN-GOOD`, which the script does not do on deploy;
3. deploy `routes.candidate.toml`.

After step 2 both restore paths point at a Q3-free manifest, so step 3 — and any later
rollback — is safe. This ordering was verified rather than assumed, with a failure-injection
harness on the host that drives the real `activate-release.sh`: deploy → promote → deploy with
an induced health failure → automatic restore → manual rollback → redeploy. It asserts 22
properties, including that the automatically restored manifest is byte-identical to the
recovery manifest and mentions no retired profile. The negative control omits the `promote`
step and the manual rollback reintroduces twelve retired identifiers, so the harness fails
when the fix is removed rather than passing for the wrong reason.

## Deploying

Config-only: the binary is unchanged, so each step is one manifest swap and one restart.

Stage root-owned, so nothing under a user-writable path is read as root, and confirm the hash:

```sh
sudo -n install -m 0640 -o root -g switchyard routes.recovery.toml \
  /var/lib/switchyard/staging/routes.recovery.toml
sudo -n install -m 0640 -o root -g switchyard routes.candidate.toml \
  /var/lib/switchyard/staging/routes.candidate.toml
sudo -n sha256sum /var/lib/switchyard/staging/routes.{recovery,candidate}.toml
```

Validate each with the deployed binary before asking for a swap. This needs the unit's own
environment, because `api_key_env` is resolved at config load:

```sh
sudo -n bash -c 'set -a
  . /etc/switchyard/secrets/providers.env
  . /var/lib/switchyard/oauth/switchyard-openai.env
  set +a
  /opt/switchyard/bin/switchyard-server \
    --config /var/lib/switchyard/staging/routes.candidate.toml --dry-run'
```

Then the two deploys and the promotion between them:

```sh
sudo -n /opt/switchyard/bin/activate-release.sh deploy \
  /var/lib/switchyard/staging/routes.recovery.toml \
  505c8dc632601059f196b281d69054fa4866709735a63c0503a5b21fc8f392d6 \
  /opt/switchyard/bin/switchyard-server \
  eceabe3a6f7bcbd215d1fc9d4b1eae1e4abce659b5f99d1e76abb5dd1a92df6d \
  --capabilities /var/lib/switchyard/releases/CURRENT/capabilities.json \
  --label q3-quarantine-20261113 \
  --source-commit 1a21604fb2b1e1c85c0bc30c8e8905fda01f83c8 \
  --source-branch HEAD \
  --basis "remove every lane that names a Studio profile deleted from the ComfyNinja registry, so no restore path can reintroduce one"

sudo -n /opt/switchyard/bin/activate-release.sh promote

sudo -n /opt/switchyard/bin/activate-release.sh deploy \
  /var/lib/switchyard/staging/routes.candidate.toml \
  9627a389f8b5abbcb4af99302579e0e3ece4d56e38c5ef92d877da3296719ccc \
  /opt/switchyard/bin/switchyard-server \
  eceabe3a6f7bcbd215d1fc9d4b1eae1e4abce659b5f99d1e76abb5dd1a92df6d \
  --capabilities /var/lib/switchyard/releases/CURRENT/capabilities.json \
  --label comfyninja-km-routes-20261113 \
  --source-commit 1a21604fb2b1e1c85c0bc30c8e8905fda01f83c8 \
  --source-branch HEAD \
  --basis "retire four ComfyNinja profiles that no longer exist in the Studio registry and retarget smartlocal R2 to the retained Q4-MTP-VL profile; direct-route contexts corrected to served-peak verified values"
```

`--capabilities` may reuse the deployed file only because the binary is byte-identical; a
rebuilt binary requires a capability contract generated from that build.

Exit codes: `0` activated and healthy, `1` refused before changing anything, `3` activation
failed with the previous release restored and verified, `4` failed and the restore could not be
verified (manual attention).

Two restarts, not one: the quarantine step restarts the service, and so does the candidate.
That is the cost of not letting the rollback target be the hazard.

## Rolling back

```sh
sudo -n /opt/switchyard/bin/activate-release.sh rollback
```

Defaults to `LAST-KNOWN-GOOD`. After the promotion above that is the Q3-free recovery release,
so a rollback lands on a manifest with no local lane — not on the pre-change manifest. To
return to the pre-change manifest deliberately, name it:

```sh
sudo -n /opt/switchyard/bin/activate-release.sh rollback 20261111T130253Z-eceabe3a6f7bcbd2-858590db
```

Do not do that while the ComfyNinja profile registry lacks those four profiles; that release is
the reason this change exists.

## Post-deployment acceptance

1. `activate-release.sh verify` — stored bytes and live files agree.
2. `GET /v1/models` advertises 42 routes and no route whose id contains `q3`; the two
   ComfyNinja leaves report `context_length` 179 712 and 152 832; `smartlocal` and
   `smartlocal-bounded` report 81 920.
3. `scripts/release/route_audit.py /etc/switchyard/routes.toml --models-url http://127.0.0.1:4000`
   — manifest and advertisement agree, every leaf advertised, no composite leaked.
4. One request each through `direct-comfyninja-qwen3_8-27b-q4-mtp` and `…-q4-mtp-vl`, and one
   through `smartlocal` small enough to stay on the HTPC leg.
5. Confirm the Studio resident is still `qwen3.8-27b-q4-mtp` afterwards and that VRAM is
   unchanged — a route that resolved to a retired profile would show up as a new download.

## Known limitation: `context_window` does not enforce anything

`context_window` is what `GET /v1/models` advertises. It is not a request limit: a leaf route
dispatches whatever the caller sends and lets the provider reject it. Composite children *are*
skipped when a declared child window cannot hold the request, but a `passthrough` route is not
a composite.

What actually bounds input here:

- `escalation_max_input_tokens = 61440` moves requests above that estimate to the ComfyNinja
  leg before any provider call, so the 8 GiB host is not handed an oversized prompt. The
  estimator is character-based and conservative for these tokenisers.
- The ComfyNinja profile's own context is enforced by llama-server, which returns an error
  above it rather than silently truncating.

81 920 is therefore a provisional advertised figure for `smartlocal`: it is the HTPC leg's
capacity, which is the smallest leg the route can land on, not an enforced ceiling and not a
promise about the ComfyNinja leg's 152 832. A clean local refusal above each candidate's own
ceiling needs the admission logic in the router (`context_shadow` measures it today). Until
that exists, 163 840 must not be advertised anywhere.
