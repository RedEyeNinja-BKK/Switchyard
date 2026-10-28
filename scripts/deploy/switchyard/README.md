# Deployment and operations scripts

Run as root on `switchyard-vm`. Each is a single-purpose tool; none touches
anything outside the paths named below.

## `activate-release.sh` — moved to `scripts/release/`

The release installer and its supporting tools now live in **`scripts/release/`**:

| file | purpose |
| --- | --- |
| `activate-release.sh` | install, roll back, verify, adopt, list releases |
| `capability_contract.py` | generated capability manifest, deployment gate |
| `capabilities.toml` | the reviewed classification the generator reconciles against |
| `release_validate.py` | validates a candidate under the unit's own environment |
| `test_activate_release.py` | the acceptance suite for all of the above |

```bash
# deploy (the positional form below is still accepted)
/opt/switchyard/bin/activate-release.sh deploy <manifest.toml> <manifest-sha256> \
                                               <binary> <binary-sha256> \
                                               --capabilities <capabilities.json> \
                                               --source-commit <sha>
/opt/switchyard/bin/activate-release.sh rollback [<release-id>]
/opt/switchyard/bin/activate-release.sh promote  [<release-id>]
/opt/switchyard/bin/activate-release.sh verify   [<release-id>]
/opt/switchyard/bin/activate-release.sh adopt    --label <name> --basis <text>
/opt/switchyard/bin/activate-release.sh list
```

### Why the previous version was replaced

The installer that lived here copied the previous binary and manifest into a
`mktemp -d` directory and deleted that directory in its `EXIT` trap **on success**.
After a healthy deployment the previous release therefore existed nowhere: a
rollback was not slow, it was impossible. Nothing on the host recorded what had
been deployed, either, so there was no way to reconstruct it afterwards.

Two failure modes came out of that, and the replacement exists to close both:

- **losing the ability to restore a working release.** Every deployment is now
  recorded durably under `/var/lib/switchyard/releases` as a pair of artefacts, and
  `rollback` restores one days later with nothing in `/var/tmp` to rely on.
- **silently deploying a binary that had lost functionality.** Metrics are created
  on first observation, so a build missing a subsystem still starts, still passes
  health, and merely stops exporting. Health checks cannot see it. A capability
  manifest is therefore generated from source, and once the current release carries
  one, every candidate is compared against it before installation.

### Behaviour that matters when something goes wrong

- every artefact is hash-checked before anything is replaced;
- the candidate manifest is validated with the **candidate** binary under the
  unit's own environment — a manifest that only the new binary can parse is caught
  before the swap, not after the restart;
- the candidate is checked against the deployed capability baseline; a candidate
  arriving without its own manifest is refused once a baseline exists;
- a verified rollback target must exist before anything is replaced, and it is
  restored **and re-verified** if any later step fails;
- installation is write-then-rename, so no reader sees a partial file;
- `CURRENT` follows whatever is installed; `LAST-KNOWN-GOOD` moves only by
  `promote`, and `--promote` waits for post-deployment verification, so a release
  whose own checks failed is never installed later as if it were known good;
- a failed post-deployment check restores the previous release **and repoints
  `CURRENT` at it**, so the store never names a release that was rolled back out;
- an artefact marked `RETAINED` is history: retention never prunes it, it does not
  consume a retention slot, and no verb will install, verify or promote it.

Provenance is recorded per release, and *attested* means the source identity was
declared when those exact bytes were installed. It is **not** a claim that the
build is reproducible: no rebuild has been compared against the stored bytes. A
release adopted from what is already running can never be attested.

All paths and commands can be overridden by environment variable
(`LIVE_BIN`, `LIVE_MANIFEST`, `RELEASE_ROOT`, `RESTART_CMD`, `HEALTH_CMD`,
`VALIDATE_CMD`, `CAPABILITY_TOOL`, `ENV_FILE_1`, `ENV_FILE_2`, `MANIFEST_GROUP`).
Those seams exist so the failure paths can be tested without touching production.

`test_release_recovery.py` was removed with the installer it exercised. Its cases
are covered, and in more detail, by `scripts/release/test_activate_release.py`
(26 acceptance tests: deploy, rollback, promote, verify, adopt, retention, the
capability gate, the validation gate, attestation, pointer independence). That
harness also defaulted to a `/var/tmp` scratch path and contained no
pytest-collectable tests, so it never ran in CI.

## `disk-retention.sh`

Watchdog for build-artifact growth. Warns at 85% used, prunes regenerable caches
at 90%, removes the debug build profile at 93%.

```bash
/opt/switchyard/bin/disk-retention.sh --dry-run
```

Never removes source, release builds, production binaries, credentials or
rollback anchors — the protected set is snapshotted before any deletion and
asserted afterwards. Skips the debug-profile removal while a build is running, so
a compile is never truncated mid-write. Runs daily from `disk-retention.timer`.

## Deployment state record - 2026-10-28

Recorded on the host and from the release store after a configuration-only
deployment landed outside a planned change window. It documents state; it changes
no behaviour, and it does not alter any release record.

| | |
| --- | --- |
| CURRENT | `20261028T163707Z-f188c5a9cab31c0b-83c33c93`, label `dualfree-tier-20261028` |
| CURRENT binary | `f188c5a9cab31c0b…` - byte-identical to the release it replaced |
| CURRENT manifest | `83c33c93…`, 37,077 B |
| CURRENT capabilities | `c64c1016…` - identical to the previous release's |
| LAST-KNOWN-GOOD | `20261028T154640Z-f188c5a9cab31c0b-407d8da5`, label `stephen-st-paid-1017` |
| also retained | `20261028T152055Z-f188c5a9cab31c0b-cc1dab83` |

All three releases hold a full binary, manifest and capability manifest, and every
recorded hash matches the bytes on disk (`verify` reports `binary ok`,
`manifest ok`, `service healthy`). The previous configurations are therefore
recoverable by `rollback`, which is a property the pre-`e40c7ff` installer did
not have.

### What the deployment changed

Three routes were added - `ST_Tier1g`, `ST_Tier2g`, `ST_Tier3g` - carrying the same
free model ids as the Stephen tiers but on the global
`SWITCHYARD_OPENROUTER_API_KEY` instead of the Stephen key. `ST_Route` and
`ST_RouteFree` gained those three ahead of `ST_Paid` and `smartlocal`. Routes
47 → 50; targets 45 → 45; composites unchanged in count. No route or target was
removed, and no existing route body changed - the diff against `407d8da5` is three
new blocks plus two `children` lines.

The intent is stated in the config itself: the Stephen credential's free allowance
is an account-wide daily cap, and when it is exhausted every Stephen free model
returns one class of 429, which previously fell straight through to paid `ST_Paid`.
A twin on a separate account lets the composite spend the second free allowance
first. Measured since activation, `switchyard_composition_cross_total` shows 30
crossings `ST_Tier1 → ST_Tier1g` with `reason="account_quota"`, so the change is
both loaded and doing what it was written to do.

### Validation

- The capability gate ran and passed. It could not have failed on inspection: the
  manifest supplied with the candidate was a byte copy of the previous release's,
  which is consistent with an unchanged binary but means the gate compared a
  manifest with itself. The route-level capability flags this config introduces
  (`vision`, `video`) are not represented in the manifest at all.
- The candidate was exercised live before installation in a separate process against
  the production credentials - three requests, one of which crossed into
  `ST_Tier1g` and returned a completion. That is a stronger check than the
  `--dry-run` this repository's `validate` verb performs.
- `validate` itself was **not** run for this candidate: the only
  `.validate-*.log` in the release root is from the 15:46 deploy.
- Post-deployment verification passed. `CURRENT` is repointed before `post_verify`
  runs, and a failure there rolls back, so `CURRENT` still naming the candidate
  plus a live service is the evidence.
- Re-verified read-only afterwards: `verify-live` PASS, every one of the 16
  `always_present` families exposed, health 200, `NRestarts=0`.

### Route capability findings

Checked against the provider's own catalogue (`GET https://openrouter.ai/api/v1/models`,
read-only metadata, no inference) and against the target declarations in the
config. The route flags are not the same kind of claim, which matters here:

- `vision = true` is a promise about **every** target the route can select.
- `video = true` is a promise that **some** target can serve video; the loader
  requires at least one child that declares it. For dispatch, the per-target fact
  is `no_video = target.video != Some(true)`, so a target either declares
  `video = true` or is video-incapable - omitting it is not a refusal at load
  time, but it does exclude the target from a video request.

| route | declared | target(s) upstream | provider input modalities | verdict |
| --- | --- | --- | --- | --- |
| `ST_Tier1g` | `vision=true` `video=true` | `dots-studio/dots-3-note-preview:free` (primary), `nvidia/nemotron-3.5-super-vl:free` (candidate) | `text,image` / `text,image,video` | correct on both; video routes to the eligible candidate |
| `ST_Tier2g` | `vision=false` | `recursive/flint-v1.0:free`, `xingchen-agi/xing4.0-29b-a4b:free` | `text` / `text` | correct, and conservative |
| `ST_Tier3g` | `vision=true` `video=true` | `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free` | `image,text,video` / `image,text,video` | correct on both |

`ST_Tier1g` advertises video while its **primary** target explicitly carries
`video = false` and its input is text and image only at the provider. That is a
narrower claim than "the primary cannot read it": the candidate selection filter
in `Route::execute` removes every target that cannot serve the request *before any
provider is contacted* (`no_video = target.video != Some(true)`, applied per
target), so the video-incapable primary is not called, not called-and-failed. A
video request selects the video-capable candidate as its first and only upstream
attempt.

Measured on the deployed binary `f188c5a9…` (loopback mock provider, scratch
configuration, no credential, no external request), with a primary declaring
`video = false` and a candidate declaring `video = true` on a route declaring
`video = true`:

| request | upstream calls | which model | caller status |
| --- | --- | --- | --- |
| video | **1** | the video-capable candidate | 200 |
| image (control) | 1 | the primary | 200 |
| video, route with no eligible candidate | **0** | none | 400 `unsupported_capability` |
| video, whole-route composite whose first child is ineligible | **1** | the eligible child | 200 |
| image, same composite (control) | 1 | the first child | 200 |

The image controls matter: they show the route still prefers its primary for
ordinary traffic, so the video case's exclusion is the capability filter working,
not a candidate list that never dispatches to the primary. The composite case
shows the same filter applied per child by composition, which skips the
ineligible child and continues.

So `video = true` on `ST_Tier1g` is correct and intentional, and no change to
`ST_Tier1g` or `ST_Tier1` is warranted: selecting the eligible video candidate is
the designed behaviour. An earlier revision of this record claimed the primary
would be called and fail into fallback, and proposed setting `video = false` on
those routes; that was an untested inference from the config alone and is
withdrawn. The `openai_responses` transport requirement is satisfied as well, so
no silent video-to-text conversion is possible on this client.

### Provenance: four separate claims

`release.json` records one field for source identity. Four different questions are
being asked of it here, and only some answers are true as written.

**Declared source commit - `7372e79`, branch `econ-restore-20261028`.** This is what
the deploying command passed with `--source-commit`. It is a declaration made at
deploy time, which the tool labels `declared-at-deploy` and reports as
`attested: true`. It is not evidence about how the bytes were produced.

**Actual source identity at build time - not the declared commit, and not fully
recoverable.** The binary cannot have been built from `7372e79`, because the commit
did not exist when the binary was made:

| event | time (UTC) |
| --- | --- |
| binary built | 13:42:44 |
| binary installed | 13:49:53 |
| `7372e79` committed | 13:55:28 |

The build artifact survives at `/var/tmp/econ-merge/target/release/switchyard-server`
and is byte-identical to the deployed binary (`f188c5a9…`, 27,582,360 B), which dates
the build to 13:42:44Z independently of the commit log. The change `7372e79` records
is absent from its parent `688f5c2` and present in `7372e79` itself, so at 13:42:44
it existed only in an uncommitted working tree. The build therefore came from a
dirty tree that was committed thirteen minutes later. Whether that tree carried
**other** uncommitted differences cannot now be established: the working tree has
since moved on and no record of its state at build time survives. That question is
recorded as unresolved rather than answered by inference.

**Behavioural functionality in the deployed binary - present, measured.** Commit
orders cannot settle this, so it was measured. The exact deployed bytes were run
against a local mock provider on a loopback port, in a scratch directory, with a
scratch configuration and no credential: no production configuration was touched
and no external request was made. Each case sends one caller request and counts the
upstream attempts the binary actually performs:

| mock response | upstream attempts | caller status |
| --- | --- | --- |
| 200 with a valid completion (control) | 1 | 200 |
| 500 `error_type="authentication"` | **1** | 500 |
| 500 `error_type="permission_denied"` | **1** | 500 |
| 500 `error_type="payment_required"` | **1** | 500 |
| 503 `error_type="authentication"` | **1** | 503 |
| 500 bare `server_error` (control) | 3 | 500 |
| 500 `error_type="rate_limit_exceeded"` (control) | 3 | 500 |

That is exactly the rule `7372e79` adds - a 5xx carrying a masked permanent
`error_type` fails fast instead of being retried - together with the controls that
commit preserves, where a bare or unrelated 5xx still retries. The fix's behaviour
**is** in the deployed binary even though its commit postdates the build. An earlier
reading of this release concluded the opposite from timestamps alone; that inference
was wrong, because a commit that does not exist yet says nothing about what a
working tree contained.

**Origin of the route configuration - staged on the host, in no commit.** The three
new routes exist in no commit on any ref (`git log --all -S ST_Tier1g` is empty).
The file was authored as `/tmp/routes_dualfree2.toml` on `switchyard-vm` and copied
to `/opt/switchyard/staged/routes-dualfree-20261028.toml` immediately before the
deploy, with the capability manifest copied from `CURRENT` in the same command.

**What the record should say.** The `config-only-dualfree-tiers` basis is accurate
for the route change. The declared commit is accurate as a declaration and
misleading as provenance: it names a commit the binary cannot have been built from,
and says nothing about where the route change came from. The accurate summary is:
*binary and config from an uncommitted working tree at 13:42:44Z; routes staged on
the host and present in no commit; declared commit `7372e79` records the binary's
request-retry behaviour and was committed after the build.*

Nothing here retroactively attests the release. The `attested: true` flag keeps its
narrow, declared meaning.

### Authorization

Not established. The deployment ran as
`activate-release.sh deploy /opt/switchyard/staged/routes-dualfree-20261028.toml …`
from an authenticated `turnstone-mgmt` SSH session originating on the Turnstone
management host, minutes after that same host staged the file and copied the
capability manifest. The release store, the staged documents and the CI workstream
carry no approval, ticket or operator instruction for it.

Resolving it needs one of: the initiating session's transcript, an operator
confirmation, or the change record that authorized `dualfree-tier-20261028`.

The deployment is healthy and is left in place. Reversing it would be a production
change with its own risk, and doing so on an unknown authorization would compound
one unknown with another.

### Open items

- The authorization gap above.
- `CURRENT` is attested only in the declared sense; `LAST-KNOWN-GOOD` is not
  attested at all, so the rollback target rests on observed bytes alone.
- The build-time source state is unresolved beyond what the surviving artifact and
the deployed binary's behaviour prove; it may have contained other uncommitted
  changes that no longer exist anywhere.
- `ST_Tier1g` (and the pre-existing `ST_Tier1`) advertise video with a
  video-incapable primary. Verified correct: the candidate filter excludes the
  primary before dispatch, so the eligible candidate is selected first. No change
  needed; see the capability findings above.
- Route-level `vision` and `video` are not represented in the capability manifest,
  so the release gate cannot notice a wrong modality claim. That remains true even
  though these particular claims are right.
