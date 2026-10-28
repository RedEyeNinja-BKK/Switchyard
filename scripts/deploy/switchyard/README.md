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

### Provenance caveat

`release.json` declares `source_commit=7372e79` on branch `econ-restore-20261028`
with `attested: true`. That commit does not describe this deployment. It is an
`llm-client` fix, it is not present in the running binary (whose bytes are
unchanged from the previous release, which was built before it), and the route
change exists in no commit at all - the config was authored as
`/tmp/routes_dualfree2.toml` on the host and staged from there.

The `config-only-dualfree-tiers` basis is accurate. The source identity attached to
it is not, and an operator reading `SOURCE-COMMIT.txt`-style provenance would
conclude the deployed config came from `7372e79`, which it did not. The accurate
record is: config-only, no source commit, staged file authored on `switchyard-vm`.

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
- `ST_Tier1g` and `ST_Tier3g` declare `vision = true` and `video = true`. Modality
  support for those free models was not verified here and is not covered by the
  capability contract.
