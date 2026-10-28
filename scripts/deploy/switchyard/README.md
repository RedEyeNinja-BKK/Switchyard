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

## Continuous integration

Workflow runs arrive through `.github/workflows/ci.yml` on pull requests and on
pushes to `main`. Changes under `scripts/release/` are covered by the release
safety job, which runs the capability-contract tests and the installer acceptance
suite; the suite creates its own sandbox, so it never reads or writes the release
store of a real host.
