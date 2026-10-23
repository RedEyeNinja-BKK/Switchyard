# Deployment and operations scripts

Run as root on `switchyard-vm`. Each is a single-purpose tool; none touches
anything outside the paths named below.

## `activate-release.sh`

Installs a release (binary + manifest) and restarts once.

```bash
/opt/switchyard/bin/activate-release.sh <manifest.toml> <manifest-sha256> \
                                        <binary> <binary-sha256>
```

Behaviour that matters when something goes wrong:

- every artefact is hash-checked before anything is replaced;
- the candidate manifest is validated with the **candidate** binary under the
  unit's own environment — a manifest that only the new binary can parse is
  caught before the swap, not after the restart;
- the previous binary and manifest are copied aside first, so "roll back" means
  restoring known-good bytes rather than restarting whatever is on disk;
- installation is write-then-rename, so no reader sees a partial file;
- health is verified after the restart; a failure restores the previous release
  **and restarts onto it**;
- an `EXIT` trap covers the whole window between "files replaced" and "release
  verified", so an interruption cannot leave disk and memory on different
  versions;
- once the release is live, healthy and byte-verified it is authoritative: a
  later bookkeeping error (mirroring files into `staged/`) logs a warning and
  does not revert a working deployment.

All paths and commands can be overridden by environment variable
(`LIVE_BIN`, `LIVE_MANIFEST`, `RESTART_CMD`, `HEALTH_CMD`, `VALIDATE_CMD`,
`ENV_FILE_1`, `ENV_FILE_2`, `MANIFEST_GROUP`). Those seams exist so the failure
paths can be tested without touching production.

## `test_release_recovery.py`

Fault injection for the above, in a throwaway sandbox. Every path and command is
pointed at a temporary directory; production is never referenced.

```bash
sudo python3 /opt/switchyard/bin/test-release-recovery.py
```

Covers: success, an unhealthy candidate, an install failure after the binary was
replaced, a hash mismatch, and a `SIGTERM` delivered mid-restart. Each case
asserts what ends up running, that disk matches the running process, and how many
restarts happened.

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
