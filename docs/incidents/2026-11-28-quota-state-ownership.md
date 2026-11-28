# 2026-11-28 — service start refused on root-owned quota counters

## What happened

At 18:26:48Z a release was deployed (`20261128T182648Z-6a8209f97101563b-02587e3e`).
The swap appeared to fail: the unit would not start, and `activate-release.sh`
reported `ROLLBACK INCOMPLETE`.

Two daily-request counters in `/var/lib/switchyard/quota` were owned by
`root:root` instead of `switchyard:switchyard`:

```
google-ai-studio-free-stephen-gemini-3.6-flash-debc4ea0d8bfc3e6.json
google-ai-studio-free-stephen-gemini-3.1-flash-lite-2233d11d876696ce.json
```

Every start attempt failed with:

```
the daily request counter ... cannot be opened read-write by this process:
Permission denied (os error 13)
```

Repair: `chown switchyard:switchyard` on the two files, `systemctl reset-failed`,
start. Backups of the pre-repair bytes were kept under
`/var/tmp/quota-ownerfix-20261129/`.

`ROLLBACK INCOMPLETE` was a **verification** failure, not a partial restore — the
release bytes had in fact been restored. The banner is produced by the rollback
path and was not itself a second defect, but it cost time and should be read with
that in mind.

## What is proven, and what is not

**Proven:** the candidate dry-run is a real load of the real configuration, so it
reaches the same lazy counter-creation path a real service start does. Anything
the dry-run creates while running as root is `root:root 0644`, and the
unprivileged service then cannot write it. Pre-existing counter ownership was
never checked by the validator.

**Not proven:** that these two specific counters were created by a validation
run on 2026-11-28. Three reproductions against the pre-fix validator — an empty
state directory, a populated directory with the two counters removed, and a
populated directory containing a counter with a stale `day` — created zero files
and did not rewrite the aged counter.

The fix therefore removes the **capability**, and does not rest on the
unestablished attribution. This distinction is deliberate: an incident write-up
that states an unproven cause as measurement is worse than one that says what it
checked.

## Correction

`scripts/release/release_validate.py` gained two independent, always-on
defences. Both fail closed.

1. **Isolation.** The candidate child runs in a private mount namespace
   (`unshare --mount --propagation private`) with a 32 MiB tmpfs mounted over the
   state directory. The directory is taken from the candidate's own
   `[quota].state_dir`, falling back to `/var/lib/switchyard/quota`, so
   validation protects the directory the release will actually use. Anything the
   dry-run creates is discarded with the namespace; the underlying filesystem is
   never written.
2. **Identity.** Counters that already exist are opened read-write as the account
   that owns the state directory — the account the service runs as — before any
   pointer or byte is installed. A counter the service cannot use becomes a
   refusal here instead of an outage after the swap.

If isolation cannot be established the validator refuses with exit `2`. There is
no fallback to running unsandboxed, because the unsandboxed path is the defect.

Exit code `2` now documents as `usage, environment-file or isolation problem`.

## Proof

`/var/tmp/prove_quota_fix.py` on switchyard-vm, run as root against the installed
validator, 21 checks, 0 failures (`proof.json` beside each run):

| Case | What it establishes |
| --- | --- |
| A | Missing counters in an isolated state dir stay missing; nothing created, no ownership change, no root-owned file. The receipt names the isolation and the number of counters opened as the service account. |
| B | A writer aimed at the state dir reaches it when run unsandboxed (control), and does **not** when run through the validator. Control and subject are the same command against the same path. |
| C | A `root:root 0644` counter produces exit `1`, the message names the account and the file, and the refusal precedes the candidate dry-run. |
| D | Correct ownership validates, and the probe really opened every counter as the service account (67 of 67), creating nothing. |
| E | Live production state dir unchanged in names and owners; no root-owned counter. |
| F | Stored CURRENT verifies byte-for-byte; CURRENT and LAST-KNOWN-GOOD both unchanged by the harness; CURRENT carries the live manifest identity. |

B's control matters: without it, a leak that never reached the path would look
like successful isolation.

Installed and re-proven against the live manifest:

```
P9 PASS direct=55 qualified=55 failures=0 manifest_sha256=02587e3e...
quota: isolated tmpfs over /var/lib/switchyard/quota;
       71 existing counter(s) opened read-write as switchyard(999):switchyard(988)
```

Production census identical before and after (71 entries), zero root-owned.

Installed validator sha256 `19670d34cec74be5975287aa63c7b8d8c7166d1ffa83144a4a487f7a330a883d`.
Pre-fix bytes preserved at `/var/tmp/relvalidate-backup-20261129T0100Z.py`
(sha256 `247508fe7bac2191d16f47a30dad3f0041e1950d5509baeb079a072061b656e1`).

No allowance was reset and the service was not restarted to install this change:
the validator is not part of the running process.

## Residual risk

- The identity probe only sees counters named after the candidate's own quota
  groups. A counter for a group absent from the candidate is neither checked nor
  protected — but it is also not on the candidate's write path.
- Isolation depends on `unshare` being usable. If it is not, releases stop; that
  is the intended failure direction.
- The rollback banner still reports `ROLLBACK INCOMPLETE` when only verification
  failed. Deferred: it needs a distinction between "bytes restored" and "service
  verified healthy", which is a separate change to `activate-release.sh`.
