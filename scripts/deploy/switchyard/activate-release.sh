#!/usr/bin/env bash
# Install a release (binary + manifest) with one restart, and restore on failure.
#
# Contract
# --------
#   * Every artefact is hash-checked before anything is replaced.
#   * The candidate manifest is validated with the CANDIDATE binary, under the
#     unit's own environment, before it becomes live.
#   * The previous release is copied aside first, so "roll back" means restoring
#     known-good bytes -- not restarting whatever happens to be on disk.
#   * Installation is a rename, so no reader ever sees a half-written file.
#   * Health is verified after the restart. A failure restores the previous
#     release and restarts again.
#   * Any failure between "files replaced" and "health verified" triggers the same
#     restore from an EXIT trap, so an interrupted run cannot leave the disk and
#     the running process on different versions.
#   * The restore is skipped once the release is verified, so a later, unrelated
#     error cannot revert a good deployment.
#
# Seams for fault-injection tests: every path and command can be overridden by an
# environment variable, and the defaults below are the production values.
#
# Usage: activate-release.sh <manifest.toml> <manifest-sha256> <binary> <binary-sha256>
set -Eeuo pipefail
set +x
umask 027

M_SRC=$1
M_WANT=$2
B_SRC=$3
B_WANT=$4

UNIT=${UNIT:-switchyard-neutral-green.service}
LIVE_MANIFEST=${LIVE_MANIFEST:-/etc/switchyard/routes.toml}
LIVE_BIN=${LIVE_BIN:-/opt/switchyard/bin/switchyard-server}
STAGED_MANIFEST=${STAGED_MANIFEST:-/opt/switchyard/staged/routes_deployed.toml}
STAGED_BIN=${STAGED_BIN:-/opt/switchyard/staged/switchyard-server-candidate}
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:4000/health}
# A file-comparison probe stands in for the HTTP check during fault injection.
HEALTH_CMD=${HEALTH_CMD:-}
RESTART_CMD=${RESTART_CMD:-systemctl restart "$UNIT"}
HEALTH_ATTEMPTS=${HEALTH_ATTEMPTS:-80}
# The two environment files the unit reads, and the validation invocation.
ENV_FILE_1=${ENV_FILE_1:-/etc/switchyard/secrets/providers.env}
ENV_FILE_2=${ENV_FILE_2:-/var/lib/switchyard/oauth/switchyard-openai.env}
VALIDATE_CMD=${VALIDATE_CMD:-}
MANIFEST_GROUP=${MANIFEST_GROUP:-switchyard}

say() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { say "FATAL: $*" >&2; exit 1; }

healthy() {
  if [ -n "$HEALTH_CMD" ]; then
    "$HEALTH_CMD" >/dev/null 2>&1
    return
  fi
  curl -sS -m 2 "$HEALTH_URL" 2>/dev/null | grep -q '"status":"ok"'
}

wait_healthy() {
  local attempt
  for attempt in $(seq 1 "$HEALTH_ATTEMPTS"); do
    healthy && return 0
    sleep 0.5
  done
  return 1
}

# ---------------------------------------------------------------- preconditions
[ "$(id -u)" -eq 0 ] || die "must run as root"
[ -f "$M_SRC" ] || die "missing manifest $M_SRC"
[ -f "$B_SRC" ] || die "missing binary $B_SRC"

[ "$(sha256sum "$M_SRC" | cut -d' ' -f1)" = "$M_WANT" ] || die "manifest hash mismatch"
[ "$(sha256sum "$B_SRC" | cut -d' ' -f1)" = "$B_WANT" ] || die "binary hash mismatch"
say "artefacts match the approved hashes"

# --------------------------------------------------------- capture known-good
# A copy, not a reference: the restore must survive the live file being replaced,
# and it must be the exact bytes that were running when we started.
BACKUP=$(mktemp -d /var/tmp/switchyard-release-backup.XXXXXX)
chmod 0700 "$BACKUP"
cp -a "$LIVE_BIN" "$BACKUP/switchyard-server"
cp -a "$LIVE_MANIFEST" "$BACKUP/routes.toml"
OLD_BIN=$(sha256sum "$LIVE_BIN" | cut -d' ' -f1)
OLD_MANIFEST=$(sha256sum "$LIVE_MANIFEST" | cut -d' ' -f1)
say "current release: binary $OLD_BIN manifest $OLD_MANIFEST"
[ "$(sha256sum "$BACKUP/switchyard-server" | cut -d' ' -f1)" = "$OLD_BIN" ] \
  || die "backup copy is not the live binary"
[ "$(sha256sum "$BACKUP/routes.toml" | cut -d' ' -f1)" = "$OLD_MANIFEST" ] \
  || die "backup copy is not the live manifest"

RELEASE_VERIFIED=0
RESTORED=0

restore_previous() {
  [ "$RESTORED" -eq 1 ] && return 0
  RESTORED=1
  say "restoring the previous release"
  install -o root -g root -m 0755 "$BACKUP/switchyard-server" "$LIVE_BIN.restore"
  install -o root -g "$MANIFEST_GROUP" -m 0640 "$BACKUP/routes.toml" "$LIVE_MANIFEST.restore"
  [ "$(sha256sum "$LIVE_BIN.restore" | cut -d' ' -f1)" = "$OLD_BIN" ] \
    || { say "restored binary does not match the previous release"; return 1; }
  [ "$(sha256sum "$LIVE_MANIFEST.restore" | cut -d' ' -f1)" = "$OLD_MANIFEST" ] \
    || { say "restored manifest does not match the previous release"; return 1; }
  mv -f "$LIVE_BIN.restore" "$LIVE_BIN"
  mv -f "$LIVE_MANIFEST.restore" "$LIVE_MANIFEST"
  say "previous bytes installed; restarting"
  eval "$RESTART_CMD" || say "restart during restore failed"
  if wait_healthy; then
    say "rollback complete: running the previous release, health ok"
    return 0
  fi
  say "rollback restarted but health is still failing"
  return 1
}

on_exit() {
  local status=$?
  if [ "$status" -ne 0 ] && [ "$RELEASE_VERIFIED" -eq 0 ]; then
    say "activation did not complete; restoring the previous release"
    restore_previous || say "RESTORE FAILED - manual attention required"
  fi
  rm -rf "$BACKUP"
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

# ------------------------------------------------------------- pre-flight checks
healthy || die "production is not healthy before activation; refusing to start"
say "production healthy before activation"

say "validating the candidate with the candidate binary"
set -a
for env_file in "$ENV_FILE_1" "$ENV_FILE_2"; do
  [ -f "$env_file" ] || continue
  # shellcheck disable=SC1090
  . "$env_file"
done
set +a
if [ -n "$VALIDATE_CMD" ]; then
  eval "$VALIDATE_CMD" >/tmp/candidate-dryrun.txt 2>&1 \
    || { cat /tmp/candidate-dryrun.txt >&2; die "candidate does not validate; production untouched"; }
else
  "$B_SRC" --config "$M_SRC" --dry-run >/tmp/candidate-dryrun.txt 2>&1 \
    || { cat /tmp/candidate-dryrun.txt >&2; die "candidate does not validate; production untouched"; }
fi
say "candidate validates: $(head -1 /tmp/candidate-dryrun.txt | cut -c1-64)..."

# ------------------------------------------------------------------- install
# Write beside the target and rename: a reader either sees the old file or the
# new one, never a partial write.
install -o root -g root -m 0755 "$B_SRC" "$LIVE_BIN.new"
[ "$(sha256sum "$LIVE_BIN.new" | cut -d' ' -f1)" = "$B_WANT" ] || die "staged binary is not the candidate"
mv -f "$LIVE_BIN.new" "$LIVE_BIN"
say "binary installed"

install -o root -g "$MANIFEST_GROUP" -m 0640 "$M_SRC" "$LIVE_MANIFEST.new"
[ "$(sha256sum "$LIVE_MANIFEST.new" | cut -d' ' -f1)" = "$M_WANT" ] || die "staged manifest is not the candidate"
mv -f "$LIVE_MANIFEST.new" "$LIVE_MANIFEST"
say "manifest installed"

# --------------------------------------------------------------- activate
say "restarting $UNIT"
eval "$RESTART_CMD" || die "restart failed"

if ! wait_healthy; then
  die "health check failed after restart"
fi
say "health ok after restart"

# --------------------------------------------------------------- verify
[ "$(sha256sum "$LIVE_BIN" | cut -d' ' -f1)" = "$B_WANT" ] || die "installed binary is not the candidate"
[ "$(sha256sum "$LIVE_MANIFEST" | cut -d' ' -f1)" = "$M_WANT" ] || die "installed manifest is not the candidate"
say "installed bytes verified"

# The release is now live, healthy and byte-verified. From this point it is
# authoritative: a later error is bookkeeping, and reverting a working deployment
# because a mirror copy failed would be the worse outcome.
RELEASE_VERIFIED=1

# Keep the deployed pair in staging so the next release can diff against it.
# `install` refuses to copy a file onto itself, which under errexit would abort an
# otherwise successful release -- so the same-file case is skipped deliberately.
same() { [ "$1" -ef "$2" ] 2>/dev/null; }
same "$M_SRC" "$STAGED_MANIFEST" || install -o root -g root -m 0644 "$M_SRC" "$STAGED_MANIFEST" \
  || say "WARNING: could not mirror the manifest into staging"
same "$B_SRC" "$STAGED_BIN" || install -o root -g root -m 0755 "$B_SRC" "$STAGED_BIN" \
  || say "WARNING: could not mirror the binary into staging"

say "active=$( { systemctl is-active "$UNIT" 2>/dev/null || true; } ) pid=$( { systemctl show -p MainPID --value "$UNIT" 2>/dev/null || true; } )"
say "binary   $(sha256sum "$LIVE_BIN" | cut -d' ' -f1)"
say "manifest $(sha256sum "$LIVE_MANIFEST" | cut -d' ' -f1)"
say "ACTIVATION COMPLETE"
