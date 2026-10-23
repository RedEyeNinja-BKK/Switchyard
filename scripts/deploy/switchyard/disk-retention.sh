#!/usr/bin/env bash
# Build-artifact retention guard.
#
# The failure this prevents: a debug build and test artefacts fill the root
# filesystem, after which nothing on the host works — including the commands
# needed to investigate. Storage therefore gets its own scheduled check rather
# than being noticed at the point of failure.
#
# Three thresholds on the filesystem holding a watched path:
#   WARN   (%d)  record a warning; delete nothing
#   PRUNE  (%d)  remove regenerable build caches and stale test temp files
#   HARD   (%d)  additionally remove the whole debug build tree
#
# What is NEVER removed, in any branch: source, the release build, production
# binaries and configuration, credentials, and rollback anchors. Those are named
# in PROTECTED and verified to survive.
#
# Usage: disk-retention.sh [--dry-run]
# Tunables (environment): WARN_PCT PRUNE_PCT HARD_PCT
set -Eeuo pipefail
set +x

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

WARN_PCT=${WARN_PCT:-85}
PRUNE_PCT=${PRUNE_PCT:-90}
HARD_PCT=${HARD_PCT:-93}

TARGETS=${TARGETS:-/var/lib/turnstone/switchyard-upstream/recovered-6de85da1/target}
PYTEST_GLOB=${PYTEST_GLOB:-/tmp/pytest-of-*}

# Anything here must still exist after a run. Losing one of these would be worse
# than a full disk: it would be unrecoverable.
# This list is host-independent: each host protects the ones it actually has.
PROTECTED=(
  /var/lib/switchyard/rollback
  /var/lib/switchyard/routes.toml
  /opt/switchyard/bin/switchyard-server
  /opt/switchyard/bin/stephen-gateway.py
  /etc/switchyard/routes.toml
  /etc/switchyard/secrets
  /var/lib/turnstone/switchyard-upstream/recovered-6de85da1/recovered
  /var/lib/turnstone/credentials
  "$TARGETS/release"
)

say() { printf '%s\n' "$*"; }
warn() { say "$*"; command -v logger >/dev/null 2>&1 && logger -t disk-retention "$*" || true; }

remove() {
  local path=$1
  [ -e "$path" ] || return 0
  if [ "$DRY_RUN" -eq 1 ]; then
    say "  would remove $path ($(du -x -sh "$path" 2>/dev/null | cut -f1))"
    return 0
  fi
  local size
  size=$(du -x -sh "$path" 2>/dev/null | cut -f1 || echo "?")
  rm -rf -- "$path"
  say "  removed $path ($size)"
}

# Records which protected paths exist right now. Only these are asserted
# afterwards: a path that never existed on this host is not a failure, and
# asserting it would make the check meaningless noise on the build host.
SURVIVORS=()
snapshot_protected() {
  SURVIVORS=()
  local path
  for path in "${PROTECTED[@]}"; do
    # An `if`, not `[ ... ] && ...`: a failing test as the last statement of the
    # loop makes this function return non-zero, which errexit turns into an exit.
    if [ -e "$path" ]; then
      SURVIVORS+=("$path")
    fi
  done
  return 0
}

assert_protected() {
  local missing=0 path
  for path in "${SURVIVORS[@]}"; do
    if [ ! -e "$path" ]; then
      say "ERROR: protected path disappeared: $path"
      missing=1
    fi
  done
  return "$missing"
}

# Percentage used on the filesystem holding $1.
used_pct() {
  df -P "$1" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5}'
}

main() {
  # `|| true` on both: with pipefail a missing path would otherwise abort the
  # script silently, which is the least useful possible response to bad input.
  local anchor
  anchor=$( { df -P "$TARGETS" 2>/dev/null || true; } | awk 'NR==2 {print $6}')
  if [ -z "$anchor" ]; then
    say "cannot resolve a filesystem for $TARGETS; nothing to check"
    return 0
  fi

  local pct
  pct=$(used_pct "$TARGETS" || true)
  if [ -z "$pct" ]; then
    say "cannot read usage for $TARGETS; nothing to check"
    return 0
  fi
  say "filesystem $anchor is ${pct}% used (warn ${WARN_PCT} / prune ${PRUNE_PCT} / hard ${HARD_PCT})"
  snapshot_protected
  say "protecting ${#SURVIVORS[@]} existing path(s) on this host"

  if [ "$pct" -lt "$WARN_PCT" ]; then
    say "nothing to do"
    return 0
  fi

  warn "storage pressure on $anchor: ${pct}% used"

  if [ "$pct" -ge "$PRUNE_PCT" ]; then
    say "pruning regenerable caches"
    # Incremental compilation state: pure cache, rebuilt on demand.
    remove "$TARGETS/debug/incremental"
    # Test sandboxes older than a day. A live run's directory is left alone.
    local dir
    for dir in $PYTEST_GLOB; do
      [ -d "$dir" ] || continue
      if [ -n "$(find "$dir" -maxdepth 0 -mtime +1 2>/dev/null)" ]; then
        remove "$dir"
      fi
    done
    # Disposable capture files from earlier investigations.
    find /tmp -maxdepth 1 -type f -name '*.json' -mtime +3 -size +1M 2>/dev/null | while read -r f; do
      remove "$f"
    done
  fi

  if [ "$pct" -ge "$HARD_PCT" ]; then
    # Never pull the build tree out from under a compile that is running: the
    # output would be truncated mid-write, and the failure would look like a
    # source problem rather than a storage one. The cheaper caches above have
    # already been reclaimed by this point.
    if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1 \
       || pgrep -f "cargo build|cargo test|cargo check" >/dev/null 2>&1; then
      warn "a build is running; skipping the debug-profile removal at ${pct}%"
    else
      warn "emergency: removing the debug build profile on $anchor"
      remove "$TARGETS/debug"
    fi
  fi

  assert_protected || { warn "protected path check FAILED"; exit 1; }

  local after
  after=$(used_pct "$TARGETS")
  say "now ${after}% used"
  # Re-check: if still above HARD after pruning, this needs a human.
  if [ "${after:-0}" -ge "$HARD_PCT" ]; then
    warn "still ${after}% used after pruning; manual intervention required"
    exit 1
  fi
  return 0
}

main
