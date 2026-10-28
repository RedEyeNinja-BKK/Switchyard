#!/usr/bin/env bash
# Install, roll back, verify and adopt Switchyard releases.
#
# CONTRACT
# --------
#   * Every artefact is hash-checked before anything is replaced.
#   * The candidate manifest is validated with the CANDIDATE binary, under the
#     unit's own environment, before it becomes live. Secrets are read by the
#     validation process and never printed, copied, or exported into this shell.
#   * A release is durable: binary, manifest, capability manifest and provenance
#     are written to a versioned directory that survives the run. Rolling back is
#     therefore possible AFTER this script has exited, which a /var/tmp copy is not.
#   * Installation is a rename, so no reader ever sees a half-written file.
#   * Health is verified after the restart. A failure restores the previous release
#     and restarts again, and the restore's own outcome is what the exit code reports.
#   * Any failure between "files replaced" and "health verified" triggers the same
#     restore from an EXIT trap, so an interrupted run cannot leave the disk and the
#     running process on different versions.
#   * The restore is skipped once the release is verified, so a later, unrelated
#     error cannot revert a good deployment.
#   * LAST-KNOWN-GOOD is advanced only by an explicit `promote`. A release that
#     merely started successfully is not thereby known good, and overwriting the
#     rollback target on every deploy would leave nothing to roll back to.
#
# VERBS
# -----
#   deploy <manifest> <manifest-sha> <binary> <binary-sha> [options]
#       Activate a release. Current positional form, kept because it is the form
#       existing runbooks use.
#   rollback [<release-id>]
#       Restore a stored release and restart. Defaults to LAST-KNOWN-GOOD.
#   promote [<release-id>]
#       Make a stored release the rollback target. Defaults to CURRENT.
#   adopt [--label L] [--source-commit SHA] [--basis TEXT]
#       Capture the live pair as the initial retained baseline, without restarting.
#   verify [<release-id>]
#       Re-check a stored release's bytes, and the live files when it is CURRENT.
#   list
#       Show stored releases, their ages and which is CURRENT / LAST-KNOWN-GOOD.
#
# EXIT CODES
#   0  ok
#   1  refused before changing anything (preconditions, hashes, validation, gate)
#   3  activation failed, previous release restored and VERIFIED running
#   4  activation failed and the restore could NOT be verified -- manual attention
#
# Seams for fault-injection tests: every path and command can be overridden by an
# environment variable, and the defaults below are the production values.
set -Eeuo pipefail
set +x
umask 027

SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

UNIT=${UNIT:-switchyard-neutral-green.service}
LIVE_MANIFEST=${LIVE_MANIFEST:-/etc/switchyard/routes.toml}
LIVE_BIN=${LIVE_BIN:-/opt/switchyard/bin/switchyard-server}
STAGED_MANIFEST=${STAGED_MANIFEST:-/opt/switchyard/staged/routes_deployed.toml}
STAGED_BIN=${STAGED_BIN:-/opt/switchyard/staged/switchyard-server-candidate}
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:4000/health}
METRICS_URL=${METRICS_URL:-http://127.0.0.1:4000/metrics}
# A file-comparison probe stands in for the HTTP check during fault injection.
HEALTH_CMD=${HEALTH_CMD:-}
RESTART_CMD=${RESTART_CMD:-systemctl restart "$UNIT"}
HEALTH_ATTEMPTS=${HEALTH_ATTEMPTS:-80}
# The two environment files the unit reads. Their CONTENTS are never copied: the
# release records a hash so the configuration is identifiable without the secret
# leaving its file.
ENV_FILE_1=${ENV_FILE_1:-/etc/switchyard/secrets/providers.env}
ENV_FILE_2=${ENV_FILE_2:-/var/lib/switchyard/oauth/switchyard-openai.env}
MANIFEST_GROUP=${MANIFEST_GROUP:-switchyard}
RELEASE_ROOT=${RELEASE_ROOT:-/var/lib/switchyard/releases}
RETAIN=${RETAIN:-5}
VALIDATE_CMD=${VALIDATE_CMD:-}
VALIDATOR=${VALIDATOR:-$SELF_DIR/release_validate.py}
CAPABILITY_TOOL=${CAPABILITY_TOOL:-$SELF_DIR/capability_contract.py}

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

sha() { sha256sum "$1" | cut -d' ' -f1; }

# ---------------------------------------------------------------- preconditions
[ "$(id -u)" -eq 0 ] || die "must run as root"
mkdir -p "$RELEASE_ROOT"
chmod 0750 "$RELEASE_ROOT"

CURRENT_LINK=$RELEASE_ROOT/CURRENT
LKG_LINK=$RELEASE_ROOT/LAST-KNOWN-GOOD

# The release a symlink names, or empty. A dangling link is reported rather than
# followed: a release directory removed underneath us must not read as "none".
link_target() {
  local link=$1 target
  [ -L "$link" ] || return 0
  target=$(readlink "$link")
  if [ -f "$link/release.json" ]; then
    printf '%s\n' "$target"
  else
    printf 'DANGLING:%s\n' "$target"
  fi
}

# Every mutation of the live pair goes through here, so the restore path has one
# shape. Returns non-zero when the bytes could not be installed or do not match.
install_pair() {
  local bin_src=$1 bin_sha=$2 man_src=$3 man_sha=$4
  install -o root -g root -m 0755 "$bin_src" "$LIVE_BIN.restore" || return 1
  install -o root -g "$MANIFEST_GROUP" -m 0640 "$man_src" "$LIVE_MANIFEST.restore" || return 1
  [ "$(sha "$LIVE_BIN.restore")" = "$bin_sha" ] \
    || { say "staged binary does not match $bin_sha"; return 1; }
  [ "$(sha "$LIVE_MANIFEST.restore")" = "$man_sha" ] \
    || { say "staged manifest does not match $man_sha"; return 1; }
  mv -f "$LIVE_BIN.restore" "$LIVE_BIN" || return 1
  mv -f "$LIVE_MANIFEST.restore" "$LIVE_MANIFEST" || return 1
  return 0
}

# Brings a stored release live and proves it, returning one of three verdicts.
# The caller decides what to do; nothing here exits, so a failure during rollback
# cannot re-enter the failure path that called it.
activate_release() {
  local dir=$1 id
  id=$(basename "$dir")
  if is_retained "$id"; then
    say "$id is a retained historical artefact, not a deployable release"
    return 1
  fi
  [ -f "$dir/switchyard-server" ] || { say "release $id has no binary"; return 1; }
  [ -f "$dir/routes.toml" ] || { say "release $id has no manifest"; return 1; }
  local want_bin want_man
  want_bin=$(json_field "$dir/release.json" binary sha256)
  want_man=$(json_field "$dir/release.json" manifest sha256)
  [ "$(sha "$dir/switchyard-server")" = "$want_bin" ] \
    || { say "stored binary in $id does not match its recorded hash"; return 1; }
  [ "$(sha "$dir/routes.toml")" = "$want_man" ] \
    || { say "stored manifest in $id does not match its recorded hash"; return 1; }
  install_pair "$dir/switchyard-server" "$want_bin" "$dir/routes.toml" "$want_man" \
    || return 1
  say "restarting $UNIT onto $id"
  eval "$RESTART_CMD" >/dev/null 2>&1 || say "restart command reported failure"
  wait_healthy || return 1
  [ "$(sha "$LIVE_BIN")" = "$want_bin" ] || { say "live binary is not $id"; return 1; }
  [ "$(sha "$LIVE_MANIFEST")" = "$want_man" ] || { say "live manifest is not $id"; return 1; }
  return 0
}

# Minimal JSON reader for the fields this script wrote itself. Avoids a jq
# dependency on a host that has none, and refuses rather than guessing.
# Two forms: `json_field FILE KEY` for a top-level field, and
# `json_field FILE SECTION KEY` for a nested one. An absent path or field is an
# error, never an empty string: a caller that reads "" cannot tell a missing value
# from a real one, and that is how `list` came to print nothing at all.
json_field() {
  # The program goes in argv, not on stdin. A heredoc-fed interpreter inside a
  # command substitution inside a `while read` loop can consume the loop's own
  # input stream; an argument has no such interaction.
  python3 -c '
import json, sys
path, section, key = sys.argv[1], sys.argv[2], sys.argv[3] or None
with open(path) as fh:
    doc = json.load(fh)
value = doc
if section:
    for part in section.split("."):
        value = value[part]
print(value if key is None else value[key])
' "$1" "$2" "${3-}"
}
# A boolean field, as the word `true`/`false`. Python prints `True`, which is a
# different spelling from the JSON it was read from and reads like a mistake.
json_bool() {
  python3 -c '
import json, sys
with open(sys.argv[1]) as fh:
    value = json.load(fh)
for part in sys.argv[2].split("."):
    value = value[part]
print("true" if value else "false")
' "$1" "$2"
}
# --------------------------------------------------------------- release store
#
# A release directory is written to a temporary sibling and renamed into place,
# so a reader never sees a half-written release. Its contents are the whole of
# what a later rollback needs; nothing is kept in /var/tmp.

# A release is a PAIR of artefacts, so its identity is the pair: the same binary
# with a different configuration is a different release, and two of them can be
# deployed inside one second.
release_id() {
  printf '%s-%s-%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" \
    "$(printf '%s' "$1" | cut -c1-16)" "$(printf '%s' "$2" | cut -c1-8)"
}

# Records the release being replaced and installation, without the secret. A hash
# identifies a configuration; the file keeps the secret.
env_identity_json() {
  local path out="" entry
  for path in "$ENV_FILE_1" "$ENV_FILE_2"; do
    if [ -f "$path" ]; then
      entry=$(printf '{"path":"%s","sha256":"%s","bytes":%s,"mode":"%s","owner":"%s","group":"%s"}' \
        "$path" "$(sha "$path")" "$(stat -c %s "$path")" "$(stat -c %a "$path")" \
        "$(stat -c %U "$path")" "$(stat -c %G "$path")")
    else
      entry=$(printf '{"path":"%s","present":false}' "$path")
    fi
    out=${out:+$out,}$entry
  done
  printf '[%s]' "$out"
}

unit_identity_json() {
  local path fragment
  path=$(systemctl show -p FragmentPath --value "$UNIT" 2>/dev/null || true)
  if [ -n "$path" ] && [ -f "$path" ]; then
    fragment=$(sha "$path")
  else
    path=""
    fragment=""
  fi
  printf '{"unit":"%s","fragment":"%s","fragment_sha256":"%s"}' "$UNIT" "$path" "$fragment"
}

# store_release <dir> <binary> <binary-sha> <manifest> <manifest-sha> <caps-or-empty> \
#               <action> <label> <commit> <branch> <dirty> <basis>
store_release() {
  local dir=$1 b_src=$2 b_sha=$3 m_src=$4 m_sha=$5 caps=$6 action=$7
  local label=$8 commit=$9 branch=${10} dirty=${11} basis=${12}
  local work caps_json=""
  work="$dir.new"
  rm -rf "$work"
  mkdir -p "$work"
  chmod 0750 "$work"

  install -o root -g root -m 0755 "$b_src" "$work/switchyard-server"
  [ "$(sha "$work/switchyard-server")" = "$b_sha" ] \
    || { say "release copy is not the candidate binary"; rm -rf "$work"; return 1; }
  install -o root -g "$MANIFEST_GROUP" -m 0640 "$m_src" "$work/routes.toml"
  [ "$(sha "$work/routes.toml")" = "$m_sha" ] \
    || { say "release copy is not the candidate manifest"; rm -rf "$work"; return 1; }

  if [ -n "$caps" ] && [ -f "$caps" ]; then
    install -o root -g root -m 0644 "$caps" "$work/capabilities.json"
    caps_json=$(printf '{"file":"capabilities.json","sha256":"%s","bytes":%s}' \
      "$(sha "$work/capabilities.json")" "$(stat -c %s "$work/capabilities.json")")
  else
    caps_json='null'
  fi

  local created
  created=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # Attestation records WHEN the provenance was captured, and binds it to the
  # binary hash. A deploy captures the source the operator declared at activation
  # time; `adopt` can only describe bytes that are already running and never
  # attests. This is deliberately NOT a claim of byte-reproducibility: a rebuild
  # of the same commit has not been compared, and the field says so on its face.
  local attested=false attestation
  if [ "$action" = deploy ] && [ -n "$commit" ]; then
    attested=true ; attestation="declared-at-deploy"
  elif [ "$action" = deploy ]; then
    attestation="deploy-without-source-commit"
  else
    attestation="captured-after-the-fact-from-running-bytes"
  fi
  local dirty_note=""
  [ "$dirty" = true ] && dirty_note=" (working tree dirty)"
  cat >"$work/release.json" <<JSON
{
  "schema": 2,
  "release_id": "$(basename "$dir")",
  "created_at": "$created",
  "action": "$action",
  "label": "$label",
  "binary": {"file": "switchyard-server", "sha256": "$b_sha", "bytes": $(stat -c %s "$work/switchyard-server"), "mode": "$(stat -c %a "$work/switchyard-server")"},
  "manifest": {"file": "routes.toml", "sha256": "$m_sha", "bytes": $(stat -c %s "$work/routes.toml"), "mode": "$(stat -c %a "$work/routes.toml")", "group": "$MANIFEST_GROUP"},
  "capabilities": $caps_json,
  "provenance": {"source_commit": "$commit", "source_branch": "$branch", "source_dirty": $dirty, "basis": "$basis", "attested": $attested, "attestation": "$attestation", "note": "Attested means the source identity was declared when these exact bytes were installed and is bound to the binary sha256 above. It does not claim the build is reproducible: no rebuild has been compared against it."},
  "configuration": {"unit": $(unit_identity_json), "env_files": $(env_identity_json), "note": "Environment file contents are deliberately not copied: the hash identifies the configuration, and the secret stays in its own file with its own permissions."}
}
JSON
  # A human-readable companion, so the attestation status of a stored release can be
  # read without parsing JSON and without anyone having to remember what it meant.
  {
    printf 'release       %s\n' "$(basename "$dir")"
    printf 'recorded      %s (%s)\n' "$created" "$action"
    printf 'binary        %s\n' "$b_sha"
    printf 'manifest      %s\n' "$m_sha"
    printf 'source commit %s\n' "${commit:-not recorded}"
    printf 'source branch %s\n' "${branch:-not recorded}${dirty_note}"
    printf 'basis         %s\n' "$basis"
    printf 'attested      %s (%s)\n' "$attested" "$attestation"
    printf '\n'
    if [ "$attested" = true ]; then
      printf 'The source identity above was declared when these exact bytes were installed,\n'
      printf 'and is bound to the binary sha256. The build has not been shown to be\n'
      printf 'reproducible: no rebuild of that commit was compared against this binary.\n'
    else
      printf 'This release is NOT attested. It records bytes observed on this host; it does\n'
      printf 'not record a build. The source commit is advisory at best and may be unknown.\n'
      printf 'Treat the binary and manifest hashes as the whole of the evidence.\n'
    fi
  } >"$work/PROVENANCE.txt"
  chmod 0640 "$work/PROVENANCE.txt"
  chmod 0640 "$work/release.json"
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$work/release.json" \
    || { say "release metadata is not valid JSON"; rm -rf "$work"; return 1; }

  mv -T "$work" "$dir"
}

# Writes a release record for the pair that is already live, without restarting.
# Used to establish the baseline on a host whose earlier binary is gone: the
# provenance is recorded as unattested because nothing can attest a build that no
# longer exists.
adopt_release() {
  local label=$1 commit=$2 branch=$3 dirty=$4 basis=$5 caps=$6
  [ -f "$LIVE_BIN" ] || die "no live binary at $LIVE_BIN"
  [ -f "$LIVE_MANIFEST" ] || die "no live manifest at $LIVE_MANIFEST"
  local b_sha m_sha id dir
  b_sha=$(sha "$LIVE_BIN")
  m_sha=$(sha "$LIVE_MANIFEST")
  id=$(release_id "$b_sha" "$m_sha")
  dir=$RELEASE_ROOT/$id
  [ -e "$dir" ] && die "release $id already exists"
  say "adopting the live pair: binary $b_sha manifest $m_sha"
  store_release "$dir" "$LIVE_BIN" "$b_sha" "$LIVE_MANIFEST" "$m_sha" "$caps" \
    adopt "$label" "$commit" "$branch" "$dirty" "$basis" || die "could not record the release"
  point_at CURRENT "$id"
  # The adopted release is the only evidence of a good state this host has, so it
  # is also the rollback target. A later deploy will NOT move this pointer.
  point_at LAST-KNOWN-GOOD "$id"
  say "ADOPTED $id (CURRENT and LAST-KNOWN-GOOD)"
}

point_at() {
  local link=$1 id=$2
  # Callers may name the pointer ("CURRENT") or give its path. Normalising here
  # keeps the verification below reading the same file the rename just wrote --
  # checking a relative path against the shell's working directory silently
  # reports every pointer as missing.
  case $link in
    /*) ;;
    *) link=$RELEASE_ROOT/$link ;;
  esac
  [ -d "$RELEASE_ROOT/$id" ] || die "release $id does not exist"
  ln -sfn "$id" "$RELEASE_ROOT/.$id.link"
  mv -Tf "$RELEASE_ROOT/.$id.link" "$RELEASE_ROOT/$(basename "$link")" 2>/dev/null \
    || ln -sfn "$id" "$link"
  [ "$(link_target "$link")" = "$id" ] || die "could not point $link at $id"
}

# A stored directory may carry a RETAINED marker. Such an artefact is kept for the
# record and is exempt from every automatic decision about it: retention never
# prunes it, it never consumes a retention slot, and no verb will install, verify or
# promote it as a release. It exists for history that cannot be re-established --
# an artefact whose provenance predates the store, or one held deliberately because
# it is evidence of what ran at a point in time.
is_retained() { [ -f "$RELEASE_ROOT/$1/RETAINED" ]; }

# Keeps the newest RETAIN releases plus CURRENT and LAST-KNOWN-GOOD. The two
# pointers are protected by identity, not by age: an old release that is still the
# rollback target must survive any amount of newer activity.
prune_releases() {
  local keep_current keep_lkg id dir n=0 keep=()
  keep_current=$(link_target "$CURRENT_LINK")
  keep_lkg=$(link_target "$LKG_LINK")
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    # A retained artefact is never pruned, and it does not consume a retention slot:
    # otherwise marking one thing as permanent would silently shorten the history of
    # everything else.
    if is_retained "$id"; then
      keep+=("$id")
      continue
    fi
    n=$((n + 1))
    if [ "$n" -le "$RETAIN" ] || [ "$id" = "$keep_current" ] || [ "$id" = "$keep_lkg" ]; then
      keep+=("$id")
    fi
  done < <(list_release_ids)
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    local found=0
    for dir in "${keep[@]:-}"; do
      [ "$dir" = "$id" ] && { found=1; break; }
    done
    if [ "$found" -eq 0 ]; then
      say "pruning release $id (retention $RETAIN)"
      rm -rf "${RELEASE_ROOT:?}/$id"
    fi
  done < <(list_release_ids)
}

list_release_ids() {
  # A `.new` directory is a write that never completed. It is not a release and
  # must not be listed, counted for retention, or offered as a rollback target.
  find "$RELEASE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
    | grep -v '^\.' | grep -v '\.new$' | sort -r || true
}
# ------------------------------------------------------------------ validation
#
# The candidate is validated with the candidate binary under the unit's own
# environment. Three properties are non-negotiable here:
#
#   1. systemd EnvironmentFile SYNTAX, not bash. A value containing `$`, a
#      backtick or an unquoted space is legal for systemd and is either mangled or
#      executed by `. file`. The helper parses the file the way systemd does.
#   2. The variables exist only in the child process. Sourcing them into this
#      shell would put provider keys in the environment of every later command,
#      including the restart and anything an operator runs afterwards.
#   3. Nothing is echoed. The child's diagnostics are scrubbed of `KEY=value`
#      before they reach a log, and its full output goes to a 0600 file under the
#      release root for the operator, not to the terminal.
validate_candidate() {
  local binary=$1 manifest=$2 log=$3
  if [ -n "$VALIDATE_CMD" ]; then
    # Testing seam: the real check would need a working binary and real keys. The
    # override runs with the same parsed environment so an injected failure still
    # exercises the same path.
    "$VALIDATOR" --binary "$binary" --manifest "$manifest" \
      --env-file "$ENV_FILE_1" --env-file "$ENV_FILE_2" \
      --command "$VALIDATE_CMD" --log "$log"
    return
  fi
  "$VALIDATOR" --binary "$binary" --manifest "$manifest" \
    --env-file "$ENV_FILE_1" --env-file "$ENV_FILE_2" --log "$log"
}

capability_gate() {
  local binary=$1 caps=$2 baseline=$3
  [ -n "$caps" ] || return 0
  [ -f "$caps" ] || { say "capability manifest $caps not found"; return 1; }
  if [ ! -f "$baseline" ]; then
    say "no deployed capability baseline at $baseline"
    say "the preservation check cannot be made without one; adopt or deploy a release"
    say "that carries a capability manifest before replacing this binary"
    return 1
  fi
  python3 "$CAPABILITY_TOOL" preflight --binary "$binary" --manifest "$caps" \
    --baseline "$baseline"
}

post_verify() {
  local caps=$1
  healthy || { say "service is not healthy"; return 1; }
  say "health ok"
  if [ -n "$caps" ] && [ -f "$caps" ]; then
    python3 "$CAPABILITY_TOOL" verify-live --manifest "$caps" --metrics-url "$METRICS_URL" \
      || return 1
  else
    say "no capability manifest for this release; skipped the always-present check"
  fi
  return 0
}

# ---------------------------------------------------------------------- deploy
cmd_deploy() {
  local m_src="" m_want="" b_src="" b_want="" label="" caps="" commit="" branch="" dirty=false
  local basis="build-session" promote=0
  while [ $# -gt 0 ]; do
    case $1 in
      --label) label=$2; shift 2 ;;
      --capabilities) caps=$2; shift 2 ;;
      --source-commit) commit=$2; shift 2 ;;
      --source-branch) branch=$2; shift 2 ;;
      --source-dirty) dirty=true; shift ;;
      --basis) basis=$2; shift 2 ;;
      --promote) promote=1; shift ;;
      -*) die "unknown option $1" ;;
      *)
        if [ -z "$m_src" ]; then m_src=$1
        elif [ -z "$m_want" ]; then m_want=$1
        elif [ -z "$b_src" ]; then b_src=$1
        elif [ -z "$b_want" ]; then b_want=$1
        else die "unexpected argument $1"
        fi
        shift ;;
    esac
  done
  [ -n "$b_want" ] || die "usage: $0 deploy <manifest> <manifest-sha> <binary> <binary-sha>"
  [ -f "$m_src" ] || die "missing manifest $m_src"
  [ -f "$b_src" ] || die "missing binary $b_src"
  [ "$(sha "$m_src")" = "$m_want" ] || die "manifest hash mismatch"
  [ "$(sha "$b_src")" = "$b_want" ] || die "binary hash mismatch"
  say "artefacts match the approved hashes"

  # The rollback target must exist BEFORE anything is replaced. A deployment with
  # nowhere to go back to is the situation this whole change exists to remove.
  local prev_id prev_dir
  prev_id=$(link_target "$CURRENT_LINK")
  case $prev_id in
    DANGLING:*) die "CURRENT points at ${prev_id#DANGLING:} which is not a release; repair before deploying" ;;
    "") die "no CURRENT release recorded; run '$0 adopt' first so a rollback target exists" ;;
  esac
  prev_dir=$RELEASE_ROOT/$prev_id
  [ -f "$prev_dir/release.json" ] || die "release $prev_id has no metadata; run '$0 verify'"
  [ "$(sha "$prev_dir/switchyard-server")" = "$(json_field "$prev_dir/release.json" binary sha256)" ] \
    || die "rollback target $prev_id has a binary that does not match its record"
  [ "$(sha "$prev_dir/routes.toml")" = "$(json_field "$prev_dir/release.json" manifest sha256)" ] \
    || die "rollback target $prev_id has a manifest that does not match its record"
  say "current release: $prev_id"

  healthy || die "production is not healthy before activation; refusing to start"
  say "production healthy before activation"

  local log=$RELEASE_ROOT/.validate-$(date -u +%Y%m%dT%H%M%SZ).log
  say "validating the candidate with the candidate binary, under the unit environment"
  if ! validate_candidate "$b_src" "$m_src" "$log"; then
    die "candidate does not validate; production untouched (detail: $log)"
  fi
  say "candidate validates"

  say "capability gate"
  if ! capability_gate "$b_src" "$caps" "$prev_dir/capabilities.json"; then
    die "capability gate refused this candidate; production untouched"
  fi

  local restored=0
  restore_and_report() {
    if [ "$restored" -eq 1 ]; then return 0; fi
    restored=1
    say "restoring $prev_id"
    if activate_release "$prev_dir"; then
      # The pointer must name the binary that is actually running. This path is
      # reached after CURRENT has already moved to the candidate, so without this
      # the store would claim a release was current that had just been rolled back
      # out -- and `verify` would report the live pair as mismatched.
      point_at CURRENT "$prev_id" || say "could not repoint CURRENT to $prev_id"
      say "ROLLBACK VERIFIED: $prev_id is live, byte-verified and healthy"
      return 0
    fi
    say "ROLLBACK INCOMPLETE: $prev_id could not be restored and verified" >&2
    return 1
  }
  fail_and_rollback() {
    say "FATAL: $1" >&2
    if restore_and_report; then exit 3; else exit 4; fi
  }

  RELEASE_VERIFIED=0
  on_exit() {
    local status=$?
    if [ "$status" -ne 0 ] && [ "$RELEASE_VERIFIED" -eq 0 ]; then
      restore_and_report || say "RESTORE FAILED - manual attention required" >&2
    fi
    exit "$status"
  }
  trap on_exit EXIT
  trap 'exit 130' INT TERM

  say "installing"
  install_pair "$b_src" "$b_want" "$m_src" "$m_want" || fail_and_rollback "could not install the candidate"
  say "restarting $UNIT"
  eval "$RESTART_CMD" >/dev/null 2>&1 || fail_and_rollback "restart failed"
  wait_healthy || fail_and_rollback "health check failed after restart"
  say "health ok after restart"
  [ "$(sha "$LIVE_BIN")" = "$b_want" ] || fail_and_rollback "installed binary is not the candidate"
  [ "$(sha "$LIVE_MANIFEST")" = "$m_want" ] || fail_and_rollback "installed manifest is not the candidate"
  say "installed bytes verified"
  RELEASE_VERIFIED=1

  # The release is durable before the pointers move: a crash between these two
  # statements leaves the store consistent, with CURRENT still on the previous
  # release even though the new binary is running -- which is recoverable, unlike
  # a pointer to a directory that does not exist.
  local id dir reuse=0
  id=$(release_id "$b_want" "$m_want")
  dir=$RELEASE_ROOT/$id
  # Deploying the same pair twice is a legitimate no-op, and two deployments of
  # one binary inside the same second would otherwise collide on the name. A
  # stored release is reused only when BOTH artefacts match; the same binary with
  # a different configuration is a different release and gets its own directory.
  if [ -d "$dir" ]; then
    if [ "$(json_field "$dir/release.json" binary sha256)" = "$b_want" ] \
       && [ "$(json_field "$dir/release.json" manifest sha256)" = "$m_want" ]; then
      say "release $id already records these exact bytes; reusing it"
      reuse=1
    else
      # Same name, different bytes: only possible if a stored release was modified
      # in place. Refuse rather than overwrite the evidence.
      fail_and_rollback "release $id exists with different recorded bytes"
    fi
  fi
  if [ "$reuse" -eq 0 ] \
     && ! store_release "$dir" "$b_src" "$b_want" "$m_src" "$m_want" "$caps" \
          deploy "$label" "$commit" "$branch" "$dirty" "$basis"; then
    fail_and_rollback "could not record the release durably"
  fi
  point_at CURRENT "$id"

  say "post-deployment verification"
  if ! post_verify "$caps"; then
    fail_and_rollback "post-deployment verification failed"
  fi

  # Promotion follows post-deployment verification, never the restart. Moving this
  # above the check is a one-line change that points LAST-KNOWN-GOOD at a release
  # whose own checks had not yet run, and a later rollback would then install it as
  # if it were known good -- which is the opposite of what the pointer means.
  if [ "$promote" -eq 1 ]; then
    point_at LAST-KNOWN-GOOD "$id"
    say "promoted $id to LAST-KNOWN-GOOD after post-deployment verification"
  fi

  same() { [ "$1" -ef "$2" ] 2>/dev/null; }
  same "$m_src" "$STAGED_MANIFEST" || install -o root -g root -m 0644 "$m_src" "$STAGED_MANIFEST" \
    || say "WARNING: could not mirror the manifest into staging"
  same "$b_src" "$STAGED_BIN" || install -o root -g root -m 0755 "$b_src" "$STAGED_BIN" \
    || say "WARNING: could not mirror the binary into staging"

  prune_releases
  rm -f "$log"

  say "active=$( { systemctl is-active "$UNIT" 2>/dev/null || true; } ) pid=$( { systemctl show -p MainPID --value "$UNIT" 2>/dev/null || true; } )"
  say "release  $id"
  say "binary   $(sha "$LIVE_BIN")"
  say "manifest $(sha "$LIVE_MANIFEST")"
  say "rollback target stays $prev_id until '$0 promote'"
  say "ACTIVATION COMPLETE"
}
# -------------------------------------------------------------------- rollback
#
# The whole point of a durable store: this runs after the activation script has
# exited, possibly days later, with nothing in /var/tmp to rely on.
cmd_rollback() {
  local target=${1:-} from_id from_dir
  from_id=$(link_target "$CURRENT_LINK")
  case $from_id in
    DANGLING:*) die "CURRENT points at ${from_id#DANGLING:} which is not a release" ;;
  esac
  if [ -z "$target" ]; then
    target=$(link_target "$LKG_LINK")
    case $target in
      DANGLING:*) die "LAST-KNOWN-GOOD points at ${target#DANGLING:} which is not a release" ;;
      "") die "no LAST-KNOWN-GOOD recorded; name a release id explicitly" ;;
    esac
  fi
  [ -d "$RELEASE_ROOT/$target" ] || die "no release $target in $RELEASE_ROOT"
  if is_retained "$target"; then
    die "$target is a retained historical artefact and is not deployable"
  fi
  if [ "$target" = "$from_id" ]; then
    say "CURRENT is already $target; verifying instead of reinstalling"
  fi
  say "rolling back from ${from_id:-none} to $target"
  local want_bin want_man
  want_bin=$(json_field "$RELEASE_ROOT/$target/release.json" binary sha256)
  want_man=$(json_field "$RELEASE_ROOT/$target/release.json" manifest sha256)

  if activate_release "$RELEASE_ROOT/$target"; then
    point_at CURRENT "$target"
    say "ROLLBACK VERIFIED: $target is live, byte-verified and healthy"
    say "binary   $want_bin"
    say "manifest $want_man"
    if [ -n "$from_id" ] && [ -d "$RELEASE_ROOT/$from_id" ]; then
      say "the release that was replaced remains at $RELEASE_ROOT/$from_id"
    fi
    exit 0
  fi

  # Restoring the release we just left is the only remaining move, and it is only
  # possible if that release is itself intact.
  say "rollback to $target did not verify" >&2
  if [ -n "$from_id" ] && [ -d "$RELEASE_ROOT/$from_id" ]; then
    say "restoring $from_id" >&2
    if activate_release "$RELEASE_ROOT/$from_id"; then
      say "ROLLBACK INCOMPLETE: $target failed, $from_id restored and verified" >&2
      exit 4
    fi
  fi
  say "ROLLBACK INCOMPLETE: neither release could be verified; manual attention required" >&2
  exit 4
}

cmd_promote() {
  local id=${1:-}
  if [ -z "$id" ]; then
    id=$(link_target "$CURRENT_LINK")
    [ -n "$id" ] || die "no CURRENT release to promote"
  fi
  [ -d "$RELEASE_ROOT/$id" ] || die "no release $id"
  if is_retained "$id"; then
    die "$id is a retained historical artefact and cannot become the rollback target"
  fi
  # Only a release that is live AND healthy may become the rollback target.
  local live
  live=$(link_target "$CURRENT_LINK")
  if [ "$id" = "$live" ]; then
    healthy || die "$id is CURRENT but the service is not healthy; not promoting"
  else
    say "note: $id is not CURRENT; promoting a release that is not running"
  fi
  [ "$(sha "$RELEASE_ROOT/$id/switchyard-server")" = "$(json_field "$RELEASE_ROOT/$id/release.json" binary sha256)" ] \
    || die "$id binary does not match its record"
  [ "$(sha "$RELEASE_ROOT/$id/routes.toml")" = "$(json_field "$RELEASE_ROOT/$id/release.json" manifest sha256)" ] \
    || die "$id manifest does not match its record"
  point_at LAST-KNOWN-GOOD "$id"
  say "LAST-KNOWN-GOOD is now $id"
}

cmd_verify() {
  local id=${1:-} rc=0
  if [ -z "$id" ]; then
    id=$(link_target "$CURRENT_LINK")
    [ -n "$id" ] || die "no CURRENT release; name a release id"
  fi
  local dir=$RELEASE_ROOT/$id
  [ -d "$dir" ] || die "no release $id"
  if is_retained "$id"; then
    die "$id is a retained historical artefact: it is kept for the record and is not \
a release, so there is nothing to verify against a record it never had"
  fi
  local want_bin want_man
  want_bin=$(json_field "$dir/release.json" binary sha256)
  want_man=$(json_field "$dir/release.json" manifest sha256)
  local got_bin got_man
  got_bin=$(sha "$dir/switchyard-server")
  got_man=$(sha "$dir/routes.toml")
  printf '%-16s %-18s %s\n' release "$id" stored
  if [ "$got_bin" = "$want_bin" ]; then
    printf '  binary    %s  ok\n' "$got_bin"
  else
    printf '  binary    %s  MISMATCH (recorded %s)\n' "$got_bin" "$want_bin"; rc=1
  fi
  if [ "$got_man" = "$want_man" ]; then
    printf '  manifest  %s  ok\n' "$got_man"
  else
    printf '  manifest  %s  MISMATCH (recorded %s)\n' "$got_man" "$want_man"; rc=1
  fi
  if [ "$(link_target "$CURRENT_LINK")" = "$id" ]; then
    [ -f "$LIVE_BIN" ] && [ "$(sha "$LIVE_BIN")" = "$want_bin" ] \
      || { printf '  live binary does not match this release\n'; rc=1; }
    [ -f "$LIVE_MANIFEST" ] && [ "$(sha "$LIVE_MANIFEST")" = "$want_man" ] \
      || { printf '  live manifest does not match this release\n'; rc=1; }
    healthy && printf '  service   healthy\n' || { printf '  service   NOT healthy\n'; rc=1; }
  fi
  return $rc
}

cmd_list() {
  local cur lkg id created bin mark
  cur=$(link_target "$CURRENT_LINK")
  lkg=$(link_target "$LKG_LINK")
  printf '%-34s %-20s %-17s %s\n' RELEASE CREATED BINARY MARKS
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    created=$(json_field "$RELEASE_ROOT/$id/release.json" created_at 2>/dev/null || echo '?')
    bin=$(json_field "$RELEASE_ROOT/$id/release.json" binary sha256 2>/dev/null | cut -c1-16 || echo '?')
    mark=""
    [ "$id" = "$cur" ] && mark="$mark CURRENT"
    [ "$id" = "$lkg" ] && mark="$mark LAST-KNOWN-GOOD"
    is_retained "$id" && mark="$mark RETAINED"
    # Attestation is the one property an operator must not have to open release.json
    # to discover: an unattested release is a record of bytes, not of a build.
    if [ -f "$RELEASE_ROOT/$id/release.json" ] \
       && [ "$(json_bool "$RELEASE_ROOT/$id/release.json" provenance.attested 2>/dev/null)" != "true" ]; then
      mark="$mark UNATTESTED"
    fi
    printf '%-34s %-20s %-17s %s\n' "$id" "$created" "$bin" "${mark# }"
  done < <(list_release_ids)
}

# ------------------------------------------------------------------------ main
RELEASE_VERIFIED=0

usage() {
  sed -n '1,40p' "${BASH_SOURCE[0]}" | sed -n '/^# VERBS/,/^# EXIT/p' | sed 's/^# \{0,1\}//'
}

verb=${1:-}
case $verb in
  deploy)
    shift
    # The positional form predates the verb, so a bare
    # `activate-release.sh <manifest> <sha> <binary> <sha>` still works.
    cmd_deploy "$@"
    ;;
  rollback) shift; cmd_rollback "$@" ;;
  promote) shift; cmd_promote "$@" ;;
  verify) shift; cmd_verify "$@" ;;
  list) shift; cmd_list "$@" ;;
  adopt)
    shift
    label="" ; commit="" ; branch="" ; dirty=false ; basis="adopted-unattested" ; caps=""
    while [ $# -gt 0 ]; do
      case $1 in
        --label) label=$2; shift 2 ;;
        --source-commit) commit=$2; shift 2 ;;
        --source-branch) branch=$2; shift 2 ;;
        --source-dirty) dirty=true; shift ;;
        --basis) basis=$2; shift 2 ;;
        --capabilities) caps=$2; shift 2 ;;
        *) die "unknown option $1" ;;
      esac
    done
    adopt_release "$label" "$commit" "$branch" "$dirty" "$basis" "$caps"
    ;;
  ""|-h|--help|help) usage ;;
  *)
    # Backwards compatibility: no verb means the old positional deploy.
    if [ -f "$verb" ]; then cmd_deploy "$@"; else die "unknown command '$verb'"; fi
    ;;
esac
