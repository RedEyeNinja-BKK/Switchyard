#!/usr/bin/env python3
"""Mutation controls for the environment-file discovery fix.

Each mutant breaks one property the fix is supposed to hold. The suite is then run
against the mutated copy and MUST fail: a mutant that survives means the property
is asserted by nothing, or asserted by a case that cannot distinguish it.

Run as root on the deployment host, from a directory holding pristine copies of
activate-release.sh, release_validate.py, test_activate_release.py and
capability_contract.py:

    python3 mutants_env_files.py

Nothing in the production paths is touched: every mutant is applied to a scratch
copy and the suite is run there.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FILES = ("activate-release.sh", "release_validate.py", "test_activate_release.py",
         "capability_contract.py")

SCRIPT = "activate-release.sh"
VALIDATOR = "release_validate.py"

MUTANTS = [
    dict(
        name="M1 discovery failure substitutes the old two-file default",
        file=SCRIPT,
        old='''  if ! raw=$("$SYSTEMCTL" show -p EnvironmentFiles --value "$UNIT" 2>/dev/null); then
    return 1
  fi
  [ -n "$raw" ] || return 1''',
        new='''  raw=$("$SYSTEMCTL" show -p EnvironmentFiles --value "$UNIT" 2>/dev/null || true)
  [ -n "$raw" ] || raw="/etc/switchyard/secrets/providers.env /var/lib/switchyard/oauth/switchyard-openai.env"''',
        caught_by="test_discovery_that_answers_nothing_is_refused_rather_than_guessed",
    ),
    dict(
        name="M2 the optional marker is ignored, so an absent optional file is fatal",
        file=SCRIPT,
        old='''    case $line in
      *"(ignore_errors=yes)"*) flag=optional ;;
    esac''',
        new='''    case $line in
      *"(ignore_errors=yes)"*) : ;;
    esac''',
        caught_by="test_a_missing_optional_environment_file_is_skipped",
    ),
    dict(
        name="M3 a missing required file is not checked before validation",
        file=SCRIPT,
        old='''    die "the unit declares ${ENV_PATHS[$i]} but it is missing: systemd would refuse to start $UNIT, so no candidate can be validated under the service's real environment"''',
        new='''    : "the missing file is left to the validator"''',
        caught_by="test_a_missing_required_environment_file_is_refused",
    ),
    dict(
        name="M4 read order is discarded by reversing the resolved list",
        file=SCRIPT,
        old='''    ENV_FLAGS+=("$flag")
    ENV_PATHS+=("$path")''',
        new='''    ENV_FLAGS=("$flag" ${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"})
    ENV_PATHS=("$path" ${ENV_PATHS[@]+"${ENV_PATHS[@]}"})''',
        caught_by="test_a_later_file_wins_when_two_declare_the_same_variable",
    ),
    dict(
        name="M5 only the first resolved file is used",
        file=SCRIPT,
        old='resolve_environment_files\n  say "the unit reads',
        new='''resolve_environment_files
  ENV_FLAGS=("${ENV_FLAGS[0]}")
  ENV_PATHS=("${ENV_PATHS[0]}")
  say "the unit reads''',
        caught_by="test_a_credential_that_exists_only_in_the_fourth_file_is_what_loads",
    ),
    dict(
        name="M6 the release record names only the first file",
        file=SCRIPT,
        old='''  for i in "${!ENV_PATHS[@]}"; do
    path=${ENV_PATHS[$i]}
    required=true''',
        new='''  for i in 0; do
    path=${ENV_PATHS[$i]}
    required=true''',
        caught_by="test_the_release_record_lists_four_files_in_read_order_with_hashes",
    ),
    dict(
        name="M7 the validator reads a missing file as an empty one",
        file=VALIDATOR,
        old='''    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such environment file")''',
        new='''    if not os.path.isfile(path):
        return {}''',
        caught_by="test_the_validator_refuses_an_environment_file_it_cannot_read",
    ),
    dict(
        name="M8 a quoted path containing a space is split",
        file=SCRIPT,
        old='''        \\"*)
          path=${line#\\"}
          case $path in
            *\\"*) ;;
            *) return 1 ;;
          esac
          rest=${path#*\\"}
          path=${path%%\\"*}
          line=$rest
          ;;''',
        new='''        \\"*)
          path=${line%%[[:space:]]*}
          line=${line#"$path"}
          ;;''',
        caught_by="test_a_quoted_path_with_a_space_is_taken_whole",
    ),
    dict(
        name="M9 an empty discovery is not treated as a failure",
        file=SCRIPT,
        old='''  [ -n "$raw" ] || return 1
  while IFS= read -r line; do''',
        new='''  while IFS= read -r line; do''',
        also=(('''  done <<< "$raw"
  [ "$emitted" -gt 0 ]
}''', '''  done <<< "$raw"
  return 0
}'''),),
        caught_by="test_discovery_that_answers_nothing_is_refused_rather_than_guessed",
    ),
]


def run_case(tree, case):
    env = dict(os.environ)
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    completed = subprocess.run(
        [sys.executable, os.path.join(tree, "test_activate_release.py"), case],
        cwd=tree, capture_output=True, text=True, env=env, timeout=900)
    return completed


def apply_edits(text, pairs):
    """Apply every (old, new) pair, each exactly once. Returns (text, problem)."""
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            return text, f"anchor matched {count} times: {old.splitlines()[0][:60]!r}"
        text = text.replace(old, new, 1)
    return text, None


def main():
    if os.geteuid() != 0:
        print("must run as root (the suite refuses otherwise)", file=sys.stderr)
        return 2

    failures = 0
    for mutant in MUTANTS:
        with tempfile.TemporaryDirectory(prefix="envmut-") as scratch:
            tree = pathlib.Path(scratch) / "tree"
            tree.mkdir()
            for name in FILES:
                shutil.copy2(HERE / name, tree / name)

            target = tree / mutant["file"]
            pairs = [(mutant["old"], mutant["new"])] + list(mutant.get("also", ()))
            text, problem = apply_edits(target.read_text(), pairs)
            if problem:
                print(f"FAIL {mutant['name']}: {problem}")
                failures += 1
                continue

            target.write_text(text)

            result = run_case(str(tree), mutant["caught_by"])
            if result.returncode == 0:
                print(f"SURVIVED {mutant['name']} "
                      f"(case {mutant['caught_by']} still passed)")
                failures += 1
            else:
                tail = (result.stdout + result.stderr).strip().splitlines()
                print(f"caught   {mutant['name']} -> {tail[-1] if tail else ''}")

    print(f"\n{len(MUTANTS) - failures}/{len(MUTANTS)} mutants caught")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
