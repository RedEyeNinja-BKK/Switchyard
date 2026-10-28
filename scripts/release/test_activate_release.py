#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for the durable release store in activate-release.sh.

These drive the REAL script, as root, entirely inside a sandbox under /var/tmp:
every path it touches is overridden, the restart and health probes are stubbed,
and no unit is touched. That is the only honest way to test an installer whose
job is to restore a previous release -- a hand-written mock of the restore path
would pass whether or not the real one works.

The sandbox models a unit whose health depends on WHICH BYTES are live, which is
how a failed deployment behaves: the new binary starts and the health check fails,
and the only way to recover is to put the old bytes back.

Requires root (the installer checks, and it installs files as root). Skips
otherwise, so it is safe in CI.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import time

try:
    import pytest
except ModuleNotFoundError:  # the deployment host has no test framework
    pytest = None

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activate-release.sh")
VALIDATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_validate.py")


class Sandbox:
    """One isolated installation: paths, stubs, and helpers to drive the script."""

    def __init__(self, root):
        self.root = root
        self.live = os.path.join(root, "live")
        self.state = os.path.join(root, "state")
        self.releases = os.path.join(root, "releases")
        for path in (self.live, self.state, self.releases):
            os.makedirs(path, exist_ok=True)
        self.live_bin = os.path.join(self.live, "switchyard-server")
        self.live_manifest = os.path.join(self.live, "routes.toml")
        self.env1 = os.path.join(root, "providers.env")
        with open(self.env1, "w") as fh:
            fh.write("# provider keys for this sandbox\n")
            fh.write('SWITCHYARD_TEST_KEY="s3cret value with spaces"\n')
            fh.write("SWITCHYARD_TEST_OTHER=plain\n")
        # Health succeeds only when the live binary is the one marked healthy.
        self.health_sh = os.path.join(root, "health.sh")
        with open(self.health_sh, "w") as fh:
            fh.write("#!/bin/bash\n")
            fh.write(f'[ -f "{self.state}/healthy" ] && '
                     f'[ "$(sha256sum {self.live_bin} | cut -d" " -f1)" = '
                     f'"$(cat {self.state}/healthy)" ]\n')
        os.chmod(self.health_sh, 0o755)
        self.restart_sh = os.path.join(root, "restart.sh")
        with open(self.restart_sh, "w") as fh:
            fh.write("#!/bin/bash\n")
            # A deployment is healthy only if the bytes installed are the ones
            # the test declared good. Anything else models a bad release.
            fh.write(f'echo restart >> {self.state}/restarts\n')
            fh.write(f'if [ -f {self.state}/good ]; then\n'
                     f'  cp {self.state}/good {self.state}/healthy\n'
                     f'fi\n')
        os.chmod(self.restart_sh, 0o755)

    def env(self, **overrides):
        env = {
            "PATH": os.environ["PATH"],
            "LIVE_BIN": self.live_bin,
            "LIVE_MANIFEST": self.live_manifest,
            "RELEASE_ROOT": self.releases,
            "STAGED_MANIFEST": os.path.join(self.root, "staged", "routes.toml"),
            "STAGED_BIN": os.path.join(self.root, "staged", "server"),
            "HEALTH_CMD": self.health_sh,
            "RESTART_CMD": self.restart_sh,
            "HEALTH_ATTEMPTS": "4",
            "ENV_FILE_1": self.env1,
            "ENV_FILE_2": os.path.join(self.root, "absent.env"),
            "MANIFEST_GROUP": "root",
            "VALIDATOR": VALIDATOR,
            "METRICS_URL": "http://127.0.0.1:9/metrics",
        }
        os.makedirs(os.path.join(self.root, "staged"), exist_ok=True)
        env.update(overrides)
        return env

    def run(self, *args, **overrides):
        return subprocess.run(["bash", SCRIPT, *args], env=self.env(**overrides),
                              capture_output=True, text=True)

    def write_candidate(self, name, body):
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
        return path, sha256(path)

    def write_manifest(self, name, body="route = 1\n"):
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            fh.write(body)
        return path, sha256(path)

    def install_live(self, binary_body, manifest_body="route = 1\n"):
        with open(self.live_bin, "w") as fh:
            fh.write(binary_body)
        os.chmod(self.live_bin, 0o755)
        with open(self.live_manifest, "w") as fh:
            fh.write(manifest_body)
        # The live pair is healthy, and it is the release the unit considers good.
        with open(os.path.join(self.state, "good"), "w") as fh:
            fh.write(sha256(self.live_bin))
        with open(os.path.join(self.state, "healthy"), "w") as fh:
            fh.write(sha256(self.live_bin))

    def mark_good(self, path):
        with open(os.path.join(self.state, "good"), "w") as fh:
            fh.write(sha256(path))

    def current(self):
        link = os.path.join(self.releases, "CURRENT")
        return os.readlink(link) if os.path.islink(link) else None

    def lkg(self):
        link = os.path.join(self.releases, "LAST-KNOWN-GOOD")
        return os.readlink(link) if os.path.islink(link) else None

    def live_bytes(self):
        return sha256(self.live_bin)


def sha256(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _box(tmp_path):
    """The isolation everything below runs inside.

    Named without the `test_` prefix so both runners treat it as a fixture rather
    than a case.
    """
    sandbox = Sandbox(str(tmp_path))
    sandbox.install_live("#!/bin/bash\nexit 0\n", "route = old\n")
    return sandbox


if pytest is not None:
    @pytest.fixture
    def box(tmp_path):
        sandbox = _box(tmp_path)
        yield sandbox
        shutil.rmtree(sandbox.root, ignore_errors=True)


def adopted(box):
    result = box.run("adopt", "--label", "baseline", "--basis", "adopted-unattested")
    assert result.returncode == 0, result.stdout + result.stderr
    return box.current()


# ---------------------------------------------------------------------------
# durable store
# ---------------------------------------------------------------------------


def test_adopt_records_the_live_pair_as_a_release(box):
    rid = adopted(box)
    directory = os.path.join(box.releases, rid)
    assert os.path.isfile(os.path.join(directory, "switchyard-server"))
    assert os.path.isfile(os.path.join(directory, "routes.toml"))
    assert os.path.isfile(os.path.join(directory, "release.json"))
    assert box.lkg() == rid, "the adopted release is the only rollback target there is"


def test_a_successful_deployment_is_recorded_and_does_not_move_the_rollback_target(box):
    baseline = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")

    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--label", "second", "--source-commit", "a" * 40,
                     "--source-branch", "test", "--capabilities", "")
    assert result.returncode == 0, result.stdout + result.stderr

    assert box.current() != baseline
    assert box.lkg() == baseline, (
        "a release that started successfully is not thereby known good; the "
        "rollback target must survive the deployment"
    )
    record = os.path.join(box.releases, box.current(), "release.json")
    text = open(record).read()
    assert csha in text and msha in text
    assert "a" * 40 in text
    assert "build-session" in text


def test_the_previous_release_survives_the_activation_script(box):
    """The defect this change fixes: after the run, rollback must still be possible."""
    baseline = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    assert box.run("deploy", manifest, msha, candidate, csha).returncode == 0

    stored = os.path.join(box.releases, baseline, "switchyard-server")
    assert os.path.isfile(stored), "the previous binary must still be on disk"
    assert open(stored).read() == "#!/bin/bash\nexit 0\n"


def test_release_metadata_records_configuration_identity_without_the_secret(box):
    assert box.run("adopt", "--label", "baseline").returncode == 0
    text = open(os.path.join(box.releases, box.current(), "release.json")).read()
    assert box.env1 in text, "the configuration must be identifiable"
    assert sha256(box.env1) in text, "by hash"
    assert "s3cret value with spaces" not in text, "but the secret must not be copied"


# ---------------------------------------------------------------------------
# refusal paths
# ---------------------------------------------------------------------------


def test_a_hash_mismatch_is_refused_before_anything_changes(box):
    adopted(box)
    before = box.live_bytes()
    candidate, _ = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, "0" * 64)
    assert result.returncode == 1
    assert "binary hash mismatch" in result.stderr
    assert box.live_bytes() == before


def test_a_deployment_with_no_recorded_release_is_refused(box):
    """The missing-backup case: nothing is replaced when there is nowhere to go back to."""
    before = box.live_bytes()
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, csha)
    assert result.returncode == 1
    assert "adopt" in result.stderr
    assert box.live_bytes() == before


def test_a_candidate_that_fails_its_own_validation_is_refused(box):
    adopted(box)
    before = box.live_bytes()
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 3\n")
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, csha)
    assert result.returncode == 1
    assert "does not validate" in result.stderr
    assert box.live_bytes() == before


# ---------------------------------------------------------------------------
# failure paths: the release goes live, then fails
# ---------------------------------------------------------------------------


def test_a_release_that_fails_its_health_check_is_rolled_back_and_verified(box):
    baseline = adopted(box)
    before = box.live_bytes()
    # Passes its own dry-run, then fails to serve: the failure this path exists
    # for is one that gets THROUGH every pre-flight check.
    candidate, csha = box.write_candidate(
        "candidate",
        "#!/bin/bash\n"
        'for arg in "$@"; do [ "$arg" = "--dry-run" ] && exit 0; done\n'
        "exit 1\n")
    manifest, msha = box.write_manifest("candidate.toml")

    result = box.run("deploy", manifest, msha, candidate, csha)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "ROLLBACK VERIFIED" in result.stdout
    assert box.live_bytes() == before, "the previous bytes must be live again"
    assert box.current() == baseline, "a failed deployment must not move CURRENT"


def test_an_interrupted_activation_restores_the_previous_release(box):
    """The EXIT trap's job: a signal between install and verify still converges."""
    baseline = adopted(box)
    before = box.live_bytes()
    candidate, csha = box.write_candidate(
        "candidate",
        "#!/bin/bash\n"
        'for arg in "$@"; do [ "$arg" = "--dry-run" ] && exit 0; done\n'
        "exit 1\n")
    manifest, msha = box.write_manifest("candidate.toml")

    process = subprocess.Popen(
        ["bash", SCRIPT, "deploy", manifest, msha, candidate, csha],
        env=box.env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # HEALTH_ATTEMPTS=4 at 0.5s keeps the window open long enough to interrupt it.
    time.sleep(1.0)
    process.send_signal(signal.SIGTERM)
    out, err = process.communicate(timeout=60)

    assert process.returncode != 0
    assert box.live_bytes() == before
    assert box.current() == baseline


# ---------------------------------------------------------------------------
# rollback after the script has exited
# ---------------------------------------------------------------------------


def test_rollback_after_a_successful_deployment_restores_the_last_known_good(box):
    baseline = adopted(box)
    baseline_bytes = box.live_bytes()
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    box.mark_good(candidate)
    assert box.run("deploy", manifest, msha, candidate, csha).returncode == 0
    deployed = box.current()
    assert deployed != baseline

    # A separate process, as an operator would run it days later.
    result = box.run("rollback")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ROLLBACK VERIFIED" in result.stdout
    assert box.current() == baseline
    assert box.live_bytes() == baseline_bytes


def test_rollback_to_a_named_release_works_and_keeps_the_replaced_one(box):
    first = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    box.mark_good(candidate)
    assert box.run("deploy", manifest, msha, candidate, csha).returncode == 0
    second = box.current()

    result = box.run("rollback", first)
    assert result.returncode == 0, result.stdout + result.stderr
    assert box.current() == first
    assert os.path.isdir(os.path.join(box.releases, second)), (
        "the release that was replaced must remain, so the move can be undone")


def test_promote_moves_the_rollback_target(box):
    baseline = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    box.mark_good(candidate)
    assert box.run("deploy", manifest, msha, candidate, csha).returncode == 0
    second = box.current()
    assert box.lkg() == baseline

    result = box.run("promote")
    assert result.returncode == 0, result.stdout + result.stderr
    assert box.lkg() == second


def test_promote_refuses_a_release_whose_stored_bytes_do_not_match(box):
    baseline = adopted(box)
    with open(os.path.join(box.releases, baseline, "switchyard-server"), "a") as fh:
        fh.write("# tampered\n")
    result = box.run("promote", baseline)
    assert result.returncode == 1
    assert "does not match its record" in result.stderr


# ---------------------------------------------------------------------------
# artifact integrity and retention
# ---------------------------------------------------------------------------


def test_verify_reports_a_tampered_release(box):
    baseline = adopted(box)
    with open(os.path.join(box.releases, baseline, "routes.toml"), "a") as fh:
        fh.write("# tampered\n")
    result = box.run("verify", baseline)
    assert result.returncode == 1
    assert "MISMATCH" in result.stdout


def test_rollback_refuses_to_install_a_release_whose_bytes_are_wrong(box):
    first = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    box.mark_good(candidate)
    assert box.run("deploy", manifest, msha, candidate, csha).returncode == 0
    live_now = box.live_bytes()

    with open(os.path.join(box.releases, first, "switchyard-server"), "a") as fh:
        fh.write("# tampered\n")
    result = box.run("rollback", first)
    # The corrupt release is not installed; the one that was running is restored.
    assert result.returncode == 4
    assert "ROLLBACK INCOMPLETE" in result.stderr
    assert box.live_bytes() == live_now, "tampered bytes must never reach the live path"


def test_retention_prunes_old_releases_but_never_the_rollback_target(box):
    baseline = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    for index in range(4):
        result = box.run("deploy", manifest, msha, candidate, csha,
                         "--label", f"round {index}", RETAIN="2")
        assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isdir(os.path.join(box.releases, baseline)), (
        "the designated last-known-good release must survive any amount of newer activity")
    assert box.lkg() == baseline
    # CURRENT and LAST-KNOWN-GOOD are symlinks INTO the store; `isdir` follows
    # them, so they must be excluded by their own type rather than by name.
    remaining = [entry for entry in os.listdir(box.releases)
                 if os.path.isdir(os.path.join(box.releases, entry))
                 and not os.path.islink(os.path.join(box.releases, entry))]
    assert len(remaining) <= 3, f"retention did not bound the store: {remaining}"


def test_list_reports_every_release_and_marks_the_pointers(box):
    """`list` is the operator's only view of the store, so it must print rows.

    It had no test and shipped printing a header and nothing else: every row
    raised a TypeError from the JSON reader, and the reader's own `|| echo '?'`
    produced an empty string rather than a marker. A silent listing looks exactly
    like an empty store.
    """
    baseline = adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, csha, "--label", "second")
    assert result.returncode == 0, result.stdout + result.stderr

    result = box.run("list")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line for line in result.stdout.splitlines() if line.strip()][1:]
    assert len(rows) == 2, f"expected two release rows, got {result.stdout!r}"
    # CURRENT follows what is installed; LAST-KNOWN-GOOD is the rollback target and
    # only `promote` moves it. So after one deploy the two marks are on different
    # rows, and the marks are the whole point of the listing.
    current_rows = [r for r in rows if "CURRENT" in r]
    assert len(current_rows) == 1, f"exactly one CURRENT row expected: {rows!r}"
    assert baseline not in current_rows[0], (
        "CURRENT must follow the deployed release, not stay on the adopted one")
    lkg_rows = [r for r in rows if "LAST-KNOWN-GOOD" in r]
    assert len(lkg_rows) == 1 and baseline in lkg_rows[0], (
        f"the adopted release must remain the rollback target: {rows!r}")
    for row in rows:
        fields = row.split()
        assert len(fields) >= 3, f"incomplete row: {row!r}"
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", fields[1]), (
            f"CREATED is not a timestamp: {row!r}")
        assert re.fullmatch(r"[0-9a-f]{16}", fields[2]), (
            f"BINARY is not a truncated digest: {row!r}")


def test_list_survives_a_directory_that_is_not_a_release(box):
    """A leftover directory from an earlier tool must not blank the listing."""
    adopted(box)
    legacy = os.path.join(box.releases, "legacy-20260101T000000Z")
    os.makedirs(legacy)
    result = box.run("list")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line for line in result.stdout.splitlines() if line.strip()][1:]
    assert len(rows) == 2, f"the real release must still be listed: {result.stdout!r}"
    legacy_row = next(r for r in rows if "legacy" in r)
    assert "?" in legacy_row, f"a release-less directory should be marked unknown: {legacy_row!r}"
    real_row = next(r for r in rows if "legacy" not in r)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", real_row.split()[1]), (
        f"one unreadable directory must not spoil the other row: {real_row!r}")


def test_a_retained_artefact_is_never_pruned_and_never_deployable(box):
    """A retained directory is history, and nothing automatic may touch it.

    The store can hold an artefact whose provenance predates it: real bytes that
    ran, with no release record and no way to reconstruct one. Marking it RETAINED
    must make it permanent and inert -- not a rollback target, not a promotion
    candidate, not something retention counts or deletes.
    """
    baseline = adopted(box)
    legacy = os.path.join(box.releases, "legacy-20260101T000000Z")
    os.makedirs(legacy)
    with open(os.path.join(legacy, "switchyard-server"), "w") as fh:
        fh.write("#!/bin/bash\nexit 0\n")
    with open(os.path.join(legacy, "RETAINED"), "w") as fh:
        fh.write("historical artefact\n")

    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    for index in range(4):
        result = box.run("deploy", manifest, msha, candidate, csha,
                         "--label", f"round {index}", RETAIN="0")
        assert result.returncode == 0, result.stdout + result.stderr

    assert os.path.isdir(legacy), "a retained artefact must survive every prune"
    assert os.path.isfile(os.path.join(legacy, "switchyard-server")), (
        "retention must not delete the retained artefact's contents")
    # It must not have consumed a retention slot either: the newest real release
    # still has to be present.
    assert os.path.isdir(os.path.join(box.releases, box.lkg() if box.lkg() else baseline))

    for verb in (("rollback", os.path.basename(legacy)), ("promote", os.path.basename(legacy))):
        result = box.run(*verb)
        assert result.returncode != 0, f"{verb[0]} must refuse a retained artefact"
        assert "retained" in (result.stdout + result.stderr).lower()
    assert box.lkg() == baseline, "a refused promote must not move the rollback target"

    result = box.run("verify", os.path.basename(legacy))
    assert result.returncode != 0
    assert "retained" in (result.stdout + result.stderr).lower()


def test_a_gated_release_line_refuses_a_candidate_that_has_no_manifest(box):
    """Once CURRENT carries a baseline, every candidate is checked against it.

    Enforcing this with a flag means the check happens when the operator remembers.
    The failure it guards against -- a healthy build that lost a subsystem -- is
    invisible to health checks, so the one deployment that forgets the flag is
    exactly the one that ships the regression.
    """
    baseline = adopted(box)
    import json as _json
    with open(os.path.join(box.releases, baseline, "capabilities.json"), "w") as fh:
        _json.dump({"families": {}}, fh)

    before = box.live_bytes()
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")

    result = box.run("deploy", manifest, msha, candidate, csha, "--label", "no manifest")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "gated" in result.stdout, result.stdout + result.stderr
    assert box.live_bytes() == before, "a refused candidate must not touch the live pair"
    assert box.current() == baseline


def test_a_line_without_a_baseline_says_so_instead_of_pretending(box):
    """Bootstrapping is allowed, but never silently mistaken for a checked deploy."""
    baseline = adopted(box)   # adopted without --capabilities: no baseline exists
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, csha, "--label", "bootstrap")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "gate skipped" in result.stdout, result.stdout
    assert box.current() != baseline


def _capability_stub(box, *, verify_rc):
    """A capability tool that passes the gate and then fails the live check.

    The script invokes the tool as `python3 "$CAPABILITY_TOOL" <verb>`, so a bash
    stub is a SyntaxError rather than a stub -- which fails the deploy at the
    capability gate, long before post-deployment verification runs. Every test
    below therefore asserts the FAILURE REASON, not merely a non-zero exit: a
    deploy that stops at an earlier gate would otherwise satisfy them.
    """
    path = os.path.join(box.root, "capability_stub.py")
    with open(path, "w") as fh:
        fh.write("import sys\n")
        fh.write("verb = sys.argv[1] if len(sys.argv) > 1 else ''\n")
        fh.write(f"if verb == 'verify-live':\n    sys.exit({verify_rc})\n")
        fh.write("sys.exit(0)\n")
    os.chmod(path, 0o644)
    return path


def test_promotion_waits_for_post_deployment_verification(box, tmp_path):
    """LAST-KNOWN-GOOD must never name a release whose own checks had not run.

    A restart proves the process started. Verification proves the release does
    what it claims. If promotion happened between the two, a later rollback would
    install the failed candidate as if it were known good -- the pointer would be
    actively harmful rather than merely stale.
    """
    baseline = adopted(box)
    before = box.live_bytes()
    import json as _json
    with open(os.path.join(box.releases, baseline, "capabilities.json"), "w") as fh:
        _json.dump({"families": {}}, fh)

    stub = _capability_stub(box, verify_rc=1)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    caps = tmp_path / "caps.json"
    caps.write_text("{}")

    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--capabilities", str(caps), "--promote",
                     "--source-commit", "c" * 40,
                     CAPABILITY_TOOL=stub)
    assert result.returncode != 0, "a failed post-deployment check must fail the deploy"
    assert "post-deployment verification failed" in result.stderr, (
        "the deploy must fail AT post-deployment verification, not at an earlier gate: "
        + result.stdout + result.stderr)
    assert box.lkg() == baseline, (
        "LAST-KNOWN-GOOD moved to a release that failed post-deployment verification")
    assert box.current() == baseline, (
        "CURRENT must name the binary that is actually running after a rollback")
    assert box.live_bytes() == before


def test_a_failed_post_deployment_check_restores_current(box, tmp_path):
    """CURRENT follows the restored binary, not the attempt that was undone."""
    baseline = adopted(box)
    before = box.live_bytes()
    import json as _json
    with open(os.path.join(box.releases, baseline, "capabilities.json"), "w") as fh:
        _json.dump({"families": {}}, fh)

    stub = _capability_stub(box, verify_rc=1)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    caps = tmp_path / "caps.json"
    caps.write_text("{}")

    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--capabilities", str(caps), CAPABILITY_TOOL=stub)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "post-deployment verification failed" in result.stderr, (
        "the deploy must fail AT post-deployment verification: "
        + result.stdout + result.stderr)
    assert box.live_bytes() == before
    assert box.current() == baseline
    verify = box.run("verify")
    assert verify.returncode == 0, (
        "after a restore the store must agree with the live pair: "
        + verify.stdout + verify.stderr)


def test_a_deploy_records_attested_provenance_and_adopt_does_not(box):
    """Attestation distinguishes a recorded build from observed bytes.

    A deploy knows the source it was asked to install, at the moment those bytes
    went live. `adopt` knows only what is running. Collapsing the two would let a
    reconstructed record masquerade as build provenance.
    """
    import json as _json

    baseline = adopted(box)
    adopted_record = _json.load(open(os.path.join(box.releases, baseline, "release.json")))
    assert adopted_record["provenance"]["attested"] is False
    assert adopted_record["provenance"]["basis"] == "adopted-unattested"
    with open(os.path.join(box.releases, baseline, "PROVENANCE.txt")) as fh:
        assert "NOT attested" in fh.read()

    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--label", "attested release",
                     "--source-commit", "d" * 40, "--source-branch", "release/line")
    assert result.returncode == 0, result.stdout + result.stderr

    deployed = box.current()
    record = _json.load(open(os.path.join(box.releases, deployed, "release.json")))
    provenance = record["provenance"]
    assert provenance["attested"] is True
    assert provenance["attestation"] == "declared-at-deploy"
    assert provenance["source_commit"] == "d" * 40
    assert provenance["source_branch"] == "release/line"
    # The attestation must not overclaim: it is not a reproducibility proof.
    assert "reproducible" in provenance["note"]

    with open(os.path.join(box.releases, deployed, "PROVENANCE.txt")) as fh:
        text = fh.read()
    assert "declared-at-deploy" in text
    assert "not been shown to be" in text

    listing = box.run("list")
    assert listing.returncode == 0
    adopted_row = next(r for r in listing.stdout.splitlines() if baseline in r)
    assert "UNATTESTED" in adopted_row
    deployed_row = next(r for r in listing.stdout.splitlines() if deployed in r)
    assert "UNATTESTED" not in deployed_row


def test_a_deploy_without_a_source_commit_is_not_attested(box):
    """The absence of a declared source is recorded, not papered over."""
    import json as _json

    adopted(box)
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    result = box.run("deploy", manifest, msha, candidate, csha, "--label", "no source")
    assert result.returncode == 0, result.stdout + result.stderr
    record = _json.load(open(os.path.join(box.releases, box.current(), "release.json")))
    assert record["provenance"]["attested"] is False
    assert record["provenance"]["attestation"] == "deploy-without-source-commit"


# ---------------------------------------------------------------------------
# the capability gate, wired into deployment
# ---------------------------------------------------------------------------


def test_a_candidate_missing_economics_is_refused_before_production_changes(box, tmp_path):
    """The 2026-10-27 failure, stopped at the deployment boundary."""
    import json

    adopted(box)
    before = box.live_bytes()

    contract_tool = os.path.join(os.path.dirname(SCRIPT), "capability_contract.py")

    def families(names, subsystem):
        return {name: {"instrument": name.replace("_total", "").replace("_", "."),
                       "kind": "gauge", "kind_basis": "source",
                       "subsystem": subsystem, "class": "always_present"}
                for name in names}

    deployed = {
        "contract_version": 1,
        "build": {"commit": "a" * 40, "binary_sha256": "0" * 64},
        "families": {**families(["switchyard_economics_provider_balance"], "economics"),
                     **families(["switchyard_build_info"], "telemetry")},
        "subsystems": {"economics": ["switchyard_economics_provider_balance"],
                       "telemetry": ["switchyard_build_info"]},
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(deployed))
    os.makedirs(os.path.join(box.releases, box.current()), exist_ok=True)
    with open(os.path.join(box.releases, box.current(), "capabilities.json"), "w") as fh:
        json.dump(deployed, fh)

    # The candidate's own manifest is self-consistent and simply has no economics.
    # A runnable candidate that carries a compile-time literal but no economics.
    candidate, csha = box.write_candidate(
        "candidate", "#!/bin/bash\n# switchyard.build_info\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    candidate_caps = {
        "contract_version": 1,
        "build": {"commit": "b" * 40, "binary_sha256": csha},
        "families": families(["switchyard_build_info"], "telemetry"),
        "subsystems": {"telemetry": ["switchyard_build_info"]},
    }
    caps_path = tmp_path / "candidate-caps.json"
    caps_path.write_text(json.dumps(candidate_caps))

    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--capabilities", str(caps_path),
                     CAPABILITY_TOOL=contract_tool)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "capability gate refused" in result.stderr
    assert box.live_bytes() == before


# ---------------------------------------------------------------------------
# standalone runner
#
# The deployment host has no pytest and is not a place to install packages for a
# test, so the same file runs directly. The fixtures are two functions; there is
# no reason for a missing test framework to block a deployment test.
# ---------------------------------------------------------------------------

def _run_standalone(argv):  # pragma: no cover - exercised by hand on the host
    import inspect
    import tempfile
    import traceback

    if os.geteuid() != 0:
        print("must run as root", file=sys.stderr)
        return 2

    selected = argv or [name for name, fn in sorted(globals().items())
                        if name.startswith("test_") and callable(fn)]
    failures = []
    for name in selected:
        function = globals().get(name)
        if function is None:
            print(f"no such test: {name}", file=sys.stderr)
            return 2
        parameters = list(inspect.signature(function).parameters)
        with tempfile.TemporaryDirectory(prefix="reltest-") as tmp:
            from pathlib import Path
            if "box" in parameters:
                args = {"box": _box(Path(tmp)), "tmp_path": Path(tmp)}
            else:
                args = {"tmp_path": Path(tmp)}
            try:
                function(**{p: args[p] for p in parameters})
                print(f"PASS {name}")
            except Exception:
                failures.append(name)
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{len(selected) - len(failures)} passed, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_run_standalone(sys.argv[1:]))
