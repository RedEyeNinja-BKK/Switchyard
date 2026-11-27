#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# A candidate manifest that the P9 direct-capability contract accepts. The suite
# below is about restore mechanics, not routing content, so every candidate it
# deploys carries the smallest manifest that passes P9 -- which keeps the REAL
# gate in the deploy path instead of bypassing it with the --command test seam.
# (The seam is deliberately forbidden for the production release store.)
P9_VALID_MANIFEST = """\
[routes.direct-fixture]
type = "passthrough"
target = "fixture-target"
tool_calling = true
vision = false
video = false
context_window = 8192

[targets.fixture-target]
id = "fixture/model-1"
llm_client = "fixture-client"
tool_calling = true
vision = false
video = false
context_window = 8192

[llm_clients.fixture-client]
format = "openai_chat"
"""


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
        # The unit is asked which EnvironmentFiles it reads. The sandbox answers with
        # one file by default, so the answer never depends on whether the host runs a
        # unit with this name. Cases that need a different answer -- four files, a
        # missing one, an empty or failing `systemctl` -- build their own stub.
        self.default_systemctl = self.write_systemctl("systemctl-default", [self.env1])

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
            "SYSTEMCTL": self.default_systemctl,
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

    def write_env_file(self, name, assignments):
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            for key, value in assignments.items():
                fh.write(f"{key}={value}\n")
        return path

    def write_systemctl(self, name, output, markers=True, optional=()):
        """A stub for `systemctl show -p EnvironmentFiles --value`.

        `markers=True` reproduces current systemd, one `PATH (ignore_errors=no)` per
        line; `markers=False` reproduces the older single-line form. Both shapes are
        in the wild, so both are tested. Paths in `optional` are reported with
        `ignore_errors=yes`, which is how a unit says the file may be absent.
        """
        path = os.path.join(self.root, name)

        def quote(item):
            # systemd quotes a path that contains a space, and only then.
            return f'"{item}"' if " " in item else item

        with open(path, "w") as fh:
            fh.write("#!/bin/bash\n")
            if markers:
                for item in output:
                    flag = "yes" if item in optional else "no"
                    argument = f"{quote(item)} (ignore_errors={flag})"
                    fh.write(f"printf '%s\\n' '{argument}'\n")
            else:
                argument = " ".join(quote(item) for item in output)
                fh.write(f"printf '%s\\n' '{argument}'\n")
        os.chmod(path, 0o755)
        return path

    def write_systemctl_silent(self, name, exit_code=0):
        """A `systemctl` that answers nothing: not installed, or a unit it cannot read."""
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            fh.write(f"#!/bin/bash\nexit {exit_code}\n")
        os.chmod(path, 0o755)
        return path

    def write_candidate(self, name, body):
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            fh.write(body)
        os.chmod(path, 0o755)
        return path, sha256(path)

    def write_manifest(self, name, body=P9_VALID_MANIFEST):
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


def test_a_candidate_that_fails_p9_is_refused_before_anything_is_installed(box):
    """The negative control for the new release gate.

    The deploy path must refuse a manifest the P9 contract rejects, and the
    refusal must happen while the live pair is still the old one. Without this
    case, a broken checker or a bypassed validator would leave every other test
    in this file passing.
    """
    adopted(box)
    before = box.live_bytes()
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    # Structurally valid TOML with a direct alias that has no target: the
    # installer's own checks pass, so only P9 can refuse it.
    manifest, msha = box.write_manifest(
        "candidate.toml",
        '[routes.direct-unqualified]\ntype = "passthrough"\ntarget = "absent"\n')
    result = box.run("deploy", manifest, msha, candidate, csha)
    assert result.returncode != 0, "a P9-refused candidate must not deploy"
    assert "P9 direct-route capability release gate" in result.stderr
    assert box.live_bytes() == before, "the refused candidate must not be installed"


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
# which environment the candidate is validated under
#
# The unit declares its EnvironmentFiles. This script used to know two of them by
# name, which was correct until a third was added as a drop-in -- at which point a
# candidate that needed only that third file was refused, with a message naming a
# variable the operator could see on disk. The unit is asked instead; these four
# cases hold that behaviour in place.
# ---------------------------------------------------------------------------


def _three_file_setup(box):
    """Three env files, and a candidate that loads only when the third is read."""
    files = (
        box.write_env_file("one.env", {"FIRST_FILE_KEY": "one"}),
        box.write_env_file("two.env", {"SECOND_FILE_KEY": "two"}),
        box.write_env_file("three.env", {"THIRD_FILE_KEY": "from-the-third-file"}),
    )
    candidate, csha = box.write_candidate(
        "candidate",
        "#!/bin/bash\n"
        '[ "${THIRD_FILE_KEY:-}" = "from-the-third-file" ] || exit 3\n'
        'echo "server OK: fixture"\n')
    manifest, msha = box.write_manifest("candidate.toml")
    return files, (candidate, csha, manifest, msha)


def test_the_candidate_is_validated_under_every_environment_file_the_unit_declares(box):
    adopted(box)
    files, (candidate, csha, manifest, msha) = _three_file_setup(box)
    systemctl = box.write_systemctl("systemctl-three", files)
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_unit_file_this_script_never_heard_of_is_not_skipped(box):
    """The negative control for the case above.

    Same candidate, same first two files; the stub now reports only those two. The
    candidate must be refused -- which is what proves the pass came from the third
    file being read, not from validation being toothless.
    """
    adopted(box)
    files, (candidate, csha, manifest, msha) = _three_file_setup(box)
    before = box.live_bytes()
    systemctl = box.write_systemctl("systemctl-two", files[:2])
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 1
    assert "does not validate" in result.stderr
    assert box.live_bytes() == before


def test_the_older_single_line_environment_file_list_is_read_too(box):
    adopted(box)
    files, (candidate, csha, manifest, msha) = _three_file_setup(box)
    systemctl = box.write_systemctl("systemctl-old", files, markers=False)
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_release_record_names_every_file_it_was_validated_under(box):
    """The record is what an auditor reads; it must not name a subset."""
    adopted(box)
    files, (candidate, csha, manifest, msha) = _three_file_setup(box)
    systemctl = box.write_systemctl("systemctl-three", files)
    box.mark_good(candidate)
    assert box.run("deploy", manifest, msha, candidate, csha,
                   SYSTEMCTL=systemctl).returncode == 0
    text = open(os.path.join(box.releases, box.current(), "release.json")).read()
    for path in files:
        assert path in text, f"{path} must be recorded"
        assert sha256(path) in text, f"{path} must be recorded by hash"


# ---------------------------------------------------------------------------
# the production shape: four files, a credential in the fourth, and the ways
# discovery can be wrong
#
# The host that motivated this has four EnvironmentFiles: two on the unit and two
# drop-ins. The fourth is a credential file, so a validator that reads the first
# three cannot load a candidate that needs it -- which is exactly what happened,
# and what a hand-named pair of files could never have caught.
# ---------------------------------------------------------------------------


def _four_file_setup(box):
    """Four files where the credential the candidate needs is only in the fourth."""
    files = (
        box.write_env_file("f1.env", {"FIRST_FILE_KEY": "one"}),
        box.write_env_file("f2.env", {"SECOND_FILE_KEY": "two"}),
        box.write_env_file("f3.env", {"THIRD_FILE_KEY": "three"}),
        box.write_env_file("f4.env", {"FOURTH_FILE_KEY": "from-the-fourth-file"}),
    )
    candidate, csha = box.write_candidate(
        "candidate",
        "#!/bin/bash\n"
        '[ "${FOURTH_FILE_KEY:-}" = "from-the-fourth-file" ] || exit 3\n'
        'echo "server OK: fixture"\n')
    manifest, msha = box.write_manifest("candidate.toml")
    return files, (candidate, csha, manifest, msha)


def test_a_credential_that_exists_only_in_the_fourth_file_is_what_loads(box):
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    systemctl = box.write_systemctl("systemctl-four", files)
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_fourth_file_is_not_optional_in_practice(box):
    """The negative control: with only the first three, this candidate must fail.

    Without this, the case above would also pass if validation were toothless.
    """
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    before = box.live_bytes()
    systemctl = box.write_systemctl("systemctl-three-of-four", files[:3])
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 1
    assert "does not validate" in result.stderr
    assert box.live_bytes() == before


def test_the_release_record_lists_four_files_in_read_order_with_hashes(box):
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    systemctl = box.write_systemctl("systemctl-four", files)
    box.mark_good(candidate)
    assert box.run("deploy", manifest, msha, candidate, csha,
                   SYSTEMCTL=systemctl).returncode == 0
    text = open(os.path.join(box.releases, box.current(), "release.json")).read()
    positions = [text.index(path) for path in files]
    assert positions == sorted(positions), "the record must preserve read order"
    for path in files:
        assert sha256(path) in text, f"{path} must be recorded by hash"


def test_a_later_file_wins_when_two_declare_the_same_variable(box):
    """Preserved order is what makes this true, for systemd and for the validator."""
    adopted(box)
    first = box.write_env_file("p1.env", {"PRECEDENCE_KEY": "loser"})
    second = box.write_env_file("p2.env", {"PRECEDENCE_KEY": "winner"})
    candidate, csha = box.write_candidate(
        "candidate-precedence",
        "#!/bin/bash\n"
        '[ "${PRECEDENCE_KEY:-}" = "winner" ] || exit 3\n'
        'echo "server OK: fixture"\n')
    manifest, msha = box.write_manifest("candidate.toml")
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha,
                     SYSTEMCTL=box.write_systemctl("systemctl-order", [first, second]))
    assert result.returncode == 0, result.stdout + result.stderr

    # And the proof that the order is what decided it: swap the files and the same
    # candidate must now be refused, because the value it needs is no longer last.
    before = box.live_bytes()
    swapped = box.write_systemctl("systemctl-swapped", [second, first])
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=swapped)
    assert result.returncode == 1
    assert box.live_bytes() == before


def test_a_missing_required_environment_file_is_refused(box):
    """systemd would refuse to start with this file absent, so validation cannot proceed."""
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    before = box.live_bytes()
    ghost = os.path.join(box.root, "never-written.env")
    systemctl = box.write_systemctl("systemctl-ghost", (files[0], ghost))
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 1
    assert "systemd would refuse to start" in result.stderr
    assert box.live_bytes() == before


def test_a_missing_optional_environment_file_is_skipped(box):
    """ignore_errors=yes is systemd's own statement that the file may be absent."""
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    ghost = os.path.join(box.root, "optional-absent.env")
    systemctl = box.write_systemctl("systemctl-optional", files + (ghost,),
                                    optional=(ghost,))
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=systemctl)
    assert result.returncode == 0, result.stdout + result.stderr
    text = open(os.path.join(box.releases, box.current(), "release.json")).read()
    assert ghost in text and '"present":false' in text


def test_discovery_that_answers_nothing_is_refused_rather_than_guessed(box):
    """The two-file fallback is gone: no answer means no deployment.

    Substituting a default here would validate the candidate under an environment
    the service never gets, and the failure would surface later as a service that
    cannot start -- with a message about a missing variable, not about this.
    """
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    before = box.live_bytes()
    silent = box.write_systemctl_silent("systemctl-silent", 0)
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=silent)
    assert result.returncode == 1
    assert "cannot resolve the EnvironmentFiles" in result.stderr
    assert "ENV_FILES" in result.stderr, "the refusal must name the way out"
    assert box.live_bytes() == before


def test_a_systemctl_that_fails_is_not_read_as_no_files(box):
    adopted(box)
    files, (candidate, csha, manifest, msha) = _four_file_setup(box)
    before = box.live_bytes()
    failing = box.write_systemctl_silent("systemctl-failing", 1)
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha, SYSTEMCTL=failing)
    assert result.returncode == 1
    assert "cannot resolve the EnvironmentFiles" in result.stderr
    assert box.live_bytes() == before


def test_a_discovered_file_that_does_not_parse_is_refused_before_the_swap(box):
    """A file systemd would reject is a file whose environment cannot be reproduced."""
    adopted(box)
    good = box.write_env_file("good.env", {"GOOD_KEY": "value"})
    bad = os.path.join(box.root, "bad.env")
    with open(bad, "w") as fh:
        fh.write("this line is not an assignment\n")
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    manifest, msha = box.write_manifest("candidate.toml")
    before = box.live_bytes()
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha,
                     SYSTEMCTL=box.write_systemctl("systemctl-bad", [good, bad]))
    assert result.returncode == 1
    assert "does not validate" in result.stderr
    assert box.live_bytes() == before


def test_a_quoted_path_with_a_space_is_taken_whole(box):
    """systemd quotes such a path; splitting it would hand the validator half a file."""
    adopted(box)
    spaced = box.write_env_file("a file with spaces.env", {"SPACED_KEY": "present"})
    candidate, csha = box.write_candidate(
        "candidate-spaced",
        "#!/bin/bash\n"
        '[ "${SPACED_KEY:-}" = "present" ] || exit 3\n'
        'echo "server OK: fixture"\n')
    manifest, msha = box.write_manifest("candidate.toml")
    box.mark_good(candidate)
    result = box.run("deploy", manifest, msha, candidate, csha,
                     SYSTEMCTL=box.write_systemctl("systemctl-spaced", [spaced]))
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_validator_refuses_an_environment_file_it_cannot_read(tmp_path):
    """The second layer: a missing file is not an empty one, whoever asks."""
    candidate = os.path.join(str(tmp_path), "candidate")
    with open(candidate, "w") as fh:
        fh.write("#!/bin/bash\nexit 0\n")
    os.chmod(candidate, 0o755)
    manifest = os.path.join(str(tmp_path), "candidate.toml")
    with open(manifest, "w") as fh:
        fh.write("route = 1\n")
    result = subprocess.run(
        [sys.executable, VALIDATOR, "--binary", candidate, "--manifest", manifest,
         "--env-file", os.path.join(str(tmp_path), "absent.env")],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "no such environment file" in result.stderr


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


def _capability_pair(box, tmp_path, *, keep_economics):
    """A deployed baseline and a candidate manifest that lost one family.

    Everything about the candidate is self-consistent: it describes its own
    binary, and it simply does not carry that family any more. Only the
    cross-release comparison against the DEPLOYED manifest can see that.

    `keep_economics` decides whether the economics subsystem survives. Losing a
    family and losing a whole subsystem are two different events to the gate, so
    both shapes are needed to state the rule.
    """
    import json

    def families(names, subsystem):
        return {name: {"instrument": name.replace("_total", "").replace("_", "."),
                       "kind": "gauge", "kind_basis": "source",
                       "subsystem": subsystem, "class": "present_only_in_deployed"}
                for name in names}

    deployed = {
        "contract_version": 1,
        "build": {"commit": "a" * 40, "binary_sha256": "0" * 64},
        "families": {**families(["switchyard_economics_provider_balance",
                                 "switchyard_economics_provider_cost"], "economics"),
                     **families(["switchyard_build_info"], "telemetry")},
        "subsystems": {"economics": ["switchyard_economics_provider_balance",
                                     "switchyard_economics_provider_cost"],
                       "telemetry": ["switchyard_build_info"]},
    }
    binary, bsha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    kept = ["switchyard_economics_provider_cost"] if keep_economics else []
    candidate = {
        "contract_version": 1,
        "build": {"commit": "b" * 40, "binary_sha256": bsha},
        "families": {**families(kept, "economics"),
                     **families(["switchyard_build_info"], "telemetry")},
        "subsystems": ({"economics": kept, "telemetry": ["switchyard_build_info"]}
                       if keep_economics else
                       {"telemetry": ["switchyard_build_info"]}),
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(deployed))
    caps = tmp_path / "candidate-caps.json"
    caps.write_text(json.dumps(candidate))
    return binary, baseline, caps


def test_an_approved_removal_is_accepted_and_an_unapproved_one_is_refused(tmp_path):
    """Approval has to be per family and carry a reason; silence is not consent."""
    import json

    tool = os.path.join(os.path.dirname(SCRIPT), "capability_contract.py")
    binary, baseline, caps = _capability_pair(_box(tmp_path), tmp_path,
                                              keep_economics=True)

    def preflight(*extra):
        return subprocess.run(
            [sys.executable, tool, "preflight", "--binary", binary,
             "--manifest", str(caps), "--baseline", str(baseline), *extra],
            capture_output=True, text=True)

    refused = preflight()
    assert refused.returncode == 7, refused.stdout + refused.stderr
    assert "switchyard_economics_provider_balance" in refused.stdout
    assert "UNAPPROVED" in refused.stdout

    approvals = tmp_path / "approvals.json"
    approvals.write_text(json.dumps({"removed": {
        "switchyard_economics_provider_balance":
            "the provider balance gauge was retired with the economics module"}}))
    accepted = preflight("--approvals", str(approvals))
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert "APPROVED" in accepted.stdout

    # An approval for a different family must not cover this one.
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"removed": {"switchyard_something_else_total": "x"}}))
    still_refused = preflight("--approvals", str(elsewhere))
    assert still_refused.returncode == 7, still_refused.stdout + still_refused.stderr

    # Losing every family in a subsystem is a second, separate event: the
    # families alone do not cover it, because a subsystem that vanished takes its
    # whole capability surface with it.
    binary, baseline, caps = _capability_pair(_box(tmp_path), tmp_path,
                                              keep_economics=False)
    both = tmp_path / "both.json"
    both.write_text(json.dumps({"removed": {
        "switchyard_economics_provider_balance": "retired",
        "switchyard_economics_provider_cost": "retired"}}))
    lost_subsystem = preflight("--approvals", str(both))
    assert lost_subsystem.returncode == 7, lost_subsystem.stdout + lost_subsystem.stderr
    assert "subsystems that disappeared entirely" in lost_subsystem.stdout

    with_subsystem = tmp_path / "with-subsystem.json"
    with_subsystem.write_text(json.dumps({"removed": {
        "switchyard_economics_provider_balance": "retired",
        "switchyard_economics_provider_cost": "retired",
        "economics": "the module was removed, so its subsystem went with it"}}))
    whole = preflight("--approvals", str(with_subsystem))
    assert whole.returncode == 0, whole.stdout + whole.stderr


def test_the_deploy_wrapper_passes_the_approvals_file_to_the_gate(box, tmp_path):
    """The flag has to reach the tool, not merely be accepted by the parser."""
    import json

    baseline = adopted(box)
    with open(os.path.join(box.releases, baseline, "capabilities.json"), "w") as fh:
        json.dump({"families": {}}, fh)

    recorded = os.path.join(box.root, "gate-argv.txt")
    stub = os.path.join(box.root, "capability_stub.py")
    with open(stub, "w") as fh:
        fh.write("import sys\n")
        fh.write(f"open({recorded!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n")
        fh.write("sys.exit(0)\n")
    os.chmod(stub, 0o644)

    approvals = tmp_path / "approvals.json"
    approvals.write_text(json.dumps({"removed": {"switchyard_old_total": "retired"}}))
    candidate, csha = box.write_candidate("candidate", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("candidate.toml")
    caps = tmp_path / "caps.json"
    caps.write_text("{}")

    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--capabilities", str(caps),
                     "--capability-approvals", str(approvals),
                     CAPABILITY_TOOL=stub)
    assert result.returncode == 0, result.stdout + result.stderr
    with open(recorded) as fh:
        argv = fh.read()
    assert "--approvals" in argv, argv
    assert str(approvals) in argv, argv


# ---------------------------------------------------------------------------
# serialisation: one writer at a time, decided before any state is read
#
# Two people deploying at once is not a hypothetical on this host: a deployment
# takes minutes, and the failure mode is silent -- each writer snapshots the same
# CURRENT, each believes it owns the rollback target, and the one that fails later
# restores over the release the other just activated. The lock is the fix, and
# these cases are what prove it is taken, that it is taken before the store is
# read, and that a refused writer leaves nothing behind.
# ---------------------------------------------------------------------------

def _hold_release_lock(box):
    """Take the store's own lock, the way a competing operation would.

    flock is per open-file-description, so holding it here blocks the script's
    `flock` for as long as this descriptor stays open -- which is exactly the
    contention the deploy path has to survive.
    """
    import fcntl
    path = os.path.join(box.releases, ".release.lock")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _stable(sandbox):
    """What the store looks like, for proving a refusal changed nothing."""
    names = sorted(os.listdir(sandbox.releases))
    return (names, sandbox.current(), sandbox.lkg(), sandbox.live_bytes())


def test_a_candidate_that_changes_while_waiting_is_refused_under_the_lock(box):
    """The pre-lock hash check is a fast rejection, not a proof.

    The candidate path is normally a staging directory under /var/tmp, which anyone
    can write, so the bytes compared before the lock was taken are not necessarily
    the bytes that would be installed. This drives the real TOCTOU: the manifest is
    approved and correct while the script waits, and is rewritten by another process
    during the wait. The lock is released only after the swap, so the deployment must
    re-derive the digest under the lock and refuse rather than install an unapproved
    manifest whose recorded hash would then be a lie.
    """
    baseline = adopted(box)
    candidate, csha = box.write_candidate("swap-bin", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("swap.toml")
    approved = open(manifest).read()
    before = _stable(box)

    fd = _hold_release_lock(box)
    swapper = subprocess.Popen(
        ["bash", "-c",
         f"while [ ! -e {box.root}/go ]; do sleep 0.05; done; "
         f"printf '%s' 'route = tampered\\n' > {manifest}"])
    try:
        proc = subprocess.Popen(
            ["bash", SCRIPT, "deploy", manifest, msha, candidate, csha,
             "--label", "swapped", "--capabilities", ""],
            env=box.env(RELEASE_LOCK_TIMEOUT="120"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(1.0)          # the deploy is blocked on the lock by now
        open(os.path.join(box.root, "go"), "w").close()   # release the swapper
        swapper.wait(timeout=30)
        os.close(fd)             # only now can the deployment proceed
        out, err = proc.communicate(timeout=180)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    text = out + err
    assert proc.returncode != 0, (
        "a manifest rewritten while the deployment waited must be refused, not "
        "installed: " + text)
    assert "changed while this deployment waited" in text, text
    assert open(manifest).read() != approved, "the fixture must actually have tampered"

    # Nothing was installed: the store and the live pair are as they were.
    assert _stable(box) == before, text
    assert box.current() == baseline


def test_a_lock_file_belonging_to_another_account_is_refused(box):
    """A lock another account can replace cannot serialise writers.

    Only asserted as root, which is how the suite runs: an unprivileged run owns its
    own sandbox lock legitimately. The point is the production case -- a release store
    whose lock is not root-owned is one where a second account could delete and
    recreate the inode, leaving the holder asleep on a file nobody else will open.
    """
    if os.geteuid() != 0:
        return
    adopted(box)
    lock = os.path.join(box.releases, ".release.lock")
    with open(lock, "w"):
        pass
    os.chmod(lock, 0o666)          # world-writable: replaceable by any account

    candidate, csha = box.write_candidate("perm-bin", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("perm.toml")
    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--label", "perm", "--capabilities", "")

    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "world-writable" in text or "group- or world-writable" in text, text


def test_the_release_lock_is_stable_across_the_whole_deployment(box):
    """The lock inode is never unlinked, so a contender locks the same object.

    Comparing inode NUMBERS does not prove this: a file deleted and recreated in the
    same directory is routinely handed the same inode number, so that check passes on
    exactly the defect it was written for (measured -- the mutant survived it). The
    reuse-immune proof is a descriptor held across the deployment: unlinking the lock
    drops that descriptor's link count to zero, and no reallocation can undo it.
    """
    adopted(box)
    candidate, csha = box.write_candidate("stable-bin", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("stable.toml")

    lock = os.path.join(box.releases, ".release.lock")
    if not os.path.exists(lock):
        with open(lock, "w"):
            pass
    holder = os.open(lock, os.O_WRONLY | os.O_APPEND)
    try:
        assert os.fstat(holder).st_nlink == 1
        result = box.run("deploy", manifest, msha, candidate, csha,
                         "--label", "stable", "--capabilities", "")
        assert result.returncode == 0, result.stdout + result.stderr
        assert os.fstat(holder).st_nlink == 1, (
            "the deployment unlinked the release lock; a contender that opened the "
            "old file would then be locking an object the store no longer listens "
            "to, and both writers would proceed while believing they were alone")
    finally:
        os.close(holder)


def test_a_child_does_not_inherit_the_release_lock(box):
    """A child that keeps fd 9 would hold the lock after the deployment exits.

    Proved from outside, without looking at the script: the restart command is
    replaced with one that writes its own descriptor table, and the deployment is
    then required to leave the store immediately available. If the child inherited
    fd 9, a second writer would block until the child died.
    """
    adopted(box)
    candidate, csha = box.write_candidate("fd-bin", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("fd.toml")

    dumped = os.path.join(box.root, "child-fds.txt")
    restart = os.path.join(box.root, "restart-dump.sh")
    with open(restart, "w") as fh:
        fh.write("#!/bin/bash\n")
        # Ask the kernel for the descriptor itself. `ls -l /proc/$$/fd` prints
        # " 9 -> target", not the path, so a substring test for "/fd/9" could never
        # fail -- it survived the mutant that removes the close (measured).
        fh.write(f"if [ -e /proc/$$/fd/9 ]; then echo PRESENT > {dumped}; "
                 f"else echo ABSENT > {dumped}; fi\n")
        fh.write(f"if [ -f {box.state}/good ]; then cp {box.state}/good {box.state}/healthy; fi\n")
        fh.write("exit 0\n")
    os.chmod(restart, 0o755)

    result = box.run("deploy", manifest, msha, candidate, csha,
                     "--label", "fd", "--capabilities", "",
                     RESTART_CMD=restart)
    assert result.returncode == 0, result.stdout + result.stderr

    verdict = open(dumped).read().strip()
    assert verdict == "ABSENT", (
        "the restart child inherited the release lock descriptor; a child that "
        f"outlives the deploy would then hold the store (probe said {verdict!r})")

    # And the store is usable straight away, which is the consequence that matters.
    fd = _hold_release_lock(box)
    try:
        second = box.run("promote", RELEASE_LOCK_TIMEOUT="1")
    finally:
        os.close(fd)
    assert second.returncode != 0
    assert "concurrently" in (second.stdout + second.stderr), (
        "the store must be free to lock once the deployment has exited")


def test_a_second_writer_is_refused_by_the_lock_and_changes_nothing(box):
    baseline = adopted(box)
    candidate, csha = box.write_candidate("contended", "#!/bin/bash\nexit 0\n")
    box.mark_good(candidate)
    manifest, msha = box.write_manifest("contended.toml")
    before = _stable(box)

    fd = _hold_release_lock(box)
    try:
        result = box.run("deploy", manifest, msha, candidate, csha,
                         "--label", "contended", RELEASE_LOCK_TIMEOUT="1")
    finally:
        os.close(fd)

    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert ".release.lock" in text, text
    assert "concurrently" in text, text
    assert _stable(box) == before, (
        "a refused writer must not have installed, stored, pointed at or restored "
        "anything: " + text
    )
    assert box.current() == baseline


def test_argument_checks_do_not_wait_on_the_lock(box):
    """The lock is taken before the store is read, not before the arguments are checked.

    A malformed invocation has to fail instantly instead of waiting out a timeout to
    be told its hash is wrong. Proving that also pins the ordering that matters: the
    hash comparison runs outside the lock, so everything after it -- the environment
    discovery, the validation, the swap, the readback -- is inside.
    """
    adopted(box)
    candidate, csha = box.write_candidate("instant", "#!/bin/bash\nexit 0\n")
    manifest, _ = box.write_manifest("instant.toml")

    fd = _hold_release_lock(box)
    started = time.monotonic()
    try:
        # A lock budget far beyond the assertion window: if the lock were taken
        # first, this would stall for two minutes and the elapsed check would fail.
        result = box.run("deploy", manifest, "0" * 64, candidate, csha,
                         RELEASE_LOCK_TIMEOUT="120")
    finally:
        os.close(fd)
    elapsed = time.monotonic() - started

    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "manifest hash mismatch" in text, text
    assert "concurrently" not in text, (
        "the hash check must not be gated by the lock: " + text
    )
    assert elapsed < 10, f"a hash mismatch waited {elapsed:.1f}s on the release lock"


def test_two_concurrent_deploys_serialise_into_a_consistent_store(box):
    """Both writers succeed, in order, and the store describes exactly what is live.

    The first deployment is made slow on purpose so the second is genuinely
    contending while the first holds the lock. With serialisation the second reads
    the first's CURRENT as its rollback target and the pair leaves two complete
    releases, a CURRENT whose bytes are the live ones, and the adopted baseline
    still as LAST-KNOWN-GOOD (neither deploy promoted).

    Measured scope, so this is not read as more than it is: with the lock removed
    from the deploy path this case still passed, because two writes that happen to
    land in sequence leave a coherent store too. It is the end-state check -- live
    bytes, CURRENT and the stored records must agree -- not the lock detector. The
    deterministic detector is `test_a_second_writer_is_refused_by_the_lock_and_
    changes_nothing`, and it does fail when the lock is removed.
    """
    import json

    baseline = adopted(box)
    # Different bytes, both loadable: the validator runs the candidate, so a
    # non-zero exit would refuse the deploy before it ever reached the store.
    candidate_a, a_sha = box.write_candidate("concurrent-a", "#!/bin/bash\necho a\nexit 0\n")
    candidate_b, b_sha = box.write_candidate("concurrent-b", "#!/bin/bash\necho b\nexit 0\n")
    manifest, msha = box.write_manifest("concurrent.toml")

    def always_ok(name):
        path = os.path.join(box.root, name)
        with open(path, "w") as fh:
            fh.write("#!/bin/bash\nexit 0\n")
        os.chmod(path, 0o755)
        return path

    slow_health = os.path.join(box.root, "slow-health.sh")
    with open(slow_health, "w") as fh:
        fh.write("#!/bin/bash\nsleep 2\nexit 0\n")
    os.chmod(slow_health, 0o755)

    first = subprocess.Popen(
        ["bash", SCRIPT, "deploy", manifest, msha, candidate_a, a_sha,
         "--label", "first", "--capabilities", ""],
        env=box.env(HEALTH_CMD=slow_health), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(0.5)  # the first writer is inside the lock and its health wait
        second = box.run("deploy", manifest, msha, candidate_b, b_sha,
                         "--label", "second", "--capabilities", "",
                         HEALTH_CMD=always_ok("ok.sh"))
    finally:
        out1, err1 = first.communicate(timeout=120)

    assert first.returncode == 0, out1 + err1
    text2 = second.stdout + second.stderr
    assert second.returncode == 0, text2

    stored = sorted(name for name in os.listdir(box.releases)
                    if os.path.isdir(os.path.join(box.releases, name))
                    and not os.path.islink(os.path.join(box.releases, name))
                    and name != baseline)
    assert len(stored) == 2, f"both releases must survive, store holds {stored}"

    # Each stored release is internally intact: what it records is what it holds.
    recorded = {}
    for name in stored:
        directory = os.path.join(box.releases, name)
        with open(os.path.join(directory, "release.json")) as fh:
            record = json.load(fh)
        binary_sha = record["binary"]["sha256"]
        assert sha256(os.path.join(directory, "switchyard-server")) == binary_sha
        recorded[name] = binary_sha
    assert sorted(recorded.values()) == sorted([a_sha, b_sha]), recorded

    # The readback: CURRENT names a real release, and the live bytes are that one.
    current = box.current()
    assert current in recorded, f"CURRENT {current} is not one of the stored releases"
    assert box.live_bytes() == recorded[current], (
        "the live bytes must be the release CURRENT names, or the next rollback "
        "will restore the wrong thing"
    )
    assert box.lkg() == baseline, "neither deploy promoted, so the baseline stands"


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
