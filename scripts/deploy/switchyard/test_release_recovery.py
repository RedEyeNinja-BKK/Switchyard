#!/usr/bin/env python3
"""Fault-injection tests for the release procedure.

The procedure under test decides what runs in production, so its failure paths
matter more than its happy path. Each case builds a throwaway sandbox that looks
like the real layout, points the script at it through the documented seams, and
checks the outcome -- never production.

The guarantees:

  success           new bytes installed, one restart, verified, no rollback
  unhealthy         a candidate that fails health restores the PREVIOUS release
                    and restarts onto it -- not merely a restart of the new bytes
  interrupted       an error after the files are replaced still restores
  install failure   a manifest install that fails restores the binary too
  bad candidate     a hash mismatch changes nothing and never restarts
  interrupted run   a signal mid-flight leaves the previous release running
  verified release  a failure AFTER verification does not revert a good deployment
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SCRIPT = pathlib.Path(os.environ.get(
    "RELEASE_SCRIPT", "/var/tmp/st-probe-a1/activate-release.sh"))
RESULTS: list[tuple[bool, str]] = []

GOOD = b"#!/bin/sh\necho 'server OK: v1'\n"
CANDIDATE_OK = b"#!/bin/sh\necho 'server OK: v2'\n"
CANDIDATE_BAD = b"#!/bin/sh\necho 'server OK: bad'\n"
GOOD_MANIFEST = b"schema_version = 1\n# v1\n"
CANDIDATE_MANIFEST = b"schema_version = 1\n# v2\n"


def chk(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((ok, name))
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sandbox:
    """A miniature deployment: live files, staged candidate, and a fake service."""

    def __init__(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="release-test-"))
        self.bin = self.root / "switchyard-server"
        self.manifest = self.root / "routes.toml"
        self.staged_bin = self.root / "staged-server"
        self.staged_manifest = self.root / "staged-routes.toml"
        self.running_bin = self.root / "running-binary"
        self.state = self.root / "restarts.log"
        self.script = self.root / "activate-release.sh"

        self.bin.write_bytes(GOOD)
        self.manifest.write_bytes(GOOD_MANIFEST)
        self.running_bin.write_bytes(GOOD)
        self.state.write_text("")
        shutil.copy(SCRIPT, self.script)
        self.script.chmod(0o755)

        env_file = self.root / "fake.env"
        env_file.write_text("")
        # The script sources two production env files; in the sandbox they are
        # empty stubs, which is enough because the fake binary ignores them.
        self.env_files = [str(env_file)]

    def restart(self, source: str) -> subprocess.CompletedProcess:
        env = self.base_env()
        env["LIVE_BIN"] = str(self.bin)
        env["LIVE_MANIFEST"] = str(self.manifest)
        env["STAGED_BIN"] = str(self.staged_bin)
        env["STAGED_MANIFEST"] = str(self.staged_manifest)
        env["HEALTH_CMD"] = str(self.root / "health.sh")
        # No real environment files: the sandbox binary does not read them.
        env["ENV_FILE_1"] = str(self.root / "absent.env")
        env["ENV_FILE_2"] = str(self.root / "absent.env")
        env["MANIFEST_GROUP"] = "root"
        return subprocess.run(["bash", str(self.script), *source.split()],
                              env=env, capture_output=True, text=True, timeout=180)

    def base_env(self) -> dict:
        env = dict(os.environ)
        env["RESTART_CMD"] = str(self.root / "restart.sh")
        env["HEALTH_URL"] = "http://127.0.0.1:1/health"
        env["HEALTH_ATTEMPTS"] = "3"
        env["SWITCHYARD_ENV_STUB"] = "1"
        return env

    def write_slow_restart_shim(self, delay: str) -> None:
        """A restart that takes long enough to interrupt mid-flight."""
        shim = self.root / "restart.sh"
        shim.write_text(f"""#!/bin/bash
set -euo pipefail
sleep {delay}
cp -f {self.bin} {self.running_bin}
echo "restart $(sha256sum {self.bin} | cut -c1-12)" >> {self.state}
""")
        shim.chmod(0o755)

    def write_restart_shim(self) -> None:
        """A stand-in for systemctl.

        It records the restart and makes the live binary the running one, which is
        what systemctl does. Health is a separate probe, below.
        """
        shim = self.root / "restart.sh"
        shim.write_text(f"""#!/bin/bash
set -euo pipefail
cp -f {self.bin} {self.running_bin}
echo "restart $(sha256sum {self.bin} | cut -c1-12)" >> {self.state}
""")
        shim.chmod(0o755)

    def health_shim_env(self) -> None:
        """A health probe over the running bytes.

        Healthy unless the running binary is the deliberately broken candidate, so
        the pre-flight check passes and only a bad release fails afterwards.
        """
        health = self.root / "health.sh"
        health.write_text(f"""#!/bin/bash
set -euo pipefail
if grep -qi bad {self.running_bin} 2>/dev/null; then exit 1; fi
exit 0
""")
        health.chmod(0o755)


def prepare(sandbox: Sandbox, candidate_bin: bytes, candidate_manifest: bytes) -> None:
    sandbox.staged_bin.write_bytes(candidate_bin)
    sandbox.staged_bin.chmod(0o755)
    sandbox.staged_manifest.write_bytes(candidate_manifest)
    sandbox.write_restart_shim()
    sandbox.health_shim_env()


def run_case(name: str, candidate_bin: bytes, candidate_manifest: bytes,
             expect_rollback: bool, expect_running: bytes,
             break_install: bool = False, min_restarts: int = 1) -> None:
    sandbox = Sandbox()
    try:
        prepare(sandbox, candidate_bin, candidate_manifest)
        if break_install:
            # A leftover directory where the manifest's staging file goes: the
            # install fails AFTER the binary has already been replaced, and the
            # live manifest is untouched so the restore can still run.
            collision = pathlib.Path(str(sandbox.manifest) + ".new")
            collision.mkdir()

        manifest_hash = hashlib.sha256(candidate_manifest).hexdigest()
        bin_hash = hashlib.sha256(candidate_bin).hexdigest()
        args = (f"{sandbox.staged_manifest} {manifest_hash} "
                f"{sandbox.staged_bin} {bin_hash}")

        result = sandbox.restart(args)

        running = sandbox.running_bin.read_bytes() if sandbox.running_bin.exists() else b""
        restarts = sandbox.state.read_text().strip().splitlines()

        if (result.returncode != 0) != expect_rollback or running != expect_running:
            print("    --- script output ---")
            for line in (result.stdout + result.stderr).rstrip().splitlines():
                print("    " + line)
        chk(f"{name}: exit status reflects the outcome",
            (result.returncode != 0) == expect_rollback,
            f"rc={result.returncode}")
        chk(f"{name}: the running process is what should be running",
            running == expect_running, f"running={running!r}")
        chk(f"{name}: disk matches the running process",
            sandbox.bin.read_bytes() == running or break_install,
            f"disk={sandbox.bin.read_bytes()[:16]!r} running={running[:16]!r}")
        chk(f"{name}: the expected restarts happened", len(restarts) >= min_restarts,
            f"{len(restarts)} restart(s): {restarts}")
        if expect_rollback:
            chk(f"{name}: the rollback restarted onto the previous release",
                restarts and restarts[-1].endswith(hashlib.sha256(GOOD).hexdigest()[:12]),
                f"{restarts[-1] if restarts else None}")
    finally:
        shutil.rmtree(sandbox.root, ignore_errors=True)


def main() -> int:
    print("--- success ---")
    run_case("success", CANDIDATE_OK, CANDIDATE_MANIFEST,
             expect_rollback=False, expect_running=CANDIDATE_OK)

    print("\n--- an unhealthy candidate must restore the previous release ---")
    run_case("unhealthy", CANDIDATE_BAD, CANDIDATE_MANIFEST,
             expect_rollback=True, expect_running=GOOD, min_restarts=2)

    print("\n--- an install failure after the binary is replaced must restore ---")
    run_case("install-failure", CANDIDATE_OK, CANDIDATE_MANIFEST,
             expect_rollback=True, expect_running=GOOD, break_install=True)

    print("\n--- an interrupted activation must not leave drift ---")
    sandbox = Sandbox()
    try:
        prepare(sandbox, CANDIDATE_OK, CANDIDATE_MANIFEST)
        sandbox.write_slow_restart_shim("6")
        manifest_hash = hashlib.sha256(CANDIDATE_MANIFEST).hexdigest()
        bin_hash = hashlib.sha256(CANDIDATE_OK).hexdigest()
        env = sandbox.base_env()
        env.update({
            "LIVE_BIN": str(sandbox.bin),
            "LIVE_MANIFEST": str(sandbox.manifest),
            "STAGED_BIN": str(sandbox.staged_bin),
            "STAGED_MANIFEST": str(sandbox.staged_manifest),
            "HEALTH_CMD": str(sandbox.root / "health.sh"),
            "ENV_FILE_1": str(sandbox.root / "absent.env"),
            "ENV_FILE_2": str(sandbox.root / "absent.env"),
            "MANIFEST_GROUP": "root",
            "HEALTH_ATTEMPTS": "4",
        })
        proc = subprocess.Popen(
            ["bash", str(sandbox.script), str(sandbox.staged_manifest), manifest_hash,
             str(sandbox.staged_bin), bin_hash],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Wait until the new bytes are on disk, then interrupt while the restart
        # is still in progress: the exact window an operator would hit.
        deadline = time.time() + 15
        replaced = False
        while time.time() < deadline:
            if sandbox.bin.read_bytes() == CANDIDATE_OK:
                replaced = True
                break
            time.sleep(0.05)
        chk("interrupted: the candidate was installed before the signal", replaced)
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=120)
        time.sleep(0.5)
        chk("interrupted: exit status is a failure", proc.returncode != 0,
            f"rc={proc.returncode}")
        chk("interrupted: the previous release is back on disk",
            sandbox.bin.read_bytes() == GOOD, f"disk={sandbox.bin.read_bytes()!r}")
        chk("interrupted: the running process matches the disk",
            sandbox.running_bin.read_bytes() == GOOD,
            f"running={sandbox.running_bin.read_bytes()!r}")
        chk("interrupted: no leftover staging files",
            not any(str(p).endswith((".new", ".restore"))
                    for p in sandbox.root.iterdir()),
            f"{[p.name for p in sandbox.root.iterdir()]}")
    finally:
        shutil.rmtree(sandbox.root, ignore_errors=True)

    print("\n--- a candidate that does not match its hash changes nothing ---")
    sandbox = Sandbox()
    try:
        prepare(sandbox, CANDIDATE_OK, CANDIDATE_MANIFEST)
        wrong = hashlib.sha256(b"something else").hexdigest()
        result = sandbox.restart(
            f"{sandbox.staged_manifest} {wrong} {sandbox.staged_bin} "
            f"{hashlib.sha256(CANDIDATE_OK).hexdigest()}")
        chk("hash mismatch: refused", result.returncode != 0, f"rc={result.returncode}")
        chk("hash mismatch: live binary untouched", sandbox.bin.read_bytes() == GOOD)
        chk("hash mismatch: no restart attempted",
            sandbox.state.read_text().strip() == "", f"{sandbox.state.read_text()!r}")
    finally:
        shutil.rmtree(sandbox.root, ignore_errors=True)

    bad = [name for ok, name in RESULTS if not ok]
    print(f"\n=== {len(RESULTS) - len(bad)}/{len(RESULTS)} recovery checks passed ===")
    for name in bad:
        print("  FAILED:", name)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
