#!/usr/bin/env python3
"""The build chain stops on a refusal, tolerates a written-but-unresolved step.

The reconciler exits 2 when it has written its output and still reports gaps,
which is the normal mid-pipeline state. Treating that as fatal stopped the whole
candidate from ever reaching the gate, so the distinction is pinned here by
stubbing the step runner: the stub honours only the exit codes the chain allows
for that step, which is the contract under test.
"""
import importlib.util
import sys
from pathlib import Path
from unittest import mock
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "build_p9_candidate", HERE / "build_p9_candidate.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

BASELINE = """\
[routes.direct-a]
type = "passthrough"
target = "t-a"
context_window = 8192
tool_calling = true
vision = false
video = false

[targets.t-a]
id = "vendor/model-a"
llm_client = "c-a"
vision = false
video = false

[llm_clients.c-a]
format = "openai_chat"
"""


def stub_runner(codes: dict[str, int]):
    """Answer each step from its script name; ok only for an accepted code."""
    calls: list[tuple[str, str, int, tuple[int, ...]]] = []

    def _run(step, argv, accept=(0,)):
        script = Path(argv[1]).name
        code = codes.get(script, 0)
        calls.append((step, script, code, accept))
        return code in accept, ""

    return _run, calls


class ChainTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.baseline = self.dir / "baseline.toml"
        self.baseline.write_text(BASELINE)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, codes):
        runner, calls = stub_runner(codes)
        with mock.patch.object(builder, "run", runner):
            rc = builder.main(["--baseline", str(self.baseline),
                               "--out-dir", str(self.dir / "out")])
        return rc, [script for _, script, _, _ in calls], calls

    def test_a_written_but_unresolved_reconcile_does_not_stop_the_chain(self):
        rc, scripts, _ = self.build({"reconcile_direct_capability_declarations.py": 2})
        self.assertIn("direct_capability_gate.py", scripts,
                      "the gate was never reached, so rc=2 stopped the chain")
        self.assertNotEqual(rc, 1)

    def test_only_the_reconciler_accepts_a_nonzero_exit(self):
        _, _, calls = self.build({})
        for step, script, _, accept in calls:
            if script == "reconcile_direct_capability_declarations.py":
                self.assertEqual(accept, (0, 2), step)
            else:
                self.assertEqual(accept, (0,), step)

    def test_a_refusing_reconcile_stops_the_chain(self):
        rc, scripts, _ = self.build({"reconcile_direct_capability_declarations.py": 3})
        self.assertEqual(rc, 1)
        self.assertNotIn("direct_capability_gate.py", scripts)

    def test_a_refusing_applier_stops_before_the_gate(self):
        rc, scripts, _ = self.build({"remove_retired_direct_leaf.py": 3})
        self.assertEqual(rc, 1)
        self.assertNotIn("direct_capability_gate.py", scripts)

    def test_a_refused_gate_returns_two_not_one(self):
        rc, scripts, _ = self.build({"direct_capability_gate.py": 3})
        self.assertEqual(rc, 2)
        self.assertIn("direct_capability_gate.py", scripts)


if __name__ == "__main__":
    unittest.main()
