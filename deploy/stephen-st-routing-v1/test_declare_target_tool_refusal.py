#!/usr/bin/env python3
"""A target may only be declared tool-incapable behind a route that already refuses.

The declaration is worth nothing unless it cannot change behaviour: every test
below that removes one of those preconditions must end in a refusal.
"""
import importlib.util
import sys
from pathlib import Path
import tomllib
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "declare_target_tool_refusal", HERE / "declare_target_tool_refusal.py")
applier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(applier)

BASELINE = """\
[routes.direct-a]
type = "passthrough"
target = "t-a"
context_window = 8192
tool_calling = false
vision = false
video = false

[routes.direct-b]
type = "passthrough"
target = "t-b"
context_window = 8192
tool_calling = true
vision = false
video = false

[targets.t-a]
id = "vendor/model-a"
llm_client = "c-a"
vision = false
video = false

[targets.t-b]
id = "vendor/model-b"
llm_client = "c-a"
vision = false
video = false

[llm_clients.c-a]
format = "openai_chat"
"""

REFUSAL = {
    "t-a": {
        "basis": "PROVIDER_REFUSES_TOOLS",
        "provider_status": 400,
        "provider_message": '"auto" tool choice requires --enable-auto-tool-choice',
        "receipt": "route-tool-probe-20261123b.jsonl",
    },
}


class RefusalDeclarationTests(unittest.TestCase):
    def test_declares_false_on_provider_evidence_alone(self):
        before = tomllib.loads(BASELINE)
        out, declared = applier.apply(BASELINE, REFUSAL)
        after = tomllib.loads(out)
        self.assertEqual(declared, ["t-a"])
        self.assertIs(after["targets"]["t-a"]["tool_calling"], False)
        # Nothing else moved, and the route keeps refusing on its own.
        self.assertEqual(after["routes"], before["routes"])
        self.assertEqual(after["llm_clients"], before["llm_clients"])
        self.assertEqual(after["targets"]["t-b"],
                         {**before["targets"]["t-b"]})

    def test_route_that_still_admits_tools_blocks_the_declaration(self):
        """Otherwise the target claim would change behaviour for that route."""
        text = BASELINE.replace(
            'target = "t-b"\ncontext_window = 8192\ntool_calling = true',
            'target = "t-b"\ncontext_window = 8192\ntool_calling = true')
        text = text.replace('[routes.direct-a]\ntype = "passthrough"\ntarget = "t-a"\n'
                            'context_window = 8192\ntool_calling = false',
                            '[routes.direct-a]\ntype = "passthrough"\ntarget = "t-a"\n'
                            'context_window = 8192\ntool_calling = true')
        with self.assertRaises(applier.Refused) as err:
            applier.apply(text, REFUSAL)
        self.assertIn("REFUSED_ROUTE_STILL_ADMITS_TOOLS", str(err.exception))

    def test_two_consumers_block_the_declaration(self):
        text = BASELINE + """
[routes.direct-c]
type = "passthrough"
target = "t-a"
context_window = 8192
tool_calling = false
vision = false
video = false
"""
        with self.assertRaises(applier.Refused) as err:
            applier.apply(text, REFUSAL)
        self.assertIn("REFUSED_NOT_SOLE_CONSUMER", str(err.exception))

    def test_a_quota_response_is_not_a_tool_refusal(self):
        evidence = {"t-a": {**REFUSAL["t-a"], "provider_status": 429}}
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, evidence)
        self.assertIn("REFUSED_STATUS", str(err.exception))

    def test_an_unavailable_upstream_is_not_a_tool_refusal(self):
        evidence = {"t-a": {**REFUSAL["t-a"], "provider_status": 502}}
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, evidence)
        self.assertIn("REFUSED_STATUS", str(err.exception))

    def test_a_message_without_a_receipt_is_refused(self):
        evidence = {"t-a": {k: v for k, v in REFUSAL["t-a"].items() if k != "receipt"}}
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, evidence)
        self.assertIn("REFUSED_EVIDENCE_INCOMPLETE", str(err.exception))

    def test_a_different_basis_is_refused(self):
        evidence = {"t-a": {**REFUSAL["t-a"], "basis": "LIVE_TOOL_CALL"}}
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, evidence)
        self.assertIn("REFUSED_BASIS", str(err.exception))

    def test_an_existing_declaration_is_never_overwritten(self):
        # The sole consumer must already refuse, or the earlier gate fires first;
        # the point here is the key, so the shape has to reach that check.
        text = BASELINE.replace(
            '[targets.t-a]\nid = "vendor/model-a"\nllm_client = "c-a"\n',
            '[targets.t-a]\nid = "vendor/model-a"\nllm_client = "c-a"\ntool_calling = false\n')
        with self.assertRaises(applier.Refused) as err:
            applier.apply(text, REFUSAL)
        self.assertIn("REFUSED_KEY_ALREADY_PRESENT", str(err.exception))

    def test_unknown_target_is_refused(self):
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, {"t-zzz": REFUSAL["t-a"]})
        self.assertIn("REFUSED_UNKNOWN_TARGET", str(err.exception))


if __name__ == "__main__":
    unittest.main()
