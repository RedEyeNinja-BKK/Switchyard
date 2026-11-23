#!/usr/bin/env python3
"""The widening applier may only perform the correction it names, from evidence."""
import importlib.util
import json
import sys
from pathlib import Path
import tomllib
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(
    "apply_route_tool_admission", HERE / "apply_route_tool_admission.py")
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

EVIDENCE = {"direct-a": {"basis": "LIVE_TOOL_CALL", "receipt": "probe-20261123.jsonl"}}


class WideningTests(unittest.TestCase):
    def test_widens_the_placeholder_and_nothing_else(self):
        before = tomllib.loads(BASELINE)
        out, widened = applier.apply(BASELINE, EVIDENCE)
        after = tomllib.loads(out)
        self.assertEqual(widened, ["direct-a"])
        self.assertIs(after["routes"]["direct-a"]["tool_calling"], True)
        # No other declaration moved.
        self.assertEqual(after["routes"]["direct-b"], before["routes"]["direct-b"])
        self.assertEqual(after["targets"], before["targets"])
        self.assertEqual(after["llm_clients"], before["llm_clients"])
        # The target is never given a tool claim by a route-level widening.
        self.assertNotIn("tool_calling", after["targets"]["t-a"])

    def test_target_that_refuses_tools_blocks_the_widening(self):
        text = BASELINE.replace(
            '[targets.t-a]\nid = "vendor/model-a"\nllm_client = "c-a"\n',
            '[targets.t-a]\nid = "vendor/model-a"\nllm_client = "c-a"\ntool_calling = false\n')
        with self.assertRaises(applier.Refused) as err:
            applier.apply(text, EVIDENCE)
        self.assertIn("REFUSED_CONTRADICTED_BY_TARGET", str(err.exception))

    def test_an_already_true_route_is_not_a_placeholder(self):
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, {"direct-b": EVIDENCE["direct-a"]})
        self.assertIn("REFUSED_NOT_A_PLACEHOLDER", str(err.exception))

    def test_evidence_without_a_receipt_is_refused(self):
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, {"direct-a": {"basis": "LIVE_TOOL_CALL"}})
        self.assertIn("REFUSED_EVIDENCE_INCOMPLETE", str(err.exception))

    def test_unknown_evidence_basis_is_refused(self):
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, {"direct-a": {"basis": "SEEMS_FINE",
                                                  "receipt": "x"}})
        self.assertIn("REFUSED_EVIDENCE_INCOMPLETE", str(err.exception))

    def test_hidden_substitution_blocks_the_widening(self):
        text = BASELINE.replace('target = "t-a"\n',
                                'target = "t-a"\ncandidates = ["t-b"]\n', 1)
        with self.assertRaises(applier.Refused) as err:
            applier.apply(text, EVIDENCE)
        self.assertIn("REFUSED_HIDDEN_SUBSTITUTION", str(err.exception))

    def test_non_direct_alias_is_refused(self):
        text = BASELINE.replace("[routes.direct-a]", "[routes.pooled-a]")
        with self.assertRaises(applier.Refused) as err:
            applier.apply(text, {"pooled-a": EVIDENCE["direct-a"]})
        self.assertIn("REFUSED_NOT_A_DIRECT_ALIAS", str(err.exception))

    def test_unknown_route_is_refused(self):
        with self.assertRaises(applier.Refused) as err:
            applier.apply(BASELINE, {"direct-zzz": EVIDENCE["direct-a"]})
        self.assertIn("REFUSED_UNKNOWN_ROUTE", str(err.exception))

    def test_evidence_file_shape_is_route_keyed(self):
        # The pipeline hands the applier the file as written on disk: a mapping
        # from route name to record, never an envelope.
        payload = json.loads(json.dumps(EVIDENCE))
        out, widened = applier.apply(BASELINE, payload)
        self.assertEqual(widened, ["direct-a"])
        self.assertTrue(out.startswith("[routes.direct-a]"))


if __name__ == "__main__":
    unittest.main()
