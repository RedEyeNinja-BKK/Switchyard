#!/usr/bin/env python3
"""Removing a dead leaf must be provably bounded to that leaf."""
import importlib.util
import sys
from pathlib import Path
import tomllib
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "remove_retired_direct_leaf", HERE / "remove_retired_direct_leaf.py")
remover = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remover)

BASELINE = """\
[routes.direct-dead]
type = "passthrough"
target = "t-dead"
context_window = 8192
tool_calling = true
vision = false
video = false

[routes.direct-live]
type = "passthrough"
target = "t-live"
context_window = 8192
tool_calling = true
vision = false
video = false

[targets.t-dead]
id = "vendor/model-dead:free"
llm_client = "c-a"
vision = false
video = false

[targets.t-live]
id = "vendor/model-live:free"
llm_client = "c-a"
vision = false
video = false

[llm_clients.c-a]
format = "openai_chat"
"""

EVIDENCE = {
    "route": "direct-dead",
    "target": "t-dead",
    "provider_status": 404,
    "provider_message": "no longer available on OpenRouter: no provider serves this model anymore",
    "receipt": "route-tool-probe-20261123b.jsonl",
}


class RemoveRetiredLeafTests(unittest.TestCase):
    def test_removes_exactly_the_route_and_its_target(self):
        before = tomllib.loads(BASELINE)
        out, removed = remover.apply(BASELINE, EVIDENCE)
        after = tomllib.loads(out)
        self.assertEqual(removed, ["direct-dead", "t-dead"])
        self.assertNotIn("direct-dead", after["routes"])
        self.assertNotIn("t-dead", after["targets"])
        # The sibling leaf and the shared client are untouched, value for value.
        self.assertEqual(after["routes"]["direct-live"], before["routes"]["direct-live"])
        self.assertEqual(after["targets"]["t-live"], before["targets"]["t-live"])
        self.assertEqual(after["llm_clients"], before["llm_clients"])

    def test_the_payload_still_parses_and_keeps_its_sections(self):
        out, _ = remover.apply(BASELINE, EVIDENCE)
        parsed = tomllib.loads(out)
        self.assertEqual(set(parsed), {"routes", "targets", "llm_clients"})

    def test_a_longer_identifier_is_not_a_reference(self):
        """`direct-x-nt` and `t-x-extra` are different names, not references.

        A substring audit reads them as consumers of this leaf and refuses a
        legitimate removal; production aliases genuinely share prefixes.
        """
        text = BASELINE + """
[routes.pooled]
type = "passthrough"
target = "t-live"
candidates = ["direct-dead-extra"]
terminal_fallback = ["t-dead-extra"]
context_window = 8192
tool_calling = true
vision = false
video = false
"""
        out, removed = remover.apply(text, EVIDENCE)
        self.assertEqual(removed, ["direct-dead", "t-dead"])
        parsed = tomllib.loads(out)
        self.assertIn("direct-dead-extra", parsed["routes"]["pooled"]["candidates"])
        self.assertIn("t-dead-extra", parsed["routes"]["pooled"]["terminal_fallback"])

    def test_a_second_consumer_blocks_the_removal(self):
        text = BASELINE + """
[routes.pooled]
type = "passthrough"
target = "t-live"
candidates = ["t-dead"]
context_window = 8192
tool_calling = true
vision = false
video = false
"""
        with self.assertRaises(remover.Refused) as err:
            remover.apply(text, EVIDENCE)
        self.assertIn("REFUSED_CONSUMER_EXISTS", str(err.exception))

    def test_a_route_referenced_as_a_candidate_blocks_the_removal(self):
        text = BASELINE + """
[routes.pooled]
type = "passthrough"
target = "t-live"
candidates = ["direct-dead"]
context_window = 8192
tool_calling = true
vision = false
video = false
"""
        with self.assertRaises(remover.Refused) as err:
            remover.apply(text, EVIDENCE)
        self.assertIn("REFUSED_ROUTE_BEING_REFERENCED", str(err.exception))

    def test_a_quota_response_does_not_retire_a_model(self):
        with self.assertRaises(remover.Refused) as err:
            remover.apply(BASELINE, {**EVIDENCE, "provider_status": 429})
        self.assertIn("REFUSED_NOT_RETIRED", str(err.exception))

    def test_a_missing_message_does_not_retire_a_model(self):
        with self.assertRaises(remover.Refused) as err:
            remover.apply(BASELINE, {**EVIDENCE, "provider_message": "rate limited"})
        self.assertIn("REFUSED_NOT_RETIRED", str(err.exception))

    def test_a_target_mismatch_is_refused(self):
        with self.assertRaises(remover.Refused) as err:
            remover.apply(BASELINE, {**EVIDENCE, "target": "t-live"})
        self.assertIn("REFUSED_TARGET_MISMATCH", str(err.exception))

    def test_a_route_with_candidates_is_not_a_leaf(self):
        text = BASELINE.replace('target = "t-dead"\n',
                                'target = "t-dead"\ncandidates = ["t-live"]\n', 1)
        with self.assertRaises(remover.Refused) as err:
            remover.apply(text, EVIDENCE)
        self.assertIn("REFUSED_ROUTE_HAS_CANDIDATES", str(err.exception))

    def test_evidence_without_a_receipt_is_refused(self):
        with self.assertRaises(remover.Refused) as err:
            remover.apply(BASELINE,
                          {k: v for k, v in EVIDENCE.items() if k != "receipt"})
        self.assertIn("REFUSED_EVIDENCE_INCOMPLETE", str(err.exception))


if __name__ == "__main__":
    unittest.main()
