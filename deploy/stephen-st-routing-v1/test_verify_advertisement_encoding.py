#!/usr/bin/env python3
"""The expected-on-wire rule must reproduce a payload, and must be able to fail."""
import importlib.util
import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "verify_advertisement_encoding", HERE / "verify_advertisement_encoding.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

MANIFEST = {
    "routes": {
        "direct-declared": {
            "tool_calling": True, "vision": False, "video": False,
            "context_window": 8192,
        },
        "direct-undeclared": {"context_window": 4096},
        "direct-carried": {
            "tool_calling": True, "vision": True, "video": True,
            "video_carrier": "inline", "image_carrier": "remote",
            "context_window": 8192,
        },
        "direct-held": {"held": {"reason": "no servable target"}},
    }
}


def payload():
    return {"data": [
        {"id": "direct-carried", "context_length": 8192, "capabilities": {
            "tool_calling": True, "vision": True, "video": True,
            "video_carriers": ["inline"], "image_carriers": ["remote"]}},
        {"id": "direct-declared", "context_length": 8192, "capabilities": {
            "tool_calling": True, "vision": False, "video": False,
            "video_carriers": None, "image_carriers": None}},
        {"id": "direct-undeclared", "context_length": 4096, "capabilities": {
            "tool_calling": None, "vision": None, "video": None,
            "video_carriers": None, "image_carriers": None}},
    ]}


class EncodingProofTests(unittest.TestCase):
    def test_an_absent_flag_is_advertised_as_null(self):
        expected = verifier.expected_route_values(MANIFEST["routes"]["direct-undeclared"])
        self.assertEqual(expected, {"context_length": 4096, "tool_calling": None,
                                    "vision": None, "video": None,
                                    "video_carriers": None, "image_carriers": None})

    def test_a_carrier_is_only_advertised_for_an_admitted_medium(self):
        route = {"tool_calling": True, "vision": False, "video": False,
                 "video_carrier": "inline", "image_carrier": "inline"}
        expected = verifier.expected_route_values(route)
        self.assertIsNone(expected["video_carriers"])
        self.assertIsNone(expected["image_carriers"])
        route["vision"] = True
        self.assertEqual(verifier.expected_route_values(route)["image_carriers"],
                         ["inline"])

    def test_both_carriers_are_advertised_as_null(self):
        route = {"vision": True, "image_carrier": "both"}
        self.assertIsNone(verifier.expected_route_values(route)["image_carriers"])

    def test_the_rule_reproduces_a_matching_payload(self):
        problems, summary = verifier.compare(MANIFEST, payload())
        self.assertEqual(problems, [], problems)
        self.assertIn("checked 3 advertised routes", summary[0])

    def test_a_held_route_is_not_required_in_the_payload(self):
        problems, _ = verifier.compare(MANIFEST, payload())
        self.assertNotIn("direct-held", " ".join(problems))

    def test_a_declared_false_against_a_null_payload_is_a_mismatch(self):
        observed = payload()
        observed["data"][1]["capabilities"]["video"] = None
        problems, _ = verifier.compare(MANIFEST, observed)
        self.assertIn("ENCODING_MISMATCH direct-declared.video", " ".join(problems))

    def test_a_declared_true_against_a_missing_key_is_a_mismatch(self):
        observed = payload()
        del observed["data"][1]["capabilities"]["tool_calling"]
        problems, _ = verifier.compare(MANIFEST, observed)
        self.assertIn("ENCODING_MISMATCH direct-declared.tool_calling", " ".join(problems))

    def test_a_context_mismatch_is_reported(self):
        observed = payload()
        observed["data"][1]["context_length"] = 1
        problems, _ = verifier.compare(MANIFEST, observed)
        self.assertIn("ENCODING_MISMATCH direct-declared.context_length",
                      " ".join(problems))

    def test_route_sets_must_match_both_ways(self):
        data = payload()["data"]
        data.append({"id": "direct-extra", "context_length": 1, "capabilities": {}})
        del data[0]
        problems, _ = verifier.compare(MANIFEST, {"data": data})
        joined = " ".join(problems)
        self.assertIn("ROUTE_SET_MISMATCH direct-carried: defined but not advertised", joined)
        self.assertIn("ROUTE_SET_MISMATCH direct-extra: advertised but not defined", joined)

    def test_a_payload_without_capabilities_is_refused_not_crashed(self):
        observed = payload()
        observed["data"][1]["capabilities"] = "none"
        problems, _ = verifier.compare(MANIFEST, observed)
        self.assertIn("PAYLOAD_SHAPE", " ".join(problems))


if __name__ == "__main__":
    unittest.main()
