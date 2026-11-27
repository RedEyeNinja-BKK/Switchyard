#!/usr/bin/env python3
"""Negative regression tests: the Switchyard contract is checked, not inferred."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("direct_capability_gate", HERE / "direct_capability_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture():
    google = {"id": "gemini-3.6-flash", "llm_client": "owner-google",
              "tool_calling": True, "vision": False, "video": False}
    stephen = {**google, "llm_client": "stephen-google"}
    non_google = {"id": "nvidia/nemotron-3.5-super-vl:free", "llm_client": "openrouter",
                  "tool_calling": True, "vision": True, "video": True,
                  "image_carrier": "inline", "video_carrier": "inline"}
    doc = {
        "llm_clients": {
            "owner-google": {"format": "openai_chat", "api_key_env": "OWNER_GOOGLE",
                             "quota_group": "owner_google"},
            "stephen-google": {"format": "openai_chat", "api_key_env": "STEPHEN_GOOGLE",
                               "quota_group": "stephen_google"},
            "openrouter": {"format": "openai_responses", "api_key_env": "OWNER_OPENROUTER"},
        },
        "targets": {
            "google-3.6": google, "stephen-google-3.6": stephen, "vl": non_google,
        },
        "routes": {
            "direct-google-gemini-3.6-flash": {
                "id": "direct-google-gemini-3.6-flash", "type": "passthrough",
                "target": "google-3.6", "candidates": [],
                "context_window": 262144,
                "tool_calling": True, "vision": False, "video": False,
            },
            "direct-google-stephen-gemini-3.6-flash": {
                "id": "direct-google-stephen-gemini-3.6-flash", "type": "passthrough",
                "target": "stephen-google-3.6", "candidates": [],
                "context_window": 262144,
                "tool_calling": True, "vision": False, "video": False,
            },
            "direct-openrouter-nemotron-vl": {
                "id": "direct-openrouter-nemotron-vl", "type": "passthrough",
                "target": "vl", "candidates": [],
                "context_window": 262144,
                "tool_calling": True, "vision": True, "video": True,
                "image_carrier": "inline", "video_carrier": "inline",
            },
        },
    }
    return doc


class GateTests(unittest.TestCase):
    def test_clean_known_metadata_passes_without_question_marks(self):
        result = gate.analyze(fixture(), set(fixture()["routes"]))
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(result["direct_count"], 3)
        self.assertNotIn("?", str(result))

    def test_upstream_google_modalities_remain_visible_without_falsely_enabling_route(self):
        d = fixture()
        spec = {"gemini-3.6-flash": {
            "model_input": {"image": "DOCUMENTED", "video": "DOCUMENTED_INPUT",
                            "audio": "DOCUMENTED"},
            "model_output": {"text": "DOCUMENTED"},
            "modality_source": "GOOGLE_MODEL_CARD",
        }}
        result = gate.analyze(d, canonical=spec)
        self.assertEqual(result["status"], "PASS")
        self.assertIn("UPSTREAM_IMAGE_NOT_ADMITTED",
                      str(result["upstream_vs_admission_gaps"]))
        self.assertIn("UPSTREAM_VIDEO_NOT_ADMITTED",
                      str(result["upstream_vs_admission_gaps"]))
        self.assertIn("UPSTREAM_AUDIO_NOT_ADMITTED",
                      str(result["upstream_vs_admission_gaps"]))
        route = next(row for row in result["routes"]
                     if row["route"] == "direct-google-stephen-gemini-3.6-flash")
        self.assertIs(route["route_capabilities"]["vision"], False)
        self.assertEqual(route["upstream_model_input"]["image"], "DOCUMENTED")

    def test_missing_route_vision_fails(self):
        d = fixture()
        d["routes"]["direct-openrouter-nemotron-vl"].pop("vision")
        self.assertIn("ROUTE_CAPABILITY_NOT_EXPLICIT",
                      " ".join(gate.analyze(d)["failures"]))

    def test_missing_target_tool_support_fails(self):
        d = fixture()
        d["targets"]["vl"].pop("tool_calling")
        self.assertIn("TARGET_CAPABILITY_UNQUALIFIED",
                      " ".join(gate.analyze(d)["failures"]))

    def test_target_video_absent_is_not_converted_to_false(self):
        d = fixture()
        d["targets"]["vl"].pop("video")
        result = gate.analyze(d)
        self.assertIn("TARGET_CAPABILITY_UNQUALIFIED", str(result["failures"]))
        self.assertEqual(
            next(row for row in result["routes"] if row["target"] == "vl")
                ["target_capabilities"]["video"], "NOT_DECLARED")

    def test_missing_safe_context_bound_refused(self):
        d = fixture()
        d["routes"]["direct-openrouter-nemotron-vl"].pop("context_window")
        self.assertIn("CONTEXT_CAP_UNQUALIFIED",
                      str(gate.analyze(d)["failures"]))

    def test_context_overclaim_refused(self):
        d = fixture()
        d["targets"]["vl"]["context_window"] = 131072
        self.assertIn("CONTEXT_OVERCLAIM", str(gate.analyze(d)["failures"]))

    def test_unadvertised_alias_is_a_failure(self):
        d = fixture()
        shown = set(d["routes"]) - {"direct-openrouter-nemotron-vl"}
        self.assertIn("DIRECT_NOT_ADVERTISED", str(gate.analyze(d, shown)["failures"]))

    def test_actual_model_endpoint_must_match_runtime_route_flags(self):
        d = fixture()
        rows = {}
        for name, route in d["routes"].items():
            rows[name] = {
                "id": name, "context_length": route["context_window"],
                "capabilities": {
                    "tool_calling": route["tool_calling"],
                    "vision": route["vision"],
                    "video": route["video"],
                    "image_carriers": (["inline"] if route.get("image_carrier") == "inline" else None),
                    "video_carriers": (["inline"] if route.get("video_carrier") == "inline" else None),
                },
            }
        self.assertEqual(gate.analyze(d, rows)["status"], "PASS")
        rows["direct-openrouter-nemotron-vl"]["capabilities"]["vision"] = False
        self.assertIn("RUNTIME_ADVERTISEMENT_DRIFT", str(gate.analyze(d, rows)["failures"]))
        rows["direct-openrouter-nemotron-vl"]["capabilities"]["vision"] = True
        rows["direct-openrouter-nemotron-vl"]["capabilities"]["image_carriers"] = None
        self.assertIn("RUNTIME_CARRIER_DRIFT", str(gate.analyze(d, rows)["failures"]))
        rows["direct-openrouter-nemotron-vl"]["capabilities"]["image_carriers"] = ["inline"]
        rows["direct-openrouter-nemotron-vl"]["context_length"] = 1
        self.assertIn("RUNTIME_CONTEXT_DRIFT", str(gate.analyze(d, rows)["failures"]))

    def test_duplicate_runtime_model_ids_refused(self):
        with self.assertRaises(gate.ContractError) as err:
            gate._model_entries({"data": [{"id": "direct-a"}, {"id": "direct-a"}]})
        self.assertEqual(err.exception.token, "MODEL_LIST_DUPLICATE_ID")

    def test_fake_video_overclaim_is_caught(self):
        d = fixture()
        d["targets"]["vl"]["video"] = False
        self.assertIn("DIRECT_OVERCLAIM", str(gate.analyze(d)["failures"]))

    def test_wrong_video_wire_is_caught(self):
        d = fixture()
        d["llm_clients"]["openrouter"]["format"] = "openai_chat"
        self.assertIn("VIDEO_WIRE_MISMATCH", str(gate.analyze(d)["failures"]))

    def test_carrier_mismatch_is_caught(self):
        d = fixture()
        d["routes"]["direct-openrouter-nemotron-vl"]["video_carrier"] = "both"
        self.assertIn("CARRIER_OVERCLAIM", str(gate.analyze(d)["failures"]))

    def test_silent_capability_suppression_is_caught(self):
        d = fixture()
        d["routes"]["direct-openrouter-nemotron-vl"]["vision"] = False
        self.assertIn("DIRECT_SUPPRESSED_CAPABILITY", str(gate.analyze(d)["failures"]))

    def test_credential_boundary_is_not_merged(self):
        d = fixture()
        d["llm_clients"]["stephen-google"]["quota_group"] = "owner_google"
        self.assertIn("GOOGLE_ACCOUNT_BOUNDARY_BROKEN", str(gate.analyze(d)["failures"]))

    def test_shared_google_model_facts_must_be_equal(self):
        d = fixture()
        d["targets"]["stephen-google-3.6"]["vision"] = True
        self.assertIn("GOOGLE_SHARED_CAPABILITY_DRIFT", str(gate.analyze(d)["failures"]))

    def test_missing_stephen_model_is_a_failure(self):
        d = fixture()
        del d["routes"]["direct-google-stephen-gemini-3.6-flash"]
        self.assertIn("GOOGLE_DUAL_PROFILE_MODEL_DRIFT", str(gate.analyze(d)["failures"]))

    def test_unexpected_fallback_is_a_hard_refusal(self):
        d = fixture()
        d["routes"]["direct-openrouter-nemotron-vl"]["candidates"] = ["google-3.6"]
        self.assertIn("DIRECT_HAS_FALLBACK", str(gate.analyze(d)["failures"]))

    def _rows_for(self, d, overrides=None):
        rows = {}
        for name, route in d["routes"].items():
            rows[name] = {
                "id": name, "context_length": route["context_window"],
                "capabilities": {
                    "tool_calling": route.get("tool_calling"),
                    "vision": route.get("vision"),
                    "video": route.get("video"),
                    "image_carriers": (["inline"] if route.get("image_carrier") == "inline" else None),
                    "video_carriers": (["inline"] if route.get("video_carrier") == "inline" else None),
                },
            }
        for name, patch in (overrides or {}).items():
            rows[name]["capabilities"].update(patch)
        return rows

    def test_undeclared_flag_advertised_as_null_is_not_drift(self):
        """`capabilities` is Option<bool>: an undeclared flag reaches the wire as null.

        Comparing that null against a "not declared" marker reported drift on every
        field the config left absent, which punished a truthful advertisement.
        """
        d = fixture()
        route = d["routes"]["direct-openrouter-nemotron-vl"]
        route.pop("video")
        route.pop("video_carrier")
        d["targets"]["vl"].pop("video")
        d["targets"]["vl"].pop("video_carrier")
        rows = self._rows_for(d)
        failures = " ".join(gate.analyze(d, rows)["failures"])
        self.assertNotIn("RUNTIME_ADVERTISEMENT_DRIFT direct-openrouter-nemotron-vl.video",
                         failures)
        self.assertIn("ROUTE_CAPABILITY_NOT_EXPLICIT", failures)

    def test_declared_false_advertised_as_null_is_still_drift(self):
        """Normalising absence must not neuter the check: a declared value is a claim."""
        d = fixture()
        rows = self._rows_for(d, {"direct-openrouter-nemotron-vl": {"video": None}})
        failures = " ".join(gate.analyze(d, rows)["failures"])
        self.assertIn("RUNTIME_ADVERTISEMENT_DRIFT direct-openrouter-nemotron-vl.video",
                      failures)

    def test_missing_wire_key_stays_distinct_from_null(self):
        d = fixture()
        rows = self._rows_for(d)
        del rows["direct-openrouter-nemotron-vl"]["capabilities"]["vision"]
        self.assertIn("RUNTIME_ADVERTISEMENT_DRIFT direct-openrouter-nemotron-vl.vision",
                      " ".join(gate.analyze(d, rows)["failures"]))

    def test_retired_native_25_is_refused(self):
        d = fixture()
        d["routes"]["direct-google-gemini-2.5-flash"] = copy.deepcopy(
            d["routes"]["direct-google-gemini-3.6-flash"])
        with self.assertRaises(gate.ContractError) as err:
            gate.analyze(d)
        self.assertEqual(err.exception.token, "RETIRED_DIRECT_DEFINED")


if __name__ == "__main__":
    unittest.main()
