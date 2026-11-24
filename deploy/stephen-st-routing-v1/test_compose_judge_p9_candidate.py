"""Negative controls for the composed #51+#52 release manifest, no provider or release."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compose_judge_p9_candidate",
                                                HERE / "compose_judge_p9_candidate.py")
comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comp)


def fixtures():
    original = {
        "llm_clients": {"htpc": {"format": "openai_responses"}},
        "targets": {"htpc-qwen3_5-9b-mtp": {"id": "Qwen3.5-9B-MTP"}},
        "routes": {
            "smart-judge": {
                "candidates": ["google-gemma-26b", "google-gemma-31b"],
                "terminal_fallback": ["openai-gpt-6-luna-nt"],
                "terminal_policy": "quota_only",
            },
            "direct-htpc-qwen3_5-9b-mtp": {"target": "htpc-qwen3_5-9b-mtp"},
            comp.RETIRED_ALIAS: {"target": comp.RETIRED_TARGET},
        },
    }
    original["targets"][comp.RETIRED_TARGET] = {"id": "liquid/lfm-2.5-2.6b:free"}
    final = copy.deepcopy(original)
    del final["routes"][comp.RETIRED_ALIAS]
    del final["targets"][comp.RETIRED_TARGET]
    final["llm_clients"]["htpc-nt"] = {"quota_group": "htpc"}
    final["targets"][comp.NEW_TARGET] = {
        "id": "Qwen3.5-9B-MTP", "reasoning_policy": "disabled",
        "reasoning_dialect": "llama_cpp_enable_thinking",
        "llm_client": "htpc-nt", "tool_calling": True,
        "vision": False, "video": False,
    }
    final["routes"][comp.NEW_ALIAS] = {
        "target": comp.NEW_TARGET, "tool_calling": True,
        "vision": False, "video": False, "reasoning": False,
    }
    final["routes"]["smart-judge"]["terminal_policy"] = "availability"
    final["routes"]["smart-judge"]["terminal_fallback"] = copy.deepcopy(comp.TIERS)
    report = {"status": "PASS", "failures": [], "direct_count": 2}
    return original, final, report


class ComposedCandidateTests(unittest.TestCase):
    def test_correctly_composed_candidate_is_accepted(self):
        before, after, report = fixtures()
        comp.verify_composition(before, after, report)

    def test_p9_only_manifest_is_not_the_judge_manifest(self):
        before, after, report = fixtures()
        after["routes"]["smart-judge"] = before["routes"]["smart-judge"]
        with self.assertRaisesRegex(comp.Refused, "JUDGE_TIERS"):
            comp.verify_composition(before, after, report)

    def test_missing_new_htpc_alias_refused_even_if_p9_says_pass(self):
        before, after, report = fixtures()
        del after["routes"][comp.NEW_ALIAS]
        report["direct_count"] = 1
        with self.assertRaisesRegex(comp.Refused, "ROUTE_SET"):
            comp.verify_composition(before, after, report)

    def test_locality_and_reasoning_critical_alias_fields_are_preserved(self):
        before, after, report = fixtures()
        after["targets"][comp.NEW_TARGET]["reasoning_policy"] = "enabled"
        with self.assertRaisesRegex(comp.Refused, "HTPC_LEAF"):
            comp.verify_composition(before, after, report)

    def test_primary_gemma_pool_cannot_silently_change(self):
        before, after, report = fixtures()
        after["routes"]["smart-judge"]["candidates"] = ["google-gemma-31b"]
        with self.assertRaisesRegex(comp.Refused, "PRIMARY_POOL"):
            comp.verify_composition(before, after, report)

    def test_gate_refusal_is_never_called_success(self):
        before, after, report = fixtures()
        report["status"] = "REFUSED"
        report["failures"] = ["TARGET_CAPABILITY_UNQUALIFIED test target.tool_calling"]
        with self.assertRaisesRegex(comp.Refused, "COMPOSITION_P9"):
            comp.verify_composition(before, after, report)

    def test_baseline_sha_drift_refused_pre_write(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "routes.toml"
            baseline.write_text("[routes]\n")
            with self.assertRaisesRegex(comp.Refused, "STALE_BASELINE"):
                comp.require_safe_input_output(baseline, Path(temp) / "scratch", "0" * 64)

    def test_no_scratch_output_into_production_release_store(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "routes.toml"
            baseline.write_text("[routes]\n")
            with self.assertRaisesRegex(comp.Refused, "protected production root"):
                comp.require_safe_input_output(
                    baseline, Path("/var/lib/switchyard/releases/forbidden"), comp.sha256(baseline))


if __name__ == "__main__":
    unittest.main()
