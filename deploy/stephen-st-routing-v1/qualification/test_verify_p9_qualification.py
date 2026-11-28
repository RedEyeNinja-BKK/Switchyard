#!/usr/bin/env python3
"""Mutation controls for the independent P9 qualification verifier.

Every test states one way the matrix could lie, applies it, and requires the
verifier to refuse with that specific check. A verifier that passes the pristine
matrix but also passes a mutated one is worthless, so each mutant is graded by the
check name that caught it rather than by "it failed somehow".

The controls run against the committed artifacts only, so they need no access to a
release store and cannot drift from the delivery they grade.
"""
import copy
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "evidence" / "tool-qualification-20261123"
VERIFIER = HERE / "verify_p9_qualification.py"


def verifier_argv(matrix: pathlib.Path) -> list[str]:
    return [sys.executable, str(VERIFIER), "--matrix", str(matrix),
            "--candidate-declarations", str(HERE / "candidate-declarations.json"),
            "--chain-declarations", str(HERE / "chain-declarations.json"),
            "--gate-candidate", str(HERE / "gate-candidate.json"),
            "--gate-chain", str(HERE / "gate-chain.json"),
            "--evidence-dir", str(EVIDENCE)]


class VerifierCase(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="p9verify-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.matrix = json.loads((HERE / "p9-matrix.json").read_text())

    def run_mutated(self, matrix: dict,
                    chain: pathlib.Path | None = None) -> tuple[int, str]:
        path = self.tmp / "matrix.json"
        path.write_text(json.dumps(matrix, indent=2))
        argv = verifier_argv(path)
        if chain is not None:
            index = argv.index("--chain-declarations") + 1
            argv[index] = str(chain)
        completed = subprocess.run(argv, capture_output=True, text=True)
        return completed.returncode, completed.stdout + completed.stderr

    def assert_refused(self, matrix: dict, check: str,
                       chain: pathlib.Path | None = None) -> None:
        code, output = self.run_mutated(matrix, chain)
        self.assertEqual(code, 3, f"{check}: verifier returned {code}\n{output}")
        self.assertIn(f"FAIL {check}", output,
                      f"{check}: refused, but not from this check\n{output}")

    # --- the control -------------------------------------------------------
    def test_the_matrix_as_committed_is_qualified(self):
        completed = subprocess.run(verifier_argv(HERE / "p9-matrix.json"),
                                   capture_output=True, text=True)
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("QUALIFIED", output)

    # --- C1: coverage ------------------------------------------------------
    def test_a_dropped_finding_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["findings"] = matrix["findings"][1:]
        self.assert_refused(matrix, "C1_coverage")

    def test_a_duplicated_finding_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["findings"].append(copy.deepcopy(matrix["findings"][0]))
        self.assert_refused(matrix, "C1_coverage")

    # --- C2: the finding line is the authority for route and target --------
    def test_a_retargeted_finding_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["findings"][0]["route"] = "direct-somewhere-else"
        self.assert_refused(matrix, "C2_parse")

    def test_a_relabelled_finding_target_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"] if r["layer"] == "route")
        row["finding_target"] = "invented"
        self.assert_refused(matrix, "C2_parse")

    # --- C3: the evidence class and the value it can support ---------------
    def test_a_blocked_row_promoted_to_a_declaration_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"] if not r["chain_resolved"])
        row["resolution"] = "DECLARATION_ONLY_MEASURED_TOOL_CALL"
        row["proposed_declaration"] = {"kind": "declare", "route": row["route"],
                                       "capability": "tool_calling",
                                       "route_value": True, "target": row["target"],
                                       "target_value": True}
        self.assert_refused(matrix, "C3_evidence")

    def test_a_mislabelled_evidence_class_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["evidence_class"] == "LIVE_TOOL_CALL")
        row["evidence_class"] = "PRIOR_MEASUREMENT"
        self.assert_refused(matrix, "C3_evidence")

    def test_a_shared_spec_row_relabelled_as_a_live_call_is_refused(self):
        """The owner accepted these four as model-level evidence only.

        Promoting one to LIVE_TOOL_CALL is the exact rewrite the acceptance
        forbids: it would claim the owner's own credential path served a tool
        call. The class must stay re-derived from the receipt.
        """
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["evidence_class"] == "SHARED_MODEL_SPEC")
        owner_route, witness = row["route"], row["witness_route"]
        row["evidence_class"] = "LIVE_TOOL_CALL"
        self.assert_refused(matrix, "C3_evidence")
        # And the refusal is about the class, not an incidental missing field.
        self.assertNotEqual(owner_route, witness)
        self.assertIsNotNone(witness)

    def test_a_flipped_capability_value_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["resolution"] == "DECLARATION_ONLY_EXPLICIT_REFUSAL")
        row["proposed_declaration"]["route_value"] = True
        self.assert_refused(matrix, "C3_evidence")

    # --- C4: a positive claim needs a positive receipt ---------------------
    def test_a_shared_spec_claim_without_a_witness_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["evidence_class"] == "SHARED_MODEL_SPEC")
        row["witness_route"] = None
        self.assert_refused(matrix, "C4_receipt")

    def test_a_shared_spec_claim_witnessed_on_the_same_credential_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["evidence_class"] == "SHARED_MODEL_SPEC")
        row["witness_route"] = row["route"]
        self.assert_refused(matrix, "C4_receipt")

    def test_a_receipt_that_does_not_contain_the_call_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["evidence_class"] == "LIVE_TOOL_CALL")
        row["evidence_receipt"] = "LIVE_TOOL_CALL 2026-11-23 nowhere.jsonl"
        self.assert_refused(matrix, "C4_receipt")

    def test_a_receipt_attribution_overstated_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r.get("receipt_attribution") == "annotated")
        row["receipt_attribution"] = "recorded"
        self.assert_refused(matrix, "C4_receipt")

    def test_a_receipt_attribution_understated_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r.get("receipt_attribution") == "recorded")
        row["receipt_attribution"] = "annotated"
        self.assert_refused(matrix, "C4_receipt")

    # --- C5: the chain candidate must carry the proposal -------------------
    def test_a_chain_that_does_not_carry_the_proposal_is_refused(self):
        """The matrix is right; the chain artifact is what disagrees."""
        chain = json.loads((HERE / "chain-declarations.json").read_text())
        row = next(r for r in self.matrix["findings"]
                   if r["chain_resolved"] and r["layer"] == "target"
                   and r["capability"] == "tool_calling"
                   and r["resolution"] == "DECLARATION_ONLY_MEASURED_TOOL_CALL")
        self.assertIs(chain["targets"][row["target"]]["tool_calling"], True)
        chain["targets"][row["target"]]["tool_calling"] = False
        path = self.tmp / "chain-declarations.json"
        path.write_text(json.dumps(chain, indent=2))
        self.assert_refused(self.matrix, "C5_chain", chain=path)

    # --- C6: the residual set is exactly the blocked set -------------------
    def test_a_blocked_row_declared_resolved_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"] if not r["chain_resolved"])
        row["chain_resolved"] = True
        self.assert_refused(matrix, "C6_residual")

    # --- C7: documented gaps are not admissions ----------------------------
    def test_a_documentation_gap_dropped_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["upstream_vs_admission_gaps"] = matrix["upstream_vs_admission_gaps"][1:]
        self.assert_refused(matrix, "C7_gaps")

    def test_a_video_gap_promoted_to_true_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        row = next(r for r in matrix["findings"]
                   if r["capability"] == "video" and r["proposed_declaration"])
        row["proposed_declaration"]["route_value"] = True
        self.assert_refused(matrix, "C7_gaps")

    # --- C8: the artifact is pinned to what it describes -------------------
    def test_a_stale_candidate_pin_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["summary"]["inputs"]["candidate_sha256"] = "0" * 64
        self.assert_refused(matrix, "C8_identity")

    def test_a_projection_from_another_manifest_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["summary"]["inputs"]["chain_declarations_sha256"] = "1" * 64
        self.assert_refused(matrix, "C8_identity")

    # --- vacuity -----------------------------------------------------------
    def test_an_empty_matrix_is_refused_not_passed(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["findings"] = []
        code, output = self.run_mutated(matrix)
        self.assertEqual(code, 2, output)
        self.assertIn("examined nothing", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
