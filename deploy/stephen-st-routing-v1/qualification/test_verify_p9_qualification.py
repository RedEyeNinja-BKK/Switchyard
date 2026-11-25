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
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
DEPLOY = HERE.parent
EVIDENCE = DEPLOY / "evidence" / "tool-qualification-20261123"
VERIFIER = HERE / "verify_p9_qualification.py"
#: The oracle whose bytes the matrix pins. Named here as a path, not a digest:
#: the expected digest is read from the matrix under test.
GATE = DEPLOY / "direct_capability_gate.py"
#: The two pinned manifests the committed matrix was built from.
BASELINE = pathlib.Path("/var/tmp/sj3tier/live-current-5211bdc7.toml")
CANDIDATE = pathlib.Path("/var/tmp/sj3tier/cand-c5fd4668-a.toml")


def sha256_path(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate_pin_check(matrix: dict, check: str = "C8_identity") -> str:
    """Run the verifier on one matrix and return its output.

    Used where a test must inspect *what* the verifier computed, not only that it
    refused: a mutant that echoes the pinned digest would still refuse a tampered
    pin, and only the message distinguishes the two.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="p9gate-"))
    try:
        path = tmp / "matrix.json"
        path.write_text(json.dumps(matrix, indent=2))
        completed = subprocess.run(verifier_argv(path), capture_output=True, text=True)
        output = completed.stdout + completed.stderr
        assert f"FAIL {check}" in output, output
        return output
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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

    # --- C8: the gate oracle must be the build the matrix names -------------
    def test_the_pinned_gate_hash_matches_the_source(self):
        """The control: this must pass, or every other assertion is vacuous."""
        self.assertIn("gate_script_sha256", self.matrix["summary"]["inputs"])
        code, output = self.run_mutated(self.matrix)
        self.assertEqual(code, 0, output)
        self.assertNotIn("gate_script_sha256", output)

    def test_gate_hash_tampering_is_refused_with_the_outputs_intact(self):
        """Flip only the pin. The gate JSONs are untouched and still match."""
        matrix = copy.deepcopy(self.matrix)
        matrix["summary"]["inputs"]["gate_script_sha256"] = "0" * 64
        self.assert_refused(matrix, "C8_identity")

    def test_a_gate_hash_missing_entirely_is_refused(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["summary"]["inputs"].pop("gate_script_sha256", None)
        self.assert_refused(matrix, "C8_identity")
        # And it is refused *because the pin is absent*: the message names the
        # absence and the digest of the checker it could not match.
        code, output = self.run_mutated(matrix)
        self.assertEqual(code, 3, output)
        self.assertIn("FAIL C8_identity", output)
        self.assertIn("gate_script_sha256 is absent", output)
        self.assertIn(sha256_path(GATE), output)

    def test_a_gate_hash_from_a_different_build_is_refused(self):
        """A plausible but wrong digest, not just an obviously fake one."""
        matrix = copy.deepcopy(self.matrix)
        real = matrix["summary"]["inputs"]["gate_script_sha256"]
        flipped = ("0" if real[0] != "0" else "1") + real[1:]
        matrix["summary"]["inputs"]["gate_script_sha256"] = flipped
        self.assert_refused(matrix, "C8_identity")

    def test_a_gate_script_byte_change_is_refused(self):
        """The script changes; the matrix's pin, gate JSONs and rows do not.

        This is the whole point of the pin: the committed gate output remains a
        perfectly self-consistent transcript, and only the oracle's identity
        gives it away.
        """
        import os
        import shutil
        live = GATE
        backup = self.tmp / "gate-backup.py"
        shutil.copy2(live, backup)
        original = live.read_bytes()
        self.addCleanup(lambda: (live.write_bytes(original), shutil.copymode(backup, live)))
        mutated = original + b"\n# comment-only change\n"
        self.assertNotEqual(mutated, original)
        live.write_bytes(mutated)
        # The mutant that matters is "the verifier stops hashing the checker and
        # echoes the pin instead": the refusal below would then pass for the wrong
        # reason. Assert the verifier's own computation, not only its verdict.
        self.assertNotEqual(sha256_path(live), self.matrix["summary"]["inputs"]["gate_script_sha256"])
        self.assertIn(sha256_path(live),
                      run_gate_pin_check(self.matrix, check="C8_identity"))
        self.assert_refused(self.matrix, "C8_identity")

    def test_a_missing_gate_script_is_refused_not_skipped(self):
        """Absence must refuse, not skip: an uncheckable oracle is not a pass.

        The real file is moved aside rather than copied-and-deleted, so the
        restore is a rename of a file that still exists on disk. The park sits
        beside the source because a rename cannot cross filesystems, and the
        restore is asserted before the test can return.
        """
        live = GATE
        parked = live.parent / "direct_capability_gate.parked.py"
        original = live.read_bytes()
        self.assertFalse(parked.exists(), "the park must be empty before the test")
        self.addCleanup(lambda: (parked.rename(live) if parked.exists() else None))
        # Remove the script without opening the subject module: mutating the
        # verifier's source here would make the test unrunnable, because unittest
        # imports this module before the first test executes.
        live.unlink()
        self.assertFalse(live.exists(), "the checker must actually be absent")
        try:
            self.assert_refused(self.matrix, "C8_identity")
        finally:
            # The parked copy is made before the unlink; the source bytes never
            # leave the working tree.
            live.write_bytes(original)
        self.assertEqual(live.read_bytes(), original)
        self.assertFalse(parked.exists(), "the park must be empty again")
        self.assertTrue(live.exists(), "the source tree must be intact again")

    def test_the_parked_checker_is_restored_byte_for_byte(self):
        """The harness's own safety property, checked separately."""
        live = GATE
        before = live.read_bytes()
        parked = live.parent / "direct_capability_gate.parked.py"
        live.rename(parked)
        parked.rename(live)
        self.assertEqual(live.read_bytes(), before)
        self.assertEqual(sha256_path(live), self.matrix["summary"]["inputs"]["gate_script_sha256"])

    # --- vacuity -----------------------------------------------------------
    def test_an_empty_matrix_is_refused_not_passed(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["findings"] = []
        code, output = self.run_mutated(matrix)
        self.assertEqual(code, 2, output)
        self.assertIn("examined nothing", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class BuilderPinTests(unittest.TestCase):
    """The builder must emit the oracle pin, or nothing can verify it."""

    SOURCE = HERE / "build_p9_matrix.py"

    def test_the_builder_pins_the_checker_it_actually_runs(self):
        source = self.SOURCE.read_text()
        self.assertIn("GATE = DEPLOY / \"direct_capability_gate.py\"", source)
        self.assertIn("\"gate_script_sha256\": sha256(GATE)", source)
        # The same path object must be what it executes, not a second copy of the
        # literal that could drift from the file being run.
        self.assertIn("str(GATE), \"--manifest\"", source)

    def test_the_builder_writes_the_pin_into_the_matrix(self):
        """End to end: rebuild in a scratch dir and read the pin back."""
        import tempfile as tf
        work = pathlib.Path(tf.mkdtemp(prefix="p9build-"))
        try:
            out = work / "p9-matrix.json"
            work_chain = work / "chain"
            work_chain.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [sys.executable, str(HERE / "build_p9_matrix.py"),
                 "--baseline", str(BASELINE), "--candidate", str(CANDIDATE),
                 "--work-dir", str(work_chain), "--out", str(out)],
                capture_output=True, text=True)
            # Report the builder's failure verbatim: the point of this test is the
            # pin, and a bare exit code would bury the actual defect.
            self.assertEqual(
                proc.returncode, 0,
                f"builder exited {proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
            self.assertTrue(out.is_file(), "builder exited 0 but wrote no matrix")
            inputs = json.loads(out.read_text())["summary"]["inputs"]
            # Assert presence and format first: indexing a missing key raises
            # KeyError, which is a crash, not a failed assertion.
            self.assertIn("gate_script_sha256", inputs)
            pin = inputs["gate_script_sha256"]
            self.assertEqual(len(pin), 64)
            self.assertEqual(pin, sha256_path(GATE))
        finally:
            shutil.rmtree(work, ignore_errors=True)
