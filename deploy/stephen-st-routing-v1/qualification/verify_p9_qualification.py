#!/usr/bin/env python3
"""Independently re-derive a P9 qualification matrix and refuse when it disagrees.

This verifier does not read the matrix's own verdict labels as authority. For every
gate finding it recomputes, from the evidence files and the two manifests alone:

  * which route/target/capability the finding line names;
  * whether a receipt exists for that route, and of which evidence class;
  * the exact declaration that receipt can support, and whether the chain candidate
    actually carries it;
  * whether an unqualified finding is genuinely unqualified, and not merely a
    receipt that a 429/503 was quietly promoted into a capability claim.

Exit codes: 0 clean, 3 a check failed, 2 a check examined nothing (a vacuous pass is
a defect, not a success).

Usage:
  python3 verify_p9_qualification.py \
      --matrix p9-matrix.json --work-dir /var/tmp/p9-matrix \
      --candidate-declarations candidate-declarations.json \
      --chain-declarations chain-declarations.json \
      --gate-candidate gate-candidate.json --gate-chain gate-chain.json \
      --evidence-dir ../evidence/tool-qualification-20261123

This file must stay independent of `build_p9_matrix.py`: it deliberately does not
import anything from the builder, so a defect in the builder cannot also hide itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tomllib

# Evidence bases that may support a positive `tool_calling` declaration.
POSITIVE_BASES = ("LIVE_TOOL_CALL", "PRIOR_MEASUREMENT", "SHARED_MODEL_SPEC")
# A receipt is tool-positive only on a completed answer that carried a tool call.
TOOL_POSITIVE_VERDICTS = ("TOOL_CALL",)
TOOL_POSITIVE_FINISH = ("tool_calls",)
INCONCLUSIVE_STATUSES = (429, 500, 502, 503, 504)

#: The checker whose verdicts this matrix reports. Its bytes are pinned in
#: ``summary.inputs.gate_script_sha256`` and re-computed here from the deployed
#: source, never trusted from the matrix.
GATE = pathlib.Path(__file__).resolve().parent.parent / "direct_capability_gate.py"


def tool_positive(record: dict) -> bool:
    if record.get("verdict") in TOOL_POSITIVE_VERDICTS:
        return True
    return (record.get("finish_reason") in TOOL_POSITIVE_FINISH
            and bool(record.get("tool_calls") or record.get("tool_call_count")))


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def read_toml(path: pathlib.Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_finding(finding: str) -> dict:
    parts = finding.split()
    code = parts[0]
    route = parts[1] if len(parts) > 1 else ""
    if code == "TARGET_CAPABILITY_UNQUALIFIED" and len(parts) == 3:
        target, _, capability = parts[2].rpartition(".")
        return {"code": code, "route": route, "target": target,
                "capability": capability, "layer": "target"}
    if code == "ROUTE_CAPABILITY_NOT_EXPLICIT" and len(parts) == 3:
        return {"code": code, "route": route, "target": None,
                "capability": parts[2], "layer": "route"}
    return {"code": code, "route": route, "target": None, "capability": None,
            "layer": "other"}


class Receipts:
    """Index every receipt record by the route or model name it is about."""

    def __init__(self, evidence_dir: pathlib.Path):
        self.dir = evidence_dir
        self.by_name: dict[str, list[dict]] = {}
        self.by_file: dict[str, list[dict]] = {}
        self.files: dict[str, dict] = {}
        for path in sorted(evidence_dir.glob("*.jsonl")):
            self._index_file(path)

    def _index_file(self, path: pathlib.Path) -> None:
        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"unparseable": line[:120]})
        self.files[path.name] = {"sha256": sha256(path), "records": len(records)}
        self.by_file[path.name] = records
        for record in records:
            normalised = dict(record, _receipt_file=path.name)
            if "http" in record and "status" not in record:
                normalised["status"] = record["http"]
            for key in ("route", "alias", "model"):
                name = record.get(key)
                if isinstance(name, str):
                    self.by_name.setdefault(name, []).append(normalised)

    def positive(self, name: str, files: set[str] | None = None) -> list[dict]:
        hits = []
        for record in self.by_name.get(name, []):
            if files is not None and record["_receipt_file"] not in files:
                continue
            if tool_positive(record):
                hits.append(record)
        return hits

    def inconclusive(self, name: str) -> list[dict]:
        return [record for record in self.by_name.get(name, [])
                if record.get("status") in INCONCLUSIVE_STATUSES
                or record.get("verdict") in ("RATE_LIMITED", "QUOTA_429")]

    def cited_files(self, receipt: str | None) -> set[str]:
        if not receipt:
            return set()
        return {name for name in self.files if name in receipt}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-declarations", type=pathlib.Path, required=True,
                        help="capability projection of the candidate manifest")
    parser.add_argument("--chain-declarations", type=pathlib.Path, required=True,
                        help="capability projection of the chain candidate manifest")
    parser.add_argument("--gate-candidate", type=pathlib.Path, required=True)
    parser.add_argument("--gate-chain", type=pathlib.Path, required=True)
    parser.add_argument("--evidence-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    matrix = load(args.matrix)
    gate_candidate = load(args.gate_candidate)
    gate_chain = load(args.gate_chain)
    chain_doc = load(args.chain_declarations)
    candidate_doc = load(args.candidate_declarations)
    chain_routes = chain_doc.get("direct_routes") or {}
    chain_targets = chain_doc.get("targets") or {}
    cand_routes = candidate_doc.get("direct_routes") or {}
    cand_targets = candidate_doc.get("targets") or {}
    receipts = Receipts(args.evidence_dir)
    tools = load(args.evidence_dir / "target_tool_evidence.json").get("routes", {})
    refusals = load(args.evidence_dir / "target_tool_refusal_evidence.json")
    retired = load(args.evidence_dir / "retired_direct_leaf_evidence.json")
    annotations = load(args.evidence_dir / "receipt-annotations.json").get("files", {})

    rows = matrix["findings"]
    failures: list[str] = []
    examined = {"rows": 0, "resolved": 0, "blocked": 0, "receipts": 0, "gaps": 0}

    def fail(check: str, detail: str) -> None:
        failures.append(f"{check}: {detail}")

    def expected_disposition(row: dict) -> tuple[str, dict | None]:
        """Recompute the evidence class and declaration from the evidence files."""
        route, target, capability = row["route"], row["target"], row["capability"]
        if route == retired.get("route"):
            return "UPSTREAM_RETIRED_404", {"kind": "retire"}
        if capability == "tool_calling":
            entry = tools.get(route)
            if entry:
                return entry["basis"], {"kind": "declare", "route_value": True,
                                        "target_value": bool(entry.get("value"))}
            if target in refusals:
                return "PROVIDER_REFUSES_TOOLS", {"kind": "declare",
                                                  "route_value": False,
                                                  "target_value": False}
            return "UNQUALIFIED", None
        if capability in ("video", "vision"):
            return "ADMISSION_NARROWING", {"kind": "declare", "route_value": False,
                                           "target_value": False}
        return "UNHANDLED", None

    # C1 — the matrix covers the gate exactly, once each.
    covered = [row["finding"] for row in rows]
    if len(covered) != len(set(covered)):
        fail("C1_coverage", "duplicate findings in the matrix")
    missing = sorted(set(gate_candidate["failures"]) - set(covered))
    extra = sorted(set(covered) - set(gate_candidate["failures"]))
    for item in missing:
        fail("C1_coverage", f"gate failure absent from matrix: {item}")
    for item in extra:
        fail("C1_coverage", f"matrix row absent from gate: {item}")
    examined["rows"] = len(rows)
    if not rows:
        return _report(examined, failures, "C1_coverage")

    # C2..C5 — per-row re-derivation.
    for row in rows:
        parsed = parse_finding(row["finding"])
        for field in ("code", "layer", "route", "capability"):
            if row.get(field) != parsed[field]:
                fail("C2_parse", f"{row['finding']}: {field} is {row.get(field)!r}, "
                                 f"finding says {parsed[field]!r}")
        # The finding line names a target only in the target-layer shape; a
        # direct route IS a one-to-one passthrough, so the row resolves it and
        # records what the line itself said.
        if row.get("finding_target") != parsed["target"]:
            fail("C2_parse", f"{row['finding']}: finding_target is "
                             f"{row.get('finding_target')!r}, finding says "
                             f"{parsed['target']!r}")
        if parsed["layer"] == "route":
            resolved = (cand_routes.get(row["route"]) or {}).get("target")
            if row.get("target") != resolved:
                fail("C2_parse", f"{row['finding']}: resolved target is "
                                 f"{row.get('target')!r}, route points at {resolved!r}")
        expected_class, expected_decl = expected_disposition(row)
        if row.get("evidence_class") != expected_class:
            fail("C3_evidence", f"{row['finding']}: evidence class is "
                                f"{row.get('evidence_class')!r}, evidence supports "
                                f"{expected_class!r}")
        proposed = row.get("proposed_declaration")
        if expected_decl is None:
            if proposed is not None:
                fail("C3_evidence", f"{row['finding']}: unqualified row proposes a "
                                    f"declaration {proposed!r}")
            if row.get("resolution") != "BLOCKED_PROVIDER_PROOF":
                fail("C3_evidence", f"{row['finding']}: unqualified row is labelled "
                                    f"{row.get('resolution')!r}")
            examined["blocked"] += 1
        else:
            if not proposed or proposed.get("kind") != expected_decl["kind"]:
                fail("C3_evidence", f"{row['finding']}: proposal {proposed!r} does not "
                                    f"match evidence {expected_decl!r}")
            elif expected_decl["kind"] == "declare":
                for side in ("route_value", "target_value"):
                    if proposed.get(side) != expected_decl[side]:
                        fail("C3_evidence", f"{row['finding']}: {side} is "
                                            f"{proposed.get(side)!r}, evidence supports "
                                            f"{expected_decl[side]!r}")
            examined["resolved"] += 1

        # C4 — a positive declaration needs a positive receipt in the cited file.
        if expected_class in POSITIVE_BASES:
            examined["receipts"] += 1
            cited = receipts.cited_files(row.get("evidence_receipt"))
            if not cited:
                fail("C4_receipt", f"{row['finding']}: no receipt file resolves from "
                                   f"{row.get('evidence_receipt')!r}")
                continue

            subjects = {row["route"], row.get("model")}
            if row.get("witness_route"):
                subjects.add(row["witness_route"])
            naming = sorted({name for name in cited
                             for record in receipts.by_file.get(name, [])
                             if tool_positive(record)
                             and (record.get("route") or record.get("alias")
                                  or record.get("model")
                                  or (annotations.get(name) or {}).get("subject"))
                             in subjects})
            if not naming:
                claimed = "unrecorded"
            elif any(name not in annotations for name in naming):
                claimed = "recorded"
            else:
                claimed = "annotated"
            if row.get("receipt_attribution") != claimed:
                fail("C4_receipt", f"{row['finding']}: attribution is "
                                   f"{row.get('receipt_attribution')!r}, receipts "
                                   f"support {claimed!r}")

            if expected_class == "LIVE_TOOL_CALL":
                if not receipts.positive(row["route"], cited):
                    fail("C4_receipt", f"{row['finding']}: LIVE claim but {row['route']} "
                                       f"has no tool-positive record in {sorted(cited)}")
                for record in receipts.inconclusive(row["route"]):
                    if record["_receipt_file"] in cited:
                        fail("C4_receipt", f"{row['finding']}: LIVE claim on a route that "
                                           f"also returned {record.get('status')} in "
                                           f"{record['_receipt_file']}")

            elif expected_class == "PRIOR_MEASUREMENT":
                model = row.get("model") or ""
                if not receipts.positive(model, cited):
                    annotation = next((annotations[name] for name in sorted(cited)
                                       if annotations.get(name, {}).get("subject") == model),
                                      {})
                    if annotation.get("subject") != model:
                        fail("C4_receipt", f"{row['finding']}: cited receipts "
                                           f"{sorted(cited)} carry no tool call for {model!r} "
                                           "and no annotation names it")
                    else:
                        # An annotation is weaker than a recorded field, so at least
                        # require the file to read as that model's probe series.
                        series = [record for record in receipts.by_name.get(model, [])
                                  if record["_receipt_file"] in cited]
                        if not any(record.get("status") in INCONCLUSIVE_STATUSES
                                   for record in series):
                            fail("C4_receipt", f"{row['finding']}: {sorted(cited)} is "
                                               "annotated to this model but does not read "
                                               "as that model's probe series")

            else:  # SHARED_MODEL_SPEC
                witness = row.get("witness_route")
                if not witness:
                    fail("C4_receipt", f"{row['finding']}: shared-spec claim names no witness")
                else:
                    observed = cand_targets.get(
                        (cand_routes.get(witness) or {}).get("target"), {})
                    here = cand_targets.get(row["target"], {})
                    if observed.get("id") != here.get("id"):
                        fail("C4_receipt", f"{row['finding']}: witness {witness} is a "
                                           f"different model ({observed.get('id')!r} vs "
                                           f"{here.get('id')!r})")
                    if observed.get("llm_client") == here.get("llm_client"):
                        fail("C4_receipt", f"{row['finding']}: witness {witness} uses the "
                                           "same credential, so this is not a shared-spec "
                                           "claim")
                    if not receipts.positive(witness, cited):
                        fail("C4_receipt", f"{row['finding']}: witness {witness} has no "
                                           f"tool-positive record in {sorted(cited)}")

        # C5 — the chain candidate implements the proposal it is credited with.
        if row.get("chain_resolved"):
            spec = chain_routes.get(row["route"])
            target_spec = chain_targets.get(row["target"])
            if expected_decl and expected_decl["kind"] == "retire":
                if spec is not None:
                    fail("C5_chain", f"{row['finding']}: credited as resolved but "
                                     f"{row['route']} still exists in the chain candidate")
            elif expected_decl:
                capability = row["capability"]
                for label, holder, want in (
                        ("route", spec, expected_decl["route_value"]),
                        ("target", target_spec, expected_decl["target_value"])):
                    if holder is None:
                        fail("C5_chain", f"{row['finding']}: chain candidate has no {label}")
                    elif holder.get(capability) != want:
                        fail("C5_chain", f"{row['finding']}: chain {label} "
                                         f"{row['route'] if label == 'route' else row['target']}"
                                         f".{capability} is {holder.get(capability)!r}, "
                                         f"proposal requires {want!r}")

    # C6 — the residual set is exactly the blocked set, and each is a tool gap.
    residual = set(gate_chain["failures"])
    blocked = {row["finding"] for row in rows if not row.get("chain_resolved")}
    for item in sorted(residual - blocked):
        fail("C6_residual", f"chain still reports {item} but the matrix does not call it blocked")
    for item in sorted(blocked - residual):
        fail("C6_residual", f"matrix calls {item} blocked but the chain gate no longer reports it")
    for item in sorted(residual):
        parsed = parse_finding(item)
        if parsed["capability"] != "tool_calling":
            fail("C6_residual", f"residual {item} is not a tool_calling gap")
    if not residual:
        fail("C6_residual", "the chain reports no residual findings")

    # C7 — documented upstream gaps are carried through, never turned into admissions.
    gaps = matrix.get("upstream_vs_admission_gaps") or []
    examined["gaps"] = len(gaps)
    if set(gaps) != set(gate_candidate["upstream_vs_admission_gaps"]):
        fail("C7_gaps", "matrix gaps differ from the gate's documented gaps")
    if len(gaps) != len(gate_candidate["upstream_vs_admission_gaps"]):
        fail("C7_gaps", "matrix gap count differs from the gate's")
    for row in rows:
        if row["capability"] in ("video", "vision") and row["proposed_declaration"]:
            if row["proposed_declaration"].get("route_value") is True:
                fail("C7_gaps", f"{row['finding']}: a documented admission gap was "
                                "promoted to an admission")
    for capability in ("video", "vision"):
        declared = [name for name, spec in chain_routes.items()
                    if spec.get(capability) is True]
        for name in declared:
            if any(row["route"] == name and row["capability"] == capability for row in rows):
                fail("C7_gaps", f"chain declares {name}.{capability} = true while the gate "
                                "still reports it as an admission gap")

    # C8 — the candidate and chain must be the manifests the summary names.
    summary = matrix.get("summary") or {}
    inputs = summary.get("inputs") or {}
    for key, path in (
            ("candidate_declarations_sha256", args.candidate_declarations),
            ("chain_declarations_sha256", args.chain_declarations),
            ("gate_candidate_sha256", args.gate_candidate),
            ("gate_chain_sha256", args.gate_chain)):
        if inputs.get(key) != sha256(path):
            fail("C8_identity", f"{key} is {inputs.get(key)!r}, file is {sha256(path)!r}")

    # The oracle itself, fail closed. Every verdict above came from these bytes; if
    # the source changed, the matrix describes a checker that no longer exists and
    # must be rebuilt. A missing pin, a missing script or an unreadable script is a
    # refusal, never a skip: "could not check" is not "checked and fine".
    #
    # A missing script must be its own refusal, not the `OSError` of the read
    # below: both are refusals, but they say different things to a reader, and a
    # single try block would report an absent oracle as an unreadable one.
    try:
        actual = sha256(GATE)
    except OSError as exc:
        fail("C8_identity",
             f"gate_script_sha256: checker {GATE} is missing or unreadable ({exc}), "
             "so the matrix verdicts cannot be attributed to any build")
    else:
        pinned = inputs.get("gate_script_sha256")
        if not pinned:
            fail("C8_identity",
                 "gate_script_sha256 is absent from summary.inputs, so the matrix "
                 f"verdicts are unattributable (checker is {actual})")
        elif pinned != actual:
            fail("C8_identity", f"gate_script_sha256 is {pinned!r}, checker file is {actual!r}")
    if candidate_doc.get("manifest_sha256") != inputs.get("candidate_sha256"):
        fail("C8_identity", "candidate projection was not derived from the pinned manifest")
    if chain_doc.get("manifest_sha256") != inputs.get("chain_candidate_sha256"):
        fail("C8_identity", "chain projection was not derived from the pinned manifest")
    if summary.get("candidate", {}).get("findings") != len(rows):
        fail("C8_identity", "summary finding count does not match the rows")
    if summary.get("chain", {}).get("findings") != len(residual):
        fail("C8_identity", "summary residual count does not match the chain gate")

    # C9 — the new HTPC leaf must not carry a finding of its own.
    alias = summary.get("new_alias") or {}
    if alias.get("findings"):
        fail("C9_alias", f"new alias carries findings: {alias['findings']}")
    if alias.get("name") not in cand_routes:
        fail("C9_alias", f"candidate does not define {alias.get('name')!r}")

    return _report(examined, failures, None)


def _report(examined: dict, failures: list[str], short_circuit: str | None) -> int:
    print(json.dumps({"examined": examined}, indent=2))
    for detail in failures:
        print(f"FAIL {detail}")
    if short_circuit:
        print(f"REFUSED: {short_circuit} examined nothing")
        return 2
    if failures:
        print(f"REFUSED: {len(failures)} disagreement(s)")
        return 3
    for label, value in examined.items():
        if value == 0:
            print(f"REFUSED: check scope {label} is empty")
            return 2
    print("QUALIFIED: every matrix cell re-derives from evidence and the chain candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
