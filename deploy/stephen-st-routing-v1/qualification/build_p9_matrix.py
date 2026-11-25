#!/usr/bin/env python3
"""Build the route-by-route P9 qualification matrix for the direct capability gate.

Every cell is either a fact read from a file in this tree or is recorded as
UNQUALIFIED. Nothing here contacts a provider, reads a credential or writes to a
release store.

Two manifests are involved and they are not interchangeable:

  * the CANDIDATE is the manifest a change would ship (for example the smart-judge
    three-tier candidate). The matrix enumerates the gate findings over it.
  * the BASELINE is what the declaration/retirement chain is built from, because
    the chain's appliers refuse to overwrite a declaration that already exists.
    Its output is a separate declaration candidate, not the change candidate.

So the matrix answers: "of the findings on the change candidate, which ones does
the declaration chain resolve, on what evidence, and with which exact edit?"

Usage:
  python3 build_p9_matrix.py \
      --baseline /path/to/live-CURRENT/routes.toml \
      --candidate /path/to/candidate.toml \
      --work-dir /var/tmp/p9-matrix \
      --out p9-matrix.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
DEPLOY = HERE.parent
EVIDENCE = DEPLOY / "evidence" / "tool-qualification-20261123"

#: The checker whose verdicts these artifacts contain. Pinned by sha256 into
#: ``summary.inputs.gate_script_sha256`` and re-computed independently by
#: ``verify_p9_qualification.py``: a matrix whose verdicts came from a different
#: build must be rebuilt, not reinterpreted.
GATE = DEPLOY / "direct_capability_gate.py"

CAPABILITIES = ("tool_calling", "vision", "video")

# Evidence bases that may support a positive `tool_calling` declaration, weakest
# last. Anything not in this mapping cannot justify `true`.
POSITIVE_BASES = {
    "LIVE_TOOL_CALL": "measured on this route under its own credential",
    "PRIOR_MEASUREMENT": "measured upstream on this model under this route's credential",
    "SHARED_MODEL_SPEC": "measured on the same upstream model under a DIFFERENT credential",
}
POSITIVE_RESOLUTION = {
    "LIVE_TOOL_CALL": "DECLARATION_ONLY_MEASURED_TOOL_CALL",
    "PRIOR_MEASUREMENT": "DECLARATION_ONLY_MEASURED_TOOL_CALL",
    "SHARED_MODEL_SPEC": "DECLARATION_ONLY_SHARED_MODEL_SPEC",
}


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def manifest(path: pathlib.Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def parse_finding(finding: str) -> dict:
    """Split one gate failure line into its parts.

    The gate emits two shapes:
        TARGET_CAPABILITY_UNQUALIFIED <route> <target>.<capability>
        ROUTE_CAPABILITY_NOT_EXPLICIT <route> <capability>
    """
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


def run_chain(baseline: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Run the declaration/retirement chain and return its candidate manifest.

    The chain keeps every intermediate file so each stage can be reviewed. Its
    exit status is not a success signal for this function: the chain ends by
    running the gate, which is expected to REFUSE while provider blockers remain.
    """
    command = [sys.executable, str(DEPLOY / "build_p9_candidate.py"),
               "--baseline", str(baseline), "--out-dir", str(out_dir)]
    subprocess.run(command, capture_output=True, text=True, check=False)
    candidate = out_dir / "5-candidate.toml"
    if not candidate.exists():
        raise SystemExit(f"chain produced no candidate at {candidate}")
    return candidate


def run_gate(manifest_path: pathlib.Path, out: pathlib.Path) -> dict:
    command = [sys.executable, str(GATE), "--manifest", str(manifest_path), "--json"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    text = completed.stdout.strip()
    if not text:
        raise SystemExit(
            f"gate produced no JSON for {manifest_path}: {completed.stderr.strip()}")
    report = json.loads(text)
    # Keep only what a reviewer needs to check the verdict. The full report also
    # names each client's credential environment variable and quota group; those
    # are deployment detail, not qualification evidence, so they stay out of the
    # committed artifact.
    projection = {key: report[key] for key in
                  ("direct_count", "status", "failures", "upstream_vs_admission_gaps")}
    out.write_text(json.dumps(projection, indent=2) + "\n")
    return report


def tool_evidence(route: str, target: str, tools: dict, refusals: dict) -> dict | None:
    """The recorded tool_calling evidence for this route, if any exists."""
    entry = (tools.get("routes") or {}).get(route)
    if entry:
        return {
            "class": entry["basis"],
            "value": bool(entry.get("value")),
            "receipt": entry.get("receipt"),
            "note": entry.get("note", ""),
            "file": "evidence/tool-qualification-20261123/target_tool_evidence.json",
            "key": route,
        }
    refusal = refusals.get(target)
    if refusal:
        return {
            "class": "PROVIDER_REFUSES_TOOLS",
            "value": False,
            "receipt": refusal.get("receipt"),
            "note": (f"HTTP {refusal.get('provider_status')}: "
                     f"{refusal.get('provider_message', '')[:160]}"),
            "file": ("evidence/tool-qualification-20261123/"
                     "target_tool_refusal_evidence.json"),
            "key": target,
        }
    return None


def sibling_witness(candidate_doc: dict, route: str, target_name: str,
                    client: str | None, receipts: dict) -> str | None:
    """A direct route on the SAME model but a DIFFERENT credential.

    A model-level tool call proves the model, not the credential the route in
    question uses, so the matrix must name the sibling it borrowed the proof from
    rather than assert a shared capability anonymously.
    """
    targets = candidate_doc.get("targets") or {}
    model = (targets.get(target_name) or {}).get("id")
    for name, spec in (candidate_doc.get("routes") or {}).items():
        if name == route or not name.startswith("direct-"):
            continue
        other = targets.get(spec.get("target")) or {}
        if other.get("id") != model or other.get("llm_client") == client:
            continue
        if receipts.get(name, {}).get("verdict") == "TOOL_CALL":
            return name
    return None


def row_for(finding: str, candidate_doc: dict, tools: dict, refusals: dict,
            retired: dict, probe: dict, annotations: dict,
            evidence_dir: pathlib.Path) -> dict:
    parsed = parse_finding(finding)
    route = parsed["route"]
    capability = parsed["capability"]
    spec = (candidate_doc.get("routes") or {}).get(route, {})
    target_name = parsed["target"] or spec.get("target")
    target = (candidate_doc.get("targets") or {}).get(target_name, {})
    client = target.get("llm_client")
    transport = ((candidate_doc.get("llm_clients") or {}).get(client, {}) or {}).get("format")

    row = {
        "finding": finding,
        "code": parsed["code"],
        "layer": parsed["layer"],
        "route": route,
        "finding_target": parsed["target"],
        "target": target_name,
        "model": target.get("id"),
        "client": client,
        "transport": transport,
        "capability": capability,
        "route_declaration": spec.get(capability, "NOT_DECLARED"),
        "target_declaration": target.get(capability, "NOT_DECLARED"),
    }

    if route == retired.get("route"):
        row.update({
            "evidence_class": "UPSTREAM_RETIRED_404",
            "evidence_file": ("evidence/tool-qualification-20261123/"
                              "retired_direct_leaf_evidence.json"),
            "evidence_key": route,
            "evidence_receipt": retired.get("receipt"),
            "evidence_note": retired.get("note", ""),
            "proposed_declaration": {"kind": "retire", "route": route,
                                    "target": retired.get("target")},
            "proposed_correction": (
                f"remove route {route} and target {retired.get('target')}: the provider "
                f"answers {retired.get('provider_status')} and no provider serves the "
                "model, so the leaf cannot hold a capability qualification"),
            "resolution": "DECLARATION_ONLY_RETIRED_ROUTE",
        })
    elif capability == "tool_calling":
        evidence = tool_evidence(route, target_name, tools, refusals)
        if evidence is None:
            row.update({
                "evidence_class": "UNQUALIFIED",
                "evidence_file": None,
                "evidence_key": None,
                "evidence_receipt": None,
                "evidence_note": ("no receipt in this tree: the provider was never asked, "
                                  "or was asked and gave an inconclusive answer"),
                "proposed_declaration": None,
                "proposed_correction": (
                    f"none available: a bounded tool-call probe on {target.get('id')} "
                    "under this route's own credential is required before any declaration"),
                "resolution": "BLOCKED_PROVIDER_PROOF",
            })
        elif evidence["class"] == "PROVIDER_REFUSES_TOOLS":
            row.update({
                "evidence_class": evidence["class"],
                "evidence_file": evidence["file"],
                "evidence_key": evidence["key"],
                "evidence_receipt": evidence["receipt"],
                "evidence_note": evidence["note"],
                "proposed_declaration": {
                    "kind": "declare", "route": route, "capability": "tool_calling",
                    "route_value": False, "target": target_name, "target_value": False},
                "proposed_correction": (
                    f"declare target {target_name}.tool_calling = false and route "
                    f"{route}.tool_calling = false; the provider refused a tool request "
                    "and the route must refuse it at ingress rather than forward it"),
                "resolution": "DECLARATION_ONLY_PROVIDER_REFUSAL",
            })
        elif evidence["class"] in POSITIVE_BASES:
            witness = None
            if evidence["class"] == "SHARED_MODEL_SPEC":
                witness = sibling_witness(candidate_doc, route, target_name, client, probe)
                if witness is None:
                    raise SystemExit(
                        f"SHARED_MODEL_SPEC for {route} names no tool-positive sibling on a "
                        "different credential, so the class overstates the evidence")
            # Whether the receipt names its own subject, or only a filename does.
            cited = {name for name in _jsonl_names(evidence_dir)
                     if name in str(evidence["receipt"])}
            subjects = {route, target.get("id")}
            if witness:
                subjects.add(witness)
            pairs = [name for subject, name in
                     positive_subjects(evidence_dir, cited, annotations)
                     if subject in subjects]
            if not pairs:
                attribution = "unrecorded"
            elif any(name not in annotations for name in pairs):
                attribution = "recorded"
            else:
                attribution = "annotated"
            row.update({
                "evidence_class": evidence["class"],
                "evidence_file": evidence["file"],
                "evidence_key": evidence["key"],
                "evidence_receipt": evidence["receipt"],
                "evidence_note": evidence["note"],
                "receipt_attribution": attribution,
                "witness_route": witness,
                "proposed_declaration": {
                    "kind": "declare", "route": route, "capability": "tool_calling",
                    "route_value": True, "target": target_name, "target_value": True},
                "proposed_correction": (
                    f"declare target {target_name}.tool_calling = true and route "
                    f"{route}.tool_calling = true"),
                "resolution": POSITIVE_RESOLUTION[evidence["class"]],
            })
        else:
            raise SystemExit(f"unknown evidence basis {evidence['class']!r} for {route}")
    elif capability in ("video", "vision"):
        row.update({
            "evidence_class": "ADMISSION_NARROWING",
            "evidence_file": "direct_capability_gate.py",
            "evidence_key": capability,
            "evidence_receipt": None,
            "evidence_note": (
                "the runner refuses video unless the flag is literally true and vision "
                "unless it is not literally false; an absent flag is therefore already a "
                "refusal, and stating it makes the existing refusal auditable without "
                "changing admission. This is an admission decision, not an upstream "
                "inability: the documented upstream gap is kept separately"),
            "proposed_declaration": {
                "kind": "declare", "route": route, "capability": capability,
                "route_value": False, "target": target_name, "target_value": False},
            "proposed_correction": (
                f"declare route {route}.{capability} = false and target "
                f"{target_name}.{capability} = false, making the existing refusal explicit"),
            "resolution": "DECLARATION_ONLY_EXPLICIT_REFUSAL",
        })
    else:
        raise SystemExit(f"unhandled finding shape: {finding}")
    return row


def leg_targets(route_spec: dict) -> set[str]:
    out: set[str] = set()
    if isinstance(route_spec.get("target"), str):
        out.add(route_spec["target"])
    out.update(route_spec.get("candidates") or [])
    for tier in route_spec.get("terminal_fallback") or []:
        out.update([tier] if isinstance(tier, str) else tier)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True,
                        help="manifest the declaration chain is built from")
    parser.add_argument("--candidate", type=pathlib.Path, required=True,
                        help="manifest the findings are enumerated over")
    parser.add_argument("--work-dir", type=pathlib.Path, required=True,
                        help="scratch directory for chain intermediates and gate JSON")
    parser.add_argument("--out", type=pathlib.Path,
                        default=HERE / "p9-matrix.json")
    parser.add_argument("--evidence-dir", type=pathlib.Path, default=EVIDENCE)
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    chain_dir = args.work_dir / "chain"

    candidate_doc = manifest(args.candidate)
    tools = load(args.evidence_dir / "target_tool_evidence.json")
    refusals = load(args.evidence_dir / "target_tool_refusal_evidence.json")
    retired = load(args.evidence_dir / "retired_direct_leaf_evidence.json")
    annotations = load(args.evidence_dir / "receipt-annotations.json").get("files", {})
    probe = probe_verdicts(args.evidence_dir)

    gate_candidate_path = args.out.parent / "gate-candidate.json"
    gate_chain_path = args.out.parent / "gate-chain.json"
    gate_candidate = run_gate(args.candidate, gate_candidate_path)
    chain_manifest = run_chain(args.baseline, chain_dir)
    gate_chain = run_gate(chain_manifest, gate_chain_path)

    rows = [row_for(finding, candidate_doc, tools, refusals, retired, probe,
                    annotations, args.evidence_dir)
            for finding in gate_candidate["failures"]]

    residual = set(gate_chain["failures"])
    for row in rows:
        row["chain_resolved"] = row["finding"] not in residual
        if row["chain_resolved"] and row["resolution"] == "BLOCKED_PROVIDER_PROOF":
            raise SystemExit(
                f"contradiction: {row['finding']} is blocked but the chain resolved it")

    chain_doc = manifest(chain_manifest)
    judge_targets = leg_targets((candidate_doc.get("routes") or {}).get("smart-judge", {}))
    nt_targets = leg_targets((candidate_doc.get("routes") or {}).get("smart-nt", {}))
    direct_specs = {name: spec for name, spec in (candidate_doc.get("routes") or {}).items()
                    if name.startswith("direct-")}

    def reachable(targets: set[str]) -> list[str]:
        return sorted(name for name, spec in direct_specs.items()
                      if spec.get("target") in targets)

    def rows_on(names) -> list[dict]:
        selected = set(names)
        return [row for row in rows if row["route"] in selected]

    new_alias = "direct-htpc-qwen3_5-9b-mtp-nt"
    alias_spec = (candidate_doc.get("routes") or {}).get(new_alias)
    summary = {
        "inputs": {
            "baseline_sha256": sha256(args.baseline),
            "candidate_sha256": sha256(args.candidate),
            "chain_candidate_sha256": sha256(chain_manifest),
            "candidate_declarations_sha256": None,
            "chain_declarations_sha256": None,
            "gate_candidate_sha256": sha256(gate_candidate_path),
            "gate_chain_sha256": sha256(gate_chain_path),
            # The verdicts above came from these exact bytes.
            "gate_script_sha256": sha256(GATE),
        },
        "candidate": {
            "direct_routes": gate_candidate["direct_count"],
            "findings": len(gate_candidate["failures"]),
            "gaps": len(gate_candidate["upstream_vs_admission_gaps"]),
            "status": gate_candidate["status"],
        },
        "chain": {
            "direct_routes": gate_chain["direct_count"],
            "findings": len(gate_chain["failures"]),
            "status": gate_chain["status"],
            "residual": list(gate_chain["failures"]),
        },
        "resolution_classes": count(rows, "resolution"),
        "evidence_classes": count(rows, "evidence_class"),
        "by_code": count(rows, "code"),
        "by_capability": count(rows, "capability"),
        "chain_resolved": sum(1 for row in rows if row["chain_resolved"]),
        "judge_legs": {
            "smart_judge": {"routes": reachable(judge_targets),
                            "findings": [row["finding"] for row in
                                         rows_on(reachable(judge_targets))]},
            "smart_nt": {"routes": reachable(nt_targets),
                         "findings": [row["finding"] for row in
                                      rows_on(reachable(nt_targets))]},
            "union": {"routes": reachable(judge_targets | nt_targets),
                      "findings": [row["finding"] for row in
                                   rows_on(reachable(judge_targets | nt_targets))]},
        },
        "new_alias": {
            "name": new_alias,
            "findings": [row["finding"] for row in rows if row["route"] == new_alias],
            "declarations": (
                {capability: {
                    "route": alias_spec.get(capability, "NOT_DECLARED"),
                    "target": (candidate_doc.get("targets") or {})
                              .get(alias_spec.get("target"), {})
                              .get(capability, "NOT_DECLARED")}
                 for capability in CAPABILITIES}
                if alias_spec else None),
        },
    }

    candidate_projection = args.out.parent / "candidate-declarations.json"
    chain_projection = args.out.parent / "chain-declarations.json"
    candidate_projection.write_text(json.dumps(
        dict(capability_projection(candidate_doc),
             manifest_sha256=sha256(args.candidate)), indent=2) + "\n")
    chain_projection.write_text(json.dumps(
        dict(capability_projection(manifest(chain_manifest)),
             manifest_sha256=sha256(chain_manifest),
             retired_routes=sorted(
                 set(name for name in (manifest(args.baseline).get("routes") or {})
                     if name.startswith("direct-"))
                 - set(name for name in (manifest(chain_manifest).get("routes") or {})
                       if name.startswith("direct-")))),
        indent=2) + "\n")
    summary["inputs"]["candidate_declarations_sha256"] = sha256(candidate_projection)
    summary["inputs"]["chain_declarations_sha256"] = sha256(chain_projection)

    matrix = {
        "note": (
            "Each row is one direct_capability_gate failure on the candidate manifest, "
            "paired with the evidence that exists for it and the exact declaration the "
            "chain proposes. 'chain_resolved' is read from the gate over the chain's own "
            "candidate, not asserted here."),
        "summary": summary,
        "findings": rows,
        "upstream_vs_admission_gaps": gate_candidate["upstream_vs_admission_gaps"],
    }
    args.out.write_text(json.dumps(matrix, indent=2, sort_keys=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nmatrix written to {args.out} ({args.out.stat().st_size} bytes)")
    return 0


def is_tool_positive(record: dict) -> bool:
    if record.get("verdict") == "TOOL_CALL":
        return True
    return (record.get("finish_reason") == "tool_calls"
            and bool(record.get("tool_calls") or record.get("tool_call_count")))


def positive_subjects(evidence_dir: pathlib.Path, files: set[str],
                      annotations: dict) -> list[tuple[str, str]]:
    """(subject, file) for every tool-positive record in these files.

    A file that carries no subject of its own falls back to its annotation, so the
    pair always names what the record is about.
    """
    out: list[tuple[str, str]] = []
    for path in sorted(evidence_dir.glob("*.jsonl")):
        if path.name not in files:
            continue
        default = (annotations.get(path.name) or {}).get("subject")
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not is_tool_positive(record):
                continue
            subject = (record.get("route") or record.get("alias")
                       or record.get("model") or default)
            if subject:
                out.append((subject, path.name))
    return out


def receipt_subjects(evidence_dir: pathlib.Path, files: set[str]) -> set[str]:
    """Every route/alias/model name that any record in these files names itself."""
    out: set[str] = set()
    for path in sorted(evidence_dir.glob("*.jsonl")):
        if path.name not in files:
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("route", "alias", "model"):
                if isinstance(record.get(key), str):
                    out.add(record[key])
    return out


def probe_verdicts(evidence_dir: pathlib.Path) -> dict[str, dict]:
    """The measured verdict per direct alias, across every route-level probe file."""
    out: dict[str, dict] = {}
    for path in sorted(evidence_dir.glob("route-tool-probe*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record.get("route") or record.get("alias")
            if name and record.get("verdict") == "TOOL_CALL":
                out[name] = record
    return out


def _jsonl_names(evidence_dir: pathlib.Path) -> set[str]:
    return {path.name for path in evidence_dir.glob("*.jsonl")}


def capability_projection(doc: dict) -> dict:
    """The declaration surface the gate judges, with credentials and endpoints removed.

    Only the fields a capability claim needs are kept: which target a direct route
    points at, the model id and client that identify the target's credential class,
    and the three admission flags. No base URL, credential name, limit or header.
    """
    targets = doc.get("targets") or {}
    routes = doc.get("routes") or {}
    direct = {}
    for name in sorted(routes):
        if not name.startswith("direct-"):
            continue
        spec = routes[name]
        target = spec.get("target")
        entry = {"target": target}
        for capability in CAPABILITIES:
            if capability in spec:
                entry[capability] = spec[capability]
        direct[name] = entry
    used = sorted({entry["target"] for entry in direct.values() if entry.get("target")})
    target_view = {}
    for name in used:
        spec = targets.get(name) or {}
        entry = {"id": spec.get("id"), "llm_client": spec.get("llm_client")}
        for capability in CAPABILITIES:
            if capability in spec:
                entry[capability] = spec[capability]
        target_view[name] = entry
    return {"direct_routes": direct, "targets": target_view}


def sha256(path: pathlib.Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
