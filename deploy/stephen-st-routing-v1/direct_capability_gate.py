#!/usr/bin/env python3
"""Strict, OFFLINE direct-route capability contract, using the ACTIVE Switchyard manifest.

This is a RELEASE GATE, not a dashboard/Google-only model-card renderer.
It consumes the fields the Switchyard runner actually uses for admission:
route-level tool_calling/vision/video and image/video carriers, plus the
target-level corresponding facts. It never turns vendor model docs into an
admitted wire claim. Missing evidence is a named BLOCKER, not a question mark.

Usage:
  python3 direct_capability_gate.py --manifest /var/lib/switchyard/releases/CURRENT/routes.toml
  python3 direct_capability_gate.py --manifest candidate.toml --models models.json
  python3 direct_capability_gate.py --manifest candidate.toml --json

Exit 0 only for fully qualified active direct routes. No secrets, no API calls,
no service changes or account mutations.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RETIRED = frozenset({"gemini-2.5-flash", "gemini-2.5-flash-lite"})
BOOL_FIELDS = ("tool_calling", "vision", "video")
CARRIERS = frozenset({"inline", "remote", "both"})
# Distinct from any TOML value, so "the route never declared this" cannot be
# confused with "the route declared the string NOT_DECLARED".
ABSENT = object()
# Distinct from JSON null: a key missing from /v1/models is a shape problem, not
# an absence of the fact.
UNREPORTED = object()


class ContractError(ValueError):
    def __init__(self, token: str, route: str, detail: str):
        self.token, self.route, self.detail = token, route, detail
        super().__init__(f"{token} [{route}]: {detail}")


def _fail(token: str, route: str, detail: str) -> None:
    raise ContractError(token, route, detail)


def _direct_names(doc: dict) -> list[str]:
    routes = doc.get("routes")
    if not isinstance(routes, dict):
        _fail("DIRECT_ROUTES_NOT_A_TABLE", "manifest", "routes must be a table")
    direct = sorted(name for name in routes if name.startswith("direct-"))
    if not direct:
        _fail("DIRECT_ROUTES_EMPTY", "manifest", "no direct aliases")
    for name in direct:
        if any(name.endswith("-" + suffix) for suffix in RETIRED):
            _fail("RETIRED_DIRECT_DEFINED", name, "native Gemini 2.5 must be removed, not held")
    return direct


def _one_target(doc: dict, route_name: str) -> tuple[dict, str, dict]:
    route = doc["routes"][route_name]
    if route.get("type") != "passthrough":
        _fail("DIRECT_NOT_PASSTHROUGH", route_name, "a direct alias must be 1:1")
    candidates = route.get("candidates", [])
    if candidates:
        _fail("DIRECT_HAS_FALLBACK", route_name, repr(candidates))
    for key in ("terminal_fallback", "escalation_target", "pool_groups"):
        if route.get(key):
            _fail("DIRECT_HIDDEN_SUBSTITUTION", route_name, key)
    target_name = route.get("target")
    targets = doc.get("targets") or {}
    if not isinstance(target_name, str) or target_name not in targets:
        _fail("DIRECT_TARGET_MISSING", route_name, str(target_name))
    target = targets[target_name]
    if not target.get("id") or target.get("held"):
        _fail("DIRECT_TARGET_NOT_SERVABLE", route_name, target_name)
    if not target.get("llm_client") or target["llm_client"] not in doc.get("llm_clients", {}):
        _fail("DIRECT_CLIENT_MISSING", route_name, target_name)
    return route, target_name, target


def _boolean(value) -> bool:
    return type(value) is bool


def _carriers(route_name: str, route: dict, target: dict, medium: str, errors: list[str]):
    field = "image_carrier" if medium == "vision" else "video_carrier"
    rc, tc = route.get(field), target.get(field)
    if rc is not None and rc not in CARRIERS:
        errors.append(f"CARRIER_INVALID {route_name} route.{field}={rc!r}")
    if tc is not None and tc not in CARRIERS:
        errors.append(f"CARRIER_INVALID {route_name} target.{field}={tc!r}")
    if route[medium]:
        if medium == "video" and tc is None:
            errors.append(f"VIDEO_TRANSPORT_NOT_QUALIFIED {route_name} missing target.{field}")
        if rc in CARRIERS and tc in CARRIERS:
            needed = {"inline", "remote"} if rc == "both" else {rc}
            supported = {"inline", "remote"} if tc == "both" else {tc}
            if not needed <= supported:
                errors.append(f"CARRIER_OVERCLAIM {route_name} {field} route={rc} target={tc}")
        elif rc is None and tc in ("inline", "remote"):
            errors.append(f"CARRIER_UNADVERTISED {route_name} route.{field} must be {tc}")
    elif rc is not None:
        errors.append(f"CARRIER_WITH_DISABLED_MEDIUM {route_name} {field}")


def canonical_google_cards() -> dict:
    """Existing upstream model-spec source, never a runtime admission substitute."""
    path = HERE / "gemini-suite" / "canonical-google-model-capabilities-20261123.json"
    return json.loads(path.read_text())["models"]


def analyze(doc: dict, advertised: set[str] | dict | None = None,
            canonical: dict | None = None) -> dict:
    names = _direct_names(doc)
    listed = set(advertised) if advertised is not None else None
    live_entries = advertised if isinstance(advertised, dict) else None
    rows = []
    errors: list[str] = []
    documented_gaps: list[str] = []
    canonical = canonical or {}
    for name in names:
        try:
            route, target_name, target = _one_target(doc, name)
        except ContractError as exc:
            errors.append(str(exc))
            continue
        client = doc["llm_clients"][target["llm_client"]]
        issues: list[str] = []
        for key in BOOL_FIELDS:
            if not _boolean(route.get(key)):
                issues.append(f"ROUTE_CAPABILITY_NOT_EXPLICIT {name} {key}")
            if not _boolean(target.get(key)):
                issues.append(f"TARGET_CAPABILITY_UNQUALIFIED {name} {target_name}.{key}")
            if _boolean(route.get(key)) and _boolean(target.get(key)):
                if route[key] and not target[key]:
                    issues.append(f"DIRECT_OVERCLAIM {name} {key} true against target false")
                if not route[key] and target[key]:
                    issues.append(f"DIRECT_SUPPRESSED_CAPABILITY {name} {key}: "
                                  "must document intentional route narrowing or correct source")
        if all(_boolean(route.get(k)) for k in BOOL_FIELDS):
            _carriers(name, route, target, "vision", issues)
            _carriers(name, route, target, "video", issues)
        if listed is not None and name not in listed:
            issues.append(f"DIRECT_NOT_ADVERTISED {name}")
        if live_entries is not None and name in live_entries:
            live = live_entries[name]
            capabilities = live.get("capabilities")
            if not isinstance(capabilities, dict):
                issues.append(f"RUNTIME_CAPABILITY_METADATA_MISSING {name}")
            else:
                for key in BOOL_FIELDS:
                    declared = route.get(key, ABSENT)
                    advertised = capabilities.get(key, UNREPORTED)
                    # The runtime carries Option<bool>: a flag the config does not
                    # declare reaches the wire as JSON null, which is the encoding of
                    # absence rather than a third value. Comparing that null against a
                    # "not declared" marker reported drift on every undeclared field,
                    # so absence is normalised; a declared value stays an exact claim
                    # and a key the payload never carried stays its own failure.
                    expected = None if declared is ABSENT else declared
                    if advertised != expected:
                        shown_declared = ("NOT_DECLARED" if declared is ABSENT
                                          else repr(declared))
                        shown_get = ("MISSING_KEY" if advertised is UNREPORTED
                                     else repr(advertised))
                        issues.append(f"RUNTIME_ADVERTISEMENT_DRIFT {name}.{key}: "
                                      f"manifest={shown_declared}, "
                                      f"expected_on_wire={expected!r}, "
                                      f"GET={shown_get}")
                if live.get("context_length", "MISSING") != route.get("context_window"):
                    issues.append(f"RUNTIME_CONTEXT_DRIFT {name}: "
                                  f"manifest={route.get('context_window')!r}, "
                                  f"GET={live.get('context_length', 'MISSING')!r}")
                for medium, carrier_field in (("vision", "image_carriers"),
                                               ("video", "video_carriers")):
                    declared_carrier = route.get("image_carrier" if medium == "vision"
                                                 else "video_carrier", "both")
                    expected_carrier = ([declared_carrier]
                                        if route.get(medium) is True
                                        and declared_carrier in ("inline", "remote") else None)
                    if capabilities.get(carrier_field, "MISSING") != expected_carrier:
                        issues.append(f"RUNTIME_CARRIER_DRIFT {name}.{carrier_field}: "
                                      f"manifest={expected_carrier!r}, "
                                      f"GET={capabilities.get(carrier_field, 'MISSING')!r}")
        if target.get("video") is True and client.get("format") != "openai_responses":
            issues.append(f"VIDEO_WIRE_MISMATCH {name} {client.get('format')}")
        route_window = route.get("context_window")
        target_window = target.get("context_window")
        input_ceiling = target.get("max_input_tokens")
        if type(route_window) is not int or route_window <= 0:
            issues.append(f"ROUTE_CONTEXT_UNQUALIFIED {name}: /v1/models context_length "
                          "would be null; set a source-backed conservative route window")
        if not any(type(value) is int and value > 0
                   for value in (route_window, target_window, input_ceiling)):
            issues.append(f"CONTEXT_CAP_UNQUALIFIED {name}: no positive bounded "
                          "route window, target window, or input ceiling")
        if type(route_window) is int and type(target_window) is int and route_window > target_window:
            issues.append(f"CONTEXT_OVERCLAIM {name}: route {route_window} > target {target_window}")
        if type(input_ceiling) is int and type(target_window) is int and input_ceiling > target_window:
            issues.append(f"INPUT_CAP_OVERCLAIM {name}: input {input_ceiling} > target context {target_window}")
        upstream = canonical.get(target["id"])
        upstream_input = upstream.get("model_input", {}) if upstream else {}
        upstream_output = upstream.get("model_output", {}) if upstream else {}
        # A vendor supports an input and the current route refuses it. That
        # is NOT a claim that the vendor model is incapable; it is an actionable
        # admission gap. Video/audio can require a different native wire.
        for medium, field in (("image", "vision"), ("video", "video")):
            documented = upstream_input.get(medium)
            if documented in ("DOCUMENTED", "DOCUMENTED_INPUT") and (
                route.get(field) is not True or target.get(field) is not True
            ):
                documented_gaps.append(
                    f"UPSTREAM_{medium.upper()}_NOT_ADMITTED {name} model={target['id']} "
                    f"wire={client.get('format', 'NOT_CONFIGURED')}; "
                    "qualify the exact provider transport before changing admission"
                )
        if upstream_input.get("audio") == "DOCUMENTED":
            documented_gaps.append(
                f"UPSTREAM_AUDIO_NOT_ADMITTED {name}: existing Switchyard audio ingress "
                "refuses audio regardless of Google upstream model support"
            )
        row = {
            "route": name,
            "target": target_name,
            "model": target["id"],
            "client": target["llm_client"],
            "credential_env_name": client.get("api_key_env", "NOT_CONFIGURED"),
            "quota_group": client.get("quota_group", "NOT_CONFIGURED"),
            "transport": client.get("format", "NOT_CONFIGURED"),
            "route_capabilities": {key: route.get(key, "NOT_DECLARED") for key in BOOL_FIELDS},
            "target_capabilities": {key: target.get(key, "NOT_DECLARED") for key in BOOL_FIELDS},
            "route_image_carrier": route.get("image_carrier", "NO_NARROWING_DECLARED"),
            "target_image_carrier": target.get("image_carrier", "NO_NARROWING_DECLARED"),
            "route_video_carrier": route.get("video_carrier", "NO_NARROWING_DECLARED"),
            "target_video_carrier": target.get("video_carrier", "NO_NARROWING_DECLARED"),
            "route_context_window": route.get("context_window", "NOT_DECLARED"),
            "target_context_window": target.get("context_window", "NOT_DECLARED"),
            "max_input_tokens": target.get("max_input_tokens", "NOT_DECLARED"),
            "min_output_tokens": target.get("min_output_tokens", "NOT_DECLARED"),
            "max_output_tokens": target.get("max_output_tokens", "NOT_DECLARED"),
            "route_reasoning_policy": route.get("route_reasoning_policy", "INHERIT_TARGET"),
            "route_reasoning_control": route.get("reasoning", "NOT_DECLARED"),
            "target_reasoning_effort": target.get("reasoning_effort", "NOT_DECLARED"),
            "target_reasoning_policy": target.get("reasoning_policy", "NOT_DECLARED"),
            "target_reasoning_dialect": target.get("reasoning_dialect", "NOT_DECLARED"),
            "hosted_reasoning": target.get("hosted_reasoning", "NOT_DECLARED"),
            "accepts_reasoning_control": target.get("accepts_reasoning_control", "NOT_DECLARED"),
            "free_tier": target.get("free_tier", "NOT_DECLARED"),
            "daily_request_limit": target.get("daily_request_limit", "NOT_DECLARED"),
            "requests_per_minute": target.get("requests_per_minute", "NOT_DECLARED"),
            "input_tokens_per_minute": target.get("input_tokens_per_minute", "NOT_DECLARED"),
            "max_retries": client.get("max_retries", "NOT_DECLARED"),
            "locality": client.get("locality", "NOT_DECLARED"),
            "admission_input": {
                "text": "CHAT_INPUT",
                "image": "ADMITTED" if route.get("vision") is True and target.get("vision") is True else "REFUSED",
                "video": "ADMITTED" if route.get("video") is True and target.get("video") is True else "REFUSED",
                "audio": "REFUSED_BY_SWITCHYARD_CURRENT_INGRESS",
                "pdf": "NOT_A_STANDALONE_ROUTE_ADMISSION_FLAG",
            },
            "output_generation": "NOT_A_CHAT_DIRECT_CAPABILITY",
            "upstream_model_input": upstream_input or "NO_GOOGLE_SPEC_FOR_THIS_MODEL",
            "upstream_model_output": upstream_output or "NO_GOOGLE_SPEC_FOR_THIS_MODEL",
            "upstream_modality_source": (upstream.get("modality_source", "NOT_SPECIFIED")
                                         if upstream else "NON_GOOGLE_QUALIFICATION_SEPARATE"),
            "state": "QUALIFIED_DECLARATIONS" if not issues else "EXECUTION_METADATA_BLOCKED",
            "problems": issues,
        }
        rows.append(row)
        errors.extend(issues)

    google = [r for r in rows if r["route"].startswith("direct-google-")]
    owner = {r["model"]: r for r in google if not r["route"].startswith("direct-google-stephen-")}
    stephen = {r["model"]: r for r in google if r["route"].startswith("direct-google-stephen-")}
    if set(owner) != set(stephen):
        errors.append(f"GOOGLE_DUAL_PROFILE_MODEL_DRIFT owner={sorted(owner)} stephen={sorted(stephen)}")
    for model in set(owner) & set(stephen):
        a, b = owner[model], stephen[model]
        for key in ("target_capabilities", "route_capabilities", "max_input_tokens"):
            if a[key] != b[key]:
                errors.append(f"GOOGLE_SHARED_CAPABILITY_DRIFT {model}.{key}")
        if a["quota_group"] == b["quota_group"] or a["credential_env_name"] == b["credential_env_name"]:
            errors.append(f"GOOGLE_ACCOUNT_BOUNDARY_BROKEN {model}")
    return {"direct_count": len(names), "routes": rows, "failures": errors,
            "upstream_vs_admission_gaps": documented_gaps,
            "status": "PASS" if not errors else "REFUSED"}


def _model_entries(blob: dict) -> dict[str, dict]:
    rows = blob.get("data")
    if not isinstance(rows, list):
        _fail("MODEL_LIST_BAD_SHAPE", "/v1/models", "data must be a list")
    entries = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            _fail("MODEL_LIST_INVALID_ENTRY", "/v1/models", repr(row)[:128])
        if row["id"] in entries:
            _fail("MODEL_LIST_DUPLICATE_ID", row["id"], "duplicate runtime advertisement")
        entries[row["id"]] = row
    return entries


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--models", type=Path, default=None, help="saved live GET /v1/models JSON")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        doc = tomllib.loads(a.manifest.read_text())
        advertised = _model_entries(json.loads(a.models.read_text())) if a.models else None
        report = analyze(doc, advertised, canonical_google_cards())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['status']} direct={report['direct_count']} "
              f"qualified={len(report['routes'])-len(set(row['route'] for row in report['routes'] if row['problems']))} "
              f"failures={len(report['failures'])}")
        for err in report["failures"]:
            print(f"REFUSED {err}")
        for gap in report["upstream_vs_admission_gaps"]:
            print(f"NONADMITTED_DOCUMENTED_INPUT {gap}")
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
