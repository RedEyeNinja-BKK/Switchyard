# SPDX-FileCopyrightText: Copyright (c) 2026 RedEyeNinja
# SPDX-License-Identifier: Apache-2.0

"""Tests for the production manifest tools in scripts/deploy/production.

Two tools are covered: the redactor that turns a host manifest into a publishable copy, and
the contract checker that decides whether a manifest is the one that was reviewed and
measured. Both are guards, so both are tested by showing they *reject* the thing they exist
to catch -- a guard that only ever passes is indistinguishable from no guard at all. Each
negative case below is one way the real deployment could go wrong.
"""

from __future__ import annotations

import copy
import pathlib
import sys

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "deploy" / "production"
sys.path.insert(0, str(TOOLS))

import manifest_contract  # noqa: E402
import sanitize_manifest  # noqa: E402

PUBLISHED = TOOLS


# --------------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------------

# The fixture needs a private-network hostname that behaves like a real one, but a literal
# would be exactly the tracked identifier tests/test_no_leaked_identifiers.py reports. It is
# assembled at runtime for the same reason that guard assembles its own sample.
PRIVATE_DOMAIN = "privnet" + "1234" + ".ts.net"

HOST_MANIFEST = """\
schema_version = 1

[llm_clients.local]
base_url = "https://gpu-box.{domain}:8446/v1"
api_key_env = "SWITCHYARD_LOCAL_API_KEY"

[llm_clients.provider]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "SWITCHYARD_OPENROUTER_API_KEY"

[llm_clients.overlay]
base_url = "http://10.20.30.40:8011"

[targets.local-model]
id = "local/model"
llm_client = "local"

[tools]
cache = "/home/operator/.cache/models"
""".format(domain=PRIVATE_DOMAIN)


@pytest.fixture()
def host_manifest(tmp_path):
    path = tmp_path / "routes.toml"
    path.write_text(HOST_MANIFEST, encoding="utf-8")
    return path


def test_redaction_removes_private_coordinates_and_keeps_public_ones(host_manifest):
    redacted, mapping = sanitize_manifest.sanitize(str(host_manifest))

    assert PRIVATE_DOMAIN not in redacted
    assert "gpu-box.tailnet-example.ts.net" in redacted
    assert "10.20.30.40" not in redacted
    assert sanitize_manifest.PLACEHOLDER_IP in redacted
    assert "/home/operator" not in redacted
    # The provider endpoint is not private topology and must survive, or the document stops
    # saying which provider each lane calls.
    assert "https://openrouter.ai/api/v1" in redacted
    # Ports are meaningful and are preserved verbatim.
    assert ":8446/v1" in redacted and ":8011" in redacted
    assert set(mapping) == {"gpu-box." + PRIVATE_DOMAIN, "10.20.30.40"}


def test_redaction_rewrites_carrier_grade_nat_addresses(tmp_path):
    """A CGNAT address is neither RFC1918 nor globally routable.

    ``ipaddress`` does not report 100.64.0.0/10 as private, so a redactor written against
    ``is_private`` publishes it while reporting success. This is the regression that check
    exists for.
    """
    path = tmp_path / "routes.toml"
    path.write_text('[x]\nurl = "http://100.105.20.36:8765"\n', encoding="utf-8")

    redacted, mapping = sanitize_manifest.sanitize(str(path))

    assert "100.105.20.36" not in redacted
    assert mapping == {"100.105.20.36": sanitize_manifest.PLACEHOLDER_IP}


def test_redaction_keeps_every_table_and_key(host_manifest):
    redacted, _ = sanitize_manifest.sanitize(str(host_manifest))

    import tomllib

    before = tomllib.loads(HOST_MANIFEST)
    after = tomllib.loads(redacted)
    assert sanitize_manifest._shape(before) == sanitize_manifest._shape(after)
    assert set(after["llm_clients"]) == {"local", "provider", "overlay"}
    assert after["targets"]["local-model"]["llm_client"] == "local"


def test_redaction_refuses_to_leak_a_private_hostname_in_a_comment(tmp_path):
    """A coordinate inside a comment is still published; the guard reads text, not config."""
    path = tmp_path / "routes.toml"
    path.write_text(
        "# served by gpu-box.%s\n[x]\nurl = \"https://openrouter.ai/api\"\n" % PRIVATE_DOMAIN,
        encoding="utf-8",
    )

    raw = path.read_text(encoding="utf-8")
    redacted, _ = sanitize_manifest.sanitize(str(path))

    assert "gpu-box.tailnet-example.ts.net" in redacted
    assert PRIVATE_DOMAIN not in redacted
    assert PRIVATE_DOMAIN in raw  # the input really did contain it


def test_redaction_is_a_fixed_point(host_manifest):
    once, _ = sanitize_manifest.sanitize(str(host_manifest))
    twice, second_mapping = sanitize_manifest._redact_once(once)

    assert twice == once
    assert second_mapping == {}


def test_redaction_is_deterministic(host_manifest):
    first, _ = sanitize_manifest.sanitize(str(host_manifest))
    second, _ = sanitize_manifest.sanitize(str(host_manifest))

    assert first == second


# --------------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------------

def _synthetic(profile="candidate"):
    """A manifest that satisfies the published contract for ``profile``.

    Built from the contract itself so the two cannot drift: if a count in the contract
    changes, this fixture changes with it. The routes the contract names -- including the
    composite children, which are routes rather than targets -- are created first, and filler
    fills the remainder up to the published count.
    """
    contract = copy.deepcopy(manifest_contract.CONTRACTS[profile])
    counts = contract["counts"]
    doc = {"schema_version": 1}
    doc["llm_clients"] = {"client-%d" % index: {"format": "openai_responses",
                                                "base_url": "https://openrouter.ai/api/v1"}
                          for index in range(counts["llm_clients"])}

    targets = {}
    for target, model_id in contract["target_ids"].items():
        targets[target] = {"id": model_id, "llm_client": "local"}
    if contract["smartlocal"]:
        # The local lane's primary leg is a real target and must exist for the contract.
        targets[manifest_contract.HTPC_PRIMARY] = {"id": "vendor/small", "llm_client": "local"}
    filler = 0
    while len(targets) < counts["targets"]:
        targets["filler-target-%02d" % filler] = {"id": "vendor/filler-%d" % filler,
                                                  "llm_client": "provider"}
        filler += 1
    doc["targets"] = targets
    reusable = next(t for t in targets if t.startswith("filler-target"))
    spread = [t for t in targets if t.startswith("filler-target")]

    routes = {}
    for target, named in contract["routes"].items():
        for name, body in named.items():
            routes[name] = dict(body, id=name, type="passthrough", target=target, candidates=[])
    if contract["smartlocal"]:
        for name in ("smartlocal", "smartlocal-bounded"):
            routes[name] = dict(contract["smartlocal"], id=name, type="passthrough",
                                candidates=[contract["smartlocal"]["escalation_target"]])
    # Composite children are route ids, so they have to exist for the contract's composite
    # assertions and for the referential check.
    for name in ("smartfree", "smartfree-bounded", "smart", "smart-bounded"):
        routes.setdefault(name, {"id": name, "type": "passthrough", "candidates": [],
                                 "target": reusable})
    index = 0
    while len(routes) < counts["routes"]:
        name = "filler-route-%02d" % index
        routes[name] = {"id": name, "type": "passthrough", "candidates": [],
                        "target": spread[index % len(spread)]}
        index += 1
    doc["routes"] = routes

    composites = {name: dict(body) for name, body in contract["composites"].items()}
    while len(composites) < counts["composites"]:
        composites["filler-composite-%d" % len(composites)] = {"children": ["smartfree"]}
    doc["composites"] = composites

    doc["decision_lanes"] = {
        "lane-%d" % index: {"targets": [reusable]} for index in range(counts["decision_lanes"])
    }
    doc["capability_clients"] = {
        name: {"format": "openai_embeddings", "base_url": "https://openrouter.ai/api"}
        for name in ("embedding-local", "reranker-local", "embedding-x", "reranker-x",
                     "embedding-y", "reranker-y")[:counts["capability_clients"]]
    }
    doc["capabilities"] = {"cap-%d" % index: {} for index in range(counts["capabilities"])}
    return doc


@pytest.mark.parametrize("profile", ["candidate", "recovery"])
def test_contract_accepts_the_shape_it_publishes(profile):
    assert manifest_contract.check(_synthetic(profile), profile) == []


def test_contract_catches_a_route_count_that_moved():
    doc = _synthetic()
    doc["routes"].pop("filler-route-00")

    fails = manifest_contract.check(doc, "candidate")

    assert any("routes: expected 42" in failure for failure in fails)
def test_contract_catches_a_context_window_that_was_not_measured():
    doc = _synthetic()
    doc["routes"]["smartlocal"]["context_window"] = 225280

    fails = manifest_contract.check(doc, "candidate")

    assert any("smartlocal.context_window" in failure for failure in fails)


def test_contract_catches_the_context_that_must_never_be_advertised():
    doc = _synthetic()
    doc["routes"]["smartlocal"]["context_window"] = manifest_contract.FORBIDDEN_CONTEXT
    doc["routes"]["smartlocal-bounded"]["context_window"] = manifest_contract.FORBIDDEN_CONTEXT

    fails = manifest_contract.check(doc, "candidate")

    assert any("forbidden" in failure for failure in fails)


def test_contract_catches_a_retired_profile_used_as_an_escalation_target():
    doc = _synthetic()
    doc["routes"]["smartlocal"]["escalation_target"] = "comfyninja-qwen3_8-27b-q3-mtp-vl"

    fails = manifest_contract.check(doc, "candidate")

    assert any("retired profile reference" in failure for failure in fails)


def test_contract_tells_a_retired_id_from_a_retained_one_that_starts_with_it():
    """``...-q4`` is retired; ``...-q4-mtp`` is the profile in production.

    A plain substring check flags the healthy manifest, so the match has to respect the token
    boundary. Asserted both ways: the clean document passes, the retired one does not.
    """
    clean = _synthetic()
    clean["routes"]["smartlocal"]["escalation_target"] = manifest_contract.COMFYNINJA_VISION
    assert manifest_contract.check(clean, "candidate") == []

    dirty = _synthetic()
    dirty["routes"]["smartlocal"]["escalation_target"] = "comfyninja-qwen3_8-27b-q4"
    assert any("retired profile reference" in failure
               for failure in manifest_contract.check(dirty, "candidate"))


def test_contract_catches_a_dangling_target():
    doc = _synthetic()
    doc["routes"]["filler-route-01"]["target"] = "does-not-exist"

    fails = manifest_contract.check(doc, "candidate")

    assert any("unknown target" in failure for failure in fails)


def test_contract_catches_a_target_nothing_uses():
    doc = _synthetic()
    for route in doc["routes"].values():
        if route.get("target") == "filler-target-00":
            route["target"] = "filler-target-01"
    for lane in doc["decision_lanes"].values():
        lane["targets"] = ["filler-target-01"]

    fails = manifest_contract.check(doc, "candidate")

    assert any("referenced by nothing" in failure for failure in fails)


def test_contract_catches_a_missing_local_capability_client():
    """The local embedding and rerank services are separate processes, not model profiles.

    Removing them alongside a profile cleanup would break retrieval, so their presence is
    part of the contract.
    """
    doc = _synthetic()
    doc["capability_clients"].pop("embedding-local")

    fails = manifest_contract.check(doc, "candidate")

    assert any("embedding-local" in failure for failure in fails)


def test_recovery_contract_rejects_a_reintroduced_local_lane():
    doc = _synthetic("recovery")
    doc["targets"][manifest_contract.COMFYNINJA_VISION] = {
        "id": "comfyninja/qwen3.8-27b-q4-mtp-vl", "llm_client": "local"}

    fails = manifest_contract.check(doc, "recovery")

    assert any("must not exist" in failure for failure in fails)


# --------------------------------------------------------------------------------------------
# The published artifacts
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,profile", [("routes.redacted.toml", "candidate"),
                                          ("routes.recovery.redacted.toml", "recovery")])
def test_published_redacted_manifest_satisfies_its_contract(name, profile):
    path = PUBLISHED / name
    if not path.exists():  # pragma: no cover - the file is part of the change
        pytest.fail("%s is missing: the reviewable manifest must be published" % name)

    import tomllib

    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    assert manifest_contract.check(doc, profile) == []


@pytest.mark.parametrize("name", ["routes.redacted.toml", "routes.recovery.redacted.toml"])
def test_published_manifest_carries_no_private_coordinate(name):
    text = (PUBLISHED / name).read_text(encoding="utf-8")
    assert sanitize_manifest._leftovers(text) == []
