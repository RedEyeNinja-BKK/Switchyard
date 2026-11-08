# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The route manifest contract, tested without a server.

`route_audit.py` compares a manifest against the model list a server advertises.
The interesting half is not "does a correct manifest pass" -- it is whether each
invariant actually rejects the mistake it exists for. Every case below takes a
manifest that passes, breaks exactly one thing, and requires a specific failure.

Two of these controls are here because the first version of the audit was wrong:

  * `unbounded` is not a bounded route. A substring test says it is, which would
    fail a correct manifest, so the rule is a suffix test and this case pins it.
  * candidate fallback ACROSS providers is designed (`smart` falls from a local
    model to another provider's). Rejecting it would reject the deployment, so the
    credential rule is opt-in and only fires for a pair the caller declares.
"""

import os
import sys
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "release"))

import route_audit  # noqa: E402

FIXTURE = os.path.join(
    ROOT, "crates", "switchyard-server", "tests", "fixtures", "routes.audit.toml"
)


@pytest.fixture()
def manifest() -> dict:
    with open(FIXTURE, "rb") as handle:
        return tomllib.load(handle)


def advertised_from(config: dict) -> dict:
    """The advertisement a correct server would produce for this manifest."""
    return {
        name: {
            "id": name,
            "context_length": route.get("context_window"),
            "capabilities": {
                "vision": route.get("vision"),
                "tool_calling": route.get("tool_calling"),
                "video": route.get("video"),
            },
        }
        for name, route in config["routes"].items()
    }


def verdict(config: dict, advertised: dict, exclusive=()) -> list[str]:
    _, failures = route_audit.audit(config, advertised, exclusive)
    return failures


def test_the_committed_fixture_satisfies_every_invariant(manifest):
    assert verdict(manifest, advertised_from(manifest)) == []


def test_a_composite_must_not_be_advertised(manifest):
    advertised = advertised_from(manifest)
    advertised["cascade"] = {"id": "cascade", "context_length": 262144, "capabilities": {}}
    failures = verdict(manifest, advertised)
    assert any("composite is advertised" in failure for failure in failures), failures


def test_an_unadvertised_leaf_is_a_failure(manifest):
    advertised = advertised_from(manifest)
    del advertised["direct-remote-b-flash"]
    failures = verdict(manifest, advertised)
    assert any("not advertised" in failure for failure in failures), failures


def test_advertisement_must_agree_on_the_context_window(manifest):
    advertised = advertised_from(manifest)
    advertised["direct-remote-a-omni"]["context_length"] = 1
    failures = verdict(manifest, advertised)
    assert any("context_window" in failure for failure in failures), failures


def test_advertisement_must_agree_on_each_declared_capability(manifest):
    advertised = advertised_from(manifest)
    advertised["direct-remote-a-omni"]["capabilities"]["vision"] = False
    failures = verdict(manifest, advertised)
    assert any("vision" in failure for failure in failures), failures


def test_a_bounded_route_must_pin_reasoning_none(manifest):
    # The deployed defect this rule exists for: the name says bounded, the policy
    # says inherit, so the route advertises a contract it does not keep.
    manifest["routes"]["prims-unbounded-bounded"]["route_reasoning_policy"] = "inherit"
    failures = verdict(manifest, advertised_from(manifest))
    assert any("bounded route must pin" in failure for failure in failures), failures


def test_unbounded_is_not_a_bounded_route(manifest):
    # `prims-unbounded` inherits reasoning and must NOT be required to pin none.
    # A substring test for "bounded" would fail this correct manifest.
    assert manifest["routes"]["prims-unbounded"]["route_reasoning_policy"] == "inherit"
    assert verdict(manifest, advertised_from(manifest)) == []


def test_a_bounded_composite_must_name_only_bounded_children(manifest):
    manifest["composites"]["tier-a-bounded"]["children"] = [
        "prims-unbounded",
        "prims-local-bounded",
    ]
    failures = verdict(manifest, advertised_from(manifest))
    assert any("unbounded child" in failure for failure in failures), failures


def test_a_nested_composite_is_refused(manifest):
    manifest["composites"]["tier-a"]["children"] = ["cascade"]
    failures = verdict(manifest, advertised_from(manifest))
    assert any("itself a composite" in failure for failure in failures), failures


def test_video_on_a_chat_transport_is_refused(manifest):
    # The Chat codec has no faithful video representation, so a video target there
    # would admit a request and deliver text.
    manifest["llm_clients"]["remote_a"]["format"] = "openai_chat"
    failures = verdict(manifest, advertised_from(manifest))
    assert any("non-responses client" in failure for failure in failures), failures


def test_candidate_fallback_across_providers_is_allowed(manifest):
    # Designed behaviour: a route may fall from one provider to another. Refusing
    # this would refuse the deployment.
    manifest["routes"]["direct-remote-b-flash"]["candidates"] = ["remote-a-omni"]
    assert verdict(manifest, advertised_from(manifest)) == []


def test_mixing_declared_exclusive_credentials_is_refused(manifest):
    # Same manifest, but the caller declares these two clients must not share a
    # route. Which accounts those are is deployment knowledge, not a constant.
    manifest["routes"]["direct-remote-b-flash"]["candidates"] = ["remote-a-omni"]
    failures = verdict(manifest, advertised_from(manifest), ({"remote_a", "remote_b"},))
    assert any("mixes credentials" in failure for failure in failures), failures


def test_an_unknown_candidate_is_refused(manifest):
    manifest["routes"]["direct-remote-a-text"]["candidates"] = ["no-such-target"]
    failures = verdict(manifest, advertised_from(manifest))
    assert any("is not defined" in failure for failure in failures), failures


def test_an_unknown_target_client_is_refused(manifest):
    manifest["targets"]["remote-a-text"]["llm_client"] = "unconfigured"
    failures = verdict(manifest, advertised_from(manifest))
    assert any("is not configured" in failure for failure in failures), failures


def test_inventory_covers_every_definition_including_composites(manifest):
    inventory, _ = route_audit.audit(manifest, advertised_from(manifest))
    leaves = set(manifest["routes"])
    composites = set(manifest.get("composites", {}))
    rows = {row["route"] for row in inventory}
    # No definition may disappear from the audit because another shares its model.
    assert rows == leaves | composites
    assert len(inventory) == len(leaves) + len(composites)


def test_inventory_classifies_a_bounded_route_by_name_not_by_model(manifest):
    inventory, _ = route_audit.audit(manifest, advertised_from(manifest))
    by_name = {row["route"]: row for row in inventory}
    # Two routes share the local target but are distinct definitions with distinct
    # reasoning contracts; collapsing them by model would hide one.
    assert by_name["prims-local"]["target"] == by_name["prims-local-bounded"]["target"]
    assert by_name["prims-local"]["policy"] == "medium"
    assert by_name["prims-local-bounded"]["policy"] == "none"
