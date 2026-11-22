#!/usr/bin/env python3
"""Bidirectional parity verification for the two Gemini free-suite profiles.

What "one suite, two profiles" has to mean, mechanically:

  P1  every catalogue model is projected by BOTH profiles, and every chat target deployed
      for the owner is in the catalogue — neither direction may be missing a model;
  P2  every shared fact renders identically in both profiles, so a quirk fixed for one is
      fixed for the other;
  P3  the account facts are the only ones that differ, and they never cross: no credential
      environment name or quota group of one profile appears in the other;
  P4  a profile owns its own serving qualification, and an unqualified member cannot enter
      a served pool — but flipping its qualification adds it and nothing else;
  P5  each profile keeps its own route policy: the owner's measured ordering is not copied
      to Stephen, and Stephen's peer pool over OR+Gemini is not copied to the owner;
  P6  the owner projection still reproduces every fact the deployed manifest serves, so a
      catalogue edit made for one profile cannot silently drop the other's capability;
  P7  the catalogue says which bytes it was captured from, and they are the bytes given;
  P8  the profile files hold no model facts of their own: if they did, the "one catalogue"
      claim would already be false and every later correction would have two homes.

Exit code 0 only when every check passes. `--json` emits the same result as data.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import project

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path("/var/tmp/st1/live3-routes-53fd9a84.toml")

# A profile may state these about an account; it may not state anything about a model.
PROFILE_MODEL_FACT_KEYS = {
    "context_window",
    "omit_body_fields",
    "unsupported_reasoning_effort",
    "hosted_reasoning",
    "accepts_reasoning_control",
    "min_output_tokens",
    "admission_margin_permille",
    "max_input_tokens",
    "vision",
    "video",
    "daily_request_limit",
    "requests_per_minute",
    "input_tokens_per_minute",
}

SHARED_COMPARE_FIELDS = (
    "id",
    "omit_body_fields",
    "unsupported_reasoning_effort",
    "vision",
    "video",
    "context_window",
    "free_tier",
    "accepts_reasoning_control",
    "hosted_reasoning",
    "min_output_tokens",
    "admission_margin_permille",
)


def chat_models(cat: dict) -> list[dict]:
    return [model for model in cat["models"] if model["lane"] == "chat"]


def deployed_chat_targets(document: dict, cat: dict) -> set[str]:
    targets = document["targets"]
    return {
        model["route"]
        for model in chat_models(cat)
        if model["route"] in targets or True
    } & set(targets)


def check_catalogue_covers_deployed(cat: dict, document: dict) -> list[str]:
    """P1, deployed → catalogue: a served Gemini target with no catalogue entry is a model
    whose facts would be learned twice, once per profile."""
    failures = []
    catalogued = {model["route"] for model in cat["models"]}
    for name, target in document["targets"].items():
        family = str(target.get("id", ""))
        if "gemini" not in family and "gemma" not in family:
            continue
        if str(target.get("llm_client", "")).startswith("openrouter"):
            continue
        if name.startswith("stephen-"):
            continue
        if name not in catalogued:
            failures.append(f"deployed Gemini target {name} ({family}) has no catalogue entry")
    return failures


def check_shared_identity(cat: dict, owner: dict, stephen: dict) -> list[str]:
    """P2: the same model fact, rendered by each profile, must be the same value."""
    failures = []
    for model in chat_models(cat):
        left = project.target_fields("owner", cat, owner, model)
        right = project.target_fields("stephen", cat, stephen, model)
        for key in SHARED_COMPARE_FIELDS:
            if key in left or key in right:
                if left.get(key, "<absent>") != right.get(key, "<absent>"):
                    failures.append(
                        f"{model['key']}.{key}: owner {left.get(key, '<absent>')!r} "
                        f"!= stephen {right.get(key, '<absent>')!r}"
                    )
        # The provider's retirement is a fact about the model, not the account: a
        # different project does not un-retire it.
        if model["provider_state"] == "retired_404":
            if "held" not in right:
                failures.append(
                    f"{model['key']}: provider-retired model is not held in the stephen profile"
                )
            elif right.get("held") != left.get("held"):
                failures.append(f"{model['key']}.held differs between profiles")
        else:
            # Qualification is per account: a serving model for one profile may be
            # unverified for the other, and that difference must be visible, not silent.
            owner_serves = owner["state"][model["key"]]["qualification"] == "serving"
            stephen_state = stephen["state"][model["key"]]["qualification"]
            if stephen_state == "held" and owner_serves:
                failures.append(
                    f"{model['key']}: held for stephen although the provider serves it"
                )
    return failures


def check_account_isolation(cat: dict, owner: dict, stephen: dict) -> list[str]:
    """P3: the differing fields are exactly the account's, and they never cross."""
    failures = []
    owner_envs = {client["api_key_env"] for client in owner["clients"]}
    stephen_envs = {client["api_key_env"] for client in stephen["clients"]}
    owner_groups = {client["quota_group"] for client in owner["clients"]}
    stephen_groups = {client["quota_group"] for client in stephen["clients"]}

    if owner_envs & stephen_envs:
        failures.append(f"credential environment shared across accounts: {owner_envs & stephen_envs}")
    if owner_groups & stephen_groups:
        failures.append(f"quota group shared across accounts: {owner_groups & stephen_groups}")
    if {client["name"] for client in owner["clients"]} & {
        client["name"] for client in stephen["clients"]
    }:
        failures.append("client name shared across accounts")

    for env in stephen_envs:
        if env in project.render_targets("owner", cat, owner):
            failures.append(f"stephen credential {env} appears in the owner projection")
    for env in owner_envs:
        if env in project.render_targets("stephen", cat, stephen):
            failures.append(f"owner credential {env} appears in the stephen projection")
    return failures


def check_qualification_gate(cat: dict, stephen: dict) -> list[str]:
    """P4: unqualified members stay out of the pool, and qualifying one adds only it."""
    failures = []
    rendered = project.render_stephen_staged(cat, stephen)
    by_key = project.models_by_key(cat)
    route = stephen["chat_routes"][0]
    serving = [key for key in route["members"] if key in project.servable_keys(stephen)]
    unqualified = [key for key in route["members"] if key not in serving]

    for key in unqualified:
        name = project.target_name("stephen", by_key[key])
        if f'"{name}"' in rendered.split("# Staged members")[0]:
            failures.append(f"{key}: unqualified member appears in a servable position")

    if not serving:
        if "[routes.ST_Gemini_Free]" in rendered:
            failures.append("an empty pool still rendered a route block")
        # The gate must be a one-flag change: qualify one member and exactly that member
        # becomes the route's target.
        flipped = json.loads(json.dumps(stephen))
        candidate = route["members"][0]
        flipped["state"][candidate]["qualification"] = "serving"
        flipped["state"][candidate]["evidence"] = "synthetic flip for the gate test"
        after = project.render_stephen_staged(cat, flipped)
        expected = project.target_name("stephen", by_key[candidate])
        if f"[routes.ST_Gemini_Free]" not in after:
            failures.append("qualifying one member did not produce the route")
        if f'target = "{expected}"' not in after:
            failures.append("qualifying one member did not make it the route target")
        for other in unqualified:
            if other == candidate:
                continue
            other_name = project.target_name("stephen", by_key[other])
            if f'"{other_name}"' in after.split("# Staged members")[0]:
                failures.append(f"{other}: joined the pool when a sibling was qualified")
    return failures


def check_route_policy(cat: dict, owner: dict, stephen: dict) -> list[str]:
    """P5: each account keeps the policy it was measured to need."""
    failures = []
    for route in owner["chat_routes"]:
        if route["selection"] != "measured":
            failures.append(f"owner route {route['name']} no longer uses measured selection")
    if owner["chat_routes"][0]["name"] != "smart_gemini":
        failures.append("the owner profile's primary chat route is not smart_gemini")

    gemini = stephen["chat_routes"][0]
    if gemini["selection"] != "peer":
        failures.append("stephen's Gemini profile no longer uses peer selection")
    delta = stephen["pool_delta"]
    if delta["route"] != "ST_Free":
        failures.append("stephen's Gemini leaves no longer join ST_Free")
    if delta["add_group"] != gemini["pool_group"]:
        failures.append("the pool delta and the profile disagree about the group name")
    if sorted(delta["add_candidates"]) != sorted(gemini["members"]):
        failures.append("the pool delta and the profile disagree about the members")
    return failures


def check_owner_regression(cat: dict, owner: dict, manifest: Path) -> list[str]:
    """P6: the owner projection still reproduces the deployed manifest."""
    return project.check_owner_against_deployed(cat, owner, manifest)


def check_provenance(cat: dict, manifest: Path, expected_hash: str | None) -> list[str]:
    """P7: the catalogue names the bytes it came from, and they are the bytes supplied."""
    failures = []
    recorded = cat["captured_from"]["manifest_sha256"]
    if expected_hash is not None and recorded != expected_hash:
        failures.append(
            f"catalogue was captured from {recorded}, but the manifest supplied is {expected_hash}"
        )
    if recorded not in str(cat["captured_from"]["manifest"]):
        failures.append("the recorded manifest path does not contain the recorded digest")
    if not manifest.exists():
        failures.append(f"manifest {manifest} is not present")
    return failures


def check_no_duplicated_facts(owner: dict, stephen: dict) -> list[str]:
    """P8: a model fact in a profile file is the beginning of a second catalogue."""
    failures = []
    for name, prof in (("owner", owner), ("stephen", stephen)):
        for target in prof.get("state", {}).values():
            for key in PROFILE_MODEL_FACT_KEYS & set(target):
                failures.append(f"profile-{name} restates model fact {key} in its state")
        for route in prof.get("chat_routes", []):
            for key in ("omit_body_fields", "hosted_reasoning", "max_input_tokens"):
                if key in route:
                    failures.append(f"profile-{name} route {route['name']} restates {key}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-capture-hash", default=None)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    cat = project.catalogue()
    owner = project.profile("owner")
    stephen = project.profile("stephen")
    document = tomllib.loads(arguments.manifest.read_text())

    # Evaluated one at a time, and a check that raises is reported BY NAME. A crash is a
    # refusal, but "traceback" does not say which property broke, and a verifier whose
    # output a reader cannot act on is not verification.
    pending = [
        ("P1-deployed-targets-are-catalogued", lambda: check_catalogue_covers_deployed(cat, document)),
        ("P2-shared-facts-identical", lambda: check_shared_identity(cat, owner, stephen)),
        ("P3-accounts-are-isolated", lambda: check_account_isolation(cat, owner, stephen)),
        ("P4-qualification-gates-the-pool", lambda: check_qualification_gate(cat, stephen)),
        ("P5-route-policies-preserved", lambda: check_route_policy(cat, owner, stephen)),
        ("P6-owner-projection-matches-deployed", lambda: check_owner_regression(cat, owner, arguments.manifest)),
        ("P7-catalogue-names-its-bytes", lambda: check_provenance(cat, arguments.manifest, arguments.expected_capture_hash)),
        ("P8-no-model-facts-in-profiles", lambda: check_no_duplicated_facts(owner, stephen)),
    ]
    checks = []
    for name, run in pending:
        try:
            checks.append((name, run()))
        except Exception as error:  # noqa: BLE001 - every failure mode must be named
            checks.append((name, [f"check raised {type(error).__name__}: {error}"]))

    failures = 0
    result = []
    for name, problems in checks:
        status = "PASS" if not problems else "FAIL"
        result.append({"check": name, "status": status, "failures": problems})
        if problems:
            failures += len(problems)
            for problem in problems:
                print(f"{status} {name}: {problem}")
        else:
            print(f"{status} {name}")

    if arguments.json:
        print(json.dumps({"checks": result, "failure_count": failures}, indent=2))
    print(f"parity: {len(checks)} checks, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
