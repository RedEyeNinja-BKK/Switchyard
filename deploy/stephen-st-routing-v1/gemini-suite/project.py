#!/usr/bin/env python3
"""Project the one Gemini Free Suite catalogue onto two account profiles.

The catalogue holds facts about a MODEL: its native id, its window, the wire quirk that
makes the compatibility surface accept an OpenAI-shaped body, the thought-tag marker, the
thinking floor. The profiles hold facts about an ACCOUNT: which credential and quota group
serves it, whether it has been qualified for that account, and which pool it joins.

Rendering is deterministic text, so a diff of the rendered fragment is a diff of the
intent. `--check` goes further for the owner profile: every fact it renders for a model
must equal the fact deployed in the manifest, which is how an optimisation made for one
profile is proven not to have been lost from the other.

No network, no provider call, no deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path("/var/tmp/st1/live3-routes-53fd9a84.toml")

# Fields the catalogue owns. Two profiles project these identically, so a correction is
# made once. `limits` is deliberately absent: an allowance belongs to the account.
SHARED_TARGET_FIELDS = (
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
    "decision_transport",
    "decision_path",
    "decision_base_url",
    "decision_api_key_env",
    "supported_types",
    "state_forms",
)

# Fields a profile owns, because they describe one account's budget rather than the model.
ACCOUNT_TARGET_FIELDS = (
    "daily_request_limit",
    "requests_per_minute",
    "input_tokens_per_minute",
    "max_input_tokens",
)


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def catalogue() -> dict:
    return load("catalogue.json")


def profile(which: str) -> dict:
    return load(f"profile-{which}.json")


def models_by_key(cat: dict) -> dict[str, dict]:
    return {model["key"]: model for model in cat["models"]}


def scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(scalar(item) for item in value) + "]"
    if isinstance(value, dict):
        # Inline table, sorted, so the rendered text is stable across runs.
        inner = ", ".join(f"{key} = {scalar(value[key])}" for key in sorted(value))
        return "{ " + inner + " }"
    raise TypeError(f"cannot render {value!r}")


def target_name(which: str, model: dict) -> str:
    """The target table this profile uses for one catalogue model.

    The owner keeps the deployed R6 names; a second account cannot reuse them, so Stephen's
    targets carry the same `stephen-` prefix its OpenRouter leaves already use.
    """
    if which == "owner":
        return model["route"]
    return f"stephen-google-{model['key']}"


def route_name(which: str, model: dict) -> str:
    if which == "owner":
        return model["route"]
    return f"direct-google-stephen-{model['key']}"


def cat_credential_env(prof: dict, key: str) -> str:
    name = prof["client_for"][key]
    for client in prof["clients"]:
        if client["name"] == name:
            return client["api_key_env"]
    raise KeyError(f"profile names client {name} for {key} but does not declare it")


def target_fields(which: str, cat: dict, prof: dict, model: dict) -> dict:
    # The chat wire quirks are facts about the OpenAI-compatibility surface. The decision
    # lane speaks google_decisions directly and never sends an OpenAI-shaped body, so
    # applying them there would describe a request that lane never makes.
    fields: dict = (
        dict(cat["shared_target_fields"]) if model["wire"] == "openai_compat" else {}
    )
    for key in SHARED_TARGET_FIELDS:
        if key in model:
            fields[key] = model[key]
        elif key in model.get("behaviour", {}):
            fields[key] = model["behaviour"][key]
    for key in ACCOUNT_TARGET_FIELDS:
        if key in model.get("limits", {}):
            fields[key] = model["limits"][key]
    fields["llm_client"] = prof["client_for"][model["key"]]
    if model["lane"] == "decision":
        # The key that authenticates a decision call is the account's, not the model's, so
        # it comes from the profile like every other credential reference.
        fields["decision_api_key_env"] = cat_credential_env(prof, model["key"])

    state = prof["state"][model["key"]]
    if model["provider_state"] == "retired_404":
        # The provider's own reason, held once. A profile may choose to hold a model the
        # provider still serves, and that reason is the profile's; a retirement is not.
        fields["held"] = model["retirement_evidence"]
    elif state["qualification"] == "held":
        fields["held"] = state["held"]
    # An undeclared fact renders as absence, not as a null: the loader treats an omitted
    # window as unconstrained, which is the deployed meaning for a model that states none.
    return {key: value for key, value in fields.items() if value is not None}


def render_targets(which: str, cat: dict, prof: dict) -> str:
    chunks = []
    for model in cat["models"]:
        if model["lane"] == "decision" and which == "stephen":
            # The decision lane is catalogued so both profiles agree about it, but it is
            # not deployed for this account; rendering it would claim a lane that is not
            # there.
            continue
        name = target_name(which, model)
        fields = target_fields(which, cat, prof, model)
        body = "\n".join(f"{key} = {scalar(value)}" for key, value in fields.items())
        chunks.append(f'\n[targets."{name}"]\n{body}\n')
    return "".join(chunks)


def render_owner_leaf_routes(which: str, cat: dict, prof: dict) -> str:
    chunks = []
    for model in cat["models"]:
        if model["lane"] != "chat":
            continue
        name = route_name(which, model)
        fields = {
            "id": name,
            "type": "passthrough",
            "target": target_name(which, model),
            "candidates": [],
        }
        body = "\n".join(f"{key} = {scalar(value)}" for key, value in fields.items())
        chunks.append(f'\n[routes."{name}"]\n{body}\n')
    return "".join(chunks)


def render_owner_routes(which: str, cat: dict, prof: dict) -> str:
    by_key = models_by_key(cat)
    chunks = []
    for route in prof["chat_routes"]:
        missing = [key for key in [route["target"], *route["candidates"]] if key not in by_key]
        if missing:
            raise KeyError(
                f"profile route {route['name']} names models absent from the catalogue: {missing}"
            )
        fields = {
            "id": route["name"],
            "type": route["kind"],
            "target": target_name(which, by_key[route["target"]]),
            "candidates": [target_name(which, by_key[key]) for key in route["candidates"]],
            "selection": route["selection"],
            "context_window": route["context_window"],
            "tool_calling": route["tool_calling"],
            "reasoning": route["reasoning"],
            "vision": route["vision"],
            "video": route["video"],
        }
        body = "\n".join(f"{key} = {scalar(value)}" for key, value in fields.items())
        chunks.append(f'\n[routes."{route["name"]}"]\n{body}\n')
    return "".join(chunks)


def render_clients(which: str, cat: dict, prof: dict) -> str:
    chunks = []
    for client in prof["clients"]:
        fields = {
            "format": cat["wire"]["format"],
            "base_url": cat["wire"]["base_url"],
            "locality": cat["wire"]["locality"],
            "api_key_env": client["api_key_env"],
            "quota_group": client["quota_group"],
            "max_retries": client["max_retries"],
        }
        if "failure_cooldown_ms" in client:
            fields["failure_cooldown_ms"] = client["failure_cooldown_ms"]
        body = "\n".join(f"{key} = {scalar(value)}" for key, value in fields.items())
        chunks.append(f'\n[llm_clients.{client["name"]}]\n{body}\n')
    return "".join(chunks)


def servable_keys(prof: dict) -> list[str]:
    return [
        key
        for key, state in prof["state"].items()
        if state["qualification"] == "serving"
    ]


def render_stephen_staged(cat: dict, prof: dict) -> str:
    """Stephen's own route and pool delta, for the members it can actually serve.

    A target with no serving receipt under this project is still rendered as a staged
    record: it must not enter the pool, because the first request after a deploy would
    spend a call learning that it cannot answer. The rendering states the gate and the
    exact members it would add, so qualifying a model is a one-flag change and nothing
    else moves.
    """
    by_key = models_by_key(cat)
    route = prof["chat_routes"][0]
    unknown = [key for key in route["members"] if key not in by_key]
    if unknown:
        # Answering with a narrower pool would hide the defect; a member the catalogue
        # does not hold is a profile/catalogue disagreement, and it must be visible.
        raise KeyError(f"profile route {route['name']} names uncatalogued models: {unknown}")
    serving = [key for key in route["members"] if key in servable_keys(prof)]
    staged = [key for key in route["members"] if key not in servable_keys(prof)]
    delta = prof["pool_delta"]

    lines = [
        "# ST_Gemini_Free — Stephen's account profile of the same Gemini free suite.",
        "#",
        "# Generated from gemini-suite/catalogue.json + profile-stephen.json. The facts",
        "# below are the catalogue's, so a quirk corrected for either profile is present in",
        "# both. What differs is the account: credential, quota group, qualification and",
        "# pool policy.",
        f"#",
        f"# Servable members ({len(serving)}): {', '.join(serving) if serving else 'none'}",
        f"# Staged members ({len(staged)}): {', '.join(staged) if staged else 'none'}",
        "#",
        "# A staged member is NOT a candidate. It has no serving receipt under this",
        "# project, and putting it in the pool would spend the first request after every",
        "# deploy discovering that.",
    ]
    if serving:
        fields = {
            "id": route["name"],
            "type": "passthrough",
            "target": target_name("stephen", by_key[serving[0]]),
            "candidates": [target_name("stephen", by_key[key]) for key in serving[1:]],
            "selection": route["selection"],
            "pool_groups": [route["pool_group"]],
            "context_window": route["context_window"],
            "tool_calling": route["tool_calling"],
            "reasoning": route["reasoning"],
            "vision": route["vision"],
            "video": route["video"],
        }
        body = "\n".join(f"{key} = {scalar(value)}" for key, value in fields.items())
        lines.append(f"\n[routes.{route['name']}]\n{body}\n")
    else:
        lines.append("")
        lines.append("# No member is servable under this account yet, so no route block is")
        lines.append("# emitted: an empty pool route would advertise a capability that")
        lines.append("# refuses every request. The gate is a serving receipt per member.")

    if serving:
        lines.append(
            "\n# Pool delta: ST_Free becomes the unified free pool."
            f"\n#   pool_groups += {scalar(delta['add_group'])}"
            f"\n#   candidates  += {scalar([target_name('stephen', by_key[key]) for key in serving])}"
        )
    return "\n".join(lines) + "\n"


def check_owner_against_deployed(cat: dict, prof: dict, manifest: Path) -> list[str]:
    """Prove the owner projection still carries every deployed fact.

    This is the regression half of parity: the shared catalogue may be edited for Stephen's
    sake, but it may not quietly drop or alter a fact the owner's live manifest serves.
    """
    document = tomllib.loads(manifest.read_text())
    targets = document.get("targets", {})
    routes = document.get("routes", {})
    failures: list[str] = []

    for model in cat["models"]:
        name = target_name("owner", model)
        deployed = targets.get(name)
        if deployed is None:
            failures.append(f"{name}: rendered by the owner profile but absent from {manifest.name}")
            continue
        rendered = target_fields("owner", cat, prof, model)
        for key, value in rendered.items():
            if key == "llm_client":
                continue
            if deployed.get(key, "<absent>") != value:
                failures.append(
                    f"{name}.{key}: catalogue {value!r} != deployed {deployed.get(key, '<absent>')!r}"
                )
        # A deployed fact the renderer does not know about is drift in the other
        # direction, and it is the direction that loses capability.
        known = set(rendered) | {"id", "llm_client"}
        for key in deployed:
            if key not in known:
                failures.append(f"{name}.{key}: deployed {deployed[key]!r} is not in the catalogue")

    for route in prof["chat_routes"]:
        deployed = routes.get(route["name"])
        if deployed is None:
            failures.append(f"{route['name']}: declared by the profile but absent from the manifest")
            continue
        by_key = models_by_key(cat)
        expected = {
            "target": target_name("owner", by_key[route["target"]]),
            "candidates": [
                target_name("owner", by_key[key]) for key in route["candidates"]
            ],
            "selection": route["selection"],
            "context_window": route["context_window"],
            "tool_calling": route["tool_calling"],
            "reasoning": route["reasoning"],
            "vision": route["vision"],
            "video": route["video"],
        }
        for key, value in expected.items():
            if deployed.get(key, "<absent>") != value:
                failures.append(
                    f"routes.{route['name']}.{key}: profile {value!r} != deployed {deployed.get(key, '<absent>')!r}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=HERE / "rendered")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    cat = catalogue()
    owner = profile("owner")
    stephen = profile("stephen")

    arguments.out.mkdir(parents=True, exist_ok=True)
    written = {
        "owner-clients.toml": render_clients("owner", cat, owner),
        "owner-targets.toml": render_targets("owner", cat, owner),
        "owner-leaves.toml": render_owner_leaf_routes("owner", cat, owner),
        "owner-routes.toml": render_owner_routes("owner", cat, owner),
        "stephen-clients.toml": render_clients("stephen", cat, stephen),
        "stephen-targets.toml": render_targets("stephen", cat, stephen),
        "stephen-routes.toml": render_stephen_staged(cat, stephen),
    }
    for name, text in written.items():
        (arguments.out / name).write_text(text)
    # Parse every fragment, so a rendering bug is a failure here rather than a broken
    # deployment later.
    for name in written:
        tomllib.loads((arguments.out / name).read_text())

    if not arguments.check:
        print(f"rendered {len(written)} fragments into {arguments.out}")
        return 0

    failures = check_owner_against_deployed(cat, owner, arguments.manifest)
    for failure in failures:
        print(f"OWNER-PARITY FAIL: {failure}")
    print(f"owner profile vs {arguments.manifest.name}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
