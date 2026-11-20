#!/usr/bin/env python3
"""Turn the owner-attested Gemini free-project crosswalk into a truthful inventory.

The crosswalk is evidence about what the owner's AI Studio project *allocates*.
It is not evidence that a model is reachable over the wire, that Switchyard has a
transport for its operation, or that a route should be activated. Those are three
separate questions, and this module keeps them separate:

  allocation  -> carried through verbatim, including zero-RPD rows, because a
                 zero observed peak is not the absence of an allowance.
  transport   -> derived from the crosswalk's own operation contract. Only
                 `chat.complete` maps onto an OpenAI-compatible chat leaf; every
                 other operation needs an adapter Switchyard does not have yet.
  activation  -> only a model that is both chat-transport and wire-qualified may
                 appear in a route candidate list.

Nothing here contacts Google, and nothing here writes a production manifest. It
emits an inventory, a route graph and a reviewable TOML fragment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = 1

# Operations a chat-completions leaf can serve. Everything else is a typed
# operation with its own wire shape, and pretending otherwise is the defect this
# whole module exists to prevent.
CHAT_OPERATIONS = frozenset({"chat.complete"})

# What an operator would have to build before each non-chat operation could be
# routed. Deliberately concrete: "adapter required" alone reads as a promise,
# and these are the actual shapes.
ADAPTERS = {
    "agent.run": (
        "managed-agent session API (agents.run) over an asynchronous task/session "
        "handle; not a chat completion. Needs an adapter that submits, polls and "
        "projects agent output into Switchyard's response shape."
    ),
    "session.live": (
        "bidirectional Live API session (BidiGenerateContent over a WebSocket). "
        "Needs a stateful streaming session adapter with audio in/out; a request/"
        "response leaf cannot carry it."
    ),
    "audio.synthesize": (
        "speech synthesis (generateContent with responseModalities=AUDIO) returning "
        "audio bytes rather than chat text. Needs an audio-payload response path and "
        "a voice/format selector."
    ),
    "audio.transcribe": (
        "transcription (generateContent over an audio part) returning text and "
        "segment metadata. Needs an audio-input carrier and a transcription-shaped "
        "response, not a chat completion."
    ),
    "vector.embed": (
        "embedding endpoint (:embedContent / :batchEmbedContents) returning vectors. "
        "Switchyard has no embeddings surface for this deployment; needs a typed "
        "embedding route, not a chat leaf."
    ),
    "decision.score": (
        "typed decision transport. Switchyard already has a dedicated decision wire "
        "family and a per-target decision_* configuration block; this row needs that "
        "transport bound to it, not a chat leaf."
    ),
    "image.generate": (
        "image generation (responseModalities=IMAGE) returning image parts. Needs an "
        "image-payload response path and a size/aspect selector."
    ),
    "music.stream": (
        "music generation endpoint returning an audio stream. Needs a streaming "
        "audio adapter with its own session lifetime."
    ),
    "robotics.perceive": (
        "robotics ER endpoint with embodied-action outputs. Needs a robotics adapter; "
        "the response is not text."
    ),
    "world.generate": (
        "world-model generation producing interactive world state. Needs a bespoke "
        "adapter; no part of it fits a chat leaf."
    ),
    "multimodal.generate": (
        "omni multimodal generation returning mixed-modality parts. Needs a "
        "multi-part response adapter with per-modality egress accounting."
    ),
    "unknown": (
        "operation not identified in the evidence. It must be classified before any "
        "adapter work; routing it as chat would guess."
    ),
}

# A per-minute token budget below this cannot carry a long prompt in one minute,
# so the model is qualified for small bounded work only regardless of its RPD.
SMALL_CONTEXT_TPM_CEILING = 100_000


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(raw)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quota(model: dict) -> dict:
    display = model.get("quota_display") or {}

    def one(key: str) -> dict:
        value = display.get(key) or {}
        return {
            "limit": value.get("limit"),
            "peak_usage_selected_1d": value.get("peak_usage_selected_1d"),
        }

    return {"rpm": one("rpm"), "tpm": one("tpm"), "rpd": one("rpd")}


def _classify(model: dict) -> tuple[str, str | None, bool]:
    """Return (state, adapter_requirement, chat_capable).

    `state` says what may be done with this row *today*; it never claims that the
    allocation is absent.
    """
    operation = model.get("operation_contract") or "unknown"
    status = model.get("project_quota_status") or ""
    native = model.get("native_model_id")

    if operation not in CHAT_OPERATIONS:
        return ("HELD_NON_CHAT_TRANSPORT", ADAPTERS.get(operation, ADAPTERS["unknown"]), False)

    if native is None:
        return (
            "HELD_NATIVE_ID_UNRESOLVED",
            "chat.complete with no resolved native model id. A name must be observed, "
            "never derived from the display name.",
            True,
        )

    if status == "ZERO_RPD":
        return (
            "HELD_ZERO_RPD_ALLOWANCE_UNSHOWN",
            "chat.complete with a native id, but the project shows no request "
            "allowance for it. Retained because a zero observed peak is not proof "
            "of no allocation; not routable until an allowance is observed.",
            True,
        )

    if model.get("direct_contract_is_executable"):
        return ("EXECUTABLE_AND_DEPLOYED", None, True)

    return (
        "HELD_WIRE_UNQUALIFIED",
        "chat.complete with a cached native id. The transport exists but no call "
        "has proven the id, version mapping and serving behaviour on the wire.",
        True,
    )


def build(crosswalk_path: Path) -> dict:
    crosswalk = _load(crosswalk_path)
    models = crosswalk["models"]

    entries = []
    seen_native: dict[str, list[str]] = {}
    for model in models:
        state, adapter, chat_capable = _classify(model)
        quota = _quota(model)
        native = model.get("native_model_id")
        tpm = quota["tpm"]["limit"]
        # Only a chat-capable row is constrained by a per-minute budget on a
        # prompt; a TTS or embedding row's TPM bounds its own transport.
        bounded_only = (
            chat_capable and isinstance(tpm, int) and 0 < tpm <= SMALL_CONTEXT_TPM_CEILING
        )
        entry = {
            "index": model["index"],
            "display_name": model["display_name"],
            "category": model.get("ai_studio_category"),
            "operation_contract": model.get("operation_contract"),
            "native_model_id": native,
            "native_model_id_candidates": model.get("native_model_id_candidates") or [],
            "native_id_evidence": model.get("native_id_evidence"),
            "native_input_limit": model.get("cached_input_token_limit"),
            "native_output_limit": model.get("cached_output_token_limit"),
            "native_generation_methods": model.get("cached_supported_generation_methods") or [],
            "proposed_direct_contract_name": model.get("proposed_direct_contract_name"),
            "quota": quota,
            "project_quota_status": model.get("project_quota_status"),
            "state": state,
            "chat_capable": chat_capable,
            "routable_today": state == "EXECUTABLE_AND_DEPLOYED",
            "small_bounded_only": bounded_only,
            "adapter_requirement": adapter,
            "capability_route_candidates": model.get("capability_route_candidates") or [],
            "qualification_notes": model.get("qualification_notes"),
        }
        if bounded_only and chat_capable:
            entry["bounded_note"] = (
                f"per-minute input budget {tpm} tokens qualifies this model for "
                "small bounded agent tasks only; a 256K prompt cannot be served "
                "within its own TPM allowance"
            )
        entries.append(entry)
        if native:
            seen_native.setdefault(native, []).append(entry["index"])

    by_state: dict[str, int] = {}
    for entry in entries:
        by_state[entry["state"]] = by_state.get(entry["state"], 0) + 1

    shared_native = {k: v for k, v in seen_native.items() if len(v) > 1}

    features = []
    for row in crosswalk.get("feature_quotas") or []:
        features.append(
            {
                "index": row["index"],
                "display_name": row["display_name"],
                "rpd": row.get("rpd"),
                "project_feature_status": row.get("project_feature_status"),
                "is_model_direct_route": False,
                "routing_role": "HELD_TOOL_QUOTA_NOT_A_MODEL_ENDPOINT",
                "notes": row.get("notes"),
            }
        )

    inventory = {
        "schema": SCHEMA,
        "kind": "gemini-free-project-capability-inventory",
        "generated_from": {
            "crosswalk_sha256": _digest(crosswalk_path),
            "owner_confirmed_date": crosswalk.get("owner_confirmed_date"),
            "project": crosswalk.get("project"),
            "tier": crosswalk.get("tier"),
            "source_official_doc": crosswalk.get("source_official_doc"),
            "current_switchyard_release": crosswalk.get("current_switchyard_release"),
        },
        "totals": {
            "models": len(entries),
            "feature_quotas": len(features),
            "chat_capable": sum(1 for e in entries if e["chat_capable"]),
            "routable_today": sum(1 for e in entries if e["routable_today"]),
            "held": sum(1 for e in entries if not e["routable_today"]),
            "small_bounded_only": sum(1 for e in entries if e["small_bounded_only"]),
            "by_state": by_state,
        },
        "invariants": {
            "declared_models": crosswalk["totals"]["all_models"],
            "matches_crosswalk": len(entries) == crosswalk["totals"]["all_models"],
            "declared_feature_rows": crosswalk["totals"]["feature_rows"],
            "feature_rows_match": len(features) == crosswalk["totals"]["feature_rows"],
            "native_ids_shared_between_rows": shared_native,
            "one_quota_counter_per_native_model": not shared_native,
        },
        "models": entries,
        "feature_quotas": features,
    }
    return inventory


def route_graph(inventory: dict) -> dict:
    """The staged graph: primitives by capability, then the top-level composite.

    Primitives are filled in order of specificity and each one may only claim
    candidates no earlier primitive claimed. Two primitives sharing a candidate
    would give one provider-model quota two legs, so a failed request would spend
    the same daily counter twice; a primitive left with nothing is not emittable
    rather than a copy of another. The composite lists primitives only, because a
    composite child that is itself a composite is refused at load.
    """
    routable = [e for e in inventory["models"] if e["routable_today"]]
    claimed: set[str] = set()

    def claim(predicate) -> list[str]:
        taken = []
        for entry in routable:
            name = entry["proposed_direct_contract_name"]
            if not name or name in claimed:
                continue
            if predicate(entry):
                claimed.add(name)
                taken.append(name)
        return taken

    def names(state: str) -> list[str]:
        return [
            e["proposed_direct_contract_name"]
            for e in inventory["models"]
            if e["state"] == state and e["proposed_direct_contract_name"]
        ]

    bounded_pool = [
        e["proposed_direct_contract_name"]
        for e in inventory["models"]
        if e["chat_capable"] and e["small_bounded_only"]
    ]
    pending_chat = names("HELD_WIRE_UNQUALIFIED")

    bounded_today = claim(lambda e: e["small_bounded_only"])
    chat_today = claim(lambda e: True)
    long_today = claim(
        lambda e: (e["quota"]["tpm"]["limit"] or 0) >= 1_000_000
    )

    primitives = {
        "smart-gemini-bounded": {
            "purpose": (
                "small bounded agent tasks only; every member is capped by its own "
                "per-minute input budget, so this primitive never advertises a long "
                "context"
            ),
            "candidates_today": bounded_today,
            "candidates_pending_wire_qualification": bounded_pool,
            "context_window": 16384,
            "emittable": bool(bounded_today),
            "reason_not_emittable": (
                "no small-budget member is wire-qualified yet; emitting the route "
                "would claim a capability no target can serve"
            ),
        },
        "smart-gemini-chat": {
            "purpose": "general free-project chat",
            "candidates_today": chat_today,
            "candidates_pending_wire_qualification": [
                n for n in pending_chat if n not in bounded_pool
            ],
            "context_window": 1048576,
            "emittable": bool(chat_today),
            "reason_not_emittable": "no wire-qualified chat member remains unclaimed",
        },
        "smart-gemini-long-context": {
            "purpose": "prompts sized against a million-token per-minute budget",
            "candidates_today": long_today,
            "candidates_pending_wire_qualification": [],
            "context_window": 1048576,
            "emittable": bool(long_today),
            "reason_not_emittable": (
                "every remaining member was already claimed by a more specific "
                "primitive; a second leg over the same models would spend one "
                "provider-model quota twice"
            ),
        },
        "smart-gemini-vision": {
            "purpose": "image-bearing free-project requests",
            "candidates_today": [],
            "candidates_pending_wire_qualification": [],
            "context_window": 1048576,
            "emittable": False,
            "reason_not_emittable": (
                "the deployed Gemini leaves declare vision = false; the two "
                "wire-qualified Lite models are not image-qualified in the "
                "deployment, so a vision primitive would promise what they refuse"
            ),
        },
        "smart-gemini-antigravity": {
            "purpose": "managed agent runs",
            "candidates_today": [],
            "candidates_pending_wire_qualification": [],
            "context_window": None,
            "emittable": False,
            "reason_not_emittable": "agent.run has no adapter; see adapter_requirement",
        },
        "smart-gemini-live": {
            "purpose": "realtime sessions",
            "candidates_today": [],
            "candidates_pending_wire_qualification": [],
            "context_window": None,
            "emittable": False,
            "reason_not_emittable": "session.live has no adapter; see adapter_requirement",
        },
    }

    composite_children = [name for name, p in primitives.items() if p["emittable"]]
    # One child is indirection, not capability: reparenting the live composite
    # onto a single primitive would insert a layer that changes nothing a caller
    # can observe. Two is the smallest shape that can route differently.
    composite_emittable = len(composite_children) >= 2
    composite = {
        "smart_gemini": {
            "children": composite_children,
            "context_window": 1048576,
            "emittable": composite_emittable,
            "reason_not_emittable": None
            if composite_emittable
            else (
                f"only {len(composite_children)} primitive(s) have an executable "
                "candidate; a composite needs two legs before reparenting the live "
                "composite buys anything, and emitting a primitive without it would "
                "give one provider-model quota a second route"
            ),
            "note": (
                "children must be routes, never other composites; a child that a "
                "request is not cleared for is dropped before dispatch, so an "
                "uncleared request falls through instead of being sent to Google"
            ),
        }
    }

    # One provider-model quota may not be reachable through two legs, and no
    # primitive may be listed twice in the composite. Both are checked here so a
    # future edit to the grouping fails loudly instead of quietly doubling spend.
    seen: dict[str, str] = {}
    duplicates: list[dict] = []
    for name, primitive in primitives.items():
        for candidate in primitive["candidates_today"]:
            if candidate in seen:
                duplicates.append(
                    {"candidate": candidate, "primitives": [seen[candidate], name]}
                )
            else:
                seen[candidate] = name

    return {
        "schema": SCHEMA,
        "kind": "gemini-free-project-capability-route-graph",
        "quota_partition": {
            "one_primitive_per_candidate": not duplicates,
            "duplicates": duplicates,
            "unique_candidates": sorted(seen),
        },
        "primitives": primitives,
        "composite": composite,
        "non_google_fallback": {
            "chain": ["smart-nt", "luna-nt", "deepseek-flash-nt"],
            "note": (
                "the egress gate already drops a Google leg for an uncleared "
                "request before dispatch; what the graph must guarantee is that "
                "the legs after it are strict non-training targets"
            ),
        },
    }


def staged_toml(inventory: dict, graph: dict) -> str:
    """Emit only what is valid today, and document the rest as held.

    Emission is gated on the composite, not on the primitives: a primitive with
    an executable candidate that is emitted while its composite is held would be
    a second route to a quota the live deployment already reaches.
    """
    composite = graph["composite"]["smart_gemini"]
    apply_today = composite["emittable"]
    lines = [
        "# Gemini capability staging fragment -- generated, do not hand-edit.",
        f"# source crosswalk sha256: {inventory['generated_from']['crosswalk_sha256']}",
        "#",
        f"# apply today: {'yes' if apply_today else 'no'}",
    ]
    if not apply_today:
        lines.append("# " + composite["reason_not_emittable"])
    lines += [
        "#",
        "# Only routes whose every candidate is wire-qualified are emitted. Held",
        "# primitives are recorded as comments with the gate that releases them, so",
        "# nothing here promises a capability the deployment cannot serve.",
        "",
    ]

    emitted = (
        [(name, p) for name, p in graph["primitives"].items() if p["emittable"]]
        if apply_today
        else []
    )
    held = [
        (name, p)
        for name, p in graph["primitives"].items()
        if not p["emittable"] or not apply_today
    ]

    for name, primitive in emitted:
        lines.append(f"[routes.{name}]")
        lines.append('type = "passthrough"')
        lines.append(f"context_window = {primitive['context_window']}")
        lines.append("candidates = [")
        for candidate in primitive["candidates_today"]:
            lines.append(f'  "{candidate}",')
        lines.append("]")
        lines.append("")
    composite = graph["composite"]["smart_gemini"]
    children = composite["children"]
    if apply_today and children:
        lines.append("[composites.smart_gemini]")
        lines.append("children = [")
        for child in children:
            lines.append(f'  "{child}",')
        lines.append("]")
        lines.append(f"context_window = {composite['context_window']}")
        lines.append("tool_calling = true")
        lines.append("reasoning = true")
        lines.append("vision = false")
        lines.append("video = false")
    else:
        # A composite with no children is a load error, and a one-child composite
        # is indirection, so an unready graph is recorded as held rather than
        # emitted as an invalid or pointless route.
        lines.append("# HELD [composites.smart_gemini] -- "
                     + str(composite["reason_not_emittable" or "no child primitive"]))
    lines.append("")

    for name, primitive in held:
        if primitive["emittable"]:
            lines.append(
                f"# HELD [routes.{name}] -- emit gated on the composite: "
                f"{primitive['purpose']}"
            )
            lines.append("#   candidates available today:")
            for candidate in primitive["candidates_today"]:
                lines.append(f"#     {candidate}")
            lines.append("")
            continue
        lines.append(f"# HELD [routes.{name}] -- {primitive['reason_not_emittable']}")
        if primitive["candidates_pending_wire_qualification"]:
            lines.append("#   releases when wire-qualified:")
            for candidate in primitive["candidates_pending_wire_qualification"]:
                lines.append(f"#     {candidate}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def matrix(inventory: dict) -> str:
    rows = [
        "| # | model | operation | native id | RPD | state | routable |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in inventory["models"]:
        rpd = entry["quota"]["rpd"]["limit"]
        native = entry["native_model_id"] or "-- unresolved --"
        rows.append(
            "| {i} | {name} | {op} | {native} | {rpd} | {state} | {routable} |".format(
                i=entry["index"],
                name=entry["display_name"],
                op=entry["operation_contract"],
                native=native,
                rpd=rpd,
                state=entry["state"],
                routable="yes" if entry["routable_today"] else "no",
            )
        )
    totals = inventory["totals"]
    head = [
        "# Gemini free-project inventory and verified/held matrix",
        "",
        f"Generated from crosswalk `{inventory['generated_from']['crosswalk_sha256']}`",
        f"(owner-confirmed {inventory['generated_from']['owner_confirmed_date']}, "
        f"project {inventory['generated_from']['project']}, tier "
        f"{inventory['generated_from']['tier']}).",
        "",
        f"- models: {totals['models']}",
        f"- chat-capable: {totals['chat_capable']}",
        f"- routable today: {totals['routable_today']}",
        f"- held: {totals['held']}",
        f"- small-bounded-only (per-minute input budget): {totals['small_bounded_only']}",
        f"- feature quotas held as tool quota: {totals['feature_quotas']}",
        "",
        "Held is a statement about our transport and qualification, never about the",
        "owner's allocation.",
        "",
    ]
    return "\n".join(head + rows) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crosswalk", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    inventory = build(args.crosswalk)
    if not inventory["invariants"]["matches_crosswalk"]:
        print("inventory does not cover every crosswalk model", file=sys.stderr)
        return 1
    if not inventory["invariants"]["feature_rows_match"]:
        print("inventory does not cover every feature row", file=sys.stderr)
        return 1
    if inventory["invariants"]["native_ids_shared_between_rows"]:
        print(
            "two rows claim one native model id; one provider-model quota cannot be "
            "multiplied across contracts",
            file=sys.stderr,
        )
        return 1

    graph = route_graph(inventory)
    if not graph["quota_partition"]["one_primitive_per_candidate"]:
        for duplicate in graph["quota_partition"]["duplicates"]:
            print(
                "candidate {c} is reachable through {p}; one provider-model quota "
                "would be spent twice".format(
                    c=duplicate["candidate"], p=" and ".join(duplicate["primitives"])
                ),
                file=sys.stderr,
            )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "direct-google-capability-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    (args.out_dir / "capability-route-graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (args.out_dir / "gemini-capability-matrix.md").write_text(matrix(inventory))
    (args.out_dir / "gemini-r2-routes.staged.toml").write_text(staged_toml(inventory, graph))
    print(
        "models={models} routable={routable} held={held} features={features}".format(
            models=inventory["totals"]["models"],
            routable=inventory["totals"]["routable_today"],
            held=inventory["totals"]["held"],
            features=inventory["totals"]["feature_quotas"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
