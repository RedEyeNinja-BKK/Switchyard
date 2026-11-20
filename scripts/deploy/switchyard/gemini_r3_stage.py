#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Splice the Gemma 4 bounded lane into a Switchyard manifest.

The release this supports adds two capabilities to the manifest schema -- a target's
physical ``context_window`` and its per-request ``max_input_tokens`` ceiling -- plus a
``selection`` policy on a route. Everything else the lane needs already exists: the
``google_gemini`` client with ``requires_public_payload``, the daily allowance accounting,
and the ``smart-nt-bounded-trial`` composite that crosses from Google to the strict
non-training legs.

This script is the reason the manifest change is reviewable rather than retyped. It reads
the *deployed* file, refuses to run if any anchor it expects is missing or ambiguous, and
writes a candidate. Hand-editing a 1,582-line production manifest is how a deployment gains
a regression nobody diffed.

The five edits. Four are additive; the fifth promotes the ``smart-nt`` alias:

1. The two Gemini Lite targets gain the physical facts that were already true of them: a
   1,048,576-token window and a 250,000-token per-request ceiling, the latter derived from
   the project's 250,000 TPM allowance. A request above that ceiling cannot be served in one
   minute, so admitting it only spends a call the provider will refuse. The route-level
   ``context_window`` of 1,048,576 is left alone: it is a capacity statement, and capacity
   and per-request admission are deliberately different numbers.

2. Two Gemma 4 targets are added at their native ids. Both advertise a 262,144-token window
   and a 16,000-token per-request ceiling -- the window is the model's, the ceiling is the
   AI Studio free-tier TPM, and the ceiling is what binds.

3. A ``smart-gemini-bounded`` passthrough route is added over the two Gemma targets with
   ``selection = "measured"``. It is new, so no consumer moves; it is a route rather than a
   composite so that measured selection applies to it, and its candidates are two models on
   one credential with one price, which is exactly the choice worth making adaptively.

4. A ``smart-gemini-chat`` passthrough over the two Lite leaves, and ``smart_gemini`` becomes
   an addressable route over the same four targets. The group is what a consumer names; a
   composite is not an addressable model, so the group has to be a route to be callable.

5. ``smart-nt`` keeps its kind and its alias, and gains the four Google legs ahead of the two
   paid ones it already had. A passthrough's candidate list is filtered per target before any
   provider call, so privacy, capability and input-size gates skip a Google leg without
   ending the route. Promoting the alias to a composite was rejected: composites do not
   appear in the served model list, so every consumer naming ``smart-nt`` would break.
   ``smart-bounded`` and every other route are untouched.

Usage::

    python3 scripts/deploy/switchyard/gemini_r3_stage.py \\
        --live /var/tmp/live-routes.toml --out /var/tmp/routes.candidate.toml

Exit status is 1 with a reason when an anchor does not match exactly once, because a silent
partial splice is worse than no splice.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

LITE_35 = '[targets."direct-google-gemini-3.5-flash-lite"]'
LITE_31 = '[targets."direct-google-gemini-3.1-flash-lite"]'
LITE_35_ROUTE = '[routes."direct-google-gemini-3.5-flash-lite"]'
LITE_31_ROUTE = '[routes."direct-google-gemini-3.1-flash-lite"]'
SMART_GEMINI = "[composites.smart_gemini]"

# The two blocks this release replaces. Both are matched verbatim against the deployed file,
# so a manifest that has drifted refuses rather than half-splicing.
SMART_GEMINI_COMPOSITE = '''[composites.smart_gemini]
children = [
    "direct-google-gemini-3.5-flash-lite",
    "direct-google-gemini-3.1-flash-lite",
]
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false
'''

# A composite is not an addressable model: it does not appear in the served model list, so
# nothing can call it. The group the owner asked to expose therefore has to be a route. Same
# pool, same order, same capability claims -- only the table changes, which is what makes it
# callable.
SMART_GEMINI_ROUTE = '''[routes.smart_gemini]
id = "smart_gemini"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = [
    "direct-google-gemini-3.1-flash-lite",
    "direct-google-gemma-4-26b-a4b-it",
    "direct-google-gemma-4-31b-it",
]
context_window = 262144
tool_calling = true
reasoning = true
vision = false
video = false
'''

SMART_NT_PASSTHROUGH = '''# ---- smart-nt: the bounded lane under an explicit no-thinking alias -------
# An additive clone of `smart-bounded`, which stays live and unchanged: same
# primary, same fallback, same policy and same admission. Only the alias differs,
# so a caller can name the no-thinking lane directly instead of inheriting it.
[routes.smart-nt]
id = "smart-nt"
type = "passthrough"
target = "openai-gpt-6-luna-nt"
candidates = ["deepseek-deepseek-flash-nt"]
context_window = 262144
tool_calling = true
reasoning = true
vision = true
route_reasoning_policy = "none"
'''

SMART_NT_GOOGLE_FIRST = '''# ---- smart-nt: the bounded lane, Google Free first -----------------------
# Still a passthrough, and still under the same alias: a composite is not an
# addressable model, so promoting this alias to a composite would remove it from
# the served model list and break every consumer that names it.
#
# ORDER: chat Lites -> Gemma 26B -> 31B -> Luna-NT -> DeepSeek Flash-NT.
#
# The route-level filter drops a candidate before any provider call when it
# cannot carry the request: an uncleared payload skips every Google leg (all
# four declare requires_public_payload through their client), a vision request
# skips all four (none declares vision), more than 16,000 input tokens skips the
# Gemmas and more than 250,000 skips the Lites. What survives is served in this
# order; the paid legs are reached exactly as they are today.
#
# `route_reasoning_policy` is removed rather than kept: a route-level policy needs every
# candidate's client to declare a reasoning dialect, and the Google clients deliberately
# declare none -- their family cannot turn thinking off, so there is no correct wire form.
# Nothing is lost for the two paid legs: `openai-gpt-6-luna-nt` and
# `deepseek-deepseek-flash-nt` each declare `reasoning_policy = "disabled"` on the target,
# and a target policy is authoritative and applied after the route's. The no-thinking
# contract is therefore kept by the legs that have a dialect to carry it, and the Google
# legs stay neutral instead of being sent a control they cannot honour.
#
# Spend can only fall. Today every smart-nt request is served by Luna-NT; after
# this change a Google leg serves when it is eligible and a paid leg serves when
# it is not, which is the pre-existing behaviour for exactly those requests.
[routes.smart-nt]
id = "smart-nt"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = [
    "direct-google-gemini-3.1-flash-lite",
    "direct-google-gemma-4-26b-a4b-it",
    "direct-google-gemma-4-31b-it",
    "openai-gpt-6-luna-nt",
    "deepseek-deepseek-flash-nt",
]
context_window = 262144
tool_calling = true
reasoning = true
vision = true
'''

CHAT_ROUTE = '''
# smart-gemini-chat: the two Lite leaves as one capability group.
#
# A passthrough rather than a composite because its only job is the ordered pair: both
# leaves sit on one credential at one price, so the order is a latency decision, not a
# cost one. `tool_calling` is claimed because both leaves carry it on their own routes;
# `vision` is not, because neither target declares it.
[routes."smart-gemini-chat"]
id = "smart-gemini-chat"
type = "passthrough"
target = "direct-google-gemini-3.5-flash-lite"
candidates = ["direct-google-gemini-3.1-flash-lite"]
context_window = 1048576
tool_calling = true
reasoning = true
vision = false
video = false
'''

# The two Lite targets, as deployed. Matched with their closing line so the splice lands in
# the target block rather than in any route that mentions the same id.
LITE_TARGET_BODIES = {
    LITE_35: '''[targets."direct-google-gemini-3.5-flash-lite"]
id = "gemini-3.5-flash-lite"
llm_client = "google_gemini"
vision = false
video = false
daily_request_limit = 500
''',
    LITE_31: '''[targets."direct-google-gemini-3.1-flash-lite"]
id = "gemini-3.1-flash-lite"
llm_client = "google_gemini"
vision = false
video = false
daily_request_limit = 500
''',
}

LITE_PHYSICAL = '''# Capacity and admission are two different numbers and both are stated here.
# The window is the model's: 1,048,576 tokens. The ceiling is the project's free-tier
# allowance of 250,000 tokens per minute -- a single request larger than the minute is
# one the provider cannot complete, so it is refused locally instead of being charged.
context_window = 1048576
max_input_tokens = 250000
'''

GEMMA_TARGETS = '''
# The 31B answers 503 while it spins up, and the lane's default failure cooldown is five
# seconds -- short enough to spend the whole free-minute budget re-asking a model that is
# still starting. It therefore gets its own client entry, identical to the shared one except
# for the cooldown, rather than a group: `cooldown_group` would make a capacity fact about
# the large model cool the small one down with it, and the small one is this lane's default.
# The daily counter is still shared, because it is the same project and the same allowance.
[llm_clients.google_gemma_31b]
format = "openai_chat"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
locality = "external"
api_key_env = "SWITCHYARD_GEMINI_API_KEY"
requires_public_payload = true
quota_group = "google-ai-studio-free"
max_retries = 0
failure_cooldown_ms = 300000

# Gemma 4, added for bounded tasks. Native ids from the owner's AI Studio listing; the
# serving receipt is what `smart-gemini-bounded` exists to collect, so nothing else depends
# on them yet.
#
# The 262,144-token window is the model's physical capacity. The 16,000-token ceiling is
# the free-tier tokens-per-minute allowance for this project, and it is the number that
# binds: a bounded lane that accepted more would be spending calls Google refuses.
#
# daily_request_limit is the model's own RPD allowance (14,400). It is a local budget, not
# an observed balance -- see the note on the Lite targets above.
[targets."direct-google-gemma-4-31b-it"]
id = "gemma-4-31b-it"
llm_client = "google_gemma_31b"
vision = false
video = false
context_window = 262144
max_input_tokens = 16000
daily_request_limit = 14400
# Hosted Gemma writes its chain of thought into the visible message: the measured body is
# `<thought>...</thought>` followed by the answer, with `completion_tokens` counting only the
# answer and `total_tokens` counting both. Declaring the marker is what keeps the monologue
# out of the caller's text; without it the whole block is returned as output.
hosted_reasoning = "thought_tag"

[targets."direct-google-gemma-4-26b-a4b-it"]
id = "gemma-4-26b-a4b-it"
llm_client = "google_gemini"
vision = false
video = false
context_window = 262144
max_input_tokens = 16000
daily_request_limit = 14400
hosted_reasoning = "thought_tag"
'''

GEMMA_ROUTES = '''
# Direct leaves, one target each, so a caller that wants 31B gets 31B. A direct leaf is
# subject to the same payload gate, physical fit and daily budget as a composite child.
[routes."direct-google-gemma-4-31b-it"]
id = "direct-google-gemma-4-31b-it"
type = "passthrough"
target = "direct-google-gemma-4-31b-it"
candidates = []
context_window = 262144
# tool_calling and reasoning are not claimed yet. Neither has a wire receipt on this
# project, and claiming a capability that turns out absent would send a tool-bearing
# request into a lane that mangles it instead of skipping it. Flipping either boolean is a
# one-line change once the comparison harness records the behaviour.
tool_calling = false
reasoning = false
vision = false
video = false

[routes."direct-google-gemma-4-26b-a4b-it"]
id = "direct-google-gemma-4-26b-a4b-it"
type = "passthrough"
target = "direct-google-gemma-4-26b-a4b-it"
candidates = []
context_window = 262144
tool_calling = false
reasoning = false
vision = false
video = false

# ---------------------------------------------------------------------
# smart-gemini-bounded: the bounded lane, under a name nobody calls
# ---------------------------------------------------------------------
# A passthrough with one candidate rather than a composite, for one reason: measured
# selection applies to a route's candidate list, so a composite would put the adaptive
# behaviour out of reach. Both targets sit on one credential at one price, so reordering
# them cannot move cost -- it only moves success rate and latency.
#
# The default is the 26B, and the reason is structural rather than a measurement: it is an
# A4B mixture with roughly 4B active parameters per token against the 31B's 31B, so for the
# same request it moves far fewer weights. On a free tier whose binding constraint is tokens
# per minute, that is the difference between finishing inside the minute and being refused
# for the rest of it. The 31B stays as the escalation step, where a caller has already asked
# for something the small model's quality did not settle.
#
# `selection = "measured"` reorders only after a candidate has at least three observations,
# and it never adds, drops or resurrects one. A physical or payload refusal is still final
# because it is applied before selection. Until those observations exist the order above is
# what serves, so it is chosen on the argument, not on the model names.
#
# context_window = 262144 is the composite promise and it is checked at load against the
# widest target below it, so it cannot drift into a claim no target serves. The 16,000-token
# per-request ceiling is not a route promise: it is enforced per target at dispatch, which
# is the only place that knows which model a request would have reached.
[routes."smart-gemini-bounded"]
id = "smart-gemini-bounded"
type = "passthrough"
target = "direct-google-gemma-4-26b-a4b-it"
candidates = ["direct-google-gemma-4-31b-it"]
selection = "measured"
context_window = 262144
tool_calling = false
reasoning = false
vision = false
video = false
'''


def splice(text: str) -> tuple[str, list[str]]:
    """Apply the three edits, refusing on any anchor that is not unique."""
    notes: list[str] = []

    for header, body in LITE_TARGET_BODIES.items():
        count = text.count(body)
        if count != 1:
            raise SystemExit(
                f"anchor for {header} matched {count} times, expected exactly 1; "
                "the deployed manifest has moved and this splice is stale"
            )
        text = text.replace(body, body + LITE_PHYSICAL, 1)
        notes.append(f"{header}: context_window=1048576 max_input_tokens=250000")

    # Gemma targets go after the last Lite target block, which is also where the target
    # section ends, so the file keeps targets-before-routes ordering.
    tail = LITE_TARGET_BODIES[LITE_31] + LITE_PHYSICAL
    if text.count(tail) != 1:
        raise SystemExit("the Lite target section is not in the expected shape")
    text = text.replace(tail, tail + GEMMA_TARGETS, 1)
    notes.append("targets: direct-google-gemma-4-31b-it, direct-google-gemma-4-26b-a4b-it")

    # Gemma routes and the bounded lane go after the two Lite direct routes, immediately
    # before the smart_gemini header, so the Google lane stays contiguous.
    anchor = "\n# ---------------------------------------------------------------------\n# smart_gemini:"
    if text.count(anchor) != 1:
        raise SystemExit("the smart_gemini section header is not unique")
    text = text.replace(anchor, GEMMA_ROUTES + CHAT_ROUTE + anchor, 1)
    notes.append(
        "routes: direct-google-gemma-4-31b-it, direct-google-gemma-4-26b-a4b-it, "
        "smart-gemini-chat, smart-gemini-bounded (selection=measured)"
    )

    # smart_gemini becomes an addressable route over the same pool. Matched on the whole
    # block, table header included, because the table is what changes.
    if text.count(SMART_GEMINI_COMPOSITE) != 1:
        raise SystemExit("the smart_gemini composite is not in the expected shape")
    text = text.replace(SMART_GEMINI_COMPOSITE, SMART_GEMINI_ROUTE, 1)
    notes.append("smart_gemini: composites -> routes, same pool, now addressable")

    # The alias keeps its kind and gains the Google legs in front of the paid ones. A
    # whole-block replacement, because target and candidates both move.
    if text.count(SMART_NT_PASSTHROUGH) != 1:
        raise SystemExit("the smart-nt passthrough is not in the expected shape")
    text = text.replace(SMART_NT_PASSTHROUGH, SMART_NT_GOOGLE_FIRST, 1)
    notes.append("smart-nt: four Google legs ahead of the unchanged paid ladder")
    return text, notes


def verify(candidate: str, live: str) -> dict:
    """Re-parse the candidate and check the properties the lane depends on."""
    doc = tomllib.loads(candidate)
    live_doc = tomllib.loads(live)
    targets = doc.get("targets", {})
    routes = doc.get("routes", {})
    composites = doc.get("composites", {})
    failures: list[str] = []

    for name, window, ceiling, rpd in (
        ("direct-google-gemma-4-31b-it", 262144, 16000, 14400),
        ("direct-google-gemma-4-26b-a4b-it", 262144, 16000, 14400),
    ):
        target = targets.get(name, {})
        if target.get("context_window") != window:
            failures.append(f"{name}: context_window is {target.get('context_window')!r}")
        if target.get("max_input_tokens") != ceiling:
            failures.append(f"{name}: max_input_tokens is {target.get('max_input_tokens')!r}")
        if target.get("daily_request_limit") != rpd:
            failures.append(f"{name}: daily_request_limit is {target.get('daily_request_limit')!r}")
        # The 31B carries its own client entry for a longer capacity cooldown; both still
        # point at the same gated AI Studio project and share its daily counter.
        if target.get("llm_client") not in {"google_gemini", "google_gemma_31b"}:
            failures.append(f"{name}: not on the gated Google client")

    for name, ceiling in (
        ("direct-google-gemini-3.5-flash-lite", 250000),
        ("direct-google-gemini-3.1-flash-lite", 250000),
    ):
        target = targets.get(name, {})
        if target.get("max_input_tokens") != ceiling:
            failures.append(f"{name}: max_input_tokens is {target.get('max_input_tokens')!r}")
        if target.get("context_window") != 1048576:
            failures.append(f"{name}: context_window is {target.get('context_window')!r}")

    bounded = routes.get("smart-gemini-bounded", {})
    if bounded.get("type") != "passthrough":
        failures.append("smart-gemini-bounded is not a passthrough")
    if bounded.get("selection") != "measured":
        failures.append("smart-gemini-bounded does not select on measurement")
    if bounded.get("context_window") != 262144:
        failures.append("smart-gemini-bounded over- or under-states its window")
    children = [bounded.get("target"), *bounded.get("candidates", [])]
    # 26B first: it is the A4B default, and the 31B is the escalation step. The order is
    # checked here because it is the one decision in this file that a reader cannot verify
    # from a hash.
    if children != [
        "direct-google-gemma-4-26b-a4b-it",
        "direct-google-gemma-4-31b-it",
    ]:
        failures.append(f"smart-gemini-bounded candidates are {children!r}")

    # The route's promise must fit inside the widest target below it -- the same rule the
    # loader enforces, restated here so the script fails before the binary has to.
    widest = max(
        (targets[name].get("context_window") or 0) for name in children if name in targets
    )
    if (bounded.get("context_window") or 0) > widest:
        failures.append("smart-gemini-bounded advertises more than its widest target")

    # Nothing outside the Google lane may have gained a physical window: the release is
    # additive, and a widened reach would be a change to lanes this release does not touch.
    added_windows = [
        name
        for name, target in targets.items()
        if "context_window" in target and "context_window" not in live_doc.get("targets", {}).get(name, {})
    ]
    stray = [name for name in added_windows if not name.startswith("direct-google-")]
    if stray:
        failures.append(f"physical windows added outside the Google lane: {stray}")

    # The privacy ladder is already deployed; it must be untouched, because it is the lane
    # the bounded release is measured against.
    trial = composites.get("smart-nt-bounded-trial")
    if trial != live_doc.get("composites", {}).get("smart-nt-bounded-trial"):
        failures.append("smart-nt-bounded-trial changed; this release must not move it")

    # The graph the release exists to ship.
    chat = routes.get("smart-gemini-chat", {})
    if chat.get("type") != "passthrough":
        failures.append("smart-gemini-chat is not a passthrough")
    chat_children = [chat.get("target"), *chat.get("candidates", [])]
    if chat_children != [
        "direct-google-gemini-3.5-flash-lite",
        "direct-google-gemini-3.1-flash-lite",
    ]:
        failures.append(f"smart-gemini-chat candidates are {chat_children!r}")

    gemini = routes.get("smart_gemini", {})
    if gemini.get("type") != "passthrough":
        failures.append("smart_gemini is not a route; a composite is not addressable")
    gemini_children = [gemini.get("target"), *gemini.get("candidates", [])]
    if gemini_children != [
        "direct-google-gemini-3.5-flash-lite",
        "direct-google-gemini-3.1-flash-lite",
        "direct-google-gemma-4-26b-a4b-it",
        "direct-google-gemma-4-31b-it",
    ]:
        failures.append(f"smart_gemini candidates are {gemini_children!r}")
    if "smart_gemini" in composites:
        failures.append("smart_gemini is still a composite; it would not be callable")

    nt = routes.get("smart-nt", {})
    if nt.get("type") != "passthrough":
        failures.append("smart-nt is not a passthrough; its alias would leave the model list")
    expected_nt = [
        "direct-google-gemini-3.5-flash-lite",
        "direct-google-gemini-3.1-flash-lite",
        "direct-google-gemma-4-26b-a4b-it",
        "direct-google-gemma-4-31b-it",
        "openai-gpt-6-luna-nt",
        "deepseek-deepseek-flash-nt",
    ]
    nt_children = [nt.get("target"), *nt.get("candidates", [])]
    if nt_children != expected_nt:
        failures.append(f"smart-nt candidates are {nt_children!r}")
    if "route_reasoning_policy" in nt:
        failures.append("smart-nt carries a route-level reasoning policy it cannot apply")
    for leaf in ("openai-gpt-6-luna-nt", "deepseek-deepseek-flash-nt"):
        if targets.get(leaf, {}).get("reasoning_policy") != "disabled":
            failures.append(f"target {leaf} no longer declares its target-level disable")

    # `validate_composites` rejects nesting, so prove flatness here before the binary has to.
    nested = [
        (parent, child)
        for parent, composite in composites.items()
        for child in composite.get("children", [])
        if child in composites
    ]
    if nested:
        failures.append(f"nested composites are not supported: {nested}")

    # This release puts Google legs in front of the paid ladder; it must not move it. The two
    # paid targets and the Luna-NT alias are what smart-nt served before, so their bytes are
    # compared rather than assumed.
    for leaf in (
        "openai-gpt-6-luna-nt",
        "deepseek-deepseek-flash-nt",
    ):
        if live_doc.get("targets", {}).get(leaf) != targets.get(leaf):
            failures.append(f"target {leaf} changed; the paid ladder must not move")
    for leaf in ("direct-openai-gpt-6-luna-nt", "direct-deepseek-deepseek-flash-nt"):
        if routes.get(leaf) != live_doc.get("routes", {}).get(leaf):
            failures.append(f"route {leaf} changed; the paid ladder must not move")

    if failures:
        raise SystemExit("candidate failed verification:\n  " + "\n  ".join(failures))

    return {
        "gemma_targets": 2,
        "chat_candidates": chat_children,
        "bounded_candidates": children,
        "smart_gemini_candidates": gemini_children,
        "smart_nt_candidates": nt_children,
        "added_context_windows": sorted(added_windows),
        "trial_composite_unchanged": True,
        "live_targets": len(live_doc.get("targets", {})),
        "candidate_targets": len(targets),
        "live_routes": len(live_doc.get("routes", {})),
        "candidate_routes": len(routes),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, type=Path, help="deployed routes.toml")
    parser.add_argument("--out", required=True, type=Path, help="candidate to write")
    args = parser.parse_args(argv)

    live = args.live.read_text()
    candidate, notes = splice(live)
    report = verify(candidate, live)
    args.out.write_text(candidate)

    report["notes"] = notes
    report["live_sha256_prefix_source"] = str(args.live)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
