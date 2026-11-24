#!/usr/bin/env python3
"""Declare the local no-thinking leaf as smart-judge's third tier.

The production route manifest is not stored in this tree — it names internal hosts. This
applier therefore reads the deployed bytes and writes a candidate; it never writes to
production.

Three objects are added, and one declaration is rewritten:

- a second client on the existing local endpoint. The loader refuses two targets that name
  one model on one client with different reasoning policies, and the whole point of the leaf
  is a different policy on the same weights, so the client is the seam;
- a target for those weights with reasoning disabled, and the capability facts the release
  gate demands;
- a 1:1 alias for the target, so the leg can be exercised directly and not only through a
  cascade;
- smart-judge's `terminal_fallback`, from one flat leg to two ordered tiers.

The applier refuses on any anchor that is missing or doubled, and then refuses again unless
the finished document differs from the baseline exactly as declared. The last check is the
one that matters for review: it asserts that the existing reasoning-enabled local target and
its alias are byte-identical to the deployed ones, so "the new leaf did not change the old
route" is enforced rather than observed.

Usage:
  python3 apply_smart_judge_three_tier.py --baseline deployed.toml --out candidate.toml \
      [--diff]
"""
from __future__ import annotations

import argparse
import difflib
import sys
import tomllib

CLIENT_ANCHOR = """[llm_clients.htpc]
format = "openai_responses"
base_url = "http://htpc-wsl.tailc8ef7b.ts.net:8889/v1"
locality = "local"
api_key_env = "SWITCHYARD_HTPC_API_KEY"
max_retries = 2

billing = "local"
"""

CLIENT_ADDED = """
# The no-thinking local leg. Same transport as `htpc`, separate client on purpose: the loader
# refuses two targets that name one model on one client with different reasoning policies, and
# the third tier exists precisely because the thinking leg's budget is the reason it cannot
# answer a strict-JSON verdict in time.
[llm_clients.htpc-nt]
format = "openai_responses"
base_url = "http://htpc-wsl.tailc8ef7b.ts.net:8889/v1"
locality = "local"
api_key_env = "SWITCHYARD_HTPC_API_KEY"
max_retries = 2
billing = "local"
# One physical machine, one account. Both HTPC clients reach the same server and the same
# weights, so the accounting unit is the machine rather than the client entry.
quota_group = "htpc"

"""

TARGET_ANCHOR = """[targets.htpc-qwen3_5-9b-mtp]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc"
vision = false
reasoning_policy = "enabled"
reasoning_dialect = "llama_cpp_enable_thinking"
"""

TARGET_ADDED = """
# The same local model asked not to think. `llama_cpp_enable_thinking` renders the refusal as
# `chat_template_kwargs.enable_thinking = false`, which is what the llama.cpp-family server
# behind this target reads. The policy is on the TARGET, not the route, because a route's
# terminal leg is resolved from target names: a route-level override could not reach it.
#
# The three capability flags are target FACTS, measured on the live deployment rather than
# copied from the sibling or from a model card; the receipt is
# `evidence/smart-judge-three-tier-20261123/htpc-nt-wire-evidence.json`. The server reports
# `modalities = {vision: false, video: false, audio: false}` AND refuses an image and a video
# request in its own words, which is what `vision = false` and `video = false` state.
# `tool_calling = true` was called, not inferred: a request with `tools` + `tool_choice =
# "auto"` and `enable_thinking = false` came back with `finish_reason = "tool_calls"`, a
# populated `tool_calls` array and `content = null`.
[targets.htpc-qwen3_5-9b-mtp-nt]
id = "Qwen3.5-9B-MTP"
llm_client = "htpc-nt"
vision = false
video = false
tool_calling = true
reasoning_policy = "disabled"
reasoning_dialect = "llama_cpp_enable_thinking"
"""

ROUTE_ANCHOR = """[routes.direct-htpc-qwen3_5-9b-mtp]
id = "direct-htpc-qwen3_5-9b-mtp"
type = "passthrough"
target = "htpc-qwen3_5-9b-mtp"
candidates = []
context_window = 81920
tool_calling = true
reasoning = true
vision = false
"""

ROUTE_ADDED = """
# The third tier's own alias, so the leg can be exercised directly instead of only through a
# whole cascade. `reasoning = false` states the contract the target enforces. The capability
# flags repeat the target's facts because the load validates route against target, and a route
# has to say explicitly what it narrows.
[routes.direct-htpc-qwen3_5-9b-mtp-nt]
id = "direct-htpc-qwen3_5-9b-mtp-nt"
type = "passthrough"
target = "htpc-qwen3_5-9b-mtp-nt"
candidates = []
context_window = 81920
tool_calling = true
reasoning = false
vision = false
video = false
"""

POLICY_ANCHOR = """terminal_fallback = ["openai-gpt-6-luna-nt"]
terminal_policy = "quota_only"
"""

POLICY_REPLACED = """terminal_fallback = [["openai-gpt-6-luna-nt"], ["htpc-qwen3_5-9b-mtp-nt"]]
terminal_policy = "availability"
"""

DOC_ANCHOR = """# The paid leg is a terminal fallback, not a candidate. `terminal_fallback` is a
# separate model set reached only through the route's terminal decision, and this
# route declares `terminal_policy = "quota_only"`: Luna NT can be spent only when
# the free pool is provably out of quota -- every dispatched capable leg failed on
# quota evidence (daily allowance exhausted, minute window closed, or a 429 on a
# free-allowance credential), or every capable leg is covered by a quota-scoped
# credential block. A 5xx, transport error, timeout, transient unavailability, a
# rejected credential or a payment fault is an outage, not an exhausted allowance,
# and never reaches the paid leg. Neither does a 400, 403, context-window refusal,
# empty answer or translation fault: those are facts about the request, and
# retrying them on a paid leg would fail again with a bill attached.
"""

DOC_REPLACED = """# Two terminal TIERS, not one flat leg, and the distinction is load-bearing. A flat
# list would make these two peers, ranked against each other by the route's selection
# policy; written as a list of lists, tier 2 is reachable only once tier 1 has itself
# been proved out. The declared order is the escalation order and no selector may
# re-decide it, so the metered remote leg is always consulted before the local one.
#
# `terminal_policy = "availability"` is what each boundary is judged by: the whole
# tier must have failed on capacity or availability -- allowance exhausted, a minute
# window closed, a 429, a retryable 5xx, a transport fault, a leg that went
# unavailable, or an attempt that timed out -- before the next tier may be spent.
# A rejected credential or a refused balance is NOT capacity: it says the deployment
# is misconfigured or unpaid, and answering it with a further call turns a
# configuration fault into cost. Neither is a 400, 403, context-window refusal, empty
# answer or translation fault, which are facts about the request and would fail again
# elsewhere with a bill attached.
#
# The order is deliberate. Tier 1 is the metered remote leg that exists so a spent
# free allowance does not become the caller's problem, and tier 2 is the local
# no-thinking leg that costs nothing to try last. A local leg never outranks a
# healthy remote one, and a healthy earlier tier is never skipped on cost grounds.
"""

EDITS = (
    ("local client", CLIENT_ANCHOR, CLIENT_ANCHOR + CLIENT_ADDED),
    ("local no-thinking target", TARGET_ANCHOR, TARGET_ANCHOR + TARGET_ADDED),
    ("direct alias route", ROUTE_ANCHOR, ROUTE_ANCHOR + ROUTE_ADDED),
    ("tiered declaration", POLICY_ANCHOR, POLICY_REPLACED),
    ("documented rule", DOC_ANCHOR, DOC_REPLACED),
)

NEW_CLIENT = "htpc-nt"
NEW_TARGET = "htpc-qwen3_5-9b-mtp-nt"
NEW_ROUTE = "direct-htpc-qwen3_5-9b-mtp-nt"
UNCHANGED_SIBLINGS = ("htpc-qwen3_5-9b-mtp", "direct-htpc-qwen3_5-9b-mtp")
UNTOUCHED_SECTIONS = ("composites", "capabilities", "decision_lanes", "quota")


class EditError(ValueError):
    pass


def apply(text: str) -> str:
    out = text
    for label, anchor, replacement in EDITS:
        found = out.count(anchor)
        if found != 1:
            raise EditError(f"{label} anchor matched {found} times, expected 1")
        out = out.replace(anchor, replacement, 1)
    assert_scope(tomllib.loads(text), tomllib.loads(out))
    return out


def assert_scope(before: dict, after: dict) -> None:
    """The candidate must differ from the baseline exactly as declared."""
    for section, added in (
        ("llm_clients", NEW_CLIENT),
        ("targets", NEW_TARGET),
        ("routes", NEW_ROUTE),
    ):
        old, new = before.get(section) or {}, after.get(section) or {}
        removed = set(old) - set(new)
        if removed:
            raise EditError(f"unexpected {section} removed: {sorted(removed)}")
        if set(new) - set(old) != {added}:
            raise EditError(
                f"{section} delta is {sorted(set(new) - set(old))}, expected only {added}"
            )

    for section in UNTOUCHED_SECTIONS:
        if before.get(section) != after.get(section):
            raise EditError(f"[{section}] changed; this edit must not touch it")

    for name in UNCHANGED_SIBLINGS:
        for section in ("targets", "routes"):
            if (before.get(section) or {}).get(name) != (after.get(section) or {}).get(name):
                raise EditError(
                    f"{section}.{name} changed; the reasoning-enabled local route must be "
                    "byte-identical to the deployed one"
                )

    for name, config in (before.get("llm_clients") or {}).items():
        if after["llm_clients"].get(name) != config:
            raise EditError(f"llm_clients.{name} changed outside the declared delta")

    for name, config in (before.get("targets") or {}).items():
        if after["targets"].get(name) != config:
            raise EditError(f"targets.{name} changed outside the declared delta")

    for name, config in (before.get("routes") or {}).items():
        want = dict(config)
        if name == "smart-judge":
            want["terminal_fallback"] = [["openai-gpt-6-luna-nt"], [NEW_TARGET]]
            want["terminal_policy"] = "availability"
        if after["routes"].get(name) != want:
            raise EditError(f"routes.{name} changed outside the declared delta")

    leaf = after["routes"][NEW_ROUTE]
    if leaf.get("type") != "passthrough" or leaf.get("candidates") != []:
        raise EditError(f"{NEW_ROUTE} must be a 1:1 passthrough route")
    for key in ("terminal_fallback", "escalation_target", "pool_groups"):
        if leaf.get(key):
            raise EditError(f"{NEW_ROUTE} must not declare {key}")

    tiers = after["routes"]["smart-judge"].get("terminal_fallback")
    if tiers != [["openai-gpt-6-luna-nt"], [NEW_TARGET]]:
        raise EditError(f"smart-judge terminal tiers are {tiers!r}, expected two ordered tiers")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="deployed manifest to read")
    parser.add_argument("--out", required=True, help="candidate path to write")
    parser.add_argument("--diff", action="store_true", help="print a unified diff")
    args = parser.parse_args(argv)

    with open(args.baseline, "rb") as handle:
        original = handle.read().decode()
    try:
        candidate = apply(original)
    except EditError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    if candidate == original:
        print("REFUSED: the edit produced no change", file=sys.stderr)
        return 2

    with open(args.out, "w") as handle:
        handle.write(candidate)

    if args.diff:
        sys.stdout.writelines(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                candidate.splitlines(keepends=True),
                fromfile=args.baseline,
                tofile=args.out,
                n=2,
            )
        )
    print(f"wrote {args.out} ({len(candidate.encode())} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
