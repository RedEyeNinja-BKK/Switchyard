#!/usr/bin/env python3
"""Batch F1: give `smart-nt` its authorized terminal leg and drop held 404 clutter.

Edits are line-anchored rather than re-serialised: the owner's manifest is a reviewed
document with comments that record why each leg is or is not a candidate, and rewriting it
through a TOML writer would silently delete that reasoning.

Three changes, each refused if its anchor is not found exactly once:

1. `[routes.smart-nt]` loses `direct-google-gemini-2.5-flash` (held, Google answers 404)
   and `deepseek-deepseek-flash-nt` (moves to the terminal leg).
2. `[routes.smart-nt]` gains `terminal_fallback = ["deepseek-deepseek-flash-nt"]`.
   DeepSeek is reached only when the primary pool is provably unavailable.
3. `[routes.smart_gemini]` loses `direct-google-gemini-2.5-flash`, and the stale comment
   claiming 3.6/3.8 are absent is corrected to say what the route actually lists.

Usage:
    apply_batch_f1.py INPUT OUTPUT
    apply_batch_f1.py --verify ORIGINAL OUTPUT
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tomllib
from pathlib import Path

ROUTE_NT = "[routes.smart-nt]"
ROUTE_GEM = "[routes.smart_gemini]"
TERMINAL = "deepseek-deepseek-flash-nt"
HELD_25 = "direct-google-gemini-2.5-flash"

TERMINAL_LINE = f'terminal_fallback = ["{TERMINAL}"]'


def die(message: str) -> None:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)


def block(lines: list[str], header: str) -> tuple[int, int]:
    """Return [start, end) of the TOML table headed by `header`."""
    starts = [index for index, line in enumerate(lines) if line.strip() == header]
    if len(starts) != 1:
        die(f"expected exactly one {header}, found {len(starts)}")
    start = starts[0]
    for index in range(start + 1, len(lines)):
        stripped = lines[index].lstrip()
        if stripped.startswith("[") and not stripped.startswith("#"):
            return start, index
    return start, len(lines)


def remove_candidate(lines: list[str], start: int, end: int, model: str) -> None:
    needle = f'"{model}",'
    hits = [index for index in range(start, end) if lines[index].strip() == needle]
    if len(hits) != 1:
        die(f"expected exactly one candidate line {needle} in block, found {len(hits)}")
    del lines[hits[0]]


def apply(source: str) -> str:
    lines = source.splitlines(keepends=True)

    # Refuse before mutating anything: a document that already declares the terminal leg
    # has been F1-edited (or edited against it), and re-running would duplicate the key.
    start, end = block(lines, ROUTE_NT)
    if any("terminal_fallback" in line for line in lines[start:end]):
        die("smart-nt already declares terminal_fallback; refusing to edit a moved baseline")
    start, end = block(lines, ROUTE_NT)
    remove_candidate(lines, start, end, HELD_25)
    start, end = block(lines, ROUTE_NT)
    remove_candidate(lines, start, end, TERMINAL)

    selection = [index for index in range(start, end) if lines[index].startswith("selection =")]
    if len(selection) != 1:
        die(f"expected one selection line in smart-nt, found {len(selection)}")
    lines.insert(selection[0] + 1, TERMINAL_LINE + "\n")

    start, end = block(lines, ROUTE_GEM)
    remove_candidate(lines, start, end, HELD_25)

    # The comment above smart_gemini said 3.6 and 3.8 were absent while the list included
    # them. Replace the paragraph with what the route now does.
    stale = (
        "# Complete pool, in preference order. gemini-2.5-flash-lite is absent\n"
        "# because it is advertised by /v1beta/models and 404s on generateContent\n"
        "# in this project: including it would put a known-dead leg in the\n"
        "# ordering. 3.6-flash and 3.8-flash are absent because neither has a\n"
        "# current free-tier quota qualification; one 200 does not earn pool\n"
        "# membership, and they remain catalogued as held.\n"
    )
    corrected = (
        "# Complete pool, in preference order. The 2.5 models are absent: Google\n"
        "# answers 404 for them in this project, so listing them would put a\n"
        "# known-dead leg in the ordering; their target records stay held so the\n"
        "# hold remains auditable. 3.6-flash and 3.8-flash are listed and are NOT\n"
        "# held: the previous comment claimed the opposite of what this route does.\n"
    )
    text = "".join(lines)
    if text.count(stale) != 1:
        die("smart_gemini comment paragraph not found exactly once; baseline moved")
    text = text.replace(stale, corrected, 1)
    return text


def require(condition: object, message: str) -> None:
    """A verifier that refuses must say what it refused, or a green run proves nothing."""
    if not condition:
        die(message)


def verify(original: str, output: str) -> None:
    before = tomllib.loads(original)
    after = tomllib.loads(output)

    routes = after["routes"]
    nt = routes["smart-nt"]
    candidates = nt["candidates"]
    require(TERMINAL not in candidates, "the terminal leg is still ranked as a candidate")
    require(HELD_25 not in candidates, "the held 404 model is still a candidate of smart-nt")
    require(
        nt.get("terminal_fallback") == [TERMINAL],
        f"smart-nt terminal_fallback is {nt.get('terminal_fallback')!r}, expected [{TERMINAL!r}]",
    )
    require(
        nt["target"] == before["routes"]["smart-nt"]["target"],
        "the primary target of smart-nt moved",
    )

    gem = routes["smart_gemini"]
    require(HELD_25 not in gem["candidates"], "the held 404 model is still a candidate of smart_gemini")
    require("terminal_fallback" not in gem, "a Gemini-only profile must not gain a paid leg")
    expected = [name for name in before["routes"]["smart_gemini"]["candidates"] if name != HELD_25]
    require(
        gem["candidates"] == expected,
        f"smart_gemini candidates changed unexpectedly: {gem['candidates']!r}",
    )

    # Every other route is untouched, and every target record survives.
    require(after["targets"] == before["targets"], "a target record changed")
    for name, route in before["routes"].items():
        if name in {"smart-nt", "smart_gemini"}:
            continue
        require(after["routes"][name] == route, f"an unrelated route changed: {name}")

    # The terminal target must exist and must be callable.
    terminal_target = after["targets"][TERMINAL]
    require("held" not in terminal_target, "a held target cannot be a last resort")

    # Comment preservation: the reviewed reasoning must still be there.
    for marker in (
        "[routes.smart-nt]",
        "[routes.smart_gemini]",
        TERMINAL_LINE,
        "3.6-flash and 3.8-flash are listed and are NOT",
    ):
        require(marker in output, f"reviewed comment marker lost: {marker}")

    # Idempotence, and specifically idempotence on the terminal line: a second run must
    # refuse because the leg is already declared. Accepting an earlier refusal for a
    # different reason would make this check pass on a file that was never F1-edited.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            apply(output)
    except SystemExit as exit:
        require(exit.code == 2, f"re-application exited {exit.code}")
    else:
        die("re-applying batch F1 to its own output was accepted")
    require(
        "already declares terminal_fallback" in buffer.getvalue(),
        "re-application refused for an unrelated reason: "
        + buffer.getvalue().strip().replace("REFUSED: ", ""),
    )
    print(f"PASS: {len(candidates)} primary candidates, terminal {TERMINAL}")
    print(f"PASS: smart-nt  sha256 {hashlib.sha256(output.encode()).hexdigest()}")


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--verify":
        verify(Path(argv[2]).read_text(), Path(argv[3]).read_text())
        return 0
    if len(argv) == 3:
        source = Path(argv[1]).read_text()
        Path(argv[2]).write_text(apply(source))
        print(f"wrote {argv[2]}")
        return 0
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
