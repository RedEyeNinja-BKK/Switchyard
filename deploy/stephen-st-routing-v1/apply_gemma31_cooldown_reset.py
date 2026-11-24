#!/usr/bin/env python3
"""Optional, exact one-field 31B incident candidate fix (SOURCE ONLY).

Only removes the known 300000ms override from google_gemma_31b, restoring the
runner's 5000ms default. No other client, retries, routing, or model policy may
change. Apply to a disposable candidate, NOT live CURRENT.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import tomllib


class Refused(ValueError):
    pass


def apply(text: str) -> str:
    before = tomllib.loads(text)
    clients = before.get("llm_clients") or {}
    client = clients.get("google_gemma_31b")
    if not isinstance(client, dict) or client.get("failure_cooldown_ms") != 300000:
        raise Refused("google_gemma_31b must have exact 300000ms incident override; "
                      "otherwise establish current value instead of guessing")
    lines = text.splitlines(keepends=True)
    headings = [(i, line.strip()) for i, line in enumerate(lines)
                if line.startswith("[") and line.rstrip().endswith("]")]
    spans = {h: (i, headings[n+1][0] if n+1 < len(headings) else len(lines))
             for n, (i, h) in enumerate(headings)}
    headings_for_client = [h for h in
                           ("[llm_clients.google_gemma_31b]",
                            '[llm_clients."google_gemma_31b"]') if h in spans]
    if len(headings_for_client) != 1:
        raise Refused("missing or duplicate source heading google_gemma_31b")
    start, end = spans[headings_for_client[0]]
    positions = [i for i in range(start+1, end)
                 if lines[i].partition("=")[0].strip() == "failure_cooldown_ms"]
    if len(positions) != 1 or lines[positions[0]].partition("=")[2].strip() != "300000":
        raise Refused("override line not exactly 300000, or more than one")
    del lines[positions[0]]
    out = "".join(lines)
    after = tomllib.loads(out)
    want = {key: value for key, value in client.items()
            if key != "failure_cooldown_ms"}
    for section, value in before.items():
        if section != "llm_clients" and after.get(section) != value:
            raise Refused(f"scope drift outside llm_clients: {section}")
    if set(after["llm_clients"]) != set(clients):
        raise Refused("client names changed")
    for name, value in clients.items():
        if after["llm_clients"][name] != (want if name == "google_gemma_31b" else value):
            raise Refused(f"unexpected llm client diff: {name}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.baseline.resolve() == a.out.resolve():
        print("REFUSED: cannot overwrite input", file=sys.stderr)
        return 3
    try:
        a.out.write_text(apply(a.baseline.read_text()))
    except (OSError, Refused, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    print("CANDIDATE_ONLY: removed google_gemma_31b 300000ms override; "
          "runner 5000ms default requires binary check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
