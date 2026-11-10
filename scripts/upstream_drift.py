#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only upstream drift watch.

The fork has to know when upstream NVIDIA-NeMo/Switchyard moves, and which of
those moves touch code this fork has changed. That is all this program does: it
reads, it summarises, and it keeps one issue comment current. It does not merge,
push, rebase or deploy anything, and it never writes to either repository.

Usage (GitHub Actions supplies GITHUB_TOKEN and GITHUB_REPOSITORY):

    python scripts/upstream_drift.py --issue 35 --overlap .github/upstream-overlap-paths.txt

Exit codes: 0 when the check completed (whether or not there was news), 1 when
the check could not complete. A failed check must not look like "no drift".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MARKER_PREFIX = "<!-- switchyard-upstream-drift v1"
MARKER_SUFFIX = "-->"
# The marker is rendered on its own line at the end of the comment. It must be
# matched as a line, not as a substring: a post that merely *mentions* the marker
# -- a design note, or a report explaining this file -- contains the same text,
# and treating that post as the report would rewrite somebody else's comment.
# This happened once, to the alignment report that described this workflow.
MARKER_LINE = re.compile(rf"^{re.escape(MARKER_PREFIX)}", re.MULTILINE)
UPSTREAM = "NVIDIA-NeMo/Switchyard"
# The paths this fork has changed since the pinned common ancestor. Regenerate
# with: git diff --name-only <ancestor> origin/main | sort
DEFAULT_OVERLAP = ".github/upstream-overlap-paths.txt"
COMMENT_TAG = "switchyard-upstream-drift"
# The report stays short enough to read; the embedded state keeps the full list.
MAX_LISTED_PRS = 20
# Only the most recent merges are kept in the state: the pin advances past the
# older ones, so they can never be reported as news again.
MAX_STORED_PRS = 40


def load_overlap_paths(path: Path | str) -> set[str]:
    """Read the pinned overlap list, ignoring comments and blank lines."""
    paths: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.add(line)
    return paths


def marker_span(body: str) -> tuple[int, int] | None:
    """Return the span of the report marker and its payload, or None.

    The marker must start a line. Only the rendered report puts it there; prose
    that quotes the marker does not, so a comment about this workflow can never
    be mistaken for its output.
    """
    match = MARKER_LINE.search(body)
    if match is None:
        return None
    end = body.find(MARKER_SUFFIX, match.start())
    if end < 0:
        return None
    return match.start(), end


def parse_state(body: str) -> dict[str, Any]:
    """Recover the state embedded in a previous report, or {} when absent."""
    span = marker_span(body)
    if span is None:
        return {}
    start, end = span
    payload = body[start + len(MARKER_PREFIX) : end].strip()
    try:
        parsed = json.loads(payload)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def render_comment(state: dict[str, Any], summary: str) -> str:
    """Render the single report comment, with its state embedded for the next run."""
    sha = state.get("upstream_sha", "")
    release = state.get("release") or "none"
    prerelease = state.get("release_prerelease")
    ahead = state.get("ahead", 0)
    behind = state.get("behind", 0)
    prs = state.get("overlapping_prs") or []

    lines = [
        f"### Upstream drift watch — `{str(sha)[:8]}`",
        "",
        summary,
        "",
        "| item | value |",
        "| --- | --- |",
        f"| upstream `main` | `{sha}` |",
        f"| newest release | `{release}`{' (pre-release)' if prerelease else ''} |",
        f"| fork ahead | {ahead} |",
        f"| upstream ahead | {behind} |",
        "",
    ]
    if prs:
        lines += ["Merged upstream changes touching files this fork has changed:", ""]
        for pr in prs[:MAX_LISTED_PRS]:
            detail = f"{pr.get('overlap_count', 0)} overlapping file(s)"
            if pr.get("docs_only"):
                detail += ", docs only"
            lines.append(
                f"- #{pr.get('number')} — {pr.get('title', '')}"
                f" (merged {str(pr.get('merged_at') or '')[:10]}; {detail})"
            )
        if len(prs) > MAX_LISTED_PRS:
            lines.append(f"- … and {len(prs) - MAX_LISTED_PRS} more")
    elif state.get("baseline"):
        lines.append("Baseline recorded. Deltas will be reported from this pin onwards.")
    else:
        lines.append(
            "No merged upstream change since the pin touches a file this fork has changed."
        )
    lines += [
        "",
        "<sub>Read-only check: it never merges, rebases, pushes or deploys. "
        "This comment is updated in place, so an unchanged upstream produces no new comment.</sub>",
        "",
        f"{MARKER_PREFIX}\n{json.dumps(state, sort_keys=True)}\n{MARKER_SUFFIX}",
    ]
    return "\n".join(lines)


def overlapping(changed: Iterable[str], local: set[str]) -> list[str]:
    """Return the changed paths that this fork also changed."""
    return sorted(path for path in changed if path in local)


def is_substantive(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """True when the report should be published, not merely remembered.

    A moved upstream SHA by itself is not news: it still carries the same
    release and the same overlapping merges. A new release, or a merged pull
    request that touches a file this fork changed, is.
    """
    if not previous:
        return True
    if current.get("release") != previous.get("release"):
        return True
    reported = {pr.get("number") for pr in previous.get("overlapping_prs") or []}
    for pr in current.get("overlapping_prs") or []:
        if pr.get("number") not in reported:
            return True
    return False


class Github:
    """Minimal read-only GitHub client. It issues GET only, by construction."""

    def __init__(self, token: str, api: str = "https://api.github.com") -> None:
        self.token = token
        self.api = api.rstrip("/")

    def get(self, path: str) -> Any:
        request = urllib.request.Request(
            f"{self.api}{path}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": COMMENT_TAG,
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"null")


def collect(
    github: Any,
    *,
    upstream: str,
    fork: str,
    overlap_paths: set[str],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Read the current upstream state. No repository is modified."""
    head = github.get(f"/repos/{upstream}/commits/main")
    upstream_sha = head["sha"]

    releases = github.get(f"/repos/{upstream}/releases?per_page=10")
    newest = releases[0] if releases else {}

    # base is upstream, head is the fork: ahead_by counts fork-only commits.
    fork_owner = fork.split("/")[0]
    compare = github.get(f"/repos/{upstream}/compare/{upstream_sha}...{fork_owner}:main")

    state: dict[str, Any] = {
        "upstream_sha": upstream_sha,
        "release": newest.get("tag_name"),
        "release_prerelease": bool(newest.get("prerelease")),
        "ahead": int(compare.get("ahead_by") or 0),
        "behind": int(compare.get("behind_by") or 0),
        "overlapping_prs": [],
    }

    previous_sha = previous.get("upstream_sha")
    if not previous_sha:
        # First observation: establish the baseline. The aim is a pin future runs
        # can diff against, not a backlog of everything since the common ancestor.
        state["baseline"] = True
        return state
    if upstream_sha == previous_sha:
        # Nothing moved: neither the commit walk nor the PR fan-out is needed.
        state["overlapping_prs"] = list(previous.get("overlapping_prs") or [])
        return state

    seen: set[int] = set()
    found: list[dict[str, Any]] = []
    commits = github.get(f"/repos/{upstream}/commits?sha={upstream_sha}&per_page=100")
    for commit in commits:
        sha = commit.get("sha")
        if not sha or sha == previous_sha:
            break
        for pr in github.get(f"/repos/{upstream}/commits/{sha}/pulls"):
            number = pr.get("number")
            if number in seen or not pr.get("merged_at"):
                continue
            seen.add(number)
            files = github.get(f"/repos/{upstream}/pulls/{number}/files?per_page=100")
            paths = [item.get("filename", "") for item in files]
            touched = overlapping(paths, overlap_paths)
            if touched:
                found.append(
                    {
                        "number": number,
                        "title": pr.get("title", ""),
                        "merged_at": pr.get("merged_at"),
                        "overlap_count": len(touched),
                        # A documentation-only overlap is still reported, but named as
                        # such, so a reader can dismiss it without opening the PR.
                        "docs_only": all(
                            path.startswith("docs/") or path.endswith(".md") for path in touched
                        ),
                    }
                )
    state["overlapping_prs"] = found[:MAX_STORED_PRS]
    return state


def find_marker_comment(github: Any, issue: int) -> dict[str, Any] | None:
    """Return this bot's previous report comment, if it exists."""
    comments = github.get(f"/repos/{_repo()}/issues/{issue}/comments?per_page=100")
    for comment in comments:
        if marker_span(comment.get("body") or "") is not None:
            return comment
    return None


def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY", "")


def publish(github: Any, issue: int, body: str) -> str:
    """Create or update the single report comment. Returns the action taken."""
    existing = find_marker_comment(github, issue)
    if existing is not None:
        if existing.get("body") == body:
            return "unchanged"
        github_request("PATCH", f"/repos/{_repo()}/issues/comments/{existing['id']}", body)
        return "updated"
    github_request("POST", f"/repos/{_repo()}/issues/{issue}/comments", body)
    return "created"


def github_request(method: str, path: str, body_text: str) -> Any:
    """The only write this program performs: the issue comment itself."""
    data = json.dumps({"body": body_text}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": COMMENT_TAG,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"null")


def summarise(previous: dict[str, Any], current: dict[str, Any]) -> str:
    if not previous:
        return (
            f"Baseline established at `{str(current['upstream_sha'])[:8]}`. "
            "Future runs report releases and merged upstream changes that touch a file "
            "this fork has changed, after this pin."
        )
    if current.get("release") != previous.get("release"):
        return f"New upstream release `{current.get('release')}` (was `{previous.get('release')}`)."
    new_prs = [
        pr
        for pr in current.get("overlapping_prs") or []
        if pr.get("number") not in {p.get("number") for p in previous.get("overlapping_prs") or []}
    ]
    if new_prs:
        return f"{len(new_prs)} overlapping upstream merge(s) since the pin."
    if current.get("upstream_sha") != previous.get("upstream_sha"):
        return "Upstream moved, but no merged change touches a file this fork changed."
    return "No change since the last observation."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--overlap", default=DEFAULT_OVERLAP)
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument("--fork", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1
    if not args.fork:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 1

    github = Github(token)
    try:
        existing = find_marker_comment(github, args.issue)
        previous = parse_state((existing or {}).get("body") or "")
        current = collect(
            github,
            upstream=args.upstream,
            fork=args.fork,
            overlap_paths=load_overlap_paths(args.overlap),
            previous=previous,
        )
        body = render_comment(current, summarise(previous, current))
        if args.dry_run:
            print(body)
            return 0
        # Even an unchanged SHA is recorded, so the pin advances without a new
        # comment; the body is only patched when it actually differs.
        action = publish(github, args.issue, body)
    except (urllib.error.URLError, urllib.error.HTTPError) as error:
        print(f"upstream drift check failed: {error}", file=sys.stderr)
        return 1

    print(
        f"{action}: {json.dumps({k: current[k] for k in ('upstream_sha', 'release', 'ahead', 'behind')})}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
