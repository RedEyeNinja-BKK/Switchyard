# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic tests for the read-only upstream drift watch.

The watch answers four questions and nothing else: what upstream ``main`` is now,
which releases exist, how far the fork has diverged, and which newly merged
upstream pull requests touch files this fork has changed. It must never merge,
push or deploy, and it must not post a second report when the answer has not
changed.

These tests use a canned GitHub client, so nothing here reaches the network and
no rate limit can turn a green run into a red one.
"""

from pathlib import Path

from scripts import upstream_drift as drift


def test_state_marker_round_trips_through_a_comment_body():
    state = {
        "upstream_sha": "9cea42af",
        "release": "v0.4.0a1",
        "ahead": 115,
        "behind": 227,
    }

    body = drift.render_comment(state, summary="No overlapping upstream changes.")

    parsed = drift.parse_state(body)

    assert parsed["upstream_sha"] == "9cea42af"
    assert parsed["ahead"] == 115
    assert parsed["behind"] == 227


def test_a_body_without_a_marker_parses_as_no_state():
    assert drift.parse_state("Just a normal comment.") == {}


def test_state_marker_survives_an_untrusted_body_with_no_json():
    assert drift.parse_state("<!-- switchyard-upstream-drift v1\nnot json\n-->") == {}


def test_overlap_is_decided_by_path_prefix_free_equality():
    changed = [
        "crates/switchyard-server/src/sse.rs",
        "docs/README.md",
        "README.md",
        "README.md.bak",
    ]
    local = {"crates/switchyard-server/src/sse.rs", "README.md"}

    # A directory prefix is not an overlap: only the exact file counts, so a new
    # file beside a changed one is not reported as touching it.
    assert drift.overlapping(changed, local) == [
        "README.md",
        "crates/switchyard-server/src/sse.rs",
    ]


def test_an_unchanged_upstream_sha_is_not_substantive():
    previous = {"upstream_sha": "6a924d55", "release": "v0.4.0a1"}
    current = {"upstream_sha": "6a924d55", "release": "v0.4.0a1"}

    assert drift.is_substantive(previous, current) is False


def test_a_moved_upstream_sha_with_no_overlap_and_no_release_is_not_substantive():
    previous = {"upstream_sha": "6a924d55", "release": "v0.4.0a1", "behind": 226}
    current = {
        "upstream_sha": "deadbeef",
        "release": "v0.4.0a1",
        "behind": 227,
        "overlapping_prs": [],
    }

    assert drift.is_substantive(previous, current) is False


def test_a_new_release_is_substantive_even_without_overlap():
    previous = {"upstream_sha": "6a924d55", "release": "v0.4.0a1"}
    current = {"upstream_sha": "6a924d55", "release": "v0.5.0", "overlapping_prs": []}

    assert drift.is_substantive(previous, current) is True


def test_a_newly_merged_overlapping_pr_is_substantive():
    previous = {"upstream_sha": "6a924d55", "release": "v0.4.0a1", "overlapping_prs": []}
    current = {
        "upstream_sha": "9cea42af",
        "release": "v0.4.0a1",
        "overlapping_prs": [{"number": 1300, "title": "preserve Responses refusal streams"}],
    }

    assert drift.is_substantive(previous, current) is True


def test_an_overlapping_pr_that_was_already_reported_is_not_substantive_twice():
    reported = [{"number": 1300, "title": "preserve Responses refusal streams"}]
    previous = {"upstream_sha": "9cea42af", "release": "v0.4.0a1", "overlapping_prs": reported}
    current = {
        "upstream_sha": "9cea42af",
        "release": "v0.4.0a1",
        "overlapping_prs": list(reported),
    }

    assert drift.is_substantive(previous, current) is False


class FakeGithub:
    """A GitHub client with canned answers and a record of what was asked."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, path):
        self.calls.append(path)
        matches = [prefix for prefix in self.routes if path.startswith(prefix)]
        if not matches:
            raise AssertionError(f"unexpected request: {path}")
        # Longest prefix wins, so /commits/<sha>/pulls is not answered by /commits/.
        return self.routes[max(matches, key=len)]


def test_collect_reports_pin_release_divergence_and_overlapping_merges():
    github = FakeGithub(
        {
            "/repos/NVIDIA-NeMo/Switchyard/commits/main": {"sha": "9cea42af" + "0" * 32},
            "/repos/NVIDIA-NeMo/Switchyard/releases": [
                {
                    "tag_name": "v0.4.0a1",
                    "prerelease": True,
                    "published_at": "2026-10-29T22:20:21Z",
                },
                {"tag_name": "v0.3.0", "prerelease": False, "published_at": "2026-09-22T18:06:59Z"},
            ],
            "/repos/NVIDIA-NeMo/Switchyard/compare/": {
                "ahead_by": 115,
                "behind_by": 227,
                "merge_base_commit": {"sha": "16cbe593"},
            },
            "/repos/NVIDIA-NeMo/Switchyard/commits": [
                {"sha": "9cea42af", "html_url": "https://example.invalid/c"},
            ],
            "/repos/NVIDIA-NeMo/Switchyard/commits/9cea42af/pulls": [
                {
                    "number": 1300,
                    "title": "fix(translation): preserve Responses refusal streams",
                    "merged_at": "2026-11-10T22:32:38Z",
                }
            ],
            "/repos/NVIDIA-NeMo/Switchyard/pulls/1300/files": [
                {"filename": "crates/switchyard-translation/src/codecs/responses/stream.rs"}
            ],
        }
    )

    state = drift.collect(
        github,
        upstream="NVIDIA-NeMo/Switchyard",
        fork="RedEyeNinja-BKK/Switchyard",
        overlap_paths={"crates/switchyard-translation/src/codecs/responses/stream.rs"},
        previous={"upstream_sha": "6a924d55" + "0" * 32, "release": "v0.4.0a1"},
    )

    assert state["upstream_sha"] == "9cea42af" + "0" * 32
    assert state["release"] == "v0.4.0a1"
    assert state["ahead"] == 115
    assert state["behind"] == 227
    assert [pr["number"] for pr in state["overlapping_prs"]] == [1300]


def test_collect_skips_prs_whose_files_do_not_overlap_the_fork():
    github = FakeGithub(
        {
            "/repos/NVIDIA-NeMo/Switchyard/commits/main": {"sha": "abc"},
            "/repos/NVIDIA-NeMo/Switchyard/releases": [{"tag_name": "v0.4.0a1"}],
            "/repos/NVIDIA-NeMo/Switchyard/compare/": {
                "ahead_by": 1,
                "behind_by": 2,
                "merge_base_commit": {"sha": "base"},
            },
            "/repos/NVIDIA-NeMo/Switchyard/commits": [{"sha": "abc"}],
            "/repos/NVIDIA-NeMo/Switchyard/commits/abc/pulls": [
                {"number": 1, "title": "docs: unrelated"}
            ],
            "/repos/NVIDIA-NeMo/Switchyard/pulls/1/files": [{"filename": "docs/other.md"}],
        }
    )

    state = drift.collect(
        github,
        upstream="NVIDIA-NeMo/Switchyard",
        fork="RedEyeNinja-BKK/Switchyard",
        overlap_paths={"crates/switchyard-server/src/sse.rs"},
        previous={"upstream_sha": "abc"},
    )

    assert state["overlapping_prs"] == []


def test_collect_does_not_walk_commits_already_covered_by_the_pin():
    github = FakeGithub(
        {
            "/repos/NVIDIA-NeMo/Switchyard/commits/main": {"sha": "same"},
            "/repos/NVIDIA-NeMo/Switchyard/releases": [{"tag_name": "v0.4.0a1"}],
            "/repos/NVIDIA-NeMo/Switchyard/compare/": {
                "ahead_by": 1,
                "behind_by": 2,
                "merge_base_commit": {"sha": "base"},
            },
        }
    )

    state = drift.collect(
        github,
        upstream="NVIDIA-NeMo/Switchyard",
        fork="RedEyeNinja-BKK/Switchyard",
        overlap_paths=set(),
        previous={"upstream_sha": "same"},
    )

    assert state["overlapping_prs"] == []
    assert not any("/commits/" in call and call.endswith("/pulls") for call in github.calls)


def test_comment_body_names_the_pin_and_stays_free_of_claims_it_cannot_support():
    state = {
        "upstream_sha": "9cea42af",
        "release": "v0.4.0a1",
        "release_prerelease": True,
        "ahead": 115,
        "behind": 227,
        "overlapping_prs": [
            {
                "number": 1300,
                "title": "fix(translation): preserve Responses refusal streams",
                "merged_at": "2026-11-10T22:32:38Z",
            }
        ],
    }

    body = drift.render_comment(state, summary="1 overlapping upstream merge since the pin.")

    assert "9cea42af" in body
    assert "v0.4.0a1" in body
    assert "#1300" in body
    assert "| fork ahead | 115 |" in body
    assert "| upstream ahead | 227 |" in body
    assert "no merge" in body.lower() or "read-only" in body.lower()


def test_overlap_path_file_is_read_defensively(tmp_path: Path):
    path = tmp_path / "paths.txt"
    path.write_text("# comment\n\ncrates/a.rs\ncrates/b.rs\n", encoding="utf-8")

    assert drift.load_overlap_paths(path) == {"crates/a.rs", "crates/b.rs"}


def test_a_documentation_only_overlap_is_labelled_and_a_long_list_is_capped():
    state = {
        "upstream_sha": "abc",
        "release": "v0.4.0a1",
        "ahead": 1,
        "behind": 2,
        "overlapping_prs": [
            {
                "number": n,
                "title": "t",
                "merged_at": "2026-11-10T00:00:00Z",
                "overlap_count": 1,
                "docs_only": True,
            }
            for n in range(1, 26)
        ],
    }

    body = drift.render_comment(state, "25 overlapping upstream merges since the pin.")

    assert "docs only" in body
    assert "#20" in body
    assert "#21" not in body
    assert "… and 5 more" in body


def test_the_first_run_establishes_a_pin_without_replaying_the_backlog():
    github = FakeGithub(
        {
            "/repos/NVIDIA-NeMo/Switchyard/commits/main": {"sha": "head"},
            "/repos/NVIDIA-NeMo/Switchyard/releases": [{"tag_name": "v0.4.0a1"}],
            "/repos/NVIDIA-NeMo/Switchyard/compare/": {
                "ahead_by": 1,
                "behind_by": 2,
                "merge_base_commit": {"sha": "base"},
            },
        }
    )

    state = drift.collect(
        github,
        upstream="NVIDIA-NeMo/Switchyard",
        fork="RedEyeNinja-BKK/Switchyard",
        overlap_paths={"crates/switchyard-server/src/sse.rs"},
        previous={},
    )

    assert state["overlapping_prs"] == []
    assert not any(call.endswith("/pulls") for call in github.calls)
    assert "Baseline established" in drift.summarise({}, state)


def test_publish_keeps_one_comment_and_does_not_rewrite_an_identical_body(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "RedEyeNinja-BKK/Switchyard")
    writes = []

    def record(method, path, body):
        writes.append((method, path))
        return {}

    monkeypatch.setattr(drift, "github_request", record)
    body = drift.render_comment({"upstream_sha": "abc"}, "No change since the last observation.")

    same = FakeGithub(
        {"/repos/RedEyeNinja-BKK/Switchyard/issues/35/comments": [{"id": 11, "body": body}]}
    )
    assert drift.publish(same, 35, body) == "unchanged"
    assert writes == []

    changed = FakeGithub(
        {"/repos/RedEyeNinja-BKK/Switchyard/issues/35/comments": [{"id": 11, "body": body}]}
    )
    assert drift.publish(changed, 35, body + "\\n") == "updated"
    assert writes == [("PATCH", "/repos/RedEyeNinja-BKK/Switchyard/issues/comments/11")]

    fresh = FakeGithub({"/repos/RedEyeNinja-BKK/Switchyard/issues/35/comments": []})
    assert drift.publish(fresh, 35, body) == "created"
    assert writes[-1] == ("POST", "/repos/RedEyeNinja-BKK/Switchyard/issues/35/comments")


def test_the_baseline_report_does_not_claim_a_delta_it_has_not_measured():
    state = {"upstream_sha": "abc", "baseline": True, "overlapping_prs": []}

    body = drift.render_comment(state, drift.summarise({}, state))

    assert "Baseline recorded" in body
    assert "No merged upstream change since the pin" not in body
