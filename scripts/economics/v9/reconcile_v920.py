#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline V9.20 reconciliation: live revision-29 export in, V9.20 out.

The base is the published economics document (the byte-exact Grafana read at
revision 29). The transform reconciles the metadata to the live routing release
and appends one dated anchor to the correction panel; it changes no expression,
because no economics instrument changed with this release.

Anchors must match exactly once, or the build refuses.
"""
import argparse
import json

RELEASE = "20261128T182648Z-6a8209f97101563b-02587e3e"
LIVE_TITLE = "SWITCHYARD ECONOMICS & TOKEN ACCOUNTING"
CONTENT_VERSION = 20
NEXT_REVISION = 30

ANCHOR = """

**2026-11-29 \u2014 release anchor.** The running release is `%s` (label `stephen-google-capability-twins-20261128`; config-only, binary `6a8209f9\u2026` unchanged from the qualified build). It adds eleven Stephen-keyed Google free capability twins \u2014 two embeddings, six Live models, music, robotics and transcription. None of them changes an economics source: provider-account reads, allowance windows and cost families are the same instruments on the same cadence, and those free-tier capability lanes carry no provider spend this build reports. What this dashboard cannot show is unchanged: per-candidate attribution, exported headroom, and exact post-anchor monetary reconciliation.""" % RELEASE

NEW_DESCRIPTION = (
    "SWITCHYARD V9.20 \u2014 reconciled to routing release `%s` (2026-11-29); no economics "
    "instrument changed with it. The V9.17 description below is retained as written, and its "
    "correction lives in full on panel 1047.\n\n" % RELEASE
)


def _flatten(panels):
    flat = {}
    def walk(items):
        for panel in items:
            flat[panel["id"]] = panel
            walk(panel.get("panels", []))
    walk(panels)
    return flat


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconcile V9 economics to the live release.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-version", type=int, required=True)
    parser.add_argument("--uid", default="turnstone-switchyard-economics-v9")
    args = parser.parse_args(argv)

    with open(args.base, encoding="utf-8") as handle:
        base = json.load(handle)
    document = json.loads(json.dumps(base["document"] if "document" in base else base))

    document["title"] = f"{LIVE_TITLE} V9.{CONTENT_VERSION}"
    document["version"] = NEXT_REVISION
    original = document.get("description") or ""
    if original.startswith("SWITCHYARD V9.20"):
        raise SystemExit("base description is already reconciled; refusing to prepend twice")
    document["description"] = NEW_DESCRIPTION + original

    flat = _flatten(document["panels"])
    panel = flat[1047]
    content = panel["options"]["content"]
    marker = "**2026-11-29 \u2014 release anchor.**"
    if marker in content:
        raise SystemExit("panel 1047 already carries the 2026-11-29 anchor; refusing to duplicate")
    panel["options"]["content"] = content.rstrip() + ANCHOR
    if panel.get("content") is not None:
        panel["content"] = panel["options"]["content"]

    if document["version"] != args.expected_version + 1:
        raise SystemExit(f"version {document['version']} != {args.expected_version + 1}")
    if document["title"] != f"{LIVE_TITLE} V9.{CONTENT_VERSION}":
        raise SystemExit("title invariant failed")

    request = {
        "op": "update",
        "uid": args.uid,
        "expected_version": args.expected_version,
        "folder_uid": None,
        "message": (
            "V9.20: reconcile metadata to release 02587e3e and append one dated release anchor "
            "to panel 1047. No expression changed; no economics instrument changed."
        ),
        "dashboard": document,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: title {document['title']} version {document['version']} "
          f"expected_version {args.expected_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
