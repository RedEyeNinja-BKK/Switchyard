# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the routing dashboard's back-link to the economics dashboard.

WHY THIS EXISTS
  The economics dashboard links to the routing dashboard so an operator can move
  from usage and cost back to routing and serving. The reverse link makes the
  pairing discoverable from the dashboard an operator is already looking at.

  The routing dashboard is operator-authored and is read-only to this tooling's
  write path in one respect: an ``update`` needs the dashboard's current
  ``expected_version``, and a wrong one is refused. So the version is read live,
  and the document is written to a file for a human to review before anything is
  published. Nothing here writes to Grafana.

  A full copy of the current routing dashboard is preserved next to the prepared
  file, so an update that goes wrong can be rolled back by writing the copy
  back.

WHAT IT CHANGES
  One dashboard-level link, and nothing else. The panels are read from the live
  document and written back unchanged; the script compares them and refuses to
  write the request if they differ.
"""

import argparse
import json
import subprocess
import sys

import build_economics_dashboard as economics


def read_dashboard(launcher, uid):
    """Reads one dashboard raw, preserving the panel geometry and options.

    The projection form of the read discards panel geometry and options, and
    writing that back would destroy the dashboard. So the raw document is used.
    """
    proc = subprocess.run(
        ["sudo", "-n", launcher, "dashboard"],
        input=json.dumps({"uid": uid, "raw": True}).encode(),
        capture_output=True,
        timeout=90,
    )
    if proc.returncode != 0:
        raise SystemExit(f"read failed: {proc.stderr.decode() or proc.stdout.decode()}")
    return json.loads(proc.stdout.decode())


def back_link(economics_uid, economics_title):
    """Builds the link the routing dashboard gains."""
    return {
        "asDropdown": False,
        "icon": "external link",
        "includeVars": False,
        "keepTime": True,
        "tags": [],
        "targetBlank": False,
        "title": economics_title,
        "tooltip": "Provider, route and cost-class attribution, token accounting and "
        "provider account state.",
        "type": "link",
        "url": f"/d/{economics_uid}",
    }


def prepare(document, version, routing_uid, link):
    """Returns the update request, and the panels as they were.

    The panels are returned so the caller can check that they came through
    unchanged.
    """
    panels_before = document.get("panels")

    links = list(document.get("links") or [])
    if not any(existing.get("url") == link["url"] for existing in links):
        links.append(link)
    document["links"] = links

    request = {
        "op": "update",
        "uid": routing_uid,
        "expected_version": version,
        "message": f"Link to {link['title']}",
        "dashboard": document,
    }
    return request, panels_before


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare the routing dashboard's back-link request."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="The economics dashboard configuration, which names both dashboards.",
    )
    parser.add_argument(
        "--launcher",
        required=True,
        help="The publishing tool's launcher, invoked with the 'dashboard' verb.",
    )
    parser.add_argument("--out", required=True, help="Where to write the update request.")
    parser.add_argument(
        "--backup",
        required=True,
        help="Where to write the copy of the live document, for rollback.",
    )
    args = parser.parse_args(argv)

    config = economics.load_config(args.config)
    routing_uid = config["routing_uid"]

    body = read_dashboard(args.launcher, routing_uid)
    document = body.get("document", body)
    version = body.get("meta", {}).get("version", document.get("version"))
    if version is None:
        raise SystemExit("could not determine the routing dashboard's current version")

    # A copy of what is live right now, so an update can be undone.
    with open(args.backup, "w", encoding="utf-8") as handle:
        json.dump({"document": document, "version": version}, handle, indent=1)

    link = back_link(config["uid"], config["title"])
    request, panels_before = prepare(document, version, routing_uid, link)

    if request["dashboard"].get("panels") != panels_before:
        raise SystemExit("the panels changed; refusing to prepare the request")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1)

    print(f"current routing dashboard version: {version}")
    print(f"backup of the live document: {args.backup}")
    print(f"prepared update request: {args.out}")
    print(f"panels unchanged: {len(panels_before or [])}")
    print(f"links after the update: {len(request['dashboard']['links'])}")
    print("NOTHING WAS WRITTEN TO GRAFANA. Review the file, then publish it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
