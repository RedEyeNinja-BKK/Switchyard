<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Economics dashboard

The Switchyard economics dashboard, its generator and the tool that prepares it
for publication.

Everything here reads from Switchyard itself. The provider account figures are
collected by Switchyard, with the credentials Switchyard already holds to serve
inference, and published through Switchyard's own metric surface. No external
exporter or agent-based accounting path is involved.

## The files

| File | What it is |
|---|---|
| `economics_dashboard.toml` | The dashboard's identity and the deployment it reads: the Prometheus job, the Grafana datasource, the dashboard uid and title, and the routing dashboard it links to. |
| `build_economics_dashboard.py` | Builds the dashboard document from that configuration. Writes a file; publishes nothing. |
| `prepare_v7_backlink.py` | Reads the live routing dashboard, adds one link to this dashboard, and writes the update request and a rollback copy. Writes a file; publishes nothing. |

The values in the configuration are deployment facts. The generator hardcodes
none of them, so pointing this at a different deployment is a configuration
change rather than a code change.

## Building the document

```bash
python3 scripts/economics/build_economics_dashboard.py \
  --config scripts/economics/economics_dashboard.toml \
  --out /tmp/economics-v8.json
```

The output is a publishing *request*, not a raw Grafana body. The publishing
tool accepts one request shape:

```json
{"op": "create", "uid": "...", "message": "...", "folder_uid": null, "dashboard": {...}}
```

and refuses any other field. A raw Grafana POST body carries `overwrite`, which
the tool rejects as an unsupported field, and omits `op` and `uid`, which the
tool requires. The tool builds the upstream payload itself, including
`overwrite=True`, so the request must not carry it.

## Validating before publishing

The build is not the last step. A dashboard that parses can still query the
wrong series, so the expressions are checked against real scraped data before
anything is published.

Throughout this section `$LAUNCHER` stands for the publishing tool's launcher
for your environment, which accepts the verbs `query`, `dashboard` and
`dashboard-write`.

### 1. The request is accepted by the publishing tool's own parser

Publishing is a mutation. To check the parser without publishing, import the
tool's runtime module and call the parser directly:

```bash
python3 - <<'EOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("runtime", "$RUNTIME")
m = importlib.util.module_from_spec(spec)
sys.modules["runtime"] = m
spec.loader.exec_module(m)
request = m.parse_dashboard_write_request(open("/tmp/economics-v8.json", "rb").read())
print("accepted:", request["op"], request["uid"], len(request["dashboard"]["panels"]))
EOF
```

### 2. Every expression evaluates against real data

A parse-only pass proves nothing. The economics families do not exist in a
deployment that does not run this code, so a pass against such a job returns no
rows for every expression and proves only that the expressions are syntactically
valid.

The real check runs the same expressions against a deployment whose series are
genuinely scraped. That deployment's Prometheus job label differs from the one
the dashboard ships, so the job selector is rewritten and nothing else is
changed:

```bash
printf '%s' '{"datasource_uid":"<uid>","expr":"<expression>","kind":"instant"}' | \
  sudo -n "$LAUNCHER" query
```

Every expression must return real data, with one exception that is expected:
`switchyard_economics_account_auth_failures_total` returns no rows when no
account read has been rejected. That expression is proved non-vacuous against a
deployment holding deliberately invalid credentials, which must publish the
failure series.

### 3. The accounting rules still hold

The panels encode rules that a reader would otherwise have to infer. The
regression tests pin the metric side of those rules:

```bash
cargo test -p switchyard-server --test economics_account_labels \
  --test economics_accounts --test economics_attribution
python3 -m pytest tests/test_economics_dashboard.py
```

`economics_account_labels` pins the published label set. The currency, the
accounting scope and the reporting period are three separate labels, and the
purchased credit, the consumed credit and the remaining balance are three
separate series. A composite label would make two different amounts
indistinguishable and would let a query add series that were never meant to be
added.

`test_economics_dashboard.py` pins the panel side: no windowed `increase`, no
component added to the total that contains it, and every provider account query
naming both its scope and its period.

## Preparing the back-link

```bash
python3 scripts/economics/prepare_v7_backlink.py \
  --config scripts/economics/economics_dashboard.toml \
  --launcher "$LAUNCHER" \
  --out /tmp/routing-with-economics-link.json \
  --backup /tmp/routing-backup.json
```

This reads the live routing dashboard, adds one dashboard-level link and writes
the update request. It changes nothing else: the script compares the panels
before and after and refuses to write the request if they differ.

The request carries the live `expected_version`, because the publishing tool
refuses an update whose version does not match. Read the version fresh
immediately before publishing; if the dashboard has changed since, the request
is refused rather than applied to the wrong document.

## Publishing

Publishing is a separate, explicit step. It is not done by these scripts, and it
is not done as part of a build.

```bash
# Create the economics dashboard.
printf '%s' "$(cat /tmp/economics-v8.json)" | sudo -n "$LAUNCHER" dashboard-write

# Update the routing dashboard to add the back-link.
printf '%s' "$(cat /tmp/routing-with-economics-link.json)" | \
  sudo -n "$LAUNCHER" dashboard-write
```

Each call backs up the dashboard it changes before writing, and the backup path
is in the tool's output.

## Rolling back

To undo the routing dashboard update, write the backup copy back as an update
with the current version as `expected_version`. To remove the economics
dashboard, delete it with the publishing tool's delete verb, which only removes
dashboards in the tool's own uid namespace.
