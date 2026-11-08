# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Server builds and route contracts

Two things that belong together because both answer one question: *is the binary
in front of a caller the one we think it is, running the routes we think it runs?*

## Building the server

`.github/workflows/switchyard-server.yml` builds `switchyard-server` and publishes
it with the identity needed to prove what it is.

| Input | Pinned to | Why |
|---|---|---|
| runner | `ubuntu-24.04` | the deployment image's native toolchain |
| Rust | `1.99.0` | `rust-toolchain.toml` names the *channel* `1.99`, which now resolves to a later patch release, so two builds of one commit can differ |
| GCC | major 13 | recorded and asserted; a different compiler is the same source but a different binary |
| dependencies | `--locked` | a lockfile change is a source change |

The job fails if the toolchain it actually used is not the one above, so an
artifact cannot silently claim a toolchain it was not built with. It also refuses
an artifact that cannot run on the deployment image: a binary linked against a
newer glibc loads on the build host and fails at activation, so the highest
required `GLIBC_` symbol is compared against Ubuntu 24.04's 2.39.

`build/provenance.json` travels with the binary and records the SHA-256, source
commit, run identity and compiler versions.

Nothing in this workflow publishes packages, tags a release, or deploys.

## Auditing a route manifest

`route_audit.py` compares a manifest with the model list a server advertises:

```bash
# Against a running server (start it on the manifest first).
route_audit.py /etc/switchyard/routes.toml --models-url http://127.0.0.1:4000

# Against a captured payload.
route_audit.py routes.toml models.json --inventory inventory.json
```

It exits non-zero with one line per violation. The invariants it enforces, and
why each one is not already caught by config load:

| Invariant | Fails when |
|---|---|
| advertised set == leaf set | a composite is advertised, or a leaf is unreachable by name |
| advertisement agrees with the manifest | a route's context window or capability differs from what the server reports |
| targets and clients resolve | a route names a target that does not exist, or a client that is not configured |
| `video` only on a responses transport | a video target sits on a client whose codec cannot carry video |
| composite children are leaves | a composite nests another, or names an unknown child |
| bounded contracts hold | a `-bounded` route does not pin `route_reasoning_policy = "none"`, or a `-bounded` composite names a child that does not |
| credentials stay separate | a route mixes two clients the caller declared exclusive |

Two rules are deliberately narrow. **Candidate fallback across providers is
allowed** — a route may fall from a local model to another provider's — so the
credential rule is opt-in per pair, because which accounts must not share a route
is deployment knowledge. And **`unbounded` is not a bounded route**: the bounded
test is a suffix match, not a substring one.

### The bounded contract

A route whose id ends in `-bounded` promises the caller that no reasoning reaches
the provider, whatever the caller asked for and whatever the target prefers. That
must hold on the wire, not only in configuration, so
`a_bounded_route_sends_reasoning_disabled_to_a_thinking_target` asserts the
outbound body against a thinking target, with the caller asking for effort, and
its control asserts that removing the route policy restores the target's switch.

To apply the contract to a deployment, change the route's
`route_reasoning_policy` to `"none"` and verify with `--dry-run` before
restarting. The audit reports any bounded definition that does not hold it.
