# Deployment example

`routes.example.toml` is a complete deployment file. It shows how a fleet
that serves several providers at once is written: one direct route per
model, a few routes that fall back through a list of candidates, tiers that
order whole routes, decision lanes, and the embedding and reranking
capabilities.

## What it declares

| Section | Count | What it is |
|---|---:|---|
| `[llm_clients]` | 7 | One entry per upstream, each naming the environment variable that holds its key. |
| `[targets]` | 28 | A model on one client, with the capabilities it really has. |
| `[routes]` | 30 | 24 direct routes, plus 6 that fall back through a candidate list. |
| `[composites]` | 4 | Tiers that order whole routes rather than single models. |
| `[decision_lanes]` | 2 | Lanes of physical decision targets. |
| `[decision_lane_composition]` | 1 | A tier over those lanes. |
| `[capabilities]` | 3 | Embedding and reranking, each bound to its own client. |

A route falls back in the order of its `candidates` list. A tier composes
child *routes*, so a change to a child is inherited by every tier that
contains it.

## Running it

Every `base_url` points at a placeholder host, so the file validates but
does not serve anything until you point those at your own servers. Export
the environment variables named by each `api_key_env`, then check the file
without binding a socket:

```bash
export SWITCHYARD_OPENROUTER_API_KEY=...
switchyard-server --config routes.example.toml --dry-run
```

The file holds no secret. It names the environment variable that holds each
key, and the server reads that variable at startup.

## The default decision lane

`default_decision_lane` is a top-level key. It names the lane that serves a
request which names no lane at all, so it must be a lane the file actually
declares under `[decision_lanes]`. A composed lane cannot serve as the
default, and the server rejects the file at startup if one is named:

```
invalid server config routes.toml: default_decision_lane names tier1-decision,
which is not a configured decision lane
```

This example omits the key. A request that names no lane then fails at
request time rather than being served by a default, which is the right
behaviour for a file you are reading as a template. Add the key when you
know which lane should answer an unaddressed request.
