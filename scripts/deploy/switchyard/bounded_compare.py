#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the bounded lane against the strict non-training DeepSeek leg.

Five axes, one run, one artifact:

* **completion** -- did the route answer, and was the answer right on a task with a checkable
  answer rather than a subjective one;
* **tools** -- did a tool-bearing request reach a provider, and what came back;
* **latency** -- wall-clock milliseconds per call, reported as median and p95 rather than a
  mean, because one timeout moves a mean and says nothing;
* **fallbacks** -- which model actually served each call, so a crossing leg is visible
  instead of being credited to the route that was asked for;
* **cost** -- tokens from the response's own usage block, turned into money only when a
  price table is supplied. A price this script does not know is reported as ``null``, not
  guessed.

Two modes:

``--self-test`` starts a local stub that speaks the OpenAI chat shape and runs the entire
measurement path against it. Nothing leaves the machine, and the five axes are computed from
a response this script produced, which is what makes the harness trustworthy before anyone
authorises a real call.

Live mode requires ``--authorize LIVE`` and an explicit ``--base-url``. That is deliberate:
the default must not be a paid provider. Live mode makes no call it was not asked for and
writes one JSON artifact plus a readable summary.

Usage::

    python3 bounded_compare.py --self-test
    python3 bounded_compare.py --authorize LIVE --base-url http://switchyard-vm:8443 \\
        --token-file /srv/turnstone/credentials/... --tasks 12 --out bounded-20261121.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
# Checkable answers, short enough to be a bounded task rather than a benchmark. Each one is
# something a small model either gets right or does not, so "completion" means correctness
# and not merely a 200.

ARITHMETIC = [
    ("What is 17 * 23? Reply with the number only.", "391"),
    ("What is 144 / 12? Reply with the number only.", "12"),
    ("What is 61 + 87? Reply with the number only.", "148"),
    ("How many minutes are in 2.5 hours? Reply with the number only.", "150"),
    ("What is 9 squared minus 4 squared? Reply with the number only.", "65"),
    ("A box holds 12 rows of 8 items. How many items? Reply with the number only.", "96"),
]

EXTRACTION = [
    (
        "From this line, reply with the order id only: order 4471 shipped 2026-11-03.",
        "4471",
    ),
    (
        "From this line, reply with the city only: shipment left Rotterdam at 06:40.",
        "Rotterdam",
    ),
]

TOOL_TASK = "What is the weather in Bangkok? Use the get_weather tool."

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@dataclass
class Call:
    route: str
    kind: str  # completion | tool
    task: str
    expected: str | None
    status: int | None = None
    served_model: str | None = None
    content: str = ""
    tool_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    error: str | None = None

    @property
    def correct(self) -> bool | None:
        if self.expected is None:
            return None
        return normalise(self.content) == normalise(self.expected)

    @property
    def crossed(self) -> bool:
        """True when something other than the requested route answered."""
        return self.served_model is not None and not self.served_model.startswith("smart")


def normalise(text: str) -> str:
    return text.strip().strip(".").strip().lower()


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def post_chat(base_url: str, body: dict, headers: dict[str, str], timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"error": {"message": raw.decode(errors="replace")}}


def measure_route(
    base_url: str,
    route: str,
    headers: dict[str, str],
    timeout: float,
    limit: int | None,
) -> list[Call]:
    tasks: list[Call] = [
        Call(route=route, kind="completion", task=task, expected=expected)
        for task, expected in (ARITHMETIC + EXTRACTION)[: limit or None]
    ]
    tasks.append(Call(route=route, kind="tool", task=TOOL_TASK, expected=None))

    for call in tasks:
        body: dict = {
            "model": route,
            "messages": [{"role": "user", "content": call.task}],
            "max_tokens": 256,
        }
        if call.kind == "tool":
            body["tools"] = [WEATHER_TOOL]
            body["tool_choice"] = "auto"
        started = time.monotonic()
        try:
            status, payload = post_chat(base_url, body, headers, timeout)
        except Exception as error:  # noqa: BLE001 - the failure itself is a measurement
            call.latency_ms = (time.monotonic() - started) * 1000
            call.error = f"{type(error).__name__}: {error}"
            continue
        call.latency_ms = (time.monotonic() - started) * 1000
        call.status = status
        if status != 200:
            call.error = payload.get("error", {}).get("message") or json.dumps(payload)[:200]
            continue
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        call.content = message.get("content") or ""
        call.tool_calls = len(message.get("tool_calls") or [])
        call.served_model = payload.get("model")
        usage = payload.get("usage") or {}
        call.prompt_tokens = usage.get("prompt_tokens")
        call.completion_tokens = usage.get("completion_tokens")
    return tasks


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarise(calls: list[Call], prices: dict[str, tuple[float, float]] | None) -> dict:
    completions = [call for call in calls if call.kind == "completion"]
    graded = [call for call in completions if call.expected is not None]
    answered = [call for call in completions if call.status == 200 and call.content]
    latencies = sorted(call.latency_ms for call in calls if call.latency_ms is not None)
    p95_index = max(0, int(round(0.95 * (len(latencies) - 1)))) if latencies else 0

    def money(route: str) -> float | None:
        if not prices or route not in prices:
            return None
        prompt_price, completion_price = prices[route]
        total = 0.0
        for call in calls:
            if call.route != route:
                continue
            total += (call.prompt_tokens or 0) / 1_000_000 * prompt_price
            total += (call.completion_tokens or 0) / 1_000_000 * completion_price
        return round(total, 6)

    by_route: dict[str, dict] = {}
    for route in sorted({call.route for call in calls}):
        mine = [call for call in calls if call.route == route]
        my_completions = [call for call in mine if call.kind == "completion"]
        my_graded = [call for call in my_completions if call.expected is not None]
        my_latencies = sorted(call.latency_ms for call in mine if call.latency_ms is not None)
        by_route[route] = {
            "calls": len(mine),
            "answered": sum(1 for c in my_completions if c.status == 200 and c.content),
            "correct": sum(1 for c in my_graded if c.correct),
            "graded": len(my_graded),
            "refused": sum(1 for c in mine if c.status in (400, 403, 422)),
            "errored": sum(1 for c in mine if c.error and c.status not in (400, 403, 422)),
            "tool_calls_returned": sum(c.tool_calls for c in mine if c.kind == "tool"),
            "served_by_other": sum(1 for c in mine if c.crossed),
            "median_latency_ms": round(statistics.median(my_latencies), 1) if my_latencies else None,
            "p95_latency_ms": round(my_latencies[max(0, int(round(0.95 * (len(my_latencies) - 1))))], 1)
            if my_latencies
            else None,
            "prompt_tokens": sum(c.prompt_tokens or 0 for c in mine),
            "completion_tokens": sum(c.completion_tokens or 0 for c in mine),
            "cost": money(route),
        }

    return {
        "calls": len(calls),
        "answered": len(answered),
        "graded_accuracy": (
            round(sum(1 for call in graded if call.correct) / len(graded), 3) if graded else None
        ),
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_latency_ms": round(latencies[p95_index], 1) if latencies else None,
        "by_route": by_route,
        "fallbacks": sum(1 for call in calls if call.crossed),
        "unpriced_routes": sorted(route for route in by_route if by_route[route]["cost"] is None),
    }


# ---------------------------------------------------------------------------
# Self-test stub
# ---------------------------------------------------------------------------


def start_stub() -> tuple[ThreadingHTTPServer, str]:
    """A minimal OpenAI-chat endpoint that answers the checkable tasks correctly."""

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the self-test output readable
            return

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            model = body.get("model", "")
            text = ""
            for item in body.get("messages") or []:
                text = item.get("content") or ""
            if body.get("tools"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Bangkok"}),
                            },
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": self.answer(text)}
            payload = {
                "id": "stub",
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": max(1, len(text) // 4),
                    "completion_tokens": max(1, len(message.get("content") or "") // 4),
                },
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        @staticmethod
        def answer(text: str) -> str:
            lookup = {task: expected for task, expected in ARITHMETIC + EXTRACTION}
            return lookup.get(text, "unknown")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run against a local stub")
    parser.add_argument("--authorize", default=None, help='pass "LIVE" to permit real calls')
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--routes", default="smart-gemini-bounded,smart-nt", help="comma separated")
    parser.add_argument("--tasks", type=int, default=None, help="limit the graded task count")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.self_test:
        server, base_url = start_stub()
        headers: dict[str, str] = {}
        try:
            calls: list[Call] = []
            for route in args.routes.split(","):
                calls.extend(measure_route(base_url, route, headers, args.timeout, args.tasks))
        finally:
            server.shutdown()
        summary = summarise(calls, prices=None)
        # The stub answers every graded task correctly and returns one tool call, so a
        # harness that computed the axes wrongly shows up here rather than in a live run.
        assert summary["graded_accuracy"] == 1.0, summary
        assert summary["median_latency_ms"] is not None, summary
        assert all(
            route["tool_calls_returned"] == 1 for route in summary["by_route"].values()
        ), summary
        assert summary["unpriced_routes"], "an unpriced route must be reported, not guessed"
        print(json.dumps({"self_test": "PASS", "summary": summary}, indent=2))
        return 0

    if args.authorize != "LIVE":
        parser.error(
            "live mode needs --authorize LIVE; nothing was sent. Run --self-test to check the harness."
        )
    if not args.base_url:
        parser.error("live mode needs --base-url")
    headers = {}
    if args.token_file:
        headers["authorization"] = f"Bearer {args.token_file.read_text().strip()}"

    calls = []
    for route in args.routes.split(","):
        calls.extend(measure_route(args.base_url, route, headers, args.timeout, args.tasks))
    summary = summarise(calls, prices=None)
    artifact = {"mode": "live", "route_calls": [asdict(call) for call in calls], "summary": summary}
    if args.out:
        args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
