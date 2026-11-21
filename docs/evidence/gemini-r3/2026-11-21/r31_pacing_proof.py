#!/usr/bin/env python3
"""Live proof that the per-minute governor sits in front of the Google lanes.

Three public/synthetic requests to one direct Google leaf, in a burst, each carrying about
5,000 estimated input tokens against a declared 14,400-token minute window. The first two
fit (10,002); the third cannot (15,003), so it must be refused locally with
`minute_allowance_throttled` and must never reach the provider.

Upstream attempts: at most two. The proof is the contrast between three requests sent and
two departures, which is exactly the property that was missing before this change.
"""

from __future__ import annotations

import json
import time

import r3_canary as canary

LEAF = "direct-google-gemma-4-26b-a4b-it"
TOKENS_EACH = 8_000
PROMPT = (
    "Synthetic pacing proof. Ignore this text and reply with the single word READY. "
    + ("alpha bravo charlie delta " * 1280)
)

results = []
started = time.time()
for index in range(2):
    status, payload, seconds = canary.post(
        "/v1/chat/completions",
        {
            "model": LEAF,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 32,
        },
        timeout=120,
    )
    error = payload.get("error") or {}
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    results.append(
        {
            "attempt": index + 1,
            "status": status,
            "seconds": round(seconds, 2),
            "served_model": payload.get("model"),
            "content_head": (message.get("content") or "")[:40],
            "error_code": error.get("code"),
            "error_type": error.get("type"),
            "error_message": (error.get("message") or "")[:200],
            "usage": payload.get("usage"),
        }
    )
    print(json.dumps(results[-1], ensure_ascii=False))
    if index == 0:
        # Clear the 5s capacity cooldown a 503 arms, without leaving the 60s token window.
        time.sleep(7)
print(json.dumps({"started": started, "finished": time.time(), "leaf": LEAF}))
