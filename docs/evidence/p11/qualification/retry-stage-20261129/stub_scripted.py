#!/usr/bin/env python3
"""Scripted upstream stub for retry-contract staging.

Control surface (never proxied):
  POST /control/reset  {"script": [<step>, ...]}   -> {"ok": true, "len": N}
  GET  /control/log                                -> {"calls": [...], "remaining": M}

Each /v1/chat/completions request consumes one step. A step is:
  {"status": 200}                                    fixed success
  {"status": 429, "retry_after": "7", "code": "...", "message": "..."}
  {"status": 503, "code": "...", "message": "..."}

Every consumed step is appended to log.jsonl with wall-clock arrival time, so a caller can
prove how many outbound attempts actually happened and how far apart they were.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT_DIR = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
os.makedirs(OUT_DIR, exist_ok=True)

state = {"script": [], "calls": []}
lock = threading.Lock()


def ok_body():
    return {
        "id": "stub-1", "object": "chat.completion", "created": 0, "model": "stub",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "stub ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, payload, headers=None):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") == "/control/log":
            with lock:
                self._send(200, {"calls": list(state["calls"]),
                                 "remaining": len(state["script"])})
            return
        self._send(404, {"error": "no such control path"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        path = self.path.rstrip("/")

        if path == "/control/reset":
            try:
                doc = json.loads(raw or b"{}")
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": "bad json: %s" % exc})
                return
            with lock:
                state["script"] = list(doc.get("script") or [])
                state["calls"] = []
            self._send(200, {"ok": True, "len": len(doc.get("script") or [])})
            return

        # any other POST is treated as an upstream inference call
        with lock:
            step = state["script"].pop(0) if state["script"] else {"status": 200}
            idx = len(state["calls"]) + 1
            state["calls"].append({
                "idx": idx,
                "t": time.time(),
                "path": self.path,
                "bytes": len(raw),
                "status": step.get("status", 200),
            })
            record = {"idx": idx, "t": time.time(), "path": self.path, "bytes": len(raw),
                      "step": step}
        with open(os.path.join(OUT_DIR, "log.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

        status = int(step.get("status", 200))
        if status == 200:
            self._send(200, ok_body())
            return
        headers = {}
        if step.get("retry_after") is not None:
            headers["Retry-After"] = str(step["retry_after"])
        body = {"error": {"message": step.get("message", "stub refusal"),
                          "type": "rate_limit_error" if status == 429 else "server_error",
                          "code": step.get("code", "stub_%d" % status)}}
        self._send(status, body, headers)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
