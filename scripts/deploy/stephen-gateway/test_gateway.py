#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gateway acceptance tests: authentication, grants, accounting, streaming.

Every check pins a property a plausible implementation gets wrong. The accounting
cases matter most, because getting them wrong is invisible until a budget is
either silently exceeded or silently forgiven:

  streaming      a stream that reports usage settles to it; one that does not
                 still charges its reservation
  interruption   a client that disconnects mid-stream, and an upstream that dies
                 after producing output, must both remain chargeable
  restart        a reservation left by a dead process becomes spend, not a refund
  malformed      negative, non-numeric or absurd usage falls back to the bound
  concurrency    the ceiling holds when requests race
  discovery      a granted composite is visible, an ungranted one is not

Run: python3 test_gateway.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATEWAY = os.environ.get(
    "GATEWAY_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway.py"))
UPSTREAM_PORT = 14970
PAID_SLEEP = float(os.environ.get("PAID_SLEEP", "1.2"))

CATALOG = ["ST_Paid", "ST_Tier1", "ST_Tier2", "ST_Tier3",
           "ST_Rerank", "ST_Embed", "smartfree", "direct-openai-gpt-6-sol"]
# Served-model names the mock reports back, so pricing can be checked per target.
SERVED = {"ST_Paid": "z-ai/glm-5.3-flash"}

SEEN_AUTH: list[str | None] = []
RESULTS: list[tuple[bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((ok, name))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        SEEN_AUTH.append(self.headers.get("Authorization"))
        body = json.dumps({"object": "list", "data": [{"id": m} for m in CATALOG]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ helpers
    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _sse_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse_send(self, payload: bytes) -> None:
        self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
        self.wfile.flush()

    def _json(self, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        SEEN_AUTH.append(self.headers.get("Authorization"))
        body = self._body()
        model = body.get("model", "")
        marker = body.get("input")
        served = SERVED.get(model, model)
        # Deliberately modest: a ~60 byte body cannot carry 1000 tokens, and a mock
        # that claims otherwise would be testing arithmetic no real request produces.
        usage = {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}

        if model == "ST_Paid" and marker != "stream":
            time.sleep(PAID_SLEEP)

        # --- buffered responses ------------------------------------------------
        if marker == "bad-usage":
            self._json({"id": "r", "object": "response", "model": served,
                        "usage": {"input_tokens": -5, "output_tokens": "NaN", "cost": -1}})
            return
        if marker == "string-usage":
            self._json({"id": "r", "object": "response", "model": served,
                        "usage": {"input_tokens": "many", "output_tokens": None,
                                  "cost": "free"}})
            return
        if marker == "huge-usage":
            self._json({"id": "r", "object": "response", "model": served,
                        "usage": {**usage, "cost": 999.0}})
            return
        if marker == "reported-cost":
            self._json({"id": "r", "object": "response", "model": served,
                        "usage": {**usage, "cost": 0.0000432}})
            return
        if marker == "r2-served":
            self._json({"id": "r", "object": "response",
                        "model": "deepseek/deepseek-v4.1-flash", "usage": usage})
            return

        # --- streaming ---------------------------------------------------------
        if marker == "stream-usage":
            self._sse_head()
            self._sse_send(b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n')
            time.sleep(0.2)
            self._sse_send(
                b'data: {"type":"response.completed","response":{"usage":'
                b'{"input_tokens":40,"output_tokens":20,"cost":0.00002}}}\n\n')
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        if marker == "stream-nousage":
            self._sse_head()
            self._sse_send(b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n')
            time.sleep(0.2)
            self._sse_send(b'data: {"type":"response.output_text.done"}\n\n')
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        if marker in ("stream-cut", "stream"):
            # Produces output, then the connection dies without a terminator:
            # stands in for an upstream failure after tokens were generated.
            self._sse_head()
            self._sse_send(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
            time.sleep(0.3)
            # BaseHTTPRequestHandler keeps makefile() handles open, and closing the
            # socket defers the real close while they exist -- so the peer would
            # never see the end. Closing them first is what ends the connection.
            try:
                self.wfile.close()
                self.rfile.close()
            except OSError:
                pass
            try:
                self.connection.close()
            except OSError:
                pass
            return

        # --- default -----------------------------------------------------------
        self._json({"id": "r", "object": "response", "model": served,
                    "status": "completed",
                    "output": [{"type": "message", "id": "m", "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "ok",
                                             "annotations": []}]}],
                    "usage": usage})


def post(gw_port: int, token: str | None, model: str,
         extra: dict | None = None, timeout: float = 60.0):
    payload = {"model": model, "input": "hi"}
    if extra:
        payload.update(extra)
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}/v1/responses",
        data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            return response.status, _parse(raw)
    except urllib.error.HTTPError as error:
        return error.code, _parse(error.read().decode())
    except Exception as error:  # noqa: BLE001
        return 0, str(error)


def _parse(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def models(gw_port: int, token: str) -> list[str]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}/v1/models",
        headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=30) as response:
        return sorted(m["id"] for m in json.load(response)["data"])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_settled(state_path: pathlib.Path, client: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        try:
            last = json.loads(state_path.read_text())
        except (OSError, ValueError):
            time.sleep(0.05)
            continue
        row = last.get(client) or {}
        if abs(float(row.get("reserved_usd") or 0)) < 1e-9:
            return row
        time.sleep(0.05)
    return last.get(client) or {}


def spawn(gw_port: int, work: pathlib.Path, env_extra: dict | None = None):
    launcher = work / "launch.py"
    launcher.write_text(
        "import sys\n"
        f"src = open({GATEWAY!r}).read()\n"
        f"src = src.replace('UPSTREAM = (\"127.0.0.1\", 4000)', 'UPSTREAM = (\"127.0.0.1\", {UPSTREAM_PORT})')\n"
        f"src = src.replace('LISTEN = (\"127.0.0.1\", 4010)', 'LISTEN = (\"127.0.0.1\", {gw_port})')\n"
        "exec(compile(src, 'gateway', 'exec'), {'__name__': '__main__'})\n"
    )
    env = dict(os.environ)
    env["STEPHEN_GATEWAY_CONFIG"] = str(work / "clients.json")
    env["STEPHEN_GATEWAY_STATE"] = str(work / "state.json")
    if env_extra:
        env.update(env_extra)
    log = open(work / "gw.log", "a")
    proc = subprocess.Popen([sys.executable, str(launcher)], env=env,
                            stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", gw_port), timeout=0.3):
                return proc, log
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("gateway did not start")


def stop(proc: subprocess.Popen, log) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="gw5-", dir="/var/tmp/st-probe-a1"))
    cfg = {
        "clients": [
            {"name": "free-client", "token_sha256": sha("tok-free"),
             "allow": ["ST_Tier1", "ST_Route", "ST_Rerank"], "monthly_spend_ceiling_usd": 0},
            {"name": "paid-client", "token_sha256": sha("tok-paid"),
             "allow": ["ST_Paid", "ST_Tier1"], "monthly_spend_ceiling_usd": 5.0},
            {"name": "paid-tiny", "token_sha256": sha("tok-tiny"),
             "allow": ["ST_Paid"], "monthly_spend_ceiling_usd": 0.01},
            {"name": "no-budget", "token_sha256": sha("tok-nobudget"),
             "allow": ["ST_Paid"], "monthly_spend_ceiling_usd": 0},
            {"name": "other-paid", "token_sha256": sha("tok-other"),
             "allow": ["ST_Paid"], "monthly_spend_ceiling_usd": 5.0},
        ],
        "pricing": {"z-ai/glm-5.3-flash": [0.225, 2.0],
                    "deepseek/deepseek-v4.1-flash": [0.6, 2.4]},
        "max_request_bytes": 4096,
        "upstream_timeout_seconds": 60,
    }
    (work / "clients.json").write_text(json.dumps(cfg))
    (work / "clients.json").chmod(0o600)

    upstream = ThreadingHTTPServer(("127.0.0.1", UPSTREAM_PORT), Upstream)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    gw_port = free_port()
    state_path = work / "state.json"
    proc, log = spawn(gw_port, work)
    try:
        run(gw_port, work, state_path)
        restart_checks(work, state_path)
    finally:
        stop(proc, log)
        upstream.shutdown()

    bad = [name for ok, name in RESULTS if not ok]
    print(f"\n=== {len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed ===")
    for name in bad:
        print("  FAILED:", name)
    return 1 if bad else 0


def run(gw_port: int, work: pathlib.Path, state_path: pathlib.Path) -> None:
    print("\n--- authentication ---")
    before = len(SEEN_AUTH)
    chk("no token refused", post(gw_port, None, "ST_Tier1")[0] == 401)
    chk("unknown token refused", post(gw_port, "nope", "ST_Tier1")[0] == 401)
    chk("empty token refused", post(gw_port, "", "ST_Tier1")[0] == 401)
    chk("refused requests never reached the upstream",
        len(SEEN_AUTH) == before, f"{len(SEEN_AUTH) - before} call(s)")

    print("\n--- grants ---")
    chk("allowed route passes", post(gw_port, "tok-free", "ST_Tier1")[0] == 200)
    chk("unlisted route denied", post(gw_port, "tok-free", "ST_Tier3")[0] == 403)
    chk("global paid route not reachable",
        post(gw_port, "tok-free", "direct-openai-gpt-6-sol")[0] == 403)
    chk("paid route denied without a grant", post(gw_port, "tok-free", "ST_Paid")[0] == 403)

    print("\n--- paid authorization ---")
    chk("zero-budget client denied",
        post(gw_port, "tok-nobudget", "ST_Paid", {"max_output_tokens": 64})[0] == 403)
    chk("paid without an output bound refused",
        post(gw_port, "tok-paid", "ST_Paid")[0] == 403)
    chk("negative output bound refused",
        post(gw_port, "tok-paid", "ST_Paid", {"max_output_tokens": -10})[0] == 403)
    chk("grant plus budget passes",
        post(gw_port, "tok-paid", "ST_Paid", {"max_output_tokens": 64})[0] == 200)

    print("\n--- accounting: buffered ---")
    row = wait_settled(state_path, "paid-client")
    first = float(row.get("spend_usd") or 0)
    # usage is 40 in / 20 out at USD 0.225 / 2.0 per million = 0.000049
    chk("buffered call charged the reported usage", abs(first - 0.000049) < 1e-9,
        f"spent={first}")
    chk("no reservation left outstanding", abs(float(row.get("reserved_usd") or 0)) < 1e-9,
        f"reserved={row.get('reserved_usd')}")

    print("\n--- accounting: the provider's own cost is authoritative ---")
    post(gw_port, "tok-paid", "ST_Paid",
         {"max_output_tokens": 64, "input": "reported-cost"})
    row = wait_settled(state_path, "paid-client")
    delta = float(row.get("spend_usd") or 0) - first
    chk("reported cost trusted over the rate table", abs(delta - 0.0000432) < 1e-9,
        f"delta={delta:.8f}")

    print("\n--- accounting: malformed usage falls back to the bound ---")
    for marker, label in (("bad-usage", "negative and non-numeric"),
                          ("string-usage", "string values"),
                          ("huge-usage", "absurd cost above the reservation")):
        before_row = wait_settled(state_path, "paid-client")
        before_spend = float(before_row.get("spend_usd") or 0)
        chk(f"{label}: request served",
            post(gw_port, "tok-paid", "ST_Paid",
                 {"max_output_tokens": 64, "input": marker})[0] == 200)
        after_row = wait_settled(state_path, "paid-client")
        spent = float(after_row.get("spend_usd") or 0) - before_spend
        # The reservation for a ~90 byte body and 64 output tokens is tiny but real.
        chk(f"{label}: falls back to a charge, never free or negative",
            0 < spent < 0.01, f"charged={spent:.8f}")

    print("\n--- accounting: streaming ---")
    before_row = wait_settled(state_path, "paid-client")
    before_spend = float(before_row.get("spend_usd") or 0)
    status, raw = post(gw_port, "tok-paid", "ST_Paid",
                       {"max_output_tokens": 1000, "input": "stream-usage"})
    row = wait_settled(state_path, "paid-client")
    streamed = float(row.get("spend_usd") or 0) - before_spend
    chk("stream with final usage served", status == 200, f"http={status}")
    chk("stream settles to its reported usage", abs(streamed - 0.00002) < 1e-9,
        f"charged={streamed:.8f}")

    before_spend = float(wait_settled(state_path, "paid-client").get("spend_usd") or 0)
    status, _ = post(gw_port, "tok-paid", "ST_Paid",
                     {"max_output_tokens": 1000, "input": "stream-nousage"})
    row = wait_settled(state_path, "paid-client")
    charged = float(row.get("spend_usd") or 0) - before_spend
    chk("stream without usage served", status == 200, f"http={status}")
    chk("stream without usage keeps a conservative charge",
        charged > 0, f"charged={charged:.8f}")
    # body bytes at the dearest prompt rate plus the declared output bound at the
    # dearest completion rate: 0.6 and 2.4 USD per million.
    chk("that charge is the whole reservation, not a partial one",
        0.0023 < charged < 0.0026, f"charged={charged:.8f}")

    print("\n--- accounting: interruption must stay chargeable ---")
    before_spend = float(wait_settled(state_path, "paid-client").get("spend_usd") or 0)
    try:
        status, _ = post(gw_port, "tok-paid", "ST_Paid",
                         {"max_output_tokens": 1000, "input": "stream-cut"}, timeout=10)
    except Exception:  # noqa: BLE001
        status = 0
    row = wait_settled(state_path, "paid-client")
    charged = float(row.get("spend_usd") or 0) - before_spend
    chk("upstream failure after output still charges", charged > 0,
        f"charged={charged:.8f} (http={status})")
    chk("the reservation was not silently refunded",
        abs(float(row.get("reserved_usd") or 0)) < 1e-9,
        f"reserved={row.get('reserved_usd')}")

    # A client that walks away mid-stream: the provider was already working.
    before_spend = float(wait_settled(state_path, "paid-client").get("spend_usd") or 0)
    abandon(gw_port, "tok-paid",
            {"model": "ST_Paid", "input": "stream-nousage", "max_output_tokens": 1000})
    row = wait_settled(state_path, "paid-client")
    charged = float(row.get("spend_usd") or 0) - before_spend
    chk("client disconnect mid-stream still charges", charged > 0,
        f"charged={charged:.8f}")
    chk("disconnect leaves no dangling reservation",
        abs(float(row.get("reserved_usd") or 0)) < 1e-9,
        f"reserved={row.get('reserved_usd')}")

    print("\n--- accounting: selected model ---")
    before_spend = float(wait_settled(state_path, "paid-client").get("spend_usd") or 0)
    post(gw_port, "tok-paid", "ST_Paid", {"max_output_tokens": 64, "input": "r2-served"})
    row = wait_settled(state_path, "paid-client")
    charged = float(row.get("spend_usd") or 0) - before_spend
    expected = 40 / 1_000_000 * 0.6 + 20 / 1_000_000 * 2.4
    chk("R1 to R2 fallback is priced at the model that served",
        abs(charged - expected) < 1e-9, f"charged={charged:.8f} expected={expected:.8f}")

    print("\n--- ceiling under concurrency ---")
    if state_path.exists():
        state_path.unlink()
    codes: list[int] = []
    lock = threading.Lock()

    def fire() -> None:
        status, _ = post(gw_port, "tok-tiny", "ST_Paid", {"max_output_tokens": 1000})
        with lock:
            codes.append(status)

    threads = [threading.Thread(target=fire) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    allowed = sum(1 for c in codes if c == 200)
    refused = sum(1 for c in codes if c == 403)
    chk("concurrent paid requests cannot overrun the ceiling", allowed == 4,
        f"accepted={allowed} refused={refused} of {len(codes)}")
    chk("every request was decided", allowed + refused == len(codes), f"{sorted(set(codes))}")

    row = wait_settled(state_path, "paid-tiny")
    chk("settled spend stays within the ceiling",
        float(row.get("spend_usd") or 0) <= 0.01 + 1e-9,
        f"spent={row.get('spend_usd')}")

    print("\n--- independent clients ---")
    before_a = float(wait_settled(state_path, "paid-client").get("spend_usd") or 0)
    before_b = float(wait_settled(state_path, "other-paid").get("spend_usd") or 0)
    results: list[tuple[str, int]] = []

    def fire_client(token: str) -> None:
        status, _ = post(gw_port, token, "ST_Paid", {"max_output_tokens": 64})
        with lock:
            results.append((token, status))

    pair = [threading.Thread(target=fire_client, args=(tok,))
            for tok in ("tok-paid", "tok-other")]
    for thread in pair:
        thread.start()
    for thread in pair:
        thread.join(timeout=120)
    chk("both independent clients were served",
        sorted(status for _, status in results) == [200, 200], f"{results}")
    after_a = float(wait_settled(state_path, "paid-client").get("spend_usd") or 0)
    after_b = float(wait_settled(state_path, "other-paid").get("spend_usd") or 0)
    chk("each client is charged its own request",
        after_a > before_a and after_b > before_b,
        f"client-a +{after_a - before_a:.8f} client-b +{after_b - before_b:.8f}")

    print("\n--- model discovery ---")
    chk("free client sees its grants plus the composite",
        models(gw_port, "tok-free") == ["ST_Rerank", "ST_Route", "ST_Tier1"],
        f"{models(gw_port, 'tok-free')}")
    paid_seen = models(gw_port, "tok-paid")
    chk("paid client sees its paid grant", "ST_Paid" in paid_seen, f"{paid_seen}")
    chk("a client never sees a route it was not granted",
        "ST_Route" not in paid_seen, f"{paid_seen}")
    chk("the composite is marked as such",
        any(m.get("switchyard", {}).get("kind") == "composite"
            for m in _model_rows(gw_port, "tok-free") if m.get("id") == "ST_Route"))
    chk("no capability is invented for a composite",
        not any("context_length" in m for m in _model_rows(gw_port, "tok-free")
                if m.get("id") == "ST_Route"))

    print("\n--- bounds and paths ---")
    chk("oversized body refused",
        post(gw_port, "tok-free", "ST_Tier1", {"input": "x" * 8192})[0] == 413)
    chk("non-JSON body refused", raw_post(gw_port, "tok-free", b"not json") == 400)
    for path in ("/v1/chat/completions", "/admin", "/"):
        chk(f"path {path} refused", raw_post(gw_port, "tok-free", b"{}", path) == 403)

    print("\n--- secrecy ---")
    chk("no caller token forwarded upstream",
        all(a is None or not a.startswith("Bearer tok-") for a in SEEN_AUTH),
        f"{sorted({str(a) for a in SEEN_AUTH})}")
    log_text = (work / "gw.log").read_text(errors="replace")
    chk("no token in the gateway log",
        "tok-free" not in log_text and "tok-paid" not in log_text)


def _model_rows(gw_port: int, token: str) -> list[dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}/v1/models",
        headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["data"]


def raw_post(gw_port: int, token: str, body: bytes, path: str = "/v1/responses") -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}{path}", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception:  # noqa: BLE001
        return 0


def abandon(gw_port: int, token: str, payload: dict) -> None:
    """Send a request, read a little, then close the socket mid-stream."""
    body = json.dumps(payload).encode()
    sock = socket.create_connection(("127.0.0.1", gw_port), timeout=10)
    try:
        request = (
            f"POST /v1/responses HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{gw_port}\r\n"
            f"Authorization: Bearer {token}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode() + body
        sock.sendall(request)
        sock.recv(64)  # headers, at least
    finally:
        sock.close()


def restart_checks(work: pathlib.Path, state_path: pathlib.Path) -> None:
    """A reservation left by a dead process must become spend, not a refund."""
    print("\n--- restart with an outstanding reservation ---")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    row = state.setdefault("paid-client", {})
    row.setdefault("spend_usd", 0.0)
    row.setdefault("requests", 0)
    row["reservations"] = {"deadbeef": {"amount": 0.0042, "model": "ST_Paid",
                                        "opened_at": time.time() - 60}}
    row["reserved_usd"] = 0.0042
    state_path.write_text(json.dumps(state))
    before = float(row["spend_usd"])

    gw_port = free_port()
    proc, log = spawn(gw_port, work)
    try:
        deadline = time.time() + 15
        settled: dict = {}
        while time.time() < deadline:
            settled = json.loads(state_path.read_text()).get("paid-client") or {}
            if abs(float(settled.get("reserved_usd") or 0)) < 1e-9:
                break
            time.sleep(0.05)
        chk("the reservation was converted to spend",
            abs(float(settled.get("spend_usd") or 0) - (before + 0.0042)) < 1e-9,
            f"before={before:.8f} after={settled.get('spend_usd')}")
        chk("and no reservation remains", abs(float(settled.get("reserved_usd") or 0)) < 1e-9,
            f"reserved={settled.get('reserved_usd')}")
        log_text = (work / "gw.log").read_text(errors="replace")
        chk("the reconciliation is reported in the log",
            "reconciled" in log_text and "outstanding" in log_text)

        # A second start must not charge it again.
        proc2, log2 = spawn(free_port(), work)
        try:
            time.sleep(0.6)
            again = json.loads(state_path.read_text()).get("paid-client") or {}
            chk("a second start does not charge it twice",
                abs(float(again.get("spend_usd") or 0)
                    - (before + 0.0042)) < 1e-9,
                f"spent={again.get('spend_usd')}")
        finally:
            stop(proc2, log2)
    finally:
        stop(proc, log)


if __name__ == "__main__":
    sys.exit(main())
