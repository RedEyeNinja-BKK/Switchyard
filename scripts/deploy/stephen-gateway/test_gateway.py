#!/usr/bin/env python3
"""Gateway v4 acceptance tests. Loopback only, mock upstream, no credentials.

Every check pins a property that a plausible implementation gets wrong:

  auth      absent / unknown / non-Bearer tokens never reach the upstream
  grants    a client's allow-list is exactly what it may call
  paid      ST_Paid needs an explicit grant AND a positive ceiling
  ceiling   concurrent paid requests cannot each see the same remaining budget
  bounds    an oversized body is refused before it is read
  streaming SSE is relayed incrementally, not batched at the end
  secrecy   the caller's token never reaches the upstream or the log
  models    /v1/models is filtered per client
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

GATEWAY = os.environ.get("GATEWAY_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway.py"))
UPSTREAM_PORT = 14970
PAID_SLEEP = float(os.environ.get("PAID_SLEEP", "1.5"))

CATALOG = ["ST_Paid", "ST_Tier1", "ST_Tier2", "ST_Tier3", "ST_Route",
           "ST_Rerank", "ST_Embed", "smartfree", "direct-openai-gpt-6-sol"]

SEEN_AUTH: list[str | None] = []
SEEN_PATHS: list[str] = []
results: list[tuple[bool, str, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        SEEN_AUTH.append(self.headers.get("Authorization"))
        SEEN_PATHS.append(self.path)
        body = json.dumps({"object": "list", "data": [{"id": m} for m in CATALOG]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        SEEN_AUTH.append(self.headers.get("Authorization"))
        SEEN_PATHS.append(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("model") == "ST_Paid":
            time.sleep(PAID_SLEEP if not body.get("input") == "reported" else 0)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for piece in (b'data: {"n":1}\n\n', b'data: {"n":2}\n\n'):
                self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
                self.wfile.flush()
                time.sleep(1.0)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        # A paid route is served by a concrete model, and the response names it.
        served = "z-ai/glm-5.3-flash" if body.get("model") == "ST_Paid" else body.get("model")
        payload = json.dumps({
            "id": "r", "object": "response", "model": served, "status": "completed",
            "output": [{"type": "message", "id": "m", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": ({"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500,
                       "cost": 0.0004321}
                      if body.get("input") == "reported"
                      else {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def post(gw_port: int, token: str | None, model: str, extra: dict | None = None,
         stream: bool = False, timeout: float = 60.0) -> tuple[int, dict | str]:
    payload = {"model": model, "input": "hi"}
    if stream:
        payload["stream"] = True
    if extra:
        payload.update(extra)
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(f"http://127.0.0.1:{gw_port}/v1/responses",
                                     data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            return response.status, _parse(raw)
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        return error.code, _parse(raw)
    except Exception as error:  # noqa: BLE001
        return 0, str(error)


def _parse(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="gw4-", dir="/var/tmp/st-probe-a1"))
    cfg_path = work / "clients.json"
    state_path = work / "state.json"
    log_path = work / "gw.log"
    gw_port = free_port()

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
            {"name": "paid-reported", "token_sha256": sha("tok-reported"),
             "allow": ["ST_Paid"], "monthly_spend_ceiling_usd": 5.0},
        ],
        "pricing": {"z-ai/glm-5.3-flash": [0.225, 2.0],
                    "deepseek/deepseek-v4.1-flash": [0.6, 2.4]},
        "max_request_bytes": 4096,
        "upstream_timeout_seconds": 60,
    }
    cfg_path.write_text(json.dumps(cfg))
    cfg_path.chmod(0o600)

    upstream = ThreadingHTTPServer(("127.0.0.1", UPSTREAM_PORT), Upstream)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    # Point the gateway at the mock and a free port, without editing its source.
    launcher = work / "launch.py"
    launcher.write_text(
        "import runpy, sys\n"
        f"src = open({GATEWAY!r}).read()\n"
        f"src = src.replace('UPSTREAM = (\"127.0.0.1\", 4000)', 'UPSTREAM = (\"127.0.0.1\", {UPSTREAM_PORT})')\n"
        f"src = src.replace('LISTEN = (\"127.0.0.1\", 4010)', 'LISTEN = (\"127.0.0.1\", {gw_port})')\n"
        "exec(compile(src, 'gateway', 'exec'), {'__name__': '__main__'})\n"
    )
    env = dict(os.environ)
    env["STEPHEN_GATEWAY_CONFIG"] = str(cfg_path)
    env["STEPHEN_GATEWAY_STATE"] = str(state_path)
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, str(launcher)], env=env,
                            stdout=log, stderr=log, start_new_session=True)
    try:
        for _ in range(80):
            try:
                with socket.create_connection(("127.0.0.1", gw_port), timeout=0.3):
                    break
            except OSError:
                time.sleep(0.1)
        run(gw_port, work)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        upstream.shutdown()
        log.close()

    bad = [r for r in results if not r[0]]
    print(f"\n=== {len(results) - len(bad)}/{len(results)} checks passed ===")
    for _ok, name, detail in bad:
        print("  FAILED:", name, detail)
    return 1 if bad else 0


def run(gw_port: int, work: pathlib.Path) -> None:
    print("\n--- authentication ---")
    before = len(SEEN_AUTH)
    st, _ = post(gw_port, None, "ST_Tier1")
    chk("no token refused", st == 401, f"http={st}")
    st, _ = post(gw_port, "wrong-token", "ST_Tier1")
    chk("unknown token refused", st == 401, f"http={st}")
    st, _ = post(gw_port, "", "ST_Tier1")
    chk("empty token refused", st == 401, f"http={st}")
    chk("refused requests never reached the upstream", len(SEEN_AUTH) == before,
        f"upstream saw {len(SEEN_AUTH) - before} call(s)")

    print("\n--- grants ---")
    st, body = post(gw_port, "tok-free", "ST_Tier1")
    chk("free client: allowed route passes", st == 200, f"http={st}")
    st, _ = post(gw_port, "tok-free", "ST_Tier3")
    chk("free client: unlisted route denied", st == 403, f"http={st}")
    st, _ = post(gw_port, "tok-free", "direct-openai-gpt-6-sol")
    chk("free client: a global paid route is not reachable", st == 403, f"http={st}")
    st, _ = post(gw_port, "tok-free", "ST_Paid")
    chk("free client: paid route denied", st == 403, f"http={st}")

    print("\n--- paid authorization ---")
    st, _ = post(gw_port, "tok-nobudget", "ST_Paid", {"max_output_tokens": 64})
    chk("client with no budget denied", st == 403, f"http={st}")
    st, _ = post(gw_port, "tok-paid", "ST_Paid")
    chk("paid without max_output_tokens refused", st == 403, f"http={st}")
    st, body = post(gw_port, "tok-paid", "ST_Paid", {"max_output_tokens": 64})
    chk("paid with grant and budget passes", st == 200, f"http={st}")

    print("\n--- ceiling is atomic under concurrency ---")
    # Each paid request reserves 1000 output tokens at 2.4/M plus body bytes at
    # 0.6/M: about 0.0025 USD. A 0.01 ceiling affords four. The mock holds every
    # request for PAID_SLEEP seconds so all reservations are taken before any
    # settles, which is exactly the window a check-then-charge design loses.
    state_path = work / "state.json"

    # The buffered response above named z-ai/glm-5.3-flash and reported
    # 1000 in / 500 out, so the charge must be the real 0.001225 USD rather than
    # the much smaller reservation taken before dispatch.
    settle_deadline = time.time() + 10
    paid_state: dict = {}
    while time.time() < settle_deadline:
        paid_state = json.loads(state_path.read_text())
        if abs(float(paid_state.get("paid-client", {}).get("reserved_usd", 0))) < 1e-9:
            break
        time.sleep(0.05)
    charged = float(paid_state.get("paid-client", {}).get("spend_usd", 0))
    expected = 1000 / 1_000_000 * 0.225 + 500 / 1_000_000 * 2.0
    chk("charged the served model's real usage", abs(charged - expected) < 1e-9,
        f"charged={charged} expected={expected}")

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
    chk("every other concurrent request was refused", allowed + refused == len(codes),
        f"{sorted(set(codes))}")

    # A client sees its response body before the server has finished writing the
    # ledger, so settlement is polled rather than assumed immediate.
    deadline = time.time() + 10
    state: dict = {}
    while time.time() < deadline:
        state = json.loads(state_path.read_text())
        if abs(float(state.get("paid-tiny", {}).get("reserved_usd", 0))) < 1e-9:
            break
        time.sleep(0.05)
    spent = float(state.get("paid-tiny", {}).get("spend_usd", 0))
    reserved = float(state.get("paid-tiny", {}).get("reserved_usd", 0))
    chk("settled spend never exceeds the ceiling", spent <= 0.01 + 1e-9, f"spent={spent}")
    chk("settled spend accounts for every accepted request", spent > 0, f"spent={spent}")
    chk("no reservation is left outstanding", abs(reserved) < 1e-9, f"reserved={reserved}")

    st, _ = post(gw_port, "tok-reported", "ST_Paid",
                 {"max_output_tokens": 64, "input": "reported"})
    deadline = time.time() + 10
    reported_state: dict = {}
    while time.time() < deadline:
        reported_state = json.loads(state_path.read_text())
        if abs(float(reported_state.get("paid-reported", {}).get("reserved_usd", 0))) < 1e-9:
            break
        time.sleep(0.05)
    reported_charge = float(reported_state.get("paid-reported", {}).get("spend_usd", 0))
    chk("the provider's reported cost is trusted over the rate table",
        abs(reported_charge - 0.0004321) < 1e-9, f"charged={reported_charge}")

    print("\n--- bounds ---")
    big = "x" * 8192
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}/v1/responses",
        data=json.dumps({"model": "ST_Tier1", "input": big}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer tok-free"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            code = response.status
    except urllib.error.HTTPError as error:
        code = error.code
    except Exception as error:  # noqa: BLE001
        code = 0
    chk("oversized body refused", code == 413, f"http={code}")

    print("\n--- streaming stays streaming ---")
    started = time.time()
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}/v1/responses",
        data=json.dumps({"model": "ST_Tier1", "input": "hi", "stream": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer tok-free"},
        method="POST")
    first_chunk_at = None
    pieces = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            while True:
                chunk = response.read1(4096) if hasattr(response, "read1") else response.read(4096)
                if not chunk:
                    break
                if first_chunk_at is None:
                    first_chunk_at = time.time() - started
                pieces += 1
    except Exception as error:  # noqa: BLE001
        chk("stream completed", False, str(error))
    total = time.time() - started
    chk("first streamed chunk arrives before the stream ends",
        first_chunk_at is not None and first_chunk_at < total - 0.5,
        f"first={first_chunk_at:.2f}s total={total:.2f}s")
    chk("stream delivered in more than one piece", pieces >= 2, f"pieces={pieces}")

    print("\n--- models filtering ---")
    def models(token: str) -> list[str]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{gw_port}/v1/models",
            headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=30) as response:
            return sorted(m["id"] for m in json.load(response)["data"])

    chk("free client sees only its grants",
        models("tok-free") == ["ST_Rerank", "ST_Route", "ST_Tier1"],
        f"{models('tok-free')}")
    chk("paid client sees its paid grant", "ST_Paid" in models("tok-paid"),
        f"{models('tok-paid')}")

    print("\n--- secrecy ---")
    chk("no caller token was forwarded upstream",
        all(a is None or not a.startswith("Bearer tok-") for a in SEEN_AUTH),
        f"distinct auth seen: {sorted({str(a) for a in SEEN_AUTH})}")
    log_text = (work / "gw.log").read_text(errors="replace")
    chk("no token in the gateway log", "tok-free" not in log_text and "tok-paid" not in log_text)
    chk("config holds no usable token",
        all("tok-" not in str(c.get("token_sha256")) for c in json.loads(
            (work / "clients.json").read_text())["clients"]))

    print("\n--- path surface ---")
    for path in ("/v1/chat/completions", "/admin", "/"):
        request = urllib.request.Request(f"http://127.0.0.1:{gw_port}{path}",
                                        headers={"Authorization": "Bearer tok-free"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                code = response.status
        except urllib.error.HTTPError as error:
            code = error.code
        except Exception:  # noqa: BLE001
            code = 0
        chk(f"path {path} refused", code == 403, f"http={code}")

    print("\n--- malformed input ---")
    request = urllib.request.Request(
        f"http://127.0.0.1:{gw_port}/v1/responses", data=b"not json",
        headers={"Content-Type": "application/json", "Authorization": "Bearer tok-free"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            code = response.status
    except urllib.error.HTTPError as error:
        code = error.code
    except Exception:  # noqa: BLE001
        code = 0
    chk("non-JSON body refused", code == 400, f"http={code}")


if __name__ == "__main__":
    sys.exit(main())
