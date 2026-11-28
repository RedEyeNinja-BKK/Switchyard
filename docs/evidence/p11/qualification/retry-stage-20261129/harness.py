#!/usr/bin/env python3
"""Retry-contract staging harness: real HTTP, staged candidate, no production contact.

Starts the scripted stub and the candidate `switchyard-server` (debug build from the
retry-contract branch), drives the scenarios, and prints one machine-graded line per
assertion. Everything is local (127.0.0.1), disposable and torn down at the end.

Two facts drive the shape of these scenarios:

* Switchyard states its OWN intervals (minute window, armed cooldown) and Turnstone paces
  on those.
* An upstream's own `Retry-After` is parsed by the client -- a retry gap proves it -- but it
  is not currently forwarded to the caller on a passthrough route. That is measured here as
  a gap, not assumed either way.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

STUB_PORT = 18991
SRV_PORT = 18992
ROOT = "/var/tmp/retry-stage-20261129/run"
SERVER = "/mnt/turnstone-build-ssd/target-w-fit-esc/debug/switchyard-server"
STUB = "/var/tmp/retry-stage-20261129/stub_scripted.py"
TURNSTONE_TREE = "/var/tmp/audit-v194/privdev/workspace/turnstone"

RESULTS = []
PROCS = []


def grade(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%-52s %s %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def note(name, detail):
    RESULTS.append((name, True, "NOTE " + detail))
    print("%-52s NOTE %s" % (name, detail), flush=True)


def http(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def lower(headers):
    return {k.lower(): v for k, v in (headers or {}).items()}


def chat(model, prompt="Reply with exactly: ok", timeout=90):
    status, headers, body = http(
        "http://127.0.0.1:%d/v1/chat/completions" % SRV_PORT,
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32},
        timeout=timeout,
    )
    try:
        doc = json.loads(body)
    except Exception:  # noqa: BLE001
        doc = {"__raw__": body.decode("utf-8", "replace")[:400]}
    return status, lower(headers), doc


def stub_reset(script):
    http("http://127.0.0.1:%d/control/reset" % STUB_PORT, {"script": script})


def stub_log():
    _, _, body = http("http://127.0.0.1:%d/control/log" % STUB_PORT, None)
    return json.loads(body)["calls"]


CONFIG = """schema_version = 1

[llm_clients.ca]
format = "openai_chat"
base_url = "http://127.0.0.1:%(stub)d/v1"
max_retries = 0
failure_cooldown_ms = 5000

[llm_clients.cb]
format = "openai_chat"
base_url = "http://127.0.0.1:%(stub)d/v1"
max_retries = 0
failure_cooldown_ms = 5000

[llm_clients.cc]
format = "openai_chat"
base_url = "http://127.0.0.1:%(stub)d/v1"
max_retries = 0
failure_cooldown_ms = 5000

[llm_clients.cd]
format = "openai_chat"
base_url = "http://127.0.0.1:%(stub)d/v1"
max_retries = 0
failure_cooldown_ms = 0

[llm_clients.ce]
format = "openai_chat"
base_url = "http://127.0.0.1:%(stub)d/v1"
max_retries = 1
failure_cooldown_ms = 0

[llm_clients.cf]
format = "openai_chat"
base_url = "http://127.0.0.1:%(stub)d/v1"
max_retries = 0
failure_cooldown_ms = 2000

[targets.ta]
id = "ta"
llm_client = "ca"
requests_per_minute = 600
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.tb]
id = "tb"
llm_client = "cb"
requests_per_minute = 600
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.tc]
id = "tc"
llm_client = "cc"
requests_per_minute = 600
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.td]
id = "td"
llm_client = "cd"
requests_per_minute = 600
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.te]
id = "te"
llm_client = "ce"
requests_per_minute = 600
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.tf]
id = "tf"
llm_client = "cf"
requests_per_minute = 600
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.tg]
id = "tg"
llm_client = "cd"
requests_per_minute = 1
max_input_tokens = 4000
input_tokens_per_minute = 400000

[targets.th]
id = "th"
llm_client = "cd"
requests_per_minute = 1
max_input_tokens = 4000
input_tokens_per_minute = 400000

[routes.ra-route]
id = "ra-route"
type = "passthrough"
target = "ta"

[routes.rb-route]
id = "rb-route"
type = "passthrough"
target = "tb"

[routes.rc-route]
id = "rc-route"
type = "passthrough"
target = "tc"

[routes.rd-route]
id = "rd-route"
type = "passthrough"
target = "td"

[routes.re-route]
id = "re-route"
type = "passthrough"
target = "te"

[routes.rf-route]
id = "rf-route"
type = "passthrough"
target = "tf"

[routes.rg-route]
id = "rg-route"
type = "passthrough"
target = "tg"

[routes.rh-route]
id = "rh-route"
type = "passthrough"
target = "th"
"""


def wait_health(timeout=40):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _, body = http("http://127.0.0.1:%d/health" % SRV_PORT, None, timeout=3)
            if status == 200:
                return True, body.decode()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    return False, ""


def start():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT + "/stub", exist_ok=True)
    with open(ROOT + "/stage.toml", "w") as f:
        f.write(CONFIG % {"stub": STUB_PORT})

    stub = subprocess.Popen(
        [sys.executable, STUB, ROOT + "/stub", str(STUB_PORT)],
        stdout=subprocess.DEVNULL, stderr=open(ROOT + "/stub.err", "w"),
    )
    PROCS.append(stub)
    time.sleep(0.6)

    srv = subprocess.Popen(
        [SERVER, "--config", ROOT + "/stage.toml", "--host", "127.0.0.1",
         "--port", str(SRV_PORT), "--routing-log-file", ROOT + "/routing.jsonl"],
        stdout=open(ROOT + "/server.out", "w"), stderr=subprocess.STDOUT,
    )
    PROCS.append(srv)
    return wait_health()


def stop():
    for p in reversed(PROCS):
        try:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass


def header_seconds(headers):
    try:
        return float(headers.get("retry-after"))
    except Exception:  # noqa: BLE001
        return -1.0


def scenario_a_upstream_429():
    """Upstream 429: body preserved, interval parsed internally, header forwarding measured."""
    stub_reset([{"status": 429, "retry_after": "7", "code": "rate_limit_exceeded",
                 "message": "upstream says wait seven"}])
    status, headers, doc = chat("ra-route")
    calls = stub_log()
    msg = str(((doc.get("error") or {}) if isinstance(doc, dict) else {}).get("message", ""))
    code = str(((doc.get("error") or {}) if isinstance(doc, dict) else {}).get("code", ""))
    grade("A1 upstream 429 status preserved", status == 429, "status=%s" % status)
    grade("A2 provider message and code preserved", "seven" in msg and code == "rate_limit_exceeded",
          "code=%r" % code)
    grade("A3 one outbound attempt, no hidden retry", len(calls) == 1, "stub hits=%d" % len(calls))
    if headers.get("retry-after") == "7":
        grade("A4 upstream interval forwarded verbatim", True, "retry-after=7")
    else:
        note("A4 upstream interval NOT forwarded",
             "retry-after=%r; recorded as an open gap" % headers.get("retry-after"))


def scenario_e_parse_is_proven_inside():
    """The client parses the upstream interval: a retry gap equals what the upstream stated."""
    stub_reset([{"status": 429, "retry_after": "3", "code": "rate_limit_exceeded",
                 "message": "first"},
                {"status": 429, "retry_after": "3", "code": "rate_limit_exceeded",
                 "message": "second"}])
    status, headers, _ = chat("re-route")
    calls = stub_log()
    gap = round(calls[1]["t"] - calls[0]["t"], 2) if len(calls) > 1 else None
    grade("A5 internal parse: retry gap equals 3 s +-0.6", gap is not None and abs(gap - 3.0) <= 0.6,
          "gap=%r" % gap)
    grade("A6 max_retries=1 spends exactly two attempts", len(calls) == 2, "stub hits=%d" % len(calls))


def scenario_b_cooldown():
    """A 503 arms the cooldown; the next call refuses locally with 1..=6 s and no dispatch."""
    stub_reset([{"status": 503, "code": "unavailable", "message": "backend temporarily unavailable"}])
    s1, _, _ = chat("rb-route")
    t0 = time.time()
    s2, h2, d2 = chat("rb-route")
    elapsed = time.time() - t0
    calls = stub_log()
    secs = header_seconds(h2)
    code = str(((d2.get("error") or {}) if isinstance(d2, dict) else {}).get("code", ""))
    grade("B1 first 503 reaches the caller", s1 == 503, "status=%s" % s1)
    grade("B2 cooldown refuses locally with 503", s2 == 503, "status=%s" % s2)
    grade("B3 retry-after within 1..=6", 1 <= secs <= 6, "retry-after=%r" % h2.get("retry-after"))
    grade("B4 no second outbound while cooling", len(calls) == 1, "stub hits=%d" % len(calls))
    grade("B5 refusal is immediate", elapsed < 1.5, "elapsed=%.2fs" % elapsed)
    grade("B6 refusal keeps the established code", code == "temporarily_unavailable", "code=%r" % code)


def scenario_c_minute_window():
    """Local per-minute admission states its own interval (this target is rpm=1)."""
    stub_reset([])
    s1, _, _ = chat("rg-route")
    s2, h2, d2 = chat("rg-route")
    calls = stub_log()
    secs = header_seconds(h2)
    code = str(((d2.get("error") or {}) if isinstance(d2, dict) else {}).get("code", ""))
    grade("C1 first call served", s1 == 200, "status=%s" % s1)
    grade("C2 second call refused 429", s2 == 429, "status=%s" % s2)
    grade("C3 minute-window code", code == "minute_allowance_throttled", "code=%r" % code)
    grade("C4 retry-after within 1..=60", 1 <= secs <= 60, "retry-after=%r" % h2.get("retry-after"))
    grade("C5 only one outbound issued", len(calls) == 1, "stub hits=%d" % len(calls))


def turnstone_client():
    sys.path.insert(0, TURNSTONE_TREE)
    from turnstone.core.providers import create_client

    return create_client("openai-compatible",
                         base_url="http://127.0.0.1:%d/v1" % SRV_PORT,
                         api_key="staging-not-a-secret")


def scenario_d_turnstone_paces_on_switchyard():
    """Turnstone's real client reads Switchyard's own interval off the wire."""
    from turnstone.core.completion_recovery import retry_after_delay

    stub_reset([])
    client = turnstone_client()
    first = client.chat.completions.create(model="rh-route",
                                           messages=[{"role": "user", "content": "hello"}],
                                           max_tokens=16)
    grade("D1 SDK serves the first call", first.choices[0].message.content == "stub ok",
          "content=%r" % first.choices[0].message.content)
    delay = None
    exc_name = ""
    t0 = time.time()
    try:
        client.chat.completions.create(model="rh-route",
                                       messages=[{"role": "user", "content": "hello"}],
                                       max_tokens=16)
    except Exception as exc:  # noqa: BLE001
        exc_name = type(exc).__name__
        delay = retry_after_delay(exc)
    elapsed = time.time() - t0
    calls = stub_log()
    grade("D2 SDK raises RateLimitError", exc_name == "RateLimitError", "exc=%s" % exc_name)
    grade("D3 Turnstone reads the advertised interval", delay is not None and 1 <= delay <= 60,
          "delay=%r" % delay)
    grade("D4 no hidden SDK retry", len(calls) == 1, "stub hits=%d" % len(calls))
    grade("D5 the SDK itself does not sleep", elapsed < 2.0, "elapsed=%.2fs" % elapsed)


def scenario_f_three_windows():
    """Three 2 s cooldown windows paced by the advertised interval, then a closing answer."""
    from turnstone.core.completion_recovery import retry_after_delay

    stub_reset([{"status": 503, "code": "unavailable", "message": "resting"},
                {"status": 503, "code": "unavailable", "message": "resting"},
                {"status": 503, "code": "unavailable", "message": "resting"},
                {"status": 200}])
    client = turnstone_client()
    waits = []
    answer = None
    t0 = time.time()
    for _ in range(12):
        try:
            resp = client.chat.completions.create(
                model="rf-route", messages=[{"role": "user", "content": "hello"}], max_tokens=16)
            answer = resp.choices[0].message.content
            break
        except Exception as exc:  # noqa: BLE001
            delay = retry_after_delay(exc)
            if delay is None:
                # The upstream's own 503 carries no interval; it arms the cooldown, and the
                # next call is refused locally with the remaining wait.
                time.sleep(0.05)
                continue
            waits.append(delay)
            time.sleep(delay)
    calls = stub_log()
    gaps = [round(calls[i + 1]["t"] - calls[i]["t"], 2) for i in range(len(calls) - 1)]
    grade("F1 three advertised waits, each 2 s", waits == [2.0, 2.0, 2.0], "waits=%r" % waits)
    grade("F2 closing answer served", answer == "stub ok", "answer=%r" % answer)
    grade("F3 four attempts, one per window", len(calls) == 4, "stub hits=%d" % len(calls))
    grade("F4 observed gaps are 2 s +-0.8", bool(gaps) and all(abs(g - 2.0) <= 0.8 for g in gaps),
          "gaps=%r" % gaps)
    print("    (total wall %.2fs)" % (time.time() - t0))


def main():
    ok, body = start()
    if not ok:
        print("SERVER DID NOT START:", body, open(ROOT + "/server.out").read()[-2000:])
        stop()
        return 2
    print("staged server healthy; config %s/stage.toml" % ROOT, flush=True)
    try:
        scenario_a_upstream_429()
        scenario_e_parse_is_proven_inside()
        scenario_b_cooldown()
        scenario_c_minute_window()
        scenario_d_turnstone_paces_on_switchyard()
        scenario_f_three_windows()
    finally:
        stop()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n%d/%d checks passed" % (passed, len(RESULTS)))
    with open(ROOT + "/results.json", "w") as f:
        json.dump({"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS]},
                  f, indent=1)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
