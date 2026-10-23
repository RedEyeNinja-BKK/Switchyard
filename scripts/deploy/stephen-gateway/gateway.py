#!/usr/bin/env python3
"""Authenticated ingress gateway for the ST routes.

One listen address, loopback only. Every request must carry a bearer token whose
SHA-256 digest matches exactly one configured client; that client's entry decides
which routes it may call and how much money it may spend.

Configuration (root-owned, 0600), re-read per request so adding or removing an
Agent Stack never needs a restart:

  {
    "clients": [
      {"name": "stephen-openclaw",
       "token_sha256": "<hex>",
       "allow": ["ST_Tier1", "ST_Tier2", "ST_Tier3", "ST_Route",
                 "ST_Rerank", "ST_Embed"],
       "monthly_spend_ceiling_usd": 0},
      {"name": "operator-test",
       "token_sha256": "<hex>",
       "allow": ["ST_Paid", "ST_Tier1"],
       "monthly_spend_ceiling_usd": 0.25}
    ],
    "pricing": {"z-ai/glm-5.3-flash": [0.225, 2.0],
                "deepseek/deepseek-v4.1-flash": [0.6, 2.4]},
    "max_request_bytes": 4194304,
    "upstream_timeout_seconds": 600
  }

Pricing is the WORST-CASE published rate across every endpoint of a model, in USD
per million tokens, so a ceiling cannot be quietly exceeded by the provider
routing a request to a dearer endpoint.

Accounting model
----------------
A ceiling checked before dispatch and charged afterwards is a race: concurrent
requests all observe the same remaining budget. Each paid request therefore
creates a RESERVATION -- its worst-case cost -- under the same lock that reads
the budget, before dispatch. Outstanding reservations plus settled spend can
never exceed the ceiling, so the ceiling holds under concurrency.

The reservation is then settled:

  * buffered 2xx with a usable usage report -> charged the reported cost, which
    is authoritative; capped at the reservation so a malformed or absurd report
    cannot push the ledger past what was reserved;
  * streamed 2xx -> charged its reservation, because usage arrives inside the
    stream that has already been relayed. Where the stream does report usage, the
    ledger is corrected down to it;
  * 2xx that produced no usage report, a stream the client abandoned, or an
    upstream failure AFTER a 2xx began -> still charged its reservation. The
    provider may have generated tokens; guessing that it did not would be the
    one error a ceiling must never make;
  * a request that never got a 2xx -> released untouched, because a refused call
    is not billable.

A reservation can therefore outlive the process. On startup every outstanding
reservation is converted to spend and reported, so a crash or restart cannot
silently forgive a billable request.

Reproducing verification
------------------------
    python3 test_gateway.py

Fail-closed rules
-----------------
* unreadable or malformed config denies;
* unknown, absent or non-Bearer credentials deny;
* a path outside the exposed set denies;
* a model outside the client's allow-list denies;
* ST_Paid additionally needs a positive ceiling, a declared output bound and an
  affordable reservation;
* an oversized body is refused before it is read;
* the caller's Authorization header is never forwarded upstream.
"""
from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import math
import os
import pathlib
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ("127.0.0.1", 4000)
LISTEN = ("127.0.0.1", 4010)
CONFIG = pathlib.Path(
    os.environ.get("STEPHEN_GATEWAY_CONFIG", "/etc/switchyard/stephen-gateway.json")
)
STATE = pathlib.Path(
    os.environ.get("STEPHEN_GATEWAY_STATE", "/var/lib/switchyard/stephen-gateway-state.json")
)

ALLOWED_PATHS = {"/v1/responses", "/v1/embeddings", "/v1/rerank", "/v1/models", "/health"}
MODEL_PATHS = {"/v1/responses", "/v1/embeddings", "/v1/rerank"}
PAID_ROUTE = "ST_Paid"

FORWARD_REQUEST_HEADERS = ("content-type", "accept", "openai-beta", "idempotency-key")
STRIP_RESPONSE_HEADERS = {
    "connection", "keep-alive", "transfer-encoding", "content-length",
    "upgrade", "server", "date",
}

DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_UPSTREAM_TIMEOUT = 600

# How much of a stream's tail is kept in order to find a usage report. Bounded so a
# long stream cannot grow memory, and large enough for the closing events of an SSE
# response.
STREAM_TAIL_BYTES = 64 * 1024

# Rounding of currency in the ledger. Anything finer is noise from float arithmetic.
LEDGER_QUANTUM = 8

_lock = threading.Lock()


class GatewayRefused(Exception):
    """A request the gateway will not forward."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------- config
def load_config() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def client_for(token: str, cfg: dict) -> dict | None:
    digest = hashlib.sha256(token.encode()).hexdigest()
    for client in cfg.get("clients", []):
        if isinstance(client, dict) and hmac.compare_digest(
            digest, str(client.get("token_sha256", ""))
        ):
            return client
    return None


# ---------------------------------------------------------------------- money
def finite_non_negative(value) -> float | None:
    """`value` as a cost or token count, or None when it cannot be trusted.

    `bool` is rejected before the numeric branch because Python makes it an int
    subclass, so `True` would otherwise read as one token.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def cost_usd(prices, input_tokens: float, output_tokens: float) -> float:
    """Cost in USD. Prices are USD per million tokens."""
    prompt, completion = float(prices[0]), float(prices[1])
    return (input_tokens / 1_000_000) * prompt + (output_tokens / 1_000_000) * completion


def worst_paid_prices(cfg: dict):
    """The dearest configured paid model, as (prompt, completion) per million tokens.

    A route may pick either paid target and the gateway sees only the route name,
    so a reservation has to cover the dearer of them.
    """
    priced = []
    for value in (cfg.get("pricing") or {}).values():
        try:
            priced.append((float(value[0]), float(value[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if not priced:
        return None
    return (max(p[0] for p in priced), max(p[1] for p in priced))


def usage_tokens(usage: dict) -> tuple[float, float] | None:
    """Input and output token counts from a usage report, or None if unusable."""
    if not isinstance(usage, dict):
        return None
    prompt = finite_non_negative(usage.get("input_tokens"))
    if prompt is None:
        prompt = finite_non_negative(usage.get("prompt_tokens"))
    completion = finite_non_negative(usage.get("output_tokens"))
    if completion is None:
        completion = finite_non_negative(usage.get("completion_tokens"))
    if prompt is None or completion is None:
        return None
    return prompt, completion


def settle_amount(usage: dict | None, prices, reservation: float) -> float:
    """What one paid request should be charged, never more than it reserved.

    The reservation is this request's worst case. A report above it means the
    gateway's bound was wrong or the report is malformed; charging the bound keeps
    the ledger honest without letting a bad number overrun the ceiling.
    """
    if reservation <= 0:
        return 0.0
    reported = finite_non_negative((usage or {}).get("cost"))
    if reported is not None:
        return min(reported, reservation)
    tokens = usage_tokens(usage or {})
    if tokens is not None and prices:
        computed = cost_usd(prices, tokens[0], tokens[1])
        return min(computed, reservation)
    # No usable report: the reservation stands.
    return reservation


def usage_from_sse(tail: bytes) -> dict | None:
    """The last usage report in a captured SSE tail, if the stream carried one."""
    if not tail:
        return None
    for line in reversed(tail.split(b"\n")):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        for candidate in (event.get("usage"), (event.get("response") or {}).get("usage")):
            if isinstance(candidate, dict):
                return candidate
    return None


# ---------------------------------------------------------------------- state
def load_state() -> dict:
    try:
        with open(STATE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=STATE.parent, prefix=".gwstate-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(path, STATE)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def client_row(state: dict, name: str) -> dict:
    row = state.get(name)
    if not isinstance(row, dict):
        row = {}
        state[name] = row
    row.setdefault("spend_usd", 0.0)
    row.setdefault("reserved_usd", 0.0)
    row.setdefault("requests", 0)
    reservations = row.get("reservations")
    if not isinstance(reservations, dict):
        row["reservations"] = {}
    return row


def reconcile_outstanding(announce=None) -> int:
    """Convert reservations left by a previous process into settled spend.

    A reservation exists because a request was authorised to spend. If the process
    died before settling it, the provider may well have billed it, so forgiving the
    amount would understate what was spent and quietly widen the ceiling. Converting
    is the conservative direction and is idempotent: the reservation list is
    consumed, so a second start has nothing to convert.
    """
    converted = 0
    total = 0.0
    with _lock:
        state = load_state()
        changed = False
        for name, row in list(state.items()):
            if not isinstance(row, dict):
                continue
            row = client_row(state, name)
            reservations = row.get("reservations") or {}
            if not reservations:
                if float(row.get("reserved_usd") or 0) != 0:
                    # Nothing outstanding, so any residual is drift; clear it rather
                    # than reserving budget forever against nothing.
                    row["reserved_usd"] = 0.0
                    changed = True
                continue
            for reservation in reservations.values():
                amount = finite_non_negative(
                    (reservation or {}).get("amount") if isinstance(reservation, dict) else 0
                )
                if amount:
                    row["spend_usd"] = round(
                        float(row.get("spend_usd") or 0) + amount, LEDGER_QUANTUM
                    )
                    total += amount
                    converted += 1
            row["reservations"] = {}
            row["reserved_usd"] = 0.0
            changed = True
        if changed:
            save_state(state)
    if converted and announce:
        announce(
            f"reconciled {converted} outstanding paid reservation(s) totalling "
            f"${total:.8f} into settled spend"
        )
    return converted


class Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "stephen-gateway/5"
    sys_version = ""

    # A hung peer must not hold a thread forever.
    timeout = 900

    def log_message(self, fmt, *args):
        # Method and path only. Never the Authorization header, never a body.
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

    # ---------------------------------------------------------------- responses
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny(self, code: int, message: str) -> None:
        self._json(code, {"error": {"message": message, "type": "gateway_refused"}})

    # ------------------------------------------------------------ authn / config
    def _config(self) -> dict:
        try:
            return load_config()
        except (OSError, ValueError) as error:
            raise GatewayRefused(
                503, f"gateway policy unavailable: {type(error).__name__}"
            ) from None

    def _client(self, cfg: dict) -> dict:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise GatewayRefused(401, "missing or invalid bearer token")
        token = header[7:].strip()
        if not token:
            raise GatewayRefused(401, "missing or invalid bearer token")
        client = client_for(token, cfg)
        if client is None:
            raise GatewayRefused(401, "missing or invalid bearer token")
        return client

    # --------------------------------------------------------------- accounting
    def _reserve(self, client: dict, amount: float, model: str) -> str | None:
        """Reserve `amount` for this request if the ceiling allows it."""
        reservation_id = uuid.uuid4().hex
        with _lock:
            ceiling = float(client.get("monthly_spend_ceiling_usd") or 0)
            state = load_state()
            row = client_row(state, client["name"])
            settled = float(row.get("spend_usd") or 0)
            reserved = float(row.get("reserved_usd") or 0)
            if settled + reserved + amount > ceiling:
                return None
            row["reservations"][reservation_id] = {
                "amount": round(amount, LEDGER_QUANTUM),
                "model": model,
                "opened_at": time.time(),
            }
            row["reserved_usd"] = round(reserved + amount, LEDGER_QUANTUM)
            save_state(state)
            return reservation_id

    def _settle(self, client: dict, reservation_id: str, charge: float | None) -> None:
        """Close a reservation, charging `charge` (or nothing when None)."""
        with _lock:
            state = load_state()
            row = client_row(state, client["name"])
            reservation = row["reservations"].pop(reservation_id, None)
            if reservation is None:
                # Already settled, or never existed. Charging again would
                # double-count the request against the ceiling.
                return
            if isinstance(reservation, dict):
                amount = finite_non_negative(reservation.get("amount")) or 0.0
                row["reserved_usd"] = round(
                    max(0.0, float(row.get("reserved_usd") or 0) - amount), LEDGER_QUANTUM
                )
            if charge is not None:
                row["spend_usd"] = round(
                    float(row.get("spend_usd") or 0) + charge, LEDGER_QUANTUM
                )
                row["requests"] = int(row.get("requests") or 0) + 1
            save_state(state)

    # ---------------------------------------------------------------- admission
    def _admit(self, client: dict, body: bytes, cfg: dict) -> tuple[str, str | None]:
        """Decide whether to forward. Returns (model, reservation_id)."""
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise GatewayRefused(400, "request body is not JSON") from None
        if not isinstance(payload, dict):
            raise GatewayRefused(400, "request body must be a JSON object")
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise GatewayRefused(400, "a model is required")
        if model not in set(client.get("allow") or []):
            raise GatewayRefused(403, "this client is not authorised for that route")
        if model != PAID_ROUTE:
            return model, None

        if float(client.get("monthly_spend_ceiling_usd") or 0) <= 0:
            raise GatewayRefused(403, "this client has no paid spending allowance")
        prices = worst_paid_prices(cfg)
        if prices is None:
            raise GatewayRefused(503, "no paid pricing is configured")
        declared = payload.get("max_output_tokens")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            # Without a bound the cost has no upper limit, so a hard ceiling would be
            # a claim the gateway cannot keep.
            raise GatewayRefused(
                403, "paid requests must declare max_output_tokens so the cost is bounded"
            )
        reservation = cost_usd(prices, len(body), int(declared))
        reservation_id = self._reserve(client, reservation, model)
        if reservation_id is None:
            raise GatewayRefused(403, "this client's spending ceiling is reached")
        return model, reservation_id

    # ------------------------------------------------------------------ proxying
    def _proxy(self, body: bytes, client: dict, model: str,
               reservation_id: str | None, cfg: dict) -> None:
        timeout = float(cfg.get("upstream_timeout_seconds") or DEFAULT_UPSTREAM_TIMEOUT)
        try:
            conn = http.client.HTTPConnection(*UPSTREAM, timeout=timeout)
            headers = {k: self.headers[k] for k in FORWARD_REQUEST_HEADERS if self.headers.get(k)}
            conn.request(self.command, self.path, body=body or None, headers=headers)
            upstream = conn.getresponse()
        except Exception as error:  # noqa: BLE001
            # Nothing was dispatched that the provider could bill.
            if reservation_id:
                self._settle(client, reservation_id, None)
            return self._deny(502, f"upstream unreachable: {type(error).__name__}")

        # From here the provider may have produced output, so a paid request stays
        # chargeable even if the relay below fails.
        billable = upstream.status < 400
        settled = False
        stream_tail = bytearray()
        try:
            self.send_response(upstream.status)
            declared = None
            for key, value in upstream.getheaders():
                lowered = key.lower()
                if lowered in STRIP_RESPONSE_HEADERS:
                    if lowered == "content-length":
                        declared = value
                    continue
                self.send_header(key, value)

            if declared is not None:
                data = upstream.read()
                self.send_header("Content-Length", declared)
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                if reservation_id:
                    usage: dict = {}
                    served = model
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, dict):
                            usage = parsed.get("usage") or {}
                            # A paid route may be served by either of its paid
                            # targets, and the response names which. Pricing the
                            # route name instead would charge the dearest target
                            # for a cheaper call.
                            named = parsed.get("model")
                            if isinstance(named, str) and named:
                                served = named
                    except ValueError:
                        usage = {}
                    self._settle_paid(client, reservation_id, served, cfg, usage, billable)
                    settled = True
            else:
                # No length: the upstream is streaming. Relay it as it arrives.
                # `read1` returns whatever is buffered instead of waiting for a
                # full block, which is what keeps SSE incremental rather than
                # batched into 2 KiB lumps.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = upstream.read1(2048)
                    if not chunk:
                        break
                    stream_tail.extend(chunk)
                    if len(stream_tail) > STREAM_TAIL_BYTES:
                        del stream_tail[: len(stream_tail) - STREAM_TAIL_BYTES]
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                if reservation_id:
                    self._settle_paid(client, reservation_id, model, cfg,
                                      usage_from_sse(bytes(stream_tail)), billable)
                    settled = True
        except Exception as error:  # noqa: BLE001
            # The relay broke: a truncated upstream, or a client that walked away.
            # Nothing more can be written to a half-delivered stream, so close the
            # connection and let the settlement below record what it may have cost.
            # One line, not a traceback: this is an operational event, not a defect.
            self.close_connection = True
            sys.stderr.write(
                f"{self.command} {self.path.split('?')[0]}: relay ended early "
                f"({type(error).__name__})\n")
        finally:
            conn.close()
            if reservation_id and not settled:
                try:
                    self._settle_paid(client, reservation_id, model, cfg, None, billable)
                except Exception as error:  # noqa: BLE001
                    # Never leave a reservation open because settlement itself
                    # failed: that would block the budget forever. Close it at its
                    # reserved amount, which is the conservative reading.
                    sys.stderr.write(
                        f"settlement failed ({type(error).__name__}: {error}); "
                        "closing at the reserved amount\n")
                    try:
                        amount = self._reservation_amount(client, reservation_id)
                        self._settle(client, reservation_id, amount if billable else None)
                    except Exception as fallback_error:  # noqa: BLE001
                        sys.stderr.write(
                            "closing the reservation failed: "
                            f"{type(fallback_error).__name__}\n")

    def _settle_paid(self, client: dict, reservation_id: str, model: str, cfg: dict,
                     usage: dict | None, billable: bool) -> None:
        """Close a paid reservation under the settlement rules."""
        if not billable:
            # A refused call is not billable.
            self._settle(client, reservation_id, None)
            return
        amount = self._reservation_amount(client, reservation_id)
        prices = (cfg.get("pricing") or {}).get(model)
        if prices is None:
            prices = worst_paid_prices(cfg)
        charge = settle_amount(usage, prices, amount)
        self._settle(client, reservation_id, charge)

    def _reservation_amount(self, client: dict, reservation_id: str) -> float:
        """The reserved amount, read without holding the lock across the settle."""
        state = load_state()
        row = state.get(client.get("name", ""))
        if not isinstance(row, dict):
            return 0.0
        reservation = (row.get("reservations") or {}).get(reservation_id)
        if not isinstance(reservation, dict):
            return 0.0
        return finite_non_negative(reservation.get("amount")) or 0.0

    def _models(self, client: dict) -> None:
        try:
            conn = http.client.HTTPConnection(*UPSTREAM, timeout=30)
            conn.request("GET", "/v1/models")
            payload = json.loads(conn.getresponse().read() or b"{}")
            conn.close()
        except Exception as error:  # noqa: BLE001
            return self._deny(502, f"upstream unreachable: {type(error).__name__}")

        allow = set(client.get("allow") or [])
        rows = [m for m in payload.get("data", [])
                if isinstance(m, dict) and m.get("id") in allow]

        # Switchyard lists leaf routes. A composite such as ST_Route is callable by
        # name but absent from that list, so a client granted one could not discover
        # it. Add exactly the granted names that the catalog omits, and claim no
        # capabilities for them: the grant says the route exists, but only Switchyard
        # knows what it can do, and inventing facts would be worse than omitting them.
        listed = {m.get("id") for m in rows}
        existing = payload.get("data", [])
        known = {m.get("id") for m in existing if isinstance(m, dict)}
        for name in sorted(allow - listed):
            if name not in known:
                rows.append({
                    "id": name,
                    "object": "model",
                    "type": "model",
                    "created": 0,
                    "owned_by": "switchyard",
                    "display_name": name,
                    "switchyard": {"kind": "composite"},
                })
        self._json(200, {"object": "list", "data": rows})

    # --------------------------------------------------------------------- verbs
    def _guard(self):
        """Shared admission: config, identity, path."""
        cfg = self._config()
        client = self._client(cfg)
        path = self.path.split("?")[0]
        if path not in ALLOWED_PATHS:
            raise GatewayRefused(403, "path not exposed by this gateway")
        return cfg, client, path

    def do_GET(self):
        try:
            cfg, client, path = self._guard()
        except GatewayRefused as refusal:
            return self._deny(refusal.status, refusal.message)
        if path == "/v1/models":
            return self._models(client)
        self._proxy(b"", client, "", None, cfg)

    def do_POST(self):
        try:
            cfg, client, path = self._guard()
            limit = int(cfg.get("max_request_bytes") or DEFAULT_MAX_REQUEST_BYTES)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise GatewayRefused(400, "invalid Content-Length") from None
            if length < 0:
                raise GatewayRefused(400, "invalid Content-Length")
            if length > limit:
                # Refused before reading: an oversized body cannot be used to
                # exhaust memory. The body is unread, so the connection closes.
                self.close_connection = True
                raise GatewayRefused(413, f"request body exceeds {limit} bytes")
            body = self.rfile.read(length) if length else b""
            model, reservation_id = "", None
            if path in MODEL_PATHS:
                model, reservation_id = self._admit(client, body, cfg)
        except GatewayRefused as refusal:
            return self._deny(refusal.status, refusal.message)
        self._proxy(body, client, model, reservation_id, cfg)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # Refuse to bind anything but loopback, whatever the caller configured.
        if self.server_address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise SystemExit("refusing to bind a non-loopback address")
        super().server_bind()


def main() -> int:
    if not CONFIG.exists():
        print(f"config not found: {CONFIG}", file=sys.stderr)
        return 1
    try:
        cfg = load_config()
    except (OSError, ValueError) as error:
        print(f"config unreadable: {error}", file=sys.stderr)
        return 1

    def announce(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    reconcile_outstanding(announce)

    server = Server(LISTEN, Gateway)
    print(f"listening on {LISTEN[0]}:{LISTEN[1]} -> {UPSTREAM[0]}:{UPSTREAM[1]}; "
          f"{len(cfg.get('clients', []))} client(s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
