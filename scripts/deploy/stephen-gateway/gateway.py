#!/usr/bin/env python3
"""Authenticated ingress gateway for the ST routes.

One listen address, loopback only. Every request must carry a bearer token whose
SHA-256 digest matches exactly one configured client; that client's entry decides
which routes it may call and how much money it may spend. Nothing about access is
implied by "having a token": two clients with different grants are different
principals.

Configuration (root-only, 0600), reloaded per request so adding or removing an
Agent Stack never needs a restart:

  {
    "clients": [
      {"name": "operator-test",
       "token_sha256": "<hex>",
       "allow": ["ST_Paid", "ST_Tier1"],
       "monthly_spend_ceiling_usd": 0.25},
      {"name": "stephen-openclaw",
       "token_sha256": "<hex>",
       "allow": ["ST_Tier1", "ST_Tier2", "ST_Tier3", "ST_Route",
                 "ST_Rerank", "ST_Embed"],
       "monthly_spend_ceiling_usd": 0}
    ],
    "pricing": {"z-ai/glm-5.3-flash": [0.225, 2.0],
                "deepseek/deepseek-v4.1-flash": [0.6, 2.4]},
    "max_request_bytes": 4194304,
    "upstream_timeout_seconds": 600
  }

Pricing is the WORST-CASE published rate across every endpoint of the model, in
USD per million tokens, so a ceiling cannot be quietly exceeded by OpenRouter
routing a request to a more expensive provider.

Why the arithmetic is a reservation rather than a running total
---------------------------------------------------------------
A ceiling checked before dispatch and charged after the response is a race: N
concurrent requests all see the same remaining budget and all proceed. Here the
worst-case cost of a request is reserved under the same lock that reads the
budget, before dispatch, and settled afterwards. Concurrent requests therefore
serialise on the reservation, and outstanding reservations plus settled spend
can never exceed the ceiling. Input tokens are bounded by request bytes (every
token is at least one byte), which needs no tokenizer and cannot underestimate.

A streamed paid call is charged at its reservation, because the usage report
arrives inside the stream that has already been relayed. That overcharges the
ledger rather than undercharging it, which is the safe direction for a ceiling;
buffered calls are settled to their reported usage.

Fail-closed rules
-----------------
* unreadable or malformed config denies (never allows);
* unknown, absent or non-Bearer credentials deny;
* a path outside the exposed set denies;
* a model outside the client's allow-list denies;
* ST_Paid additionally needs a positive ceiling, a declared output bound and an
  affordable reservation;
* an oversized body is refused before it is read into memory;
* the caller's Authorization header is never forwarded upstream.
"""
from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import pathlib
import sys
import tempfile
import threading
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

_lock = threading.Lock()


class GatewayRefused(Exception):
    """A request the gateway will not forward."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def load_config() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


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


def client_for(token: str, cfg: dict) -> dict | None:
    digest = hashlib.sha256(token.encode()).hexdigest()
    for client in cfg.get("clients", []):
        if hmac.compare_digest(digest, str(client.get("token_sha256", ""))):
            return client
    return None


def cost_usd(prices, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD. Prices are USD per million tokens."""
    prompt, completion = float(prices[0]), float(prices[1])
    return (input_tokens / 1_000_000) * prompt + (output_tokens / 1_000_000) * completion


def worst_paid_prices(cfg: dict):
    """The most expensive configured paid model, as (prompt, completion).

    A route may pick either paid target, and the gateway sees only the route
    name, so the reservation must cover the dearer of them.
    """
    priced = [tuple(float(x) for x in p) for p in (cfg.get("pricing") or {}).values()]
    if not priced:
        return None
    return (max(p[0] for p in priced), max(p[1] for p in priced))


class Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "stephen-gateway/4"
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
            # Fail closed: a gateway that cannot read its policy cannot authorize.
            raise GatewayRefused(503, f"gateway policy unavailable: {type(error).__name__}")

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
    def _reserve(self, client: dict, amount: float) -> bool:
        """Atomically reserve `amount` if it fits under the ceiling."""
        with _lock:
            ceiling = float(client.get("monthly_spend_ceiling_usd") or 0)
            state = load_state()
            row = state.setdefault(client["name"], {})
            settled = float(row.get("spend_usd", 0.0))
            reserved = float(row.get("reserved_usd", 0.0))
            if settled + reserved + amount > ceiling:
                return False
            row["reserved_usd"] = round(reserved + amount, 8)
            save_state(state)
            return True

    def _apply(self, client: dict, reservation: float, charge: float | None) -> None:
        """Replace this request's reservation with its charge.

        `charge=None` releases the reservation without charging, for a request
        that produced no billable response.
        """
        with _lock:
            state = load_state()
            row = state.setdefault(client["name"], {})
            reserved = float(row.get("reserved_usd", 0.0))
            row["reserved_usd"] = round(max(0.0, reserved - reservation), 8)
            if charge is not None:
                row["spend_usd"] = round(float(row.get("spend_usd", 0.0)) + charge, 8)
                row["requests"] = int(row.get("requests", 0)) + 1
            save_state(state)

    # ---------------------------------------------------------------- admission
    def _admit(self, client: dict, body: bytes, cfg: dict) -> tuple[str, float]:
        """Decide whether to forward, and reserve money if it costs any.

        Returns (model, reservation). Raises GatewayRefused otherwise.
        """
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
            return model, 0.0

        if float(client.get("monthly_spend_ceiling_usd") or 0) <= 0:
            raise GatewayRefused(403, "this client has no paid spending allowance")
        prices = worst_paid_prices(cfg)
        if prices is None:
            raise GatewayRefused(503, "no paid pricing is configured")
        declared = payload.get("max_output_tokens")
        if not isinstance(declared, int) or declared <= 0:
            raise GatewayRefused(
                403, "paid requests must declare max_output_tokens so the cost is bounded"
            )
        reservation = cost_usd(prices, len(body), int(declared))
        if not self._reserve(client, reservation):
            raise GatewayRefused(403, "this client's spending ceiling is reached")
        return model, reservation

    # ------------------------------------------------------------------ proxying
    def _proxy(self, body: bytes, client: dict, model: str,
               reservation: float, cfg: dict) -> None:
        timeout = float(cfg.get("upstream_timeout_seconds") or DEFAULT_UPSTREAM_TIMEOUT)
        try:
            conn = http.client.HTTPConnection(*UPSTREAM, timeout=timeout)
            headers = {k: self.headers[k] for k in FORWARD_REQUEST_HEADERS if self.headers.get(k)}
            conn.request(self.command, self.path, body=body or None, headers=headers)
            upstream = conn.getresponse()
        except Exception as error:  # noqa: BLE001
            self._apply(client, reservation, None)
            return self._deny(502, f"upstream unreachable: {type(error).__name__}")

        billed = False
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
                if upstream.status == 200:
                    usage = {}
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, dict):
                            usage = parsed.get("usage") or {}
                    except ValueError:
                        usage = {}
                    # Price the model that actually served, which the response
                    # names, rather than the route that was asked for. Charging a
                    # paid route at its reservation would overstate a small call
                    # by an order of magnitude and exhaust a legitimate ceiling
                    # long before the money was spent.
                    served = parsed.get("model") if isinstance(parsed, dict) else None
                    pricing = cfg.get("pricing") or {}
                    prices = pricing.get(served) or pricing.get(model)
                    reported = usage.get("cost") if isinstance(usage, dict) else None
                    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
                        # The provider states what it actually charged. That is
                        # authoritative; the configured rates are only for the
                        # pre-dispatch reservation and for responses that omit it.
                        charge = max(0.0, float(reported))
                    elif prices and isinstance(usage, dict):
                        charge = cost_usd(
                            prices,
                            int(float(usage.get("input_tokens")
                                      or usage.get("prompt_tokens") or 0)),
                            int(float(usage.get("output_tokens")
                                      or usage.get("completion_tokens") or 0)),
                        )
                    elif reservation:
                        charge = reservation
                    else:
                        charge = 0.0
                    self._apply(client, reservation, charge)
                    billed = True
            else:
                # No length: the upstream is streaming. Relay as it arrives.
                # `read1` returns what is buffered instead of waiting for a full
                # block, which is what keeps SSE incremental rather than batched
                # into 2 KiB lumps.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = upstream.read1(2048)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                # Usage arrives inside the stream that has already been relayed,
                # so a streamed paid call is charged at its reservation: over
                # rather than under.
                if reservation:
                    self._apply(client, reservation, reservation)
                    billed = True
        finally:
            conn.close()
            if not billed:
                self._apply(client, reservation, None)

    def _models(self, client: dict) -> None:
        try:
            conn = http.client.HTTPConnection(*UPSTREAM, timeout=30)
            conn.request("GET", "/v1/models")
            payload = json.loads(conn.getresponse().read() or b"{}")
            conn.close()
        except Exception as error:  # noqa: BLE001
            return self._deny(502, f"upstream unreachable: {type(error).__name__}")
        allow = set(client.get("allow") or [])
        rows = [m for m in payload.get("data", []) if m.get("id") in allow]
        self._json(200, {"object": "list", "data": rows})

    # --------------------------------------------------------------------- verbs
    def _guard(self):
        """Shared admission: config, identity, path. Returns (cfg, client, path)."""
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
        self._proxy(b"", client, "", 0.0, cfg)

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
            model, reservation = "", 0.0
            if path in MODEL_PATHS:
                model, reservation = self._admit(client, body, cfg)
        except GatewayRefused as refusal:
            return self._deny(refusal.status, refusal.message)
        self._proxy(body, client, model, reservation, cfg)


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
