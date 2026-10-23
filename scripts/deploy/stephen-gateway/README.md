# Authenticated ingress gateway

A loopback proxy that puts per-client authentication, per-client route grants and
per-client paid spending limits in front of Switchyard. It exists because
Switchyard itself authenticates the caller's *provider* credential, not the
caller: any peer that can reach the listener can name any route.

```
client --bearer--> :4010 gateway :4010 --> 127.0.0.1:4000 Switchyard --> provider
```

The gateway is published only where an authenticated surface is wanted. In this
deployment that is Tailscale Serve on `:8443`; the unrestricted `:443` path is
left exactly as it was.

## What it enforces

| Property | How |
|---|---|
| Caller identity | bearer token, compared by SHA-256 digest with `hmac.compare_digest` |
| Route access | per-client `allow` list; an unlisted model is refused before dispatch |
| Paid access | `ST_Paid` additionally needs a positive ceiling |
| Cost bound | worst-case cost reserved **before** dispatch, settled after |
| Interruption | a stream that dies, or a client that walks away, is still charged |
| Restart | outstanding reservations become spend, never a refund |
| Discovery | granted composite routes appear in `/v1/models` for that client |
| Request size | `max_request_bytes`, refused before the body is read |
| Streaming | relayed with `read1`, so SSE stays incremental |
| Secret hygiene | the caller's `Authorization` is never forwarded; logs carry method and path only |
| Failure | unreadable config, unknown token, unknown path, oversized body all deny |

The spending ceiling is a reservation, not a running total. Checking a budget
before a request and charging after it is a race: concurrent requests all see the
same remaining budget. Reserving under the lock that reads the budget means the
sum of reservations plus settled spend cannot exceed the ceiling. A streamed paid
call is charged at its reservation because its usage arrives inside the stream
that has already been relayed — over rather than under.

## Configuration

Root-owned, `0640`, reloaded per request so adding or removing a client never
needs a restart. Only digests are stored; the plaintext tokens belong in a
separate `0600` file.

```json
{
  "clients": [
    {"name": "example",
     "token_sha256": "<hex>",
     "allow": ["ST_Tier1", "ST_Tier2", "ST_Tier3", "ST_Route"],
     "monthly_spend_ceiling_usd": 0}
  ],
  "pricing": {"z-ai/glm-5.3-flash": [0.225, 2.0],
              "deepseek/deepseek-v4.1-flash": [0.6, 2.4]},
  "max_request_bytes": 4194304,
  "upstream_timeout_seconds": 600
}
```

Pricing is USD per **million** tokens, using the dearest published endpoint of
each model, so routing to a more expensive provider cannot quietly exceed a
ceiling. A response that reports its own `usage.cost` is charged at that figure.

## Running

```bash
python3 gateway.py            # binds 127.0.0.1:4010, refuses any other address
python3 test_gateway.py       # 30 checks against a mock upstream, no credentials
```

`TestGateway` refuses to bind a non-loopback address, which is deliberate: this
component is designed to sit behind Serve, not to listen on a network.

## Tests worth knowing about

- **Concurrency**: twelve simultaneous paid requests against a ceiling that
  affords four must accept exactly four. Replace the reservation with a
  check-then-charge and the same test accepts twelve, spending 2.4× the ceiling.
- **Streaming**: the first chunk must arrive before the stream ends. Buffer the
  relay and this fails.
- **Secrecy**: no caller token may appear upstream or in the log.
- **Charging**: the served model's reported cost is preferred over the rate table;
  an absurd report is capped at the reservation rather than trusted.
- **Interruption**: a stream cut mid-flight and a client that disconnects are both
  charged, and neither leaves a dangling reservation.
- **Restart**: a reservation left behind is converted to spend exactly once.
- **Discovery**: a granted composite is visible and marked `composite`, with no
  invented capability facts; a route the client was not granted never appears.
