#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 RedEyeNinja
# SPDX-License-Identifier: Apache-2.0
"""Produce a redacted, publishable projection of a Switchyard deployment manifest.

The real manifest carries private deployment topology: private-network hostnames and the
addresses and ports of internal services. It carries no credential -- the file names
environment variables, it never contains their values -- but the topology does not belong in
a public repository, so the reviewable copy is generated rather than committed by hand.

Only endpoint coordinates are rewritten:

* ``<label>.<private-net>.ts.net`` -> ``<label>.tailnet-example.ts.net``
* any IPv4 that is not globally routable -> ``192.0.2.10`` (RFC 5737 TEST-NET-1)
* any remaining non-public host -> ``host-<n>.example.com`` (RFC 2606)
* ``/home/<user>``, ``/Users/<user>`` and ``/mnt/<path>`` -> ``<HOST_LOCAL_PATH>``

Public provider endpoints (``openrouter.ai`` and friends) are kept verbatim: which provider a
lane calls is the point of the document and identifies nothing private. Everything else --
tables, key order, every value, every comment -- is left byte-identical, so the redacted copy
diffs against the real manifest cleanly and can be checked against the published digest.

Four properties are enforced rather than assumed, because a redactor that silently drops a
setting, misses one address, or rewrites non-deterministically is worse than none:

1. **Structure preserved** -- the output parses, and its set of table paths and keys equals
   the input's. Nothing lost, nothing invented.
2. **Nothing private survives** -- no private-network hostname, no non-placeholder address,
   no home or mount path, no credential shape.
3. **Fixed point** -- re-running the redactor on its own output changes nothing, which is the
   only proof that the output contains no real coordinate left to rewrite.
4. **Deterministic** -- repeated runs produce identical bytes, so the published file can be
   regenerated and compared.

Usage:
    sanitize_manifest.py <real.toml> <out.toml> [--print-mapping]

Exit status: 0 written and verified, 1 a property failed (nothing written), 2 usage.
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("needs python >= 3.11 for tomllib")

# Reserved ranges only. Nothing here can resolve to a real service, so a redacted manifest
# cannot be deployed by mistake and cannot leak the real target.
PLACEHOLDER_IP = "192.0.2.10"                      # RFC 5737 TEST-NET-1
PLACEHOLDER_TAILNET = "tailnet-example.ts.net"     # matches tests/test_no_leaked_identifiers.py
PLACEHOLDER_DOMAIN = "example.com"                 # RFC 2606
PLACEHOLDER_PATH = "<HOST_LOCAL_PATH>"

# Prefixed to the published copy. The file sits next to the deploy commands, so it has to say
# what it is before anyone pastes a path from it into a shell.
BANNER = (
    "# REDACTED REVIEW COPY -- NOT DEPLOYABLE.\n"
    "# Every endpoint below is a reserved placeholder. The canonical manifest stays on the\n"
    "# host; see README.md for its digest and for the commands that are safe to run.\n"
    "\n"
)

# Public, provider-operated endpoints. Kept verbatim. Exact match, so a private host that
# merely shares a suffix is still rewritten.
PUBLIC_HOSTS = frozenset({
    "openrouter.ai", "api.deepseek.com", "chatgpt.com", "thaillm.or.th",
    "api.openai.com", "api.anthropic.com",
})

URL_RE = re.compile(r'(?P<scheme>https?://)(?P<host>[^/:"\s]+)(?P<port>:\d+)?(?P<rest>[^"\s]*)')
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# A tailnet hostname has at least one label in front of the private network label:
# ``htpc-wsl.<net>.ts.net``. Anchored so the first label is the one preserved.
TAILNET_HOST_RE = re.compile(r"(?P<label>[A-Za-z0-9-]+)\.(?:[A-Za-z0-9-]+\.)+ts\.net$", re.IGNORECASE)
# Scan form: finds any ``...ts.net`` occurrence, including one inside a longer string. The
# host form above is anchored and would miss a trailing path, so this one guards the output.
TAILNET_SCAN_RE = re.compile(r"[A-Za-z0-9-]+(?:(?:\.[A-Za-z0-9-]+)*)\.ts\.net\b", re.IGNORECASE)
# Bare occurrence, i.e. one that is not inside a URL -- typically prose in a comment. Prose is
# published too, so it is rewritten rather than left to the leftover check to reject.
BARE_TAILNET_RE = re.compile(r"(?<![A-Za-z0-9.-])(?P<label>[A-Za-z0-9-]+)\.(?:[A-Za-z0-9-]+\.)+ts\.net\b",
                             re.IGNORECASE)
HOME_RE = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+")
MNT_RE = re.compile(r"/mnt/[A-Za-z0-9_./-]+")

# Credential shapes, duplicated from the repository guard so this tool is safe to run
# standalone on a file the guard has never seen.
CREDENTIAL_PATTERNS = [
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}"), "openrouter key"),
    (re.compile(r"\bsk-(?!or-v1-)[A-Za-z0-9]{32,}"), "provider key"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"), "github token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), "github fine-grained token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws access key id"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "slack token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "jwt"),
]


def _is_public_host(host: str) -> bool:
    return (host in PUBLIC_HOSTS
            or host.endswith("." + PLACEHOLDER_DOMAIN)
            or host.endswith("." + PLACEHOLDER_TAILNET)
            or host in {"localhost", "127.0.0.1"})


def _redact_once(text: str) -> tuple[str, dict[str, str]]:
    """Rewrite private coordinates. Pure: same input, same output, same mapping."""
    mapping: dict[str, str] = {}
    counter = {"n": 0}

    def remember(real: str, fake: str) -> str:
        mapping.setdefault(real, fake)
        return fake

    def sub_url(m: re.Match[str]) -> str:
        host = m.group("host")
        if _is_public_host(host) or host == PLACEHOLDER_IP:
            return m.group(0)
        tailnet = TAILNET_HOST_RE.match(host)
        if tailnet:
            fake = remember(host, "%s.%s" % (tailnet.group("label"), PLACEHOLDER_TAILNET))
        elif IPV4_RE.fullmatch(host):
            fake = remember(host, PLACEHOLDER_IP)
        else:
            counter["n"] += 1
            fake = remember(host, "host-%d.%s" % (counter["n"], PLACEHOLDER_DOMAIN))
        return "%s%s%s%s" % (m.group("scheme"), fake, m.group("port") or "", m.group("rest") or "")

    text = URL_RE.sub(sub_url, text)

    def sub_ip(m: re.Match[str]) -> str:
        raw = m.group(0)
        if raw == PLACEHOLDER_IP:
            return raw  # already a placeholder; keep the mapping free of identity entries
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            return raw
        if addr.is_global:
            return raw
        return remember(raw, PLACEHOLDER_IP)

    text = IPV4_RE.sub(sub_ip, text)

    def sub_bare(m: re.Match[str]) -> str:
        whole = m.group(0)
        if whole.lower().endswith(PLACEHOLDER_TAILNET):
            return whole
        return remember(whole, "%s.%s" % (m.group("label"), PLACEHOLDER_TAILNET))

    text = BARE_TAILNET_RE.sub(sub_bare, text)
    text = HOME_RE.sub(PLACEHOLDER_PATH, text)
    text = MNT_RE.sub(PLACEHOLDER_PATH, text)
    return text, mapping


def _shape(obj: object, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.add(prefix + str(key))
            out |= _shape(value, prefix + str(key) + ".")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            out |= _shape(value, prefix + "[%d]." % index)
    return out


def _leftovers(text: str) -> list[str]:
    """Every coordinate or credential the redactor was supposed to remove and did not."""
    problems: list[str] = []

    for match in TAILNET_SCAN_RE.finditer(text):
        if not match.group(0).lower().endswith(PLACEHOLDER_TAILNET):
            problems.append("private-network hostname")

    for match in IPV4_RE.finditer(text):
        raw = match.group(0)
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            problems.append("malformed address %s" % raw)
            continue
        if raw == PLACEHOLDER_IP or addr.is_global:
            continue
        problems.append("non-public address")

    for pattern in (HOME_RE, MNT_RE):
        if pattern.search(text):
            problems.append("host-local path")

    for pattern, description in CREDENTIAL_PATTERNS:
        if pattern.search(text):
            problems.append(description)

    return problems


def sanitize(src: str) -> tuple[str, dict[str, str]]:
    """Return (redacted_text, mapping). Raises ValueError if a property fails."""
    raw_text = open(src, "rb").read().decode()
    # The banner is part of the published form, so a re-run strips it before redacting and
    # puts it back afterwards; otherwise every run would stack another copy and the output
    # would stop being a fixed point.
    body = raw_text[len(BANNER):] if raw_text.startswith(BANNER) else raw_text

    redacted, mapping = _redact_once(body)

    # Property 3: fixed point. A second pass that still finds something to rewrite means a
    # real coordinate survived the first, so the output is not publishable.
    again, _ = _redact_once(redacted)
    if again != redacted:
        raise ValueError("redaction is not a fixed point: a real coordinate survived")

    # Property 4: determinism.
    repeat, _ = _redact_once(body)
    if repeat != redacted:
        raise ValueError("redaction is not deterministic")

    # Property 1: structure.
    real = tomllib.loads(body)
    try:
        sanitized = tomllib.loads(redacted)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError("redacted output does not parse: %s" % exc) from exc
    missing = _shape(real) - _shape(sanitized)
    added = _shape(sanitized) - _shape(real)
    if missing or added:
        raise ValueError("structure changed: lost %d path(s), added %d"
                         % (len(missing), len(added)))

    published = BANNER + redacted

    # Property 2: nothing private survives. Checked on the text, not the parsed form, so a
    # coordinate hidden in a comment is caught too.
    problems = _leftovers(published)
    if problems:
        raise ValueError("redaction incomplete: %s" % ", ".join(sorted(set(problems))))

    return published, mapping


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    if len(args) != 2 or flags - {"--print-mapping"}:
        print(__doc__.strip().splitlines()[-1])
        return 2
    src, dst = args

    try:
        redacted, mapping = sanitize(src)
    except ValueError as exc:
        print("REFUSED: %s" % exc)
        return 1

    with open(dst, "w", encoding="utf-8", newline="") as handle:
        handle.write(redacted)

    raw = open(src, "rb").read()
    print("input      : %s (%d bytes)" % (src, len(raw)))
    print("input sha  : %s" % hashlib.sha256(raw).hexdigest())
    print("output     : %s (%d bytes)" % (dst, len(redacted.encode())))
    print("output sha : %s" % hashlib.sha256(redacted.encode()).hexdigest())
    print("verified   : structure preserved, nothing private survives, fixed point, deterministic")
    print("rewritten  : %d endpoint(s)" % len(mapping))
    if "--print-mapping" in flags:
        for real_host, fake in sorted(mapping.items()):
            print("   %-40s -> %s" % (real_host, fake))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
