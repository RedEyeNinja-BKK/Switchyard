#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  --reuse-capabilities)
    echo 'FATAL: unknown option --reuse-capabilities' >&2; exit 1;;
  --missing-artifact)
    if [ ! -f artifact.bin ]; then echo 'FATAL: missing artifact artifact.bin' >&2; exit 1; fi;;
  --hash-mismatch)
    echo 'FATAL: hash mismatch expected 0000 actual abcd' >&2; exit 1;;
  --apply)
    [ -f artifact.bin ] || { echo 'FATAL: missing artifact artifact.bin' >&2; exit 1; }
    echo 'ACTIVATION COMPLETE'; exit 0;;
  *)
    echo 'FATAL: unknown mode' >&2; exit 1;;
esac
