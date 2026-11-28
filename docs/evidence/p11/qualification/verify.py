#!/usr/bin/env python3
"""Independent verifier for the P11 qualification runs.

Reads only the artifacts on disk (never the candidate's prose) and re-runs the
test suites / recomputes expectations. Emits one JSON object per run on stdout.

usage: verify.py <run-dir> [<run-dir> ...]
"""
import hashlib
import json
import os
import subprocess
import sys

# ROOT holds expected.json and the per-task run directories. It defaults to the
# directory this file lives in, so a checkout is self-contained; override with
# QUAL_ROOT when the runs live elsewhere (for example a scratch run tree).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QUAL_ROOT") or HERE
PY = os.environ.get("QUAL_PY") or "/var/tmp/audit-v194/testenv-v194/bin/python"
EXPECTED = json.load(open(ROOT + "/expected.json"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def run_pytest(run, task):
    d = os.path.join(run, task)
    if not os.path.isdir(d):
        return False, "no dir"
    try:
        r = subprocess.run([PY, "-m", "pytest", "-q"], cwd=d,
                           capture_output=True, text=True, timeout=180)
    except Exception as e:  # noqa: BLE001
        return False, f"pytest error {e}"
    tail = (r.stdout or "").strip().splitlines()
    return r.returncode == 0, (tail[-1] if tail else "no output")


def check(run, task):
    e = EXPECTED[task]
    art = os.path.join(run, task, e["artifact"])
    txt = read(art)
    if txt is None:
        return False, f"missing {e['artifact']}"
    if task in ("t1", "t2"):
        ok, line = run_pytest(run, task)
        low = txt.lower()
        if not ok:
            return False, f"pytest still red: {line}"
        if "passed" not in low or "failed" in low:
            return False, f"summary text not a clean pass: {txt.strip()[:120]!r}"
        return True, line
    if task == "t3":
        try:
            d = json.loads(txt)
        except Exception as ex:  # noqa: BLE001
            return False, f"bad json: {ex}"
        ids = [p.get("id") for p in d.get("panels", [])]
        if ids != e["keep_panel_ids"]:
            return False, f"panel ids {ids}"
        if d.get("version") != e["keep_version"]:
            return False, f"version {d.get('version')}"
        if d.get("meta") != e["keep_meta"]:
            return False, "meta changed"
        p3 = [p for p in d["panels"] if p["id"] == 3][0]
        if p3.get("options", {}).get("extra") != [1, 2, 3]:
            return False, "panel 3 nested extra lost"
        return True, "panels/version/meta preserved"
    if task == "t4":
        lines = [l for l in txt.splitlines() if l.strip()]
        want = e["lines"]
        if lines != want:
            return False, f"lines {lines}"
        return True, "4/4 refusal+readback lines exact"
    if task == "t5":
        got = txt.strip()
        return (got == e["needle"]), ("needle exact" if got == e["needle"] else f"got {got[:60]!r}")
    if task == "t6":
        if txt.strip() != e["expect"]:
            return False, f"final {txt.strip()[:40]!r}"
        s1 = read(os.path.join(run, task, "step1.txt"))
        s2 = read(os.path.join(run, task, "step2.txt"))
        if (s1 or "").strip() != e["step1"]:
            return False, f"step1 {(s1 or '').strip()[:20]!r}"
        if (s2 or "").strip() != e["step2"]:
            return False, f"step2 {(s2 or '').strip()[:20]!r}"
        return True, "step1=5050 step2=5051 final=DONE:5051"
    return False, "unknown task"


def main(argv):
    out = []
    for run in argv[1:]:
        name = os.path.basename(os.path.dirname(run)) + "/" + os.path.basename(run)
        rec = {"run": name, "tasks": {}, "passed": 0, "total": 6}
        for t in ("t1", "t2", "t3", "t4", "t5", "t6"):
            ok, why = check(run, t)
            rec["tasks"][t] = {"pass": ok, "why": why}
            rec["passed"] += 1 if ok else 0
        rec["all_pass"] = rec["passed"] == rec["total"]
        out.append(rec)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv)