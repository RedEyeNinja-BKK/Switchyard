# P11 candidate qualification fixtures

Deterministic, machine-checkable engineering tasks used to qualify candidate
models on agentic work (tool use, multi-turn continuation, exact artifact
production). The fixtures are model-agnostic: the same six tasks are handed to
every candidate, and the artifacts are graded by `verify.py` from disk only —
never from the candidate's own prose.

This directory is the durable, versioned copy. The first pass built the fixtures
under `/var/tmp/qual-ling-out`; that scratch tree is not required to run or
verify a suite.

## Contents

| Path | Purpose |
| --- | --- |
| `fixtures/t1` | Single failing test in a two-test package; fix the source (not the tests) and record the pytest summary. |
| `fixtures/t2` | Multi-file service with an off-by-one boundary defect; fix and record the pytest summary. |
| `fixtures/t3` | Nested dashboard JSON edit: drop panel 2, preserve panel 3's nested `options.extra`, `version` and `meta` byte-for-byte. |
| `fixtures/t4` | Shell activation script: four invocations, each must produce an exact STDERR/STDOUT line and exit code. |
| `fixtures/t5` | Large log triage: locate a single release token and emit it exactly. |
| `fixtures/t6` | Two-step arithmetic with intermediate files: `step1.txt`, `step2.txt`, then `final.txt` = `DONE:5051`. |
| `expected.json` | The grading contract read by `verify.py`. |
| `verify.py` | Independent verifier: re-runs pytest for t1/t2, re-parses JSON for t3, compares exact lines/tokens for t4/t5/t6. |

## Running a suite

Each candidate run gets its own copy of the fixture tree; the candidate is told
to work only inside that directory.

```bash
RUN=/var/tmp/qual-runs/<stamp>/<candidate>
mkdir -p "$RUN"
cp -a fixtures/t1 "$RUN/t1"
chmod -R a+rwX "$RUN"          # the node service account must be able to write
```

After the turn completes, grade it independently:

```bash
QUAL_ROOT=/var/tmp/qual-runs/<stamp> \
QUAL_PY=/var/tmp/audit-v194/testenv-v194/bin/python \
python3 verify.py /var/tmp/qual-runs/<stamp>/<candidate>
```

`QUAL_ROOT` must contain `expected.json`; each argument is a run directory that
holds the six task directories. Output is one JSON object per run with
`tasks.<t>.pass`, `tasks.<t>.why`, `passed`, `total` and `all_pass`.

The interpreter needs `pytest`; on this host only the audit venv has it, so
`QUAL_PY` must point at it (or at any venv with pytest installed).

## Grading rules

- Nothing is graded from the model's reply. Every check reads a file the model
  wrote and, for t1/t2, re-runs the suite itself.
- A short summary line is only accepted when the suite is genuinely green AND
  the text contains `passed` and not `failed`.
- t3 requires structural preservation, not just the requested deletion.
- t4 requires exact strings and exit codes for all four invocations.
- t5 and t6 require exact values; partial or paraphrased answers fail.

## Integrity

SHA-256 of the fixture tree, computed as
`find fixtures -type f | sort | xargs sha256sum | sha256sum` (no caches):

```
0fdafb1c07721fd03f15c40ce15533f9f0bc91b0b37cd1977fa2f9ec329863c2
```

The fixtures contain no credentials, no hostnames of live systems and no
operator data: paths are local, tokens in t5 are synthetic, and t4 exercises a
generic activation script.
