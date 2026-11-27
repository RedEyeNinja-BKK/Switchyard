# Cost containment + peer-recovery successor — release evidence (2026-11-27)

## 1. Deployed correction (config-only, no binary change)

| field | value |
|---|---|
| release | `20261127T063310Z-fb12d2881387e338-9069fe66` (CURRENT) |
| binary | `fb12d2881387e3380d44142d69865cd3ba1e0c03293bae8d4da23dd9785ea169` (reused byte-identical) |
| manifest | `9069fe66ebfefdb7769fd59619dccae8f4be023ba833ecd543baa693c1dc1ea5` (133,371 B) |
| capabilities | `ad3a7c00f232f7268efaf94f7cf8fdd6762e86432a7e3bc404ac6f36a33cc968` |
| runtime | PID 842330, `NRestarts=0`, active, `/health` 200 |
| rollback anchor | `20261127T045138Z-fb12d2881387e338-2a20067b` (explicitly protected) |
| LKG | unchanged, `20261126T061331Z-65757ec7aaa51166-737a6887` |

Manifest-only edits against `2a20067b` (see `PATCH.diff`, 89 lines):

1. `deepseek-deepseek-v4-pro` removed from `routes.smart-route.candidates`.
2. The target's `groups = ["smart-deepseek"]` label deleted — the pool is built from the
   label, so removing only the candidate would have left the failover path open.
3. Group renamed `smart-deepseek` → `smart-deepseek-flash`, now a singleton holding
   `deepseek-deepseek-flash`.
4. `smart-route.pool_groups` names the renamed group.

`direct-deepseek-deepseek-v4-pro` and `vc-paid-core` / `VC_Paid` are untouched — V4 Pro stays
reachable for deliberate use.

### Topology as deployed — 11 effective candidates

| pool group | members |
|---|---|
| `smart-gemini-chat` | 5 (`gemini-3.1-flash-lite`, `3.5-flash-lite`, `3.6-flash`, `3.8-flash`, `4-hydrogen`) |
| `smart-openrouter-free` | 3 (`ernie-5.5-flash-preview:free`, `ling-3.2-flash:free`, `nemotron-3.5-super-vl:free`) |
| `smart-deepseek-flash` | 1 (`deepseek-flash`) — the singleton third group |
| `smart-openrouter-paid` | 2 (`glm-5.3-flash`, `tencent/hy4-flash`) |

Total 5+3+1+2 = **11**. The 10-entry `candidates` list is a subset; the 11th member
(`direct-google-gemini-4-hydrogen`) is the route `target`.

**Desired topology (documented, NOT admitted): 12 = 5+4+1+2** — `nemotron-3-ultra-550b-a55b:free`
would join `smart-openrouter-free`. Held because chat `TargetConfig` has no targeted
third-party-free admission mechanism.

### Defect measured before the fix (24 h, 6,165 handled records)

`smart-route` served 3,184; **77** dispatches reached `deepseek-v4-pro`, all entered by
failover with reason `unavailable` (75 from `nemotron-3.5-super-vl:free`, 2 from
`ling-3.2-flash:free`). No other route dispatched it.

After the 06:33:10Z deploy: **0** V4 Pro dispatches; 26 `smart-route` requests,
all on `deepseek-flash` (24) or gemini flash-lite (2).

### Proof

- `cargo test -p switchyard-server --test cost_containment` → 4 passed.
- Negative control: re-adding V4 Pro to `candidates` **and** the group label → 2 failed,
  naming `deepseek-v4-pro` in the dispatchable set.
- P9 on the deployed pair: `direct=54 qualified=54 failures=0`, manifest `9069fe66`,
  checker `72838c34…`.
- Capability gate: every `always_present` family exposed; 71 families, 0 removed.

## 2. Peer-recovery successor — qualified pair

| field | value |
|---|---|
| binary | `aaf1bec88835acaca1d9dbff556fb74cbd1647a719f32ddfd3aafd26b6f01dcf` (31,227,360 B) |
| source | `088e45b6ce2376ee2a65507e0a7805e77e658fb7`, branch `feat/nemotron-ultra-peer-recovery-20261127` |
| manifest | `9069fe66…` — the NEW cost-contained CURRENT bytes (not the Package D or successor manifest) |
| capabilities | `5312e9307f68d46bd0969ebf15b096347d6741ad2ef7884ee0623e332a0bf67b` |

Ancestry proof (not asserted — measured):

- `git archive 088e45b` → 859-file content manifest `a9782407b2be6506ae34bbd07ef04178a7d6ccd5cbd694d33b0a288facfa1db4`.
- The source tree used on `switchyard-vm` hashes to the **same** manifest, 859 files.
- The staged binary is bit-identical to the artifact built from that tree at
  `/mnt/turnstone-build-ssd/cargo-target-release-p10/release/switchyard-server`
  (mtime 12:45:28, commit 12:45:14).

Capability semantics vs the deployed baseline `ad3a7c00…`: **0 non-build differing leaves**;
identical counts (57 instruments, 71 families = 16 always_present + 55 event_driven, 8 declared_kind),
identical subsystems, `contract_version` 1. Only `build` differs (binary sha, branch, commit).

Validation of the pair: `candidate validates: server OK …` and
`P9 PASS direct=54 qualified=54 failures=0 manifest_sha256=9069fe66…`.

V4 Pro unreachable through `smart-route` on this pair: the pair's manifest is byte-identical to
the deployed manifest (`9069fe66`), which is exactly the fixture proven by the four
`cost_containment` tests above.

## 3. Reproduction

```bash
# manifest identity
sha256sum routes.9069fe66.toml            # 9069fe66ebfefdb7769fd59619dccae8f4be023ba833ecd543baa693c1dc1ea5
git apply --check PATCH.diff              # against routes.2a20067b.toml

# capability semantics on the release host
python3 /opt/switchyard/bin/capability_contract.py generate \
  --binary ./switchyard-server --source ./src-git \
  --contract ./src-git/scripts/release/capabilities.toml --out ./capabilities.pair.json

# strict pair validation
sudo -n /opt/switchyard/bin/release_validate.py \
  --binary ./switchyard-server --manifest /var/lib/switchyard/releases/CURRENT/routes.toml \
  --env-file /etc/switchyard/secrets/providers.env \
  --env-file /var/lib/switchyard/oauth/switchyard-openai.env \
  --env-file /etc/switchyard-gemini-secrets/gemini.env \
  --env-file /etc/switchyard-gemini-secrets/stephen.env
```

`routes.successor.16a97419.toml` is retained for the desired 12-candidate topology only.
It is **not** an admission request; the Ultra leaf is held.
