# R6 receipt — realtime music and spatial robotics served as native lanes

2026-11-22 · branch `feat/gemini-r3-adaptive-bounded-live` @ `94bcd306` · Switchyard PR #46
Deployed and qualified on `switchyard-vm`. Rollback target still the pre-R6 release.

## What shipped

Two Google surfaces that have no OpenAI-shaped equivalent, so both are typed lanes rather than
chat routes. Neither appears in `/v1/models`; both are reached by their own endpoint.

| Surface | Endpoint | Adapter | Provider wire |
|---|---|---|---|
| `BidiGenerateMusic` | `POST /v1/music/{capability}` (WebSocket) | `music_surface.rs` | `BidiGenerateMusic` over `wss://…/ws/google.ai.generativelanguage…` |
| `GoogleRoboticsSpatial` | `POST /v1/robotics/spatial` | `robotics_surface.rs` | `gemini-robotics-er-2-preview` generateContent with the spatial task contract |

Both surfaces reserve budget through one shared seam. `reserve_physical_call_for(capability,
estimated_tokens)` in `capability_surfaces.rs` is the single reservation path — the two private
duplicates the adapters first carried were deleted, so a pacing or quota fix lands once.
`reserve_physical_call` / `release_physical_call` are now `pub(crate)`.

**CORRECTION (2026-11-22).** An earlier version of this paragraph called both capabilities
"zero-RPD families". That was wrong, and it contradicted the numbers in the same sentence. Neither
lane is a zero-RPD model: the owner's Free-tier table gives `lyria-realtime-exp` **15 RPM / no input
TPM / 1,500 RPD** and `gemini-robotics-er-2-preview` **5 RPM / 250,000 input TPM / 20 RPD**. What
the manifest's `daily_request_limit` expresses is the **ceiling this deployment may spend** on that
lane, not an allowance that exists independently of a caller — which is why nothing here bills
without a request. The rule that kept the other 20 rows out of the build is unchanged: those are
the genuinely zero-allocated entries (Nano Banana, Gemini 2.5 Pro, Lyria 3 Clip/Pro, Genie, and the
rest of the owner's "do not activate" list), and none of them is referenced by this release. One
adjacent trap belongs here because it is easy to conflate: the ER 2 **map-grounding tool** row is
0 RPD, so it is not exposed as a tool; the spatial model lane is the only robotics capability
served.

## Measured qualification (one real call each)

Script `/var/tmp/qual-r6/qualify_r6.py`, one call per surface, no parameter sweeps.

**Robotics** — 200 in **3.04 s**. Returned `point [500, 500]` for a red square placed at that
coordinate. Reported `x-switchyard-provider-prompt-tokens: 1181` against
`x-switchyard-estimated-tokens: 348` for the scene, i.e. the image estimator under-predicted by
3.4×. That is a budgeting signal only: `max_image_bytes` bounds the payload, and the reservation is
soft, so a near-limit image is admitted on an estimate that is known to be low.

**Music** — 200, then **silence** until the caller sends `{"playbackControl": "PLAY"}`. This is the
first beat of the receipt: `BidiGenerateMusic` completing `setupComplete` is not audio. After PLAY:
**12 chunks × 384,000 bytes**, mime `audio/l16;rate=48000;channels=2`, 2 s per chunk ⇒
192,000 B/s. The declared `max_session_audio_bytes = 33,554,432` therefore caps a session at
**≈175 s** of 48 kHz stereo L16, and the session ends on the byte ceiling rather than on silence.

## Deployment

| Field | Value |
|---|---|
| Release | `20261122T055957Z-7952200d6583170b-53fd9a84` (label `music-robotics`) |
| Binary | `7952200d6583170b15a943fde445c9682ce5905d9fe6dfac8a41f29f367f0d5e` |
| Manifest deployed | `53fd9a8443322ccc7a61e51e91d0506bcc4445ef44710260ac6131607bb4ba95` |
| Capability manifest | `03bcd4ecd83e2aaebe1d10c46469ac0853c67b0f22be1ee5c9a6035ec4a5f903` |
| Capability gate | PASS — 71 families / 3 subsystems, 28 required families present |
| Source | `94bcd3060ad9554fa40f508738d7d1ae44b3417d`, clean tree |
| Rollback target | `20261122T050658Z-bf25c056b4598e65-1071be2f` (LAST-KNOWN-GOOD, not yet promoted past) |

## The canonical manifest is byte-equivalent to what runs

The source-control copy of the manifest carries operator notes added after the release was
captured, so the file is `9215e541…` while production runs `53fd9a84…`. Re-deploying a
comment-only delta was attempted and **refused by the deployment validator**, which reads at most
two environment files (`ENV_FILE_1`, `ENV_FILE_2`) while this host spreads provider credentials
across four (`/etc/switchyard/secrets/providers.env`, `/var/lib/switchyard/oauth/switchyard-openai.env`,
`/etc/switchyard-gemini-secrets/gemini.env`, `/etc/switchyard-gemini-secrets/stephen.env` supplied by
unit drop-ins). The refusal was `google_gemini could not read api_key_env SWITCHYARD_GEMINI_API_KEY`,
and production was left untouched.

The delta was then proven to be comments only, three independent ways:

1. **Parsed-document equality.** `tomllib` load of both files compares equal; both serialise to the
   same canonical JSON `fccf2ed06aff2d007b69171c301892d78fdcda2b77e903910031181470a0a0c5`.
2. **Projection equality.** `scripts/canonical_manifest_projection.py` run over each file produces
   42,615 bytes that differ only in the recorded `source_manifest_sha256` line.
3. **Byte size and diff.** +999 bytes, all of them comment lines.

`routes.canonical.toml` beside this receipt is that projection, taken from the **source-control**
copy `9215e541…`: it is the reviewable canonical manifest, tied to the deployed bytes by the
equivalence above. It keeps every id, format, budget, hold and capability ceiling and drops only
`api_key_env` values (→ `credential_present`), endpoint addresses (→ `endpoint_class`) and comments.

## Test and lint state

- `cargo test -p switchyard-runner -p switchyard-server`: green, 24 result lines, zero failures,
  including 3 new tests in `tests/music_bridge.rs` and 4 in `tests/robotics_spatial.rs`.
- `cargo clippy -p switchyard-server --all-targets`: clean for both new files after removing the
  unused `PacingError`/`PacingTicket` imports and a dead `error_body` helper.
- Mutation harness `/var/tmp/qual/mutants_music_robotics.py`: **7 mutants, all caught** — covering
  the playback-control requirement, the byte ceiling, the estimator's tile arithmetic, the spatial
  task vocabulary, the object-count ceiling and the shared reservation seam.

## What this receipt does not claim

- **No latency or throughput guarantee.** Robotics was measured once (3.04 s); music chunk cadence
  was measured once (2 s/chunk). Neither is a distribution.
- **No estimator accuracy.** The 348 → 1181 gap is one sample of a known-soft estimator; the
  receipt records it rather than treating the estimate as a bound.
- **No multi-session or adversarial coverage.** Long sessions, mid-stream refusals after the byte
  ceiling, and second-concurrent-session behaviour are unit-tested only.
- **Build reproducibility.** The release record says `attested: declared-at-deploy`: the source
  identity was declared for these exact bytes, and no rebuild was compared against them.
- **The comment-only manifest delta is not itself deployed.** It was deliberately left undeployed
  once equivalence was proven, rather than widening the deployment validator's environment file
  support inside a release.

## Open items

Status as of 2026-11-22T06:30Z; items 1–3 are annotated rather than rewritten.

1. `activate-release.sh` validates against two environment files; this host needs four. Either the
   validator grows repeated `--env-file` support or the deployment record understates the
   configuration it binds.
   **CLOSED 2026-11-22** by `63191cd5`: discovery now reads every `EnvironmentFile` the unit
   declares (`systemctl show -p EnvironmentFiles`), passes all four to the validator in read order,
   and refuses when discovery is incomplete instead of substituting the two-file default.
2. The release record's `configuration.env_files` lists only the two defaults, so the Gemini drop-in
   files are outside the recorded configuration hash. Worth closing before the next Google release.
   **CLOSED 2026-11-22** by the same commit: the record is written from the resolved set, each file
   with its hash and whether the unit marks it required. The CURRENT release for R6 was written
   before the fix, so its record still names two files; the next deploy records four.
3. `promote` to LAST-KNOWN-GOOD pending; R6 has been live and healthy but the promotion is a separate
   deliberate act.
   **CLOSED 2026-11-22T06:06:24Z**: promoted. Rollback to `20261122T050658Z-bf25c056b4598e65-1071be2f`
   remains available by explicit release id.

## Follow-ups opened from this release

- **Robotics image-token admission.** The measured 348-estimated / 1,181-reported gap on one 256×256
  scene is tracked as its own admission-accuracy item, not as a defect in this surface. The estimate
  is a floor used for minute-window admission, and it is reported beside the provider's own count
  (`x-switchyard-estimated-tokens` / `x-switchyard-provider-prompt-tokens`). If a later scene pushes
  the lane near its 250,000 input TPM, a floor is the wrong number to admit on. The implementation is
  not reopened for this. Tracked as issue **#47**, which carries the measurement, the scope and the
  explicit non-goals.

## Environment coverage addendum — 2026-11-22T06:30Z

The release record written at 05:59:57Z named two environment files because that is all the installer
knew how to name. The unit has always declared four. This addendum records the full set **as
discovered now**, and is deliberately an addendum: `release.json` is the record of what was installed,
and editing it after the fact would make it claim the deploying process knew something it did not.

```
switchyard-neutral-green.service reads 4 environment file(s), in read order:
  required e2bc20e1f4b02625061c448cffa30c6427382d127c831e3ecfebde90bd02ff06  providers.env
  required 9c12dbb4efb04c6c31c21e608d789de85e7bd8bebadcbe6b19d2360b8c4c7653  switchyard-openai.env
  required 5607e54a1996f645f7715d0ed17e240bb8d4ea09690f393f72307ecba6bf8449  gemini.env
  required 24710b8ed21ef7f831e557f1e6f7609898b11de7321e2c6a787c31837d0da97c  stephen.env
```

Order matters and is preserved: a variable declared in two files resolves to the later one. All four
are `ignore_errors=no`, so the service would refuse to start if any were absent.

The discriminating check was run against the same candidate bytes the release carries
(`7952200d…`, manifest `9215e541…`): with only the first two files the validator refuses with
`google_gemini could not read api_key_env SWITCHYARD_GEMINI_API_KEY`; with the four above it returns
`candidate validates: server OK` at rc=0. That is the defect in one pair of runs, and it is what the
new installer reproduces by construction rather than by an operator spelling four paths.

### Installer as installed

| Field | Value |
|---|---|
| `activate-release.sh` | `8e3aa7d6adb75440743fc97fe119efa2beddec054db63129857dbe5afa899513` |
| `release_validate.py` | `f096927d41685c10a8153480fea775db85e31642f47aca126cfb30a608054af8` |
| Previous copies kept | `activate-release.sh.bak-514226c6`, `release_validate.py.bak-f4589df6` |
| Source | commit `63191cd5`, branch `feat/gemini-r3-adaptive-bounded-live` |

The data plane was not touched by that change: no restart, no release, `CURRENT` and
`LAST-KNOWN-GOOD` both still `20261122T055957Z-7952200d6583170b-53fd9a84`.
