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

Both capabilities are **zero-RPD** families: `requests_per_minute = 15`,
`daily_request_limit = 1500` is the ceiling the lane may spend, not an allowance that exists
independently of the request. Nothing was activated that would bill without a caller.

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

1. `activate-release.sh` validates against two environment files; this host needs four. Either the
   validator grows repeated `--env-file` support or the deployment record understates the
   configuration it binds.
2. The release record's `configuration.env_files` lists only the two defaults, so the Gemini drop-in
   files are outside the recorded configuration hash. Worth closing before the next Google release.
3. `promote` to LAST-KNOWN-GOOD pending; R6 has been live and healthy but the promotion is a separate
   deliberate act.
