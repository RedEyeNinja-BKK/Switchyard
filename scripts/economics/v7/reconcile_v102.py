#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline V7.102 reconciliation: live revision-106 export in, V7.102 out.

This is the second-generation corrector. Unlike the V7.98 -> V7.101 chain it
does not rebuild the document from an old base: the base IS the published
document (the byte-exact Grafana read at revision 106), and this transform
applies only the deltas that reconcile it to release
``20261128T182648Z-6a8209f97101563b-02587e3e``.

Every anchor must match exactly once. A moved or duplicated anchor refuses the
build rather than silently editing the wrong paragraph.
"""
import argparse
import json
import re

RELEASE = "20261128T182648Z-6a8209f97101563b-02587e3e"
MANIFEST_SHA = "02587e3ef974b2c1a62593b5256039eb62b9cdcfc26b7fbfba314a6cba485145"
MANIFEST_BYTES = "145,413"
CAPABILITIES_SHA = "987dd039ce618d5a0d124bd5607498b3c8d6845488af0d593fb3b3f7839f77b9"
BINARY_SHA = "6a8209f97101563b58944f8614001fd0b057247da21a3daf57624ff798a22e82"
ROLLBACK = "20261128T040916Z-6a8209f97101563b-fc05b640"
LKG = "20261126T061331Z-65757ec7aaa51166-737a6887"
READ_DATE = "2026-11-29"

ROW_54 = ("| 54 | `direct-openrouter-stephen-particle-1-free` | Stephen OpenRouter | "
          "`particle-ai/particle-1:free` | 1,048,576 | Y | t\u2713 i\u2713 a? v? p? | text | "
          "\u2713 | Y/N/N | chat:cfg | [OR](https://openrouter.ai/models) |")
ROW_55 = ("| 55 | `direct-openrouter-stephen-tencent-hy4-flash` | Stephen OpenRouter | "
          "`tencent/hy4-flash` | 1,048,576 | Y | t\u2713 i\u2713 a? v? p? | text | "
          "\u2713 | Y/Y/N | chat:cfg | [OR](https://openrouter.ai/models) |")

CAPABILITY_TABLE = """**Stephen Google specialist capabilities \u2014 15 typed lanes (added by release `20261128T182648Z-\u2026-02587e3e`, 2026-11-28).**

All fifteen read `SWITCHYARD_GEMINI_STEPHEN_API_KEY` and share quota group `google-ai-studio-free-stephen`. They are capability lanes, not chat routes: none appears in the 74-route chat census and none is reachable through `/v1/chat/completions`. A contract belongs to one client, so the two Stephen embedding twins declare their own contracts rather than inheriting the owner's.

| Capability | Target model | Operation | Declared limits |
|---|---|---|---|
| `stephen-gemini-tts-3.8-flash` | `gemini-3.8-flash-tts` | speech | voice Kore \u00b7 wav \u00b7 24 kHz \u00b7 \u22648,192 in \u00b7 3 rpm \u00b7 10/day |
| `stephen-gemini-tts-3.8-flash-lite` | `gemini-3.8-flash-lite-tts` | speech | voice Kore \u00b7 wav \u00b7 24 kHz \u00b7 \u22648,192 in \u00b7 3 rpm \u00b7 10/day |
| `stephen-gemini-tts-3.1-flash` | `gemini-3.1-flash-tts-preview` | speech | voice Kore \u00b7 wav \u00b7 24 kHz \u00b7 \u22648,192 in \u00b7 3 rpm \u00b7 10/day |
| `stephen-gemini-tts-2.5-flash` | `gemini-2.5-flash-preview-tts` | speech | voice Kore \u00b7 wav \u00b7 24 kHz \u00b7 \u22648,192 in \u00b7 3 rpm \u00b7 10/day |
| `stephen-gemini-embedding` | `gemini-embedding-001` | embed | 3072D \u00b7 L2 \u00b7 batch 16 \u00b7 \u22642,048 in \u00b7 100 rpm \u00b7 1,000/day |
| `stephen-gemini-embedding-2` | `gemini-embedding-2` | embed | 3072D \u00b7 L2 \u00b7 batch 8 \u00b7 \u22648,192 in \u00b7 100 rpm \u00b7 1,000/day |
| `stephen-gemini-transcribe` | `gemini-3.5-transcribe` | transcribe | \u2264300 s \u00b7 \u226415,728,640 B \u00b7 3 rpm \u00b7 25/day |
| `stephen-gemini-robotics-er-2-spatial` | `gemini-robotics-er-2-preview` | robotics / spatial | \u2264131,072 in \u00b7 \u22648,388,608 B image \u00b7 5 rpm \u00b7 20/day |
| `stephen-gemini-music-lyria-realtime` | `lyria-realtime-exp` | music (realtime) | \u226433,554,432 B session audio \u00b7 15 rpm \u00b7 1,500/day |
| `stephen-gemini-live-2-5-flash-native-audio` | `gemini-2.5-flash-native-audio-latest` | live | response AUDIO |
| `stephen-gemini-live-3-flash` | `gemini-3.1-flash-live-preview` | live | response AUDIO |
| `stephen-gemini-live-3-8-flash` | `gemini-3.8-live` | live | response AUDIO |
| `stephen-gemini-live-3-8-flash-extended-thinking` | `gemini-3.8-live-extended-thinking` | live | response AUDIO \u00b7 thinking level HIGH |
| `stephen-gemini-live-3-5-transcribe` | `gemini-3.5-transcribe-live` | live | response TEXT \u00b7 verbatim instruction |
| `stephen-gemini-live-3-5-translate` | `gemini-3.5-live-translate-preview` | live | response TEXT \u00b7 thinking budget 0 |

**Measured on the Stephen key 2026-11-29 (free tier):** TTS `stephen-gemini-tts-3.8-flash` \u2192 200, 159,912 B, 24 kHz mono, 3.33 s; embeddings `stephen-gemini-embedding` and `-2` \u2192 200, dimension 3072, L2 norm 1.000; transcription `stephen-gemini-transcribe` \u2192 200 in 3.25 s. **Not measured:** all six Live lanes, music and robotics. A declared lane is an admission gate, not proof of a working lane."""

PROBE_NOTE = """**Stephen capability twins \u2014 first measured pass (2026-11-29).** One call each, Stephen credential, free tier: speech 200 (159,912 B \u00b7 24,000 Hz \u00b7 mono \u00b7 3.33 s, valid RIFF/WAVE); embeddings 200 (3072D, L2 norm 1.000) on both contracts; transcription 200 in 3.25 s (typed JSON body, not multipart). A transcription round-trip returns ASR paraphrase, not a verbatim string, so exact-match grading is not valid on real speech. Not measured: six Live lanes, music, robotics. `ST_Embed` (local 4096D) refused a call from a `third_party_free` route because the egress gate requires a declared data class of `public` or `synthetic`; that refusal is the gate working, not a lane defect, and no privacy setting was changed."""

SNAPSHOT = f"""**DEPLOYMENT SNAPSHOT - live P11 record (read {READ_DATE}, sha256sum-verified).**

| Item | Recorded evidence |
|---|---|
| CURRENT (deployed) | `{RELEASE}` |
| Release action / label | `deploy` / `stephen-google-capability-twins-20261128` (activated 2026-11-28T18:26:48Z) |
| Binary sha256 | `{BINARY_SHA}` (31,281,480 B) - unchanged from the qualified binary |
| Manifest sha256 | `{MANIFEST_SHA}` ({MANIFEST_BYTES} B installed routes.toml) |
| Capabilities sha256 | `{CAPABILITIES_SHA}` (25,499 B) - reused verified capsule (byte-identical binary) |
| Rollback target | `{ROLLBACK}` (predecessor, protected in store) |
| LAST-KNOWN-GOOD | `{LKG}` (not promoted) |
| Running service | PID 912928, NRestarts=0, /v1/models 74 of 74 |
| Basis | config-only on qualified binary 6a8209f9: adds eleven Stephen-keyed Google free capability twins (2 embedding, 6 Live, music, robotics, transcription) on `SWITCHYARD_GEMINI_STEPHEN_API_KEY`, quota group `google-ai-studio-free-stephen`; strict additive, no existing lane changed; census 74 routes / 60 targets / 55 direct aliases / 38 capabilities (15 Stephen) |

P11 config-only release: binary unchanged; only manifest bytes differ. No rebuild, no LKG promotion, no provider traffic.

**Superseded snapshot preserved as history:** the immediately previous record named `{ROLLBACK}` (label `ling-out`) with 72 routes / 58 targets / 53 direct aliases. That text is not repeated here; the release store keeps the release itself."""


def _flatten(panels):
    flat = {}
    def walk(items):
        for panel in items:
            flat[panel["id"]] = panel
            walk(panel.get("panels", []))
    walk(panels)
    return flat


def sub_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: anchor matched {count} times, expected exactly 1: {old[:90]!r}")
    return text.replace(old, new, 1)


def renumber(content):
    out, n = [], 0
    for line in content.splitlines():
        m = re.match(r"^\| (\d+) \| (.*)$", line)
        if m:
            n += 1
            out.append(f"| {n} | {m.group(2)}")
        else:
            out.append(line)
    return "\n".join(out)


CONTENT_VERSION = 102
NEXT_REVISION = 107


def build(document):
    document["title"] = f"SWITCHYARD DASHBOARD V7.{CONTENT_VERSION}"
    document["version"] = NEXT_REVISION
    document["description"] = (
        "P11: release 02587e3e adds eleven Stephen-keyed Google free capability twins "
        "(2 embeddings, 6 Live, music, robotics, transcription); 55 direct / 74 advertised; "
        "smart-route 3/2/1/2 = 8 (V7.102)."
    )
    flat = _flatten(document["panels"])

    # --- 12001: master direct routes, 53 -> 55 rows, re-anchored to CURRENT ---
    panel = flat[12001]
    content = panel["options"]["content"]
    content = sub_once(
        content,
        "every defined direct alias, 53 of 53, all advertised, none held.** Read from the "
        "deployed manifest (`routes.toml` sha256 `a7ef91747befd072\u2026`, 131,100 B, release "
        "`20261128T040916Z-6a8209f97101563b-fc05b640`), `/v1/models` read at 72 of 72 "
        "advertised on the service listener (2026-11-28T03:44Z)",
        "every defined direct alias, 55 of 55, all advertised, none held.** Read from the "
        f"deployed manifest (`routes.toml` sha256 `{MANIFEST_SHA[:16]}\u2026`, {MANIFEST_BYTES} B, release "
        f"`{RELEASE}`), `/v1/models` read at 74 of 74 advertised on the service listener "
        f"({READ_DATE}T19:40Z)",
        "12001 intro",
    )
    content = sub_once(
        content,
        "Row 50 is new since the previous revision: `direct-openrouter-nemotron-3-ultra-free` "
        "was admitted by release `20261127T091816Z-aaf1bec88835acac-c635222b`, so rows 50-55 "
        "were renumbered and the table is no longer '54 of 54'.",
        "Rows 54-55 are new since the previous revision: `direct-openrouter-stephen-particle-1-free` "
        "and `direct-openrouter-stephen-tencent-hy4-flash` are defined in the deployed manifest and "
        "are documented here from the OpenRouter listing read 2026-11-29 (1,048,576 context, "
        "text+image input, text output, tools listed). Row 50 records the Ultra admission of release "
        "`20261127T091816Z-aaf1bec88835acac-c635222b`; the earlier renumbering note is superseded "
        "by this table.",
        "12001 provenance sentence",
    )
    last_row = ("| 53 | `direct-openrouter-stephen-nemotron-3-5-super-vl-free` | Stephen OpenRouter | "
                "`nvidia/nemotron-3.5-super-vl:free` | 1,000,000 | Y | t\u2713 i\u2713 a? v\u2713 p? | text | "
                "\u2713 | Y/Y/Y | chat:obs | [OR](https://openrouter.ai/models) |")
    content = sub_once(content, last_row, last_row + "\n" + ROW_54 + "\n" + ROW_55,
                       "12001 append rows 54-55")
    content = renumber(content)
    content += (
        "\n\n**Two appended rows ({date}).** `direct-openrouter-stephen-particle-1-free` declines "
        "vision in this manifest (`SW t/v/vid` = Y/N/N) although the vendor listing advertises "
        "text+image input \u2014 the declaration is the admission gate and it is narrower than the "
        "vendor. `direct-openrouter-stephen-tencent-hy4-flash` is a paid-catalogue target "
        "(`ST_OR_Paid`); its Stephen direct alias is defined and advertised but belongs to no free "
        "ladder, so the row documents definition, not free availability.".format(date=READ_DATE)
    )
    panel["options"]["content"] = content
    if "content" in panel and panel.get("content") is not None:
        panel["content"] = content
    panel["title"] = "MASTER DIRECT ROUTES - 55 ALIASES (ONE TABLE, ALL DEFINED)"

    # --- 12002: topology, re-anchor to CURRENT ---
    panel = flat[12002]
    content = panel["options"]["content"]
    content = sub_once(
        content,
        "(`routes.toml` sha256 `fc05b640b49edfd2\u2026`, 131,100 B, release "
        "`20261128T040916Z-6a8209f97101563b-fc05b640`) and cross-checked against `/v1/models` "
        "on the service listener (127.0.0.1:4000, **72 of 72 advertised**, 2026-11-28).",
        f"(`routes.toml` sha256 `{MANIFEST_SHA[:16]}\u2026`, {MANIFEST_BYTES} B, release "
        f"`{RELEASE}`) and cross-checked against `/v1/models` on the service listener "
        f"(127.0.0.1:4000, **74 of 74 advertised**, {READ_DATE}).",
        "12002 anchor",
    )
    content += (
        "\n\n**Release `02587e3e` added eleven Stephen-keyed Google free capability twins** \u2014 "
        "two embeddings, six Live models, music, robotics and transcription, fifteen capability "
        "lanes in total. They are typed capability lanes, not chat routes: the advertised chat "
        "census stays 74 and every ladder described above is unchanged. Their declarations and "
        "limits are in TYPED OPERATIONS."
    )
    panel["options"]["content"] = content
    if panel.get("content") is not None:
        panel["content"] = content

    # --- 12003: typed operations, add the Stephen specialist capability table ---
    panel = flat[12003]
    content = panel["options"]["content"]
    content = content.rstrip() + "\n\n" + CAPABILITY_TABLE
    panel["options"]["content"] = content
    if panel.get("content") is not None:
        panel["content"] = content

    # --- 12004: credential / registry census ---
    panel = flat[12004]
    content = panel["options"]["content"]
    content = sub_once(
        content,
        "refreshed against release `20261128T040916Z-6a8209f97101563b-fc05b640` "
        "(`routes.toml` sha256 `fc05b640b49edfd2\u2026`, 131,100 B; 2026-11-28).",
        f"refreshed against release `{RELEASE}` (`routes.toml` sha256 "
        f"`{MANIFEST_SHA[:16]}\u2026`, {MANIFEST_BYTES} B; {READ_DATE}).",
        "12004 header",
    )
    content = sub_once(
        content,
        "**Registry census (2026-11-28):** 72 advertised chat routes on `/v1/models` \u2014 53 "
        "direct aliases (5 owner-operated local: 2 ComfyNinja, 3 HTPC) and 19 non-direct (1 "
        "non-advertised chat composite `ST_Route`, 1 decision composition `smart-decision`, 4 "
        "typed-decision lanes, embedding and reranking lanes).",
        "**Registry census (2026-11-29):** 74 advertised chat routes on `/v1/models` \u2014 55 "
        "direct aliases (5 owner-operated local: 2 ComfyNinja, 3 HTPC) and 19 non-direct (1 "
        "non-advertised chat composite `ST_Route`, 1 decision composition `smart-decision`, 4 "
        "typed-decision lanes, embedding and reranking lanes). Separately, 15 Stephen Google "
        "specialist capabilities are typed lanes and are not part of that chat census.",
        "12004 census",
    )
    panel["options"]["content"] = content
    if panel.get("content") is not None:
        panel["content"] = content

    # --- 12005: capability evidence, append the dated measured pass ---
    panel = flat[12005]
    content = panel["options"]["content"].rstrip() + "\n\n" + PROBE_NOTE
    panel["options"]["content"] = content
    if panel.get("content") is not None:
        panel["content"] = content

    # --- 12007: release snapshot ---
    panel = flat[12007]
    panel["options"]["content"] = SNAPSHOT
    if panel.get("content") is not None:
        panel["content"] = SNAPSHOT

    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconcile V7 to release 02587e3e.")
    parser.add_argument("--base", required=True, help="raw Grafana read (rev 106 export)")
    parser.add_argument("--out", required=True, help="output payload")
    parser.add_argument("--expected-version", type=int, required=True)
    parser.add_argument("--uid", default="turnstone-switchyard-v7")
    args = parser.parse_args(argv)

    with open(args.base, encoding="utf-8") as handle:
        base = json.load(handle)
    document = base["document"] if "document" in base else base
    document = build(json.loads(json.dumps(document)))

    # The operator labels a publication by its content version and its saved
    # revision ("V7.101 / r106"); the two carry a constant offset. Validate the
    # PAYLOAD's content suffix against the base document's suffix + 1, and the
    # payload's structural version against the CAS revision + 1.
    live_title = (base["document"] if "document" in base else base).get("title", "")
    match = re.search(r"V7\.(\d+)$", live_title)
    if not match:
        raise SystemExit(f"cannot read a V7 content version from base title {live_title!r}")
    expected_content = int(match.group(1)) + 1
    expected_title = f"SWITCHYARD DASHBOARD V7.{expected_content}"
    if document["title"] != expected_title:
        raise SystemExit(f"title {document['title']!r} != {expected_title!r}")
    if document["version"] != args.expected_version + 1:
        raise SystemExit(f"version {document['version']} != {args.expected_version + 1}")

    request = {
        "op": "update",
        "uid": args.uid,
        "expected_version": args.expected_version,
        "folder_uid": None,
        "message": (
            "V7.102: reconcile to release 02587e3e \u2014 55 direct aliases (adds the two Stephen "
            "OpenRouter rows the title already claimed), 74 advertised, Stephen capability twins "
            "documented, release snapshot refreshed. Historical V7.101/r106 preserved."
        ),
        "dashboard": document,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: title {document['title']} version {document['version']} "
          f"expected_version {args.expected_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
