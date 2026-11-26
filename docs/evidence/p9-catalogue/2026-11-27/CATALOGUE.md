# Deployed catalogue — P9 manifest

Deployed manifest SHA-256 `e887392c586b86cd4b75b14b6df5fd820ae6bc60a79c8bb7daab63d830c4c4d9`.

This run read the deployed manifest: `e887392c586b86cd4b75b14b6df5fd820ae6bc60a79c8bb7daab63d830c4c4d9`, 131153 bytes, schema 1.

Every route and target fact below is read from those bytes. `advertised` is the live
`/v1/models` read at the same time. A `consumer` is a Turnstone registry row; it is
credited to this manifest only when its `base_url` addresses this router and its model
string is a name the manifest declares — a route, a composite, a decision lane, a lane
composition key or a capability id. Rows that reach a provider directly are listed by
destination and are not counted as traffic this router serves. A cell not present in a
source is recorded as absent, never inferred from a sibling.

## Counts

| fact | value |
|---|---|
| routes | 73 |
| direct_routes | 54 |
| targets | 59 |
| composites | 3 |
| decision_lanes | 4 |
| llm_clients | 15 |
| advertised | 73 |
| routes_advertised | 73 |
| routes_not_advertised | none |
| advertised_not_declared | none |
| composites_advertised | none |
| targets_shared | 30 |
| targets_single_use | 29 |
| targets_unreferenced | none |
| targets_partition_holds | True |
| targets_reached_by_a_lane_only | `direct-google-gemini-decision-1`, `direct-htpc-jev-style`, `direct-htpc-laya`, `direct-openrouter-span-01-lite`, `direct-openrouter-stephen-span-01-lite` |
| capability_client_models_not_targets | `Qwen3-Embedding-0.6B`, `gemini-2.5-flash-native-audio-latest`, `gemini-2.5-flash-preview-tts`, `gemini-3.1-flash-live-preview`, `gemini-3.1-flash-tts-preview`, `gemini-3.5-live-translate-preview`, `gemini-3.5-transcribe`, `gemini-3.5-transcribe-live`, `gemini-3.8-flash-lite-tts`, `gemini-3.8-flash-tts`, `gemini-3.8-live`, `gemini-3.8-live-extended-thinking`, `gemini-embedding-001`, `gemini-embedding-2`, `gemini-robotics-er-2-preview`, `lyria-realtime-exp`, `nvidia/nemotron-3.5-embed-8b:free`, `nvidia/nemotron-3.5-rerank-8b:free`, `qwen3-reranker-0.6b-q8_0` |
| capability_clients | 21 |
| registry_rows | 91 |
| routes_with_consumer | 58 |
| consumer_rows_addressed_here | 69 |
| consumer_rows_addressed_elsewhere | 22 |
| consumer_rows_without_base_url | 0 |
| consumers_addressed_here_naming_nothing_here | none |
| consumer_models_naming_an_upstream_id | `baidu/ernie-5.5-flash-preview:free`, `comfyninja/qwen3.8-27b-q4-mtp`, `comfyninja/qwen3.8-27b-q4-mtp-vl`, `deepseek-flash`, `deepseek/deepseek-v4.1-flash-1017`, `dots-studio/dots-3-note-preview:free`, `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`, `gpt-6-luna`, `inclusionai/ling-3.2-flash:free`, `nvidia/nemotron-3.5-super-vl:free`, `qwen/qwen3.7-flash`, `qwen/qwen3.8-flash`, `recursive/flint-v1.0:free`, `z-ai/glm-5.3-flash` |
| consumer_models_undeclared | `Typhoon-S-ThaiLLM-8B-Instruct`, `deepseek/deepseek-v4.1-flash`, `gpt-6.1-sol`, `htpc/qwen3.5-9b`, `htpc/qwen3.5-9b-mtp`, `liquid/lfm-2.5-2.6b:free` |
| capability_targets_undeclared | none |
| capability contract version | 1 |
| capability families | 71 |
| consumer rows by name kind | capability: 3, composite: 2, decision_lane: 4, lane_composition: 2, route: 58, undeclared: 6, upstream_model_id: 16 |

## Target reuse

Every name here is declared by a `[targets.*]` table. The three classes are a partition of
those declarations, asserted at generation time: shared + single-use + unreferenced equals
the declared target count, and no target is in two classes. A target behind more than one
owner is a shared failure domain: an allowance, a credential or a provider incident moves
every owner below at once.

Declared targets: **59** = 30 shared + 29 single-use + 0 unreferenced.

| target | owners |
|---|---|
| `comfyninja-qwen3_8-27b-q4-mtp-vl` | `direct-comfyninja-qwen3_8-27b-q4-mtp-vl`, `smartlocal`, `smartlocal-nt` |
| `deepseek-deepseek-flash` | `direct-deepseek-deepseek-flash`, `smart`, `smart-deepseek`, `smart-route` |
| `deepseek-deepseek-flash-nt` | `direct-deepseek-deepseek-flash-nt`, `smart-deepseek-nt`, `smart-nt` |
| `deepseek-deepseek-v4-pro` | `direct-deepseek-deepseek-v4-pro`, `smart-deepseek`, `smart-route` |
| `direct-google-gemini-3.1-flash-lite` | `direct-google-gemini-3.1-flash-lite`, `smart-gemini-bounded`, `smart-gemini-chat`, `smart-nt`, `smart-route`, `smart_gemini` |
| `direct-google-gemini-3.5-flash-lite` | `direct-google-gemini-3.5-flash-lite`, `smart-gemini-bounded`, `smart-gemini-chat`, `smart-nt`, `smart-route`, `smart_gemini`, `smartfree` |
| `direct-google-gemini-3.6-flash` | `direct-google-gemini-3.6-flash`, `smart-gemini-chat`, `smart-route`, `smart_gemini` |
| `direct-google-gemini-3.8-flash` | `direct-google-gemini-3.8-flash`, `smart-gemini-chat`, `smart-route`, `smart_gemini` |
| `direct-google-gemini-4-hydrogen` | `direct-google-gemini-4-hydrogen`, `smart-gemini-chat`, `smart-route`, `smart_gemini` |
| `direct-google-gemma-4-26b-a4b-it` | `direct-google-gemma-4-26b-a4b-it`, `smart-gemini-bounded`, `smart-judge`, `smart-nt`, `smart_gemini` |
| `direct-google-gemma-4-31b-it` | `direct-google-gemma-4-31b-it`, `smart-gemini-bounded`, `smart-judge`, `smart-nt`, `smart_gemini` |
| `htpc-qwen3_5-9b-mtp` | `direct-htpc-qwen3_5-9b-mtp`, `smartlocal`, `smartlocal-nt` |
| `htpc-qwen3_5-9b-mtp-nt` | `direct-htpc-qwen3_5-9b-mtp-nt`, `smart-judge`, `smart-nt` |
| `openai-gpt-6-luna-nt` | `direct-openai-gpt-6-luna-nt`, `smart-judge`, `smart-nt`, `smart-openai-nt` |
| `openrouter-ernie-5_5-flash-preview-free` | `direct-openrouter-ernie-5_5-flash-preview-free`, `smart-openrouter-free`, `smart-route`, `smartfree` |
| `openrouter-glm-5_3-flash` | `direct-openrouter-glm-5_3-flash`, `smart-openrouter-paid`, `smart-route` |
| `openrouter-ling-3_2-flash-free` | `direct-openrouter-ling-3_2-flash-free`, `smart-openrouter-free`, `smart-route`, `smartfree` |
| `openrouter-nemotron-3_5-super-vl-free` | `direct-openrouter-nemotron-3_5-super-vl-free`, `smart-openrouter-free`, `smart-route`, `smartfree` |
| `openrouter-openrouter-free` | `direct-openrouter-openrouter-free`, `smartfree` |
| `openrouter-tencent-hy4-flash` | `direct-openrouter-tencent-hy4-flash`, `smart`, `smart-openrouter-paid`, `smart-route` |
| `stephen-dots-3-note-preview-free` | `ST_Free`, `direct-openrouter-stephen-dots-3-note-preview-free` |
| `stephen-ernie-5_5-flash-preview-free` | `ST_Free`, `direct-openrouter-stephen-ernie-5_5-flash-preview-free` |
| `stephen-flint-v1-0-free` | `ST_Free`, `direct-openrouter-stephen-flint-v1-0-free` |
| `stephen-gemma-4-26b-a4b-it-free` | `ST_Free`, `direct-openrouter-stephen-google-gemma-4-26b-a4b-it-free` |
| `stephen-gemma-4-31b-it-free` | `ST_Free`, `direct-openrouter-stephen-google-gemma-4-31b-it-free` |
| `stephen-google-gemini-3.5-flash-lite` | `ST_Free`, `ST_Gemini_Free`, `direct-google-stephen-gemini-3.5-flash-lite` |
| `stephen-ling-3_2-flash-free` | `ST_Free`, `direct-openrouter-stephen-ling-3_2-flash-free` |
| `stephen-nemotron-3-5-super-vl-free` | `ST_Free`, `direct-openrouter-stephen-nemotron-3-5-super-vl-free` |
| `stephen-paid-deepseek-v4-1-flash-1017` | `ST_Paid`, `direct-openrouter-stephen-deepseek-v4-1-flash-1017` |
| `stephen-paid-glm-5-3-flash` | `ST_Paid`, `direct-openrouter-stephen-glm-5-3-flash` |

Single-use targets: 29 (`comfyninja-qwen3_8-27b-q4-mtp`, `direct-google-gemini-3-flash`, `direct-google-gemini-decision-1`, `direct-htpc-jev-style`, `direct-htpc-laya`, `direct-openrouter-span-01-lite`, `direct-openrouter-stephen-span-01-lite`, `htpc-qwen3_5-9b`, `openai-gpt-6-luna`, `openai-gpt-6-sol`, `openrouter-deepseek-v4_1-flash-1017`, `openrouter-dots-3-note-preview-free`, `openrouter-flint-v1-0-free`, `openrouter-google-gemma-4-26b-a4b-it-free`, `openrouter-google-gemma-4-31b-it-free`, `openrouter-qwen3_7-flash`, `openrouter-qwen3_8-flash`, `stephen-google-gemini-3-flash-preview`, `stephen-google-gemini-3.1-flash-lite`, `stephen-google-gemini-3.6-flash`, `stephen-google-gemini-3.8-flash`, `stephen-google-gemini-4-hydrogen`, `stephen-google-gemma-4-26b-a4b-it`, `stephen-google-gemma-4-31b-it`, `thaillm-openthaigpt`, `thaillm-qwen3_5-9b`, `thaillm-qwen3_6-35b-a3b`, `thaillm-thalle`, `thaillm-typhoon`). Reached only by a decision lane: `direct-google-gemini-decision-1`, `direct-htpc-jev-style`, `direct-htpc-laya`, `direct-openrouter-span-01-lite`, `direct-openrouter-stephen-span-01-lite`.

## References that are not targets

A composite names routes, so its children are a call path rather than a failure domain.
They are listed here, resolved, and never counted as targets:

- `composite_child`: `ST_Free`, `ST_Paid`, `direct-deepseek-deepseek-flash-nt`, `direct-google-gemini-3.1-flash-lite`, `direct-google-gemini-3.5-flash-lite`, `direct-openai-gpt-6-luna-nt`, `direct-openrouter-glm-5_3-flash`, `direct-openrouter-openrouter-free`, `vc-paid-core`

Capability-client models that are not `[targets.*]` entries: 19 of 21 — a separate namespace (a vendor model string, not a routing target), so the target partition does not include them.

## Consumer destination

A registry row is a consumer of this router only when its `base_url` points here. The
table separates the rows addressed to this process from rows that name an upstream
model and reach the provider directly; the second group is invisible to this manifest
and is not counted as traffic it serves.

| destination | aliases |
|---|---|
| this router | 69 |
| `deepseek` | 1 |
| `openai` | 3 |
| `openrouter` | 13 |
| `private_engine` | 4 |
| `thaillm` | 1 |

Rows addressed to this router whose model name is declared nowhere in the manifest: none.

## Route → consumer ceiling

The route's declared window is the router's admission bound. A consumer's ceiling is
what Turnstone will accept before any request is sent. A positive delta means the
consumer would admit a request the route refuses; a negative delta means the consumer
is stricter than the router.

| route | route ctx | consumers (alias: ceiling) | delta |
|---|---|---|---|
| `ST_Free` | 262144 | `ST_Free`: 262144 | 0 |
| `ST_Paid` | 1048576 | `ST_Paid`: 1048576 | 0 |
| `direct-comfyninja-qwen3_8-27b-q4-mtp` | 179712 | `direct-comfyninja-qwen3_8-27b-q4-mtp`: 179712 | 0 |
| `direct-comfyninja-qwen3_8-27b-q4-mtp-vl` | 152832 | `direct-comfyninja-qwen3_8-27b-q4-mtp-vl`: 152832 | 0 |
| `direct-deepseek-deepseek-flash` | 1048576 | `direct-deepseek-deepseek-flash`: 1048576 | 0 |
| `direct-deepseek-deepseek-flash-nt` | 1048576 | `direct-deepseek-deepseek-flash-nt`: 1048576 | 0 |
| `direct-deepseek-deepseek-v4-pro` | 1048576 | `direct-deepseek-deepseek-v4-pro`: 1048576 | 0 |
| `direct-google-gemini-3-flash` | 1048576 | `direct-google-gemini-3-flash`: 1048576 | 0 |
| `direct-google-gemini-3.1-flash-lite` | 1048576 | `direct-google-gemini-3.1-flash-lite`: 1048576 | 0 |
| `direct-google-gemini-3.5-flash-lite` | 1048576 | `direct-google-gemini-3.5-flash-lite`: 1048576 | 0 |
| `direct-google-gemini-3.6-flash` | 1048576 | `direct-google-gemini-3.6-flash`: 1048576 | 0 |
| `direct-google-gemini-3.8-flash` | 1048576 | `direct-google-gemini-3.8-flash`: 1048576 | 0 |
| `direct-google-gemini-4-hydrogen` | 1048576 | `direct-google-gemini-4-hydrogen`: 1048576 | 0 |
| `direct-google-gemma-4-26b-a4b-it` | 262144 | `direct-google-gemma-4-26b-a4b-it`: 262144 | 0 |
| `direct-google-gemma-4-31b-it` | 262144 | `direct-google-gemma-4-31b-it`: 262144 | 0 |
| `direct-htpc-qwen3_5-9b` | 81920 | `direct-htpc-qwen3_5-9b`: 81920 | 0 |
| `direct-htpc-qwen3_5-9b-mtp` | 81920 | `direct-htpc-qwen3_5-9b-mtp`: 81920 | 0 |
| `direct-htpc-qwen3_5-9b-mtp-nt` | 81920 | `direct-htpc-qwen3_5-9b-mtp-nt`: 81920 | 0 |
| `direct-openai-gpt-6-luna` | 1050000 | `direct-openai-gpt-6-luna`: 1050000 | 0 |
| `direct-openai-gpt-6-luna-nt` | 1050000 | `direct-openai-gpt-6-luna-nt`: 1050000 | 0 |
| `direct-openai-gpt-6-sol` | 1050000 | `direct-openai-gpt-6-sol`: 1050000 | 0 |
| `direct-openrouter-deepseek-v4.1-flash-1017` | 1048576 | `direct-openrouter-deepseek-v4.1-flash-1017`: 1048576 | 0 |
| `direct-openrouter-dots-3-note-preview-free` | 262144 | `direct-openrouter-dots-3-note-preview-free`: 262144 | 0 |
| `direct-openrouter-ernie-5_5-flash-preview-free` | 1024000 | `direct-openrouter-ernie-5_5-flash-preview-free`: 1024000 | 0 |
| `direct-openrouter-flint-v1-0-free` | 262144 | `direct-openrouter-flint-v1-0-free`: 262144 | 0 |
| `direct-openrouter-gemma-4-26b-a4b-it-free` | 262144 | `direct-openrouter-gemma-4-26b-a4b-it-free`: 262144 | 0 |
| `direct-openrouter-gemma-4-31b-it-free` | 262144 | `direct-openrouter-gemma-4-31b-it-free`: 262144 | 0 |
| `direct-openrouter-glm-5_3-flash` | 1048576 | `direct-openrouter-glm-5_3-flash`: 1048576 | 0 |
| `direct-openrouter-ling-3_2-flash-free` | 1048576 | `direct-openrouter-ling-3_2-flash-free`: 1048576 | 0 |
| `direct-openrouter-nemotron-3_5-super-vl-free` | 1000000 | `direct-openrouter-nemotron-3_5-super-vl-free`: 1000000 | 0 |
| `direct-openrouter-openrouter-free` | 200000 | `direct-openrouter-openrouter-free`: 200000 | 0 |
| `direct-openrouter-qwen3_7-flash` | 1000000 | `direct-openrouter-qwen3_7-flash`: 1000000 | 0 |
| `direct-openrouter-qwen3_8-flash` | 1000000 | `direct-openrouter-qwen3_8-flash`: 1000000 | 0 |
| `direct-openrouter-stephen-deepseek-v4-1-flash-1017` | 1048576 | `direct-openrouter-stephen-deepseek-v4-1-flash-1017`: 1048576 | 0 |
| `direct-openrouter-stephen-dots-3-note-preview-free` | 262144 | `direct-openrouter-stephen-dots-3-note-preview-free`: 262144 | 0 |
| `direct-openrouter-stephen-ernie-5_5-flash-preview-free` | 1024000 | `direct-openrouter-stephen-ernie-5_5-flash-preview-free`: 262144 | -761856 |
| `direct-openrouter-stephen-flint-v1-0-free` | 262144 | `direct-openrouter-stephen-flint-v1-0-free`: 262144 | 0 |
| `direct-openrouter-stephen-glm-5-3-flash` | 1048576 | `direct-openrouter-stephen-glm-5-3-flash`: 1048576 | 0 |
| `direct-openrouter-stephen-google-gemma-4-26b-a4b-it-free` | 262144 | `direct-openrouter-stephen-google-gemma-4-26b-a4b-it-free`: 262144 | 0 |
| `direct-openrouter-stephen-google-gemma-4-31b-it-free` | 262144 | `direct-openrouter-stephen-google-gemma-4-31b-it-free`: 262144 | 0 |
| `direct-openrouter-stephen-ling-3_2-flash-free` | 1048576 | `direct-openrouter-stephen-ling-3_2-flash-free`: 262144 | -786432 |
| `direct-openrouter-stephen-nemotron-3-5-super-vl-free` | 1000000 | `direct-openrouter-stephen-nemotron-3-5-super-vl-free`: 262144 | -737856 |
| `direct-thaillm-openthaigpt` | 16384 | `direct-thaillm-openthaigpt`: 16384 | 0 |
| `direct-thaillm-qwen3_5-9b` | 32768 | `direct-thaillm-qwen3_5-9b`: 32768 | 0 |
| `direct-thaillm-qwen3_6-35b-a3b` | 32768 | `direct-thaillm-qwen3_6-35b-a3b`: 32768 | 0 |
| `direct-thaillm-thalle` | 16384 | `direct-thaillm-thalle`: 16384 | 0 |
| `direct-thaillm-typhoon` | 16384 | `direct-thaillm-typhoon`: 16384 | 0 |
| `smart` | 1000000 | `smart`: 262144 | -737856 |
| `smart-gemini-bounded` | 262144 | `smart-gemini-bounded`: 262144 | 0 |
| `smart-gemini-chat` | 262144 | `smart-gemini-chat`: 262144 | 0 |
| `smart-judge` | 12307 | `smart-judge`: 12307 | 0 |
| `smart-nt` | 65536 | `smart-nt`: 65536 | 0 |
| `smart-route` | 262144 | `smart-route`: 262144 | 0 |
| `smart_gemini` | 1048576 | `smart_gemini`: 262144 | -786432 |
| `smartfree` | 1000000 | `smartfree`: 1000000 | 0 |
| `smartlocal` | 152832 | `smartlocal`: 152832 | 0 |
| `smartlocal-nt` | 152832 | `smartlocal-nt`: 152832 | 0 |
| `vc-paid-core` | 1048576 | `vc-paid-core`: 1048576 | 0 |

5 route/consumer pair(s) disagree; 5 of them have the consumer stricter, so the request never reaches the router — the safe direction, but it means the route's advertised window is not what that alias can use.

## Routes without a consumer

Advertised and callable, but no Turnstone alias names them. Not a defect — a route can
be reached by another router or by a direct client — but it is the set to check before
retiring anything.

`ST_Gemini_Free`, `direct-google-stephen-gemini-3-flash-preview`, `direct-google-stephen-gemini-3.1-flash-lite`, `direct-google-stephen-gemini-3.5-flash-lite`, `direct-google-stephen-gemini-3.6-flash`, `direct-google-stephen-gemini-3.8-flash`, `direct-google-stephen-gemini-4-hydrogen`, `direct-google-stephen-gemma-4-26b-a4b-it`, `direct-google-stephen-gemma-4-31b-it`, `direct-openrouter-tencent-hy4-flash`, `smart-deepseek`, `smart-deepseek-nt`, `smart-openai-nt`, `smart-openrouter-free`, `smart-openrouter-paid`

## Not established here

- Vendor model cards, transport proofs and traffic observation are not part of this
  catalogue; the master direct panel keeps those grades and this document makes no
  claim about them.
- A route's declared capability is an admission gate. It is not evidence the upstream
  accepts the modality.
