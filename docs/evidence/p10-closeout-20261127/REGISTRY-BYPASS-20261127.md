# Registry bypass aliases and unregistered routes - 2026-11-27

Measured from the local Turnstone PostgreSQL (`model_definitions`, `workstream_config`,
`usage_events`) inside a `REPEATABLE READ, READ ONLY` transaction at 2026-11-27T08:5xZ.
Read-only: no row was modified. The owner decision of 2026-11-27 retains all 22 aliases.

## 1. The 22 provider-direct bypass aliases (rule: `provider <> 'openai-compatible' OR model <> alias`)

| Alias | Provider | Upstream host | Upstream model | Credential class | Lifetime calls | Pinned as `model_alias` | Purpose disposition |
|---|---|---|---|---|---|---|---|
| `comfyninja-qwen3.8-27b-q4-mtp` | `openai-compatible` | reninja | `comfyninja/qwen3.8-27b-q4-mtp` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `comfyninja-qwen3.8-27b-q4-mtp-vl` | `openai-compatible` | reninja | `comfyninja/qwen3.8-27b-q4-mtp-vl` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `deepseek-deepseek-flash` | `openai-compatible` | api.deepseek.com | `deepseek-flash` | static | 0 | 29 | Historical direct DeepSeek lane: pinned by 29 **closed** workstreams; 0 recorded completions under this name. Purpose beyond that: UNKNOWN. |
| `gpt-6.1-sol` | `openai` | api.openai.com | `gpt-6.1-sol` | openai_oauth | 0 | 1 | Direct OpenAI OAuth lane: pinned by 1 **closed** workstream; 0 recorded completions. Purpose beyond that: UNKNOWN. |
| `gpt-6-luna` | `openai` | api.openai.com | `gpt-6-luna` | openai_oauth | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `gpt-6-luna-nt` | `openai` | api.openai.com | `gpt-6-luna` | openai_oauth | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `htpc-qwen3.5-9b-mtp` | `openai-compatible` | htpc-wsl | `htpc/qwen3.5-9b-mtp` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `htpc-Qwen3.5-9B-Q4_0` | `openai-compatible` | htpc-wsl | `htpc/qwen3.5-9b` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-deepseek-v4.1-flash` | `openai-compatible` | openrouter.ai | `deepseek/deepseek-v4.1-flash` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-deepseek-v4.1-flash-1017` | `openai-compatible` | openrouter.ai | `deepseek/deepseek-v4.1-flash-1017` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-dots-3-note-preview-free` | `openai-compatible` | openrouter.ai | `dots-studio/dots-3-note-preview:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-ernie-5.5-flash-preview-free` | `openai-compatible` | openrouter.ai | `baidu/ernie-5.5-flash-preview:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-flint-v1.0-free` | `openai-compatible` | openrouter.ai | `recursive/flint-v1.0:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-gemma-4-26b-a4b-it-free` | `openai-compatible` | openrouter.ai | `google/gemma-4-26b-a4b-it:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-gemma-4-31b-it-free` | `openai-compatible` | openrouter.ai | `google/gemma-4-31b-it:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-glm-5.3-flash` | `openai-compatible` | openrouter.ai | `z-ai/glm-5.3-flash` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-lfm-2.5-2.6b-free` | `openai-compatible` | openrouter.ai | `liquid/lfm-2.5-2.6b:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-ling-3.2-flash-free` | `openai-compatible` | openrouter.ai | `inclusionai/ling-3.2-flash:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-nemotron-3.5-super-vl-free` | `openai-compatible` | openrouter.ai | `nvidia/nemotron-3.5-super-vl:free` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-qwen3.7-flash` | `openai-compatible` | openrouter.ai | `qwen/qwen3.7-flash` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `openrouter-qwen3.8-flash` | `openai-compatible` | openrouter.ai | `qwen/qwen3.8-flash` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |
| `thaillm-typhoon` | `openai-compatible` | thaillm.or.th | `Typhoon-S-ThaiLLM-8B-Instruct` | static | 0 | 0 | UNKNOWN - no call, no pin, no artefact establishes a purpose. |

**Credential scope.** `api_key` is store-managed and is not readable in this comparison, so scope is
recorded from naming evidence only: the 13 `openrouter-*` rows carry no `stephen-` prefix, consistent
with owner scope (the Stephen lanes are the separate `direct-openrouter-stephen-*` rows), the three
`gpt-6*` rows use `openai_oauth` (the owner OAuth grant), and the four local rows point at the two
owner-operated GPU hosts. This is a naming-and-auth-mode inference, not a credential comparison.

**Compatibility role.** All 22 appear in the `/v1/api/models` projection, so they remain selectable
client-visible names. Several are not interchangeable with the similarly named Switchyard leaves
because they draw on different credentials.

**Lifetime usage.** 0 calls under any of the 22 names across 143,557 `usage_events`
(2026-10-16T09:41:42Z → 2026-11-27T08:49:20Z). No active workstream pins any of them: the only two
with pinned references are pinned exclusively by closed workstreams (last updated 2026-11-16).

## 2. The 16 advertised-but-unregistered routes

Deliberately not registered. Advertisement does not require a Turnstone row; a name is registered
only when an authorized consumer needs it.

**Direct (9):** `direct-openrouter-tencent-hy4-flash` and eight `direct-google-stephen-*` leaves.

**Non-direct (7):** `ST_Gemini_Free`, `smart-deepseek`, `smart-deepseek-flash`, `smart-deepseek-nt`,
`smart-openai-nt`, `smart-openrouter-free`, `smart-openrouter-paid`.

## 3. The 5 registered-only aliases (preserved)

`ST_Route`, `VC_Paid`, `ST_Rerank`, `smartfree-rerank`, `smartlocal-rerank` - registered but not
advertised by the deployed manifest; retained as compatibility and capability rows.

## 4. Census cross-check (2026-11-27T08:50Z)

91 registered definitions = 69 Switchyard-bound + 22 bypass. Of the 69: 58 advertised and registered,
5 registered-only, 6 typed-decision lanes. 74 advertised − 16 unregistered = 58; 58 + 5 + 6 = 69.
