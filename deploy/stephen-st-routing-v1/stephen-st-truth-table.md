# Stephen ST direct-leaf truth table - 2026-11-22

Baseline `20261122T050658Z-bf25c056b4598e65-1071be2f`, source `0779664`, manifest `1071be2fcbda` (103416 bytes).

Evidence codes: **C** provider catalogue states it · **M** measured on this deployment's wire ·
**D** declared in configuration, not tested in this batch · **U** no authoritative evidence ·
**GAP** provider declares the modality but the target refuses it, unqualified · **-** the
provider exposes no such modality. "Traffic" says whether the 30-day routing log shows this
model serving any ST* request; it is never evidence of a capability.

| Direct leaf | Upstream id | Tariff | Route ctx | Max out | Vision | Video | Tools | Traffic | Disposition |
|---|---|---|---:|---:|---|---|---|---|---|
| `…ernie-5_5-flash-preview-free` | `baidu/ernie-5.5-flash-preview:free` | free | 1024000 (C) | 921600 | - | - | D | served | INCLUDE |
| `…ling-3_2-flash-free` | `inclusionai/ling-3.2-flash:free` | free | 1048576 (C) | 262144 | C | - | D | served | INCLUDE |
| `…nemotron-3-5-super-vl-free` | `nvidia/nemotron-3.5-super-vl:free` | free | 1000000 (C) | 131072 | M | C | D | served | INCLUDE |
| `…flint-v1-0-free` | `recursive/flint-v1.0:free` | free | 262144 (C) | 235929 | - | - | D | no receipt | INCLUDE |
| `…dots-3-note-preview-free` | `dots-studio/dots-3-note-preview:free` | free | 262144 (U) | - | D | - | D | served | CONDITIONAL |
| `…google-gemma-4-26b-a4b-it-free` | `google/gemma-4-26b-a4b-it:free` | free | 262144 (C) | 32768 | U | GAP | D | no receipt | INCLUDE |
| `…google-gemma-4-31b-it-free` | `google/gemma-4-31b-it:free` | free | 262144 (C) | 32768 | U | GAP | D | no receipt | INCLUDE |
| `…deepseek-v4-1-flash-1017` | `deepseek/deepseek-v4.1-flash-1017` | metered | 1048576 (C) | 393216 | C | - | D | served | INCLUDE |
| `…glm-5-3-flash` | `z-ai/glm-5.3-flash` | metered | 1048576 (C) | 131072 | M | M | D | served | INCLUDE |
| `…openrouter-free` | `openrouter/free` | free | - | - | D | - | D | n/a | REMOVED |

Full per-field provenance, including the three context corrections and the removed dynamic
selector, is in `stephen-st-truth-table.json`.
