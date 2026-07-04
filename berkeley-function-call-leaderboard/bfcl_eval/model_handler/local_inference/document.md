# Qwen Handler — Technical Reference

This document covers all Qwen-related handler files in
`bfcl_eval/model_handler/local_inference/`.  It is the authoritative reference
for how coached inference works end-to-end, from vLLM server management through
prompt formatting, coaching loops, and inference log structure.

---

## 1. Class Hierarchy

```
BaseHandler  (base_handler.py)
└── OSSHandler  (base_oss_handler.py)
    └── QwenFCHandler  (qwen_fc.py)
        ├── QwenFCHintedHandler  (qwen_fc_hinted.py)   — pre-exec judge+hinter (two calls)
        ├── QwenFCV1CoachHandler (qwen_fc_v1_coach.py) — post-exec single coach
        ├── QwenFCV2CoachHandler (qwen_fc_v2_coach.py) — pre-exec single coach + XML complete-check
        └── QwenFCV3CoachHandler (qwen_fc_v3_coach.py) — pre-exec single coach + silent final-coach
```

The four coaching handlers are all drop-in replacements for `QwenFCHandler`.
They override only `inference_multi_turn_prompting`; everything else (chat
template, decode, vLLM API calls) is inherited unchanged.

---

## 2. OSSHandler — `base_oss_handler.py`

### Role
Abstract base for all open-source model handlers.  Manages the vLLM/SGLang
server lifecycle and implements the BFCL prompting protocol.

### Server lifecycle

`spin_up_local_server(num_gpus, gpu_memory_utilization, backend, ...)`:
- Loads tokenizer and config to determine `max_context_length`.
- If `skip_server_setup=False`, launches a `vllm serve` or `python -m sglang.launch_server` subprocess.
- Starts two daemon threads streaming stdout/stderr until the `/v1/models` endpoint responds 200.
- Supports LoRA via `--enable-lora / --lora-modules / --max-lora-rank`.
- Supports remote vLLM endpoints via `REMOTE_OPENAI_BASE_URL` / `REMOTE_OPENAI_API_KEY` env vars.

`shutdown_local_server()`: terminates the process and joins the log threads.

### Prompting protocol

`_query_prompting(inference_data)`:
1. Calls `self._format_prompt(message, function)` — **abstract**; subclasses implement it.
2. Tokenizes the formatted prompt to count tokens.
3. Caps `max_tokens` at `min(4096, max_context_length - input_tokens - 2)`.
4. Calls `client.completions.create(...)` — OpenAI Completions API (not Chat).
5. Returns `(api_response, latency_seconds)`.

`_pre_query_processing_prompting`: injects BFCL system prompt via
`system_prompt_pre_processing_chat_model`; returns `{"message": [], "function": functions}`.

`_parse_query_response_prompting`: returns `{"model_responses": text, "input_token": N, "output_token": N}`.

Multi-turn message accumulation methods (`add_first_turn_message_prompting`,
`_add_next_turn_user_message_prompting`, `_add_assistant_message_prompting`,
`_add_execution_results_prompting`) all simply append to `inference_data["message"]`.

---

## 3. QwenFCHandler — `qwen_fc.py`

### Role
Implements the Qwen2.5 chat template manually (as a Python string concatenation)
so we have full control over formatting — needed because vLLM's Completions API
doesn't apply the template automatically.

### `_format_prompt(messages, function)`

The template mirrors the Jinja2 template shipped with Qwen2.5-Instruct exactly:

**System block (tool-use mode)**:
```
<|im_start|>system
{system message if present}

# Tools
...tool schema JSON...
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

**`last_query_index` logic**: scans messages in reverse to find the last user
message whose content does NOT start with `<tool_response>` and end with
`</tool_response>`.  This index separates the "historical multi-turn" portion
from the "active turn" portion.

**Assistant messages**:
- Before `last_query_index`: written as `<|im_start|>assistant\n{content}`.
  No `<think>` block forced even if `reasoning_content` is present — historical
  turns are reproduced verbatim.
- From `last_query_index` onward (the active turn):
  - If it is the last message OR `reasoning_content` is present: wraps in
    `<think>\n{reasoning}</think>\n\n{content}`.
  - Otherwise: written without `<think>`.
- `tool_calls` in an assistant message are serialised as `<tool_call>\n{...}\n</tool_call>` blocks.

**Tool result messages**: multiple consecutive `role=tool` messages are wrapped
in a single `<|im_start|>user` block; each result gets its own `<tool_response>` tag.

**`<think>` / reasoning_content handling**: if the model response contains
`</think>`, the handler splits on it: text before `<think>` … `</think>` becomes
`reasoning_content`; text after becomes the visible `content`.

### `_parse_query_response_prompting`

Returns `model_responses_message_for_chat_history` as:
- `{"role": "assistant", "tool_calls": [...], "content": ""}` if tool calls were found.
- `{"role": "assistant", "content": cleaned_response}` otherwise.
Both forms always include `"reasoning_content"`.

### `_extract_tool_calls(input_string)` (static)

Regex `<tool_call>\n(.*?)\n</tool_call>` with `re.DOTALL`.  Parses each match
as JSON `{"name": ..., "arguments": ...}`.  Silently skips malformed matches.

### `decode_ast` / `decode_execute`

Both call `_extract_tool_calls`.  `decode_execute` additionally calls
`convert_to_function_call` to produce BFCL execution format.

---

## 4. Common Patterns in Coaching Handlers

All four coaching handlers share the same infrastructure.

### Environment variables

| Variable | Purpose |
|---|---|
| `JUDGE_API` | `"openai"` or `"anthropic"` |
| `JUDGE_MODEL` | e.g. `"gpt-4o-2024-11-20"`, `"claude-opus-4-7"` |
| `HINT_LOG_PATH` | JSONL file for coaching events (optional) |
| `HINT_DEBUG` | `"1"` or `"true"` for verbose stdout |

### Concurrency

`_judge_semaphore = threading.Semaphore(max_judge_concurrency)` — caps
concurrent coaching API calls.  Worst case: `num_workers × max_hint_retries × 2`
parallel requests per handler instance.

`_hint_log_queue` — a `queue.Queue` consumed by a single daemon writer thread
(`_hint_log_writer`).  All inference worker threads only enqueue; one thread
does all JSONL file I/O so writes never interleave.  `shutdown_hint_log()` sends
a `None` sentinel and calls `queue.join()` to drain before process exit.
Per test-entry: `queue.join()` is also called at the end of each
`inference_multi_turn_prompting` call to guarantee log completeness without
blocking the queue.

### `_serialize_state(involved_instances)` → str

JSON dump of `{class_name: {all_attrs_except_api_description}}` for all active backend instances.
Only `_api_description` is excluded (boilerplate string, not useful to the coach); all other attributes — including private ones such as `_current_dir` and `_brakePedalForce` — are exposed.  `_current_dir` is serialised as a readable path string rather than a raw object repr.  Passed to coaching prompts so the coach can reason about current system state.

### `_fmt_messages(msgs)` → str

Converts a message list to a readable text representation for LLM prompts.
Includes user messages, assistant tool calls, and tool results.  Omits
assistant text-only messages.  V1 additionally skips user messages that start
with `"Supervisor guidance"` so the coach doesn't see its own prior instructions.

### `_build_state_log(involved_instances)` → list[dict]

Produces `{"role": "state_info", "class_name": ..., "content": {...}}` entries
for stateful backend instances (skipping `STATELESS_CLASSES` and
`OMIT_STATE_INFO_CLASSES`).  Uses `deepcopy` so future mutations don't affect
the log.

### Inference log structure

Every coaching handler produces the same log shape as `QwenFCHandler`:

```
all_inference_log = [
  [{"role": "state_info", ...}],          # initial state (if not exclude_state_log)
  {                                        # turn 0
    "begin_of_turn_query": [...],
    "step_0": [
      {"role": "assistant", "content": "...", "reasoning_content": "..."},
      {"role": "inference_input", ...},    # if include_input_log
      {"role": "handler_log", "content": "...", "model_response_decoded": [...],
       "hinted_step": bool, "final_verdict": str, "coach_latency_s": float},
      {"role": "tool", "content": "..."},
    ],
    "step_1": [...],
    ...
  },
  [{"role": "state_info", ...}],          # state after turn 0
  ...
]
```

Key fields in `handler_log`:
- `hinted_step: bool` — True if the coaching LLM intervened at least once during this step.
- `final_verdict: str` — last verdict from the coaching LLM (`"Good"`, `"Bad"`, `"Done"`, `"NotDone"`, `"coached"`, `"silent"`).
- `coach_latency_s: float` — sum of latencies (seconds) for all coaching LLM calls made during this step.

**Important**: hints, rejected FCs, and intermediate retries are **never** written
to `all_inference_log`.  The log is a clean training trajectory — only accepted
FCs and their execution results appear.

---

## 5. QwenFCHintedHandler — `qwen_fc_hinted.py`

### Strategy: pre-execution, two-step judge → hinter

After the small model produces a function call (before executing it):
1. **JUDGE_PROMPT** → evaluates correctness against 8 yes/no criteria.
   Outputs `<verdict>Good/Bad</verdict>` + plain explanation if Bad.
2. **HINTER_PROMPT** → receives the judge's full output, distils it into one
   actionable sentence without revealing the correct answer.

If the model outputs plain text (no tool call):
- **COMPLETE_PROMPT** → asks whether the task is truly done or if a tool call
  was needed.  `<verdict>Done/NotDone</verdict>` + `<hint>...</hint>`.

### `step_start_idx` invariant

```
step_start_idx = len(inference_data["message"])   # set once per step

while hint_retry <= max_hint_retries:
    del inference_data["message"][step_start_idx:]  # reset to clean boundary
    if latest_hint is not None:
        inference_data["message"].append({"role": "user", "content": latest_hint})
    # query small model ...
    # judge / complete-check ...

del inference_data["message"][step_start_idx:]  # final cleanup (remove last hint too)
```

The small model's context is always exactly: `[committed history] + [0 or 1 hint]`.
It never sees previous wrong FCs or stale hints.  After the loop the last hint is
also removed, so only the accepted FC gets committed to history.

### Hint retry flow (FC case)

```
hint_retry=0..max_hint_retries:
  query → fc_message
  if no tool call → complete_check → Done: break | NotDone: inject hint, continue
  if hint_retry < max_hint_retries:
    judge → Good: accept | Bad: hinter → hint, inject, retry
  else:
    accept (retries exhausted)
```

### Hint log events

- `type="judge_hinter"` — emitted when judge returns Bad; includes `fc_message`, `judge_raw`, `hint`.
- `type="complete_check"` — emitted on every text-only response; includes verdict, hint.

---

## 6. QwenFCV1CoachHandler — `qwen_fc_v1_coach.py`

### Strategy: post-execution, single combined coach

**No pre-execution check.  No step_start_idx.  No retry loop.**

The FC is executed unconditionally.  After execution, the coach sees the full
updated history (including the tool result) and decides whether to intervene:
- Empty / `"SILENT"` → stay silent; model proceeds normally next step.
- Any text → injected as `"Supervisor guidance for the next step only: {instruction}"`.

The same coach is also called when the model gives a text-only response
(`phase="after_attempted_final_answer"`).  If it returns an instruction,
the model is re-queried (counts as the next step).

### Key differences from hinted

- Coach runs **after** execution, not before.
- Only one LLM call per step (no separate judge/hinter split).
- Wrong FCs are impossible to intercept — the tool has already run.
- `_fmt_messages` filters out `"Supervisor guidance"` messages so the coach
  doesn't see its own prior instructions when composing new ones.
- `max_coach_instructions=0` means unlimited; set > 0 to budget instructions
  per task (tracked across all turns via `coach_instructions_total`).

### Inference log

Coach instructions **are** written to the inference log as user messages before
the step they affect.  `hinted_step` is True if the previous step had a coach
instruction injected.

---

## 7. QwenFCV2CoachHandler — `qwen_fc_v2_coach.py`

### Strategy: pre-execution, single combined coach + XML complete-check

Structurally identical to `QwenFCHintedHandler` except:
- The two-step JUDGE → HINTER pipeline is replaced by a **single COACH call**.
- COACH returns empty/SILENT to approve; any text is the hint.
- Complete-check still uses **COMPLETE_PROMPT** with `<verdict>Done/NotDone</verdict>` XML.

The `step_start_idx` invariant is identical to hinted.

### Prompt schema

**COACH_SYSTEM**: reviewer role; silent=approve, text=hint; never reveal correct answer.

**COACH_PROMPT** fields (in prompt order, matching prefix-cache optimization):
- `schema` (tool schema — largest field and constant for the entire test case; placed first to maximise prefix-cache hits)
- `user_request` (current-turn user request)
- `history` (committed turns)
- `prev_step_result` (last committed step's FC + tool result, computed once at step start before the retry sub-loop; gives the coach visibility into the previous step's execution outcome to detect state errors or failed calls)
- `previous_hints` (hint history accumulated this step)
- `current_state` (current system state)
- `tool_name`, `tool_args` (the planned FC — placed last to minimise cache invalidation)

**COMPLETE_PROMPT** fields (in prompt order): `schema`, `user_request`, `history`, `current_state`.

### Hint log events

- `type="v2_coach"` — emitted when coach returns a hint.
- `type="complete_check"` — same as hinted.

---

## 8. QwenFCV3CoachHandler — `qwen_fc_v3_coach.py`

### Strategy: pre-execution single coach + silent final-coach (no XML)

Identical to V2 for valid function calls (COACH_PROMPT → silent approve or hint).

Difference: when the model produces a text-only response, instead of
COMPLETE_PROMPT with XML verdict tags, V3 uses:

**FINAL_COACH_SYSTEM/PROMPT**: silent=accept (turn done), any text=retry instruction.
No `<verdict>Done/NotDone</verdict>` — accept/reject is implicit in whether the
response is empty.

**FINAL_COACH_PROMPT** fields (in prompt order): `schema`, `user_request`, `history`, `current_state`, `model_response`.

**COACH_PROMPT (V3)** field order is identical to V2: `schema`, `user_request`, `history`, `prev_step_result`, `previous_hints`, `current_state`, `tool_name`, `tool_args`.

This removes the XML parsing dependency and unifies the accept/reject signal
across both paths (pre-FC and post-text) into the same "silent means yes" pattern.

### Hint log events

- `type="v3_coach"` — emitted when pre-exec coach returns a hint.
- `type="v3_final_coach"` — emitted on every text-only response.

---

## 9. Handler Comparison Table

| Property | Hinted | V1 Coach | V2 Coach | V3 Coach |
|---|---|---|---|---|
| Timing | pre-exec | post-exec | pre-exec | pre-exec |
| FC acceptance | requires Good verdict | unconditional | silent=approve | silent=approve |
| LLM calls per retry | judge + hinter (2) | coach (1) | coach (1) | coach (1) |
| Retry loop | yes (step_start_idx) | no | yes (step_start_idx) | yes (step_start_idx) |
| Text-only handler | COMPLETE_PROMPT (XML) | coach re-query | COMPLETE_PROMPT (XML) | FINAL_COACH (silent) |
| Coach sees prior wrong FCs | no (context reset) | yes (full history) | no (context reset) | no (context reset) |
| Coach history filtering | n/a | skips "Supervisor guidance" | n/a | n/a |
| hint_log type | judge_hinter, complete_check | v1_coach_tool, v1_coach_final | v2_coach, complete_check | v3_coach, v3_final_coach |

---

## 10. Design Decisions

### Why erase hints from the inference log?

The inference log is the source of truth for SFT training data.
Hints injected during coaching are scaffolding, not part of the target behaviour.
If they appeared in the training trajectory, the model would learn to expect
guidance during inference, where no coach is present.

### Why step_start_idx instead of keeping all retries?

Showing the model its previous wrong FC alongside the hint pollutes its context
with a bad example.  The `step_start_idx` slice ensures the model only sees the
clean committed history plus the single most recent hint.  This also keeps
context length predictable.

### Why a semaphore for coaching API calls?

Multiple BFCL test entries run in parallel worker threads.  Without a semaphore,
worst case is `num_workers × max_hint_retries × 2` simultaneous coaching requests,
which easily exceeds rate limits.  The semaphore caps the burst.

### Why a dedicated writer thread for hint_log?

Multiple inference workers writing to the same JSONL file would interleave partial
writes without locking.  A single queue-backed writer thread serialises all I/O
with zero lock contention in the workers.

### Tool call format: flat vs OpenAI-wrapped

`QwenFCHandler` produces tool calls in the "flat" format:
```json
{"name": "fn_name", "arguments": {...}}
```
The coaching handlers handle both flat and OpenAI-wrapped formats:
```json
{"function": {"name": "fn_name", "arguments": {...}}}
```
All `_tc_name_args` helpers check for the `"function"` key and unwrap accordingly.
This matters for compatibility with `collect_snapshot` in the SFT data preparation
scripts (which produce OpenAI format).
