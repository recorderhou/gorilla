# Coached Inference Handlers

Four drop-in handler variants that augment the base `QwenFCHandler` with LLM-driven coaching during multi-turn inference. Each handler logs a complete `inference_log` per entry that can be used directly as SFT training data.

---

## Handler Overview

| Handler | Class | Coaching style | When coach acts | Context in log |
|---------|-------|---------------|-----------------|----------------|
| `qwen_fc_hinted` | `QwenFCHintedHandler` | Pre-execution judge + hinter | Before FC is committed; wrong FC triggers retry | Clean — hints erased after acceptance |
| `qwen_fc_v1_coach` | `QwenFCV1CoachHandler` | Post-execution coach (Agent Sampler style) | After FC + tool result; guidance persists in context | Full context incl. guidance as `role=user` |
| `qwen_fc_v2_coach` | `QwenFCV2CoachHandler` | Pre-execution coach | Before FC is committed; coach returns hint or empty | Clean — hints erased after acceptance |
| `qwen_fc_v3_coach` | `QwenFCV3CoachHandler` | Pre-execution coach + final coach | Before FC (coach) and at turn end (final coach) | Clean — hints erased after acceptance |

---

## Handler Details

### `qwen_fc_hinted` — Judge + Hinter (pre-execution)

Each FC the small model produces is evaluated by a judge LLM against 8 criteria (correct tool, valid args, state preconditions, etc.). If the judge returns `Bad`:

1. A hinter LLM distills the judge's explanation into one actionable sentence.
2. The hint is injected as a `role=user` message and the small model retries.
3. Only the accepted FC is committed to context — wrong FCs and hints are erased.

The model can retry up to `--max_hint_retries` times per step.

### `qwen_fc_v1_coach` — Post-execution Coach (Agent Sampler style)

After each FC and its tool result, a coach LLM reviews the full conversation and optionally emits one instruction for the next step. The instruction is prepended as `"Supervisor guidance for the next step only: ..."` before the next model query.

Key properties:
- No pre-execution rejection — every FC the model produces is executed.
- Guidance messages accumulate in context across steps (never stripped), matching Agent Sampler's approach.
- Per-task budget: `--max_coach_instructions` caps total coach interventions per task (0 = unlimited).

### `qwen_fc_v2_coach` — Pre-execution Coach

A single coach LLM both evaluates the FC and produces a hint (no separate judge/hinter). Returns a hint string or empty (silent). Same retry and context-cleanup logic as `qwen_fc_hinted`.

### `qwen_fc_v3_coach` — Pre-execution Coach + Final Coach

Extends V2 with a second coaching call at the end of each turn to decide whether the turn is complete (`Done`) or should continue (`NotDone` + hint). Useful for catching cases where the model stopped too early.

---

## Inference Log Structure

All handlers write an `inference_log` list under `metadata`. Each element is either a list of `state_info` dicts (system state after a turn) or a turn dict:

```
inference_log = [
  [state_info, ...],        # optional — initial state
  {                         # turn 0
    "begin_of_turn_query": [{"role": "user", "content": "..."}],
    "step_0": [...],
    "step_1": [...],
  },
  [state_info, ...],        # optional — state after turn 0
  { ... },                  # turn 1
  ...
]
```

Each step is a list of entries with the following roles:

| Role | Description | Present in |
|------|-------------|-----------|
| `assistant` | The accepted FC (raw `<tool_call>` string) | All handlers |
| `tool` | Tool execution result | All handlers |
| `handler_log` | Metadata: `hinted_step`, `final_verdict`, decoded FC | All handlers |
| `user` | Supervisor guidance (V1 only: `"Supervisor guidance for the next step only: ..."`) | V1 only |
| `inference_input` | Raw model input (only with `--include-input-log`) | All handlers |

### `hinted_step` and `final_verdict`

Every `handler_log` entry that marks a completed FC step carries two fields:

- **`hinted_step`** (`bool`): whether this step required at least one retry (judge/coach rejected and the model was prompted again).
- **`final_verdict`** (`str`): the last LLM evaluation on the accepted response.
  - hinted/v2/v3: `"Good"` or `"Bad"` (FC steps), `"Done"` or `"NotDone"` (text steps)
  - V1: `"coached"` (coach gave an instruction) or `"silent"` (coach stayed silent)

### Example — hinted handler, step with retry

```json
"step_0": [
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"fs.read\", \"arguments\": {\"path\": \"/a\"}}\n</tool_call>"
  },
  {
    "role": "handler_log",
    "content": "Successfully decoded model response.",
    "model_response_decoded": [{"fs.read": {"path": "/a"}}],
    "hinted_step": true,
    "final_verdict": "Good"
  },
  {
    "role": "tool",
    "content": "file contents..."
  }
]
```

### Example — V1 handler, coached step

```json
"step_1": [
  {
    "role": "user",
    "content": "Supervisor guidance for the next step only: Verify the write succeeded by reading the file back."
  },
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"fs.read\", \"arguments\": {\"path\": \"/a\"}}\n</tool_call>"
  },
  {
    "role": "handler_log",
    "content": "Successfully decoded model response.",
    "model_response_decoded": [...],
    "hinted_step": true,
    "final_verdict": "silent"
  },
  {
    "role": "tool",
    "content": "file contents..."
  },
  {
    "role": "handler_log",
    "content": "Post-execution coach.",
    "hinted_step": true,
    "final_verdict": "silent"
  }
]
```

---

## Environment Variables

All handlers read configuration from environment variables if not passed as constructor arguments:

| Variable | Description |
|----------|-------------|
| `JUDGE_API` | `"openai"` or `"anthropic"` |
| `JUDGE_MODEL` | Model ID for the judge/coach/hinter LLM |
| `OPENAI_API_KEY` | Required when `JUDGE_API=openai` |
| `ANTHROPIC_API_KEY` | Required when `JUDGE_API=anthropic` |
| `HINT_LOG_PATH` | Optional path for per-step hint/coach event log |
| `HINT_DEBUG` | Set to `"1"` for verbose per-step console output |

---

## Tests

Test samples for all four handlers are in `bfcl_eval/tests/samples/`. Each file covers multiple branches (happy path, retry, text response, etc.) and can be regenerated after code changes:

```bash
PYTHONPATH=. python bfcl_eval/tests/generate_samples.py
PYTHONPATH=. pytest bfcl_eval/tests/test_coach_handlers.py -v
```

---

## SFT Training Data

The inference logs produced by these handlers feed directly into the snapshot SFT pipeline in `hint-training/`. See `hint-training/README.md` for extraction and training scripts.
