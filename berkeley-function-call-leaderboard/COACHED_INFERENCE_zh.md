# 辅导推理 Handler（Coached Inference Handlers）

四种即插即用的 handler 变体，在多轮推理时为基础 `QwenFCHandler` 加入由 LLM 驱动的辅导机制。每个 handler 为每条记录写入完整的 `inference_log`，可直接用作 SFT 训练数据。

---

## Handler 总览

| Handler | 类名 | Coaching 风格 | Coach 介入时机 | 日志上下文 |
|---------|------|--------------|--------------|-----------|
| `qwen_fc_hinted` | `QwenFCHintedHandler` | 执行前裁判 + 提示生成 | FC 提交前；错误 FC 触发重试 | 干净——提示在接受后抹除 |
| `qwen_fc_v1_coach` | `QwenFCV1CoachHandler` | 执行后 coach（Agent Sampler 风格） | FC + 工具结果后；指导消息保留在上下文中 | 完整上下文，含 `role=user` 形式的指导消息 |
| `qwen_fc_v2_coach` | `QwenFCV2CoachHandler` | 执行前 coach + 轮末 complete-check | FC 提交前（coach）及每轮结束时（XML verdict check） | 干净——提示在接受后抹除 |
| `qwen_fc_v3_coach` | `QwenFCV3CoachHandler` | 执行前 coach + 轮末 final coach | FC 提交前（coach）及每轮结束时（同款 silent/hint 模式） | 干净——提示在接受后抹除 |

---

## Handler 详细说明

### `qwen_fc_hinted` — 裁判 + 提示生成器（执行前）

小模型生成的每个 FC 都由裁判 LLM 按 8 条标准（工具名是否正确、参数是否合法、状态前置条件等）逐一评估。若裁判返回 `Bad`：

1. 提示生成器（Hinter LLM）将裁判的解释提炼为一句可操作的提示。
2. 提示以 `role=user` 消息注入，小模型重新生成。
3. 只有被接受的 FC 才提交到上下文——错误 FC 和提示均被抹除。

每步最多重试 `--max_hint_retries` 次。

**示例——有重试的步骤（`hinted_step=True`，最终被接受）：**

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

> 注意：日志中只保留最终被接受的 FC，重试过程中的错误 FC 和提示消息均已被抹除。

---

### `qwen_fc_v1_coach` — 执行后 Coach（Agent Sampler 风格）

每次 FC 执行完并得到工具结果后，coach LLM 审阅完整对话，并可选择性地为下一步发出一条指令，以 `"Supervisor guidance for the next step only: ..."` 形式插在下次模型查询之前。

关键特性：
- **不拒绝 FC**——模型生成的每个 FC 都会被执行。
- 指导消息在各步骤中**累积**保留（不会被剥离），与 Agent Sampler 方式一致。
- 每任务预算：`--max_coach_instructions` 限制单任务 coach 介入总次数（0 = 不限）。

**示例——coach 发出了指令的步骤（`final_verdict="coached"`）：**

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

> 注意：`role=user` 的 supervisor 指导消息会永久保留在日志上下文中（不被剥离）。SFT 时需用 `strip_guidance=True` 将其去掉，否则模型会依赖这些提示才能正常工作。

---

### `qwen_fc_v2_coach` — 执行前 Coach + XML Complete-Check

单一 coach LLM 同时负责评估 FC 并生成提示（无独立裁判/提示生成器）。返回提示字符串或空字符串（沉默）。重试和上下文清理逻辑与 `qwen_fc_hinted` 相同。

每轮结束时（模型输出 text 而非 FC），用独立的 `COMPLETE_PROMPT` 调用 coach，要求返回 `<verdict>Done</verdict>` 或 `<verdict>NotDone</verdict>` XML；NotDone 时注入提示并触发继续生成。

**Coaching 判断标准：** 按 FC 对完成用户请求的贡献来评估，而非孤立判断调用合法性。请求已满足时，提示须引导模型停止调用工具并给出回复，而非尝试新的调用。仅在有明确、具体问题时介入。

**示例——coach 沉默（`hinted_step=False`，直接接受）：**

```json
"step_0": [
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"fs.write\", \"arguments\": {\"path\": \"/a\", \"content\": \"hello\"}}\n</tool_call>"
  },
  {
    "role": "handler_log",
    "content": "Successfully decoded model response.",
    "model_response_decoded": [{"fs.write": {"path": "/a", "content": "hello"}}],
    "hinted_step": false,
    "final_verdict": "Good"
  },
  {
    "role": "tool",
    "content": "None"
  }
]
```

**示例——coach 介入重试（`hinted_step=True`）：**

```json
"step_1": [
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\": \"fs.read\", \"arguments\": {\"path\": \"/b\"}}\n</tool_call>"
  },
  {
    "role": "handler_log",
    "content": "Coach hint delivered; model retried.",
    "model_response_decoded": [{"fs.read": {"path": "/a"}}],
    "hinted_step": true,
    "final_verdict": "Good"
  },
  {
    "role": "tool",
    "content": "hello"
  }
]
```

> 与 hinted 相同：日志中只保留最终接受的 FC，提示消息已被抹除。

---

### `qwen_fc_v3_coach` — 执行前 Coach + Silent/Hint Final Coach

与 V2 的区别在于轮末检查机制：V3 使用与 FC coaching 相同的 silent/hint 模式（`FINAL_COACH_SYSTEM` + `FINAL_COACH_PROMPT`）——返回空即接受，返回文字即触发继续，无需独立的 XML verdict 结构。Coaching 判断标准与 V2 相同。

**V2 vs V3 对比：**
- V2 轮末：专用 `COMPLETE_PROMPT`，coach 必须回复 `<verdict>Done/NotDone</verdict>`（结构化判断）
- V3 轮末：`FINAL_COACH_SYSTEM`，沉默=接受，有内容=继续（与 FC 前 coaching 风格一致）

**示例——模型输出 text 回答，final coach 判定轮次结束（`Done`）：**

```json
"step_2": [
  {
    "role": "assistant",
    "content": "I have completed all the requested operations."
  },
  {
    "role": "handler_log",
    "content": "Turn ended.",
    "hinted_step": false,
    "final_verdict": "Done"
  }
]
```

**示例——final coach 判定轮次未完成（`NotDone`），模型继续生成：**

```json
"step_2": [
  {
    "role": "assistant",
    "content": "The file has been written."
  },
  {
    "role": "handler_log",
    "content": "Turn ended.",
    "hinted_step": false,
    "final_verdict": "NotDone"
  }
]
```

> `NotDone` 时 final coach 会注入提示并触发下一步生成，提示同样在接受后被抹除。

---

## Inference Log 结构

所有 handler 均在 `metadata` 下写入 `inference_log` 列表。每个元素是 `state_info` dict 列表（系统状态）或轮次 dict：

```
inference_log = [
  [state_info, ...],        # 可选——初始状态
  {                         # turn 0
    "begin_of_turn_query": [{"role": "user", "content": "..."}],
    "step_0": [...],
    "step_1": [...],
  },
  [state_info, ...],        # 可选——turn 0 后的状态
  { ... },                  # turn 1
  ...
]
```

每个 step 是一组条目，各条目的 role 含义：

| Role | 描述 | 出现位置 |
|------|------|---------|
| `assistant` | 被接受的 FC（原始 `<tool_call>` 字符串）或 text 回答 | 所有 handler |
| `tool` | 工具执行结果 | 所有 handler |
| `handler_log` | 元数据：`hinted_step`、`final_verdict`、解码后的 FC | 所有 handler |
| `user` | Supervisor 指导消息（仅 V1） | 仅 V1 |
| `inference_input` | 原始模型输入（仅使用 `--include-input-log` 时） | 所有 handler |

### `hinted_step` 与 `final_verdict` 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `hinted_step` | bool | 本步骤是否经历过至少一次重试 |
| `final_verdict` | str | hinted/v2/v3 FC步骤：`"Good"` / `"Bad"`；text步骤：`"Done"` / `"NotDone"`；V1：`"silent"` / `"coached"` |

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `JUDGE_API` | `"openai"` 或 `"anthropic"` |
| `JUDGE_MODEL` | 裁判/coach/提示生成器 LLM 的模型 ID |
| `OPENAI_API_KEY` | `JUDGE_API=openai` 时必填 |
| `ANTHROPIC_API_KEY` | `JUDGE_API=anthropic` 时必填 |
| `HINT_LOG_PATH` | 可选，每步提示/coach 事件日志的写入路径 |
| `HINT_DEBUG` | 设为 `"1"` 开启逐步详细控制台输出 |

---

## 测试

```bash
PYTHONPATH=. python bfcl_eval/tests/generate_samples.py
PYTHONPATH=. pytest bfcl_eval/tests/test_coach_handlers.py -v
```

---

## SFT 训练数据

这些 handler 产生的 inference log 直接输入 `hint-training/` 中的 SFT 流水线。详见 `hint-training/prepare_and_train_zh.md`。
