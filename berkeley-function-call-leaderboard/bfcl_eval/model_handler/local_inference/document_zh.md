# Qwen Handler — 技术参考文档

本文档覆盖 `bfcl_eval/model_handler/local_inference/` 中所有 Qwen 相关的 handler 文件。
是了解 coached inference 端到端工作方式的权威参考，涵盖 vLLM 服务管理、prompt 格式化、
coaching 循环以及推理日志结构。

---

## 1. 类继承关系

```
BaseHandler  (base_handler.py)
└── OSSHandler  (base_oss_handler.py)
    └── QwenFCHandler  (qwen_fc.py)
        ├── QwenFCHintedHandler  (qwen_fc_hinted.py)   — 执行前 judge+hinter（两次调用）
        ├── QwenFCV1CoachHandler (qwen_fc_v1_coach.py) — 执行后单次 coach
        ├── QwenFCV2CoachHandler (qwen_fc_v2_coach.py) — 执行前单次 coach + XML complete-check
        └── QwenFCV3CoachHandler (qwen_fc_v3_coach.py) — 执行前单次 coach + silent final-coach
```

四个 coaching handler 都是 `QwenFCHandler` 的直接替换品。它们只覆盖
`inference_multi_turn_prompting`；chat 模板、decode、vLLM API 调用均从父类继承。

---

## 2. OSSHandler — `base_oss_handler.py`

### 职责
所有开源模型 handler 的抽象基类。管理 vLLM/SGLang 服务生命周期并实现 BFCL prompting 协议。

### 服务生命周期

`spin_up_local_server(num_gpus, gpu_memory_utilization, backend, ...)`:
- 加载 tokenizer 和 config，确定 `max_context_length`。
- 若 `skip_server_setup=False`，启动 `vllm serve` 或 `python -m sglang.launch_server` 子进程。
- 启动两个 daemon 线程流式读取 stdout/stderr，直到 `/v1/models` 端点返回 200。
- 通过 `--enable-lora / --lora-modules / --max-lora-rank` 支持 LoRA。
- 通过 `REMOTE_OPENAI_BASE_URL` / `REMOTE_OPENAI_API_KEY` 环境变量支持远程 vLLM 端点。

`shutdown_local_server()`：终止进程并等待日志线程退出。

### Prompting 协议

`_query_prompting(inference_data)`:
1. 调用 `self._format_prompt(message, function)` — **抽象方法**，子类实现。
2. 对格式化后的 prompt 做 tokenize 以统计 token 数。
3. 将 `max_tokens` 限制为 `min(4096, max_context_length - input_tokens - 2)`。
4. 调用 `client.completions.create(...)` — OpenAI Completions API（非 Chat API）。
5. 返回 `(api_response, latency_seconds)`。

`_pre_query_processing_prompting`：通过 `system_prompt_pre_processing_chat_model`
注入 BFCL 系统 prompt；返回 `{"message": [], "function": functions}`。

`_parse_query_response_prompting`：返回 `{"model_responses": text, "input_token": N, "output_token": N}`。

多轮消息追加方法（`add_first_turn_message_prompting`、`_add_next_turn_user_message_prompting`、
`_add_assistant_message_prompting`、`_add_execution_results_prompting`）均只做 append 操作。

---

## 3. QwenFCHandler — `qwen_fc.py`

### 职责
以 Python 字符串拼接手动实现 Qwen2.5 chat 模板（而非依赖 tokenizer apply_chat_template），
从而对最终格式有完全控制权——这是使用 vLLM Completions API 时必须的做法。

### `_format_prompt(messages, function)`

模板与 Qwen2.5-Instruct 自带的 Jinja2 模板完全对应：

**系统块（有工具时）**：
```
<|im_start|>system
{system 消息内容（如果有）}

# Tools
...工具 schema JSON...
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

**`last_query_index` 逻辑**：倒序遍历 messages，找到最后一条 role=user 且
内容不以 `<tool_response>` 开始、以 `</tool_response>` 结束的消息的索引。
该索引将"历史多轮"与"当前激活轮"分开。

**助手消息**：
- `last_query_index` 之前（历史轮）：写作 `<|im_start|>assistant\n{content}`，不强制 `<think>` 块。
- `last_query_index` 之后（当前激活轮）：
  - 若为最后一条消息，或存在 `reasoning_content`：包裹在 `<think>\n{reasoning}</think>\n\n{content}` 中。
  - 否则：直接写 content，不加 `<think>`。
- `tool_calls` 在助手消息中序列化为 `<tool_call>\n{...}\n</tool_call>` 块。

**工具结果消息**：多条连续 `role=tool` 消息包在同一个 `<|im_start|>user` 块里，每条结果用独立 `<tool_response>` 标签包裹。

**`<think>` / reasoning_content 处理**：若模型输出包含 `</think>`，则在 `</think>` 处拆分：前半部分（去掉 `<think>` 标签）成为 `reasoning_content`，后半部分成为可见的 `content`。

### `_parse_query_response_prompting`

返回的 `model_responses_message_for_chat_history`：
- 若解析到工具调用：`{"role": "assistant", "tool_calls": [...], "content": ""}`。
- 否则：`{"role": "assistant", "content": cleaned_response}`。
两种情况均附加 `"reasoning_content"` 字段。

### `_extract_tool_calls(input_string)` (静态方法)

正则 `<tool_call>\n(.*?)\n</tool_call>`（`re.DOTALL`）。每个匹配项解析为 JSON
`{"name": ..., "arguments": ...}`。解析失败时静默跳过。

### `decode_ast` / `decode_execute`

两者都调用 `_extract_tool_calls`。`decode_execute` 还额外调用
`convert_to_function_call` 生成 BFCL 执行格式。

---

## 4. Coaching Handler 公共模式

四个 coaching handler 共享相同的基础设施。

### 环境变量

| 变量 | 用途 |
|---|---|
| `JUDGE_API` | `"openai"` 或 `"anthropic"` |
| `JUDGE_MODEL` | 例如 `"gpt-4o-2024-11-20"`、`"claude-opus-4-7"` |
| `HINT_LOG_PATH` | coaching 事件 JSONL 文件（可选） |
| `HINT_DEBUG` | `"1"` 或 `"true"` 开启详细 stdout 输出 |

### 并发控制

`_judge_semaphore = threading.Semaphore(max_judge_concurrency)` — 限制并发 coaching API
调用数量。最坏情况：`num_workers × max_hint_retries × 2` 个并行请求。

`_hint_log_queue` — 一个 `queue.Queue`，由单个 daemon 写入线程（`_hint_log_writer`）消费。
所有推理 worker 线程只 enqueue；一个线程负责所有 JSONL 文件 I/O，写入永不交错。
`shutdown_hint_log()` 发送 `None` sentinel 并调用 `queue.join()` 等待写完。
每个 test entry 结束时也会调用 `queue.join()` 以保证该 entry 的日志完整性，
而不必关闭整个队列。

### `_serialize_state(involved_instances)` → str

JSON dump `{class_name: {所有属性（除 _api_description 外）}}`，覆盖所有活跃后端实例。
仅跳过 `_api_description` 一个私有属性（纯样板字符串，对 coach 无用）；其余所有属性——包括 `_current_dir`、`_brakePedalForce` 等私有属性——均正常暴露给 coach。`_current_dir` 会额外处理为可读路径字符串（而非 Directory 对象 repr）。传给 coaching prompt，让 coach 了解当前系统状态。

### `_fmt_messages(msgs)` → str

将 message 列表转为可读文本供 LLM prompt 使用。
包含 user 消息、助手工具调用、工具结果；忽略纯文本助手消息。
V1 额外跳过以 `"Supervisor guidance"` 开头的 user 消息，防止 coach 看到自己之前的指令。

### `_build_state_log(involved_instances)` → list[dict]

为有状态后端实例生成 `{"role": "state_info", "class_name": ..., "content": {...}}` 条目
（跳过 `STATELESS_CLASSES` 和 `OMIT_STATE_INFO_CLASSES` 中的类）。使用 `deepcopy` 防止后续状态变化影响已记录快照。

### 推理日志结构

所有 coaching handler 的日志格式与 `QwenFCHandler` 相同：

```
all_inference_log = [
  [{"role": "state_info", ...}],          # 初始状态（exclude_state_log=False 时）
  {                                        # turn 0
    "begin_of_turn_query": [...],
    "step_0": [
      {"role": "assistant", "content": "...", "reasoning_content": "..."},
      {"role": "inference_input", ...},    # include_input_log=True 时
      {"role": "handler_log", "content": "...", "model_response_decoded": [...],
       "hinted_step": bool, "final_verdict": str, "coach_latency_s": float},
      {"role": "tool", "content": "..."},
    ],
    "step_1": [...],
    ...
  },
  [{"role": "state_info", ...}],          # turn 0 结束后的状态
  ...
]
```

`handler_log` 的关键字段：
- `hinted_step: bool` — True 表示本 step 中 coaching LLM 至少介入过一次。
- `final_verdict: str` — coaching LLM 的最终判决（`"Good"`、`"Bad"`、`"Done"`、`"NotDone"`、`"coached"`、`"silent"`）。
- `coach_latency_s: float` — 本 step 所有 coaching LLM 调用的总延迟（秒）之和。

**重要**：hint、被拒绝的 FC、中间重试**绝不写入** `all_inference_log`。
日志是干净的训练轨迹——每个 step 只记录被接受的 FC 及其执行结果。

---

## 5. QwenFCHintedHandler — `qwen_fc_hinted.py`

### 策略：执行前，两步 judge → hinter

小模型产生函数调用后（执行之前）：
1. **JUDGE_PROMPT** → 按 8 条 yes/no 标准评估正确性。输出 `<verdict>Good/Bad</verdict>` + Bad 时的文字解释。
2. **HINTER_PROMPT** → 接收 judge 的完整输出，提炼为一句可操作的指导，不透露正确答案。

若模型输出纯文本（无工具调用）：
- **COMPLETE_PROMPT** → 判断任务是否真正完成。`<verdict>Done/NotDone</verdict>` + `<hint>...</hint>`。

### `step_start_idx` 不变量

```python
step_start_idx = len(inference_data["message"])   # 每个 step 只设置一次

while hint_retry <= max_hint_retries:
    del inference_data["message"][step_start_idx:]  # 回退到干净边界
    if latest_hint is not None:
        inference_data["message"].append({"role": "user", "content": latest_hint})
    # 查询小模型 ...
    # judge / complete-check ...

del inference_data["message"][step_start_idx:]  # 最终清理（删除最后一条 hint）
```

小模型的上下文始终是：`[已提交历史] + [0 或 1 条 hint]`。
它永远看不到之前错误的 FC 或过期的 hint。循环结束后最后一条 hint 也被删除，
只有被接受的 FC 会进入历史。

### Hint 重试流程（有 FC 的情况）

```
hint_retry=0..max_hint_retries:
  查询 → fc_message
  若无工具调用 → complete_check → Done: break | NotDone: 注入 hint, continue
  若 hint_retry < max_hint_retries:
    judge → Good: 接受 | Bad: hinter → hint, 注入, 重试
  else:
    接受（重试耗尽）
```

### Hint 日志事件

- `type="judge_hinter"` — judge 返回 Bad 时写入；包含 `fc_message`、`judge_raw`、`hint`。
- `type="complete_check"` — 每次纯文本响应时写入；包含 verdict 和 hint。

---

## 6. QwenFCV1CoachHandler — `qwen_fc_v1_coach.py`

### 策略：执行后，单次 coach

**无执行前检查。无 step_start_idx。无重试循环。**

FC 无条件执行。执行后 coach 看到含工具结果的完整历史，决定是否介入：
- 空响应 / `"SILENT"` → 保持沉默，模型正常进入下一 step。
- 任何文本 → 作为 `"Supervisor guidance for the next step only: {instruction}"` 注入。

同一 coach 也在模型给出纯文本响应时被调用（`phase="after_attempted_final_answer"`）。
若返回指令，模型被重新查询（计作下一个 step）。

### 与 hinted 的关键差异

- Coach 在**执行后**运行，而非执行前。
- 每个 step 只有一次 LLM 调用（无 judge/hinter 分离）。
- 无法拦截错误的 FC——工具已经运行了。
- `_fmt_messages` 过滤掉 `"Supervisor guidance"` 消息，防止 coach 看到自己之前的指令。
- `max_coach_instructions=0` 表示不限次数；设为正整数则限制整个任务（跨所有 turn）的指令总数，由 `coach_instructions_total` 追踪。

### 推理日志

Coach 指令**会**作为 user 消息写入推理日志（在它影响的 step 之前）。
`hinted_step` 为 True 表示上一个 step 有 coach 指令被注入。

---

## 7. QwenFCV2CoachHandler — `qwen_fc_v2_coach.py`

### 策略：执行前，单次 coach + XML complete-check

结构与 `QwenFCHintedHandler` 完全相同，区别在于：
- 两步 JUDGE → HINTER 流程被替换为**单次 COACH 调用**。
- COACH 返回空/SILENT 表示批准；任何文本即为 hint。
- Complete-check 仍使用带 `<verdict>Done/NotDone</verdict>` XML 的 **COMPLETE_PROMPT**。

`step_start_idx` 不变量与 hinted 完全相同。

### Prompt 设计

**COACH_SYSTEM**：审核者角色；silent=批准，文本=hint；不透露正确答案。

**COACH_PROMPT** 字段（按 prompt 顺序，对应前缀缓存优化）：
- `schema`（工具 schema，最大且在整个 test case 期间恒定，排在最前以最大化前缀缓存命中）
- `user_request`（当前 turn 的用户请求）
- `history`（已提交轮次的对话历史）
- `prev_step_result`（上一个已完成 step 的 FC + 执行结果，在每个 step 的外层循环开始时计算一次，位于重试子循环之前；让 coach 了解前一步的执行情况，从而发现需要纠正的状态错误或调用失败）
- `previous_hints`（本 step 已给出的 hint 历史）
- `current_state`（当前系统状态）
- `tool_name`、`tool_args`（计划的 FC，排在最后以减少缓存失效）

**COMPLETE_PROMPT** 字段（按 prompt 顺序）：`schema`、`user_request`、`history`、`current_state`。

### Hint 日志事件

- `type="v2_coach"` — coach 返回 hint 时写入。
- `type="complete_check"` — 与 hinted 相同。

---

## 8. QwenFCV3CoachHandler — `qwen_fc_v3_coach.py`

### 策略：执行前单次 coach + silent final-coach（无 XML）

有效函数调用时的处理与 V2 完全相同（COACH_PROMPT → silent 批准或 hint）。

区别：当模型产生纯文本响应时，不再使用带 XML verdict 标签的 COMPLETE_PROMPT，
而是使用：

**FINAL_COACH_SYSTEM/PROMPT**：silent=接受（turn 结束），任何文本=重试指令。
无 `<verdict>Done/NotDone</verdict>` ——接受/拒绝隐含在响应是否为空中。

**FINAL_COACH_PROMPT** 字段（按 prompt 顺序）：`schema`、`user_request`、`history`、`current_state`、`model_response`。

这消除了 XML 解析依赖，将两条路径（执行前 FC 检查 和 执行后文本检查）的
接受/拒绝信号统一为相同的"沉默即同意"模式。

**COACH_PROMPT（V3）** 字段顺序与 V2 相同（`schema`、`user_request`、`history`、`prev_step_result`、`previous_hints`、`current_state`、`tool_name`、`tool_args`）。

### Hint 日志事件

- `type="v3_coach"` — 执行前 coach 返回 hint 时写入。
- `type="v3_final_coach"` — 每次纯文本响应时写入。

---

## 9. Handler 对比表

| 属性 | Hinted | V1 Coach | V2 Coach | V3 Coach |
|---|---|---|---|---|
| 触发时机 | 执行前 | 执行后 | 执行前 | 执行前 |
| FC 接受条件 | 需要 Good 判决 | 无条件执行 | silent=批准 | silent=批准 |
| 每次重试的 LLM 调用数 | judge + hinter（2次） | coach（1次） | coach（1次） | coach（1次） |
| 重试循环 | 有（step_start_idx） | 无 | 有（step_start_idx） | 有（step_start_idx） |
| 纯文本响应处理 | COMPLETE_PROMPT（XML） | coach 重新查询 | COMPLETE_PROMPT（XML） | FINAL_COACH（silent） |
| Coach 是否看到之前错误的 FC | 否（上下文重置） | 是（完整历史） | 否（上下文重置） | 否（上下文重置） |
| Coach 历史过滤 | 不适用 | 跳过 "Supervisor guidance" | 不适用 | 不适用 |
| hint_log 类型 | judge_hinter, complete_check | v1_coach_tool, v1_coach_final | v2_coach, complete_check | v3_coach, v3_final_coach |

---

## 10. 设计决策

### 为什么要把 hint 从推理日志中抹去？

推理日志是 SFT 训练数据的来源。Coaching 过程中注入的 hint 是脚手架，不是目标行为。
若 hint 出现在训练轨迹里，模型会学会在推理时期待被引导，而推理时根本没有 coach。

### 为什么用 step_start_idx 而不是保留所有重试？

让模型同时看到它之前错误的 FC 和 hint 会污染上下文。`step_start_idx` 切片保证模型只看到
干净的已提交历史，加上最新的单条 hint。这也让上下文长度保持可预测。

### 为什么用信号量限制 coaching API 调用？

多个 BFCL test entry 在并行 worker 线程中运行。没有信号量，最坏情况是
`num_workers × max_hint_retries × 2` 个并发 coaching 请求，很容易超出速率限制。

### 为什么用独立写入线程处理 hint_log？

多个推理 worker 同时写同一个 JSONL 文件，若不加锁会导致写入交错。
单个队列写入线程将所有 I/O 串行化，worker 线程零锁竞争。

### 工具调用格式：flat vs OpenAI wrapped

`QwenFCHandler` 生成"flat"格式的工具调用：
```json
{"name": "fn_name", "arguments": {...}}
```
Coaching handler 同时处理 flat 格式和 OpenAI wrapped 格式：
```json
{"function": {"name": "fn_name", "arguments": {...}}}
```
所有 `_tc_name_args` 辅助函数都检查 `"function"` key 并按需解包。
这对与 SFT 数据准备脚本（`collect_snapshot` 生成 OpenAI 格式）的兼容性很重要。
