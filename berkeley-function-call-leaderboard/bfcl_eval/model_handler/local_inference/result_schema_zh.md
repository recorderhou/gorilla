# BFCL 结果与评分文件 Schema — 技术参考（中文）

本文档描述 BFCL result/score 文件的目录结构、字段含义，以及不同 handler 产生的
`inference_log` 结构差异。示例数据来自 `result_20260613/` 和 `score_20260613/`。

---

## 1. 目录结构

```
result_20260613/
├── hint_log.jsonl                          ← coaching 事件日志（仅 hinted handler 有）
├── qwen2.5-3b-hinted-FC/
│   └── multi_turn/
│       └── BFCL_v4_multi_turn_base_result.json
├── qwen2.5-3b-v1coach-FC/
│   └── multi_turn/
│       └── BFCL_v4_multi_turn_base_result.json
├── qwen2.5-3b-v2coach-FC/   ...
├── qwen2.5-3b-v3coach-FC/   ...
├── qwen2.5-3b-v1-sft-FC/    ...           ← SFT 后的模型，推理时不加 coach
├── qwen2.5-3b-v1-sft-coached-FC/  ...     ← SFT 模型 + 同版本 coach handler
├── qwen2.5-3b-v1-sft-coached-gen-FC/ ...  ← SFT 模型（泛化数据）+ coaching
└── qwen2.5-3b-v1-sft-gen-FC/ ...          ← SFT 模型（泛化数据），不加 coaching

score_20260613/
├── data_overall.csv            ← 所有类别汇总排行榜
├── data_multi_turn.csv         ← multi-turn 子类拆分
├── data_agentic.csv
├── data_live.csv
├── data_non_live.csv
├── data_format_sensitivity.csv
└── <model-name>/
    └── multi_turn/
        └── BFCL_v4_multi_turn_base_score.json
```

### 模型命名规则

| 后缀模式 | 含义 |
|---|---|
| `{handler}-FC` | base 模型 + coaching handler 推理，无 SFT |
| `{vN}-sft-FC` | SFT 后的模型，使用 base QwenFCHandler 推理（不带 coach） |
| `{vN}-sft-coached-FC` | SFT 模型 + 与训练数据同版本的 coaching handler |
| `{vN}-sft-coached-gen-FC` | SFT 模型（泛化数据训练）+ coaching |
| `{vN}-sft-gen-FC` | SFT 模型（泛化数据训练），不带 coaching |

---

## 2. Result 文件 Schema

**路径**：`result_20260613/<model>/multi_turn/BFCL_v4_multi_turn_base_result.json`
**格式**：JSONL，每行一个 JSON 对象，对应一个测试 entry（无 summary 行）。

### 顶层字段

```json
{
  "id": "multi_turn_base_0",
  "result": "...",
  "input_token_count": 1234,
  "output_token_count": 56,
  "latency": 2.31,
  "inference_log": [...]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | BFCL 测试 entry 的唯一 ID |
| `result` | str | 该 entry 最终解码后的模型输出 |
| `input_token_count` | int | 所有轮次的 input token 总数 |
| `output_token_count` | int | 所有轮次的 output token 总数 |
| `latency` | float | 所有推理调用的总耗时（秒） |
| `inference_log` | list | 所有轮次和步骤的结构化日志（详见第 4 节） |

**Coach handler 专属 metadata（仅 v1 coach；每条轨迹）：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `coach_checks` | int | coach LLM 调用总次数（含返回 SILENT 的） |
| `coach_interventions` | int | coach 返回非空 hint/instruction 的次数 |
| `coach_prompt_tokens` | int | 所有 coach 调用的 prompt token 总数 |
| `coach_completion_tokens` | int | 所有 coach 调用的 completion token 总数 |
| `coach_cached_tokens` | int | coach 的 cached prompt token 总数（`usage.prompt_tokens_details.cached_tokens`） |

计数器按轨迹重置（thread-local，因 `bfcl generate` 中一条轨迹 = 一个线程）。非 coach handler 无这些字段。

---

## 3. Score 文件 Schema

**路径**：`score_20260613/<model>/multi_turn/BFCL_v4_multi_turn_base_score.json`
**格式**：JSONL，第 0 行是汇总，第 1 行起是逐条结果。

### 第 0 行 — 汇总

```json
{ "accuracy": 0.4, "correct_count": 20, "total_count": 50 }
```

### 第 1+ 行 — 逐条结果

```json
{
  "id": "multi_turn_base_0",
  "model_name": "qwen2.5-3b-v1coach-FC",
  "test_category": "multi_turn_base",
  "valid": false,
  "error": {
    "error_message": "Model instance for GorillaFileSystem does not match ...",
    "error_type": "multi_turn:instance_state_mismatch",
    "details": { "differences": { "root": { "model": "...", "ground_truth": "..." } } }
  },
  "prompt": [...],
  "model_result_raw": "<tool_call>...</tool_call>",
  "model_result_decoded": ["mv(source='x', destination='y')"],
  "possible_answer": [...],
  "inference_log": [...]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 测试 entry ID |
| `model_name` | str | handler/模型标识符 |
| `test_category` | str | 如 `multi_turn_base`、`multi_turn_miss_func` |
| `valid` | bool | 该 entry 是否通过 |
| `error` | dict\|null | 通过则为 null；`error_type` 遵循 `multi_turn:*` 分类 |
| `prompt` | list | 发送给模型的完整对话 prompt |
| `model_result_raw` | str | 最后一次原始模型输出 |
| `model_result_decoded` | list[str] | 解码后的函数调用，格式 `fn(args)` |
| `possible_answer` | list | 评分用的标准答案 |
| `inference_log` | list | 与 result 文件相同结构（见第 4 节） |

### 汇总 CSV 文件

`data_overall.csv` 主要列：

| 列名 | 说明 |
|---|---|
| `Model` | 模型显示名称 |
| `Overall Acc` | 所有测试类别加权准确率 |
| `Multi Turn Acc` | Multi-turn 类别准确率 |
| `Multi Turn Base` | 子任务：基础 multi-turn |
| `Multi Turn Miss Func` | 子任务：缺少函数 |
| `Live Acc` | Live 类别准确率 |

`data_multi_turn.csv` 列：`Rank`、`Model`、`Multi Turn Overall Acc`、`Base`、`Miss Func`、`Miss Param`、`Long Context`。

---

## 4. inference_log 结构

`inference_log` 字段在 result 和 score 文件中结构相同，是一个**交替排列**的 list：
**状态快照**（list）和**轮次字典**（dict）交替出现。

```
inference_log = [
  [state_info, ...],          ← 初始后端状态（turn 0 之前）
  { "begin_of_turn_query": [...], "step_0": [...], "step_1": [...], ... },   ← turn 0
  [state_info, ...],          ← turn 0 执行后的状态
  { ... },                    ← turn 1
  [state_info, ...],
  ...
]
```

### 状态快照（list）

每个元素：
```json
{
  "role": "state_info",
  "class_name": "GorillaFileSystem",
  "content": { ... }
}
```

记录后端实例（如 `GorillaFileSystem`、`TwitterAPI`）在该时刻的完整状态，用于判断模型
是否把环境改到了正确的状态。

### 轮次字典（dict）

```
{
  "begin_of_turn_query": [{"role": "user", "content": "..."}],
  "step_0": [...],
  "step_1": [...],
  ...
}
```

- `begin_of_turn_query`：开启这一 BFCL turn 的用户消息。
- `step_N`：一个 agent 步骤，包含一次模型查询及可选的工具执行结果。

### 各 handler 下的 step 结构

step 是一个消息列表，不同 handler 的序列有所不同：

#### Base / SFT 模型（无 coaching）

```
["assistant", "handler_log", "tool", ...]
```

- `handler_log` 有 `model_response_decoded`，**没有** `hinted_step` / `final_verdict`。
- 模型在一步内调用多个函数时，会有多个 `tool` 消息。

#### Hinted handler（预执行，judge+hinter 两步）

```
["assistant", "handler_log", "tool"]
```

- `handler_log` 包含 `hinted_step`（bool）和 `final_verdict`（`"Good"` / `"Bad"`）。
- 日志只写入**被接受的** FC，被拒绝的错误 FC 和中间 hint 在提交前已被抹除。

#### V1 Coach handler（后执行，单步 coach）

```
["assistant", "handler_log", "tool", "handler_log"]       ← coach 沉默（不干预）
["user", "assistant", "handler_log", "tool", "handler_log"] ← coach 注入了指导
["handler_log"]                                           ← 模型只返回文本，无 tool call
```

- 第一个 `handler_log`：解码结果（`model_response_decoded`）。
- 第二个 `handler_log`：后执行 coach 事件（`content: "Post-execution coach."`, `final_verdict: "coached"` 或 `"silent"`）。
- 当 coach 注入了指导时，step 开头的 `user` 消息就是那条指导语。
- 只有 `["handler_log"]` 的步骤表示模型没有发出 tool call（纯文本回复）。

#### V2 / V3 Coach handler（预执行，单步 coach）

```
["assistant", "handler_log", "tool"]
```

- 与 hinted 形状相同。Coach 在执行前介入，只有通过的 FC 才写入日志。
- `final_verdict`：`"coached"` 表示 coach 干预过，`"silent"` 表示静默通过。

### handler_log 字段说明

| 字段 | 出现于 | 说明 |
|---|---|---|
| `content` | 所有 | 人类可读的事件描述 |
| `model_response_decoded` | 第一个 handler_log | 解码后的 `fn(args)` 字符串列表 |
| `hinted_step` | hinted / v1 / v2 / v3 | `true` 表示该步骤 coaching LLM 至少干预过一次 |
| `final_verdict` | hinted / v1 / v2 / v3 | 最终裁决：`Good`、`Bad`、`Done`、`NotDone`、`coached`、`silent` |

---

## 5. hint_log.jsonl

**路径**：`result_20260613/hint_log.jsonl`
**格式**：JSONL，每行一个事件。
**覆盖范围**：仅在推理时设置了 `HINT_LOG_PATH` 环境变量才写入。20260613 的实验中，
只有 **hinted handler** 配置了此变量，其余 v1/v2/v3 coach handler 没有，因此不产生日志。

### Schema（hinted handler，事件类型 `judge_hinter`）

```json
{
  "test_entry_id": "multi_turn_base_0",
  "turn_idx": 0,
  "step": 0,
  "hint_retry": 1,
  "fc_message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"name": "mv", "arguments": {"source": "x", "destination": "y"}}],
    "reasoning_content": ""
  },
  "judge_raw": "0. No - ...\n<verdict>Bad</verdict>\n...",
  "hint": "The destination argument should not include a path..."
}
```

| 字段 | 说明 |
|---|---|
| `test_entry_id` | BFCL 测试 entry（与 result `id` 对应） |
| `turn_idx` | 第几个 BFCL turn（从 0 开始） |
| `step` | 该 turn 的第几个步骤 |
| `hint_retry` | 重试次数（0 = 首次尝试；只有裁决为 Bad 才记录此事件） |
| `fc_message` | 小模型发出的被拒绝 FC |
| `judge_raw` | judge LLM 的完整输出，含 8 条是/否评估 + `<verdict>` 标签 |
| `hint` | hinter 蒸馏出的一句话提示（下次重试时注入给小模型） |

v2/v3 coach 的日志事件字段会不同（用 `coach_raw` 替代 `judge_raw`+`hint`），
但本次实验未配置 `HINT_LOG_PATH`，所以没有对应日志文件。

---

## 6. Multi-Turn 评分汇总（20260613）

按 Multi Turn Overall Acc 排序，所有模型均为 Qwen2.5-3B-Instruct 变体：

| 名次 | 模型 | Multi Turn Overall | Base | Miss Func |
|---|---|---|---|---|
| 1 | v2-SFT-coached-gen | 19.00% | 50.00% | 26.00% |
| 2 | v3-SFT-coached-gen | 14.00% | 40.00% | 16.00% |
| 3 | v2-SFT-coached | 11.00% | 44.00% | N/A |
| 4 | v2coach | 11.00% | 44.00% | N/A |
| 5 | v1coach | 10.00% | 40.00% | N/A |
| 6 | v3-SFT-coached | 9.50% | 38.00% | N/A |
| 7 | v3coach | 9.00% | 36.00% | N/A |
| 8 | hinted | 8.50% | 34.00% | N/A |
| 9 | v1-SFT-coached-gen | 6.00% | 20.00% | 4.00% |
| 10 | v1-SFT-coached | 5.00% | 20.00% | N/A |
| 11 | v2-SFT | 2.50% | 10.00% | N/A |
| 12 | v3-SFT | 2.50% | 10.00% | N/A |
| 13 | v1-SFT-gen | 1.50% | 6.00% | 0.00% |
| 14 | v2-SFT-gen | 0.50% | 0.00% | 2.00% |
| 15 | v3-SFT-gen | 0.50% | 0.00% | 2.00% |
| 16 | v1-SFT | 0.00% | 0.00% | N/A |
| 17 | hint-SFT | 0.00% | 0.00% | N/A |

### 观察要点

- **v2-SFT-coached-gen 最高**（19%）：v2 coach 策略 + 泛化数据 SFT + 推理时继续 coaching。
- **coaching 本身有效**：同一 SFT 模型加上 coached 推理都明显高于裸推理（如 v2-SFT 10% → v2-SFT-coached 11%，v2coach 无 SFT 也能达 11%）。
- **SFT 数据质量很关键**：v1-SFT 系列整体偏低（base 只有 0%），说明 v1 post-exec 策略产生的训练数据质量不如 v2/v3。
- **gen 数据分化明显**：v1-SFT-gen（1.5%）vs v2-SFT-gen（0.5%），泛化数据对不同版本 coach 效果不一致，需进一步分析。
