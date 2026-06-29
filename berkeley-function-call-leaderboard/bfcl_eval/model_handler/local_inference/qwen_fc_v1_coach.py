"""
QwenFCV1CoachHandler — post-execution coaching (Version 1, Agent-Sampler style).

After each accepted tool call and its observation, a coach LLM reviews the full
conversation and decides whether to add one instruction for the next step.  There
is no pre-execution retry loop.

Coach responses:
  - Empty (or "SILENT") → stay silent, no instruction.
  - Any text            → one-sentence instruction prepended to the next turn.

Training log is clean: coach instructions are stripped before writing.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from copy import deepcopy
from typing import Optional

from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from openai import OpenAI, RateLimitError

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override
from bfcl_eval.model_handler.utils import (
    add_memory_instruction_system_prompt,
    retry_with_backoff,
)
from bfcl_eval.utils import extract_test_category_from_id, is_memory, is_memory_prereq


# ── Prompts ───────────────────────────────────────────────────────────────────

COACH_SYSTEM = """You are supervising a tool-use agent completing a multi-step task.

After each tool call and its observation, choose exactly one option:
1. Stay silent — return an empty response or exactly SILENT.
2. Return ONE short instruction for the agent's next step.

Intervene only when it is likely to improve correctness. Do NOT:
- Solve the task or reveal the correct answer
- Give the exact function call, tool name, or argument values
- Repeat guidance that has already been given

If the current trajectory looks correct or sufficient, stay silent."""

COACH_PROMPT = """## Task
{user_request}

## Available Tools
{schema}

## Current System State
{current_state}

## Phase
{phase}

## Conversation so far
{history}
{budget_info}
Return an empty response to stay silent, or one short instruction for the agent's next step."""


# ── Handler ───────────────────────────────────────────────────────────────────

class QwenFCV1CoachHandler(QwenFCHandler):
    """
    Post-execution coach variant.  After each committed tool call the coach LLM
    is consulted; if it returns an instruction it is injected as a user message
    before the next small-model query.  No pre-execution retry loop.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        judge_api: Optional[str] = None,
        judge_model: Optional[str] = None,
        max_coach_instructions: int = 0,     # 0 = unlimited per turn
        max_judge_concurrency: int = 5,
        hint_log_path: Optional[str] = None,
        debug: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        judge_api   = judge_api   or os.environ["JUDGE_API"]
        judge_model = judge_model or os.environ["JUDGE_MODEL"]
        hint_log_path = hint_log_path or os.environ.get("HINT_LOG_PATH")
        self.debug = debug if debug is not None else os.environ.get("HINT_DEBUG", "").lower() in ("1", "true")

        self.judge_api   = judge_api
        self.judge_model = judge_model
        self.max_coach_instructions = max_coach_instructions

        self._judge_semaphore = threading.Semaphore(max_judge_concurrency)
        if judge_api == "openai":
            self._judge_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._anthropic_client = None
        elif judge_api == "anthropic":
            import anthropic
            self._judge_client = None
            self._anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            raise ValueError(f"Unknown judge_api: {judge_api!r}")

        self.hint_log_path = hint_log_path
        self._hint_log_queue: Optional[queue.Queue] = None
        if hint_log_path:
            import pathlib
            pathlib.Path(hint_log_path).parent.mkdir(parents=True, exist_ok=True)
            self._hint_log_queue = queue.Queue()
            t = threading.Thread(target=self._hint_log_writer, daemon=True)
            t.start()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @retry_with_backoff(error_type=RateLimitError)
    def _call_openai(self, system: str, prompt: str) -> str:
        kwargs = {"max_completion_tokens": 1024}
        if not (self.judge_model.startswith("o") or self.judge_model.startswith("gpt-5")):
            kwargs["temperature"] = 0
        resp = self._judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(self, system: str, prompt: str) -> str:
        resp = self._anthropic_client.messages.create(
            model=self.judge_model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _call_llm(self, system: str, prompt: str) -> str:
        with self._judge_semaphore:
            if self.judge_api == "openai":
                return self._call_openai(system, prompt)
            return self._call_anthropic(system, prompt)

    def _serialize_state(self, involved_instances: dict) -> str:
        try:
            from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import Directory as _Directory
        except ImportError:
            _Directory = None

        def _dir_path(d):
            parts = []
            while d is not None:
                parts.append(d.name)
                d = d.parent
            return "/".join(reversed(parts))

        state = {}
        for name, inst in involved_instances.items():
            attrs = {}
            for k, v in vars(inst).items():
                if k == "_api_description":
                    continue
                if k == "_current_dir" and _Directory is not None and isinstance(v, _Directory):
                    attrs[k] = _dir_path(v)
                else:
                    attrs[k] = v
            state[name] = attrs
        return json.dumps(state, indent=2, default=str)

    def _fmt_messages(self, msgs: list) -> str:
        parts = []
        for m in msgs:
            role = m.get("role")
            if role == "user":
                content = m.get("content") or ""
                # Skip coach instructions from history shown to the coach itself
                if str(content).startswith("Supervisor guidance"):
                    continue
                parts.append(f"User: {content}")
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if "function" in tc:
                        fn = tc["function"]
                        name = fn.get("name", "unknown")
                        args = fn.get("arguments", {})
                    else:
                        name = tc.get("name", "unknown")
                        args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    parts.append(f"Called: {name}({args})")
            elif role == "tool":
                parts.append(f"Tool result: {m.get('content', '')}")
        return "\n".join(parts) if parts else "(none)"

    def _build_state_log(self, involved_instances: dict) -> list[dict]:
        log = []
        for class_name, inst in involved_instances.items():
            if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                continue
            snap = deepcopy(inst)
            log.append({
                "role": "state_info",
                "class_name": class_name,
                "content": {k: v for k, v in vars(snap).items() if not k.startswith("_")},
            })
        return log

    def _run_coach(
        self,
        history: list,
        current_state: str,
        schema: list,
        user_request: str,
        phase: str,
        instructions_used: int = 0,
    ) -> str:
        """
        Ask the coach whether to intervene.  Returns "" to stay silent, or a
        one-sentence instruction for the agent's next step.
        """
        budget_info = (
            f"Instruction budget: {instructions_used}/{self.max_coach_instructions} already used for this task.\n"
            if self.max_coach_instructions > 0 else ""
        )
        prompt = COACH_PROMPT.format(
            user_request=user_request,
            schema=json.dumps(schema, indent=2),
            current_state=current_state,
            phase=phase,
            history=self._fmt_messages(history),
            budget_info=budget_info,
        )
        raw = self._call_llm(COACH_SYSTEM, prompt)
        raw = re.sub(r"^```\w*\s*\n?(.*?)\n?```\s*$", r"\1", raw.strip(), flags=re.DOTALL).strip()
        if raw.upper() in {"SILENT", "NONE", "NO INSTRUCTION", "NO-OP", "NOOP", ""}:
            return ""
        raw = re.sub(r"^(instruction|coach instruction)\s*:\s*", "", raw, flags=re.I).strip()
        return " ".join(raw.split())[:500]

    def _hint_log_writer(self) -> None:
        while True:
            entry = self._hint_log_queue.get()
            try:
                if entry is None:
                    break
                with open(self.hint_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            finally:
                self._hint_log_queue.task_done()

    def shutdown_hint_log(self) -> None:
        if self._hint_log_queue is not None:
            self._hint_log_queue.put(None)
            self._hint_log_queue.join()

    def _write_hint_log(self, entry: dict) -> None:
        if self._hint_log_queue is not None:
            self._hint_log_queue.put(entry)

    # ── Core override ─────────────────────────────────────────────────────────

    @override
    def inference_multi_turn_prompting(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict   = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str     = test_entry["id"]
        test_category: str     = extract_test_category_from_id(test_entry_id)
        holdout_function: dict = test_entry.get("missed_function", {})
        long_context: bool     = "long_context" in test_category or "composite" in test_category

        total_input_token_count:  list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency:            list[list[float]] = []
        all_model_response:       list[list]        = []
        all_reasoning_content:    list[list]        = []
        all_inference_log:        list              = []
        force_quit = False

        _, involved_instances = execute_multi_turn_func_call(
            [], initial_config, involved_classes,
            self.model_name_underline_replaced, test_entry_id,
            long_context=long_context, is_evaL_run=False,
        )

        if is_memory(test_category):
            assert len(involved_instances) == 1
            mem = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"], test_category, test_entry["scenario"], mem,
            )

        if not exclude_state_log:
            sl = self._build_state_log(involved_instances)
            if sl:
                all_inference_log.append(sl)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)

        coach_instructions_total = 0  # per-task, never reset between turns

        for turn_idx, current_turn_message in enumerate(test_entry["question"]):
            if str(turn_idx) in holdout_function:
                assert len(current_turn_message) == 0
                current_turn_message = [{
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                        functions=holdout_function[str(turn_idx)]
                    ),
                }]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )

            current_turn_response:           list        = []
            current_turn_reasoning_content:  list        = []
            current_turn_inference_log:      dict        = {"begin_of_turn_query": current_turn_message}
            current_turn_input_token_count:  list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency:            list[float] = []

            user_msgs = [m for m in current_turn_message if m.get("role") == "user"]
            user_request = user_msgs[0]["content"] if user_msgs else ""

            if self.debug:
                print("=" * 100)
                print(f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}")
                print(f"  [User] {user_request}")

            count = 0

            while True:
                if self.debug:
                    print("-" * 100)
                    print(f"ID: {test_entry_id.replace('multi_turn_', '')}, "
                          f"Turn: {turn_idx}, Step: {count}")

                current_step_inference_log: list[dict] = []
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                # True if the previous step's post-execution coach injected a guidance message.
                coached_step = (
                    bool(inference_data["message"])
                    and inference_data["message"][-1].get("role") == "user"
                    and str(inference_data["message"][-1].get("content") or "").startswith(
                        "Supervisor guidance for the next step only:"
                    )
                )

                # Log guidance as a user message so the inference log directly mirrors
                # the model's context (same append-only structure as Agent Sampler).
                if coached_step:
                    current_step_inference_log.append({
                        "role": "user",
                        "content": inference_data["message"][-1]["content"],
                    })

                # ── Query ─────────────────────────────────────────────────────
                api_response, latency = self._query_prompting(inference_data)
                response_data   = self._parse_query_response_prompting(api_response)
                model_responses = response_data["model_responses"]

                current_turn_input_token_count.append(response_data["input_token"])
                current_turn_output_token_count.append(response_data["output_token"])
                current_turn_latency.append(latency)

                fc_message = response_data["model_responses_message_for_chat_history"]

                try:
                    decoded = self.decode_execute(model_responses, has_tool_call_tag=False)
                    response_data["model_responses_decoded"] = decoded
                    has_tool_call = not is_empty_execute_response(decoded)
                except Exception:
                    decoded       = []
                    has_tool_call = False

                # ── No tool call: attempted final answer ───────────────────────
                if not has_tool_call:
                    if self.debug:
                        print(f"  [No FC] model gave text response")

                    inference_data["message"].append(fc_message)

                    can_intervene = (
                        self.max_coach_instructions <= 0
                        or coach_instructions_total < self.max_coach_instructions
                    ) and count < MAXIMUM_STEP_LIMIT

                    if can_intervene:
                        current_state = self._serialize_state(involved_instances)
                        instruction = self._run_coach(
                            inference_data["message"],
                            current_state,
                            inference_data["function"],
                            user_request,
                            phase="after_attempted_final_answer",
                            instructions_used=coach_instructions_total,
                        )
                        self._write_hint_log({
                            "type":          "v1_coach_final",
                            "test_entry_id": test_entry_id,
                            "turn_idx":      turn_idx,
                            "step":          count,
                            "user_request":  user_request,
                            "model_text":    model_responses,
                            "instruction":   instruction,
                        })
                        if self.debug:
                            print(f"  [Coach/Final] {instruction!r}")
                        if instruction:
                            coach_instructions_total += 1
                            current_step_inference_log.append({
                                "role": "handler_log",
                                "content": "Coach intervened on text response.",
                                "hinted_step": coached_step,
                                "final_verdict": "coached",
                            })
                            inference_data["message"].append({
                                "role": "user",
                                "content": f"Supervisor guidance for the next step only: {instruction}",
                            })
                            count += 1
                            continue

                    current_step_inference_log.append(fc_message)
                    current_step_inference_log.append({
                        "role": "handler_log",
                        "content": "Turn ended (text response, no further instruction).",
                        "hinted_step": coached_step,
                        "final_verdict": "silent",
                    })
                    break

                # ── Valid tool call ────────────────────────────────────────────
                if self.debug:
                    for _tc in fc_message.get("tool_calls") or []:
                        _n, _a = (
                            (_tc["function"]["name"], _tc["function"]["arguments"])
                            if "function" in _tc else (_tc["name"], _tc["arguments"])
                        )
                        print(f"  [FC] {_n}({json.dumps(_a) if isinstance(_a, dict) else _a})")

                # Commit the FC to history and execute.
                inference_data["message"].append(fc_message)
                model_responses   = response_data["model_responses"]
                reasoning_content = response_data.get("reasoning_content", "")
                current_turn_response.append(model_responses)
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {"role": "assistant", "content": model_responses}
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content
                current_step_inference_log.append(log_entry)

                if include_input_log:
                    current_step_inference_log.append({
                        "role": "inference_input",
                        "content": inference_data.get("inference_input_log", ""),
                    })

                decoded = response_data["model_responses_decoded"]
                current_step_inference_log.append({
                    "role": "handler_log",
                    "content": "Successfully decoded model response.",
                    "model_response_decoded": decoded,
                })

                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded, initial_config, involved_classes,
                    self.model_name_underline_replaced, test_entry_id,
                    long_context=long_context, is_evaL_run=False,
                )
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, response_data
                )
                for result in execution_results:
                    current_step_inference_log.append({"role": "tool", "content": result})

                # ── Post-execution coach ──────────────────────────────────────
                can_intervene = (
                    self.max_coach_instructions <= 0
                    or coach_instructions_total < self.max_coach_instructions
                ) and count < MAXIMUM_STEP_LIMIT - 1

                if can_intervene:
                    current_state = self._serialize_state(involved_instances)
                    instruction = self._run_coach(
                        inference_data["message"],
                        current_state,
                        inference_data["function"],
                        user_request,
                        phase="after_tool_observation",
                        instructions_used=coach_instructions_total,
                    )
                    self._write_hint_log({
                        "type":          "v1_coach_tool",
                        "test_entry_id": test_entry_id,
                        "turn_idx":      turn_idx,
                        "step":          count,
                        "user_request":  user_request,
                        "instruction":   instruction,
                    })
                    if self.debug:
                        print(f"  [Coach/Tool] {instruction!r}")
                    if instruction:
                        coach_instructions_total += 1
                        inference_data["message"].append({
                            "role": "user",
                            "content": f"Supervisor guidance for the next step only: {instruction}",
                        })
                    current_step_inference_log.append({
                        "role": "handler_log",
                        "content": "Post-execution coach.",
                        "hinted_step": coached_step,
                        "final_verdict": "coached" if instruction else "silent",
                    })

                count += 1
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append({
                        "role": "handler_log",
                        "content": f"Forced quit after {MAXIMUM_STEP_LIMIT} steps.",
                    })
                    break

            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                sl = self._build_state_log(involved_instances)
                if sl:
                    all_inference_log.append(sl)

            if force_quit:
                break

        if is_memory_prereq(test_entry_id):
            assert len(involved_instances) == 1
            list(involved_instances.values())[0]._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }
        if not all(all(c == "" for c in turn_rc) for turn_rc in all_reasoning_content):
            metadata["reasoning_content"] = all_reasoning_content

        if self._hint_log_queue is not None:
            self._hint_log_queue.join()

        return all_model_response, metadata
