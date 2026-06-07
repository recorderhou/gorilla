"""
QwenFCV2CoachHandler — pre-execution coaching with retry (Version 2, hybrid style).

Context management and trigger timing from qwen_fc_hinted:
  - Coach evaluates the function call BEFORE execution.
  - Clean step boundary (step_start_idx): only committed FCs + results in history.
  - hint_history per step accumulates all previous (FC, hint) pairs for the coach.
  - Retry loop: bad FC → hint injected → model retries (up to max_hint_retries).

Judge + hinter combined into ONE LLM call (Agent-Sampler coach style):
  - Empty response (or "SILENT") → approve the FC and execute it.
  - Any text                     → one-sentence hint; model retries.

Training log is clean: no hints, no rejected FCs — only the accepted call per step.
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
#
# Single combined coach prompt replaces the separate JUDGE_PROMPT + HINTER_PROMPT
# used in qwen_fc_hinted.  One call decides both verdict and guidance.

COACH_SYSTEM = """You are reviewing a tool-use agent's planned function call before it executes.

Choose exactly one option:
1. Return an empty response (or exactly SILENT) if the planned call is correct.
2. Return ONE short hint if there is a clear issue.

Do NOT reveal the correct answer or the correct function call.
Do NOT repeat hints that have already been given for this step.
Only intervene when there is a clear, concrete problem with the planned call."""

COACH_PROMPT = """## Task
{user_request}

## Available Tools
{schema}

## Conversation History (committed turns)
{history}

## Previous Hints This Step
{previous_hints}

## Current System State
{current_state}

## Planned Function Call
Function: {tool_name}
Arguments: {tool_args}

Return an empty response to approve this call, or one sentence to guide correction."""

# Complete-check prompt (used when model outputs text instead of a tool call).
COMPLETE_PROMPT = """You are reviewing a tool-use conversation turn.

[CRITICAL] When in doubt, output Done. If the model has made reasonable progress or executed relevant tool calls, output Done.

## User Request for This Turn
{user_request}

## Available Tools
{schema}

## Current System State
{current_state}

## Conversation History
{history}

Has the model fully completed the user's request for this turn, or does it still need to make function calls?

<verdict>Done</verdict>    — the user's request has been fully completed
<verdict>NotDone</verdict> — the model should have made a function call but didn't

<verdict>Done/NotDone</verdict>
<hint>If NotDone, one sentence explaining what tool call is still needed. Empty if Done.</hint>
"""


# ── Handler ───────────────────────────────────────────────────────────────────

class QwenFCV2CoachHandler(QwenFCHandler):
    """
    Pre-execution combined-coach variant of QwenFCHintedHandler.
    Structure is identical to qwen_fc_hinted, but the two-step judge → hinter
    pipeline is replaced by a single coach call.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        judge_api: Optional[str] = None,
        judge_model: Optional[str] = None,
        max_hint_retries: int = 3,
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
        self.max_hint_retries = max_hint_retries

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
        kwargs = {"max_completion_tokens": 300}
        if not self.judge_model.startswith("o"):
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
            max_tokens=300,
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
        state = {
            name: {k: v for k, v in vars(inst).items() if not k.startswith("_")}
            for name, inst in involved_instances.items()
        }
        return json.dumps(state, indent=2, default=str)

    def _fmt_messages(self, msgs: list) -> str:
        parts = []
        for m in msgs:
            role = m.get("role")
            if role == "user":
                parts.append(f"User: {m.get('content') or ''}")
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

    def _run_coach(
        self,
        fc_message: dict,
        history: list,
        current_state: str,
        schema: list,
        user_request: str,
        hint_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Combined judge + hinter.  Returns "" to approve the FC, or a one-sentence
        hint if the FC needs correction.

        fc_message:   assistant message with the tool call(s) to evaluate.
        history:      clean conversation history up to the current step.
        hint_history: [{fc_message, hint}] from earlier retries in this step.
        """
        tool_calls = fc_message.get("tool_calls") or []
        if not tool_calls:
            return ""  # nothing to evaluate

        def _tc_name_args(tc):
            if "function" in tc:
                fn = tc["function"]
                return fn["name"], fn["arguments"]
            return tc["name"], tc["arguments"]

        def _fmt_tc(tc):
            n, a = _tc_name_args(tc)
            return f"{n}({json.dumps(a) if isinstance(a, dict) else str(a)})"

        if len(tool_calls) == 1:
            tool_name, args = _tc_name_args(tool_calls[0])
            tool_args = json.dumps(args) if isinstance(args, dict) else str(args)
        else:
            names_args = [_tc_name_args(tc) for tc in tool_calls]
            tool_name = ", ".join(n for n, _ in names_args)
            tool_args = "\n".join(
                f"{n}: " + (json.dumps(a) if isinstance(a, dict) else str(a))
                for n, a in names_args
            )

        if hint_history:
            prev_parts = []
            for i, entry in enumerate(hint_history):
                prev_tcs = entry["fc_message"].get("tool_calls") or []
                fc_str = ", ".join(_fmt_tc(tc) for tc in prev_tcs) if prev_tcs else "(no FC)"
                prev_parts.append(f"Attempt {i + 1}:\n  FC: {fc_str}\n  Hint: {entry['hint']}")
            previous_hints = "\n".join(prev_parts)
        else:
            previous_hints = "(none)"

        prompt = COACH_PROMPT.format(
            user_request=user_request,
            schema=json.dumps(schema, indent=2),
            history=self._fmt_messages(history),
            previous_hints=previous_hints,
            current_state=current_state,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        raw = self._call_llm(COACH_SYSTEM, prompt)
        raw = re.sub(r"^```\w*\s*\n?(.*?)\n?```\s*$", r"\1", raw.strip(), flags=re.DOTALL).strip()
        if raw.upper() in {"SILENT", "NONE", "NO INSTRUCTION", "NO-OP", "NOOP", ""}:
            return ""
        return raw

    def _run_complete_check(
        self, messages: list, schema: list, current_state: str, user_request: str
    ) -> tuple[str, str]:
        raw = self._call_llm(
            "",   # no separate system prompt for complete-check
            COMPLETE_PROMPT.format(
                user_request=user_request,
                schema=json.dumps(schema, indent=2),
                current_state=current_state,
                history=self._fmt_messages(messages),
            ),
        )
        verdict_m = re.search(r"<verdict>(.*?)</verdict>", raw, re.DOTALL)
        hint_m    = re.search(r"<hint>(.*?)</hint>",    raw, re.DOTALL)
        verdict = verdict_m.group(1).strip() if verdict_m else "NotDone"
        hint    = hint_m.group(1).strip()    if hint_m    else ""
        return verdict, hint

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

                # ── Hint retry sub-loop ───────────────────────────────────────
                step_start_idx       = len(inference_data["message"])
                hint_retry           = 0
                final_response_data: Optional[dict] = None
                turn_done            = False
                latest_hint: Optional[str] = None
                step_hint_history:   list[dict] = []
                step_final_verdict:  Optional[str] = None

                while hint_retry <= self.max_hint_retries:

                    del inference_data["message"][step_start_idx:]
                    if latest_hint is not None:
                        inference_data["message"].append(
                            {"role": "user", "content": latest_hint}
                        )

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

                    # ── No tool call: complete-check ──────────────────────────
                    if not has_tool_call:
                        current_state = self._serialize_state(involved_instances)
                        verdict, complete_hint = self._run_complete_check(
                            inference_data["message"] + [fc_message],
                            inference_data["function"],
                            current_state,
                            user_request,
                        )
                        step_final_verdict = verdict
                        self._write_hint_log({
                            "type":          "complete_check",
                            "test_entry_id": test_entry_id,
                            "turn_idx":      turn_idx,
                            "step":          count,
                            "hint_retry":    hint_retry,
                            "user_request":  user_request,
                            "model_text":    model_responses,
                            "verdict":       verdict,
                            "hint":          complete_hint,
                        })
                        if self.debug:
                            print(f"  [Complete] verdict={verdict}")
                            if complete_hint:
                                print(f"  [Complete Hint] {complete_hint!r}")
                        if verdict == "Done" or count >= MAXIMUM_STEP_LIMIT:
                            turn_done = True
                            break
                        if hint_retry < self.max_hint_retries and complete_hint:
                            latest_hint = complete_hint
                            hint_retry += 1
                            continue
                        turn_done = True
                        break

                    # ── Valid FC: ask combined coach ──────────────────────────
                    if self.debug:
                        for _tc in fc_message.get("tool_calls") or []:
                            _n, _a = (
                                (_tc["function"]["name"], _tc["function"]["arguments"])
                                if "function" in _tc else (_tc["name"], _tc["arguments"])
                            )
                            print(f"  [FC] {_n}({json.dumps(_a) if isinstance(_a, dict) else _a})")

                    if hint_retry < self.max_hint_retries:
                        current_state = self._serialize_state(involved_instances)
                        hint_text = self._run_coach(
                            fc_message,
                            inference_data["message"],
                            current_state,
                            inference_data["function"],
                            user_request,
                            hint_history=step_hint_history,
                        )
                        step_final_verdict = "Bad" if hint_text else "Good"
                        if self.debug:
                            if hint_text:
                                print(f"  [Coach] hint={hint_text!r}, retry={hint_retry}")
                            else:
                                print(f"  [Coach] approved, retry={hint_retry}")
                        if hint_text:
                            self._write_hint_log({
                                "type":          "v2_coach",
                                "test_entry_id": test_entry_id,
                                "turn_idx":      turn_idx,
                                "step":          count,
                                "hint_retry":    hint_retry,
                                "user_request":  user_request,
                                "fc_message":    fc_message,
                                "hint":          hint_text,
                            })
                            step_hint_history.append({"fc_message": fc_message, "hint": hint_text})
                            latest_hint = hint_text
                            hint_retry += 1
                            continue

                    # Approved (or retries exhausted) → accept this FC.
                    final_response_data = response_data
                    break

                # ── Context cleanup ───────────────────────────────────────────
                del inference_data["message"][step_start_idx:]

                if turn_done or final_response_data is None:
                    inference_data["message"].append(fc_message)
                    current_step_inference_log.append({
                        "role": "handler_log",
                        "content": "Turn ended (Done verdict or retries exhausted).",
                        "hinted_step": hint_retry > 0,
                        "final_verdict": step_final_verdict,
                    })
                    break

                inference_data["message"].append(
                    final_response_data["model_responses_message_for_chat_history"]
                )

                model_responses   = final_response_data["model_responses"]
                reasoning_content = final_response_data.get("reasoning_content", "")
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

                decoded = final_response_data["model_responses_decoded"]
                current_step_inference_log.append({
                    "role": "handler_log",
                    "content": "Successfully decoded model response.",
                    "model_response_decoded": decoded,
                    "hinted_step": hint_retry > 0,
                    "final_verdict": step_final_verdict,
                })

                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded, initial_config, involved_classes,
                    self.model_name_underline_replaced, test_entry_id,
                    long_context=long_context, is_evaL_run=False,
                )
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, final_response_data
                )
                for result in execution_results:
                    current_step_inference_log.append({"role": "tool", "content": result})

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
