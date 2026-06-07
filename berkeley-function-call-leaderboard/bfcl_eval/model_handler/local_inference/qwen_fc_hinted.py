"""
QwenFCHintedHandler — extends QwenFCHandler with a judge-guided hint-injection
step loop.

Within each step:
  1. Small model generates (query → parse → decode_execute).
  2a. If output is empty / text only:
        - Complete-check LLM decides Done or NotDone.
        - NotDone + retries left → inject "please call a tool" hint, retry.
        - Done or retries exhausted → end this turn.
  2b. If output is a valid function call:
        - Judge LLM evaluates correctness (8 yes/no criteria → Good/Bad verdict).
        - Bad + retries left → Hinter LLM writes a one-sentence hint; small model
          retries with only the latest hint visible (prior wrong FCs are hidden).
        - Good or retries exhausted → clean context, commit the final FC, execute.

Inference log / training trajectory: contains no hint messages or wrong FCs —
only the accepted tool call and its execution result per step.
"""

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
from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt, retry_with_backoff
from bfcl_eval.utils import extract_test_category_from_id, is_memory, is_memory_prereq


# ── Prompts ───────────────────────────────────────────────────────────────────
# Three prompts are used in sequence:
#   JUDGE_PROMPT   → big model evaluates whether the small model's FC is correct
#   HINTER_PROMPT  → big model turns the judge's explanation into a one-line hint
#   COMPLETE_PROMPT → big model decides if a text-only response means the task is done

# Judge: evaluates the small model's function call against 8 criteria.
# Outputs yes/no per criterion + <verdict>Good/Bad</verdict> + plain explanation if Bad.
JUDGE_PROMPT = """You are evaluating a model's response in a tool-use conversation. The response may contain one or more tool calls.

[CRITICAL] Tool-only responses are complete. DO NOT mark a response as Bad just because it lacks a user-facing explanation or follow-up message. Multiple tool calls in a single response are ALWAYS acceptable.

## Evaluation Criteria
Evaluate ALL tool calls in the response. For each tool call:
0. Are the arguments a valid dict {{}}? (not "" or null)
1. Is the tool name valid (exists in available tools)?
2. Are all required arguments present and correctly typed?
3. Are the state preconditions satisfied? (required prior actions completed, necessary state conditions met)
4. Does the model have all the information needed to fill the arguments? (IDs or values that should have been retrieved from a prior tool call)
5. Does the selected tool name directly correspond to what the user asked for?
6. Do the argument values accurately reflect the user's specific details?
7. Is the tool call consistent with the conversation history?

Note: Only required parameters are listed in the schema. Extra parameters are acceptable.
If any answer is No, the final verdict is Bad. If all answers are Yes, the final verdict is Good.

## Available Tools
{schema}

## Conversation History (prior turns and steps)
{history}

## Previous Hints Given (earlier retries for this step)
{previous_hints}

## Current System State
{current_state}

## Function Call to Evaluate
Function: {tool_name}
Arguments: {tool_args}

Your response:
0. Yes/No - (if No, one sentence explaining why)
1. Yes/No - (if No, one sentence explaining why)
2. Yes/No - (if No, one sentence explaining why)
3. Yes/No - (if No, one sentence explaining why)
4. Yes/No - (if No, one sentence explaining why)
5. Yes/No - (if No, one sentence explaining why)
6. Yes/No - (if No, one sentence explaining why)
7. Yes/No - (if No, one sentence explaining why)
<verdict>Good/Bad</verdict>
If Bad, in plain language explain what is wrong (no question numbers, no correct answer).
"""

# Hinter: receives the judge's full output and distills it into one actionable sentence.
# Does NOT reveal the correct answer — only points out what to watch for.
HINTER_PROMPT = """A model needs guidance on generating a function call.

## Available Tools
{schema}

## Previous Hints and Retries (earlier attempts in this step)
{previous_hints}

## The Function Call with Issues
Function: {tool_name}
Arguments: {tool_args}

## Evaluation of What Needs Attention
{judge_output}

Based on the evaluation above, in ONE sentence point out what to pay attention to when generating the function call for this task.
Reference the specific tool name or argument if relevant.
If previous hints pointed out conflicting constraints, try a different strategy to satisfy both.
Do NOT reference question numbers. Do NOT give the correct answer. Do NOT ask a question.
Write as general guidance, not as a correction of a specific mistake.

Hint:"""

# Complete-check: used when the small model outputs plain text (no tool call).
# Decides whether that text response means the task is truly finished (Done)
# or whether the model should have made a tool call but didn't (NotDone).
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


# ── Handler ───────────────────────────────────────────────────────────────────

class QwenFCHintedHandler(QwenFCHandler):
    """
    Drop-in replacement for QwenFCHandler that injects judge-guided hints when
    the small model produces a wrong or missing function call.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        judge_api: Optional[str] = None,       # "openai"|"anthropic"; falls back to JUDGE_API env var
        judge_model: Optional[str] = None,     # e.g. "gpt-4o-2024-11-20"; falls back to JUDGE_MODEL env var
        max_hint_retries: int = 3,
        max_judge_concurrency: int = 5,        # semaphore limit for concurrent judge API calls
        hint_log_path: Optional[str] = None,   # falls back to HINT_LOG_PATH env var
        debug: Optional[bool] = None,          # falls back to HINT_DEBUG env var ("1"/"true")
        **kwargs,
    ):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        # Allow all judge settings to be supplied via environment variables so
        # that build_handler() (which only passes the 4 standard args) can still
        # instantiate this class when it is registered in model_config.py.
        judge_api   = judge_api   or os.environ["JUDGE_API"]
        judge_model = judge_model or os.environ["JUDGE_MODEL"]
        hint_log_path = hint_log_path or os.environ.get("HINT_LOG_PATH")
        self.debug = debug if debug is not None else os.environ.get("HINT_DEBUG", "").lower() in ("1", "true")
        self.judge_api = judge_api
        self.judge_model = judge_model
        self.max_hint_retries = max_hint_retries

        # Build the judge LLM client once; shared safely across threads.
        # Semaphore caps concurrent judge calls to avoid rate-limit bursts:
        # worst case is num_threads × max_hint_retries × 2 simultaneous requests.
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
            # Single writer thread mirrors the framework's pattern in
            # _llm_response_generation.py — inference threads only enqueue;
            # one thread does all file I/O, so writes never interleave.
            self._hint_log_queue = queue.Queue()
            t = threading.Thread(target=self._hint_log_writer, daemon=True)
            t.start()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @retry_with_backoff(error_type=RateLimitError)
    def _call_openai(self, prompt: str) -> str:
        kwargs = {"max_completion_tokens": 400}
        if not self.judge_model.startswith("o"):
            kwargs["temperature"] = 0
        resp = self._judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str) -> str:
        resp = self._anthropic_client.messages.create(
            model=self.judge_model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _call_llm(self, prompt: str) -> str:
        """Acquire the concurrency semaphore, then dispatch to the judge LLM."""
        with self._judge_semaphore:
            if self.judge_api == "openai":
                return self._call_openai(prompt)
            return self._call_anthropic(prompt)

    def _serialize_state(self, involved_instances: dict) -> str:
        """
        Snapshot the public attributes of all active backend instances.
        Private attributes (prefixed with _) are skipped — they are internal
        implementation details not relevant to the judge's evaluation.
        """
        state = {
            name: {k: v for k, v in vars(inst).items() if not k.startswith("_")}
            for name, inst in involved_instances.items()
        }
        return json.dumps(state, indent=2, default=str)

    def _fmt_messages(self, msgs: list) -> str:
        """
        Convert a message list to a plain-text representation for LLM prompts.
        Only user messages, tool calls, and tool results are shown — assistant
        text-only messages (e.g. reasoning) are omitted to keep prompts concise.
        """
        parts = []
        for m in msgs:
            role = m.get("role")
            if role == "user":
                parts.append(f"User: {m.get('content') or ''}")
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc["function"] if "function" in tc else tc
                    args = fn["arguments"]
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    parts.append(f"Called: {fn['name']}({args})")
            elif role == "tool":
                parts.append(f"Tool result: {m.get('content', '')}")
        return "\n".join(parts) if parts else "(none)"

    def _run_judge(
        self,
        fc_message: dict,
        history: list,
        current_state: str,
        schema: list,
        hint_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        """
        Ask the judge LLM to evaluate the small model's function call.

        fc_message:   the assistant message containing the tool call(s) to evaluate.
        history:      clean conversation history up to the current step (no intermediate hints).
        hint_history: list of {fc_message, hint} dicts from earlier retries in this step.
        Returns (verdict, raw_judge_output) where verdict is "Good" or "Bad".
        """
        tool_calls = fc_message.get("tool_calls") or []
        # No tool calls in the message — nothing to judge; treat as Good so
        # the outer loop can handle the text-only case separately.
        if not tool_calls:
            return "Good", ""

        # Format tool name and args for the prompt.
        # Parallel tool calls (multiple <tool_call> tags) are joined as a list.
        # QwenFCHandler uses flat format {"name":..., "arguments":...};
        # OpenAI-compatible handlers use {"function": {"name":..., "arguments":...}}.
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
                prev_parts.append(f"Retry {i + 1}:\n  FC: {fc_str}\n  Hint: {entry['hint']}")
            previous_hints = "\n".join(prev_parts)
        else:
            previous_hints = "(none)"

        prompt = JUDGE_PROMPT.format(
            current_state=current_state,
            schema=json.dumps(schema, indent=2),
            history=self._fmt_messages(history),
            previous_hints=previous_hints,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        raw = self._call_llm(prompt)
        m = re.search(r"<verdict>(.*?)</verdict>", raw, re.DOTALL)
        verdict = m.group(1).strip() if m else "Bad"  # default Bad if tag is missing
        return verdict, raw

    def _run_hinter(
        self,
        judge_raw: str,
        fc_message: dict,
        schema: list,
        hint_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Given the judge's full output, ask the hinter LLM to produce a single
        actionable hint sentence. The wrong FC, schema, and prior hints are
        included so the hinter avoids repeating guidance already given.
        """
        tool_calls = fc_message.get("tool_calls") or []

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
                prev_parts.append(f"Retry {i + 1}:\n  FC: {fc_str}\n  Hint: {entry['hint']}")
            previous_hints = "\n".join(prev_parts)
        else:
            previous_hints = "(none)"

        return self._call_llm(HINTER_PROMPT.format(
            schema=json.dumps(schema, indent=2),
            previous_hints=previous_hints,
            tool_name=tool_name,
            tool_args=tool_args,
            judge_output=judge_raw,
        )).strip()

    def _run_complete_check(
        self, messages: list, schema: list, current_state: str, user_request: str
    ) -> tuple[str, str]:
        """
        Ask the judge LLM whether a text-only response means the turn is done.

        messages: full conversation history including the model's text response.
        Returns (verdict, hint) where verdict is "Done" or "NotDone" and hint is
        the one-sentence explanation of what tool call is still needed (empty if Done).
        """
        raw = self._call_llm(COMPLETE_PROMPT.format(
            user_request=user_request,
            schema=json.dumps(schema, indent=2),
            current_state=current_state,
            history=self._fmt_messages(messages),
        ))
        verdict_m = re.search(r"<verdict>(.*?)</verdict>", raw, re.DOTALL)
        hint_m    = re.search(r"<hint>(.*?)</hint>",    raw, re.DOTALL)
        verdict = verdict_m.group(1).strip() if verdict_m else "NotDone"
        hint    = hint_m.group(1).strip()    if hint_m    else ""
        return verdict, hint

    def _build_state_log(self, involved_instances: dict) -> list[dict]:
        """
        Build a state_info log entry for each stateful backend instance.
        deepcopy prevents future mutations from altering the logged snapshot.
        Stateless and explicitly omitted classes are skipped.
        """
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
        """
        Single writer thread: dequeues entries and writes them to the hint log.
        Serialises all file I/O so concurrent inference threads never interleave.
        Stops on a None sentinel (sent by shutdown_hint_log).
        """
        while True:
            entry = self._hint_log_queue.get()
            try:
                if entry is None:  # sentinel — shut down
                    break
                with open(self.hint_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            finally:
                self._hint_log_queue.task_done()

    def shutdown_hint_log(self) -> None:
        """Wait for all pending hint log writes to complete, then stop the writer thread."""
        if self._hint_log_queue is not None:
            self._hint_log_queue.put(None)   # sentinel
            self._hint_log_queue.join()

    def _write_hint_log(self, entry: dict) -> None:
        """Enqueue one hint-correction event for the writer thread."""
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
        # extract_test_category_from_id handles format-sensitivity IDs (contain ":")
        test_category: str     = extract_test_category_from_id(test_entry_id)
        # holdout_function: maps turn index → tool definitions withheld until that turn
        # (used by the "miss function" BFCL category)
        holdout_function: dict = test_entry.get("missed_function", {})
        long_context: bool     = "long_context" in test_category or "composite" in test_category

        total_input_token_count:  list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency:            list[list[float]] = []
        all_model_response:       list[list]        = []
        all_reasoning_content:    list[list]        = []
        all_inference_log:        list              = []
        force_quit = False

        # Run with an empty action list to initialise backend instances and get
        # a reference to them for state logging and judge context.
        _, involved_instances = execute_multi_turn_func_call(
            [], initial_config, involved_classes,
            self.model_name_underline_replaced, test_entry_id,
            long_context=long_context, is_evaL_run=False,
        )

        # Memory category: prepend special system instructions to each turn.
        if is_memory(test_category):
            assert len(involved_instances) == 1
            mem = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"], test_category, test_entry["scenario"], mem,
            )

        # Log initial state before any tool calls.
        if not exclude_state_log:
            sl = self._build_state_log(involved_instances)
            if sl:
                all_inference_log.append(sl)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)

        for turn_idx, current_turn_message in enumerate(test_entry["question"]):
            # "Miss function" category: some tools are withheld at the start and
            # introduced mid-conversation as a synthetic user message.
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

            # The user's natural-language request for this turn, passed to the
            # complete-check prompt so it can reason about task completion.
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
                #
                # Key invariant: the small model's context is always
                #   [clean history up to this step] + [at most one hint]
                #
                # step_start_idx marks the boundary between committed history
                # and anything injected during the retry loop.  At the start of
                # every iteration we slice back to step_start_idx and re-inject
                # only the LATEST hint, so the model never sees its previous
                # wrong FCs or stale hints from earlier retries.
                #
                # After the loop we slice back one final time to remove even the
                # last hint, leaving only the accepted FC in the clean context.
                step_start_idx       = len(inference_data["message"])
                hint_retry           = 0
                final_response_data: Optional[dict] = None
                turn_done            = False
                latest_hint: Optional[str] = None   # most recent hint from the hinter LLM
                step_hint_history:   list[dict] = []  # all {fc_message, hint} from this step's retries
                step_final_verdict:  Optional[str] = None  # last judge/complete_check verdict

                while hint_retry <= self.max_hint_retries:

                    # Reset to clean context boundary, inject the latest hint (if any).
                    del inference_data["message"][step_start_idx:]
                    if latest_hint is not None:
                        inference_data["message"].append(
                            {"role": "user", "content": latest_hint}
                        )

                    # (1) Query: small model generates a response given the current context.
                    api_response, latency = self._query_prompting(inference_data)
                    response_data   = self._parse_query_response_prompting(api_response)
                    model_responses = response_data["model_responses"]

                    current_turn_input_token_count.append(response_data["input_token"])
                    current_turn_output_token_count.append(response_data["output_token"])
                    current_turn_latency.append(latency)

                    # fc_message: the assistant turn containing tool_calls (or text).
                    # Not yet appended to inference_data — we keep the context clean
                    # so the judge sees exactly what the small model was shown.
                    fc_message = response_data["model_responses_message_for_chat_history"]

                    # (2) Decode: parse the raw text output into executable function calls.
                    try:
                        decoded = self.decode_execute(model_responses, has_tool_call_tag=False)
                        response_data["model_responses_decoded"] = decoded
                        has_tool_call = not is_empty_execute_response(decoded)
                    except Exception:
                        decoded       = []
                        has_tool_call = False

                    # (3a) No tool call — model responded with plain text.
                    # Ask the complete-check LLM if the task is actually finished.
                    # We temporarily add fc_message to the history so the checker
                    # can read what the model said, without mutating inference_data.
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
                            # Retry only if the complete-check provided a specific hint.
                            # An empty hint means no further action is identified → Done.
                            latest_hint = complete_hint
                            hint_retry += 1
                            continue
                        # Retries exhausted, or no hint available → end the turn.
                        turn_done = True
                        break

                    # (3b) Valid function call — ask the judge whether it is correct.
                    # Skip judging on the last allowed retry to avoid an infinite loop
                    # if the judge keeps returning Bad.
                    if self.debug:
                        for _tc in fc_message.get("tool_calls") or []:
                            _n, _a = (_tc["function"]["name"], _tc["function"]["arguments"]) if "function" in _tc else (_tc["name"], _tc["arguments"])
                            print(f"  [FC] {_n}({json.dumps(_a) if isinstance(_a, dict) else _a})")
                    if hint_retry < self.max_hint_retries:
                        current_state = self._serialize_state(involved_instances)
                        verdict, judge_raw = self._run_judge(
                            fc_message,
                            inference_data["message"],  # context the model actually saw
                            current_state,
                            inference_data["function"],
                            hint_history=step_hint_history,
                        )
                        step_final_verdict = verdict
                        if self.debug:
                            print(f"  [Judge] verdict={verdict}, retry={hint_retry}")
                            print(f"  [Judge Raw] {judge_raw[:500]}")
                        if verdict == "Bad":
                            hint_text = self._run_hinter(
                                judge_raw, fc_message, inference_data["function"],
                                hint_history=step_hint_history,
                            )
                            if self.debug:
                                print(f"  [Hint] {hint_text!r}")
                            # Log the full correction event for offline analysis.
                            self._write_hint_log({
                                "type":          "judge_hinter",
                                "test_entry_id": test_entry_id,
                                "turn_idx":      turn_idx,
                                "step":          count,
                                "hint_retry":    hint_retry,
                                "user_request":  user_request,
                                "fc_message":    fc_message,
                                "judge_raw":     judge_raw,
                                "hint":          hint_text,
                            })
                            step_hint_history.append({"fc_message": fc_message, "hint": hint_text})
                            latest_hint = hint_text
                            hint_retry += 1
                            continue

                    # Good verdict or retries exhausted → accept this FC.
                    final_response_data = response_data
                    break

                # ── Context cleanup ───────────────────────────────────────────
                # Remove the last injected hint (or nothing, on first attempt).
                # From here on, inference_data["message"] only contains committed
                # history: prior tool calls and their results.
                del inference_data["message"][step_start_idx:]

                if turn_done or final_response_data is None:
                    # Mirror base handler: the final assistant message (text or
                    # empty FC) is committed to context so the next turn sees it.
                    inference_data["message"].append(fc_message)
                    current_step_inference_log.append({
                        "role": "handler_log",
                        "content": "Turn ended (Done verdict or retries exhausted).",
                        "hinted_step": hint_retry > 0,
                        "final_verdict": step_final_verdict,
                    })
                    break

                # Commit only the accepted FC to the clean context.
                inference_data["message"].append(
                    final_response_data["model_responses_message_for_chat_history"]
                )

                # ── Logging ───────────────────────────────────────────────────
                # Record only the accepted response; wrong FCs and hints are not
                # included, so the inference log is a clean training trajectory.
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

                # ── Execute ───────────────────────────────────────────────────
                decoded = final_response_data["model_responses_decoded"]
                current_step_inference_log.append({
                    "role": "handler_log",
                    "content": "Successfully decoded model response.",
                    "model_response_decoded": decoded,
                    "hinted_step": hint_retry > 0,
                    "final_verdict": step_final_verdict,
                })

                # Execute the function calls against the backend instances and
                # append the results to the conversation context.
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

            # Snapshot state after each turn for the inference log.
            if not exclude_state_log:
                sl = self._build_state_log(involved_instances)
                if sl:
                    all_inference_log.append(sl)

            if force_quit:
                break

        # Memory prereq category: flush the in-memory store to disk at end of session.
        if is_memory_prereq(test_entry_id):
            assert len(involved_instances) == 1
            list(involved_instances.values())[0]._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }
        # Only include reasoning_content in metadata if any turn produced it.
        if not all(
            all(c == "" for c in turn_rc)
            for turn_rc in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        # Wait for all hint log entries from this test entry to be written before
        # returning. This guarantees per-entry log completeness without shutting
        # down the queue (which would block subsequent entries).
        # daemon=True means the thread dies with the process, so the very last
        # batch is only guaranteed if shutdown_hint_log() is called at the end
        # of the full run — but in practice queue.join() here covers all entries.
        if self._hint_log_queue is not None:
            self._hint_log_queue.join()

        return all_model_response, metadata
