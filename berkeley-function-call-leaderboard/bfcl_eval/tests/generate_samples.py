#!/usr/bin/env python3
"""
Generate sample JSON traces for each handler version and branch.

Run from: berkeley-function-call-leaderboard/
    python bfcl_eval/tests/generate_samples.py [output_dir]

Output: one JSON file per version under output_dir (default: bfcl_eval/tests/samples/).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

# ── Stub setup (mirrors test_coach_handlers.py) ───────────────────────────────

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

for _mod in ["openai", "openai.RateLimitError", "anthropic", "overrides"]:
    if _mod not in sys.modules:
        _make_stub(_mod)

for _real_pkg in [
    "bfcl_eval", "bfcl_eval.constants",
    "bfcl_eval.eval_checker", "bfcl_eval.eval_checker.multi_turn_eval",
    "bfcl_eval.model_handler", "bfcl_eval.model_handler.local_inference",
]:
    importlib.import_module(_real_pkg)

for _mod in [
    "bfcl_eval.constants.default_prompts",
    "bfcl_eval.constants.executable_backend_config",
    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils",
    "bfcl_eval.model_handler.local_inference.qwen_fc",
    "bfcl_eval.model_handler.utils",
    "bfcl_eval.utils",
]:
    _make_stub(_mod)

sys.modules["openai"].OpenAI = MagicMock
sys.modules["openai"].RateLimitError = Exception
sys.modules["anthropic"].Anthropic = MagicMock
sys.modules["overrides"].override = lambda f: f

dp = sys.modules["bfcl_eval.constants.default_prompts"]
dp.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING = "{functions}"
dp.MAXIMUM_STEP_LIMIT = 20

ec = sys.modules["bfcl_eval.constants.executable_backend_config"]
ec.STATELESS_CLASSES = set()
ec.OMIT_STATE_INFO_CLASSES = set()

mu = sys.modules["bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"]
mu.execute_multi_turn_func_call = MagicMock(return_value=([], {}))
mu.is_empty_execute_response = MagicMock(side_effect=lambda decoded: len(decoded) == 0)

hu = sys.modules["bfcl_eval.model_handler.utils"]
hu.add_memory_instruction_system_prompt = MagicMock(side_effect=lambda q, *a, **kw: q)
hu.retry_with_backoff = lambda error_type: (lambda f: f)

u = sys.modules["bfcl_eval.utils"]
u.extract_test_category_from_id = MagicMock(return_value="multi_turn_base")
u.is_memory = MagicMock(return_value=False)
u.is_memory_prereq = MagicMock(return_value=False)

# ── Stub base class with all prompting methods ────────────────────────────────

class _BaseHandler:
    model_name_underline_replaced = "test-model"

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kw):
        self.model_name = model_name
        self.temperature = temperature
        self.registry_name = registry_name
        self.is_fc_model = is_fc_model

    def _pre_query_processing_prompting(self, test_entry):
        return {"message": [], "function": test_entry.get("function", [])}

    def add_first_turn_message_prompting(self, inference_data, messages):
        inference_data["message"].extend(messages)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data, messages):
        inference_data["message"].extend(messages)
        return inference_data

    def _add_execution_results_prompting(self, inference_data, results, response_data):
        for r in results:
            inference_data["message"].append({"role": "tool", "content": r})
        return inference_data

    # Overridden per run via patch_queries()
    def _query_prompting(self, inference_data):
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response):
        raise NotImplementedError

    def decode_execute(self, result, has_tool_call_tag=False):
        raise NotImplementedError

sys.modules["bfcl_eval.model_handler.local_inference.qwen_fc"].QwenFCHandler = _BaseHandler

# ── Import all four handlers ───────────────────────────────────────────────────

from bfcl_eval.model_handler.local_inference.qwen_fc_hinted import QwenFCHintedHandler  # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen_fc_v1_coach import QwenFCV1CoachHandler  # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen_fc_v2_coach import QwenFCV2CoachHandler  # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen_fc_v3_coach import QwenFCV3CoachHandler  # noqa: E402

# ── Scenario setup ────────────────────────────────────────────────────────────

TASK = "Send an email to bob@example.com saying 'Meeting confirmed'"

TEST_ENTRY = {
    "id": "multi_turn_base_0",
    "question": [[{"role": "user", "content": TASK}]],
    "initial_config": {},
    "involved_classes": [],
    "function": [
        {
            "name": "email.send",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":   {"type": "string", "description": "Recipient email address"},
                    "body": {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "body"],
            },
        },
        {
            "name": "email.verify",
            "description": "Verify that an email was delivered to the recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                },
                "required": ["to"],
            },
        },
    ],
}

def _fc(name, args):
    """Build a response_data dict for a tool-call model output."""
    raw = f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'
    return {
        "model_responses": raw,
        "model_responses_decoded": [{name: args}],
        "model_responses_message_for_chat_history": {
            "role": "assistant", "content": "", "reasoning_content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}],
        },
        "input_token": 120, "output_token": 25, "reasoning_content": "",
    }

def _txt(text):
    """Build a response_data dict for a text-only model output."""
    return {
        "model_responses": text,
        "model_responses_decoded": [],
        "model_responses_message_for_chat_history": {
            "role": "assistant", "content": text, "reasoning_content": "",
        },
        "input_token": 120, "output_token": 12, "reasoning_content": "",
    }

GOOD_FC    = _fc("email.send",   {"to": "bob@example.com", "body": "Meeting confirmed"})
VERIFY_FC  = _fc("email.verify", {"to": "bob@example.com"})
BAD_FC     = _fc("email.send",   {"to": "bob@example.com", "body": ""})
FINAL_TXT  = _txt("I've sent the email to bob@example.com.")
EARLY_TXT  = _txt("Sure, I'll send that right away.")

# ── Runner helpers ────────────────────────────────────────────────────────────

def make_handler(cls, llm_seq):
    """Instantiate handler with a fixed LLM response sequence."""
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        os.environ,
        {"JUDGE_API": "openai", "JUDGE_MODEL": "gpt-4o", "OPENAI_API_KEY": "sk-test"},
    ):
        h = cls("test-model", 0.0, "test", True)
    llm_iter = iter(llm_seq)
    # Both hinted (_call_llm(prompt)) and v1/v2/v3 (_call_llm(system, prompt)) are covered
    # by accepting *args.
    h._call_llm = MagicMock(side_effect=lambda *args: next(llm_iter))
    return h


def patch_queries(h, model_seq):
    """Wire _query_prompting / _parse_query_response_prompting / decode_execute."""
    resp_iter = iter(model_seq)

    h._query_prompting = MagicMock(side_effect=lambda _: (MagicMock(), 0.05))
    h._parse_query_response_prompting = MagicMock(
        side_effect=lambda _: deepcopy(next(resp_iter))
    )

    def _decode(result, has_tool_call_tag=False):
        import re as _re
        m = _re.search(r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}', str(result))
        if m:
            name = m.group(1)
            args = json.loads(m.group(2))
            return [{name: args}]
        return []

    h.decode_execute = MagicMock(side_effect=_decode)


def patch_execute(fc_result="Email sent successfully."):
    """Set execute_multi_turn_func_call to return tool result for FC calls."""
    def _side(decoded, *args, **kwargs):
        if decoded:
            return ([fc_result], {})
        return ([], {})
    mu.execute_multi_turn_func_call.side_effect = _side


def run_scenario(cls, model_seq, llm_seq):
    """
    Run inference_multi_turn_prompting for one branch scenario.
    Returns (llm_call_log, inference_log).
    """
    patch_execute()
    h = make_handler(cls, llm_seq)
    patch_queries(h, model_seq)

    # Wrap _call_llm to log calls
    _orig = h._call_llm
    llm_log = []

    def _capturing(*args):
        result = _orig(*args)
        llm_log.append({
            "system_or_prompt": str(args[0])[:300] if len(args) == 2 else "(hinted: no system)",
            "user_prompt_preview": str(args[-1])[:300],
            "response": result,
        })
        return result

    h._call_llm = _capturing

    _, metadata = h.inference_multi_turn_prompting(
        deepcopy(TEST_ENTRY),
        include_input_log=False,
        exclude_state_log=True,
    )
    return llm_log, metadata["inference_log"]


def build_sample(name, description, model_seq, llm_seq, cls):
    print(f"    {name} ...", end=" ", flush=True)
    try:
        llm_log, inf_log = run_scenario(cls, model_seq, llm_seq)
        print("OK")
        return {
            "branch": name,
            "description": description,
            "model_outputs": [r.get("model_responses", "") for r in model_seq],
            "llm_responses": llm_seq,
            "llm_calls": llm_log,
            "inference_log": inf_log,
        }
    except Exception as exc:
        import traceback
        print(f"FAILED: {exc}")
        traceback.print_exc()
        return {"branch": name, "description": description, "error": str(exc)}

# ── Branch definitions ────────────────────────────────────────────────────────

BRANCHES: dict[str, list[dict]] = {
    # Hinted: judge uses <verdict>Good/Bad</verdict> XML; complete_check uses <verdict>Done/NotDone</verdict>.
    # After each accepted FC the outer loop continues → model queried again → needs terminal FINAL_TXT.
    "hinted": [
        {
            "name": "fc_good",
            "description": "Correct FC → judge Good → execute; outer loop continues → final text accepted",
            "model_seq": [GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "<verdict>Good</verdict>",
                "<verdict>Done</verdict>\n<hint></hint>",
            ],
        },
        {
            "name": "fc_bad_retry",
            "description": "Wrong FC → judge Bad → hinter hints → retry correct FC → judge Good → final text",
            "model_seq": [BAD_FC, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "<verdict>Bad</verdict>",
                "The body argument is empty; use 'Meeting confirmed' as the message text.",
                "<verdict>Good</verdict>",
                "<verdict>Done</verdict>\n<hint></hint>",
            ],
        },
        {
            "name": "no_fc_done",
            "description": "Text response → complete_check Done → turn ends immediately",
            "model_seq": [FINAL_TXT],
            "llm_seq": ["<verdict>Done</verdict>\n<hint></hint>"],
        },
        {
            "name": "no_fc_not_done_retry",
            "description": "Premature text → NotDone hint → retry with FC → judge Good → final text",
            "model_seq": [EARLY_TXT, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "<verdict>NotDone</verdict>\n<hint>Call email.send to actually send the message.</hint>",
                "<verdict>Good</verdict>",
                "<verdict>Done</verdict>\n<hint></hint>",
            ],
        },
    ],
    # V1: coach fires POST-execution; after FC always continues outer loop;
    # turn ends only when no-FC + coach silent.
    "v1_coach": [
        {
            "name": "fc_coach_silent",
            "description": "FC executed → post-execution coach silent → outer loop continues → final text accepted",
            "model_seq": [GOOD_FC, FINAL_TXT],
            "llm_seq": ["SILENT", "SILENT"],
        },
        {
            "name": "fc_coach_instructs",
            "description": "FC1 executed → coach instructs 'verify delivery' → FC2 (verify) executed → coach silent → final text; context_snapshot at step_2 shows both FC1+result and FC2+result",
            "model_seq": [GOOD_FC, VERIFY_FC, FINAL_TXT],
            "llm_seq": [
                "Good. Now verify the email was delivered by calling email.verify.",
                "SILENT",
                "SILENT",
            ],
        },
        {
            "name": "no_fc_accepted",
            "description": "Text response → after_attempted_final_answer coach silent → turn ends",
            "model_seq": [FINAL_TXT],
            "llm_seq": ["SILENT"],
        },
        {
            "name": "no_fc_redirected",
            "description": "Premature text → coach instructs → FC executed → coach silent → final text accepted",
            "model_seq": [EARLY_TXT, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "Call email.send to actually send the message.",
                "SILENT",
                "SILENT",
            ],
        },
    ],
    # V2: pre-execution coach for FC; COMPLETE_PROMPT for no-FC;
    # after FC outer loop continues → needs terminal text + Done verdict.
    "v2_coach": [
        {
            "name": "fc_coach_silent",
            "description": "Correct FC → pre-execution coach approves → execute; outer continues → final text Done",
            "model_seq": [GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "SILENT",
                "<verdict>Done</verdict>\n<hint></hint>",
            ],
        },
        {
            "name": "fc_coach_hint_retry",
            "description": "Wrong FC → coach hints → retry correct FC → coach approves → execute; final text Done",
            "model_seq": [BAD_FC, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "The body argument is empty; set it to 'Meeting confirmed'.",
                "SILENT",
                "<verdict>Done</verdict>\n<hint></hint>",
            ],
        },
        {
            "name": "no_fc_done",
            "description": "Text response → COMPLETE_PROMPT Done → turn ends immediately",
            "model_seq": [FINAL_TXT],
            "llm_seq": ["<verdict>Done</verdict>\n<hint></hint>"],
        },
        {
            "name": "no_fc_not_done_retry",
            "description": "Premature text → NotDone hint → retry FC → coach approves → execute; final text Done",
            "model_seq": [EARLY_TXT, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "<verdict>NotDone</verdict>\n<hint>Call email.send to send the message.</hint>",
                "SILENT",
                "<verdict>Done</verdict>\n<hint></hint>",
            ],
        },
    ],
    # V3: same as V2 for FC branches; FINAL_COACH (silent/text) replaces COMPLETE_PROMPT.
    "v3_coach": [
        {
            "name": "fc_coach_silent",
            "description": "Correct FC → pre-execution coach approves → execute; outer continues → final text, FINAL_COACH silent",
            "model_seq": [GOOD_FC, FINAL_TXT],
            "llm_seq": ["SILENT", "SILENT"],
        },
        {
            "name": "fc_coach_hint_retry",
            "description": "Wrong FC → coach hints → retry correct FC → coach approves → execute; final text, FINAL_COACH silent",
            "model_seq": [BAD_FC, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "The body argument is empty; set it to 'Meeting confirmed'.",
                "SILENT",
                "SILENT",
            ],
        },
        {
            "name": "no_fc_accepted",
            "description": "Text response → FINAL_COACH silent → turn ends immediately",
            "model_seq": [FINAL_TXT],
            "llm_seq": ["SILENT"],
        },
        {
            "name": "no_fc_redirected",
            "description": "Premature text → FINAL_COACH instructs → retry FC → coach approves → execute; final text silent",
            "model_seq": [EARLY_TXT, GOOD_FC, FINAL_TXT],
            "llm_seq": [
                "Call email.send to actually send the message.",
                "SILENT",
                "SILENT",
            ],
        },
    ],
}

HANDLER_CLS = {
    "hinted":   QwenFCHintedHandler,
    "v1_coach": QwenFCV1CoachHandler,
    "v2_coach": QwenFCV2CoachHandler,
    "v3_coach": QwenFCV3CoachHandler,
}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bfcl_eval/tests/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for version, branch_list in BRANCHES.items():
        print(f"[{version}]")
        cls = HANDLER_CLS[version]
        samples = [
            build_sample(
                name=b["name"],
                description=b["description"],
                model_seq=b["model_seq"],
                llm_seq=b["llm_seq"],
                cls=cls,
            )
            for b in branch_list
        ]
        out_path = out_dir / f"{version}.json"
        out_path.write_text(
            json.dumps({"version": version, "task": TASK, "branches": samples},
                       indent=2, ensure_ascii=False)
        )
        print(f"  → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
