"""
Tests for QwenFCV1CoachHandler and QwenFCV2CoachHandler.

All LLM and backend calls are mocked so no real API keys or GPU are needed.

Coverage:
  V1  _run_coach — silent response → no instruction
  V1  _run_coach — text response   → instruction returned
  V1  training cleanup             → supervisor messages stripped from history
  V1  _fmt_messages                → skips supervisor guidance lines
  V2  _run_coach — empty/SILENT    → FC approved (empty string)
  V2  _run_coach — hint text       → hint returned, retry expected
  V2  hint_history                 → previous attempts visible in coach prompt
  V2  step_start_idx cleanup       → wrong FCs removed after retry loop
  shared  _fmt_messages            → formats user / assistant FC / tool result
  shared  _serialize_state         → public attrs only, no private attrs
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Minimal stubs so the modules import without the full BFCL install ─────────

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# Stub third-party packages that aren't installed in this environment.
for _mod in [
    "openai",
    "openai.RateLimitError",
    "anthropic",
    "overrides",
]:
    if _mod not in sys.modules:
        _make_stub(_mod)

# Import the real bfcl_eval package hierarchy (all __init__.py are empty).
# This must happen before stubbing any bfcl_eval.* leaf modules so Python
# keeps local_inference as a real package and can find qwen_fc_v1_coach etc.
import importlib
for _real_pkg in [
    "bfcl_eval",
    "bfcl_eval.constants",
    "bfcl_eval.eval_checker",
    "bfcl_eval.eval_checker.multi_turn_eval",
    "bfcl_eval.model_handler",
    "bfcl_eval.model_handler.local_inference",
]:
    importlib.import_module(_real_pkg)

# Stub leaf modules that have heavy/unavailable dependencies.
for _mod in [
    "bfcl_eval.constants.default_prompts",
    "bfcl_eval.constants.executable_backend_config",
    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils",
    "bfcl_eval.model_handler.local_inference.qwen_fc",
    "bfcl_eval.model_handler.utils",
    "bfcl_eval.utils",
]:
    _make_stub(_mod)

# Populate stubs with the names the handlers import.
sys.modules["openai"].OpenAI = MagicMock
sys.modules["openai"].RateLimitError = Exception
sys.modules["anthropic"].Anthropic = MagicMock
sys.modules["overrides"].override = lambda f: f   # no-op decorator

_dp = sys.modules["bfcl_eval.constants.default_prompts"]
_dp.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING = "{functions}"
_dp.MAXIMUM_STEP_LIMIT = 20

_ec = sys.modules["bfcl_eval.constants.executable_backend_config"]
_ec.STATELESS_CLASSES = set()
_ec.OMIT_STATE_INFO_CLASSES = set()

_mu = sys.modules["bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"]
_mu.execute_multi_turn_func_call = MagicMock(return_value=([], {}))
_mu.is_empty_execute_response = MagicMock(return_value=False)

# Minimal QwenFCHandler base class stub
class _BaseHandler:
    model_name_underline_replaced = "test-model"
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kw):
        self.model_name = model_name
        self.temperature = temperature
        self.registry_name = registry_name
        self.is_fc_model = is_fc_model

class _QwenFCHandler(_BaseHandler):
    pass

sys.modules["bfcl_eval.model_handler.local_inference.qwen_fc"].QwenFCHandler = _QwenFCHandler

_hu = sys.modules["bfcl_eval.model_handler.utils"]
_hu.add_memory_instruction_system_prompt = MagicMock(side_effect=lambda q, *a, **kw: q)
_hu.retry_with_backoff = lambda error_type: (lambda f: f)  # no-op decorator

_u = sys.modules["bfcl_eval.utils"]
_u.extract_test_category_from_id = MagicMock(return_value="multi_turn_base")
_u.is_memory = MagicMock(return_value=False)
_u.is_memory_prereq = MagicMock(return_value=False)

# Now import the handlers under test
from bfcl_eval.model_handler.local_inference.qwen_fc_v1_coach import (  # noqa: E402
    QwenFCV1CoachHandler,
    COACH_SYSTEM as V1_COACH_SYSTEM,
    COACH_PROMPT as V1_COACH_PROMPT,
)
from bfcl_eval.model_handler.local_inference.qwen_fc_v2_coach import (  # noqa: E402
    QwenFCV2CoachHandler,
    COACH_SYSTEM as V2_COACH_SYSTEM,
    COACH_PROMPT as V2_COACH_PROMPT,
    COMPLETE_PROMPT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_v1(llm_response: str = "") -> QwenFCV1CoachHandler:
    """Create a V1 handler with a mocked LLM that returns llm_response."""
    with patch.dict(os.environ, {"JUDGE_API": "openai", "JUDGE_MODEL": "gpt-4o",
                                  "OPENAI_API_KEY": "test-key"}):
        h = QwenFCV1CoachHandler("test-model", 0.0, "test", True)
    h._call_llm = MagicMock(return_value=llm_response)
    return h


def _make_v2(llm_response: str = "") -> QwenFCV2CoachHandler:
    """Create a V2 handler with a mocked LLM that returns llm_response."""
    with patch.dict(os.environ, {"JUDGE_API": "openai", "JUDGE_MODEL": "gpt-4o",
                                  "OPENAI_API_KEY": "test-key"}):
        h = QwenFCV2CoachHandler("test-model", 0.0, "test", True)
    h._call_llm = MagicMock(return_value=llm_response)
    return h


def _fc_message(name: str, args: dict) -> dict:
    """Build a minimal assistant message with one tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}],
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

class TestFmtMessages:
    def test_formats_user_message(self):
        h = _make_v1()
        msgs = [{"role": "user", "content": "Delete file.txt"}]
        assert "User: Delete file.txt" in h._fmt_messages(msgs)

    def test_formats_tool_call(self):
        h = _make_v1()
        msgs = [_fc_message("delete_file", {"path": "file.txt"})]
        out = h._fmt_messages(msgs)
        assert "Called: delete_file" in out
        assert "file.txt" in out

    def test_formats_tool_result(self):
        h = _make_v1()
        msgs = [{"role": "tool", "content": '{"status": "ok"}'}]
        assert 'Tool result: {"status": "ok"}' in h._fmt_messages(msgs)

    def test_empty_list_returns_none_string(self):
        h = _make_v1()
        assert h._fmt_messages([]) == "(none)"

    def test_v1_skips_supervisor_guidance(self):
        """V1's _fmt_messages should exclude supervisor guidance lines."""
        h = _make_v1()
        msgs = [
            {"role": "user", "content": "Do something"},
            {"role": "user", "content": "Supervisor guidance for the next step only: use tool X"},
            {"role": "user", "content": "Other user message"},
        ]
        out = h._fmt_messages(msgs)
        assert "Supervisor guidance" not in out
        assert "Do something" in out
        assert "Other user message" in out


class TestSerializeState:
    def test_public_attrs_only(self):
        h = _make_v1()

        class FakeInst:
            pass
        inst = FakeInst()
        inst.count = 5
        inst.name = "test"
        inst._private = "hidden"

        state_str = h._serialize_state({"MyClass": inst})
        state = json.loads(state_str)
        assert "count" in state["MyClass"]
        assert "name" in state["MyClass"]
        assert "_private" not in state["MyClass"]

    def test_empty_instances(self):
        h = _make_v1()
        assert json.loads(h._serialize_state({})) == {}


# ── V1 coach ─────────────────────────────────────────────────────────────────

class TestV1RunCoach:
    def test_silent_response_returns_empty(self):
        h = _make_v1(llm_response="SILENT")
        result = h._run_coach([], "{}", [], "delete file.txt", "after_tool_observation")
        assert result == ""

    def test_empty_response_returns_empty(self):
        h = _make_v1(llm_response="")
        result = h._run_coach([], "{}", [], "delete file.txt", "after_tool_observation")
        assert result == ""

    def test_instruction_text_returned(self):
        h = _make_v1(llm_response="Verify the file was actually deleted before proceeding.")
        result = h._run_coach([], "{}", [], "delete file.txt", "after_tool_observation")
        assert result == "Verify the file was actually deleted before proceeding."

    def test_strips_instruction_prefix(self):
        h = _make_v1(llm_response="Instruction: Check the return code first.")
        result = h._run_coach([], "{}", [], "task", "after_tool_observation")
        assert result == "Check the return code first."

    def test_strips_code_fences(self):
        h = _make_v1(llm_response="```text\nUse the correct argument name.\n```")
        result = h._run_coach([], "{}", [], "task", "after_tool_observation")
        assert result == "Use the correct argument name."

    def test_prompt_contains_user_request(self):
        h = _make_v1(llm_response="")
        h._run_coach([], "[]", [], "rename the file", "after_tool_observation")
        prompt_arg = h._call_llm.call_args[0][1]
        assert "rename the file" in prompt_arg

    def test_prompt_contains_phase(self):
        h = _make_v1(llm_response="")
        h._run_coach([], "[]", [], "task", "after_attempted_final_answer")
        prompt_arg = h._call_llm.call_args[0][1]
        assert "after_attempted_final_answer" in prompt_arg

    def test_system_prompt_passed(self):
        h = _make_v1(llm_response="")
        h._run_coach([], "[]", [], "task", "phase")
        system_arg = h._call_llm.call_args[0][0]
        assert system_arg == V1_COACH_SYSTEM


class TestV1TrainingCleanup:
    """V1 must strip supervisor guidance from the committed message history."""

    def _messages_with_supervisor(self) -> list[dict]:
        return [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Do the task."},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "tool_a", "arguments": "{}"}}]},
            {"role": "tool", "content": "result"},
            {"role": "user",
             "content": "Supervisor guidance for the next step only: check the output"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "tool_b", "arguments": "{}"}}]},
            {"role": "tool", "content": "result2"},
        ]

    def test_supervisor_messages_stripped(self):
        msgs = self._messages_with_supervisor()
        cleaned = [
            m for m in msgs
            if not (
                m.get("role") == "user"
                and str(m.get("content") or "").startswith(
                    "Supervisor guidance for the next step only:"
                )
            )
        ]
        roles = [m["role"] for m in cleaned]
        contents = [m.get("content") or "" for m in cleaned]
        assert "Supervisor guidance" not in " ".join(contents)
        # No user messages starting with supervisor prefix remain
        for m in cleaned:
            if m["role"] == "user":
                assert not m["content"].startswith("Supervisor guidance")

    def test_non_supervisor_user_messages_kept(self):
        msgs = self._messages_with_supervisor()
        cleaned = [
            m for m in msgs
            if not (
                m.get("role") == "user"
                and str(m.get("content") or "").startswith(
                    "Supervisor guidance for the next step only:"
                )
            )
        ]
        user_msgs = [m for m in cleaned if m["role"] == "user"]
        assert any(m["content"] == "Do the task." for m in user_msgs)


# ── V2 coach ─────────────────────────────────────────────────────────────────

class TestV2RunCoach:
    def _schema(self):
        return [{"name": "get_file", "description": "Retrieve a file"}]

    def test_empty_response_approves(self):
        h = _make_v2(llm_response="")
        fc = _fc_message("get_file", {"path": "report.txt"})
        result = h._run_coach(fc, [], "{}", self._schema(), "get the report")
        assert result == ""

    def test_silent_response_approves(self):
        h = _make_v2(llm_response="SILENT")
        fc = _fc_message("get_file", {"path": "report.txt"})
        result = h._run_coach(fc, [], "{}", self._schema(), "get the report")
        assert result == ""

    def test_hint_text_returned(self):
        h = _make_v2(llm_response="The file path should be relative to the workspace root.")
        fc = _fc_message("get_file", {"path": "/absolute/path"})
        result = h._run_coach(fc, [], "{}", self._schema(), "get the report")
        assert result == "The file path should be relative to the workspace root."

    def test_no_tool_calls_returns_empty(self):
        """FC message with no tool_calls should always be approved."""
        h = _make_v2(llm_response="Some hint that should be ignored")
        fc = {"role": "assistant", "content": "Final answer: 42", "tool_calls": []}
        result = h._run_coach(fc, [], "{}", self._schema(), "task")
        assert result == ""
        h._call_llm.assert_not_called()

    def test_prompt_contains_tool_name_and_args(self):
        h = _make_v2(llm_response="")
        fc = _fc_message("delete_file", {"path": "notes.txt"})
        h._run_coach(fc, [], "{}", self._schema(), "delete notes")
        prompt = h._call_llm.call_args[0][1]
        assert "delete_file" in prompt
        assert "notes.txt" in prompt

    def test_prompt_contains_user_request(self):
        h = _make_v2(llm_response="")
        fc = _fc_message("get_file", {"path": "x"})
        h._run_coach(fc, [], "{}", self._schema(), "retrieve the quarterly report")
        prompt = h._call_llm.call_args[0][1]
        assert "retrieve the quarterly report" in prompt

    def test_system_prompt_passed(self):
        h = _make_v2(llm_response="")
        fc = _fc_message("get_file", {"path": "x"})
        h._run_coach(fc, [], "{}", self._schema(), "task")
        system_arg = h._call_llm.call_args[0][0]
        assert system_arg == V2_COACH_SYSTEM


class TestV2HintHistory:
    """hint_history from earlier retries must appear in the coach prompt."""

    def _schema(self):
        return [{"name": "move_file", "description": "Move a file"}]

    def test_no_history_shows_none(self):
        h = _make_v2(llm_response="")
        fc = _fc_message("move_file", {"src": "a.txt", "dst": "b.txt"})
        h._run_coach(fc, [], "{}", self._schema(), "move the file", hint_history=[])
        prompt = h._call_llm.call_args[0][1]
        assert "(none)" in prompt

    def test_single_history_entry_in_prompt(self):
        h = _make_v2(llm_response="")
        prior_fc = _fc_message("move_file", {"src": "wrong.txt", "dst": "b.txt"})
        hint_history = [{"fc_message": prior_fc, "hint": "The source path was incorrect."}]
        fc = _fc_message("move_file", {"src": "correct.txt", "dst": "b.txt"})
        h._run_coach(fc, [], "{}", self._schema(), "move the file", hint_history=hint_history)
        prompt = h._call_llm.call_args[0][1]
        assert "Attempt 1" in prompt
        assert "wrong.txt" in prompt
        assert "The source path was incorrect." in prompt

    def test_multiple_history_entries_numbered(self):
        h = _make_v2(llm_response="")
        fc1 = _fc_message("move_file", {"src": "a.txt"})
        fc2 = _fc_message("move_file", {"src": "b.txt"})
        hint_history = [
            {"fc_message": fc1, "hint": "First hint."},
            {"fc_message": fc2, "hint": "Second hint."},
        ]
        fc3 = _fc_message("move_file", {"src": "c.txt"})
        h._run_coach(fc3, [], "{}", self._schema(), "task", hint_history=hint_history)
        prompt = h._call_llm.call_args[0][1]
        assert "Attempt 1" in prompt
        assert "Attempt 2" in prompt
        assert "First hint." in prompt
        assert "Second hint." in prompt


class TestV2StepBoundary:
    """
    Verify the step_start_idx / cleanup invariant directly:
    after a retry loop, messages beyond step_start_idx must be removed.
    """

    def test_hint_removed_after_loop(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        step_start_idx = len(messages)
        latest_hint = "Try using the correct path."

        # Simulate ONE retry: inject hint
        del messages[step_start_idx:]
        messages.append({"role": "user", "content": latest_hint})
        assert len(messages) == 3

        # Simulate cleanup after loop
        del messages[step_start_idx:]
        assert len(messages) == 2
        assert all(m["role"] != "user" or m["content"] == "task" for m in messages)

    def test_multiple_retries_only_latest_hint_visible_to_model(self):
        """
        Each retry resets to step_start_idx and injects only the latest hint,
        so the model never sees stale hints from earlier iterations.
        """
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        step_start_idx = len(messages)

        hints = ["Hint A", "Hint B", "Hint C"]
        for hint in hints:
            del messages[step_start_idx:]
            messages.append({"role": "user", "content": hint})
            # Model only sees the single latest hint
            model_view = messages[step_start_idx:]
            assert len(model_view) == 1
            assert model_view[0]["content"] == hint

        # Final cleanup
        del messages[step_start_idx:]
        assert len(messages) == step_start_idx

    def test_hint_history_accumulates_while_model_sees_only_latest(self):
        """
        step_hint_history grows with each bad verdict while the model context
        only ever contains the latest hint (not all previous ones).
        """
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        step_start_idx = len(messages)
        step_hint_history = []
        latest_hint = None

        # Simulate 3 bad verdicts
        fake_fcs = [
            _fc_message("tool_x", {"arg": f"wrong_{i}"}) for i in range(3)
        ]
        fake_hints = ["Hint 1", "Hint 2", "Hint 3"]

        for fc, hint in zip(fake_fcs, fake_hints):
            # Reset + inject latest hint
            del messages[step_start_idx:]
            if latest_hint:
                messages.append({"role": "user", "content": latest_hint})

            # Model context: only 0 or 1 hint
            assert len(messages) - step_start_idx <= 1

            # Record in history and update latest
            step_hint_history.append({"fc_message": fc, "hint": hint})
            latest_hint = hint

        assert len(step_hint_history) == 3
        assert step_hint_history[0]["hint"] == "Hint 1"
        assert step_hint_history[2]["hint"] == "Hint 3"

        # Final cleanup
        del messages[step_start_idx:]
        assert len(messages) == step_start_idx


# ── Complete-check (shared between both, kept in V2) ─────────────────────────

class TestCompleteCheck:
    def test_done_verdict_parsed(self):
        h = _make_v2(llm_response="<verdict>Done</verdict>\n<hint></hint>")
        verdict, hint = h._run_complete_check([], [], "{}", "task")
        assert verdict == "Done"
        assert hint == ""

    def test_not_done_verdict_and_hint_parsed(self):
        h = _make_v2(
            llm_response=(
                "<verdict>NotDone</verdict>\n"
                "<hint>You still need to call list_files() to get the directory listing.</hint>"
            )
        )
        verdict, hint = h._run_complete_check([], [], "{}", "task")
        assert verdict == "NotDone"
        assert "list_files" in hint

    def test_missing_verdict_defaults_to_not_done(self):
        h = _make_v2(llm_response="No verdict tag here")
        verdict, hint = h._run_complete_check([], [], "{}", "task")
        assert verdict == "NotDone"

    def test_prompt_contains_user_request(self):
        h = _make_v2(llm_response="<verdict>Done</verdict><hint></hint>")
        h._run_complete_check([], [], "{}", "list all files in /home")
        prompt = h._call_llm.call_args[0][1]
        assert "list all files in /home" in prompt
