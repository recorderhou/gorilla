#!/usr/bin/env python3
"""
Deterministic backward-compat check for base_oss_handler._query_prompting.

The kwargs-based refactor must produce API calls IDENTICAL to the legacy
two-branch implementation when top_p/seed are unset (the standard
`bfcl generate` path), and must inject top_p/seed only when they are set
(the multi-sample path).

This intercepts the outgoing `completions.create(**kwargs)` and asserts the
exact kwargs, so it does NOT depend on vLLM determinism, a running server, or
model weights. Exit code 0 = compatible, 1 = regression.
"""
import sys
import types
from pathlib import Path

BFCL_DIR = Path(__file__).parent
sys.path.insert(0, str(BFCL_DIR))

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler


# ── Fake OpenAI client that records the kwargs of the last create() call ──────
class _CaptureCompletions:
    def __init__(self):
        self.captured = None

    def create(self, **kwargs):
        self.captured = kwargs
        return types.SimpleNamespace(choices=[])


class _CaptureClient:
    def __init__(self):
        self.completions = _CaptureCompletions()


def _make_handler(**extra_attrs):
    """Build a bare OSSHandler with only the attributes _query_prompting reads."""
    h = OSSHandler.__new__(OSSHandler)  # skip heavy __init__
    h.client = _CaptureClient()
    h.model_path_or_id = "test/model"
    h.temperature = 0.001
    h.max_context_length = 4096
    h.tokenizer = types.SimpleNamespace(tokenize=lambda s: s.split())
    h._format_prompt = lambda message, function: "hello world prompt"
    for k, v in extra_attrs.items():
        setattr(h, k, v)
    return h


def _call(h):
    OSSHandler._query_prompting(h, {"function": [], "message": []})
    return h.client.completions.captured


# ── Expected legacy kwargs ────────────────────────────────────────────────────
# "hello world prompt".split() -> 3 tokens; leftover = min(4096, 4096-3-2) = 4091
_LEFTOVER = 4091
_LEGACY_ELSE = {
    "model": "test/model",
    "temperature": 0.001,
    "prompt": "hello world prompt",
    "max_tokens": _LEFTOVER,
    "timeout": 72000,
}


def _check(name, got, expected):
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"        expected: {expected}")
        print(f"        got:      {got}")
    return ok


def main():
    print("Backward-compat check: base_oss_handler._query_prompting")
    all_ok = True

    # 1. Standard bfcl generate path: no extra_body, no top_p, no seed.
    #    Must equal the legacy `else` branch EXACTLY (no top_p/seed/extra_body keys).
    got = _call(_make_handler())
    all_ok &= _check("no extra_body / no top_p / no seed == legacy else-branch",
                     got, _LEGACY_ELSE)

    # 2. extra_body path (stop_token_ids present) == legacy `if extra_body` branch.
    got = _call(_make_handler(stop_token_ids=[151645]))
    expected = {**_LEGACY_ELSE, "extra_body": {"stop_token_ids": [151645]}}
    all_ok &= _check("stop_token_ids -> extra_body == legacy if-branch", got, expected)

    # 3. Multi-sample path: top_p + seed injected.
    got = _call(_make_handler(top_p=0.95, seed=3))
    expected = {**_LEGACY_ELSE, "top_p": 0.95, "seed": 3}
    all_ok &= _check("top_p + seed set -> injected", got, expected)

    # 4. top_p set but seed unset -> only top_p injected (getattr(None) skipped).
    got = _call(_make_handler(top_p=0.9))
    expected = {**_LEGACY_ELSE, "top_p": 0.9}
    all_ok &= _check("top_p only -> seed skipped", got, expected)

    # 5. seed=0 must still be injected (0 is not None).
    got = _call(_make_handler(seed=0))
    expected = {**_LEGACY_ELSE, "seed": 0}
    all_ok &= _check("seed=0 -> injected (falsy but not None)", got, expected)

    print("=" * 60)
    if all_ok:
        print("RESULT: backward-compatible ✓")
        return 0
    print("RESULT: REGRESSION DETECTED ✗")
    return 1


if __name__ == "__main__":
    sys.exit(main())
