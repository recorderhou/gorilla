#!/usr/bin/env python3
"""
Collect multi-sample training trajectories for SFT data augmentation.

Each training case is run k times with different random seeds and temperature=0.7,
producing k diverse trajectories per case instead of a single greedy one.

Trial isolation: each trial uses a unique ID suffix (e.g. multi_turn_base_0_trial_2)
so the globals-based instance cache in multi_turn_utils.py creates a fresh environment
for every trial automatically — no changes to multi_turn_utils.py needed.

Usage:
    python collect_multi_sample.py --handler v1fix --k 8 --verify
    python collect_multi_sample.py --handler v1fix --k 8
    python collect_multi_sample.py --handler v2fix --k 8
"""

import argparse
import copy
import json
import os
import queue
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Training case indices (per domain, 4 domains × 40 = 160 total)
# ---------------------------------------------------------------------------
TRAINING_INDICES = set(
    list(range(0, 40))       # FileSystem
    + list(range(50, 90))    # VehicleControl
    + list(range(100, 140))  # TradingBot
    + list(range(150, 190))  # TravelAPI
)

BFCL_DIR = Path(__file__).parent
DATA_FILE = BFCL_DIR / "bfcl_eval" / "data" / "BFCL_v4_multi_turn_base.json"
RESULT_BASE = BFCL_DIR / "result"


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------
def _import_handlers():
    sys.path.insert(0, str(BFCL_DIR))
    from bfcl_eval.model_handler.local_inference.qwen_fc_v1_coach import QwenFCV1CoachHandler
    from bfcl_eval.model_handler.local_inference.qwen_fc_v2_coach import QwenFCV2CoachHandler
    return {"v1fix": QwenFCV1CoachHandler, "v2fix": QwenFCV2CoachHandler}


def build_handler(handler_key, registry_name, temperature):
    handlers = _import_handlers()
    cls = handlers[handler_key]
    handler = cls(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        temperature=temperature,
        registry_name=registry_name,
        is_fc_model=True,
    )
    return handler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_cases(verify=False):
    # Use BFCL's own loader so entries get the "function" field injected from
    # involved_classes (raw JSON has no "function" key). Reading the raw file
    # directly would crash in _pre_query_processing_prompting with KeyError.
    from bfcl_eval.utils import load_dataset_entry

    all_entries = load_dataset_entry("multi_turn_base")
    cases = []
    for entry in all_entries:
        idx = int(entry["id"].rsplit("_", 1)[-1])
        if verify:
            if idx in (0, 1):
                cases.append(entry)
        elif idx in TRAINING_INDICES:
            cases.append(entry)
    return cases


# ---------------------------------------------------------------------------
# Result writing (bypasses handler.write to avoid ID routing issue with
# trial-suffixed IDs; writes directly to the correct JSONL file)
# ---------------------------------------------------------------------------
def make_json_serializable(value):
    from bfcl_eval.utils import make_json_serializable as _mjs
    return _mjs(value)


def get_result_file(registry_name):
    result_dir = RESULT_BASE / registry_name / "multi_turn"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / "BFCL_v4_multi_turn_base_result.json"


def writer_loop(write_queue, result_file):
    # Open in "w": truncate once at the start of THIS run so re-running the same
    # registry name overwrites cleanly instead of accumulating duplicate trials.
    with open(result_file, "w") as f:
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()  # account for the sentinel so wq.join() returns
                break
            try:
                f.write(json.dumps(make_json_serializable(item)) + "\n")
                f.flush()
            finally:
                write_queue.task_done()


# ---------------------------------------------------------------------------
# Single-trial inference
# ---------------------------------------------------------------------------
def run_trial(handler, orig_case, trial_idx):
    trial_case = copy.deepcopy(orig_case)
    trial_case["id"] = f"{orig_case['id']}_trial_{trial_idx}"
    handler.seed = trial_idx

    try:
        result, metadata = handler.inference(
            trial_case,
            include_input_log=False,
            exclude_state_log=False,
        )
    except Exception as e:
        result = f"Error: {e}"
        metadata = {"traceback": traceback.format_exc()}

    return {
        "id": trial_case["id"],
        "result": result,
        **metadata,
    }


# ---------------------------------------------------------------------------
# Per-case runner (k trials, serial)
# ---------------------------------------------------------------------------
def run_case(handler, case, k, write_queue, step_stats):
    results = []
    for t in range(k):
        entry = run_trial(handler, case, t)
        write_queue.put(entry)
        results.append(entry)
        # track parse failures: a step is failed if its result item is a plain string
        result_val = entry["result"]
        if isinstance(result_val, list):
            for turn in result_val:
                if isinstance(turn, list):
                    for step in turn:
                        with step_stats["lock"]:
                            step_stats["total"] += 1
                            if not step or step == "":
                                step_stats["failed"] += 1
    return results


# ---------------------------------------------------------------------------
# Verify diagnostics
# ---------------------------------------------------------------------------
def _leaf_steps(result):
    """Flatten a (possibly nested) result into leaf step strings, any depth."""
    if isinstance(result, list):
        for item in result:
            yield from _leaf_steps(item)
    else:
        yield result


def print_verify_report(all_results, max_budget=0):
    print("\n" + "=" * 70)
    print("VERIFY REPORT")
    print("=" * 70)

    # Group by original case ID (strip _trial_N)
    by_case = {}
    for entry in all_results:
        orig_id = "_".join(entry["id"].split("_")[:-2])  # strip _trial_N
        by_case.setdefault(orig_id, []).append(entry)

    parse_total = 0
    parse_failed = 0
    coach_totals = {"checks": 0, "interv": 0, "trials": 0}
    coach_max_budget = [max_budget]
    coach_over_budget = [False]

    for case_id in sorted(by_case):
        trials = sorted(by_case[case_id], key=lambda e: int(e["id"].rsplit("_", 1)[-1]))
        print(f"\n[{case_id}] first-step tool_call + coach usage per trial:")
        first_calls = []
        for entry in trials:
            t_idx = entry["id"].rsplit("_", 1)[-1]
            result = entry.get("result", [])
            first_step = None
            if isinstance(result, list) and result:
                turn0 = result[0]
                if isinstance(turn0, list) and turn0:
                    first_step = turn0[0]
                    if isinstance(first_step, list) and first_step:
                        first_step = first_step[0]
            first_calls.append(first_step)

            # coach usage (present for v1; absent for handlers that don't track it)
            checks = entry.get("coach_checks")
            interv = entry.get("coach_interventions")
            if checks is not None:
                over = coach_max_budget[0] and interv > coach_max_budget[0]
                coach_str = f"  [coach {interv}/{checks} intervene/check]" + (" ⚠️OVER" if over else "")
                coach_totals["checks"] += checks
                coach_totals["interv"] += interv
                coach_totals["trials"] += 1
                if over:
                    coach_over_budget[0] = True
            else:
                coach_str = "  [coach n/a]"
            print(f"  trial_{t_idx}: {first_step}{coach_str}")

            # parse failure stats: a leaf step counts as failed if it is empty
            # or blank (any nesting depth).
            for step in _leaf_steps(result):
                parse_total += 1
                if not (isinstance(step, str) and step.strip()):
                    parse_failed += 1

        unique = len(set(str(c) for c in first_calls))
        if unique == 1:
            print(f"  ⚠️  All {len(first_calls)} trials produced IDENTICAL first step — check seed/temperature!")
        else:
            print(f"  ✓  {unique}/{len(first_calls)} distinct first steps (diversity confirmed)")

    print(f"\nParse failure rate: {parse_failed}/{parse_total} "
          f"({100*parse_failed/max(parse_total,1):.1f}%)")

    if coach_totals["trials"]:
        n = coach_totals["trials"]
        c = coach_totals["checks"]
        i = coach_totals["interv"]
        print(f"Coach usage (over {n} trials): "
              f"{c} checks, {i} interventions "
              f"(avg {c/n:.1f} checks / {i/n:.1f} interventions per trial); "
              f"intervene rate {100*i/max(c,1):.0f}% of checks")
        max_budget = coach_max_budget[0]
        if max_budget:
            over = coach_over_budget[0]
            status = "⚠️ some trials EXCEEDED budget!" if over else "✓ all within budget"
            print(f"Per-task intervention budget = {max_budget}: {status}")
    else:
        print("Coach usage: n/a (handler does not report coach_checks/coach_interventions)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect multi-sample coach trajectories")
    parser.add_argument("--handler", choices=["v1fix", "v2fix"], required=True)
    parser.add_argument("--k", type=int, default=8, help="trials per case")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--verify", action="store_true", help="2 cases × k trials with diagnostics")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--local-model-path", type=str, default=None,
                        help="Local dir with the student model's config/tokenizer "
                             "(needed to load the tokenizer; vLLM must already be running).")
    args = parser.parse_args()

    suffix = "verify" if args.verify else "multisample"
    registry_name = f"qwen2.5-3b-{args.handler}-{suffix}-FC"
    result_file = get_result_file(registry_name) if not args.output_dir else (
        Path(args.output_dir) / "multi_turn" / "BFCL_v4_multi_turn_base_result.json"
    )

    # Ensure hint_log goes to correct place
    # Own the hint_log path directly. setdefault would be defeated by an empty
    # HINT_LOG_PATH exported upstream (empty string is falsy -> logging disabled),
    # so assign unconditionally.
    hint_log_path = str(result_file.parent.parent / "hint_log.jsonl")
    os.environ["HINT_LOG_PATH"] = hint_log_path

    print(f"Handler     : {args.handler}")
    print(f"Registry    : {registry_name}")
    print(f"k (trials)  : {args.k}")
    print(f"temperature : {args.temperature}  top_p: {args.top_p}")
    print(f"Result file : {result_file}")
    print(f"Hint log    : {hint_log_path}")

    cases = load_training_cases(verify=args.verify)
    print(f"Cases loaded: {len(cases)} {'(verify mode)' if args.verify else ''}")

    # Build handler and set sampling params
    handler = build_handler(args.handler, registry_name, temperature=args.temperature)
    handler.top_p = args.top_p

    # Load the tokenizer / max_context_length / model_path_or_id via the normal
    # OSS setup path. skip_server_setup=True: vLLM is already running (started by
    # the shell wrapper); this only loads the tokenizer and does a quick health
    # check against the existing server — it does NOT start a new one.
    print("Loading tokenizer via spin_up_local_server(skip_server_setup=True)…")
    handler.spin_up_local_server(
        num_gpus=1,
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=args.local_model_path,
    )

    # Writer thread
    result_file.parent.mkdir(parents=True, exist_ok=True)
    wq = queue.Queue()
    wt = threading.Thread(target=writer_loop, args=(wq, result_file), daemon=True)
    wt.start()

    step_stats = {"total": 0, "failed": 0, "lock": threading.Lock()}
    all_results = []
    all_results_lock = threading.Lock()

    num_threads = min(args.num_threads, len(cases))
    total_trials = len(cases) * args.k
    done = [0]
    done_lock = threading.Lock()

    def case_task(case):
        results = run_case(handler, case, args.k, wq, step_stats)
        with all_results_lock:
            all_results.extend(results)
        with done_lock:
            done[0] += args.k
            print(f"  [{done[0]}/{total_trials}] completed trials for {case['id']}", flush=True)
        return results

    print(f"\nStarting collection ({num_threads} threads × {len(cases)} cases × {args.k} trials)…\n")
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {ex.submit(case_task, c): c for c in cases}
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"ERROR in case {futures[f]['id']}: {exc}")

    # Flush writer
    wq.put(None)
    wq.join()
    wt.join()

    print(f"\nDone. {len(all_results)} result entries written to {result_file}")

    if args.verify:
        max_budget = getattr(handler, "max_coach_instructions", 0)
        print_verify_report(all_results, max_budget=max_budget)
    else:
        with step_stats["lock"]:
            t, f_ = step_stats["total"], step_stats["failed"]
        print(f"Parse failure rate: {f_}/{t} ({100*f_/max(t,1):.1f}%)")


if __name__ == "__main__":
    main()
