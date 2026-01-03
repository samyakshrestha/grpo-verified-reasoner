# src/testing.py

import multiprocessing as mp
import traceback
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Test extraction
# ---------------------------------------------------------------------------

def get_tests_from_task(task: Dict[str, Any]) -> List[str]:
    """
    Extracts MBPP / EvalPlus test assertions from a task dict.
    Supports multiple known key formats.
    """
    # Case 1: list-based test fields
    for key in ("test_list", "tests", "plus_tests", "base_tests"):
        if key in task and task[key]:
            return list(task[key])

    # Case 2: single multiline assertion string
    if "assertion" in task and task["assertion"]:
        lines = task["assertion"].strip().splitlines()
        return [line for line in lines if line.strip()]

    raise KeyError(f"No tests found in task keys: {list(task.keys())}")

# ---------------------------------------------------------------------------
# Worker process (isolated execution)
# ---------------------------------------------------------------------------

def _exec_code_and_tests_worker(
    code: str,
    tests: List[str],
    queue: mp.Queue,
) -> None:
    """
    Executes model-generated code and its associated tests
    inside a clean execution environment.
    """
    try:
        # Shared execution environment
        env = {"__builtins__": __builtins__}

        # Load model code
        exec(code, env, env)

        # Run each test
        for test in tests:
            try:
                exec(test, env, env)
            except AssertionError:
                queue.put((False, f"Failed assertion: {test}"))
                return

        # All tests passed
        queue.put((True, None))

    except Exception:
        queue.put((False, traceback.format_exc()))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mbpp_tests(
    code: str,
    task: Dict[str, Any],
    timeout_s: float = 2.0,
) -> Tuple[bool, str | None]:
    """
    Executes MBPP tests for a given task in a subprocess with timeout.

    Returns:
        (passed, error_message)
    """
    tests = get_tests_from_task(task)

    ctx = mp.get_context("fork")  # Linux / Colab safe and fast
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_exec_code_and_tests_worker,
        args=(code, tests, queue),
    )

    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, f"Timeout after {timeout_s:.1f}s"

    if queue.empty():
        return False, "No result returned from worker process"

    return queue.get()