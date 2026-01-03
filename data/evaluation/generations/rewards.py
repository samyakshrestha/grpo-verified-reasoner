# src/rewards.py

import re
from typing import List, Dict, Any

from .schema import validate_schema, extract_solution
from .testing import run_mbpp_tests


# ---------------------------------------------------------------------------
# Format Reward
# ---------------------------------------------------------------------------

def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Rewards strict adherence to the XML schema.
    """
    rewards = []
    for completion in completions:
        is_valid, _ = validate_schema(completion)
        rewards.append(0.1 if is_valid else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reasoning Reward
# ---------------------------------------------------------------------------

_REASONING_RE = re.compile(
    r"<START_WORKING_OUT>(.*?)</END_WORKING_OUT>",
    re.IGNORECASE | re.DOTALL,
)

def reasoning_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Soft reward for producing a non-trivial reasoning block.
    Length-based, capped to avoid verbosity abuse.
    """
    rewards = []
    for completion in completions:
        match = _REASONING_RE.search(completion)
        if not match:
            rewards.append(0.0)
            continue

        reasoning = match.group(1).strip()
        length = len(reasoning)

        # Soft cap: max 0.2 reward at ~500 chars
        score = min(0.2, (length / 500.0) * 0.2)
        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# Correctness Reward
# ---------------------------------------------------------------------------

def correctness_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[Dict[str, Any]],
    **kwargs,
) -> List[float]:
    """
    Rewards functional correctness by executing unit tests.
    """
    rewards = []

    for completion, task in zip(completions, answer):
        code, status = extract_solution(completion)
        if not code:
            rewards.append(0.0)
            continue

        passed, err = run_mbpp_tests(code, task)

        if passed:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            # Failures are intentionally printed for traceability
            print(f"\n[FAIL] Task: {task.get('task_id', 'Unknown')}")
            print(f"Error: {err}")

    return rewards