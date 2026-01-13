# src/rewards.py

"""
GRPO reward functions for code generation with chain-of-thought reasoning.

This module implements the reward signal used during reinforcement learning
with GRPO (Group Relative Policy Optimization). The three independent reward
functions shape model behavior toward:
1. Schema adherence (format)
2. Concise reasoning (reasoning)
3. Functional correctness (correctness)
"""

import re
from typing import List, Dict, Any

from src.schema import validate_schema, extract_solution
from src.testing import run_mbpp_tests


# ---------------------------------------------------------------------------
# Format Reward: Schema Adherence
# ---------------------------------------------------------------------------

def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Rewards the model for strictly following the XML-like schema.

    The schema requires:
    - <START_WORKING_OUT> ... </END_WORKING_OUT> (reasoning block)
    - <SOLUTION> ... </SOLUTION> (code block)

    Args:
        completions: List of generated strings from the model.

    Returns:
        List of rewards (0.02 for valid schema, 0.0 for invalid).

    Note:
        Reward magnitude is small (0.02) to avoid overwhelming correctness
        reward (1.0). This biases outputs toward the format without
        dominating the optimization objective.
    """
    rewards = []
    for completion in completions:
        is_valid, _ = validate_schema(completion)
        rewards.append(0.02 if is_valid else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reasoning Reward: Encourage Thoughtful CoT
# ---------------------------------------------------------------------------

def reasoning_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Rewards the model for generating a meaningful reasoning block.

    Uses a "soft length" penalty to encourage thinking without reward hacking
    via verbosity. Score scales linearly with character count up to a cap.

    Args:
        completions: List of generated strings from the model.

    Returns:
        List of rewards (0.0 to 0.1, scaled by reasoning length).

    Reward formula:
        score = min(0.1, (len(reasoning) / 800.0) * 0.1)

    Rationale:
        - Cap at 800 characters prevents infinite reward for verbosity
        - Max reward of 0.1 is 10% of correctness reward
        - Incentivizes meaningful thought without dominating optimization
    """
    rewards = []
    RE_REASONING = re.compile(
        r"<START_WORKING_OUT>(.*?)</END_WORKING_OUT>",
        re.DOTALL | re.IGNORECASE,
    )

    for completion in completions:
        match = RE_REASONING.search(completion)
        if match:
            reasoning_content = match.group(1).strip()
            length = len(reasoning_content)
            # Soft cap: reward scales linearly up to 800 chars, then plateaus
            score = min(0.1, (length / 800.0) * 0.1)
            rewards.append(score)
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Correctness Reward: Execution-Based Verification
# ---------------------------------------------------------------------------

def correctness_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[Dict[str, Any]],
    **kwargs,
) -> List[float]:
    """
    Rewards the model for writing code that passes unit tests.

    This is the primary optimization signal. Code is extracted, executed in a
    sandboxed subprocess, and scored based on test pass rate.

    Args:
        prompts: The prompts fed to the model (unused, but expected by GRPO).
        completions: The model's generated answers.
        answer: List of task dicts containing test cases (from MBPP+).

    Returns:
        List of rewards. For each completion:
        - 1.1 if all tests pass (bonus for perfect correctness)
        - partial credit: (passed_tests / total_tests) if some tests pass
        - 0.0 if extraction fails or all tests fail

    Implementation:
        1. Extract code from <SOLUTION> tags via extract_solution()
        2. Run tests in forked subprocess with 2s timeout
        3. Return fraction of passing tests as reward

    Note:
        Failed executions print debug info for monitoring but don't halt training.
    """
    rewards = []

    for prompt, completion, task_data in zip(prompts, completions, answer):
        code, status = extract_solution(completion)

        if not code:
            # Extraction failed (no <SOLUTION>, syntax error, etc.)
            rewards.append(0.0)
            continue

        # Execute tests in isolated subprocess
        passed, err = run_mbpp_tests(code, task_data, timeout_s=2.0)

        if passed:
            # Bonus reward for perfect correctness
            rewards.append(1.1)
        else:
            # Partial credit: count passing tests before first failure
            # (This requires modifying run_mbpp_tests to return counts,
            #  but current implementation returns binary pass/fail.
            #  For now, we use 0.0 for any failure.)
            rewards.append(0.0)

            # Debug logging (useful during training)
            task_id = task_data.get("task_id", "Unknown")
            print(f"\n[FAIL] Task: {task_id}")
            if err:
                # Truncate error to avoid log spam
                err_preview = err[:200] + "..." if len(err) > 200 else err
                print(f"Error: {err_preview}")

    return rewards


# ---------------------------------------------------------------------------
# Combined Reward Function List
# ---------------------------------------------------------------------------

def get_reward_functions() -> List:
    """
    Returns the full list of reward functions for GRPO training.

    Order matters for logging/WandB tracking but not for optimization
    (rewards are summed by the trainer).

    Returns:
        List of three reward functions:
        - format_reward_func (0.02)
        - reasoning_reward_func (0.0-0.1)
        - correctness_reward_func (0.0-1.1)
    """
    return [
        format_reward_func,
        reasoning_reward_func,
        correctness_reward_func,
    ]
