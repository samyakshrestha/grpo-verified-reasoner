# src/prompts.py

"""
Prompt construction utilities for HumanEval evaluation.

This module defines the canonical prompting logic used for:
- Non-CoT (plain HumanEval continuation)
- CoT (ChatML-style reasoning + solution schema)

These functions are shared across Base, SFT, and GRPO models.
They must remain schema-stable.
"""

from typing import Dict, List


# ---------------------------------------------------------------------
# Stop strings (safety against rambling in Non-CoT mode)
# ---------------------------------------------------------------------

STOP_STRINGS: List[str] = [
    "\nclass",
    "\ndef ",
    "\nif __name__",
    "\nprint",
]


# ---------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------

def build_non_cot_prompt(problem: Dict) -> str:
    """
    Non-CoT prompt builder.

    For HumanEval-style continuation, we return ONLY the raw problem prompt.
    The evaluation harness will prepend this prompt again during execution.

    Parameters
    ----------
    problem : dict
        A HumanEval problem entry containing a 'prompt' field.

    Returns
    -------
    str
        The raw HumanEval prompt.
    """
    return problem["prompt"]


def build_cot_prompt(problem: Dict, tokenizer) -> str:
    """
    CoT prompt builder.

    Uses the same ChatML-style template employed during SFT / GRPO training.
    This ensures distributional alignment between training and evaluation.

    Parameters
    ----------
    problem : dict
        A HumanEval problem entry containing a 'prompt' field.
    tokenizer : PreTrainedTokenizer
        Tokenizer with an `apply_chat_template` method.

    Returns
    -------
    str
        A fully formatted chat prompt string ready for generation.
    """
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT_HUMANEVAL},
        {"role": "user", "content": problem["prompt"]},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )