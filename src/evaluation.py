# src/evaluation.py

"""
Model evaluation utilities for HumanEval-style benchmarks.

This module handles:
- Batched generation via vLLM (Unsloth wrapper)
- Proper stopping logic for CoT vs non-CoT
- Application of the cleaning pipeline
- Emission of JSONL artifacts for downstream pass@k evaluation
"""

import gc
from typing import Dict, List

import torch
from vllm import SamplingParams
from human_eval.data import write_jsonl
from unsloth import FastLanguageModel

from src.prompts import (
    build_cot_prompt,
    build_non_cot_prompt,
)
from src.cleaning import cleaner


# ---------------------------------------------------------------------
# Evaluation Driver
# ---------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    problems: Dict,
    task_ids: List[str],
    *,
    use_cot: bool,
    output_jsonl: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    min_p: float = 0.10,
    load_in_4bit: bool = True,
    gpu_memory_utilization: float = 0.7,
) -> None:
    """
    Generate HumanEval completions for a model and save results to JSONL.

    Parameters
    ----------
    model_path : str
        Path or HF identifier of the model.
    problems : dict
        Loaded HumanEval problems dict.
    task_ids : list[str]
        Subset of task IDs to evaluate.
    use_cot : bool
        Whether to use CoT prompting + extraction.
    output_jsonl : str
        Output path for JSONL samples.
    max_new_tokens : int
        Maximum tokens to generate per prompt.
    temperature : float
        Sampling temperature.
    """

    print(f"\nLoading: {model_path} | Mode: {'CoT' if use_cot else 'Non-CoT'}")

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    FastLanguageModel.for_inference(model)

    # ------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------
    prompts: List[str] = []
    for task_id in task_ids:
        problem = problems[task_id]
        if use_cot:
            prompt = build_cot_prompt(problem, tokenizer)
        else:
            prompt = build_non_cot_prompt(problem)
        prompts.append(prompt)

    # ------------------------------------------------------------
    # Stop logic
    # ------------------------------------------------------------
    if use_cot:
        stop_tokens = ["</SOLUTION>"]
    else:
        stop_tokens = ["\nclass", "\ndef ", "\nif __name__", "\nprint"]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        max_tokens=max_new_tokens,
        stop=stop_tokens,
    )

    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------
    print("Running vLLM batch generation...")
    outputs = model.fast_generate(prompts, sampling_params=sampling_params)

    # ------------------------------------------------------------
    # Post-process
    # ------------------------------------------------------------
    samples = []
    for i, task_id in enumerate(task_ids):
        problem = problems[task_id]
        raw_completion = outputs[i].outputs[0].text

        completion = cleaner(
            raw_completion,
            entry_point=problem["entry_point"],
            is_cot=use_cot,
        )

        samples.append({
            "task_id": task_id,
            "prompt": problem["prompt"],
            "completion": completion,
        })

    write_jsonl(output_jsonl, samples)
    print(f"Saved {len(samples)} samples → {output_jsonl}")

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()