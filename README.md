# Post-Training Qwen3-4B-Base for Code Generation using GRPO (RLVR)

An end-to-end **RLVR (Reinforcement Learning with Verifiable Rewards)** pipeline that transforms **Qwen3-4B-Base** into a reasoning model using Chain-of-Thought distillation, **Unsloth-optimized LoRA**, and **GRPO**, achieving **79.9% Pass@1** on HumanEval (+27.5% over base).

## Results

**79.9% pass@1 on HumanEval** with explicit reasoning (CoT). A 2.5% absolute improvement over SFT baseline and 27.5% over the base model.

| Model | Pass@1 | Reasoning | Δ from Base |
|-------|--------|-----------|-------------|
| Base (Qwen3-4B-Base) | 52.4% | No | - |
| SFT (100 distilled examples) | 77.4% | CoT | +25.0% |
| GRPO (378 MBPP+ tasks) | 79.9% | CoT | +27.5% |

> **Critical Discovery**: Merging LoRA adapters in float16 caused small GRPO updates (magnitude ~0.002) to underflow and vanish. Merging in float32 then downcasting preserved all RL refinements and enabled the final 2.5% gain.

## Overview

We train a 4B parameter code model through three stages:

1. **Distillation**: Teacher-student pipeline via deepseek-reasoner API (DeepSeek V3.2 reasoning component, 100 examples, strict schema validation)
2. **SFT**: Supervised fine-tuning on distilled data (3 epochs, LoRA rank 32 via Unsloth)
3. **GRPO**: Reinforcement learning on 378 MBPP+ coding tasks with reward shaping

Chain-of-thought reasoning is enforced via a strict XML schema: model outputs reasoning between `<START_WORKING_OUT>...</START_WORKING_OUT>` tags and code between `<SOLUTION>...</SOLUTION>` tags. This forces explicit problem decomposition and enables fine-grained reward signals.

## Pipeline Details

### Stage 1: Distillation

Generate 100 training examples using deepseek-reasoner (DeepSeek V3.2 reasoning component) as a teacher. Each example is validated against the required schema and filtered for correctness. This creates a high-quality warmup dataset.

### Stage 2: SFT (Supervised Fine-Tuning)

Fine-tune the Qwen3-4B-Base model using LoRA (rank 32, alpha 16, target modules: q_proj, k_proj, v_proj, o_proj) with Unsloth for memory efficiency and speed. Configuration:
- Batch size: 2, Gradient accumulation: 1
- Learning rate: 2e-4, Warmup ratio: 0.1, LR scheduler: cosine
- Epochs: 3
- Response tokens only (prompt tokens masked)

Output: SFT checkpoint achieving 77.4% pass@1.

### Stage 3: GRPO (Reinforcement Learning)

Train on 378 MBPP+ coding tasks using the TRL GRPOTrainer. Configuration:
- Batch size: 4, Gradient accumulation: 1
- Learning rate: 5e-6, Weight decay: 0.1, Optimizer: adamw_8bit
- vLLM sampling: temperature=0.9, top_p=0.95, num_generations=16
- Epochs: 2, Total steps: 188

Reward function combines three signals:
- **Format reward** (0.1): Schema adherence (tags present and ordered correctly)
- **Reasoning reward** (0.0-0.2): Reasoning block length incentive (capped at 500 chars to prevent verbosity)
- **Correctness reward** (1.0): Partial credit based on MBPP+ test pass rate

The correctness reward runs submitted code in a sandboxed subprocess with 2-second timeout, executing unit tests and returning the fraction of passing tests as the reward signal.

## Technical Notes

### Schema Validation

The model learns to follow a precise output format via reward shaping and schema validation. Validation checks tag presence and order (case-insensitive regex):
- `<START_WORKING_OUT>` before reasoning
- `</END_WORKING_OUT>` after reasoning
- `<SOLUTION>` before code
- `</SOLUTION>` after code

### Code Execution Safety

Test execution is isolated in a forked subprocess with a strict 2-second timeout. Code is executed in a clean environment, assertions are run sequentially, and any exception or timeout is treated as a test failure.

## Usage

### Inference

```python
from unsloth import FastLanguageModel
import torch

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="samyakshrestha/qwen3-4b-grpo",
    max_seq_length=3072,
    load_in_4bit=False,  # Full precision for stability
    fast_inference=True
)

FastLanguageModel.for_inference(model)

system_prompt = """You are a code-generation engine.
You must output your response in the following exact format:
<START_WORKING_OUT>
Concise reasoning steps required to solve the problem.
</END_WORKING_OUT>
<SOLUTION>
Valid Python code only.
</SOLUTION>
Do not output anything outside these tags."""

prompt = "Write a function to compute the factorial of n."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8
    )

input_len = inputs["input_ids"].shape[1]
response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
print(response)
```

## Model Availability

- **HuggingFace Hub**: [samyakshrestha/qwen3-4b-grpo](https://huggingface.co/samyakshrestha/qwen3-4b-grpo) (FP16)
- **GGUF format**: [samyakshrestha/qwen3-4b-grpo-GGUF](https://huggingface.co/samyakshrestha/qwen3-4b-grpo-GGUF) (quantized for CPU/edge inference)
- **LoRA adapter**: Available for further fine-tuning or inference with base model

## Training Infrastructure

Trained on Google Colab H100 80GB GPUs with **Unsloth** optimization:
- **Training time**: ~1.5 hours for 188 steps (2 epochs on 378 MBPP+ tasks)
- **Speed**: 2x faster fine-tuning via optimized CUDA kernels and smart gradient offloading
- **Memory**: Saved VRAM through optimized gradient checkpointing, enabling 16 generations/step for GRPO
- **vLLM integration**: Batched generation of 16 responses per problem during RL rollouts

**Training logs**: [WandB Run](https://wandb.ai/samyakshrestha-university-of-texas-at-dallas/mbpp-rl-project/runs/x0n9lpib)

## Evaluation

Models are evaluated on the HumanEval benchmark (164 coding problems) in both CoT and Non-CoT modes. 

**vLLM Evaluation Performance: 164 problems in <2 minutes** (vs. 1+ hour with sequential generation). Batched inference processes all test cases concurrently, achieving **~30x speedup** through vLLM's PagedAttention algorithm for efficient KV cache management, continuous batching for dynamic request scheduling, and optimized CUDA kernels for attention computation.

## Tech Stack

**Core Framework**
- PyTorch 2.0+, Transformers 4.56.2
- Unsloth (LoRA optimization and vLLM integration)
- TRL 0.22.2 (GRPOTrainer)
- vLLM 0.10.2 (batched inference)

**Training & Evaluation**
- EvalPlus (MBPP+ dataset and test execution)
- WandB (experiment tracking)
- deepseek-reasoner API (DeepSeek V3.2 reasoning component for distillation)

**Requirements**
- Python 3.10+
- CUDA-capable GPU (H100 / A100 recommended for training, T4+ for inference)
- 40GB+ VRAM for training, 16GB+ for inference

## File Structure

```
notebooks/
  01_distillation_dataset.ipynb     # Teacher generation and validation
  02_qwen3_4B_sft.ipynb             # SFT training on distilled data
  03_GRPO_optimization.ipynb        # GRPO training pipeline
  04_evaluation.ipynb               # HumanEval pass@1 evaluation
  05_GRPO_optimization_2.ipynb      # Main GRPO training (run 2)
  06_evaluation_2.ipynb             # Evaluation of run 2
  07_evaluation_2_pass@10.ipynb     # Pass@10 metrics
  08_merge_lora_adapters.ipynb      # Precision-aware model merging
  09_compare_model_outputs.ipynb    # Output comparison and analysis
  10_GGUF_conversion.ipynb          # GGUF quantization for edge deployment

src/
  evaluation.py                     # Evaluation utilities
  prompts.py                        # System and user prompts
  schema.py                         # Schema validation helpers
  cleaning.py                       # Data cleaning pipelines
  testing.py                        # Test execution utilities
```

## Author

**Samyak Shrestha**