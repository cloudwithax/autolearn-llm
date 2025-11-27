"""
FP8 GRPO Training for Code Generation

Train code models with execution-based reward signals:
- Unit test pass rate (primary signal)
- Successful execution
- Lint/style compliance
- Type safety
- Performance optimization

Targets: HumanEval, MBPP, SWE-Bench, Terminal Bench

Usage:
    python train_code_grpo.py --config code_config.yaml
    python train_code_grpo.py --model unsloth/Qwen3-1.7B --dataset bigcode/humanevalpack
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Enable memory-efficient RL
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import yaml
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from code_rewards import (
    combined_code_reward,
    syntax_reward,
    execution_reward,
    test_pass_reward,
    lint_reward,
    extract_code_blocks,
)


@dataclass
class CodeProblem:
    """A coding problem with test cases."""
    prompt: str
    entry_point: str  # Function name to implement
    test_code: str  # Unit tests
    canonical_solution: str = ""  # Reference solution (optional)
    

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_code_prompt(problem: Dict[str, Any], style: str = "instruct") -> str:
    """
    Create prompt for code generation.
    
    Styles:
    - instruct: Clear instruction format
    - completion: Docstring completion style (HumanEval)
    - chat: Conversational style
    """
    if style == "completion":
        # HumanEval style: just the docstring
        return problem.get("prompt", "")
    
    elif style == "chat":
        return f"""You are an expert Python programmer. Write clean, efficient code.

**Task:** {problem.get('prompt', '')}

Write your solution in a Python code block. The function should be named `{problem.get('entry_point', 'solution')}`.
"""
    
    else:  # instruct
        prompt = problem.get("prompt", "")
        entry_point = problem.get("entry_point", "solution")
        
        return f"""Implement the following Python function:

{prompt}

Requirements:
- Function name: `{entry_point}`
- Write clean, efficient code
- Include type hints
- Handle edge cases

```python
def {entry_point}"""


def prepare_humaneval_dataset(max_samples: int = None) -> Dataset:
    """Load and prepare HumanEval dataset."""
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process(example):
        return {
            "prompt": create_code_prompt(example, style="instruct"),
            "entry_point": example["entry_point"],
            "test_code": example["test"],
            "canonical_solution": example.get("canonical_solution", ""),
        }
    
    return dataset.map(process, remove_columns=dataset.column_names)


def prepare_mbpp_dataset(max_samples: int = None) -> Dataset:
    """Load and prepare MBPP dataset."""
    print("Loading MBPP dataset...")
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process(example):
        # MBPP has different structure
        prompt = example["prompt"]
        code = example["code"]
        
        # Extract function name from code
        match = re.search(r'def\s+(\w+)\s*\(', code)
        entry_point = match.group(1) if match else "solution"
        
        # Convert test_list to executable test code
        test_list = example.get("test_list", [])
        test_code = "\n".join([
            f"_test({test.strip()}, '{test[:30]}...')" 
            for test in test_list
        ])
        
        return {
            "prompt": create_code_prompt({
                "prompt": prompt,
                "entry_point": entry_point,
            }, style="instruct"),
            "entry_point": entry_point,
            "test_code": test_code,
            "canonical_solution": code,
        }
    
    return dataset.map(process, remove_columns=dataset.column_names)


def prepare_code_dataset(
    dataset_name: str, 
    max_samples: int = None
) -> Dataset:
    """Load coding dataset by name."""
    
    if "humaneval" in dataset_name.lower():
        return prepare_humaneval_dataset(max_samples)
    elif "mbpp" in dataset_name.lower():
        return prepare_mbpp_dataset(max_samples)
    else:
        # Generic: assume it has prompt/test columns
        print(f"Loading generic dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="test")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return dataset


class CodeRewardComputer:
    """
    Computes rewards for code completions with test execution.
    
    Caches test results for efficiency during GRPO training.
    """
    
    __name__ = "code_reward"  # Required by TRL 0.25+
    
    def __init__(
        self, 
        dataset: Dataset,
        weights: Dict[str, float] = None,
        use_execution: bool = True,
        use_lint: bool = True,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.weights = weights or {
            "test_pass": 0.5,    # Primary: tests pass
            "execution": 0.2,    # Secondary: runs without crash
            "syntax": 0.15,      # Tertiary: valid Python
            "lint": 0.1,         # Style quality
            "complexity": 0.05,  # Elegance
        }
        self.use_execution = use_execution
        self.use_lint = use_lint
        self.verbose = verbose
        
        # Build prompt -> test_code lookup
        self.test_lookup = {}
        for item in dataset:
            self.test_lookup[item["prompt"]] = item.get("test_code", "")
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """Compute rewards for a batch of completions."""
        
        # Get test cases for each prompt
        test_cases = []
        for prompt in prompts:
            test_cases.append(self.test_lookup.get(prompt, ""))
        
        # Compute combined reward
        rewards = combined_code_reward(
            prompts=prompts,
            completions=completions,
            test_cases=test_cases,
            weights=self.weights,
        )
        
        if self.verbose:
            for i, (comp, reward) in enumerate(zip(completions, rewards)):
                code = extract_code_blocks(comp)
                preview = code[0][:100] if code else comp[:100]
                print(f"  [{i}] reward={reward:.3f} | {preview}...")
        
        return rewards


def main():
    parser = argparse.ArgumentParser(description="FP8 GRPO Code Training")
    parser.add_argument("--config", type=str, default="code_config.yaml")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="humaneval")
    parser.add_argument("--max_samples", type=int, default=164)  # Full HumanEval
    parser.add_argument("--output_dir", type=str, default="./outputs/code")
    parser.add_argument("--no_execution", action="store_true", 
                        help="Disable code execution (faster, less signal)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Model settings
    model_name = args.model or config.get("model", {}).get("name", "unsloth/Qwen3-1.7B")
    max_seq_length = config.get("model", {}).get("max_seq_length", 2048)
    lora_rank = config.get("lora", {}).get("rank", 32)
    load_in_fp8 = config.get("model", {}).get("load_in_fp8", True)
    
    # Training settings
    train_cfg = config.get("training", {})
    grpo_cfg = config.get("grpo", {})
    
    print("=" * 60)
    print("FP8 GRPO Code Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"FP8 Enabled: {load_in_fp8}")
    print(f"Code Execution: {not args.no_execution}")
    print("=" * 60)
    
    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Load model
    print("\nLoading model with FP8...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        load_in_fp8=load_in_fp8,
    )
    
    # Add LoRA
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=config.get("lora", {}).get("alpha", 32),
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load dataset
    dataset = prepare_code_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Dataset: {len(dataset)} problems")
    
    # Create reward computer
    reward_weights = grpo_cfg.get("reward_weights", {
        "test_pass": 0.5,
        "execution": 0.2,
        "syntax": 0.15,
        "lint": 0.1,
        "complexity": 0.05,
    })
    
    reward_fn = CodeRewardComputer(
        dataset=dataset,
        weights=reward_weights,
        use_execution=not args.no_execution,
        verbose=args.verbose,
    )
    
    # Training config (TRL 0.25+ API)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        
        # Generation (TRL 0.25 naming)
        num_generations=train_cfg.get("num_generations", 8),
        max_completion_length=train_cfg.get("max_new_tokens", 512),
        max_prompt_length=max_seq_length // 2,
        
        # Training
        learning_rate=train_cfg.get("learning_rate", 2e-6),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        
        # Optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_8bit",
        
        # GRPO
        beta=grpo_cfg.get("beta", 0.04),
        
        # Logging
        logging_steps=10,
        save_steps=50,
        
        # Misc
        seed=42,
        bf16=True,
        report_to="none",
    )
    
    # Trainer
    print("\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    print("\nStarting code GRPO training...")
    print("Rewards: test_pass + execution + syntax + lint + complexity")
    trainer.train()
    
    # Save
    print(f"\nSaving to {args.output_dir}")
    model.save_pretrained(f"{args.output_dir}/lora")
    tokenizer.save_pretrained(f"{args.output_dir}/lora")
    
    model.save_pretrained_merged(
        f"{args.output_dir}/merged",
        tokenizer,
        save_method="merged_16bit",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA: {args.output_dir}/lora")
    print(f"Merged: {args.output_dir}/merged")
    print("=" * 60)


if __name__ == "__main__":
    main()
