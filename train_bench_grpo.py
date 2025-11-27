"""
GRPO Training on SWE-Bench and Terminal-Bench

Train models to:
- SWE-Bench: Fix real GitHub issues by generating patches
- Terminal-Bench: Complete terminal tasks by generating commands

Usage:
    python train_bench_grpo.py --benchmark swe --config bench_config.yaml
    python train_bench_grpo.py --benchmark terminal --model unsloth/Qwen3-1.7B
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

import yaml
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from bench_rewards import (
    swe_combined_reward,
    terminal_combined_reward,
    extract_patch,
    extract_commands,
)


def load_config(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# =============================================================================
# SWE-BENCH DATASET
# =============================================================================

def create_swe_prompt(item: Dict[str, Any], style: str = "instruct") -> str:
    """
    Create prompt for SWE-Bench issue resolution.
    
    Styles:
    - instruct: Clear instruction format
    - minimal: Just the issue
    """
    problem = item.get("problem_statement", "")
    repo = item.get("repo", "")
    
    if style == "minimal":
        return f"""Fix this issue in {repo}:

{problem}

Provide your fix as a unified diff patch."""
    
    else:  # instruct
        hints = item.get("hints_text", "")
        hints_section = f"\n\n**Hints from discussion:**\n{hints[:500]}" if hints else ""
        
        return f"""You are a software engineer fixing a bug in the {repo} repository.

## Issue Description
{problem}
{hints_section}

## Instructions
1. Analyze the issue carefully
2. Identify the root cause
3. Generate a patch to fix the issue

Provide your fix as a unified diff patch in the following format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```
"""


def prepare_swe_dataset(
    variant: str = "Lite",
    max_samples: int = None,
    prompt_style: str = "instruct",
) -> Dataset:
    """
    Load and prepare SWE-Bench dataset.
    
    Variants:
    - Lite: 300 instances (easier, faster)
    - Verified: 500 instances (human-verified)
    - Full: 2294 instances (complete)
    """
    print(f"Loading SWE-bench {variant}...")
    
    if variant.lower() == "lite":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    elif variant.lower() == "verified":
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    else:
        dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process(item):
        return {
            "prompt": create_swe_prompt(item, style=prompt_style),
            "instance_id": item["instance_id"],
            "gold_patch": item["patch"],
            "repo": item["repo"],
            "fail_to_pass": item.get("FAIL_TO_PASS", ""),
        }
    
    return dataset.map(process, remove_columns=dataset.column_names)


# =============================================================================
# TERMINAL-BENCH DATASET
# =============================================================================

def create_terminal_prompt(task: Dict[str, Any]) -> str:
    """Create prompt for Terminal-Bench task."""
    instruction = task.get("instruction", task.get("task", ""))
    
    return f"""You are an expert at using the Linux terminal. Complete this task:

## Task
{instruction}

## Instructions
- Provide the exact command(s) needed
- Use standard Unix/Linux tools
- Be precise and efficient

```bash
# Your command(s) here
```
"""


def prepare_terminal_dataset(
    version: str = "core",
    max_samples: int = None,
) -> Dataset:
    """
    Load Terminal-Bench tasks.
    
    Note: Terminal-Bench uses a different distribution method.
    This loads from HuggingFace if available, or creates synthetic tasks.
    """
    print(f"Loading Terminal-Bench {version}...")
    
    try:
        # Try loading from HuggingFace
        dataset = load_dataset("laude-institute/terminal-bench-core", split="test")
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        print("Creating synthetic terminal tasks for training...")
        
        # Synthetic terminal tasks for training
        tasks = [
            {"instruction": "List all files in the current directory, including hidden files", "expected": "ls -la"},
            {"instruction": "Find all Python files in the current directory and subdirectories", "expected": "find . -name '*.py'"},
            {"instruction": "Count the number of lines in all .txt files", "expected": "wc -l *.txt"},
            {"instruction": "Search for the word 'error' in all log files", "expected": "grep 'error' *.log"},
            {"instruction": "Create a directory called 'backup' and copy all .py files into it", "expected": "mkdir backup && cp *.py backup/"},
            {"instruction": "Show the last 20 lines of the file 'app.log'", "expected": "tail -n 20 app.log"},
            {"instruction": "Find all files larger than 100MB", "expected": "find . -size +100M"},
            {"instruction": "Replace all occurrences of 'foo' with 'bar' in config.txt", "expected": "sed -i 's/foo/bar/g' config.txt"},
            {"instruction": "Show disk usage of the current directory", "expected": "du -sh ."},
            {"instruction": "List all running Python processes", "expected": "ps aux | grep python"},
            {"instruction": "Create a tar archive of the 'src' directory", "expected": "tar -cvf src.tar src/"},
            {"instruction": "Show the current git branch", "expected": "git branch --show-current"},
            {"instruction": "Find and delete all .pyc files", "expected": "find . -name '*.pyc' -delete"},
            {"instruction": "Show environment variables containing 'PATH'", "expected": "env | grep PATH"},
            {"instruction": "Download a file from https://example.com/file.txt", "expected": "wget https://example.com/file.txt"},
            {"instruction": "Check if port 8080 is in use", "expected": "lsof -i :8080"},
            {"instruction": "Show the first 5 lines of each .md file", "expected": "head -n 5 *.md"},
            {"instruction": "Sort a file alphabetically and remove duplicates", "expected": "sort -u file.txt"},
            {"instruction": "Show the current system time in UTC", "expected": "date -u"},
            {"instruction": "Create a Python virtual environment called 'venv'", "expected": "python -m venv venv"},
            {"instruction": "Install requirements from requirements.txt", "expected": "pip install -r requirements.txt"},
            {"instruction": "Run pytest with verbose output", "expected": "pytest -v"},
            {"instruction": "Show git log for the last 5 commits", "expected": "git log -n 5 --oneline"},
            {"instruction": "Find files modified in the last 24 hours", "expected": "find . -mtime -1"},
            {"instruction": "Compress all .log files with gzip", "expected": "gzip *.log"},
        ]
        
        dataset = Dataset.from_list(tasks)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process(item):
        return {
            "prompt": create_terminal_prompt(item),
            "instruction": item.get("instruction", item.get("task", "")),
            "expected": item.get("expected", item.get("oracle", "")),
        }
    
    return dataset.map(process)


# =============================================================================
# REWARD COMPUTERS
# =============================================================================

class SWERewardComputer:
    """Compute rewards for SWE-Bench tasks."""
    
    __name__ = "swe_reward"  # Required by TRL 0.25+
    
    def __init__(self, dataset: Dataset, weights: Dict[str, float] = None):
        self.weights = weights or {
            "format": 0.3,
            "similarity": 0.5,
            "files": 0.2,
        }
        # Build lookup for gold patches
        self.patch_lookup = {}
        for item in dataset:
            self.patch_lookup[item["prompt"]] = item.get("gold_patch", "")
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        gold_patches = [self.patch_lookup.get(p, "") for p in prompts]
        return swe_combined_reward(
            prompts=prompts,
            completions=completions,
            gold_patches=gold_patches,
            weights=self.weights,
        )


class TerminalRewardComputer:
    """Compute rewards for Terminal-Bench tasks."""
    
    __name__ = "terminal_reward"  # Required by TRL 0.25+
    
    def __init__(self, dataset: Dataset, weights: Dict[str, float] = None):
        self.weights = weights or {
            "format": 0.2,
            "safety": 0.3,
            "execution": 0.5,
        }
        # Build lookup for expected outputs
        self.expected_lookup = {}
        for item in dataset:
            self.expected_lookup[item["prompt"]] = item.get("expected", "")
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        expected = [self.expected_lookup.get(p, "") for p in prompts]
        return terminal_combined_reward(
            prompts=prompts,
            completions=completions,
            expected_outputs=expected,
            weights=self.weights,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Training on Benchmarks")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["swe", "swe-lite", "swe-verified", "terminal"],
                        help="Benchmark to train on")
    parser.add_argument("--config", type=str, default="bench_config.yaml")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Settings
    model_name = args.model or config.get("model", {}).get("name")
    max_seq_length = config.get("model", {}).get("max_seq_length", 2048)
    lora_rank = config.get("lora", {}).get("rank", 16)
    
    train_cfg = config.get("training", {})
    
    # Determine output dir
    output_dir = args.output_dir or f"./outputs/{args.benchmark}"
    
    print("=" * 60)
    print(f"GRPO Training: {args.benchmark.upper()}")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output: {output_dir}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Load model with 4-bit quantization
    print("\nLoading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=config.get("lora", {}).get("alpha", 16),
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    
    # Load dataset
    print(f"\nLoading {args.benchmark} dataset...")
    
    if args.benchmark.startswith("swe"):
        variant = "Lite" if "lite" in args.benchmark else ("Verified" if "verified" in args.benchmark else "Full")
        dataset = prepare_swe_dataset(variant=variant, max_samples=args.max_samples)
        reward_fn = SWERewardComputer(dataset)
    else:  # terminal
        dataset = prepare_terminal_dataset(max_samples=args.max_samples)
        reward_fn = TerminalRewardComputer(dataset)
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Training config
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Generation
        num_generations=train_cfg.get("num_generations", 4),
        max_completion_length=train_cfg.get("max_completion_length", 512),
        max_prompt_length=max_seq_length // 2,
        
        # Training
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        
        # Optimization
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        
        # GRPO
        beta=config.get("grpo", {}).get("beta", 0.04),
        
        # Logging
        logging_steps=10,
        save_steps=50,
        
        seed=42,
        report_to="none",
    )
    
    # Trainer
    print("\nInitializing trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    print(f"\nStarting {args.benchmark.upper()} GRPO training...")
    trainer.train()
    
    # Save
    print(f"\nSaving to {output_dir}")
    model.save_pretrained(f"{output_dir}/lora")
    tokenizer.save_pretrained(f"{output_dir}/lora")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapter: {output_dir}/lora")
    print("=" * 60)


if __name__ == "__main__":
    main()
