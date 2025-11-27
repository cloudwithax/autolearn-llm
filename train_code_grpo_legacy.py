"""
GRPO Training for Legacy GPUs (GTX 10xx, 16xx, RTX 20xx)

No FP8 support - uses 4-bit quantization instead.
Optimized for 6GB VRAM (GTX 1060, 1660, etc.)

Key differences from FP8 version:
- 4-bit quantization (bitsandbytes)
- No vLLM (uses HF generate)
- Smaller models (0.5B-1B)
- Aggressive memory optimization

Usage:
    python train_code_grpo_legacy.py --config code_config_1060.yaml
"""

import os
import gc
import re
import argparse
from typing import List, Dict, Any

import yaml
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from code_rewards import (
    combined_code_reward,
    extract_code_blocks,
)


def load_config(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def clear_memory():
    """Aggressive memory cleanup for limited VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_prompt(problem: Dict[str, Any]) -> str:
    """Create code prompt."""
    prompt = problem.get("prompt", "")
    entry_point = problem.get("entry_point", "solution")
    
    return f"""Write a Python function to solve this problem:

{prompt}

Function name: `{entry_point}`

```python
def {entry_point}"""


def prepare_dataset(dataset_name: str, max_samples: int = None) -> Dataset:
    """Load coding dataset."""
    if "humaneval" in dataset_name.lower():
        dataset = load_dataset("openai/openai_humaneval", split="test")
    elif "mbpp" in dataset_name.lower():
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    else:
        dataset = load_dataset(dataset_name, split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process(ex):
        if "humaneval" in dataset_name.lower():
            return {
                "prompt": create_prompt(ex),
                "entry_point": ex["entry_point"],
                "test_code": ex["test"],
            }
        else:
            # MBPP format
            match = re.search(r'def\s+(\w+)\s*\(', ex.get("code", ""))
            entry = match.group(1) if match else "solution"
            tests = "\n".join(ex.get("test_list", []))
            return {
                "prompt": create_prompt({"prompt": ex["prompt"], "entry_point": entry}),
                "entry_point": entry,
                "test_code": tests,
            }
    
    return dataset.map(process, remove_columns=dataset.column_names)


class LegacyCodeReward:
    """Reward function for legacy GPU training."""
    
    __name__ = "code_reward"  # Required by TRL 0.25+
    
    def __init__(self, dataset: Dataset, weights: Dict[str, float] = None):
        self.weights = weights or {
            "test_pass": 0.50,
            "execution": 0.25,
            "syntax": 0.15,
            "lint": 0.10,
            "complexity": 0.0,
        }
        self.test_lookup = {item["prompt"]: item.get("test_code", "") for item in dataset}
    
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        test_cases = [self.test_lookup.get(p, "") for p in prompts]
        return combined_code_reward(
            prompts=prompts,
            completions=completions,
            test_cases=test_cases,
            weights=self.weights,
        )


def main():
    parser = argparse.ArgumentParser(description="GRPO Code Training (Legacy GPU)")
    parser.add_argument("--config", type=str, default="code_config_1060.yaml")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Settings
    model_name = args.model or config.get("model", {}).get("name", "unsloth/Qwen2.5-Coder-0.5B-Instruct")
    dataset_name = args.dataset or config.get("dataset", {}).get("name", "humaneval")
    max_samples = args.max_samples or config.get("dataset", {}).get("max_samples", 50)
    output_dir = args.output_dir or config.get("output", {}).get("dir", "./outputs/code_1060")
    
    max_seq_length = config.get("model", {}).get("max_seq_length", 1024)
    lora_rank = config.get("lora", {}).get("rank", 16)
    
    train_cfg = config.get("training", {})
    grpo_cfg = config.get("grpo", {})
    
    print("=" * 60)
    print("GRPO Code Training (Legacy GPU Mode)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Quantization: 4-bit (no FP8)")
    print(f"Max seq length: {max_seq_length}")
    print(f"LoRA rank: {lora_rank}")
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.1f}GB)")
    print("=" * 60)
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Extra memory savings
    )
    
    print("\nLoading model with 4-bit quantization...")
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
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_cfg = config.get("lora", {})
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("Adding LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    
    clear_memory()
    
    # Dataset
    print(f"\nLoading dataset: {dataset_name}")
    dataset = prepare_dataset(dataset_name, max_samples=max_samples)
    print(f"Samples: {len(dataset)}")
    
    # Reward function
    reward_fn = LegacyCodeReward(
        dataset=dataset,
        weights=grpo_cfg.get("reward_weights"),
    )
    
    # Training args (TRL 0.25+ API)
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Generation (TRL 0.25 naming)
        num_generations=train_cfg.get("num_generations", 2),
        max_completion_length=train_cfg.get("max_new_tokens", 256),
        max_prompt_length=max_seq_length // 2,
        
        # Training
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        
        # Memory optimization
        fp16=True,  # Use FP16 instead of BF16 for older GPUs
        bf16=False,
        optim="adamw_8bit",
        
        # GRPO
        beta=grpo_cfg.get("beta", 0.04),
        
        # Logging
        logging_steps=config.get("output", {}).get("logging_steps", 5),
        save_steps=config.get("output", {}).get("save_steps", 25),
        
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
    print("\nStarting training...")
    print("(This will be slower than FP8 on newer GPUs)")
    trainer.train()
    
    # Save
    print(f"\nSaving to {output_dir}")
    model.save_pretrained(f"{output_dir}/lora")
    tokenizer.save_pretrained(f"{output_dir}/lora")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapter saved: {output_dir}/lora")
    print("=" * 60)


if __name__ == "__main__":
    main()
