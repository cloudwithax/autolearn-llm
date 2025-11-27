"""
FP8 GRPO Training Script

Train LLMs with Group Relative Policy Optimization using FP8 precision.
Optimized for consumer GPUs (5-24GB VRAM).

Key features:
- 60% less VRAM via FP8 frozen weights
- 1.4x faster inference via vLLM FP8
- 96% of training time is inference (well-optimized)
- Memory sharing between vLLM and training

Usage:
    python train_fp8_grpo.py --config config.yaml
    python train_fp8_grpo.py --model unsloth/Qwen3-1.7B --dataset openai/gsm8k
"""

import os
import re
import argparse
from typing import List, Dict, Any

# Enable memory-efficient RL (must be before unsloth import!)
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import yaml
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from rewards import combined_reward, correctness_reward, format_reward


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(dataset_name: str, split: str = "train", max_samples: int = None):
    """
    Load and prepare dataset for GRPO training.
    
    Returns dataset with 'prompt' and 'answer' columns.
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Handle different dataset formats
    if "gsm8k" in dataset_name.lower():
        # GSM8K format: question -> prompt, answer -> extract number
        def process_gsm8k(example):
            question = example["question"]
            answer_text = example["answer"]
            
            # Extract final answer (after ####)
            match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
            answer = match.group(1).replace(',', '') if match else ""
            
            prompt = f"""Solve this math problem step by step.

Problem: {question}

Think through this carefully, showing your work. End with "The answer is [number]"."""
            
            return {"prompt": prompt, "answer": answer}
        
        dataset = dataset.map(process_gsm8k, remove_columns=dataset.column_names)
        
    elif "question" in dataset.column_names and "answer" in dataset.column_names:
        # Generic Q&A format
        def process_qa(example):
            prompt = f"Question: {example['question']}\n\nAnswer:"
            return {"prompt": prompt, "answer": str(example["answer"])}
        
        dataset = dataset.map(process_qa, remove_columns=dataset.column_names)
        
    else:
        raise ValueError(f"Unsupported dataset format. Columns: {dataset.column_names}")
    
    print(f"Dataset prepared: {len(dataset)} examples")
    return dataset


def create_reward_function(config: Dict[str, Any]):
    """Create reward function from config."""
    weights = config.get('grpo', {}).get('reward_weights', {
        'correctness': 1.0,
        'format': 0.5,
        'reasoning': 0.3,
    })
    
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Extract answers from the dataset (passed via kwargs or stored)
        answers = kwargs.get('answers', [''] * len(prompts))
        return combined_reward(prompts, completions, answers, weights=weights)
    
    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="FP8 GRPO Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument("--max_samples", type=int, help="Override max samples")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Override with command line args
    model_name = args.model or config.get('model', {}).get('name', 'unsloth/Qwen3-1.7B')
    dataset_name = args.dataset or config.get('dataset', {}).get('name', 'openai/gsm8k')
    max_samples = args.max_samples or config.get('dataset', {}).get('max_samples', 1000)
    output_dir = args.output_dir or config.get('output', {}).get('dir', './outputs')
    
    # Model config
    max_seq_length = config.get('model', {}).get('max_seq_length', 2048)
    lora_rank = config.get('lora', {}).get('rank', 32)
    load_in_fp8 = config.get('model', {}).get('load_in_fp8', True)
    
    # Training config
    train_cfg = config.get('training', {})
    grpo_cfg = config.get('grpo', {})
    
    print("=" * 60)
    print("FP8 GRPO Training")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"FP8 Enabled: {load_in_fp8}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print("WARNING: No GPU detected. FP8 training requires CUDA.")
    
    # Load model with FP8
    print("\nLoading model with FP8 quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Must be False for FP8 RL
        fast_inference=True,  # Enable vLLM
        max_lora_rank=lora_rank,
        load_in_fp8=load_in_fp8,
    )
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    lora_config = config.get('lora', {})
    target_modules = lora_config.get('target_modules', [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.0),
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Prepare dataset
    dataset = prepare_dataset(dataset_name, max_samples=max_samples)
    
    # Store answers for reward computation
    answers = dataset["answer"]
    
    # GRPO Training Configuration (TRL 0.25+ API)
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Generation settings (TRL 0.25 naming)
        num_generations=train_cfg.get('num_generations', 4),
        max_completion_length=train_cfg.get('max_new_tokens', 512),
        max_prompt_length=max_seq_length // 2,
        
        # Training settings
        learning_rate=train_cfg.get('learning_rate', 5e-6),
        num_train_epochs=train_cfg.get('num_train_epochs', 1),
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 4),
        
        # Optimization
        warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        optim="adamw_8bit",
        
        # GRPO specific
        beta=grpo_cfg.get('beta', 0.04),
        
        # Logging
        logging_steps=config.get('output', {}).get('logging_steps', 10),
        save_steps=config.get('output', {}).get('save_steps', 100),
        
        # Misc
        seed=42,
        bf16=True,
        report_to="none",  # Set to "wandb" for W&B logging
    )
    
    # Create reward function that includes answers
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # For GRPO, we need to match prompts to their answers
        # This is a simplified version - in practice you'd track indices
        batch_answers = []
        for prompt in prompts:
            # Find matching answer (simplified lookup)
            for i, ds_prompt in enumerate(dataset["prompt"]):
                if prompt == ds_prompt:
                    batch_answers.append(answers[i])
                    break
            else:
                batch_answers.append("")
        
        return combined_reward(
            prompts, completions, batch_answers,
            weights=grpo_cfg.get('reward_weights', {})
        )
    
    # Initialize trainer
    print("\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    print("\nStarting FP8 GRPO training...")
    print("(96% of time will be vLLM inference, 4% training)")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}/final")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    # Optionally save merged model (for inference without LoRA)
    print(f"Saving merged FP8 model to {output_dir}/merged")
    model.save_pretrained_merged(
        f"{output_dir}/merged",
        tokenizer,
        save_method="merged_16bit",  # or "merged_4bit_forced"
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter: {output_dir}/final")
    print(f"Merged model: {output_dir}/merged")
    print("=" * 60)


if __name__ == "__main__":
    main()
