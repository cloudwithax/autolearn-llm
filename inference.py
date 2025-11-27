"""
Inference Script for FP8 GRPO Trained Models

Test your trained model with FP8 vLLM inference.
"""

import os
import argparse
from typing import List

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import torch
from unsloth import FastLanguageModel


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load trained FP8 model for inference."""
    print(f"Loading model from {model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        load_in_fp8=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate completions for prompts using vLLM."""
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Use vLLM for fast inference
    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling_params,
    )
    
    return [output.outputs[0].text for output in outputs]


def interactive_mode(model, tokenizer):
    """Interactive chat with the model."""
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
            
            # Format as math problem
            prompt = f"""Solve this math problem step by step.

Problem: {user_input}

Think through this carefully, showing your work. End with "The answer is [number]"."""
            
            response = generate(model, tokenizer, [prompt], max_new_tokens=512)[0]
            print(f"\nModel: {response}\n")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def benchmark(model, tokenizer):
    """Quick benchmark with sample problems."""
    test_problems = [
        "If a train travels 120 miles in 2 hours, how fast is it going?",
        "A store sells apples for $2 each. If I buy 5 apples and pay with a $20 bill, how much change do I get?",
        "What is 15% of 80?",
        "If x + 5 = 12, what is x?",
    ]
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    
    for problem in test_problems:
        prompt = f"""Solve this math problem step by step.

Problem: {problem}

Think through this carefully, showing your work. End with "The answer is [number]"."""
        
        print(f"\nProblem: {problem}")
        response = generate(model, tokenizer, [prompt], max_new_tokens=256)[0]
        print(f"Response: {response[:500]}...")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="FP8 Model Inference")
    parser.add_argument("--model", type=str, default="./outputs/merged", 
                        help="Path to trained model")
    parser.add_argument("--mode", type=str, choices=["interactive", "benchmark"], 
                        default="interactive")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.max_seq_length)
    
    if args.mode == "interactive":
        interactive_mode(model, tokenizer)
    else:
        benchmark(model, tokenizer)


if __name__ == "__main__":
    main()
