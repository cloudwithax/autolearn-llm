"""
Code Evaluation Script

Evaluate trained models on coding benchmarks:
- HumanEval (pass@1, pass@10)
- MBPP
- Custom test suites

Runs actual code execution for ground truth evaluation.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import torch
from datasets import load_dataset
from tqdm import tqdm

from code_rewards import (
    execute_code_safe,
    extract_code_blocks,
    syntax_check,
    run_with_tests,
)


def load_model(model_path: str, max_seq_length: int = 4096):
    """Load model for evaluation."""
    from unsloth import FastLanguageModel
    
    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        load_in_fp8=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_solutions(
    model,
    tokenizer,
    prompts: List[str],
    n_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> List[List[str]]:
    """Generate n_samples solutions per prompt."""
    from vllm import SamplingParams
    
    all_solutions = []
    
    for prompt in tqdm(prompts, desc="Generating"):
        sampling_params = SamplingParams(
            n=n_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        
        outputs = model.fast_generate([prompt], sampling_params=sampling_params)
        solutions = [out.text for out in outputs[0].outputs]
        all_solutions.append(solutions)
    
    return all_solutions


def evaluate_humaneval(
    model,
    tokenizer,
    n_samples: int = 1,
    max_problems: int = None,
) -> Dict[str, float]:
    """
    Evaluate on HumanEval benchmark.
    
    Returns pass@k metrics.
    """
    print("\n" + "=" * 60)
    print("HumanEval Evaluation")
    print("=" * 60)
    
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
    
    results = {
        "total": len(dataset),
        "passed": 0,
        "failed": 0,
        "error": 0,
        "details": [],
    }
    
    for item in tqdm(dataset, desc="Evaluating"):
        task_id = item["task_id"]
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        
        # Generate solution
        solutions = generate_solutions(
            model, tokenizer, [prompt], 
            n_samples=n_samples,
            max_new_tokens=512,
            temperature=0.2 if n_samples == 1 else 0.8,
        )[0]
        
        # Check each solution
        any_passed = False
        for solution in solutions:
            code_blocks = extract_code_blocks(solution)
            if not code_blocks:
                # Try using raw completion
                code = prompt + solution
            else:
                code = prompt + code_blocks[0]
            
            # Run with tests
            try:
                full_code = code + "\n\n" + test_code
                result = execute_code_safe(full_code, timeout=10.0)
                
                if result.success:
                    any_passed = True
                    break
            except Exception as e:
                continue
        
        if any_passed:
            results["passed"] += 1
            status = "PASS"
        else:
            results["failed"] += 1
            status = "FAIL"
        
        results["details"].append({
            "task_id": task_id,
            "status": status,
        })
        
        if len(results["details"]) % 20 == 0:
            current_pass = results["passed"] / len(results["details"])
            print(f"  Progress: {len(results['details'])}/{len(dataset)} | pass@{n_samples}: {current_pass:.1%}")
    
    # Compute metrics
    results["pass@1"] = results["passed"] / results["total"]
    
    print(f"\nResults:")
    print(f"  pass@{n_samples}: {results['pass@1']:.1%} ({results['passed']}/{results['total']})")
    
    return results


def evaluate_mbpp(
    model,
    tokenizer,
    n_samples: int = 1,
    max_problems: int = None,
) -> Dict[str, float]:
    """Evaluate on MBPP benchmark."""
    print("\n" + "=" * 60)
    print("MBPP Evaluation")
    print("=" * 60)
    
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
    
    results = {
        "total": len(dataset),
        "passed": 0,
        "failed": 0,
        "details": [],
    }
    
    for item in tqdm(dataset, desc="Evaluating"):
        task_id = item["task_id"]
        description = item["prompt"]
        test_list = item["test_list"]
        
        # Create prompt
        prompt = f"""Write a Python function to solve the following problem:

{description}

```python
"""
        
        solutions = generate_solutions(
            model, tokenizer, [prompt],
            n_samples=n_samples,
            max_new_tokens=512,
        )[0]
        
        any_passed = False
        for solution in solutions:
            code_blocks = extract_code_blocks(solution)
            code = code_blocks[0] if code_blocks else solution
            
            # Run all tests
            try:
                full_code = code + "\n\n" + "\n".join(test_list)
                result = execute_code_safe(full_code, timeout=10.0)
                
                if result.success:
                    any_passed = True
                    break
            except:
                continue
        
        if any_passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        results["details"].append({
            "task_id": task_id,
            "status": "PASS" if any_passed else "FAIL",
        })
    
    results["pass@1"] = results["passed"] / results["total"]
    
    print(f"\nResults:")
    print(f"  pass@{n_samples}: {results['pass@1']:.1%} ({results['passed']}/{results['total']})")
    
    return results


def run_custom_tests(
    model,
    tokenizer,
    test_file: str,
) -> Dict[str, Any]:
    """Run custom test suite from JSON file."""
    print(f"\nRunning custom tests from: {test_file}")
    
    with open(test_file, 'r') as f:
        tests = json.load(f)
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    for test in tqdm(tests, desc="Testing"):
        prompt = test["prompt"]
        expected_tests = test.get("tests", "")
        
        solutions = generate_solutions(
            model, tokenizer, [prompt],
            n_samples=1,
            max_new_tokens=512,
        )[0]
        
        code = extract_code_blocks(solutions[0])
        code = code[0] if code else solutions[0]
        
        passed, total, output = run_with_tests(code, expected_tests)
        
        if passed == total and total > 0:
            results["passed"] += 1
            status = "PASS"
        else:
            results["failed"] += 1
            status = "FAIL"
        
        results["details"].append({
            "prompt": prompt[:50],
            "status": status,
            "passed": passed,
            "total": total,
        })
    
    results["pass_rate"] = results["passed"] / len(tests) if tests else 0
    
    print(f"\nCustom Tests: {results['pass_rate']:.1%} ({results['passed']}/{len(tests)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Model")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--benchmark", type=str, default="humaneval",
                        choices=["humaneval", "mbpp", "custom", "all"])
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--test_file", type=str, help="Custom test file (JSON)")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model)
    
    all_results = {}
    
    if args.benchmark in ["humaneval", "all"]:
        all_results["humaneval"] = evaluate_humaneval(
            model, tokenizer,
            n_samples=args.n_samples,
            max_problems=args.max_problems,
        )
    
    if args.benchmark in ["mbpp", "all"]:
        all_results["mbpp"] = evaluate_mbpp(
            model, tokenizer,
            n_samples=args.n_samples,
            max_problems=args.max_problems,
        )
    
    if args.benchmark == "custom" and args.test_file:
        all_results["custom"] = run_custom_tests(
            model, tokenizer, args.test_file
        )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for bench, res in all_results.items():
        if "pass@1" in res:
            print(f"  {bench}: {res['pass@1']:.1%}")
        elif "pass_rate" in res:
            print(f"  {bench}: {res['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
