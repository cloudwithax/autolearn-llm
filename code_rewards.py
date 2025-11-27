"""
Execution-Based Code Rewards for GRPO Training

Novel reward signals using actual code execution feedback:
- Syntax validity (AST parsing)
- Execution success (runs without errors)
- Test passing (unit test results)
- Linting score (ruff/pylint)
- Performance delta (execution time comparison)
- Type safety (mypy)
- Code complexity (cyclomatic complexity)

These rewards provide REAL signal vs. pattern-matching heuristics.
"""

import ast
import re
import sys
import time
import tempfile
import subprocess
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import json


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float
    
    
@dataclass 
class LintResult:
    """Result of linting."""
    score: float  # 0.0 to 1.0
    errors: int
    warnings: int
    issues: List[str]


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown-style output."""
    # Match ```python ... ``` or ```py ... ``` or ``` ... ```
    pattern = r'```(?:python|py)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches
    
    # Fallback: look for indented code blocks
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            code_lines.append(line.lstrip())
            in_code = True
        elif in_code and line.strip() == '':
            code_lines.append('')
        elif in_code:
            break
            
    if code_lines:
        return ['\n'.join(code_lines)]
    
    # Last resort: treat entire response as code
    return [text]


def syntax_check(code: str) -> Tuple[bool, str]:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def execute_code_safe(
    code: str, 
    timeout: float = 5.0,
    test_input: str = ""
) -> ExecutionResult:
    """
    Execute code in isolated subprocess with timeout.
    
    Security: Uses subprocess isolation, not exec().
    """
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.py', 
        delete=False
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        start_time = time.perf_counter()
        
        result = subprocess.run(
            [sys.executable, temp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        
        execution_time = time.perf_counter() - start_time
        
        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout[:2000],  # Limit output size
            error=result.stderr[:2000],
            exit_code=result.returncode,
            execution_time=execution_time,
        )
        
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            error="Execution timed out",
            exit_code=-1,
            execution_time=timeout,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            error=str(e),
            exit_code=-1,
            execution_time=0.0,
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


def run_with_tests(code: str, test_code: str, timeout: float = 10.0) -> Tuple[int, int, str]:
    """
    Run code with unit tests, return (passed, total, output).
    
    Injects test code after the solution and counts assertions.
    """
    full_code = f"""
{code}

# === Test Cases ===
import sys
_passed = 0
_failed = 0

def _test(condition, name="test"):
    global _passed, _failed
    if condition:
        _passed += 1
    else:
        _failed += 1
        print(f"FAILED: {{name}}", file=sys.stderr)

{test_code}

print(f"PASSED: {{_passed}}/{{_passed + _failed}}")
"""
    
    result = execute_code_safe(full_code, timeout=timeout)
    
    if not result.success:
        return 0, 1, result.error
    
    # Parse "PASSED: X/Y" from output
    match = re.search(r'PASSED:\s*(\d+)/(\d+)', result.output)
    if match:
        passed = int(match.group(1))
        total = int(match.group(2))
        return passed, total, result.output
    
    return 0, 1, result.output


def lint_code(code: str) -> LintResult:
    """
    Run ruff linter on code, return normalized score.
    
    Falls back to basic checks if ruff not installed.
    """
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.py', 
        delete=False
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        # Try ruff first (fast, modern)
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", temp_path],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        
        issues = json.loads(result.stdout) if result.stdout else []
        errors = sum(1 for i in issues if i.get('type') == 'E')
        warnings = len(issues) - errors
        
        # Score: 1.0 for clean, decreasing with issues
        max_issues = 20
        score = max(0.0, 1.0 - len(issues) / max_issues)
        
        return LintResult(
            score=score,
            errors=errors,
            warnings=warnings,
            issues=[f"{i['code']}: {i['message']}" for i in issues[:5]],
        )
        
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        # Fallback: basic style checks
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append(f"Line {i}: too long ({len(line)} > 100)")
            if line.rstrip() != line:
                issues.append(f"Line {i}: trailing whitespace")
            if '\t' in line:
                issues.append(f"Line {i}: uses tabs instead of spaces")
        
        score = max(0.0, 1.0 - len(issues) / 20)
        return LintResult(score=score, errors=0, warnings=len(issues), issues=issues[:5])
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def type_check(code: str) -> Tuple[float, List[str]]:
    """
    Run mypy type checker, return (score, errors).
    """
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.py', 
        delete=False
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["mypy", "--ignore-missing-imports", "--no-error-summary", temp_path],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        
        errors = [l for l in result.stdout.split('\n') if 'error:' in l]
        score = max(0.0, 1.0 - len(errors) / 10)
        
        return score, errors[:5]
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1.0, []  # Skip if mypy not installed
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def measure_performance(
    code: str,
    setup_code: str = "",
    n_runs: int = 3,
    timeout: float = 10.0
) -> Tuple[float, float]:
    """
    Benchmark code performance, return (mean_time, std_time).
    
    Useful for rewarding optimized solutions.
    """
    benchmark_code = f"""
import time
import statistics

{setup_code}

times = []
for _ in range({n_runs}):
    start = time.perf_counter()
    {code}
    times.append(time.perf_counter() - start)

print(f"PERF:{{statistics.mean(times):.6f}}:{{statistics.stdev(times) if len(times) > 1 else 0:.6f}}")
"""
    
    result = execute_code_safe(benchmark_code, timeout=timeout)
    
    if result.success:
        match = re.search(r'PERF:([\d.]+):([\d.]+)', result.output)
        if match:
            return float(match.group(1)), float(match.group(2))
    
    return float('inf'), 0.0


def compute_complexity(code: str) -> Dict[str, int]:
    """
    Compute code complexity metrics.
    
    Returns dict with:
    - lines: total lines
    - functions: number of function definitions
    - classes: number of class definitions
    - branches: if/for/while/try count (proxy for cyclomatic complexity)
    - nesting: max nesting depth
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"lines": 0, "functions": 0, "classes": 0, "branches": 0, "nesting": 0}
    
    metrics = {
        "lines": len(code.split('\n')),
        "functions": 0,
        "classes": 0,
        "branches": 0,
        "nesting": 0,
    }
    
    def count_nesting(node, depth=0):
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                max_depth = max(max_depth, count_nesting(child, depth + 1))
            else:
                max_depth = max(max_depth, count_nesting(child, depth))
        return max_depth
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            metrics["functions"] += 1
        elif isinstance(node, ast.ClassDef):
            metrics["classes"] += 1
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            metrics["branches"] += 1
    
    metrics["nesting"] = count_nesting(tree)
    
    return metrics


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def syntax_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for syntactically valid code.
    
    Binary: 1.0 if valid, 0.0 if syntax error.
    """
    rewards = []
    for completion in completions:
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
            
        # Check all code blocks
        all_valid = all(syntax_check(block)[0] for block in code_blocks)
        rewards.append(1.0 if all_valid else 0.0)
        
    return rewards


def execution_reward(
    prompts: List[str],
    completions: List[str],
    test_inputs: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Reward for code that executes without errors.
    
    1.0 = runs successfully
    0.5 = runs but produces stderr
    0.0 = crashes or times out
    """
    test_inputs = test_inputs or [""] * len(completions)
    rewards = []
    
    for completion, test_input in zip(completions, test_inputs):
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
        
        code = code_blocks[0]  # Use first code block
        result = execute_code_safe(code, test_input=test_input)
        
        if result.success and not result.error:
            rewards.append(1.0)
        elif result.success:  # Ran but had stderr
            rewards.append(0.5)
        else:
            rewards.append(0.0)
            
    return rewards


def test_pass_reward(
    prompts: List[str],
    completions: List[str],
    test_cases: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Reward based on unit test pass rate.
    
    Score = passed_tests / total_tests
    """
    if not test_cases:
        return [0.5] * len(completions)  # Neutral if no tests
    
    rewards = []
    
    for completion, tests in zip(completions, test_cases):
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
        
        code = code_blocks[0]
        passed, total, _ = run_with_tests(code, tests)
        
        if total == 0:
            rewards.append(0.0)
        else:
            rewards.append(passed / total)
            
    return rewards


def lint_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for clean, well-formatted code.
    
    Uses ruff linter score (0.0 to 1.0).
    """
    rewards = []
    
    for completion in completions:
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
        
        code = code_blocks[0]
        
        # Skip if syntax error
        if not syntax_check(code)[0]:
            rewards.append(0.0)
            continue
        
        result = lint_code(code)
        rewards.append(result.score)
        
    return rewards


def type_safety_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for type-safe code (mypy clean).
    """
    rewards = []
    
    for completion in completions:
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
        
        code = code_blocks[0]
        score, _ = type_check(code)
        rewards.append(score)
        
    return rewards


def performance_reward(
    prompts: List[str],
    completions: List[str],
    baseline_times: List[float] = None,
    **kwargs
) -> List[float]:
    """
    Reward for performant code (faster = better).
    
    If baseline provided: score = baseline_time / completion_time (capped at 2.0)
    Otherwise: score inversely proportional to execution time
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
        
        code = code_blocks[0]
        mean_time, _ = measure_performance(code)
        
        if mean_time == float('inf'):
            rewards.append(0.0)
            continue
        
        if baseline_times and i < len(baseline_times):
            baseline = baseline_times[i]
            # Reward speedup (2x faster = score of 2.0, capped)
            speedup = min(2.0, baseline / max(mean_time, 1e-6))
            rewards.append(speedup / 2.0)  # Normalize to 0-1
        else:
            # No baseline: reward fast execution (< 0.1s = 1.0)
            rewards.append(max(0.0, 1.0 - mean_time / 0.1))
            
    return rewards


def complexity_reward(
    prompts: List[str],
    completions: List[str],
    prefer_simple: bool = True,
    **kwargs
) -> List[float]:
    """
    Reward for appropriate code complexity.
    
    If prefer_simple: reward lower complexity
    Otherwise: neutral (for complex algorithm problems)
    """
    rewards = []
    
    for completion in completions:
        code_blocks = extract_code_blocks(completion)
        if not code_blocks:
            rewards.append(0.0)
            continue
        
        code = code_blocks[0]
        metrics = compute_complexity(code)
        
        if not prefer_simple:
            rewards.append(0.5)  # Neutral
            continue
        
        # Penalize excessive complexity
        score = 1.0
        
        # Penalize deep nesting (> 4 levels)
        if metrics["nesting"] > 4:
            score -= (metrics["nesting"] - 4) * 0.1
        
        # Penalize many branches (> 10)
        if metrics["branches"] > 10:
            score -= (metrics["branches"] - 10) * 0.05
        
        # Penalize very long code (> 50 lines for simple problems)
        if metrics["lines"] > 50:
            score -= (metrics["lines"] - 50) * 0.01
        
        rewards.append(max(0.0, score))
        
    return rewards


def combined_code_reward(
    prompts: List[str],
    completions: List[str],
    test_cases: List[str] = None,
    weights: Dict[str, float] = None,
    **kwargs
) -> List[float]:
    """
    Combined code reward with configurable weights.
    
    Default weights optimized for coding benchmarks:
    - test_pass: 0.4 (most important: does it work?)
    - execution: 0.25 (does it run at all?)
    - syntax: 0.15 (is it valid code?)
    - lint: 0.1 (is it clean?)
    - complexity: 0.1 (is it elegant?)
    """
    if weights is None:
        weights = {
            "test_pass": 0.4,
            "execution": 0.25,
            "syntax": 0.15,
            "lint": 0.1,
            "complexity": 0.1,
        }
    
    # Compute individual rewards
    syntax_scores = syntax_reward(prompts, completions)
    execution_scores = execution_reward(prompts, completions)
    test_scores = test_pass_reward(prompts, completions, test_cases=test_cases)
    lint_scores = lint_reward(prompts, completions)
    complexity_scores = complexity_reward(prompts, completions)
    
    # Weighted combination
    total_weight = sum(weights.values())
    rewards = []
    
    for i in range(len(completions)):
        # Gate on syntax: if syntax fails, heavy penalty
        if syntax_scores[i] == 0.0:
            rewards.append(0.1)  # Small reward for trying
            continue
        
        score = (
            weights.get("syntax", 0.15) * syntax_scores[i] +
            weights.get("execution", 0.25) * execution_scores[i] +
            weights.get("test_pass", 0.4) * test_scores[i] +
            weights.get("lint", 0.1) * lint_scores[i] +
            weights.get("complexity", 0.1) * complexity_scores[i]
        ) / total_weight
        
        rewards.append(score)
    
    return rewards


# Registry
CODE_REWARD_FUNCTIONS = {
    "syntax": syntax_reward,
    "execution": execution_reward,
    "test_pass": test_pass_reward,
    "lint": lint_reward,
    "type_safety": type_safety_reward,
    "performance": performance_reward,
    "complexity": complexity_reward,
    "combined": combined_code_reward,
}


def get_code_reward_function(name: str):
    """Get a code reward function by name."""
    if name not in CODE_REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward: {name}. Available: {list(CODE_REWARD_FUNCTIONS.keys())}")
    return CODE_REWARD_FUNCTIONS[name]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the rewards
    test_completions = [
        # Good code
        '''```python
def add(a, b):
    """Add two numbers."""
    return a + b

result = add(2, 3)
print(result)
```''',
        # Syntax error
        '''```python
def broken(
    return 42
```''',
        # Runtime error
        '''```python
x = 1 / 0
```''',
        # Works but poor style
        '''```python
def f(x):y=x*2;return y
print(f(5))
```''',
    ]
    
    test_cases = [
        "_test(add(2, 3) == 5, 'add works')\n_test(add(-1, 1) == 0, 'add negative')",
        "",
        "",
        "_test(f(5) == 10, 'f works')",
    ]
    
    prompts = [""] * len(test_completions)
    
    print("Testing code rewards...")
    print("=" * 50)
    
    print("\nSyntax rewards:", syntax_reward(prompts, test_completions))
    print("Execution rewards:", execution_reward(prompts, test_completions))
    print("Test pass rewards:", test_pass_reward(prompts, test_completions, test_cases=test_cases))
    print("Lint rewards:", lint_reward(prompts, test_completions))
    print("Combined rewards:", combined_code_reward(prompts, test_completions, test_cases=test_cases))
