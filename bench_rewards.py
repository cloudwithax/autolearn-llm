"""
Benchmark-Specific Rewards for SWE-Bench and Terminal-Bench

SWE-Bench: Real GitHub issue resolution
- Generate patches to fix issues
- Reward based on FAIL_TO_PASS tests

Terminal-Bench: Terminal/CLI task completion
- Generate shell commands
- Reward based on task verification scripts
"""

import os
import re
import json
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import difflib


@dataclass
class PatchResult:
    """Result of patch evaluation."""
    applies: bool       # Patch applies cleanly
    tests_pass: int     # Number of tests that pass
    tests_total: int    # Total tests
    error: str


@dataclass
class TerminalResult:
    """Result of terminal command evaluation."""
    success: bool
    output: str
    error: str
    exit_code: int


# =============================================================================
# PATCH UTILITIES (for SWE-Bench)
# =============================================================================

def extract_patch(text: str) -> Optional[str]:
    """
    Extract a unified diff patch from model output.
    
    Looks for:
    - ```diff ... ``` blocks
    - ```patch ... ``` blocks
    - Raw diff content (starts with --- or diff --git)
    """
    # Try diff/patch code blocks first
    patterns = [
        r'```(?:diff|patch)\s*\n(.*?)```',
        r'```\s*\n((?:---|\+\+\+|diff --git).*?)```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # Look for raw diff content
    lines = text.split('\n')
    diff_lines = []
    in_diff = False
    
    for line in lines:
        if line.startswith('diff --git') or line.startswith('---'):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
            # End of diff hunk
            if line.startswith('@@') and len(diff_lines) > 3:
                continue
    
    if diff_lines:
        return '\n'.join(diff_lines)
    
    return None


def validate_patch_format(patch: str) -> Tuple[bool, str]:
    """Check if patch has valid unified diff format."""
    if not patch:
        return False, "Empty patch"
    
    lines = patch.split('\n')
    has_minus = any(l.startswith('---') for l in lines)
    has_plus = any(l.startswith('+++') for l in lines)
    has_hunk = any(l.startswith('@@') for l in lines)
    
    if not (has_minus or has_plus or has_hunk):
        # Check for diff --git format
        if not any(l.startswith('diff --git') for l in lines):
            return False, "Missing diff markers (---, +++, or @@)"
    
    return True, ""


def compute_patch_similarity(generated: str, gold: str) -> float:
    """
    Compute similarity between generated and gold patch.
    Uses sequence matching on the actual changes (not headers).
    """
    def extract_changes(patch: str) -> List[str]:
        """Extract only the +/- lines (actual changes)."""
        changes = []
        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                changes.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                changes.append(line[1:])
        return changes
    
    gen_changes = extract_changes(generated or "")
    gold_changes = extract_changes(gold or "")
    
    if not gold_changes:
        return 0.0 if gen_changes else 1.0
    
    matcher = difflib.SequenceMatcher(None, gen_changes, gold_changes)
    return matcher.ratio()


# =============================================================================
# TERMINAL UTILITIES
# =============================================================================

def extract_commands(text: str) -> List[str]:
    """
    Extract shell commands from model output.
    
    Looks for:
    - ```bash/sh/shell ... ``` blocks
    - Lines starting with $ or >
    - Plain command lines
    """
    commands = []
    
    # Try bash/shell code blocks
    patterns = [
        r'```(?:bash|sh|shell|zsh)\s*\n(.*?)```',
        r'```\s*\n((?:\$|>).*?)```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            for line in match.split('\n'):
                line = line.strip()
                # Remove prompt prefixes
                if line.startswith('$ '):
                    line = line[2:]
                elif line.startswith('> '):
                    line = line[2:]
                if line and not line.startswith('#'):
                    commands.append(line)
    
    if commands:
        return commands
    
    # Fallback: look for command-like lines
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('$ '):
            commands.append(line[2:])
        elif re.match(r'^[a-z_][a-z0-9_-]*\s', line.lower()):
            # Looks like a command
            if any(cmd in line.lower() for cmd in ['cd ', 'ls', 'cat', 'echo', 'grep', 'find', 'mkdir', 'rm', 'cp', 'mv', 'git', 'python', 'pip', 'npm', 'curl', 'wget']):
                commands.append(line)
    
    return commands


def run_terminal_command(
    command: str,
    cwd: str = None,
    timeout: float = 30.0,
    env: Dict[str, str] = None,
) -> TerminalResult:
    """Execute a terminal command safely."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or tempfile.gettempdir(),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )
        return TerminalResult(
            success=result.returncode == 0,
            output=result.stdout[:2000],
            error=result.stderr[:2000],
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return TerminalResult(False, "", "Timeout", -1)
    except Exception as e:
        return TerminalResult(False, "", str(e), -1)


# =============================================================================
# SWE-BENCH REWARDS
# =============================================================================

def swe_patch_format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for valid patch format.
    
    0.0 = No patch found
    0.5 = Patch found but invalid format
    1.0 = Valid unified diff format
    """
    rewards = []
    for completion in completions:
        patch = extract_patch(completion)
        if patch is None:
            rewards.append(0.0)
            continue
        
        valid, _ = validate_patch_format(patch)
        rewards.append(1.0 if valid else 0.5)
    
    return rewards


def swe_patch_similarity_reward(
    prompts: List[str],
    completions: List[str],
    gold_patches: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Reward based on similarity to gold patch.
    
    This is a soft signal - actual test passing is the ground truth.
    """
    if not gold_patches:
        return [0.5] * len(completions)
    
    rewards = []
    for completion, gold in zip(completions, gold_patches):
        gen_patch = extract_patch(completion)
        if gen_patch is None:
            rewards.append(0.0)
            continue
        
        similarity = compute_patch_similarity(gen_patch, gold)
        rewards.append(similarity)
    
    return rewards


def swe_file_targeted_reward(
    prompts: List[str],
    completions: List[str],
    gold_patches: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Reward for targeting the correct files.
    
    Extracts file paths from patches and checks overlap with gold.
    """
    def extract_files(patch: str) -> set:
        if not patch:
            return set()
        files = set()
        for line in patch.split('\n'):
            if line.startswith('---') or line.startswith('+++'):
                # Extract path after --- a/ or +++ b/
                match = re.search(r'[ab]/(.+?)(?:\s|$)', line)
                if match:
                    files.add(match.group(1))
            elif line.startswith('diff --git'):
                matches = re.findall(r'[ab]/(\S+)', line)
                files.update(matches)
        return files
    
    if not gold_patches:
        return [0.5] * len(completions)
    
    rewards = []
    for completion, gold in zip(completions, gold_patches):
        gen_patch = extract_patch(completion)
        gen_files = extract_files(gen_patch)
        gold_files = extract_files(gold)
        
        if not gold_files:
            rewards.append(0.5)
            continue
        
        if not gen_files:
            rewards.append(0.0)
            continue
        
        # Jaccard similarity
        intersection = len(gen_files & gold_files)
        union = len(gen_files | gold_files)
        rewards.append(intersection / union if union > 0 else 0.0)
    
    return rewards


def swe_combined_reward(
    prompts: List[str],
    completions: List[str],
    gold_patches: List[str] = None,
    weights: Dict[str, float] = None,
    **kwargs
) -> List[float]:
    """
    Combined SWE-Bench reward.
    
    Weights:
    - format: Valid patch format (baseline requirement)
    - similarity: Similarity to gold patch (soft signal)
    - files: Targeting correct files (structural understanding)
    """
    if weights is None:
        weights = {
            "format": 0.3,
            "similarity": 0.5,
            "files": 0.2,
        }
    
    format_scores = swe_patch_format_reward(prompts, completions)
    similarity_scores = swe_patch_similarity_reward(prompts, completions, gold_patches)
    file_scores = swe_file_targeted_reward(prompts, completions, gold_patches)
    
    total_weight = sum(weights.values())
    rewards = []
    
    for i in range(len(completions)):
        # Gate on format - if no valid patch, heavy penalty
        if format_scores[i] == 0.0:
            rewards.append(0.1)
            continue
        
        score = (
            weights.get("format", 0.3) * format_scores[i] +
            weights.get("similarity", 0.5) * similarity_scores[i] +
            weights.get("files", 0.2) * file_scores[i]
        ) / total_weight
        rewards.append(score)
    
    return rewards


# =============================================================================
# TERMINAL-BENCH REWARDS
# =============================================================================

def terminal_command_format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for extractable terminal commands.
    
    0.0 = No commands found
    0.5 = Commands found but suspicious
    1.0 = Clean command extraction
    """
    rewards = []
    for completion in completions:
        commands = extract_commands(completion)
        if not commands:
            rewards.append(0.0)
        elif len(commands) > 20:  # Probably too many
            rewards.append(0.5)
        else:
            rewards.append(1.0)
    
    return rewards


def terminal_command_safety_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for safe (non-destructive) commands.
    
    Penalizes dangerous patterns like rm -rf, sudo, etc.
    """
    dangerous_patterns = [
        r'\brm\s+-rf\s+[/~]',     # rm -rf / or ~
        r'\bsudo\s+rm',           # sudo rm
        r'\bmkfs\b',              # Format filesystem
        r'\bdd\s+if=',            # dd (can destroy disks)
        r':(){.*};:',             # Fork bomb
        r'\bchmod\s+777\s+/',     # Dangerous permissions
        r'\bwget.*\|\s*bash',     # Pipe from web to bash
        r'\bcurl.*\|\s*bash',
    ]
    
    rewards = []
    for completion in completions:
        commands = extract_commands(completion)
        command_str = ' '.join(commands)
        
        is_dangerous = any(re.search(p, command_str) for p in dangerous_patterns)
        rewards.append(0.0 if is_dangerous else 1.0)
    
    return rewards


def terminal_execution_reward(
    prompts: List[str],
    completions: List[str],
    expected_outputs: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Reward based on command execution.
    
    Executes commands and checks for success.
    NOTE: Only use in sandboxed environments!
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        commands = extract_commands(completion)
        if not commands:
            rewards.append(0.0)
            continue
        
        # Execute first command (or combine with &&)
        command = commands[0] if len(commands) == 1 else ' && '.join(commands[:5])
        result = run_terminal_command(command, timeout=10.0)
        
        if result.success:
            if expected_outputs and i < len(expected_outputs):
                # Check if expected output is present
                expected = expected_outputs[i]
                if expected in result.output:
                    rewards.append(1.0)
                else:
                    rewards.append(0.7)  # Ran but wrong output
            else:
                rewards.append(1.0)
        else:
            rewards.append(0.2 if result.exit_code != -1 else 0.0)
    
    return rewards


def terminal_combined_reward(
    prompts: List[str],
    completions: List[str],
    expected_outputs: List[str] = None,
    weights: Dict[str, float] = None,
    **kwargs
) -> List[float]:
    """
    Combined Terminal-Bench reward.
    """
    if weights is None:
        weights = {
            "format": 0.2,
            "safety": 0.3,
            "execution": 0.5,
        }
    
    format_scores = terminal_command_format_reward(prompts, completions)
    safety_scores = terminal_command_safety_reward(prompts, completions)
    exec_scores = terminal_execution_reward(prompts, completions, expected_outputs)
    
    total_weight = sum(weights.values())
    rewards = []
    
    for i in range(len(completions)):
        # Gate on safety - dangerous commands get zero
        if safety_scores[i] == 0.0:
            rewards.append(0.0)
            continue
        
        score = (
            weights.get("format", 0.2) * format_scores[i] +
            weights.get("safety", 0.3) * safety_scores[i] +
            weights.get("execution", 0.5) * exec_scores[i]
        ) / total_weight
        rewards.append(score)
    
    return rewards


# =============================================================================
# REGISTRY
# =============================================================================

BENCH_REWARD_FUNCTIONS = {
    # SWE-Bench
    "swe_format": swe_patch_format_reward,
    "swe_similarity": swe_patch_similarity_reward,
    "swe_files": swe_file_targeted_reward,
    "swe_combined": swe_combined_reward,
    
    # Terminal-Bench
    "terminal_format": terminal_command_format_reward,
    "terminal_safety": terminal_command_safety_reward,
    "terminal_execution": terminal_execution_reward,
    "terminal_combined": terminal_combined_reward,
}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test SWE-Bench rewards
    print("Testing SWE-Bench rewards...")
    
    test_completions = [
        """Here's the fix:
```diff
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,7 +10,7 @@ def process(data):
-    return data.strip()
+    return data.strip() if data else ""
```
""",
        "I don't know how to fix this.",
        """Just change line 10:
```python
return data.strip() if data else ""
```
""",
    ]
    
    gold_patches = [
        """--- a/src/utils.py
+++ b/src/utils.py
@@ -10,7 +10,7 @@ def process(data):
-    return data.strip()
+    return data.strip() if data else ""
""",
        "",
        "",
    ]
    
    prompts = [""] * len(test_completions)
    
    print("Format rewards:", swe_patch_format_reward(prompts, test_completions))
    print("Similarity rewards:", swe_patch_similarity_reward(prompts, test_completions, gold_patches))
    print("Combined rewards:", swe_combined_reward(prompts, test_completions, gold_patches))
    
    # Test Terminal-Bench rewards
    print("\nTesting Terminal-Bench rewards...")
    
    terminal_completions = [
        """To list files:
```bash
ls -la
```
""",
        """```bash
rm -rf /
```
""",
        "I'm not sure what command to use.",
    ]
    
    print("Format rewards:", terminal_command_format_reward(prompts, terminal_completions))
    print("Safety rewards:", terminal_command_safety_reward(prompts, terminal_completions))
    print("Combined rewards:", terminal_combined_reward(prompts, terminal_completions))
