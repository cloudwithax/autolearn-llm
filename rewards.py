"""
Reward Functions for GRPO Training

GRPO (Group Relative Policy Optimization) scores multiple candidate completions
per prompt and uses relative rewards within the group for policy updates.
"""

import re
from typing import List, Dict, Any, Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model output."""
    # Look for boxed answers: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    
    # Look for "The answer is X" pattern
    answer_match = re.search(r'[Tt]he (?:final )?answer is[:\s]*([+-]?\d+(?:\.\d+)?)', text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for "#### X" pattern (GSM8K format)
    hash_match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
    if hash_match:
        return hash_match.group(1).strip()
    
    # Fallback: last number in text
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def normalize_number(s: str) -> float:
    """Normalize a number string for comparison."""
    try:
        # Remove commas and whitespace
        s = s.replace(',', '').replace(' ', '')
        return float(s)
    except (ValueError, TypeError):
        return float('nan')


def correctness_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    **kwargs
) -> List[float]:
    """
    Reward based on answer correctness.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answers: List of ground truth answers
        
    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        if predicted is None:
            rewards.append(0.0)
            continue
            
        pred_val = normalize_number(predicted)
        true_val = normalize_number(answer)
        
        # Check if answers match (with small tolerance for floats)
        if abs(pred_val - true_val) < 1e-6:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for proper formatting (chain-of-thought structure).
    
    Encourages:
    - Step-by-step reasoning
    - Clear answer presentation
    - Proper use of markers
    """
    rewards = []
    for completion in completions:
        score = 0.0
        
        # Has step-by-step markers
        if re.search(r'[Ss]tep\s*\d|[Ff]irst|[Ss]econd|[Tt]hen|[Nn]ext|[Ff]inally', completion):
            score += 0.3
            
        # Has clear answer presentation
        if re.search(r'\\boxed\{|[Tt]he (?:final )?answer is|####', completion):
            score += 0.4
            
        # Shows mathematical work
        if re.search(r'[=\+\-\*\/×÷]', completion):
            score += 0.2
            
        # Not too short (shows reasoning)
        if len(completion.split()) > 20:
            score += 0.1
            
        rewards.append(min(score, 1.0))
        
    return rewards


def reasoning_length_reward(
    prompts: List[str],
    completions: List[str],
    min_length: int = 50,
    max_length: int = 500,
    **kwargs
) -> List[float]:
    """
    Reward for appropriate reasoning length.
    
    Too short = insufficient reasoning
    Too long = rambling/inefficient
    """
    rewards = []
    for completion in completions:
        word_count = len(completion.split())
        
        if word_count < min_length:
            # Penalize very short responses
            score = word_count / min_length * 0.5
        elif word_count > max_length:
            # Slight penalty for very long responses
            score = max(0.5, 1.0 - (word_count - max_length) / max_length)
        else:
            # Optimal range
            score = 1.0
            
        rewards.append(score)
        
    return rewards


def xml_format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for XML-structured thinking (DeepSeek-R1 style).
    
    Expected format:
    <think>
    reasoning steps...
    </think>
    <answer>final answer</answer>
    """
    rewards = []
    for completion in completions:
        score = 0.0
        
        # Has thinking tags
        if '<think>' in completion and '</think>' in completion:
            score += 0.4
            
        # Has answer tags
        if '<answer>' in completion and '</answer>' in completion:
            score += 0.4
            
        # Tags are properly ordered
        think_start = completion.find('<think>')
        think_end = completion.find('</think>')
        answer_start = completion.find('<answer>')
        
        if think_start < think_end < answer_start:
            score += 0.2
            
        rewards.append(score)
        
    return rewards


def combined_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    weights: Dict[str, float] = None,
    **kwargs
) -> List[float]:
    """
    Combined reward function with configurable weights.
    
    Default weights emphasize correctness while encouraging good formatting.
    """
    if weights is None:
        weights = {
            'correctness': 1.0,
            'format': 0.5,
            'reasoning': 0.3,
        }
    
    # Compute individual rewards
    correct_scores = correctness_reward(prompts, completions, answers)
    format_scores = format_reward(prompts, completions)
    reasoning_scores = reasoning_length_reward(prompts, completions)
    
    # Weighted combination
    rewards = []
    total_weight = sum(weights.values())
    
    for i in range(len(completions)):
        score = (
            weights.get('correctness', 1.0) * correct_scores[i] +
            weights.get('format', 0.5) * format_scores[i] +
            weights.get('reasoning', 0.3) * reasoning_scores[i]
        ) / total_weight
        rewards.append(score)
        
    return rewards


# Registry of reward functions
REWARD_FUNCTIONS = {
    'correctness': correctness_reward,
    'format': format_reward,
    'reasoning': reasoning_length_reward,
    'xml_format': xml_format_reward,
    'combined': combined_reward,
}


def get_reward_function(name: str):
    """Get a reward function by name."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[name]
