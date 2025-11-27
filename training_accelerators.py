"""
Novel Training Acceleration Methods for FP8 GRPO

Implements cutting-edge optimizations:
1. Async reward computation (parallel reward evaluation)
2. Reward caching (avoid recomputing deterministic rewards)
3. Token packing (maximize GPU utilization)
4. Speculative generation hooks (for vLLM integration)
5. Curriculum learning (easy→hard scheduling)
6. Gradient accumulation with mixed precision scaling

Based on:
- FastGRPO (arXiv:2509.21792) - 2.35-2.72x speedup
- Async-GRPO (Red Hat) - decoupled pipeline
- Token packing from Unsloth optimizations
"""

import os
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
from functools import lru_cache
import json

import torch
from collections import OrderedDict


# =============================================================================
# 1. REWARD CACHING - Avoid recomputing deterministic rewards
# =============================================================================

class RewardCache:
    """
    LRU cache for deterministic reward components.
    
    Syntax and lint rewards are deterministic for the same code.
    Cache them to avoid redundant computation during GRPO iterations.
    """
    
    def __init__(self, maxsize: int = 10000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def _hash_key(self, completion: str, reward_type: str) -> str:
        """Create hash key for completion + reward type."""
        content = f"{reward_type}:{completion}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, completion: str, reward_type: str) -> Optional[float]:
        """Get cached reward if exists."""
        key = self._hash_key(completion, reward_type)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, completion: str, reward_type: str, reward: float):
        """Cache a reward value."""
        key = self._hash_key(completion, reward_type)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = reward
            # Evict oldest if over capacity
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "size": len(self.cache),
        }


# Global cache instance
_reward_cache = RewardCache(maxsize=50000)


def cached_reward(reward_type: str):
    """
    Decorator to cache deterministic reward functions.
    
    Usage:
        @cached_reward("syntax")
        def syntax_reward(completion: str) -> float:
            ...
    """
    def decorator(func: Callable):
        def wrapper(completion: str, *args, **kwargs) -> float:
            # Check cache first
            cached = _reward_cache.get(completion, reward_type)
            if cached is not None:
                return cached
            
            # Compute and cache
            result = func(completion, *args, **kwargs)
            _reward_cache.set(completion, reward_type, result)
            return result
        
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# =============================================================================
# 2. ASYNC REWARD COMPUTATION - Parallel reward evaluation
# =============================================================================

class AsyncRewardComputer:
    """
    Compute rewards asynchronously using thread pool.
    
    Separates fast rewards (syntax, format) from slow rewards (execution, tests).
    Fast rewards run synchronously, slow rewards run in parallel.
    """
    
    def __init__(
        self,
        fast_rewards: Dict[str, Callable],
        slow_rewards: Dict[str, Callable],
        weights: Dict[str, float],
        max_workers: int = 4,
    ):
        """
        Args:
            fast_rewards: Dict of reward_name -> reward_fn for fast/cached rewards
            slow_rewards: Dict of reward_name -> reward_fn for slow rewards (execution)
            weights: Dict of reward_name -> weight
            max_workers: Thread pool size for slow rewards
        """
        self.fast_rewards = fast_rewards
        self.slow_rewards = slow_rewards
        self.weights = weights
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Timing stats
        self.fast_time = 0.0
        self.slow_time = 0.0
        self.total_calls = 0
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """Compute weighted rewards with async slow rewards."""
        
        n = len(completions)
        reward_components = {name: [0.0] * n for name in 
                           list(self.fast_rewards.keys()) + list(self.slow_rewards.keys())}
        
        # 1. Compute fast rewards synchronously
        t0 = time.perf_counter()
        for name, fn in self.fast_rewards.items():
            try:
                scores = fn(prompts, completions, **kwargs)
                reward_components[name] = scores
            except Exception as e:
                print(f"Warning: Fast reward '{name}' failed: {e}")
        self.fast_time += time.perf_counter() - t0
        
        # 2. Compute slow rewards in parallel
        t0 = time.perf_counter()
        futures = {}
        for name, fn in self.slow_rewards.items():
            future = self.executor.submit(fn, prompts, completions, **kwargs)
            futures[future] = name
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                scores = future.result(timeout=30)  # 30s timeout per reward
                reward_components[name] = scores
            except Exception as e:
                print(f"Warning: Slow reward '{name}' failed: {e}")
        self.slow_time += time.perf_counter() - t0
        
        # 3. Combine with weights
        total_weight = sum(self.weights.get(name, 0) for name in reward_components)
        final_rewards = []
        
        for i in range(n):
            score = sum(
                self.weights.get(name, 0) * reward_components[name][i]
                for name in reward_components
            ) / max(total_weight, 1e-8)
            final_rewards.append(score)
        
        self.total_calls += 1
        return final_rewards
    
    def stats(self) -> Dict[str, Any]:
        """Return timing statistics."""
        total = self.fast_time + self.slow_time
        return {
            "fast_time_pct": f"{100 * self.fast_time / max(total, 1e-8):.1f}%",
            "slow_time_pct": f"{100 * self.slow_time / max(total, 1e-8):.1f}%",
            "total_time": f"{total:.2f}s",
            "calls": self.total_calls,
            "cache": _reward_cache.stats(),
        }


# =============================================================================
# 3. CURRICULUM LEARNING - Easy to hard scheduling
# =============================================================================

@dataclass
class CurriculumScheduler:
    """
    Schedule training from easy to hard examples.
    
    Strategies:
    - length: Short prompts/solutions first
    - complexity: Simpler problems first (based on reference solution)
    - success_rate: Problems with higher initial success rate first
    """
    
    strategy: str = "length"  # length, complexity, success_rate
    warmup_steps: int = 100
    current_step: int = 0
    difficulty_scores: Dict[int, float] = field(default_factory=dict)
    
    def compute_difficulty(self, dataset, strategy: str = None) -> List[float]:
        """Compute difficulty score for each example."""
        strategy = strategy or self.strategy
        difficulties = []
        
        for i, item in enumerate(dataset):
            if strategy == "length":
                # Longer prompts/solutions are harder
                prompt_len = len(item.get("prompt", ""))
                solution_len = len(item.get("canonical_solution", ""))
                diff = (prompt_len + solution_len) / 1000  # Normalize
                
            elif strategy == "complexity":
                # Count control structures in reference solution
                solution = item.get("canonical_solution", "")
                control_keywords = ["if", "for", "while", "try", "with", "def", "class"]
                diff = sum(solution.count(kw) for kw in control_keywords)
                
            elif strategy == "success_rate":
                # Use cached success rate if available, else estimate
                diff = 1.0 - self.difficulty_scores.get(i, 0.5)
                
            else:
                diff = 0.5
            
            difficulties.append(diff)
            self.difficulty_scores[i] = diff
        
        return difficulties
    
    def get_sample_weights(self, dataset, step: int = None) -> List[float]:
        """
        Get sampling weights for current curriculum position.
        
        Early: weight toward easy examples
        Late: uniform weights
        """
        step = step if step is not None else self.current_step
        
        # Compute difficulties if not cached
        if not self.difficulty_scores:
            self.compute_difficulty(dataset)
        
        # Progress ratio (0 = start, 1 = past warmup)
        progress = min(1.0, step / max(self.warmup_steps, 1))
        
        weights = []
        for i in range(len(dataset)):
            diff = self.difficulty_scores.get(i, 0.5)
            
            # Early: prefer easy (low difficulty)
            # Late: uniform
            easy_weight = 1.0 - diff
            uniform_weight = 1.0
            
            weight = (1 - progress) * easy_weight + progress * uniform_weight
            weights.append(max(0.1, weight))  # Minimum weight to avoid zeros
        
        # Normalize
        total = sum(weights)
        return [w / total for w in weights]
    
    def step(self):
        """Advance curriculum by one step."""
        self.current_step += 1
    
    def update_difficulty(self, idx: int, success: bool):
        """Update difficulty estimate based on training success."""
        old = self.difficulty_scores.get(idx, 0.5)
        # Exponential moving average
        new_signal = 0.0 if success else 1.0
        self.difficulty_scores[idx] = 0.9 * old + 0.1 * new_signal


# =============================================================================
# 4. TOKEN PACKING - Maximize GPU utilization
# =============================================================================

def pack_sequences(
    prompts: List[str],
    completions: List[str],
    tokenizer,
    max_length: int = 2048,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Pack multiple short sequences into single tensors.
    
    Returns:
        input_ids: Packed tensor [num_packs, max_length]
        attention_mask: Packed attention mask
        sequence_ranges: List of (start, end) for each original sequence
    """
    packed_ids = []
    packed_masks = []
    sequence_ranges = []
    
    current_pack = []
    current_length = 0
    pack_start_idx = 0
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        # Tokenize
        text = prompt + completion
        tokens = tokenizer.encode(text, add_special_tokens=False)
        seq_len = len(tokens)
        
        if current_length + seq_len > max_length:
            # Finalize current pack
            if current_pack:
                pack_tensor = torch.tensor(current_pack + 
                    [pad_token_id] * (max_length - len(current_pack)))
                mask_tensor = torch.tensor(
                    [1] * len(current_pack) + [0] * (max_length - len(current_pack)))
                packed_ids.append(pack_tensor)
                packed_masks.append(mask_tensor)
            
            # Start new pack
            current_pack = tokens
            current_length = seq_len
            pack_start_idx = len(packed_ids)
            sequence_ranges.append((0, seq_len))
        else:
            # Add to current pack
            start = len(current_pack)
            current_pack.extend(tokens)
            current_length += seq_len
            sequence_ranges.append((start, start + seq_len))
    
    # Handle last pack
    if current_pack:
        pack_tensor = torch.tensor(current_pack + 
            [pad_token_id] * (max_length - len(current_pack)))
        mask_tensor = torch.tensor(
            [1] * len(current_pack) + [0] * (max_length - len(current_pack)))
        packed_ids.append(pack_tensor)
        packed_masks.append(mask_tensor)
    
    return (
        torch.stack(packed_ids),
        torch.stack(packed_masks),
        sequence_ranges
    )


# =============================================================================
# 5. SPECULATIVE GENERATION HOOKS - For vLLM integration
# =============================================================================

@dataclass
class SpeculativeConfig:
    """
    Configuration for speculative decoding in GRPO.
    
    Based on FastGRPO paper - achieves 2.35-2.72x speedup.
    """
    
    enabled: bool = True
    draft_model_name: Optional[str] = None  # Smaller draft model
    num_speculative_tokens: int = 4  # Tokens to speculate per step
    
    # Concurrency-aware settings (from FastGRPO)
    dynamic_speculation: bool = True  # Adjust speculation based on batch size
    min_speculation_tokens: int = 2
    max_speculation_tokens: int = 8
    
    # Online draft learning
    update_draft_every: int = 100  # Steps between draft model updates
    draft_learning_rate: float = 1e-5


def get_speculation_tokens(
    config: SpeculativeConfig,
    batch_size: int,
    gpu_utilization: float = 0.8,
) -> int:
    """
    Dynamically adjust speculation tokens based on concurrency.
    
    Higher batch size -> fewer speculation tokens (memory constrained)
    Lower GPU utilization -> more speculation (compute available)
    """
    if not config.dynamic_speculation:
        return config.num_speculative_tokens
    
    # Scale speculation inversely with batch size
    base = config.num_speculative_tokens
    batch_factor = max(0.5, 1.0 - (batch_size - 1) * 0.1)
    util_factor = 1.0 + (1.0 - gpu_utilization) * 0.5
    
    tokens = int(base * batch_factor * util_factor)
    return max(config.min_speculation_tokens, 
               min(config.max_speculation_tokens, tokens))


# =============================================================================
# 6. MIXED PRECISION GRADIENT SCALING
# =============================================================================

class AdaptiveGradScaler:
    """
    Adaptive gradient scaling for FP8/BF16 mixed precision.
    
    Automatically adjusts scale based on gradient statistics.
    """
    
    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 100,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.steps_since_growth = 0
        self.overflow_count = 0
    
    def scale_gradients(self, grads: torch.Tensor) -> torch.Tensor:
        """Scale gradients."""
        return grads * self.scale
    
    def unscale_gradients(self, grads: torch.Tensor) -> torch.Tensor:
        """Unscale gradients."""
        return grads / self.scale
    
    def update(self, overflow: bool):
        """Update scale based on overflow detection."""
        if overflow:
            self.scale *= self.backoff_factor
            self.overflow_count += 1
            self.steps_since_growth = 0
        else:
            self.steps_since_growth += 1
            if self.steps_since_growth >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_growth = 0
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "steps_since_growth": self.steps_since_growth,
            "overflow_count": self.overflow_count,
        }


# =============================================================================
# 7. TRAINING ACCELERATOR WRAPPER
# =============================================================================

class GRPOAccelerator:
    """
    Main accelerator wrapper combining all optimizations.
    
    Usage:
        accelerator = GRPOAccelerator(config)
        
        # Wrap reward function
        reward_fn = accelerator.accelerate_rewards(my_reward_fn)
        
        # Get curriculum sampler
        sampler = accelerator.get_curriculum_sampler(dataset)
        
        # During training
        accelerator.step()
    """
    
    def __init__(
        self,
        enable_reward_cache: bool = True,
        enable_async_rewards: bool = True,
        enable_curriculum: bool = True,
        enable_speculation: bool = False,  # Requires vLLM support
        curriculum_warmup: int = 100,
        async_workers: int = 4,
    ):
        self.enable_reward_cache = enable_reward_cache
        self.enable_async_rewards = enable_async_rewards
        self.enable_curriculum = enable_curriculum
        self.enable_speculation = enable_speculation
        
        # Components
        self.curriculum = CurriculumScheduler(warmup_steps=curriculum_warmup)
        self.speculation_config = SpeculativeConfig(enabled=enable_speculation)
        self.async_workers = async_workers
        
        # Stats
        self.step_count = 0
        self.start_time = time.time()
    
    def accelerate_rewards(
        self,
        reward_fn: Callable,
        fast_rewards: Dict[str, Callable] = None,
        slow_rewards: Dict[str, Callable] = None,
        weights: Dict[str, float] = None,
    ) -> Callable:
        """
        Wrap reward function with async computation and caching.
        """
        if not self.enable_async_rewards:
            return reward_fn
        
        if fast_rewards is None or slow_rewards is None:
            # Use original function as single slow reward
            return AsyncRewardComputer(
                fast_rewards={},
                slow_rewards={"combined": reward_fn},
                weights=weights or {"combined": 1.0},
                max_workers=self.async_workers,
            )
        
        return AsyncRewardComputer(
            fast_rewards=fast_rewards,
            slow_rewards=slow_rewards,
            weights=weights,
            max_workers=self.async_workers,
        )
    
    def get_curriculum_sampler(self, dataset):
        """
        Get weighted sampler for curriculum learning.
        """
        if not self.enable_curriculum:
            return None
        
        weights = self.curriculum.get_sample_weights(dataset, self.step_count)
        
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
    
    def step(self):
        """Advance training step."""
        self.step_count += 1
        if self.enable_curriculum:
            self.curriculum.step()
    
    def stats(self) -> Dict[str, Any]:
        """Return acceleration statistics."""
        elapsed = time.time() - self.start_time
        return {
            "steps": self.step_count,
            "elapsed_time": f"{elapsed:.1f}s",
            "steps_per_second": f"{self.step_count / max(elapsed, 1):.2f}",
            "cache_stats": _reward_cache.stats(),
            "curriculum_progress": f"{min(100, 100 * self.step_count / max(self.curriculum.warmup_steps, 1)):.1f}%",
        }


# =============================================================================
# QUICK START EXAMPLE
# =============================================================================

def create_accelerated_trainer_config() -> Dict[str, Any]:
    """
    Returns recommended config for maximum training speed.
    """
    return {
        # Async rewards
        "enable_async_rewards": True,
        "async_workers": 4,  # Parallel reward threads
        
        # Caching
        "enable_reward_cache": True,
        "cache_size": 50000,
        
        # Curriculum
        "enable_curriculum": True,
        "curriculum_warmup": 100,  # Steps before full difficulty
        "curriculum_strategy": "length",  # or "complexity", "success_rate"
        
        # Generation (if using vLLM)
        "enable_speculation": False,  # Set True with compatible vLLM
        "num_speculative_tokens": 4,
        
        # Batch optimization
        "use_token_packing": True,
        "dynamic_batch_size": True,
        
        # Gradient optimization  
        "gradient_checkpointing": True,  # Already in Unsloth
        "optim": "adamw_8bit",
        "bf16": True,
    }


if __name__ == "__main__":
    # Test the accelerator
    print("Testing GRPO Accelerator...")
    
    accelerator = GRPOAccelerator(
        enable_reward_cache=True,
        enable_async_rewards=True,
        enable_curriculum=True,
    )
    
    # Simulate steps
    for _ in range(50):
        accelerator.step()
    
    print(f"Stats: {accelerator.stats()}")
    print(f"Recommended config: {create_accelerated_trainer_config()}")
