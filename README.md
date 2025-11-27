# AutoLearn-LLM: FP8 GRPO Training

Train LLMs with Group Relative Policy Optimization (GRPO) using FP8 precision on consumer GPUs.

**Now with execution-based code rewards!** Train models that actually pass tests, not just pattern-match.

## Why FP8 GRPO?

| Feature | Benefit |
|---------|---------|
| **60% less VRAM** | Train larger models on consumer GPUs |
| **1.4x faster inference** | vLLM FP8 kernels via TorchAO |
| **96% inference** | Training overhead is only 4% |
| **Memory sharing** | vLLM and training share weight buffers |

### VRAM Requirements (Approximate)

| Model | BF16 | FP8 |
|-------|------|-----|
| Qwen3-1.7B | 8GB | **5GB** |
| Llama-3.2-3B | 12GB | **8GB** |
| Qwen3-8B | 24GB | **16GB** |
| Qwen3-14B | 40GB | **24GB** |

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install base packages
pip install unsloth vllm trl transformers datasets pyyaml

# Install FP8 support (CUDA 12.8)
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
pip install --pre fbgemm-gpu fbgemm-gpu-genai --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
pip install --upgrade numba numpy
```

### 2. Configure Training

Edit `config.yaml`:

```yaml
model:
  name: "unsloth/Qwen3-1.7B"  # Change based on your VRAM
  load_in_fp8: true

training:
  num_generations: 4
  learning_rate: 5.0e-6
  
dataset:
  name: "openai/gsm8k"
  max_samples: 1000
```

### 3. Train

```bash
python train_fp8_grpo.py --config config.yaml
```

Or with command-line overrides:

```bash
python train_fp8_grpo.py \
    --model unsloth/Qwen3-4B \
    --dataset openai/gsm8k \
    --max_samples 500
```

### 4. Inference

```bash
# Interactive mode
python inference.py --model ./outputs/merged --mode interactive

# Benchmark mode  
python inference.py --model ./outputs/merged --mode benchmark
```

## How GRPO Works

GRPO (Group Relative Policy Optimization) is DeepSeek's RL algorithm:

1. **Generate** multiple candidate completions per prompt
2. **Score** each completion with reward functions
3. **Rank** completions within each group (relative rewards)
4. **Update** policy to favor higher-ranked completions

```
Prompt: "What is 2 + 2?"
  ├── Completion A: "4" → reward: 1.0
  ├── Completion B: "The answer is 4" → reward: 0.9  
  ├── Completion C: "22" → reward: 0.0
  └── Completion D: "2+2=4" → reward: 0.8

Policy update: Increase P(A), P(B), P(D); Decrease P(C)
```

## Reward Functions

Built-in reward functions in `rewards.py`:

| Function | Description |
|----------|-------------|
| `correctness` | Checks if extracted answer matches ground truth |
| `format` | Rewards step-by-step reasoning structure |
| `reasoning` | Rewards appropriate length (not too short/long) |
| `xml_format` | Rewards DeepSeek-R1 style `<think>` tags |
| `combined` | Weighted combination of above |

### Custom Rewards

```python
def my_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Your scoring logic
        score = 1.0 if "correct" in completion else 0.0
        rewards.append(score)
    return rewards
```

## Project Structure

```
autolearn-llm/
├── train_fp8_grpo.py   # Math/reasoning GRPO training
├── train_code_grpo.py  # Code generation GRPO training
├── rewards.py          # Math reward functions
├── code_rewards.py     # Execution-based code rewards
├── eval_code.py        # Benchmark evaluation (HumanEval, MBPP)
├── inference.py        # Model inference/testing
├── config.yaml         # Math training config
├── code_config.yaml    # Code training config
├── requirements.txt    # Python dependencies
├── examples/
│   └── custom_tests.json  # Example test cases
└── README.md
```

## Code Training (NEW!)

### Execution-Based Rewards

Unlike pattern-matching rewards, these use **actual code execution**:

| Reward | Signal | Weight |
|--------|--------|--------|
| `test_pass` | Unit tests pass | 0.50 |
| `execution` | Code runs without error | 0.20 |
| `syntax` | Valid Python AST | 0.15 |
| `lint` | Ruff linter score | 0.10 |
| `complexity` | Low cyclomatic complexity | 0.05 |
| `type_safety` | Mypy compliance | (optional) |
| `performance` | Execution time delta | (optional) |

### Train on Code

```bash
# HumanEval
python train_code_grpo.py --model unsloth/Qwen3-1.7B --dataset humaneval

# MBPP
python train_code_grpo.py --model unsloth/Qwen3-4B --dataset mbpp

# With custom config
python train_code_grpo.py --config code_config.yaml
```

### Evaluate

```bash
# HumanEval pass@1
python eval_code.py --model ./outputs/code/merged --benchmark humaneval

# pass@10 (sample 10 solutions)
python eval_code.py --model ./outputs/code/merged --benchmark humaneval --n_samples 10

# Custom test suite
python eval_code.py --model ./outputs/code/merged --benchmark custom --test_file examples/custom_tests.json
```

## SWE-Bench & Terminal-Bench Training

### Real-World Benchmarks

| Benchmark | Task | Size | Reward |
|-----------|------|------|--------|
| **SWE-Bench Lite** | Fix GitHub issues | 300 | Patch similarity + format |
| **SWE-Bench Verified** | Fix GitHub issues | 500 | Patch similarity + format |
| **Terminal-Bench** | Terminal tasks | 100+ | Command execution + safety |

### Train on SWE-Bench

```bash
# SWE-Bench Lite (easier, 300 samples)
python train_bench_grpo.py --benchmark swe-lite --max_samples 100

# SWE-Bench Verified (harder, human-verified)
python train_bench_grpo.py --benchmark swe-verified --max_samples 100
```

### Train on Terminal-Bench

```bash
python train_bench_grpo.py --benchmark terminal --max_samples 25
```

### Reward Signals

**SWE-Bench rewards:**
- `format` (0.3) — Valid unified diff patch
- `similarity` (0.5) — Similar to gold patch
- `files` (0.2) — Targets correct files

**Terminal-Bench rewards:**
- `format` (0.2) — Extractable commands
- `safety` (0.3) — No dangerous patterns (rm -rf /, etc.)
- `execution` (0.5) — Commands run successfully

## Tips

1. **Start small**: Test with `max_samples: 100` first
2. **Monitor rewards**: Watch for reward hacking
3. **Adjust weights**: Tune reward weights in `config.yaml`
4. **Use Wandb**: Set `report_to: "wandb"` for logging

## References

- [Unsloth FP8 RL Docs](https://docs.unsloth.ai/new/fp8-reinforcement-learning)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TorchAO FP8](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md)

## License

MIT
