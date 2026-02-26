# JAXBench

A benchmark suite for JAX and TPU kernel optimization. Contains operator-level workloads from LLM architectures — vanilla JAX baselines that Pallas optimization agents try to beat.

## Benchmark Suites

### jaxkernelbench/ — 200+ LLM-translated PyTorch→JAX operators

Automatically translated from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) using LLMs (Claude/Gemini). Each task is a self-contained JAX module with matching PyTorch reference.

| Level | Description | Tasks |
|-------|-------------|-------|
| Level 1 | Single operators (matmul, conv, activations, norms) | 101 |
| Level 2 | Fused operators (Conv+ReLU, Conv+BN+GELU) | 101 |

### real_workloads/ — 30 hand-written ops from 6 modern LLM families

Production-scale operator shapes extracted from [MaxText](https://github.com/AI-Hypercomputer/maxtext). Each file is self-contained with `CONFIG`, `create_inputs()`, `workload()`, and `benchmark()`.

| Model | Workloads | Operators |
|-------|-----------|-----------|
| Llama 3.1 (8B/70B/405B) | 11 | GQA attention, SwiGLU, RoPE, RMSNorm, Token embed |
| Llama 4 (Scout/Maverick) | 4 | GQA attention, Sparse MoE top-1, RoPE |
| Gemma 3 (4B/12B/27B) | 6 | Sliding window attention, Global attention |
| Qwen 3 (8B/14B/MoE-30B) | 4 | GQA attention, SwiGLU, Sparse MoE |
| Mixtral (8x7B/8x22B) | 2 | Sparse MoE top-2 |
| DeepSeek V3 (671B) | 3 | MLA attention, YaRN RoPE, MoE w/ shared experts |

```bash
# Run a single workload locally (prints JSON)
python real_workloads/llama3/llama3_8b_gqa.py

# Benchmark all on TPU
python -m real_workloads.run_benchmarks --tpu v6e-1 --keep-tpu
```

### tokamax/ — 12 workloads from openxla/tokamax across 6 operations

Production model configurations from [openxla/tokamax](https://github.com/openxla/tokamax). Covers attention, MoE routing, loss computation, MLP activations, normalization, and protein structure operations.

| Operation | Workloads | Models |
|-----------|-----------|--------|
| attention | 2 | Mixtral 8x7B (GQA), DeepSeek-V2 16B |
| ragged_dot | 1 | Mixtral 8x7B (grouped matmul for MoE) |
| cross_entropy | 3 | Llama 3.1 8B, Qwen 3 8B, Gemma 3 4B |
| gated_linear_unit | 2 | Llama 3.1 8B/70B (SwiGLU) |
| layer_norm | 2 | Llama 3.1 8B/70B (RMSNorm) |
| triangle_mult | 2 | AlphaFold2 384/768 (incoming/outgoing) |

```bash
# Run a single workload locally
python tokamax/attention/mixtral_8x7b_attention.py

# Benchmark all on TPU
python -m tokamax.run_benchmarks --tpu v6e-1 --keep-tpu
```

## Workload File Format

Every workload file follows the same template and runs standalone:

```python
CONFIG = {
    'name': 'model_op',
    'model': 'Model-Name',
    'operator': 'op_type',
    # ... shape parameters
}

def create_inputs(dtype=jnp.bfloat16):
    """Create input tensors with deterministic seed."""
    ...

def workload(*inputs):
    """Vanilla JAX implementation — the baseline to optimize."""
    ...

def benchmark(num_warmup=5, num_iters=100):
    """Self-contained benchmark, returns JSON-serializable dict."""
    ...

if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
```

## Repo Structure

```
JAXBench/
├── jaxkernelbench/       # 200+ LLM-translated PyTorch→JAX operators
│   ├── level1/           # 101 tasks
│   └── level2/           # 101 tasks
├── real_workloads/       # 30 hand-written ops from 6 LLM architectures
│   ├── llama3/           # 11 workloads
│   ├── llama4/           # 4 workloads
│   ├── gemma3/           # 6 workloads
│   ├── qwen3/            # 4 workloads
│   ├── mixtral/          # 2 workloads
│   ├── deepseek_v3/      # 3 workloads
│   └── run_benchmarks.py
├── tokamax/              # 12 workloads from openxla/tokamax across 6 ops
│   ├── attention/        # 2 workloads
│   ├── ragged_dot/       # 1 workload
│   ├── cross_entropy/    # 3 workloads
│   ├── gated_linear_unit/# 2 workloads
│   ├── layer_norm/       # 2 workloads
│   ├── triangle_mult/    # 2 workloads
│   └── run_benchmarks.py
├── torch_to_jax/         # Translation pipeline (PyTorch → JAX)
├── README.md
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

## Translation Pipeline

The `torch_to_jax/` module provides an LLM-powered 4-stage pipeline for translating PyTorch operators to JAX:

```
Stage 1: LLM Translation    → Generate JAX code from PyTorch (Sonnet first, Opus for retries)
Stage 2: Compilation Check   → Verify JAX code compiles on TPU via JIT
Stage 3: Correctness Check   → Compare outputs with identical inputs/weights (seed=42)
Stage 4: Performance Bench   → Measure JAX vs PyTorch/XLA timing on TPU
```

```bash
python -m torch_to_jax.run --level 1 --all --keep-tpu
python -m torch_to_jax.run --level 2 --task-ids "1,5,10" --keep-tpu
```

## Setup

```bash
pip install -r requirements.txt
```

### GCP / TPU

Place your GCP service account JSON at `credentials.json` (or set `GCP_CREDENTIALS_FILE`). SSH key at `~/.ssh/id_rsa_tpu`.

### LLM Providers

```bash
# AWS Bedrock (for Claude)
export AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." AWS_REGION="us-east-2"

# Gemini (optional)
export GEMINI_API_KEY="..."
```

## TPU Configuration

Working package versions on TPU v6e:

| Package | Version |
|---------|---------|
| jax | 0.6.2 |
| torch | 2.9.0+cpu |
| torch_xla | 2.9.0 |

## License

MIT
