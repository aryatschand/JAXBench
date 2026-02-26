# JAXBench

A benchmark suite for JAX and TPU kernel optimization. Contains operator-level workloads from LLM architectures — vanilla JAX baselines that Pallas optimization agents try to beat.

All benchmarks run on **TPU v6e-1** with **JAX 0.6.2**. Full results in [`benchmarks.json`](benchmarks.json).

## Benchmark Suites

| Suite | Workloads | Source | Description |
|-------|-----------|--------|-------------|
| [jaxkernelbench/](jaxkernelbench/) | 200 | [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | LLM-translated PyTorch→JAX operators |
| [real_workloads/](real_workloads/) | 29 | [MaxText](https://github.com/AI-Hypercomputer/maxtext) | Hand-written ops from 6 modern LLM families |
| [tokamax/](tokamax/) | 12 | [openxla/tokamax](https://github.com/openxla/tokamax) | TPU kernel benchmarks across 6 operations |

---

### real_workloads — 29 LLM operator benchmarks

Production-scale operator shapes from 6 modern LLM architectures.

| Workload | Model | Operator | Time (ms) | TFLOPS |
|----------|-------|----------|----------:|-------:|
| llama3_8b_gqa | Llama-3.1-8B | gqa_attention | 0.89 | 77.49 |
| llama3_70b_gqa | Llama-3.1-70B | gqa_attention | 1.66 | 83.01 |
| llama3_405b_gqa | Llama-3.1-405B | gqa_attention | 3.25 | 84.50 |
| llama3_8b_swiglu | Llama-3.1-8B | swiglu_mlp | 1.11 | 648.27 |
| llama3_70b_swiglu | Llama-3.1-70B | swiglu_mlp | 4.10 | 703.70 |
| llama3_405b_swiglu | Llama-3.1-405B | swiglu_mlp | 14.57 | 735.94 |
| llama3_8b_rope | Llama-3.1-8B | rope | 0.15 | 0.22 |
| llama3_70b_rope | Llama-3.1-70B | rope | 0.20 | 0.34 |
| llama3_8b_rmsnorm | Llama-3.1-8B | rmsnorm | 0.15 | 0.22 |
| llama3_70b_rmsnorm | Llama-3.1-70B | rmsnorm | 0.18 | 0.37 |
| llama3_8b_token_embed | Llama-3.1-8B | token_embed | 0.20 | 5.47 |
| gemma3_4b_sliding_window | Gemma-3-4B | sliding_window_attention | 0.37 | 92.85 |
| gemma3_12b_sliding_window | Gemma-3-12B | sliding_window_attention | 0.66 | 104.77 |
| gemma3_27b_sliding_window | Gemma-3-27B | sliding_window_attention | 1.13 | 68.57 |
| gemma3_4b_global_attn | Gemma-3-4B | global_attention | 0.37 | 94.04 |
| gemma3_12b_global_attn | Gemma-3-12B | global_attention | 0.66 | 104.77 |
| gemma3_27b_global_attn | Gemma-3-27B | global_attention | 1.13 | 68.40 |
| mixtral_8x7b_moe | Mixtral-8x7B | sparse_moe | 8.32 | 173.41 |
| mixtral_8x22b_moe | Mixtral-8x22B | sparse_moe | 13.85 | 178.70 |
| deepseek_v3_mla | DeepSeek-V3-671B | mla_attention | 4.44 | 265.75 |
| deepseek_v3_yarn_rope | DeepSeek-V3-671B | yarn_rope | 1.35 | 0.20 |
| qwen3_8b_gqa | Qwen3-8B | gqa_attention | 0.89 | 76.88 |
| qwen3_14b_gqa | Qwen3-14B | gqa_attention | 1.11 | 77.65 |
| qwen3_8b_swiglu | Qwen3-8B | swiglu_mlp | 1.09 | 663.81 |
| qwen3_moe_30b_moe | Qwen3-MoE-30B-A3B | sparse_moe_shared | 7.56 | 450.11 |
| llama4_scout_gqa | Llama-4-Scout-109B | gqa_attention | 1.27 | 80.86 |
| llama4_scout_moe | Llama-4-Scout-109B | sparse_moe_top1 | 12.17 | 677.76 |
| llama4_maverick_moe | Llama-4-Maverick-400B | sparse_moe_top1 | 49.91 | 660.91 |
| llama4_scout_rope | Llama-4-Scout-109B | rope | 0.18 | 0.41 |

### tokamax — 12 TPU kernel benchmarks

Production model configurations from [openxla/tokamax](https://github.com/openxla/tokamax).

| Workload | Model | Operator | Time (ms) | TFLOPS |
|----------|-------|----------|----------:|-------:|
| mixtral_8x7b_attention | Mixtral-8x7B | gqa_causal_attention | 0.87 | 79.05 |
| deepseek2_16b_attention | DeepSeek-V2-Lite-16B | causal_attention | 0.17 | 74.85 |
| mixtral_8x7b_ragged_dot | Mixtral-8x7B | ragged_dot | 1.37 | 702.08 |
| llama3_8b_cross_entropy | Llama-3.1-8B | fused_cross_entropy | 7.69 | 559.59 |
| qwen3_8b_cross_entropy | Qwen3-8B | fused_cross_entropy | 8.66 | 589.24 |
| gemma3_4b_cross_entropy | Gemma3-4B | fused_cross_entropy | 9.74 | 564.98 |
| llama3_8b_swiglu | Llama-3.1-8B | swiglu | 1.08 | 667.86 |
| llama3_70b_swiglu | Llama-3.1-70B | swiglu | 4.09 | 705.12 |
| llama3_8b_rmsnorm | Llama-3.1-8B | rmsnorm | 0.15 | 0.22 |
| llama3_70b_rmsnorm | Llama-3.1-70B | rmsnorm | 0.18 | 0.38 |
| alphafold_384_triangle_mult | AlphaFold2 | triangle_mult_incoming | 0.29 | 50.81 |
| alphafold_768_triangle_mult | AlphaFold2 | triangle_mult_outgoing | 1.32 | 65.87 |

### jaxkernelbench — 200 LLM-translated operators

Automatically translated from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) using Claude. Each task benchmarks JAX vs PyTorch/XLA on TPU.

**Level 1 — 100 single operators** (matmul, conv, activations, norms, pooling, reductions, losses)

| # | Operator | JAX (ms) | PyTorch/XLA (ms) | Speedup |
|---|----------|----------:|------------------:|--------:|
| 1 | Square matrix multiplication | 0.28 | 0.37 | 1.32x |
| 10 | 3D tensor matrix multiplication | 0.25 | 0.38 | 1.52x |
| 33 | BatchNorm | 7.34 | 11.23 | 1.53x |
| 35 | GroupNorm | 0.95 | 3.12 | 3.29x |
| 41 | Max Pooling 1D | 0.29 | 25.30 | 87.01x |
| 43 | Max Pooling 3D | 125.61 | 1581.79 | 12.59x |
| 96 | HuberLoss | 0.20 | 0.32 | 1.62x |
| 97 | ScaledDotProductAttention | 10.42 | 19.64 | 1.89x |

*Median speedup: 1.00x — JAX and PyTorch/XLA produce near-identical XLA HLO for most single ops. Notable wins on pooling, GroupNorm, and fused attention. Full 100 entries in `benchmarks.json`.*

**Level 2 — 100 fused operators** (multi-op compositions: Conv+BN+ReLU, Matmul+Norm+Activation, etc.)

| # | Operator | JAX (ms) | PyTorch/XLA (ms) | Speedup |
|---|----------|----------:|------------------:|--------:|
| 9 | Matmul+Subtract+Multiply+ReLU | 0.17 | 0.53 | 3.18x |
| 11 | ConvTranspose2d+BN+Tanh+MaxPool+GroupNorm | 1.20 | 3.98 | 3.33x |
| 19 | ConvTranspose2d+GELU+GroupNorm | 12.31 | 52.76 | 4.29x |
| 23 | Conv3d+GroupNorm+Mean | 0.68 | 6.98 | 10.19x |
| 56 | Matmul+Sigmoid+Sum | 0.13 | 3.84 | 30.34x |
| 59 | Matmul+Swish+Scaling | 0.14 | 3.89 | 27.05x |
| 75 | Gemm+GroupNorm+Min+BiasAdd | 0.19 | 2.41 | 12.89x |
| 92 | Conv2d+GroupNorm+Tanh+HardSwish+ResidualAdd+LogSumExp | 1.84 | 7.75 | 4.21x |

*Median speedup: 1.07x — Larger wins on fused sequences with GroupNorm, multi-activation chains, and Matmul+activation patterns. Full 100 entries in `benchmarks.json`.*

---

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
├── jaxkernelbench/       # 200 LLM-translated PyTorch→JAX operators
│   ├── level1/           # 100 single operators
│   └── level2/           # 100 fused operators
├── real_workloads/       # 29 hand-written ops from 6 LLM architectures
│   ├── llama3/           # 11 workloads
│   ├── llama4/           # 4 workloads
│   ├── gemma3/           # 6 workloads
│   ├── qwen3/            # 4 workloads
│   ├── mixtral/          # 2 workloads
│   ├── deepseek_v3/      # 3 workloads (deepseek_v3_moe OOM on single TPU)
│   └── run_benchmarks.py
├── tokamax/              # 12 workloads from openxla/tokamax
│   ├── attention/        # 2 workloads
│   ├── ragged_dot/       # 1 workload
│   ├── cross_entropy/    # 3 workloads
│   ├── gated_linear_unit/# 2 workloads
│   ├── layer_norm/       # 2 workloads
│   ├── triangle_mult/    # 2 workloads
│   └── run_benchmarks.py
├── torch_to_jax/         # Translation pipeline (PyTorch → JAX)
├── benchmarks.json       # All benchmark results (TPU v6e-1)
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

## License

MIT
