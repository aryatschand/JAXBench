# JAXBench

A benchmark suite for JAX and TPU kernel optimization. Contains operator-level workloads from LLM architectures — vanilla JAX baselines and upstream Pallas kernels for optimization evaluation.

All benchmarks run on **TPU v6e-1** with **JAX 0.6.2**. Full results in [`benchmarks.json`](benchmarks.json).

## Benchmark Suites

| Suite | Workloads | Source | Description |
|-------|-----------|--------|-------------|
| [priority_kernels/](priority_kernels/) | 10 | Hand-written | Core LLM operator baselines (attention, MLP, MoE, SSM) |
| [pallas_kernels/](pallas_kernels/) | 6 | [JAX 0.6.2 Pallas](https://github.com/jax-ml/jax) | Upstream Pallas TPU kernels for optimization |
| [jaxkernelbench/](jaxkernelbench/) | 200 | [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | LLM-translated PyTorch→JAX operators |

---

### priority_kernels — 10 core LLM operator benchmarks

Production-scale operator shapes from modern LLM architectures covering attention variants, MLPs, MoE, cross-entropy, and state-space models.

| Workload | Model | Operator | Time (ms) | TFLOPS |
|----------|-------|----------|----------:|-------:|
| gemm_llama70b | Llama-3.1-70B | dense_matmul | 5.44 | 707.81 |
| flash_attention_baseline | Baseline-MHA | causal_mha | 2.83 | 97.16 |
| llama3_405b_gqa | Llama-3.1-405B | gqa_attention | 3.26 | 84.42 |
| llama3_70b_swiglu | Llama-3.1-70B | swiglu_mlp | 4.07 | 708.40 |
| mixtral_8x7b_moe | Mixtral-8x7B | sparse_moe | 8.31 | 173.74 |
| llama3_8b_cross_entropy | Llama-3.1-8B | fused_cross_entropy | 7.65 | 562.51 |
| deepseek_v3_mla | DeepSeek-V3-671B | mla_attention | 4.46 | 264.00 |
| mixtral_8x7b_ragged_dot | Mixtral-8x7B | ragged_dot | 1.36 | 705.26 |
| retnet_6_7b_retention | RetNet-6.7B | multi_scale_retention | 0.52 | 132.51 |
| mamba2_2_7b_ssd | Mamba-2-2.7B | state_space_duality | 1.80 | 38.13 |

### pallas_kernels — 6 upstream Pallas TPU kernels

Kernels copied verbatim from JAX 0.6.2 with pure-JAX references for correctness verification.

| Kernel | Model | Time (ms) |
|--------|-------|----------:|
| matmul | Llama-3.1-70B | 22.7 |
| flash_attention | Llama-3.1-70B | 6.2 |
| splash_attention | Llama-3.1-70B | 5.9 |
| paged_attention | Llama-3.1-70B | 1.6 |
| ragged_paged_attention | Llama-3.1-70B | 1.6 |
| megablox_gmm | Qwen3-235B | 21.3 |

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
├── priority_kernels/     # 10 core LLM operator baselines
├── pallas_kernels/       # 6 upstream Pallas TPU kernels from JAX 0.6.2
│   ├── jax_references/   # Pure-JAX references for correctness checks
│   └── check_references.py
├── jaxkernelbench/       # 200 LLM-translated PyTorch→JAX operators
│   ├── level1/           # 100 single operators
│   └── level2/           # 100 fused operators
├── torch_to_jax/         # Translation pipeline (PyTorch → JAX)
├── benchmarks.json       # Benchmark results (TPU v6e-1)
├── README.md
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
