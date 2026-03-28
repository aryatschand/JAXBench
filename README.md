# JAXBench

A benchmark suite for JAX and TPU kernel optimization. Contains operator-level workloads from LLM architectures — vanilla JAX baselines and upstream Pallas kernels for optimization evaluation.

All benchmarks run on **TPU v6e-1** with **JAX 0.6.2**. Full results in [`benchmarks.json`](benchmarks.json).

## Benchmark Suites

| Suite | Workloads | Variants | Description |
|-------|-----------|----------|-------------|
| [priority_kernels/](priority_kernels/) | 17 | baseline + optimized + pallas | Core LLM operators with multi-tier optimization benchmarks |
| [pallas_kernels/](pallas_kernels/) | 6 | — | Upstream Pallas TPU kernels from JAX 0.6.2 (reference source) |
| [jaxkernelbench/](jaxkernelbench/) | 200 | — | LLM-translated PyTorch→JAX operators |

---

### priority_kernels — 17 core LLM operator benchmarks

Each workload has up to 3 implementation variants for optimization comparison:
- **baseline** — vanilla JAX (the unoptimized reference)
- **optimized** — JAX with library imports (e.g., `jax.nn.dot_product_attention`) and tuned parameters
- **pallas** — upstream Pallas TPU kernel from JAX 0.6.2

| Workload | Model | Variants | Baseline (ms) | TFLOPS |
|----------|-------|----------|----------:|-------:|
| [gemm](priority_kernels/gemm/) | Llama-3.1-70B | baseline, pallas | 5.44 | 707.81 |
| [flash_attention](priority_kernels/flash_attention/) | Baseline-MHA | baseline, optimized, pallas | 2.83 | 97.16 |
| [gqa_attention](priority_kernels/gqa_attention/) | Llama-3.1-405B | baseline, optimized | 3.26 | 84.42 |
| [swiglu_mlp](priority_kernels/swiglu_mlp/) | Llama-3.1-70B | baseline | 4.07 | 708.40 |
| [sparse_moe](priority_kernels/sparse_moe/) | Mixtral-8x7B | baseline, optimized, pallas | 8.31 | 173.74 |
| [cross_entropy](priority_kernels/cross_entropy/) | Llama-3.1-8B | baseline | 7.65 | 562.51 |
| [mla_attention](priority_kernels/mla_attention/) | DeepSeek-V3-671B | baseline, optimized | 4.46 | 264.00 |
| [ragged_dot](priority_kernels/ragged_dot/) | Mixtral-8x7B | baseline, optimized | 1.36 | 705.26 |
| [retnet_retention](priority_kernels/retnet_retention/) | RetNet-6.7B | baseline | 0.52 | 132.51 |
| [mamba2_ssd](priority_kernels/mamba2_ssd/) | Mamba-2-2.7B | baseline | 1.80 | 38.13 |
| [rms_norm](priority_kernels/rms_norm/) | Llama-3.1-70B | baseline | — | — |
| [paged_attention](priority_kernels/paged_attention/) | Llama-3.1-70B | baseline, optimized, pallas | — | — |
| [sparse_attention](priority_kernels/sparse_attention/) | Llama-3.1-70B | baseline, optimized, pallas | — | — |
| [flex_attention](priority_kernels/flex_attention/) | Llama-3.1-70B | baseline, optimized | — | — |
| [triangle_multiplication](priority_kernels/triangle_multiplication/) | AlphaFold2 | baseline | — | — |
| [ragged_paged_attention](priority_kernels/ragged_paged_attention/) | Llama-3.1-70B | baseline, pallas | — | — |
| [megablox_gmm](priority_kernels/megablox_gmm/) | Qwen3-235B | baseline, pallas | — | — |

### pallas_kernels — 6 upstream Pallas TPU kernels

Kernels copied verbatim from JAX 0.6.2 with pure-JAX references for correctness verification. These serve as the source for `pallas.py` variants in priority_kernels.

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
├── priority_kernels/           # 17 core LLM operators with multi-tier benchmarks
│   ├── gemm/                   # Each workload folder contains:
│   │   ├── baseline.py         #   Vanilla JAX implementation
│   │   ├── optimized.py        #   JAX + library imports (where applicable)
│   │   ├── pallas.py           #   Upstream Pallas kernel (where available)
│   │   └── README.md           #   Kernel overview + benchmark results
│   ├── flash_attention/
│   ├── gqa_attention/
│   ├── ... (17 workload folders)
│   ├── run_benchmarks.py       # TPU benchmark orchestrator
│   └── results.json
├── pallas_kernels/             # 6 upstream Pallas TPU kernels from JAX 0.6.2
│   ├── jax_references/         # Pure-JAX references for correctness checks
│   └── check_references.py
├── jaxkernelbench/             # 200 LLM-translated PyTorch→JAX operators
│   ├── level1/                 # 100 single operators
│   └── level2/                 # 100 fused operators
├── torch_to_jax/               # Translation pipeline (PyTorch → JAX)
├── benchmarks.json             # jaxkernelbench results (TPU v6e-1)
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
