# JAXBench: TPU Kernel Benchmark Suite

A benchmark suite of 50 JAX workloads for evaluating kernel optimization on Google Cloud TPU v6e (Trillium). The suite includes production LLM operators from MaxText and fused operator sequences from KernelBench, with both vanilla JAX baselines and hand-optimized Pallas TPU kernels.

## Quick Start

Requires **Python 3.11** with **JAX 0.9.2**. On `v2-alpha-tpuv6e` TPU VMs, install first:

```bash
python3.11 -m pip install 'jax[tpu]==0.9.2' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

```bash
# Run a single workload
PJRT_DEVICE=TPU python3.11 benchmark/1p_Flash_Attention/baseline.py

# Run Pallas-optimized variant (where available)
PJRT_DEVICE=TPU python3.11 benchmark/1p_Flash_Attention/optimized.py

# Run full benchmark suite
PJRT_DEVICE=TPU python3.11 benchmark/run_all.py
```

## Hardware Platform

All benchmarks run on a single **TPU v6e-1** chip (codename Trillium):

| Spec | Value |
|---|---|
| Chip | Google TPU v6e, 1 chip |
| MXU | 2× Matrix Multiply Units, 256×256 systolic arrays |
| Peak bf16 TFLOPS | **918 TFLOPS** per chip (2× MXU × 459 TFLOPS each) |
| HBM | 16 GB HBM2e |
| Software | JAX 0.6.2, PJRT TPU runtime |

## Workload Suite (50 Workloads)

### Priority Kernels (1p–17p): Production LLM Operators

17 operators extracted from real model architectures (Llama-3.1, DeepSeek-V3, Mixtral, Mamba-2, RetNet, AlphaFold2). These represent the compute-critical operators in modern LLM inference and training:

| # | Workload | Model Source | Operator Type |
|---|---|---|---|
| 1p | Flash Attention | Llama-3.1-70B | Multi-head causal attention |
| 2p | GQA Attention | Llama-3.1-405B | Grouped-query attention (128 Q / 8 KV heads) |
| 3p | MLA Attention | DeepSeek-V3-671B | Multi-head latent attention with low-rank KV |
| 4p | Sparse (Splash) Attention | Llama-3.1-70B | Splash attention with GQA + causal mask |
| 5p | Flex Attention | Llama-3.1-70B | Flexible attention with score modification |
| 6p | Paged Attention | Llama-3.1-70B | Paged KV-cache decode attention |
| 7p | Ragged Paged Attention | Llama-3.1-70B | Mixed prefill+decode with variable-length sequences |
| 8p | GEMM | Llama-3.1-70B | Dense matmul (8192×8192 × 8192×28672) |
| 9p | SwiGLU MLP | Llama-3.1-70B | Gated MLP with SiLU activation |
| 10p | Sparse MoE | Mixtral-8×7B | Top-2 sparse mixture of experts |
| 11p | Megablox GMM | Qwen3-235B-A22B | Grouped matrix multiply for MoE |
| 12p | RMSNorm | Llama-3.1-70B | Root mean square layer normalization |
| 13p | Cross-Entropy | Llama-3.1-8B | Fused linear + softmax cross-entropy loss |
| 14p | Ragged Dot | Mixtral-8×7B | Grouped matmul (batched per-expert) |
| 15p | RetNet Retention | RetNet-6.7B | Multi-scale retention with exponential decay |
| 16p | Mamba-2 SSD | Mamba-2-2.7B | State space duality (selective scan) |
| 17p | Triangle Multiplication | AlphaFold2 | Outgoing triangle multiplicative update |

8 of these have hand-optimized **Pallas TPU kernels** (`optimized.py`) from the upstream JAX Pallas ops library, with block sizes tuned via grid search on v6e.

### KernelBench Fused Operators (18k–50k): Synthetic Fusion Targets

33 fused operator sequences translated from KernelBench Level 2 (PyTorch → JAX). Each combines a matrix multiplication or convolution with elementwise operations (activations, normalization, pooling). These represent kernel fusion opportunities:

- **Matmul + activations** (20 workloads): GEMM/Matmul fused with ReLU, GELU, Mish, Swish, Sigmoid, Hardtanh, Softmax, Dropout, etc.
- **Matmul + normalization** (5 workloads): GEMM fused with BatchNorm, GroupNorm, LayerNorm
- **Convolution chains** (8 workloads): Conv2D/Conv3D with activations, normalization, pooling

## Workload Interface

Every workload follows a standardized interface:

```python
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'workload_name',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
    # ... operator-specific parameters
}

def create_inputs(dtype=jnp.bfloat16):
    """Returns input tensors for the workload."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4096, 8192), dtype=dtype)
    weight = jnp.zeros((8192, 8192), dtype=dtype)
    bias = jnp.zeros(8192, dtype=dtype)
    return x, weight, bias

def workload(x, weight, bias):
    """The computation to benchmark."""
    with jax.named_scope('bench_kernel'):
        x = jnp.matmul(x, weight) + bias
        return jax.nn.relu(x)
```

- **`CONFIG`**: Dictionary of hyperparameters (dimensions, model name, operator type).
- **`create_inputs(dtype)`**: Creates input tensors. The benchmark runner calls this with `dtype=jnp.bfloat16` to run all workloads in bfloat16 precision.
- **`workload(*inputs)`**: The computation to benchmark, wrapped in `jax.named_scope('bench_kernel')` for profiler annotation. The runner JIT-compiles this function via `jax.jit()`.
- **`get_flops()`** (optional): Manual FLOP count for workloads where XLA's `cost_analysis()` returns 0 (Pallas kernels, eager workloads).

Two workloads are marked with `_skip_jit = True` (Megablox GMM baseline, Ragged Paged Attention baseline) because they use data-dependent Python control flow that cannot be JIT-compiled. These run eagerly.

For workloads with Pallas-optimized variants, `optimized.py` follows the same interface and additionally defines `TUNED_PARAMS` — a dictionary of block sizes tuned for TPU v6e via grid search.

## Problem Sizes

All workloads use production-scale dimensions chosen to be representative of real LLM training and inference. Matmul-heavy workloads are sized to saturate the TPU's MXU:

| Category | Typical Dimensions | MXU Utilization |
|---|---|---|
| Matmul-heavy fused ops | (4096, 8192) × (8192, 8192) bf16 | 60–95% |
| Priority GEMM/MLP | Llama-70B: (8192, 8192) × (8192, 28672) | 75–80% |
| Attention baselines (vanilla) | B=4, H=64–128, S=4096, D=128 | 10–25% (memory-bound) |
| Attention optimized (Pallas) | Same dims, tiled algorithm | 28–35% |
| Convolutions | Batch=64–128, spatial 128×128 | 2–18% (not MXU-based) |
| MoE / Grouped matmul | Mixtral/Qwen3 production dims | 8–84% |

Attention baselines show low MXU utilization because vanilla dot-product attention materializes the full S×S attention score matrix, making the computation memory-bandwidth-bound rather than compute-bound. The Pallas-optimized flash/splash attention variants use tiled algorithms that avoid materializing this matrix, significantly improving both speed and utilization.

## Timing Methodology: Device-Side Profiling with jax.profiler

### Why Not Wall-Clock Timing

Standard wall-clock timing (`time.perf_counter()` + `block_until_ready()`) includes host-side overhead: Python dispatch, JAX runtime scheduling, and synchronization latency. For short kernels (<1ms), this overhead can be 10–20% of the measured time, distorting results. For accurate kernel-level benchmarking, we need to measure only the device-side execution.

### Protocol

For each of the 58 benchmark runs (50 baselines + 8 optimized):

1. **Module loading**: Dynamically import the workload module. Modules are registered in `sys.modules` before execution to support `dataclasses` resolution in Pallas upstream kernels.
2. **Input creation**: Call `create_inputs(dtype=jnp.bfloat16)` to generate input tensors.
3. **JIT compilation**: Compile via `jax.jit(workload_fn)` (skipped for `_skip_jit` eager workloads).
4. **Warmup**: 5 untimed iterations to warm the JIT cache and stabilize the TPU pipeline.
5. **Wall-clock measurement**: 50 timed iterations using `time.perf_counter()` bracketing `jit_fn(*inputs)` + `block_until_ready()`. Recorded for comparison but not used as the primary metric.
6. **Profiler measurement**: 50 iterations executed under `jax.profiler.trace()`, which captures an XProf-compatible trace in Perfetto format.
7. **Trace parsing**: The resulting `perfetto_trace.json.gz` is decompressed and parsed to extract per-iteration device execution times (see Event Matching below).
8. **Statistics**: Median (primary), mean, standard deviation, and minimum are computed over the 50 profiled iterations. All times reported in milliseconds.

### Event Matching in Perfetto Traces

`jax.profiler.trace()` generates a Perfetto-compatible JSON trace containing a hierarchy of events at both host and device level. We extract **`jit_<function_name>(<hash>)`** events, which represent the outermost device-side execution wrapper for each JIT-compiled function call.

These `jit_*()` events capture the **total device execution time** for the complete computation in a single iteration, including all sub-operations regardless of how XLA or Pallas decomposes them:

| Workload Type | Trace Event Structure | Example |
|---|---|---|
| **XLA single fusion** | `jit_workload(hash)` → `fusion` | GEMM: 1 fused matmul kernel |
| **XLA multi-fusion** | `jit_workload(hash)` → `fusion` + `fusion.2` + `fusion.3` + ... | Vanilla attention: QK^T, softmax, mask, AV as 4 separate fusions |
| **Pallas kernel** | `jit_workload(hash)` → `<kernel_name>.1` | Flash attention: single tiled Pallas kernel |

The `jit_*()` event captures the **total device time** for the complete computation per iteration, regardless of how XLA or Pallas decomposes it internally.

For the 2 eager workloads that cannot be JIT-compiled (`_skip_jit=True`), individual `jit_*()` events from each JAX primitive call (e.g., `jit_dot_general(...)`) are collected and grouped into per-iteration batches by dividing the total event count by the number of iterations. This captures the sum of all device ops per iteration.

### Named Scope Annotation

Every `workload()` function body is wrapped with `jax.named_scope('bench_kernel')`:

```python
def workload(x, weight, bias):
    with jax.named_scope('bench_kernel'):
        x = jnp.matmul(x, weight) + bias
        return jax.nn.relu(x)
```

This annotates the corresponding XLA HLO operations with the `bench_kernel/` prefix. While `named_scope` does not create its own timing event in the Perfetto trace (timing comes from `jit_*()` events), the annotation is visible in TensorBoard/XProf profile visualization and serves to document that the profiled region corresponds exactly to the benchmarked computation.

### Difference from Wall-Clock Timing

The device profiler consistently reports lower times than wall-clock, with the difference being the host dispatch overhead:

| Workload | Wall-Clock | Device Profiler | Host Overhead |
|---|---|---|---|
| GEMM baseline (5.3ms) | 5.48ms | 5.32ms | 3.0% |
| GEMM optimized / Pallas (5.4ms) | 5.58ms | 5.41ms | 3.1% |
| Flash Attention baseline (23ms) | 23.13ms | 22.96ms | 0.7% |
| Short matmul fused ops (~0.7ms) | 0.76ms | 0.63ms | 17.1% |

Host overhead is most significant for short-running kernels, where the fixed dispatch cost is a larger fraction of total time. The profiler removes this artifact, giving more accurate and reproducible results.

## FLOP Counting

FLOP counts are used to compute achieved TFLOPS and MXU utilization percentage. Two methods are used:

### XLA cost_analysis() (46 workloads)

For JIT-compiled vanilla JAX workloads, FLOPs are extracted from the compiled XLA HLO program:

```python
compiled = jax.jit(workload_fn).lower(*inputs).compile()
cost = compiled.cost_analysis()
flops = cost[0].get('flops', 0)  # Returns list for TPU
```

This counts all floating-point operations in the XLA graph, including matmul, elementwise, reductions, and data movement operations. The count may slightly overestimate MXU-specific FLOPs because it includes non-MXU operations (elementwise, softmax) in the total.

### Manual get_flops() (4 workloads)

For workloads where XLA returns 0 FLOPs — specifically Pallas kernels (which bypass XLA's cost accounting) and `_skip_jit` eager workloads (which are never compiled to XLA) — the workload module defines a `get_flops()` function with the analytical formula:

```python
# Example: GEMM
def get_flops():
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    return 2 * M * K * N  # Standard matmul FLOP count
```

The benchmark runner checks for `get_flops()` first and falls back to XLA `cost_analysis()` if not present.

## Utilization Calculation

MXU utilization measures how efficiently the workload uses the TPU's matrix multiply units:

```
TFLOPS_achieved = FLOP_count / median_time_seconds / 1e12
MXU_utilization = TFLOPS_achieved / 918.0 × 100%
```

Where 918.0 TFLOPS is the TPU v6e single-chip bf16 peak (source: [Google Cloud TPU v6e documentation](https://cloud.google.com/tpu/docs/v6e)).

**Interpretation**: MXU utilization reflects how much of the peak matrix multiply throughput is being used. Workloads that are memory-bandwidth-bound (attention with materialized S×S scores, normalization layers) or that don't use MXU (convolutions, elementwise operations) will show low MXU utilization even at optimal performance. This is expected — the metric is specific to matrix unit efficiency, not overall hardware utilization.

## Pallas Kernel Optimization

8 priority kernels have hand-optimized Pallas TPU kernels (`optimized.py`). These use `jax.experimental.pallas` with TPU-specific primitives:

- **`pl.BlockSpec`**: Partitions input tensors into tiles for block-wise processing
- **`pltpu.PrefetchScalarGridSpec`**: Defines the compute grid with prefetching for overlapping memory access and computation
- **`pltpu.VMEM`**: On-chip VMEM scratch buffers for float32 accumulation (matmul tiles are accumulated in float32, then cast to bf16 for output)
- **`pltpu.CompilerParams(dimension_semantics=...)`**: Specifies `"parallel"` dimensions (tiled independently) vs `"arbitrary"` dimensions (reduction axes)

### Block Size Tuning

Each Pallas kernel has a `TUNED_PARAMS` dictionary with block sizes. These were tuned via exhaustive grid search over the parameter space on TPU v6e, using the same profiler-based timing methodology as the main benchmark. The tuning script (`tune_pallas.py`) evaluated 203 configurations total across 8 kernels:

| Kernel | Parameters Tuned | Search Space | Best Config | Speedup |
|---|---|---|---|---|
| Megablox GMM | tiling [tm, tk, tn] | 27 configs | [256, 1024, 1024] | **2.79×** |
| Flash Attention | block_q, block_k_major, block_k | 44 configs | block_k=1024 | 1.10× |
| GQA Attention | block_q, block_kv, block_kv_compute | 24 configs | [2048, 2048, 1024] | 1.08× |
| Paged Attention | pages_per_compute_block | 4 configs | 128 | 1.12× |
| MLA Attention | block_q, block_k_major, block_k | 24 configs | Already optimal | 1.00× |
| Sparse Attention | block_q, block_kv, block_kv_compute | 44 configs | Already optimal | 1.00× |
| Ragged Paged Attn | num_kv_pages_per_block, num_queries_per_block | 9 configs | Already optimal | 1.00× |
| GEMM | block_shape [bm, bn], block_k | 27 configs | [1024, 2048], bk=1024 | Already optimal |

## Output Format

### results.json

```json
{
  "metadata": {
    "jax_version": "0.6.2",
    "devices": "[TpuDevice(id=0, ...)]",
    "tpu_peak_tflops_bf16": 918.0,
    "timing_method": "jax.profiler device-side (jit_*() events from Perfetto trace)",
    "num_warmup": 5,
    "num_iters": 50
  },
  "results": [
    {
      "name": "8p_GEMM",
      "variant": "baseline",
      "status": "success",
      "timing_method": "device_profiler",
      "median_ms": 5.322,
      "mean_ms": 5.325,
      "std_ms": 0.003,
      "min_ms": 5.318,
      "wall_clock_median_ms": 5.480,
      "xla_flops": 3848290697216,
      "tflops": 723.0,
      "utilization_pct": 78.8,
      "output_shape": [8192, 28672],
      "num_iters": 50
    }
  ]
}
```

### results.csv

Flat CSV sorted by workload number (1p through 50k) with columns: `workload, variant, status, timing_method, median_ms, mean_ms, std_ms, min_ms, xla_flops, tflops, utilization_pct, output_shape, error`.

## File Structure

```
benchmark/
├── run_all.py                   # Main benchmark runner with profiler timing
├── profile_workload.py          # Profile a single workload
├── tune_pallas.py               # Grid search tuning for Pallas block sizes
├── results.json                 # Full results with metadata
├── results.csv                  # Flat CSV results
├── 1p_Flash_Attention/          # Priority kernel with baseline + optimized
│   ├── baseline.py              #   Vanilla JAX multi-head attention
│   ├── optimized.py             #   Pallas flash attention kernel
│   └── README.md
├── 8p_GEMM/                     # Priority kernel with baseline + optimized
│   ├── baseline.py              #   jnp.dot(A, B)
│   ├── optimized.py             #   Pallas tiled matmul kernel
│   └── README.md
├── 19k_Matmul_Subtract_Multiply_ReLU/  # KernelBench fused op (baseline only)
│   ├── baseline.py
│   └── README.md
└── ...                          # 50 workload directories total
```
