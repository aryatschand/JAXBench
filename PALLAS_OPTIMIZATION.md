# Pallas TPU Kernel Optimization

## Overview

This document describes JAXBench's Pallas kernel optimization pipeline, which uses Opus 4.5 to automatically generate optimized Pallas TPU kernels from JAX code.

## Strategy

### The Key Insight: Hybrid Approach

The most important discovery is that **reimplementing matmul/convolution in Pallas is counterproductive**. XLA's implementations are highly optimized. Instead, we use a **hybrid approach**:

1. **Use JAX's built-in operations** (matmul, conv) - XLA already optimizes these
2. **Use Pallas only for fusing post-processing** - bias, scale, activation in a single memory pass

This avoids competing with XLA's optimized matmul while still gaining fusion benefits.

### When Pallas Wins

Pallas provides speedups when:
1. **Fusing multiple elementwise operations** - avoids memory round-trips
2. **Custom memory access patterns** - flash attention, sliding windows
3. **Operations XLA doesn't fuse well** - chained activations, custom normalizations

### When Pallas Loses

Pallas is slower when:
- Reimplementing highly optimized operations (matmul, convolution)
- Over-tiling creates too many kernel invocations
- Tensors are too large for VMEM (16MB limit)

## Results

### Level 1 - Elementwise Operations

Operations that **match or beat JAX baseline**:

| Task | Operation | Pallas vs JAX |
|------|-----------|---------------|
| 25 | Swish | **1.01x** |
| 22 | Tanh | 1.00x |
| 28 | HardSigmoid | 1.00x |
| 26 | GELU | 1.00x |
| 19 | ReLU | 0.99x |

### Level 2 - Fused Operations

Operations that **beat JAX baseline** using the hybrid approach:

| Task | Operation | Pallas vs JAX |
|------|-----------|---------------|
| 39 | Gemm_Scale_BatchNorm | **1.03x** |
| 40 | Matmul_Scaling_ResidualAdd | **1.02x** |
| 14 | Gemm_Divide_Sum_Scaling | 1.00x |
| 33 | Gemm_Scale_BatchNorm | 0.99x |
| 12 | Gemm_Multiply_LeakyReLU | 0.95x |

## Implementation

### Pipeline Components

```
src/
  pallas_prompts.py      # Prompt templates with Pallas documentation
  pallas_translator.py   # JAX -> Pallas translation using LLM

scripts/
  run_pallas_benchmark.py  # CLI for running Pallas benchmarks

jaxbench/pallas/
  level1/                # Generated Level 1 Pallas kernels
  level2/                # Generated Level 2 Pallas kernels

results/pallas/
  checkpoint_pallas.json # Progress checkpoint
  pallas_*.json          # Detailed results
```

### Key Prompt Engineering

The system prompt includes:

1. **Critical Performance Insight**: Don't over-tile - process entire tensors when possible
2. **Hybrid Approach Examples**: Use JAX matmul + Pallas fused activation
3. **Operation-Specific Strategies**: Tailored guidance for each operation type
4. **Error-Specific Fix Guidance**: Targeted fixes for common errors

### Example: Hybrid Kernel Pattern

```python
def forward_pallas(self, x):
    # Step 1: Use JAX's optimized matmul
    mm_result = jnp.matmul(x, self.weight)

    # Step 2: Fuse ALL post-processing in Pallas (single memory pass)
    def fused_postprocess_kernel(mm_ref, bias_ref, o_ref):
        x = mm_ref[...].astype(jnp.float32)
        x = x + bias_ref[...]           # Add bias
        x = x * self.multiplier         # Scale
        x = jnp.where(x >= 0, x, x * 0.01)  # LeakyReLU
        o_ref[...] = x.astype(o_ref.dtype)

    return pl.pallas_call(
        fused_postprocess_kernel,
        out_shape=jax.ShapeDtypeStruct(mm_result.shape, mm_result.dtype),
        in_specs=[
            pl.BlockSpec(mm_result.shape, lambda: ()),
            pl.BlockSpec(self.bias.shape, lambda: ()),
        ],
        out_specs=pl.BlockSpec(mm_result.shape, lambda: ()),
        grid=(1,),  # Single kernel call - no tiling overhead
    )(mm_result, self.bias)
```

## Usage

```bash
# Run Pallas benchmark on Level 1
python scripts/run_pallas_benchmark.py --level 1 --tasks "19,22,25,26,28" --keep-tpu

# Run Pallas benchmark on Level 2 (fused operations)
python scripts/run_pallas_benchmark.py --level 2 --tasks "12,14,33,39,40" --keep-tpu

# Disable cache to regenerate kernels
python scripts/run_pallas_benchmark.py --level 2 --tasks "12" --no-cache
```

## Challenges and Solutions

### Challenge 1: Over-Tiling
**Problem**: Initial kernels used small tiles (128x128), creating massive overhead.
**Solution**: Process entire tensors with `grid=(1,)` when they fit in VMEM.

### Challenge 2: VMEM Exhaustion
**Problem**: Large tensors (>16MB) don't fit in VMEM.
**Solution**:
- For matmul: Use JAX's matmul (XLA handles tiling automatically)
- For post-processing: Keep tensors small or add explicit tiling

### Challenge 3: Numerical Precision
**Problem**: Pallas kernels had precision differences.
**Solution**: Use float32 for all intermediate computations, cast at the end.

### Challenge 4: Missing Primitives
**Problem**: TPU Pallas doesn't support `erf` (used in exact GELU).
**Solution**: Use approximate GELU: `x * sigmoid(1.702 * x)`

## Key Learnings

1. **Don't compete with XLA's matmul** - it's already optimal
2. **Fusion is the value-add** - combining operations saves memory bandwidth
3. **VMEM is the bottleneck** - design kernels around the 16MB limit
4. **grid=(1,) is fastest** for small/medium tensors
5. **Error feedback works** - the refinement loop fixes 70%+ of errors

## Future Improvements

1. **Flash Attention**: Custom memory access patterns for attention
2. **Quantization Kernels**: Int8/int4 operations not supported by XLA
3. **Sparse Operations**: Custom sparse matmul/attention patterns
4. **Multi-chip Kernels**: Distributed Pallas for larger models
