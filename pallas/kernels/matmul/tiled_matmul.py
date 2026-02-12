"""
Tiled Matmul Pallas Kernels for TPU

This module provides reference implementations of tiled matrix multiplication
using Pallas for TPU. These serve as building blocks for more complex kernels.

Key patterns:
1. Basic tiled matmul with accumulator
2. BF16 matmul with FP32 accumulator
3. Fused matmul + activation
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools


# ============================================================================
# Basic Tiled Matmul
# ============================================================================

def basic_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    """
    Basic tiled matmul kernel.

    Grid: (M // bm, N // bn, K // bk)
    - program_id(0): block row index
    - program_id(1): block col index
    - program_id(2): reduction step index

    The accumulator is stored in VMEM scratch space and accumulated
    across the K dimension.
    """
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...],
                           preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def matmul_pallas(x, y, bm=128, bk=128, bn=128):
    """
    Tiled matrix multiplication using Pallas.

    Args:
        x: Input matrix (M, K)
        y: Input matrix (K, N)
        bm: Block size for M dimension
        bk: Block size for K dimension
        bn: Block size for N dimension

    Returns:
        Output matrix (M, N)
    """
    m, k = x.shape
    _, n = y.shape

    assert m % bm == 0, f"M ({m}) must be divisible by bm ({bm})"
    assert k % bk == 0, f"K ({k}) must be divisible by bk ({bk})"
    assert n % bn == 0, f"N ({n}) must be divisible by bn ({bn})"

    nsteps = k // bk

    return pl.pallas_call(
        functools.partial(basic_matmul_kernel, nsteps=nsteps),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
    )(x, y)


# ============================================================================
# BF16 Matmul with FP32 Accumulator
# ============================================================================

def bf16_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    """BF16 matmul with FP32 accumulator for better precision."""
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Accumulate in FP32 for precision
    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...],
                           preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(jnp.bfloat16)


def matmul_bf16_pallas(x, y, bm=128, bk=128, bn=128):
    """BF16 tiled matmul with FP32 accumulator."""
    m, k = x.shape
    _, n = y.shape
    nsteps = k // bk

    return pl.pallas_call(
        functools.partial(bf16_matmul_kernel, nsteps=nsteps),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
    )(x, y)


# ============================================================================
# Fused Matmul + ReLU
# ============================================================================

def matmul_relu_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    """Matmul fused with ReLU activation."""
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...],
                           preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        # Fuse ReLU with output write
        z_ref[...] = jnp.maximum(acc_ref[...], 0).astype(z_ref.dtype)


def matmul_relu_pallas(x, y, bm=128, bk=128, bn=128):
    """Fused matmul + ReLU."""
    m, k = x.shape
    _, n = y.shape
    nsteps = k // bk

    return pl.pallas_call(
        functools.partial(matmul_relu_kernel, nsteps=nsteps),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
    )(x, y)


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_matmul(size=4096, warmup=5, iters=20):
    """Compare JAX native vs Pallas matmul."""
    import time

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (size, size), dtype=jnp.float32)
    y = jax.random.normal(key, (size, size), dtype=jnp.float32)

    # JAX native
    jax_fn = jax.jit(lambda a, b: jnp.dot(a, b))
    for _ in range(warmup):
        jax_fn(x, y).block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        jax_fn(x, y).block_until_ready()
    jax_time = (time.perf_counter() - start) / iters * 1000

    # Pallas
    pallas_fn = jax.jit(matmul_pallas)
    for _ in range(warmup):
        pallas_fn(x, y).block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        pallas_fn(x, y).block_until_ready()
    pallas_time = (time.perf_counter() - start) / iters * 1000

    return {
        'size': size,
        'jax_ms': jax_time,
        'pallas_ms': pallas_time,
        'speedup': jax_time / pallas_time,
    }


if __name__ == "__main__":
    import sys

    size = 4096
    if len(sys.argv) > 1:
        size = int(sys.argv[1])

    print(f"Benchmarking {size}x{size} matmul...")
    result = benchmark_matmul(size)
    print(f"JAX:    {result['jax_ms']:.2f} ms")
    print(f"Pallas: {result['pallas_ms']:.2f} ms")
    print(f"Speedup: {result['speedup']:.2f}x")
