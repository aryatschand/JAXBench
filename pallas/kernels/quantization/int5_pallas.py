#!/usr/bin/env python3
"""
INT5 Pallas Kernel V3 - Properly passing scales

The scalar prefetch approach requires the scales to be part of the pallas_call.
Let me try a different approach: embed scales in the inputs or use a wrapper.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools
import time
import sys


WARMUP_ITERS = 5
BENCHMARK_ITERS = 20


def benchmark_fn(fn, *args, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    for _ in range(warmup):
        result = fn(*args)
        result.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        result = fn(*args)
        result.block_until_ready()
    end = time.perf_counter()

    return (end - start) / iters * 1000, result


def cosine_similarity(a, b):
    a_flat = a.flatten().astype(jnp.float32)
    b_flat = b.flatten().astype(jnp.float32)
    return float(jnp.dot(a_flat, b_flat) / (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat)))


# ============================================================================
# Pallas INT5 Kernels - Fixed scalar handling
# ============================================================================

def create_int5_fused_kernel(x_scale, w_scale):
    """Factory that creates kernel with scales baked in via closure over Python floats."""
    # Convert to Python floats to avoid JAX array capture
    xs = float(x_scale)
    ws = float(w_scale)

    bm, bk, bn = 128, 128, 128

    def int5_fused_kernel(x_ref, w_ref, z_ref, acc_ref, *, nsteps):
        """Fused dequant + matmul."""
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        # Dequant in registers using Python float scales (not JAX arrays)
        x_float = x_ref[...].astype(jnp.float32) * xs
        w_float = w_ref[...].astype(jnp.float32) * ws

        acc_ref[...] += jnp.dot(x_float, w_float, preferred_element_type=jnp.float32)

        @pl.when(pl.program_id(2) == nsteps - 1)
        def _():
            z_ref[...] = acc_ref[...].astype(z_ref.dtype)

    return int5_fused_kernel


def run_int5_pallas_fused(M, K, N, x_int5, w_int5, x_scale, w_scale):
    """Pallas fused dequant + matmul using Python float scales."""
    bm, bk, bn = 128, 128, 128

    # Create kernel with scales baked in
    kernel = create_int5_fused_kernel(x_scale, w_scale)

    def pallas_fused_matmul(x, w):
        m, k = x.shape
        _, n = w.shape
        nsteps = k // bk

        return pl.pallas_call(
            functools.partial(kernel, nsteps=nsteps),
            out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
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
        )(x, w)

    try:
        pallas_fn = jax.jit(pallas_fused_matmul)
        time_ms, result = benchmark_fn(pallas_fn, x_int5, w_int5)
        return time_ms, result, None
    except Exception as e:
        return float('inf'), None, str(e)


def create_int5_int_kernel(x_scale, w_scale):
    """Factory for INT32 accumulation kernel."""
    xs = float(x_scale)
    ws = float(w_scale)
    combined_scale = xs * ws

    bm, bk, bn = 128, 128, 128

    def int5_int_kernel(x_ref, w_ref, z_ref, acc_ref, *, nsteps):
        """INT32 accumulation, scale at the end."""
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.int32)

        acc_ref[...] += jnp.dot(
            x_ref[...].astype(jnp.int32),
            w_ref[...].astype(jnp.int32),
            preferred_element_type=jnp.int32
        )

        @pl.when(pl.program_id(2) == nsteps - 1)
        def _():
            z_ref[...] = (acc_ref[...].astype(jnp.float32) * combined_scale).astype(z_ref.dtype)

    return int5_int_kernel


def run_int5_pallas_int(M, K, N, x_int5, w_int5, x_scale, w_scale):
    """Pallas INT32 accumulation + final scale."""
    bm, bk, bn = 128, 128, 128

    kernel = create_int5_int_kernel(x_scale, w_scale)

    def pallas_int_matmul(x, w):
        m, k = x.shape
        _, n = w.shape
        nsteps = k // bk

        return pl.pallas_call(
            functools.partial(kernel, nsteps=nsteps),
            out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
                scratch_shapes=[pltpu.VMEM((bm, bn), jnp.int32)],
                grid=(m // bm, n // bn, k // bk),
            ),
        )(x, w)

    try:
        pallas_fn = jax.jit(pallas_int_matmul)
        time_ms, result = benchmark_fn(pallas_fn, x_int5, w_int5)
        return time_ms, result, None
    except Exception as e:
        return float('inf'), None, str(e)


# ============================================================================
# Main
# ============================================================================

def run_all_benchmarks(size=4096):
    M = N = K = size

    print("=" * 90)
    print(f"INT5 PALLAS BENCHMARK V3 - Size {M}x{M}")
    print("=" * 90)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    # Generate test data
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)
    ref_result = jnp.dot(x_fp32, w_fp32)

    # Quantize
    x_scale = float(jnp.max(jnp.abs(x_fp32)) / 15.0)
    w_scale = float(jnp.max(jnp.abs(w_fp32)) / 15.0)
    x_int5 = jnp.clip(jnp.round(x_fp32 / x_scale), -16, 15).astype(jnp.int8)
    w_int5 = jnp.clip(jnp.round(w_fp32 / w_scale), -16, 15).astype(jnp.int8)

    results = {}

    # 1. FP32 baseline
    print("1. FP32 Baseline...", end=" ", flush=True)
    fp32_fn = jax.jit(lambda x, w: jnp.dot(x, w))
    fp32_time, _ = benchmark_fn(fp32_fn, x_fp32, w_fp32)
    results['fp32'] = fp32_time
    print(f"{fp32_time:.2f} ms")

    # 2. JAX INT5 (INT32 accum)
    print("2. JAX INT5 (INT32 accum)...", end=" ", flush=True)
    def jax_int_matmul(x, w):
        result = jnp.dot(x.astype(jnp.int32), w.astype(jnp.int32))
        return result.astype(jnp.float32) * x_scale * w_scale
    jax_int_fn = jax.jit(jax_int_matmul)
    jax_int_time, jax_int_result = benchmark_fn(jax_int_fn, x_int5, w_int5)
    jax_int_acc = cosine_similarity(jax_int_result, ref_result)
    results['jax_int'] = jax_int_time
    print(f"{jax_int_time:.2f} ms (acc={jax_int_acc:.4f})")

    # 3. JAX INT5 (dequant + FP32)
    print("3. JAX INT5 (dequant+FP32)...", end=" ", flush=True)
    def jax_dequant_matmul(x, w):
        return jnp.dot(x.astype(jnp.float32) * x_scale, w.astype(jnp.float32) * w_scale)
    jax_dequant_fn = jax.jit(jax_dequant_matmul)
    jax_dequant_time, jax_dequant_result = benchmark_fn(jax_dequant_fn, x_int5, w_int5)
    jax_dequant_acc = cosine_similarity(jax_dequant_result, ref_result)
    results['jax_dequant'] = jax_dequant_time
    print(f"{jax_dequant_time:.2f} ms (acc={jax_dequant_acc:.4f})")

    # 4. Pallas Fused Dequant
    print("4. Pallas Fused Dequant...", end=" ", flush=True)
    pallas_fused_time, pallas_fused_result, pallas_fused_err = run_int5_pallas_fused(
        M, K, N, x_int5, w_int5, x_scale, w_scale)
    if pallas_fused_err:
        print(f"FAILED: {pallas_fused_err[:80]}")
        results['pallas_fused'] = None
    else:
        pallas_fused_acc = cosine_similarity(pallas_fused_result, ref_result)
        results['pallas_fused'] = pallas_fused_time
        print(f"{pallas_fused_time:.2f} ms (acc={pallas_fused_acc:.4f})")

    # 5. Pallas INT32 Accum
    print("5. Pallas INT32 Accum...", end=" ", flush=True)
    pallas_int_time, pallas_int_result, pallas_int_err = run_int5_pallas_int(
        M, K, N, x_int5, w_int5, x_scale, w_scale)
    if pallas_int_err:
        print(f"FAILED: {pallas_int_err[:80]}")
        results['pallas_int'] = None
    else:
        pallas_int_acc = cosine_similarity(pallas_int_result, ref_result)
        results['pallas_int'] = pallas_int_time
        print(f"{pallas_int_time:.2f} ms (acc={pallas_int_acc:.4f})")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} | {'Time (ms)':>10} | {'vs FP32':>8} | {'vs JAX INT':>10}")
    print("-" * 70)

    for name, time_ms in [
        ('FP32 Baseline', fp32_time),
        ('JAX INT5 (INT32 accum)', jax_int_time),
        ('JAX INT5 (dequant+FP32)', jax_dequant_time),
        ('Pallas Fused Dequant', results.get('pallas_fused')),
        ('Pallas INT32 Accum', results.get('pallas_int')),
    ]:
        if time_ms is None:
            print(f"{name:<30} | {'FAILED':>10} | {'-':>8} | {'-':>10}")
        else:
            vs_fp32 = fp32_time / time_ms
            vs_jax = jax_int_time / time_ms
            print(f"{name:<30} | {time_ms:>10.2f} | {vs_fp32:>7.2f}x | {vs_jax:>9.2f}x")

    print("-" * 70)

    # Key insight
    print()
    if results.get('pallas_fused') and results.get('pallas_int'):
        pallas_best = min(results['pallas_fused'], results['pallas_int'])
        jax_best = jax_int_time
        if pallas_best < jax_best:
            speedup = jax_best / pallas_best
            print(f"*** Pallas beats JAX by {speedup:.2f}x! ***")
        else:
            slowdown = pallas_best / jax_best
            print(f"*** JAX beats Pallas by {slowdown:.2f}x (Pallas adds overhead) ***")
    elif results.get('pallas_fused') is None and results.get('pallas_int') is None:
        print("*** Both Pallas kernels failed - JAX wins by default ***")

    return results


if __name__ == "__main__":
    sizes = [4096]
    if len(sys.argv) > 1:
        try:
            sizes = [int(sys.argv[1])]
        except ValueError:
            pass

    # Add 8192 for scale test
    if 8192 not in sizes:
        sizes.append(8192)

    for size in sizes:
        size = (size // 128) * 128
        if size < 128:
            size = 128
        run_all_benchmarks(size)
        print("\n")
