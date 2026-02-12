#!/usr/bin/env python3
"""
True INT5 Pallas Kernel - Packed 5-bit Quantization

This implements REAL 5-bit packing where 8 values fit into 5 bytes (40 bits).
The Pallas kernel unpacks the 5-bit values and performs the matmul.

Comparison:
1. JAX INT8 baseline (what we were measuring before - unfair comparison)
2. JAX with packed INT5 + separate unpack + matmul (memory bandwidth test)
3. Pallas with fused unpack + matmul (the interesting one)

Packing scheme:
- 8 INT5 values (each -16 to 15, stored as 0-31 unsigned) = 40 bits = 5 bytes
- Pack: v0[4:0] v1[4:0] v2[4:0] v3[4:0] v4[4:0] v5[4:0] v6[4:0] v7[4:0]
        byte0   byte1   byte2   byte3   byte4
        [v0:5][v1:3] [v1:2][v2:5][v3:1] [v3:4][v4:4] [v4:1][v5:5][v6:2] [v6:3][v7:5]

Actually, let's use a simpler scheme that's easier to implement:
- Store 6 INT5 values in 4 bytes (30 bits used, 2 wasted) = 1.33 bytes per value
- Or store values with partial packing

For clarity, let's do: 2 INT5 values packed into 10 bits, stored in uint16
- This gives 1.25 bytes per value vs 1 byte for INT8 (20% savings)

Usage:
    python int5_pallas_kernel.py [matrix_size]
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
    """Benchmark a function with warmup."""
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
# INT5 Packing/Unpacking
# ============================================================================

def pack_int5_to_bytes(values):
    """
    Pack INT5 values into bytes.

    Strategy: Pack 8 INT5 values (40 bits) into 5 bytes.
    Input shape: (..., N) where N is divisible by 8
    Output shape: (..., N * 5 // 8)

    Values are in range [-16, 15], we add 16 to make them [0, 31] (5 bits unsigned).
    """
    # Shift to unsigned [0, 31]
    unsigned = (values.astype(jnp.int32) + 16).astype(jnp.uint8)

    # Reshape to groups of 8
    orig_shape = values.shape
    flat = unsigned.reshape(-1, 8)

    # Pack 8 x 5-bit values into 5 bytes
    # byte0 = v0[4:0] << 3 | v1[4:2]
    # byte1 = v1[1:0] << 6 | v2[4:0] << 1 | v3[4]
    # byte2 = v3[3:0] << 4 | v4[4:1]
    # byte3 = v4[0] << 7 | v5[4:0] << 2 | v6[4:3]
    # byte4 = v6[2:0] << 5 | v7[4:0]

    v = flat.astype(jnp.uint32)

    byte0 = ((v[:, 0] << 3) | (v[:, 1] >> 2)).astype(jnp.uint8)
    byte1 = (((v[:, 1] & 0x03) << 6) | (v[:, 2] << 1) | (v[:, 3] >> 4)).astype(jnp.uint8)
    byte2 = (((v[:, 3] & 0x0F) << 4) | (v[:, 4] >> 1)).astype(jnp.uint8)
    byte3 = (((v[:, 4] & 0x01) << 7) | (v[:, 5] << 2) | (v[:, 6] >> 3)).astype(jnp.uint8)
    byte4 = (((v[:, 6] & 0x07) << 5) | v[:, 7]).astype(jnp.uint8)

    packed = jnp.stack([byte0, byte1, byte2, byte3, byte4], axis=1)

    # Reshape to match original (with 5/8 the size in last dim)
    new_shape = orig_shape[:-1] + (orig_shape[-1] * 5 // 8,)
    return packed.reshape(new_shape)


def unpack_int5_from_bytes(packed, orig_size):
    """
    Unpack INT5 values from bytes.

    Input shape: (..., N * 5 // 8)
    Output shape: (..., N)
    """
    orig_shape = packed.shape
    flat = packed.reshape(-1, 5)

    b = flat.astype(jnp.uint32)

    # Reverse the packing
    v0 = (b[:, 0] >> 3) & 0x1F
    v1 = ((b[:, 0] & 0x07) << 2) | (b[:, 1] >> 6)
    v2 = (b[:, 1] >> 1) & 0x1F
    v3 = ((b[:, 1] & 0x01) << 4) | (b[:, 2] >> 4)
    v4 = ((b[:, 2] & 0x0F) << 1) | (b[:, 3] >> 7)
    v5 = (b[:, 3] >> 2) & 0x1F
    v6 = ((b[:, 3] & 0x03) << 3) | (b[:, 4] >> 5)
    v7 = b[:, 4] & 0x1F

    unpacked = jnp.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=1)

    # Convert back to signed [-16, 15]
    signed = unpacked.astype(jnp.int8) - 16

    new_shape = orig_shape[:-1] + (orig_size,)
    return signed.reshape(new_shape)


# ============================================================================
# Benchmarks
# ============================================================================

def run_fp32_baseline(M, K, N, x_fp32, w_fp32):
    """FP32 baseline matmul."""
    def matmul(x, w):
        return jnp.dot(x, w)

    jax_fn = jax.jit(matmul)
    time_ms, result = benchmark_fn(jax_fn, x_fp32, w_fp32)
    return time_ms, result


def run_int8_baseline(M, K, N, x_fp32, w_fp32):
    """INT8 baseline (what we were comparing against before - unfair)."""
    x_scale = jnp.max(jnp.abs(x_fp32)) / 127.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 127.0

    x_int8 = jnp.clip(jnp.round(x_fp32 / x_scale), -128, 127).astype(jnp.int8)
    w_int8 = jnp.clip(jnp.round(w_fp32 / w_scale), -128, 127).astype(jnp.int8)

    def matmul(x, w):
        return jnp.dot(x.astype(jnp.int32), w.astype(jnp.int32))

    jax_fn = jax.jit(matmul)
    time_ms, result = benchmark_fn(jax_fn, x_int8, w_int8)

    # Dequantize for accuracy
    result_fp32 = result.astype(jnp.float32) * x_scale * w_scale
    return time_ms, result_fp32, x_scale, w_scale


def run_int5_packed_jax(M, K, N, x_fp32, w_fp32):
    """
    INT5 with real packing - JAX unpack then matmul.
    This is the fair comparison baseline for Pallas.
    """
    x_scale = jnp.max(jnp.abs(x_fp32)) / 15.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 15.0

    # Quantize to INT5 range
    x_int5 = jnp.clip(jnp.round(x_fp32 / x_scale), -16, 15).astype(jnp.int8)
    w_int5 = jnp.clip(jnp.round(w_fp32 / w_scale), -16, 15).astype(jnp.int8)

    # Pack to 5-bit
    x_packed = pack_int5_to_bytes(x_int5)
    w_packed = pack_int5_to_bytes(w_int5)

    def matmul_with_unpack(x_packed, w_packed, x_size, w_size):
        # Unpack
        x_unpacked = unpack_int5_from_bytes(x_packed, x_size)
        w_unpacked = unpack_int5_from_bytes(w_packed, w_size)
        # Matmul
        return jnp.dot(x_unpacked.astype(jnp.int32), w_unpacked.astype(jnp.int32))

    jax_fn = jax.jit(lambda xp, wp: matmul_with_unpack(xp, wp, K, N))
    time_ms, result = benchmark_fn(jax_fn, x_packed, w_packed)

    result_fp32 = result.astype(jnp.float32) * x_scale * w_scale
    return time_ms, result_fp32, x_packed, w_packed, x_scale, w_scale


def run_int5_pallas_fused(M, K, N, x_packed, w_packed, x_scale, w_scale):
    """
    INT5 with Pallas fused unpack + matmul.

    The key insight: We load packed bytes (smaller memory footprint)
    and unpack in registers before the matmul.
    """
    # Block sizes must be divisible by 8 for INT5 packing
    bm, bk, bn = 128, 128, 128

    # Packed block sizes (5/8 of original)
    bk_packed = bk * 5 // 8
    bn_packed = bn * 5 // 8

    def int5_fused_kernel(x_packed_ref, w_packed_ref, z_ref, acc_ref, *, nsteps):
        """Fused INT5 unpack + matmul kernel."""
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        # Load packed bytes
        x_packed_tile = x_packed_ref[...]  # (bm, bk_packed)
        w_packed_tile = w_packed_ref[...]  # (bk_packed, bn) - note: packed in K dim

        # Unpack x: (bm, bk_packed) -> (bm, bk)
        x_flat = x_packed_tile.reshape(-1, 5)
        b = x_flat.astype(jnp.uint32)
        v0 = (b[:, 0] >> 3) & 0x1F
        v1 = ((b[:, 0] & 0x07) << 2) | (b[:, 1] >> 6)
        v2 = (b[:, 1] >> 1) & 0x1F
        v3 = ((b[:, 1] & 0x01) << 4) | (b[:, 2] >> 4)
        v4 = ((b[:, 2] & 0x0F) << 1) | (b[:, 3] >> 7)
        v5 = (b[:, 3] >> 2) & 0x1F
        v6 = ((b[:, 3] & 0x03) << 3) | (b[:, 4] >> 5)
        v7 = b[:, 4] & 0x1F
        x_unpacked = jnp.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=1)
        x_unpacked = (x_unpacked.astype(jnp.float32) - 16.0).reshape(bm, bk)

        # Unpack w: need to handle the transposed case
        # w is (K, N), packed in K dimension -> (K_packed, N)
        # For each column of w, we have K_packed bytes that unpack to K values
        w_flat = w_packed_tile.reshape(-1, 5)
        b = w_flat.astype(jnp.uint32)
        v0 = (b[:, 0] >> 3) & 0x1F
        v1 = ((b[:, 0] & 0x07) << 2) | (b[:, 1] >> 6)
        v2 = (b[:, 1] >> 1) & 0x1F
        v3 = ((b[:, 1] & 0x01) << 4) | (b[:, 2] >> 4)
        v4 = ((b[:, 2] & 0x0F) << 1) | (b[:, 3] >> 7)
        v5 = (b[:, 3] >> 2) & 0x1F
        v6 = ((b[:, 3] & 0x03) << 3) | (b[:, 4] >> 5)
        v7 = b[:, 4] & 0x1F
        w_unpacked = jnp.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=1)
        w_unpacked = (w_unpacked.astype(jnp.float32) - 16.0).reshape(bk, bn)

        # Matmul
        acc_ref[...] += jnp.dot(x_unpacked, w_unpacked, preferred_element_type=jnp.float32)

        @pl.when(pl.program_id(2) == nsteps - 1)
        def _():
            z_ref[...] = acc_ref[...].astype(z_ref.dtype)

    def pallas_int5_matmul(x_packed, w_packed):
        m = x_packed.shape[0]
        k_packed = x_packed.shape[1]
        n = w_packed.shape[1]
        k = k_packed * 8 // 5

        nsteps = k // bk

        return pl.pallas_call(
            functools.partial(int5_fused_kernel, nsteps=nsteps),
            out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bm, bk_packed), lambda i, j, k: (i, k * bk_packed // (k_packed // nsteps) if nsteps > 0 else 0)),
                    pl.BlockSpec((bk_packed, bn), lambda i, j, k: (k * bk_packed // (k_packed // nsteps) if nsteps > 0 else 0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
                scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
                grid=(m // bm, n // bn, nsteps),
            ),
        )(x_packed, w_packed)

    try:
        pallas_fn = jax.jit(pallas_int5_matmul)
        time_ms, result = benchmark_fn(pallas_fn, x_packed, w_packed)
        result_scaled = result * x_scale * w_scale
        return time_ms, result_scaled, None
    except Exception as e:
        return float('inf'), None, str(e)


def run_int5_simple_pallas(M, K, N, x_fp32, w_fp32):
    """
    Simpler INT5 Pallas: Store as INT8 but use 5-bit range.
    Unpack/dequant fused in kernel.

    This tests if Pallas can beat JAX for the dequant+matmul fusion,
    even without the packing benefit.
    """
    bm, bk, bn = 128, 128, 128

    x_scale = jnp.max(jnp.abs(x_fp32)) / 15.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 15.0

    x_int5 = jnp.clip(jnp.round(x_fp32 / x_scale), -16, 15).astype(jnp.int8)
    w_int5 = jnp.clip(jnp.round(w_fp32 / w_scale), -16, 15).astype(jnp.int8)

    def int5_kernel(x_ref, w_ref, z_ref, acc_ref, *, nsteps, xs, ws):
        """INT5 dequant + matmul kernel."""
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        # Load int8, dequant to float32
        x_float = x_ref[...].astype(jnp.float32) * xs
        w_float = w_ref[...].astype(jnp.float32) * ws

        acc_ref[...] += jnp.dot(x_float, w_float, preferred_element_type=jnp.float32)

        @pl.when(pl.program_id(2) == nsteps - 1)
        def _():
            z_ref[...] = acc_ref[...].astype(z_ref.dtype)

    def pallas_int5_simple(x, w):
        m, k = x.shape
        _, n = w.shape
        nsteps = k // bk

        return pl.pallas_call(
            functools.partial(int5_kernel, nsteps=nsteps, xs=x_scale, ws=w_scale),
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
        pallas_fn = jax.jit(pallas_int5_simple)
        time_ms, result = benchmark_fn(pallas_fn, x_int5, w_int5)
        return time_ms, result, None
    except Exception as e:
        return float('inf'), None, str(e)


# ============================================================================
# Main
# ============================================================================

def run_all_benchmarks(size=4096):
    M = N = K = size

    # Ensure divisible by 8 for packing
    assert M % 8 == 0 and K % 8 == 0 and N % 8 == 0

    print("=" * 90)
    print("TRUE INT5 PALLAS KERNEL BENCHMARK")
    print("=" * 90)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Matrix size: {M}x{K} @ {K}x{N}")
    print()

    # Generate test data
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)
    ref_result = jnp.dot(x_fp32, w_fp32)

    results = {}

    # 1. FP32 baseline
    print("1. FP32 Baseline...", end=" ", flush=True)
    fp32_time, fp32_result = run_fp32_baseline(M, K, N, x_fp32, w_fp32)
    results['fp32'] = {'time_ms': fp32_time, 'accuracy': 1.0}
    print(f"{fp32_time:.2f} ms")

    # 2. INT8 baseline (unfair comparison)
    print("2. INT8 Baseline (unfair - more bits)...", end=" ", flush=True)
    int8_time, int8_result, _, _ = run_int8_baseline(M, K, N, x_fp32, w_fp32)
    int8_acc = cosine_similarity(int8_result, ref_result)
    results['int8'] = {'time_ms': int8_time, 'accuracy': int8_acc}
    print(f"{int8_time:.2f} ms (acc={int8_acc:.4f})")

    # 3. INT5 packed with JAX unpack+matmul
    print("3. INT5 Packed + JAX Unpack+Matmul...", end=" ", flush=True)
    int5_jax_time, int5_jax_result, x_packed, w_packed, x_scale, w_scale = run_int5_packed_jax(M, K, N, x_fp32, w_fp32)
    int5_jax_acc = cosine_similarity(int5_jax_result, ref_result)
    results['int5_jax'] = {'time_ms': int5_jax_time, 'accuracy': int5_jax_acc}
    print(f"{int5_jax_time:.2f} ms (acc={int5_jax_acc:.4f})")

    # 4. INT5 simple Pallas (int8 storage, fused dequant)
    print("4. INT5 Simple Pallas (INT8 storage, fused dequant)...", end=" ", flush=True)
    int5_simple_time, int5_simple_result, int5_simple_err = run_int5_simple_pallas(M, K, N, x_fp32, w_fp32)
    if int5_simple_err:
        print(f"FAILED: {int5_simple_err}")
        results['int5_pallas_simple'] = {'time_ms': float('inf'), 'accuracy': 0, 'error': int5_simple_err}
    else:
        int5_simple_acc = cosine_similarity(int5_simple_result, ref_result)
        results['int5_pallas_simple'] = {'time_ms': int5_simple_time, 'accuracy': int5_simple_acc}
        print(f"{int5_simple_time:.2f} ms (acc={int5_simple_acc:.4f})")

    # 5. INT5 Pallas with real packing (if we can get it working)
    print("5. INT5 Pallas with Real 5-bit Packing...", end=" ", flush=True)
    int5_pallas_time, int5_pallas_result, int5_pallas_err = run_int5_pallas_fused(
        M, K, N, x_packed, w_packed, x_scale, w_scale)
    if int5_pallas_err:
        print(f"FAILED: {int5_pallas_err}")
        results['int5_pallas_packed'] = {'time_ms': float('inf'), 'accuracy': 0, 'error': int5_pallas_err}
    else:
        int5_pallas_acc = cosine_similarity(int5_pallas_result, ref_result)
        results['int5_pallas_packed'] = {'time_ms': int5_pallas_time, 'accuracy': int5_pallas_acc}
        print(f"{int5_pallas_time:.2f} ms (acc={int5_pallas_acc:.4f})")

    # Summary
    print()
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print()
    print(f"{'Method':<45} | {'Time (ms)':>10} | {'vs FP32':>8} | {'vs INT5 JAX':>12} | {'Accuracy':>8}")
    print("-" * 90)

    for name, r in results.items():
        time_ms = r['time_ms']
        if time_ms == float('inf'):
            print(f"{name:<45} | {'FAILED':>10} | {'-':>8} | {'-':>12} | {'-':>8}")
            continue

        vs_fp32 = fp32_time / time_ms
        vs_int5_jax = int5_jax_time / time_ms if int5_jax_time > 0 else 0
        acc = r['accuracy']

        print(f"{name:<45} | {time_ms:>10.2f} | {vs_fp32:>7.2f}x | {vs_int5_jax:>11.2f}x | {acc:>8.4f}")

    print("-" * 90)

    # Analysis
    print()
    print("=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    print()
    print("Key comparisons:")
    print(f"  - FP32 baseline: {fp32_time:.2f} ms")
    print(f"  - INT8 (unfair, more bits): {int8_time:.2f} ms ({fp32_time/int8_time:.2f}x vs FP32)")
    print(f"  - INT5 JAX (packed+unpack+matmul): {int5_jax_time:.2f} ms ({fp32_time/int5_jax_time:.2f}x vs FP32)")

    if results.get('int5_pallas_simple', {}).get('time_ms', float('inf')) < float('inf'):
        t = results['int5_pallas_simple']['time_ms']
        print(f"  - INT5 Pallas simple: {t:.2f} ms ({int5_jax_time/t:.2f}x vs INT5 JAX)")

    if results.get('int5_pallas_packed', {}).get('time_ms', float('inf')) < float('inf'):
        t = results['int5_pallas_packed']['time_ms']
        print(f"  - INT5 Pallas packed: {t:.2f} ms ({int5_jax_time/t:.2f}x vs INT5 JAX)")

    print()
    print("The FAIR comparison is: INT5 Pallas vs INT5 JAX (same precision)")
    print("The unfair comparison was: INT5 vs FP32 (different precision)")

    return results


if __name__ == "__main__":
    size = 4096
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except ValueError:
            pass

    # Ensure size is divisible by 128 (block size) and 8 (packing)
    size = (size // 128) * 128
    if size < 128:
        size = 128

    results = run_all_benchmarks(size)
