#!/usr/bin/env python3
"""
JAX vs Pallas Comparison Script

Directly compares JAX native implementations against Pallas kernels
for the same operations, ensuring fair comparison (same precision, same inputs).

Usage:
    PJRT_DEVICE=TPU python pallas/scripts/compare_jax_pallas.py --size 4096
"""

import argparse
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def compare_matmul(size, dtype="float32"):
    """Compare JAX vs Pallas for basic matmul."""
    import jax
    import jax.numpy as jnp
    from pallas.utils.benchmark_utils import benchmark_fn, cosine_similarity
    from pallas.kernels.matmul.tiled_matmul import matmul_pallas

    print(f"\n=== Matmul {size}x{size} ({dtype}) ===")

    key = jax.random.PRNGKey(42)
    dtype_jnp = getattr(jnp, dtype)
    x = jax.random.normal(key, (size, size), dtype=dtype_jnp)
    y = jax.random.normal(key, (size, size), dtype=dtype_jnp)

    # JAX native
    jax_fn = jax.jit(lambda a, b: jnp.dot(a, b))
    jax_time, jax_result = benchmark_fn(jax_fn, x, y)

    # Pallas
    try:
        pallas_fn = jax.jit(matmul_pallas)
        pallas_time, pallas_result = benchmark_fn(pallas_fn, x, y)
        accuracy = cosine_similarity(pallas_result, jax_result)
        pallas_ok = True
    except Exception as e:
        pallas_time = float('inf')
        accuracy = 0
        pallas_ok = False
        print(f"  Pallas failed: {e}")

    speedup = jax_time / pallas_time if pallas_ok else 0

    print(f"  JAX:    {jax_time:.2f} ms")
    if pallas_ok:
        print(f"  Pallas: {pallas_time:.2f} ms (speedup: {speedup:.2f}x, acc: {accuracy:.4f})")
    else:
        print(f"  Pallas: FAILED")

    return {
        'operation': 'matmul',
        'size': size,
        'dtype': dtype,
        'jax_ms': jax_time,
        'pallas_ms': pallas_time if pallas_ok else None,
        'speedup': speedup,
        'accuracy': accuracy,
    }


def compare_quantized_matmul(size, bits=8):
    """Compare JAX vs Pallas for quantized matmul."""
    import jax
    import jax.numpy as jnp
    from pallas.utils.benchmark_utils import benchmark_fn, cosine_similarity

    print(f"\n=== Quantized Matmul {size}x{size} (INT{bits}) ===")

    key = jax.random.PRNGKey(42)
    x_fp32 = jax.random.normal(key, (size, size), dtype=jnp.float32)
    y_fp32 = jax.random.normal(key, (size, size), dtype=jnp.float32)

    # Quantize
    max_val = 2 ** (bits - 1) - 1
    x_scale = float(jnp.max(jnp.abs(x_fp32)) / max_val)
    y_scale = float(jnp.max(jnp.abs(y_fp32)) / max_val)

    x_int = jnp.clip(jnp.round(x_fp32 / x_scale), -max_val-1, max_val).astype(jnp.int8)
    y_int = jnp.clip(jnp.round(y_fp32 / y_scale), -max_val-1, max_val).astype(jnp.int8)

    # JAX INT32 accumulation
    def jax_quant_matmul(x, y):
        result = jnp.dot(x.astype(jnp.int32), y.astype(jnp.int32))
        return result.astype(jnp.float32) * x_scale * y_scale

    jax_fn = jax.jit(jax_quant_matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, x_int, y_int)

    # Reference FP32
    ref_result = jnp.dot(x_fp32, y_fp32)
    accuracy = cosine_similarity(jax_result, ref_result)

    # FP32 baseline for comparison
    fp32_fn = jax.jit(lambda a, b: jnp.dot(a, b))
    fp32_time, _ = benchmark_fn(fp32_fn, x_fp32, y_fp32)

    print(f"  FP32:   {fp32_time:.2f} ms")
    print(f"  INT{bits}:   {jax_time:.2f} ms (speedup: {fp32_time/jax_time:.2f}x vs FP32, acc: {accuracy:.4f})")

    return {
        'operation': f'quantized_matmul_int{bits}',
        'size': size,
        'fp32_ms': fp32_time,
        'quantized_ms': jax_time,
        'speedup_vs_fp32': fp32_time / jax_time,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="JAX vs Pallas Comparison")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size")
    parser.add_argument("--sizes", type=str, default=None,
                       help="Comma-separated sizes (overrides --size)")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "bfloat16"])
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
    else:
        sizes = [args.size]

    print("=" * 80)
    print("JAX vs PALLAS COMPARISON")
    print("=" * 80)

    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    except Exception as e:
        print(f"JAX import error: {e}")
        return

    results = []

    for size in sizes:
        # Basic matmul
        results.append(compare_matmul(size, args.dtype))

        # Quantized matmul (INT8)
        results.append(compare_quantized_matmul(size, bits=8))

        # Quantized matmul (INT5)
        results.append(compare_quantized_matmul(size, bits=5))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for r in results:
        op = r['operation']
        size = r['size']
        if 'pallas_ms' in r and r['pallas_ms']:
            print(f"{op} {size}: JAX={r['jax_ms']:.2f}ms, Pallas={r['pallas_ms']:.2f}ms, "
                  f"Speedup={r['speedup']:.2f}x")
        elif 'speedup_vs_fp32' in r:
            print(f"{op} {size}: {r['quantized_ms']:.2f}ms ({r['speedup_vs_fp32']:.2f}x vs FP32)")

    # Save
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
