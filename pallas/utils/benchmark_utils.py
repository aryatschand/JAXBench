"""Benchmark utilities for Pallas kernel evaluation."""

import time
import jax.numpy as jnp


WARMUP_ITERS = 5
BENCHMARK_ITERS = 20


def benchmark_fn(fn, *args, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """
    Benchmark a JAX function with warmup.

    Args:
        fn: JIT-compiled function to benchmark
        *args: Arguments to pass to function
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations

    Returns:
        Tuple of (time_ms, result)
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        result.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        result = fn(*args)
        result.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000
    return time_ms, result


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two arrays.

    Args:
        a: First array
        b: Second array

    Returns:
        Cosine similarity (float between -1 and 1)
    """
    a_flat = a.flatten().astype(jnp.float32)
    b_flat = b.flatten().astype(jnp.float32)
    dot = jnp.dot(a_flat, b_flat)
    norm = jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat)
    return float(dot / norm)


def max_abs_diff(a, b):
    """Compute maximum absolute difference between arrays."""
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def relative_error(result, reference):
    """Compute relative error vs reference."""
    ref_flat = reference.flatten().astype(jnp.float32)
    res_flat = result.flatten().astype(jnp.float32)
    diff = jnp.abs(ref_flat - res_flat)
    ref_abs = jnp.abs(ref_flat) + 1e-10
    return float(jnp.mean(diff / ref_abs))


def format_results_table(results, headers=None):
    """Format results as ASCII table."""
    if not results:
        return ""

    if headers is None:
        headers = list(results[0].keys())

    # Calculate column widths
    widths = {h: len(h) for h in headers}
    for r in results:
        for h in headers:
            val = r.get(h, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}" if val < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            widths[h] = max(widths[h], len(val_str))

    # Build table
    sep = "-" * (sum(widths.values()) + 3 * len(headers) + 1)
    header_row = " | ".join(h.ljust(widths[h]) for h in headers)

    rows = []
    for r in results:
        row_vals = []
        for h in headers:
            val = r.get(h, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}" if val < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            row_vals.append(val_str.ljust(widths[h]))
        rows.append(" | ".join(row_vals))

    return f"{sep}\n{header_row}\n{sep}\n" + "\n".join(rows) + f"\n{sep}"
