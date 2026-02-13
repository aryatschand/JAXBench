"""
RMSNorm - Extracted from MaxText

Root Mean Square Layer Normalization used in Llama, Gemma, Qwen, etc.

Source: MaxText/layers/normalizations.py
"""

import jax
import jax.numpy as jnp
from typing import Optional
import time


def rmsnorm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """
    Root Mean Square Layer Normalization.

    Args:
        x: Input tensor [..., features]
        scale: Learnable scale parameter [features]
        epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    # Compute RMS
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(variance + epsilon)

    # Apply scale
    return x_normed * scale


def rmsnorm_fused(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """
    Fused RMSNorm implementation (potentially faster).
    """
    dtype = x.dtype
    x = x.astype(jnp.float32)

    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(variance + epsilon)

    return (x_normed * scale).astype(dtype)


# =============================================================================
# Problem Sizes from Real LLMs
# =============================================================================

LLAMA_8B = {
    'name': 'Llama-3.1-8B',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 4096,
}

LLAMA_70B = {
    'name': 'Llama-3.1-70B',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 8192,
}

GEMMA_27B = {
    'name': 'Gemma-3-27B',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 4608,
}

QWEN_72B = {
    'name': 'Qwen-2.5-72B',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 8192,
}

DEEPSEEK_V3 = {
    'name': 'DeepSeek-V3',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 7168,
}

# Batched variants
LLAMA_8B_BATCH32 = {
    'name': 'Llama-8B-Batch32',
    'batch': 32,
    'seq_len': 512,
    'hidden_dim': 4096,
}

LLAMA_70B_BATCH8 = {
    'name': 'Llama-70B-Batch8',
    'batch': 8,
    'seq_len': 1024,
    'hidden_dim': 8192,
}

PROBLEM_SIZES = [
    LLAMA_8B,
    LLAMA_70B,
    GEMMA_27B,
    QWEN_72B,
    DEEPSEEK_V3,
    LLAMA_8B_BATCH32,
    LLAMA_70B_BATCH8,
]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_rmsnorm(config: dict, warmup: int = 5, iters: int = 50):
    """Benchmark RMSNorm for a given config."""
    batch = config['batch']
    seq_len = config['seq_len']
    hidden_dim = config['hidden_dim']

    # Generate random inputs
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch, seq_len, hidden_dim), dtype=jnp.bfloat16)
    scale = jnp.ones(hidden_dim, dtype=jnp.bfloat16)

    # JIT compile
    norm_fn = jax.jit(lambda x: rmsnorm(x, scale))

    # Warmup
    for _ in range(warmup):
        output = norm_fn(x)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        output = norm_fn(x)
        output.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000

    # Memory bandwidth (read x + write output, both same size)
    bytes_moved = 2 * batch * seq_len * hidden_dim * 2  # bfloat16 = 2 bytes
    bandwidth_gbps = bytes_moved / time_ms / 1e6  # GB/s

    return {
        'config': config['name'],
        'time_ms': time_ms,
        'bandwidth_gbps': bandwidth_gbps,
        'output_shape': output.shape,
    }


def run_all_benchmarks():
    """Run benchmarks for all problem sizes."""
    print("=" * 80)
    print("RMSNORM BENCHMARK (Real LLM Sizes)")
    print("=" * 80)

    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    except:
        pass

    print()
    print(f"{'Config':<20} | {'Time (ms)':>10} | {'BW (GB/s)':>10} | {'Shape'}")
    print("-" * 70)

    results = []
    for config in PROBLEM_SIZES:
        try:
            result = benchmark_rmsnorm(config)
            results.append(result)
            print(f"{result['config']:<20} | {result['time_ms']:>10.4f} | {result['bandwidth_gbps']:>10.1f} | {result['output_shape']}")
        except Exception as e:
            print(f"{config['name']:<20} | {'FAILED':>10} | {'-':>10} | {str(e)[:20]}")

    print("-" * 70)
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
