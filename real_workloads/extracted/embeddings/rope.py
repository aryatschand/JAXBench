"""
Rotary Position Embeddings (RoPE) - Extracted from MaxText

Position encoding used in modern LLMs (Llama, Gemma, Qwen, etc.)

Source: MaxText/layers/embeddings.py
"""

import jax
import jax.numpy as jnp
from typing import Tuple
import time


def create_sinusoidal_positions(
    seq_len: int,
    head_dim: int,
    min_timescale: float = 1.0,
    max_timescale: float = 10000.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create sinusoidal position encodings for RoPE.

    Args:
        seq_len: Sequence length
        head_dim: Dimension per attention head (must be even)
        min_timescale: Minimum timescale
        max_timescale: Maximum timescale (rope_theta)

    Returns:
        cos, sin: [seq_len, head_dim // 2]
    """
    # Frequency computation
    half_dim = head_dim // 2
    freq_exponents = jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    timescale = min_timescale * (max_timescale / min_timescale) ** freq_exponents

    # Position x frequency
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    radians = positions[:, None] / timescale[None, :]

    return jnp.cos(radians), jnp.sin(radians)


def apply_rope(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply rotary position embeddings.

    Args:
        x: Input [batch, seq_len, num_heads, head_dim]
        cos: Cosine encodings [seq_len, head_dim // 2]
        sin: Sine encodings [seq_len, head_dim // 2]

    Returns:
        Rotated tensor with same shape as input
    """
    # Split into real/imaginary parts
    x1, x2 = jnp.split(x, 2, axis=-1)

    # Rotate
    # (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    cos = cos[None, :, None, :]  # [1, seq, 1, dim//2]
    sin = sin[None, :, None, :]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return jnp.concatenate([rotated_x1, rotated_x2], axis=-1)


def apply_rope_interleaved(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply RoPE with interleaved real/imaginary pairs (Llama 3.1 style).

    In this variant, adjacent pairs (x[..., 0], x[..., 1]) form a complex number
    rather than splitting the tensor in half.
    """
    *batch_dims, seq_len, num_heads, head_dim = x.shape

    # Reshape to pairs
    x_pairs = x.reshape(*batch_dims, seq_len, num_heads, head_dim // 2, 2)

    # Expand cos/sin for broadcasting
    cos = cos[None, :, None, :, None]  # [1, seq, 1, dim//2, 1]
    sin = sin[None, :, None, :, None]

    # Rotate pairs
    x1 = x_pairs[..., 0]  # Even indices
    x2 = x_pairs[..., 1]  # Odd indices

    rotated = jnp.stack([
        x1 * cos[..., 0] - x2 * sin[..., 0],
        x1 * sin[..., 0] + x2 * cos[..., 0],
    ], axis=-1)

    return rotated.reshape(*batch_dims, seq_len, num_heads, head_dim)


# =============================================================================
# Problem Sizes from Real LLMs
# =============================================================================

LLAMA_8B = {
    'name': 'Llama-3.1-8B',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 32,
    'head_dim': 128,
    'rope_theta': 500000.0,  # Llama 3.1 uses larger theta
}

LLAMA_70B = {
    'name': 'Llama-3.1-70B',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'rope_theta': 500000.0,
}

LLAMA_70B_LONG = {
    'name': 'Llama-70B-8K',
    'batch': 1,
    'seq_len': 8192,
    'num_heads': 64,
    'head_dim': 128,
    'rope_theta': 500000.0,
}

GEMMA_27B = {
    'name': 'Gemma-3-27B',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 32,
    'head_dim': 144,
    'rope_theta': 10000.0,
}

QWEN_72B = {
    'name': 'Qwen-2.5-72B',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'rope_theta': 1000000.0,  # Qwen uses even larger theta
}

# Batched inference
LLAMA_8B_BATCH32 = {
    'name': 'Llama-8B-Batch32',
    'batch': 32,
    'seq_len': 512,
    'num_heads': 32,
    'head_dim': 128,
    'rope_theta': 500000.0,
}

PROBLEM_SIZES = [
    LLAMA_8B,
    LLAMA_70B,
    LLAMA_70B_LONG,
    GEMMA_27B,
    QWEN_72B,
    LLAMA_8B_BATCH32,
]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_rope(config: dict, warmup: int = 5, iters: int = 50):
    """Benchmark RoPE application."""
    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_heads']
    head_dim = config['head_dim']
    rope_theta = config['rope_theta']

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)

    # Precompute sin/cos
    cos, sin = create_sinusoidal_positions(seq_len, head_dim, max_timescale=rope_theta)
    cos = cos.astype(jnp.bfloat16)
    sin = sin.astype(jnp.bfloat16)

    rope_fn = jax.jit(lambda x: apply_rope(x, cos, sin))

    for _ in range(warmup):
        output = rope_fn(x)
        output.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        output = rope_fn(x)
        output.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000

    # Memory bandwidth
    bytes_moved = 2 * batch * seq_len * num_heads * head_dim * 2  # Read + write
    bandwidth_gbps = bytes_moved / time_ms / 1e6

    return {
        'config': config['name'],
        'time_ms': time_ms,
        'bandwidth_gbps': bandwidth_gbps,
    }


def run_all_benchmarks():
    """Run benchmarks for all problem sizes."""
    print("=" * 80)
    print("ROPE BENCHMARK (Real LLM Sizes)")
    print("=" * 80)

    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    except:
        pass

    print()
    print(f"{'Config':<20} | {'Time (ms)':>10} | {'BW (GB/s)':>10}")
    print("-" * 50)

    results = []
    for config in PROBLEM_SIZES:
        try:
            result = benchmark_rope(config)
            results.append(result)
            print(f"{result['config']:<20} | {result['time_ms']:>10.4f} | {result['bandwidth_gbps']:>10.1f}")
        except Exception as e:
            print(f"{config['name']:<20} | {'FAILED':>10} | {str(e)[:15]}")

    print("-" * 50)
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
