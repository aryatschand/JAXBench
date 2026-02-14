"""
Rotary Position Embedding (RoPE) for Llama 3.1

Extracted from MaxText. Llama 3.1 uses RoPE with theta=500,000
for extended context length support up to 128K tokens.

Key parameters:
- rope_max_timescale (theta): 500,000 for Llama 3.1
- head_dim: 128
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from real Llama 3.1 models
LLAMA_8B = {
    'name': 'Llama-3.1-8B',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 32,
    'head_dim': 128,
    'rope_theta': 500_000,
}

LLAMA_70B = {
    'name': 'Llama-3.1-70B',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 64,
    'head_dim': 128,
    'rope_theta': 500_000,
}

# Long context scenario
LLAMA_8B_8K = {
    'name': 'Llama-8B-8K',
    'batch': 1,
    'seq_len': 8192,
    'num_heads': 32,
    'head_dim': 128,
    'rope_theta': 500_000,
}

# Batch inference
LLAMA_8B_BATCH32 = {
    'name': 'Llama-8B-Batch32',
    'batch': 32,
    'seq_len': 512,
    'num_heads': 32,
    'head_dim': 128,
    'rope_theta': 500_000,
}


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple:
    """
    Precompute RoPE sin/cos frequencies.

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency (default 10000, Llama 3.1 uses 500000)
        dtype: Output dtype

    Returns:
        (cos, sin) of shape [max_seq_len, head_dim//2]
    """
    # Compute frequency for each dimension pair
    # freq[i] = 1 / (theta^(2i/d)) for i in [0, d/2)
    dim_pairs = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))

    # Compute position encodings
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer product: [seq_len, dim_pairs]
    angles = jnp.outer(positions, freqs)

    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)

    return cos, sin


def apply_rope(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    positions: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Apply Rotary Position Embedding to input tensor.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        cos: [max_seq_len, head_dim//2] precomputed cosines
        sin: [max_seq_len, head_dim//2] precomputed sines
        positions: Optional [batch, seq_len] position indices

    Returns:
        output: [batch, seq_len, num_heads, head_dim] with RoPE applied
    """
    batch, seq_len, num_heads, head_dim = x.shape

    # Get cos/sin for current sequence length
    cos_pos = cos[:seq_len]  # [seq_len, head_dim//2]
    sin_pos = sin[:seq_len]

    # Reshape for broadcasting: [seq, dim//2] -> [1, seq, 1, dim//2]
    cos_pos = cos_pos[None, :, None, :]
    sin_pos = sin_pos[None, :, None, :]

    # Split x into two halves for rotation
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    # Apply rotation
    # x_rotated = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    rotated_x1 = x1 * cos_pos - x2 * sin_pos
    rotated_x2 = x1 * sin_pos + x2 * cos_pos

    output = jnp.concatenate([rotated_x1, rotated_x2], axis=-1)

    return output


def rope_forward(
    x: jnp.ndarray,
    theta: float = 500000.0,
) -> jnp.ndarray:
    """
    Full RoPE forward pass (computes frequencies inline).

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        theta: RoPE base frequency

    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    batch, seq_len, num_heads, head_dim = x.shape

    # Compute frequencies
    cos, sin = compute_rope_frequencies(head_dim, seq_len, theta, x.dtype)

    return apply_rope(x, cos, sin)


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching config."""
    key = jax.random.PRNGKey(42)

    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_heads']
    head_dim = config['head_dim']

    x = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=dtype)

    return x


def benchmark_rope(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark RoPE for a given config."""
    import time

    x = create_inputs(config)
    theta = config['rope_theta']

    # JIT compile
    rope_fn = jax.jit(partial(rope_forward, theta=theta))

    # Warmup
    for _ in range(num_warmup):
        output = rope_fn(x)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = rope_fn(x)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate bandwidth (memory bound operation)
    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_heads']
    head_dim = config['head_dim']

    # Reads: x + cos + sin
    # Writes: output
    bytes_read = batch * seq_len * num_heads * head_dim * 2  # bfloat16
    bytes_written = batch * seq_len * num_heads * head_dim * 2
    total_bytes = bytes_read + bytes_written
    bandwidth_gbps = total_bytes / (avg_time_ms / 1000) / 1e9

    return {
        'config': config['name'],
        'time_ms': avg_time_ms,
        'bandwidth_gbps': bandwidth_gbps,
        'shape': list(output.shape),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("LLAMA 3.1 RoPE BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [LLAMA_8B, LLAMA_70B, LLAMA_8B_8K, LLAMA_8B_BATCH32]

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'BW (GB/s)':>10} | Output Shape")
    print("-" * 80)

    for config in configs:
        result = benchmark_rope(config)
        print(f"{result['config']:<25} | {result['time_ms']:>10.4f} | {result['bandwidth_gbps']:>10.1f} | {result['shape']}")
