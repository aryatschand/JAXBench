"""
YaRN (Yet another RoPE extensioN) for DeepSeek V3

Extracted from MaxText. YaRN is an interpolation method that extends
RoPE to longer contexts while preserving model quality.

Key features:
- Frequency scaling with beta_fast parameter
- Attention scaling with mscale
- Interleaved RoPE application (rope_interleave: True)
- Supports context extension from 4K to 163K tokens

DeepSeek V3 config:
- rope_type: "yarn"
- rope_theta: 10000 (base frequency)
- max_position_embeddings: 163840
- original_max_position_embeddings: 4096
- rope_factor: 40
- beta_fast: 32
"""

import jax
import jax.numpy as jnp
from functools import partial
import math

# Problem sizes from DeepSeek V3
DEEPSEEK_V3 = {
    'name': 'DeepSeek-V3-163K',
    'batch': 1,
    'seq_len': 8192,  # Testing with 8K context
    'num_heads': 128,
    'head_dim': 64,  # qk_rope_head_dim
    'rope_theta': 10000,
    'max_position_embeddings': 163840,
    'original_max_position_embeddings': 4096,
    'rope_factor': 40,
    'beta_fast': 32,
    'mscale': 1.0,
}

DEEPSEEK_V3_SMALL = {
    'name': 'DeepSeek-V3-4K',
    'batch': 1,
    'seq_len': 4096,
    'num_heads': 64,
    'head_dim': 64,
    'rope_theta': 10000,
    'max_position_embeddings': 163840,
    'original_max_position_embeddings': 4096,
    'rope_factor': 40,
    'beta_fast': 32,
    'mscale': 1.0,
}


def compute_yarn_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    max_position_embeddings: int = 163840,
    original_max_position_embeddings: int = 4096,
    rope_factor: float = 40.0,
    beta_fast: float = 32.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple:
    """
    Compute YaRN RoPE frequencies with frequency scaling.

    YaRN applies different scaling to different frequency bands:
    - Low frequencies: No scaling (preserve long-range info)
    - High frequencies: Full scaling (allow interpolation)
    - Middle frequencies: Smooth transition

    Args:
        head_dim: Dimension of each head (for rope part)
        max_seq_len: Maximum sequence length to generate
        theta: Base frequency
        max_position_embeddings: Extended context length
        original_max_position_embeddings: Original training context
        rope_factor: Interpolation factor
        beta_fast: Frequency scaling parameter

    Returns:
        (cos, sin) of shape [max_seq_len, head_dim]
    """
    dim_pairs = head_dim // 2

    # Base frequencies: 1 / (theta^(2i/d))
    base_freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))

    # YaRN frequency scaling
    # Different scaling for different frequency bands
    scale = max_position_embeddings / original_max_position_embeddings

    # Compute wavelengths
    wavelengths = 2 * math.pi / base_freqs

    # Compute interpolation ratios based on wavelength
    # Low frequency (long wavelength) -> ratio = 0 (no scaling)
    # High frequency (short wavelength) -> ratio = 1 (full scaling)
    low_freq_wavelen = original_max_position_embeddings / beta_fast
    high_freq_wavelen = original_max_position_embeddings / 1.0  # beta_slow = 1

    # Linear interpolation in log space
    smooth_ratios = jnp.clip(
        (wavelengths - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen),
        0.0, 1.0
    )

    # Apply YaRN scaling
    # scaled_freq = freq * scale^ratio
    # For ratio=0: freq unchanged (low frequency)
    # For ratio=1: freq * scale (high frequency, interpolated)
    yarn_freqs = base_freqs * (rope_factor ** (1 - smooth_ratios)) / rope_factor

    # Actually the standard YaRN formula:
    # new_base_freq = base_freq * factor^(1 - ratio)
    # This preserves low frequencies and scales high frequencies

    # Compute position encodings
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer product: [seq_len, dim_pairs]
    angles = jnp.outer(positions, yarn_freqs)

    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)

    return cos, sin


def apply_yarn_rope_interleaved(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    mscale: float = 1.0,
) -> jnp.ndarray:
    """
    Apply YaRN RoPE with interleaved format.

    DeepSeek V3 uses interleaved layout where rotations alternate:
    [x0, x1, x2, x3, ...] -> rotate pairs (x0, x1), (x2, x3), etc.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim//2]
        sin: [seq_len, head_dim//2]
        mscale: Attention scaling factor

    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    batch, seq_len, num_heads, head_dim = x.shape

    # Interleaved format: separate even and odd indices
    x_even = x[..., 0::2]  # [b, s, h, d//2]
    x_odd = x[..., 1::2]   # [b, s, h, d//2]

    # Reshape cos/sin for broadcasting
    cos = cos[:seq_len, :]  # [seq, d//2]
    sin = sin[:seq_len, :]

    cos = cos[None, :, None, :]  # [1, seq, 1, d//2]
    sin = sin[None, :, None, :]

    # Apply rotation
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave back
    output = jnp.stack([rotated_even, rotated_odd], axis=-1)
    output = output.reshape(batch, seq_len, num_heads, head_dim)

    # Apply mscale (attention scaling)
    output = output * mscale

    return output


def yarn_rope_forward(
    x: jnp.ndarray,
    rope_theta: float = 10000.0,
    max_position_embeddings: int = 163840,
    original_max_position_embeddings: int = 4096,
    rope_factor: float = 40.0,
    beta_fast: float = 32.0,
    mscale: float = 1.0,
) -> jnp.ndarray:
    """
    Full YaRN RoPE forward pass.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        Configuration parameters

    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    batch, seq_len, num_heads, head_dim = x.shape

    # Compute YaRN frequencies
    cos, sin = compute_yarn_frequencies(
        head_dim=head_dim,
        max_seq_len=seq_len,
        theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        beta_fast=beta_fast,
        dtype=x.dtype,
    )

    return apply_yarn_rope_interleaved(x, cos, sin, mscale)


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching config."""
    key = jax.random.PRNGKey(42)

    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_heads']
    head_dim = config['head_dim']

    x = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=dtype)

    return x


def benchmark_yarn_rope(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark YaRN RoPE."""
    import time

    x = create_inputs(config)

    rope_fn = jax.jit(partial(
        yarn_rope_forward,
        rope_theta=config['rope_theta'],
        max_position_embeddings=config['max_position_embeddings'],
        original_max_position_embeddings=config['original_max_position_embeddings'],
        rope_factor=config['rope_factor'],
        beta_fast=config['beta_fast'],
        mscale=config['mscale'],
    ))

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

    # Calculate bandwidth
    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_heads']
    head_dim = config['head_dim']

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


def compare_standard_vs_yarn():
    """Compare standard RoPE vs YaRN at extended context lengths."""
    print("\nYaRN vs Standard RoPE Frequency Analysis:")
    print("-" * 60)

    head_dim = 64
    theta = 10000.0
    dim_pairs = head_dim // 2

    # Standard frequencies
    standard_freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))

    # YaRN frequencies
    cos_yarn, sin_yarn = compute_yarn_frequencies(
        head_dim=head_dim,
        max_seq_len=1024,
        theta=theta,
        max_position_embeddings=163840,
        original_max_position_embeddings=4096,
        rope_factor=40.0,
        beta_fast=32.0,
    )

    print(f"  Head dim: {head_dim}")
    print(f"  Standard RoPE max freq: {float(standard_freqs[0]):.6f}")
    print(f"  Standard RoPE min freq: {float(standard_freqs[-1]):.6f}")
    print(f"  Context extension: 4K -> 163K ({163840/4096:.1f}x)")


if __name__ == '__main__':
    print("=" * 80)
    print("DEEPSEEK V3 YaRN RoPE BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    # Show frequency analysis
    compare_standard_vs_yarn()
    print()

    configs = [DEEPSEEK_V3_SMALL, DEEPSEEK_V3]

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'BW (GB/s)':>10} | Output Shape")
    print("-" * 80)

    for config in configs:
        try:
            result = benchmark_yarn_rope(config)
            print(f"{result['config']:<25} | {result['time_ms']:>10.4f} | {result['bandwidth_gbps']:>10.1f} | {result['shape']}")
        except Exception as e:
            print(f"{config['name']:<25} | ERROR: {e}")
