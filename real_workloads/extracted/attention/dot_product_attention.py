"""
Dot-Product Attention - Extracted from MaxText

This is the standard scaled dot-product attention used as baseline.
Real problem sizes from production LLMs.

Source: MaxText/layers/attentions.py
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import time


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
) -> jnp.ndarray:
    """
    Scaled dot-product attention.

    Args:
        query: [batch, seq_len, num_heads, head_dim]
        key: [batch, kv_len, num_kv_heads, head_dim]
        value: [batch, kv_len, num_kv_heads, head_dim]
        mask: Optional attention mask
        scale: Scaling factor (default: 1/sqrt(head_dim))
        dropout_rate: Dropout probability
        deterministic: Whether to apply dropout

    Returns:
        Output: [batch, seq_len, num_heads, head_dim]
    """
    head_dim = query.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5

    # Handle GQA: repeat KV heads if needed
    num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        key = jnp.repeat(key, repeat_factor, axis=2)
        value = jnp.repeat(value, repeat_factor, axis=2)

    # Compute attention scores
    # [batch, num_heads, seq_len, kv_len]
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key) * scale

    # Apply mask if provided
    if mask is not None:
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

    # Softmax
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # Apply attention to values
    # [batch, seq_len, num_heads, head_dim]
    output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)

    return output


def create_causal_mask(seq_len: int, kv_len: int) -> jnp.ndarray:
    """Create causal attention mask."""
    return jnp.tril(jnp.ones((seq_len, kv_len), dtype=jnp.bool_))


# =============================================================================
# Problem Sizes from Real LLMs
# =============================================================================

LLAMA_70B = {
    'name': 'Llama-3.1-70B',
    'batch': 1,
    'seq_len': 2048,
    'kv_len': 2048,
    'num_heads': 64,
    'num_kv_heads': 8,  # GQA
    'head_dim': 128,
}

LLAMA_8B = {
    'name': 'Llama-3.1-8B',
    'batch': 1,
    'seq_len': 2048,
    'kv_len': 2048,
    'num_heads': 32,
    'num_kv_heads': 8,  # GQA
    'head_dim': 128,
}

GEMMA_27B = {
    'name': 'Gemma-3-27B',
    'batch': 1,
    'seq_len': 2048,
    'kv_len': 2048,
    'num_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 144,
}

QWEN_72B = {
    'name': 'Qwen-2.5-72B',
    'batch': 1,
    'seq_len': 2048,
    'kv_len': 2048,
    'num_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}

# Long context variants
LLAMA_70B_LONG = {
    'name': 'Llama-3.1-70B-8K',
    'batch': 1,
    'seq_len': 8192,
    'kv_len': 8192,
    'num_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}

# Batched inference
LLAMA_8B_BATCHED = {
    'name': 'Llama-3.1-8B-Batch32',
    'batch': 32,
    'seq_len': 512,
    'kv_len': 512,
    'num_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
}

PROBLEM_SIZES = [
    LLAMA_8B,
    LLAMA_70B,
    GEMMA_27B,
    QWEN_72B,
    LLAMA_70B_LONG,
    LLAMA_8B_BATCHED,
]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_attention(config: dict, warmup: int = 5, iters: int = 20):
    """Benchmark dot-product attention for a given config."""
    batch = config['batch']
    seq_len = config['seq_len']
    kv_len = config['kv_len']
    num_heads = config['num_heads']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']

    # Generate random inputs
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    query = jax.random.normal(k1, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
    kv_key = jax.random.normal(k2, (batch, kv_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (batch, kv_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    # Causal mask
    mask = create_causal_mask(seq_len, kv_len)
    mask = mask[None, None, :, :]  # [1, 1, seq, kv]

    # JIT compile
    attn_fn = jax.jit(lambda q, k, v: dot_product_attention(q, k, v, mask=mask))

    # Warmup
    for _ in range(warmup):
        output = attn_fn(query, kv_key, value)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        output = attn_fn(query, kv_key, value)
        output.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000

    # Compute FLOPs
    # QK: 2 * batch * heads * seq * kv * head_dim
    # Softmax: ~5 * batch * heads * seq * kv
    # AV: 2 * batch * heads * seq * kv * head_dim
    qk_flops = 2 * batch * num_heads * seq_len * kv_len * head_dim
    av_flops = 2 * batch * num_heads * seq_len * kv_len * head_dim
    total_flops = qk_flops + av_flops
    tflops = total_flops / time_ms / 1e9  # TFLOPS

    return {
        'config': config['name'],
        'time_ms': time_ms,
        'tflops': tflops,
        'output_shape': output.shape,
    }


def run_all_benchmarks():
    """Run benchmarks for all problem sizes."""
    print("=" * 80)
    print("DOT-PRODUCT ATTENTION BENCHMARK (Real LLM Sizes)")
    print("=" * 80)

    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    except:
        pass

    print()
    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | {'Output Shape'}")
    print("-" * 80)

    results = []
    for config in PROBLEM_SIZES:
        try:
            result = benchmark_attention(config)
            results.append(result)
            print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f} | {result['output_shape']}")
        except Exception as e:
            print(f"{config['name']:<25} | {'FAILED':>10} | {'-':>8} | {str(e)[:30]}")

    print("-" * 80)
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
