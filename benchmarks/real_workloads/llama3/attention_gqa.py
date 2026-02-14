"""
Grouped Query Attention (GQA) for Llama 3.1

Extracted from MaxText. GQA uses fewer KV heads than query heads,
reducing memory bandwidth while maintaining quality.

Llama 3.1 configs:
- 8B:  32 query heads, 8 KV heads, head_dim=128
- 70B: 64 query heads, 8 KV heads, head_dim=128
- 405B: 128 query heads, 8 KV heads, head_dim=128
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from real Llama 3.1 models
LLAMA_8B = {
    'name': 'Llama-3.1-8B',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'emb_dim': 4096,
}

LLAMA_70B = {
    'name': 'Llama-3.1-70B',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'emb_dim': 8192,
}

LLAMA_405B = {
    'name': 'Llama-3.1-405B',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 128,
    'num_kv_heads': 8,
    'head_dim': 128,
    'emb_dim': 16384,
}


def gqa_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    num_kv_heads: int,
) -> jnp.ndarray:
    """
    Grouped Query Attention (GQA) as used in Llama 3.1.

    Each KV head is shared across multiple query heads.

    Args:
        query: [batch, seq_len, num_query_heads, head_dim]
        key: [batch, seq_len, num_kv_heads, head_dim]
        value: [batch, seq_len, num_kv_heads, head_dim]
        num_kv_heads: Number of key-value heads

    Returns:
        output: [batch, seq_len, num_query_heads, head_dim]
    """
    batch, seq_len, num_query_heads, head_dim = query.shape

    # Calculate number of query heads per KV head
    num_groups = num_query_heads // num_kv_heads

    # Expand KV heads to match query heads
    # key/value: [batch, seq_len, num_kv_heads, head_dim]
    # -> [batch, seq_len, num_kv_heads, 1, head_dim]
    # -> [batch, seq_len, num_kv_heads, num_groups, head_dim]
    # -> [batch, seq_len, num_query_heads, head_dim]
    key = jnp.repeat(key[:, :, :, None, :], num_groups, axis=3)
    key = key.reshape(batch, seq_len, num_query_heads, head_dim)

    value = jnp.repeat(value[:, :, :, None, :], num_groups, axis=3)
    value = value.reshape(batch, seq_len, num_query_heads, head_dim)

    # Standard scaled dot-product attention
    # query: [batch, seq_len, num_heads, head_dim]
    # key:   [batch, seq_len, num_heads, head_dim]
    scale = head_dim ** -0.5

    # Compute attention scores
    # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # -> [batch, num_heads, seq_len, seq_len]
    query = query.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)

    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

    # Causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn_weights = jnp.where(mask, attn_weights, -1e9)

    # Softmax
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # Apply attention to values
    output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)

    # Reshape back to [batch, seq_len, num_heads, head_dim]
    output = output.transpose(0, 2, 1, 3)

    return output


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching Llama config."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    batch = config['batch']
    seq_len = config['seq_len']
    num_query_heads = config['num_query_heads']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']

    query = jax.random.normal(k1, (batch, seq_len, num_query_heads, head_dim), dtype=dtype)
    key_tensor = jax.random.normal(k2, (batch, seq_len, num_kv_heads, head_dim), dtype=dtype)
    value = jax.random.normal(k3, (batch, seq_len, num_kv_heads, head_dim), dtype=dtype)

    return query, key_tensor, value


def benchmark_gqa(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark GQA attention for a given config."""
    import time

    query, key, value = create_inputs(config)

    # JIT compile
    gqa_fn = jax.jit(partial(gqa_attention, num_kv_heads=config['num_kv_heads']))

    # Warmup
    for _ in range(num_warmup):
        output = gqa_fn(query, key, value)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = gqa_fn(query, key, value)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate FLOPS
    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_query_heads']
    head_dim = config['head_dim']

    # QK: batch * heads * seq * seq * head_dim * 2
    # PV: batch * heads * seq * seq * head_dim * 2
    flops = batch * num_heads * seq_len * seq_len * head_dim * 4
    tflops = flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'],
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("LLAMA 3.1 GQA ATTENTION BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [LLAMA_8B, LLAMA_70B, LLAMA_405B]

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | Output Shape")
    print("-" * 80)

    for config in configs:
        result = benchmark_gqa(config)
        print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f} | {result['shape']}")
