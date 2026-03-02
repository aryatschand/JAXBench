"""Relative Position Attention — T5-Base (google/t5-v1_1-base).

Encoder self-attention with learned relative position bias using logarithmic
bucketing. No absolute position embeddings.

Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py
Paper: "Exploring the Limits of Transfer Learning with T5" (Raffel et al., 2020)
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 't5_base_relative_attention',
    'model': 'T5-Base',
    'operator': 'relative_position_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 12,
    'head_dim': 64,
    'd_model': 768,
    'relative_attention_num_buckets': 32,
    'relative_attention_max_distance': 128,
    'bidirectional': True,
}


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    """T5-style logarithmic bucketing of relative positions.

    Small distances get exact buckets; larger distances share log-spaced buckets.
    Bidirectional: buckets split for left/right. Unidirectional: all for backward.
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
        relative_position = jnp.abs(relative_position)
    else:
        relative_position = -jnp.clip(relative_position, a_max=0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        jnp.log(relative_position.astype(jnp.float32) / max_exact)
        / jnp.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)
    return relative_buckets + jnp.where(is_small, relative_position, relative_position_if_large)


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, relative_bias_table)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    num_buckets = CONFIG['relative_attention_num_buckets']
    query = jax.random.normal(k1, (B, S, H, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, H, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, H, D), dtype=dtype)
    # Learned bias table: (num_buckets, num_heads) — T5 Embed layer
    bias_table = jax.random.normal(k4, (num_buckets, H), dtype=dtype) * 0.02
    return query, key_t, value, bias_table


def workload(query, key, value, bias_table):
    """T5 encoder self-attention with relative position bias.

    Key difference from standard attention: position information comes from
    a learned bias added to logits, not from position embeddings in the input.
    T5 does NOT scale by 1/sqrt(d_k) — raw dot products go into softmax.
    """
    B, S, H, D = query.shape
    num_buckets = CONFIG['relative_attention_num_buckets']
    max_distance = CONFIG['relative_attention_max_distance']
    bidirectional = CONFIG['bidirectional']
    # Compute relative position bias
    context_pos = jnp.arange(S)[:, None]   # (S, 1)
    memory_pos = jnp.arange(S)[None, :]    # (1, S)
    relative_position = memory_pos - context_pos  # (S, S)
    buckets = _relative_position_bucket(
        relative_position, bidirectional, num_buckets, max_distance
    )  # (S, S) int32
    # Lookup: (S, S) -> (S, S, H)
    position_bias = bias_table[buckets]
    # Transpose to (1, H, S, S) for broadcasting
    position_bias = position_bias.transpose(2, 0, 1)[None, :, :, :]
    # Attention (T5 uses unscaled dot products)
    q = query.transpose(0, 2, 1, 3)   # (B, H, S, D)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k)  # no scaling
    attn = attn + position_bias
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    flops = B * H * S * S * D * 4
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
