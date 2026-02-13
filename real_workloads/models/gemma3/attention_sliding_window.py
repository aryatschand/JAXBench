"""
Sliding Window Attention for Gemma 3

Extracted from MaxText. Gemma 3 uses a hybrid attention pattern:
- 5 layers of LOCAL_SLIDING window attention
- 1 layer of GLOBAL attention
This pattern repeats throughout the model.

Unique features:
- Sliding window size: 4096 tokens for local attention
- QK normalization
- Attention logit soft capping

Gemma 3 configs:
- 4B:  num_query_heads=8, num_kv_heads=4, head_dim=256
- 12B: num_query_heads=16, num_kv_heads=8, head_dim=256
- 27B: num_query_heads=32, num_kv_heads=16, head_dim=144
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from real Gemma 3 models
GEMMA3_4B = {
    'name': 'Gemma-3-4B',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 8,
    'num_kv_heads': 4,
    'head_dim': 256,
    'emb_dim': 2048,
    'sliding_window': 4096,
    'attn_logits_soft_cap': 50.0,
}

GEMMA3_12B = {
    'name': 'Gemma-3-12B',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 16,
    'num_kv_heads': 8,
    'head_dim': 256,
    'emb_dim': 3584,
    'sliding_window': 4096,
    'attn_logits_soft_cap': 50.0,
}

GEMMA3_27B = {
    'name': 'Gemma-3-27B',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 32,
    'num_kv_heads': 16,
    'head_dim': 144,
    'emb_dim': 4608,
    'sliding_window': 4096,
    'attn_logits_soft_cap': 50.0,
}


def qk_normalize(q: jnp.ndarray, k: jnp.ndarray) -> tuple:
    """
    Apply QK normalization as used in Gemma 3.

    Normalizes query and key vectors to unit length before attention.

    Args:
        q: [batch, seq, heads, dim]
        k: [batch, seq, heads, dim]

    Returns:
        Normalized (q, k)
    """
    # L2 normalize along head dimension
    q_norm = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k_norm = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    return q_norm, k_norm


def soft_cap_logits(logits: jnp.ndarray, cap: float) -> jnp.ndarray:
    """
    Apply soft capping to attention logits.

    Prevents extreme attention scores using tanh squashing.

    Args:
        logits: Attention logits
        cap: Soft cap value (e.g., 50.0)

    Returns:
        Capped logits
    """
    return cap * jnp.tanh(logits / cap)


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
) -> jnp.ndarray:
    """
    Create a sliding window causal mask.

    Each token can only attend to tokens within window_size positions
    before it (plus itself).

    Args:
        seq_len: Sequence length
        window_size: Sliding window size

    Returns:
        Mask of shape [seq_len, seq_len]
    """
    # Create position indices
    positions = jnp.arange(seq_len)
    # Distance matrix: query_pos - key_pos
    distances = positions[:, None] - positions[None, :]
    # Valid if: 0 <= distance < window_size (causal + window)
    mask = (distances >= 0) & (distances < window_size)
    return mask.astype(jnp.float32)


def sliding_window_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    window_size: int,
    soft_cap: float = None,
    use_qk_norm: bool = True,
) -> jnp.ndarray:
    """
    Sliding Window Attention as used in Gemma 3.

    Args:
        query: [batch, seq_len, num_query_heads, head_dim]
        key: [batch, seq_len, num_kv_heads, head_dim]
        value: [batch, seq_len, num_kv_heads, head_dim]
        window_size: Sliding window size
        soft_cap: Optional soft cap for attention logits
        use_qk_norm: Whether to apply QK normalization

    Returns:
        output: [batch, seq_len, num_query_heads, head_dim]
    """
    batch, seq_len, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]

    # Apply QK normalization if enabled
    if use_qk_norm:
        query, key = qk_normalize(query, key)

    # Expand KV heads to match query heads (GQA)
    num_groups = num_query_heads // num_kv_heads
    key = jnp.repeat(key[:, :, :, None, :], num_groups, axis=3)
    key = key.reshape(batch, seq_len, num_query_heads, head_dim)
    value = jnp.repeat(value[:, :, :, None, :], num_groups, axis=3)
    value = value.reshape(batch, seq_len, num_query_heads, head_dim)

    # Transpose for attention computation
    query = query.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)

    # Compute attention scores
    scale = head_dim ** -0.5
    attn_logits = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

    # Apply soft cap if specified
    if soft_cap is not None:
        attn_logits = soft_cap_logits(attn_logits, soft_cap)

    # Create and apply sliding window mask
    mask = create_sliding_window_mask(seq_len, window_size)
    attn_logits = jnp.where(mask, attn_logits, -1e9)

    # Softmax
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    # Apply attention to values
    output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)

    # Reshape back
    output = output.transpose(0, 2, 1, 3)

    return output


def global_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    soft_cap: float = None,
    use_qk_norm: bool = True,
) -> jnp.ndarray:
    """
    Global (full causal) attention layer in Gemma 3.

    Every 6th layer uses global attention instead of sliding window.

    Args:
        query: [batch, seq_len, num_query_heads, head_dim]
        key: [batch, seq_len, num_kv_heads, head_dim]
        value: [batch, seq_len, num_kv_heads, head_dim]
        soft_cap: Optional soft cap for attention logits
        use_qk_norm: Whether to apply QK normalization

    Returns:
        output: [batch, seq_len, num_query_heads, head_dim]
    """
    batch, seq_len, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]

    # Apply QK normalization if enabled
    if use_qk_norm:
        query, key = qk_normalize(query, key)

    # Expand KV heads
    num_groups = num_query_heads // num_kv_heads
    key = jnp.repeat(key[:, :, :, None, :], num_groups, axis=3)
    key = key.reshape(batch, seq_len, num_query_heads, head_dim)
    value = jnp.repeat(value[:, :, :, None, :], num_groups, axis=3)
    value = value.reshape(batch, seq_len, num_query_heads, head_dim)

    # Transpose
    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)

    # Compute attention
    scale = head_dim ** -0.5
    attn_logits = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

    if soft_cap is not None:
        attn_logits = soft_cap_logits(attn_logits, soft_cap)

    # Full causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn_logits = jnp.where(mask, attn_logits, -1e9)

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)
    output = output.transpose(0, 2, 1, 3)

    return output


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching config."""
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


def benchmark_sliding_window(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark sliding window attention."""
    import time

    query, key, value = create_inputs(config)

    attn_fn = jax.jit(partial(
        sliding_window_attention,
        window_size=config['sliding_window'],
        soft_cap=config['attn_logits_soft_cap'],
        use_qk_norm=True,
    ))

    # Warmup
    for _ in range(num_warmup):
        output = attn_fn(query, key, value)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = attn_fn(query, key, value)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate FLOPS
    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_query_heads']
    head_dim = config['head_dim']

    # Note: sliding window reduces effective seq_len for attention
    effective_seq = min(seq_len, config['sliding_window'])
    flops = batch * num_heads * seq_len * effective_seq * head_dim * 4
    tflops = flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'] + '-SW',
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
    }


def benchmark_global(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark global attention."""
    import time

    query, key, value = create_inputs(config)

    attn_fn = jax.jit(partial(
        global_attention,
        soft_cap=config['attn_logits_soft_cap'],
        use_qk_norm=True,
    ))

    for _ in range(num_warmup):
        output = attn_fn(query, key, value)
        output.block_until_ready()

    start = time.perf_counter()
    for _ in range(num_iters):
        output = attn_fn(query, key, value)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_query_heads']
    head_dim = config['head_dim']

    flops = batch * num_heads * seq_len * seq_len * head_dim * 4
    tflops = flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'] + '-Global',
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("GEMMA 3 SLIDING WINDOW + GLOBAL ATTENTION BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [GEMMA3_4B, GEMMA3_12B, GEMMA3_27B]

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | Output Shape")
    print("-" * 80)

    for config in configs:
        # Benchmark sliding window (5/6 of layers)
        result_sw = benchmark_sliding_window(config)
        print(f"{result_sw['config']:<25} | {result_sw['time_ms']:>10.2f} | {result_sw['tflops']:>8.1f} | {result_sw['shape']}")

        # Benchmark global (1/6 of layers)
        result_global = benchmark_global(config)
        print(f"{result_global['config']:<25} | {result_global['time_ms']:>10.2f} | {result_global['tflops']:>8.1f} | {result_global['shape']}")
