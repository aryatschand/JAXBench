"""
Multi-head Latent Attention (MLA) for DeepSeek V3

Extracted from MaxText. MLA is a novel attention mechanism that compresses
key-value states into a lower-dimensional latent space, significantly reducing
KV cache memory during inference.

Key innovations:
1. Low-rank compression of KV: Instead of storing full K,V, compress to latent
2. Separate dimensions for position (RoPE) and content (NoPE)
3. q_lora_rank and kv_lora_rank for compression

DeepSeek V3 config (671B):
- q_lora_rank: 1536
- kv_lora_rank: 512
- qk_nope_head_dim: 128 (content)
- qk_rope_head_dim: 64 (position with RoPE)
- v_head_dim: 128
- num_heads: 128
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from DeepSeek V3
DEEPSEEK_V3 = {
    'name': 'DeepSeek-V3-671B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 7168,
    'num_heads': 128,
    'q_lora_rank': 1536,
    'kv_lora_rank': 512,
    'qk_nope_head_dim': 128,  # Content attention
    'qk_rope_head_dim': 64,   # Position attention (with RoPE)
    'v_head_dim': 128,
    'rope_theta': 10000,
}

# Smaller test config
DEEPSEEK_V3_SMALL = {
    'name': 'DeepSeek-V3-Small',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 4096,
    'num_heads': 64,
    'q_lora_rank': 768,
    'kv_lora_rank': 256,
    'qk_nope_head_dim': 64,
    'qk_rope_head_dim': 32,
    'v_head_dim': 64,
    'rope_theta': 10000,
}


def compute_rope_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Compute RoPE sin/cos frequencies."""
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply RoPE to the input tensor."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Reshape cos/sin for broadcasting: [seq, dim//2] -> [1, seq, 1, dim//2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    return rotated.reshape(x.shape)


def mla_forward(
    x: jnp.ndarray,
    # Projection weights
    q_down_proj: jnp.ndarray,  # [emb_dim, q_lora_rank]
    q_up_proj: jnp.ndarray,    # [q_lora_rank, num_heads * (qk_nope + qk_rope)]
    kv_down_proj: jnp.ndarray, # [emb_dim, kv_lora_rank + qk_rope]
    k_up_proj: jnp.ndarray,    # [kv_lora_rank, num_heads * qk_nope]
    v_up_proj: jnp.ndarray,    # [kv_lora_rank, num_heads * v_head_dim]
    o_proj: jnp.ndarray,       # [num_heads * v_head_dim, emb_dim]
    # Config
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope_theta: float = 10000.0,
) -> jnp.ndarray:
    """
    Multi-head Latent Attention forward pass.

    MLA compresses KV into a latent space and separates position (RoPE) from content.

    Architecture:
    1. Query: x -> q_down -> q_up -> [q_nope, q_rope] per head
    2. Key: x -> kv_down -> [k_latent, k_rope_raw]
           k_nope = k_latent @ k_up (decompressed per head)
           k_rope = apply_rope(k_rope_raw)
    3. Value: k_latent @ v_up (decompressed per head)
    4. Attention: separate attention for nope and rope parts, then combine
    5. Output projection

    Args:
        x: [batch, seq_len, emb_dim]
        Various projection weights
        Config parameters

    Returns:
        output: [batch, seq_len, emb_dim]
    """
    batch, seq_len, emb_dim = x.shape
    kv_lora_rank = kv_down_proj.shape[-1] - qk_rope_head_dim

    # ============ Query Projection ============
    # Low-rank factorization: x -> q_latent -> q
    q_latent = jnp.dot(x, q_down_proj)  # [b, s, q_lora_rank]
    q = jnp.dot(q_latent, q_up_proj)    # [b, s, num_heads * (nope + rope)]

    # Reshape and split into nope/rope
    q = q.reshape(batch, seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim)
    q_nope = q[..., :qk_nope_head_dim]  # [b, s, h, nope_dim]
    q_rope = q[..., qk_nope_head_dim:]  # [b, s, h, rope_dim]

    # ============ Key-Value Compression ============
    # Compress to latent space with shared rope
    kv_compressed = jnp.dot(x, kv_down_proj)  # [b, s, kv_lora_rank + rope_dim]

    # Split latent and rope parts
    k_latent = kv_compressed[..., :kv_lora_rank]     # [b, s, kv_lora_rank]
    k_rope_raw = kv_compressed[..., kv_lora_rank:]  # [b, s, rope_dim]

    # Decompress key (content part)
    k_nope = jnp.dot(k_latent, k_up_proj)  # [b, s, num_heads * nope_dim]
    k_nope = k_nope.reshape(batch, seq_len, num_heads, qk_nope_head_dim)

    # Apply RoPE to position parts
    cos, sin = compute_rope_frequencies(qk_rope_head_dim, seq_len, rope_theta)
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)

    # Expand k_rope_raw for all heads (shared across heads)
    k_rope = k_rope_raw[:, :, None, :]  # [b, s, 1, rope_dim]
    k_rope = jnp.broadcast_to(k_rope, (batch, seq_len, num_heads, qk_rope_head_dim))

    q_rope = apply_rope(q_rope, cos, sin)
    k_rope = apply_rope(k_rope, cos, sin)

    # Decompress value
    v = jnp.dot(k_latent, v_up_proj)  # [b, s, num_heads * v_dim]
    v = v.reshape(batch, seq_len, num_heads, v_head_dim)

    # ============ Attention Computation ============
    # Combine nope and rope parts for full query/key
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1)  # [b, s, h, nope+rope]
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1)

    # Transpose for attention: [b, h, s, d]
    q_full = q_full.transpose(0, 2, 1, 3)
    k_full = k_full.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = head_dim ** -0.5

    attn_logits = jnp.einsum('bhqd,bhkd->bhqk', q_full, k_full) * scale

    # Causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn_logits = jnp.where(mask, attn_logits, -1e9)

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    # Apply attention to values
    attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

    # Reshape back: [b, s, h*v_dim]
    attn_output = attn_output.transpose(0, 2, 1, 3)
    attn_output = attn_output.reshape(batch, seq_len, num_heads * v_head_dim)

    # Output projection
    output = jnp.dot(attn_output, o_proj)

    return output


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs and weights matching config."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)

    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    num_heads = config['num_heads']
    q_lora_rank = config['q_lora_rank']
    kv_lora_rank = config['kv_lora_rank']
    qk_nope = config['qk_nope_head_dim']
    qk_rope = config['qk_rope_head_dim']
    v_dim = config['v_head_dim']

    x = jax.random.normal(keys[0], (batch, seq_len, emb_dim), dtype=dtype)

    # Query projections
    q_down_proj = jax.random.normal(keys[1], (emb_dim, q_lora_rank), dtype=dtype) * 0.02
    q_up_proj = jax.random.normal(keys[2], (q_lora_rank, num_heads * (qk_nope + qk_rope)), dtype=dtype) * 0.02

    # KV projections
    kv_down_proj = jax.random.normal(keys[3], (emb_dim, kv_lora_rank + qk_rope), dtype=dtype) * 0.02
    k_up_proj = jax.random.normal(keys[4], (kv_lora_rank, num_heads * qk_nope), dtype=dtype) * 0.02
    v_up_proj = jax.random.normal(keys[5], (kv_lora_rank, num_heads * v_dim), dtype=dtype) * 0.02

    # Output projection
    o_proj = jax.random.normal(keys[6], (num_heads * v_dim, emb_dim), dtype=dtype) * 0.02

    return x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj


def benchmark_mla(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark MLA attention."""
    import time

    x, q_down, q_up, kv_down, k_up, v_up, o_proj = create_inputs(config)

    mla_fn = jax.jit(partial(
        mla_forward,
        num_heads=config['num_heads'],
        qk_nope_head_dim=config['qk_nope_head_dim'],
        qk_rope_head_dim=config['qk_rope_head_dim'],
        v_head_dim=config['v_head_dim'],
        rope_theta=config['rope_theta'],
    ))

    # Warmup
    for _ in range(num_warmup):
        output = mla_fn(x, q_down, q_up, kv_down, k_up, v_up, o_proj)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = mla_fn(x, q_down, q_up, kv_down, k_up, v_up, o_proj)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate approximate FLOPS
    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    num_heads = config['num_heads']
    q_lora = config['q_lora_rank']
    kv_lora = config['kv_lora_rank']
    head_dim = config['qk_nope_head_dim'] + config['qk_rope_head_dim']

    # Projections + attention
    proj_flops = (
        batch * seq_len * emb_dim * q_lora * 2 +  # q_down
        batch * seq_len * q_lora * (num_heads * head_dim) * 2 +  # q_up
        batch * seq_len * emb_dim * (kv_lora + config['qk_rope_head_dim']) * 2 +  # kv_down
        batch * seq_len * kv_lora * (num_heads * config['qk_nope_head_dim']) * 2 +  # k_up
        batch * seq_len * kv_lora * (num_heads * config['v_head_dim']) * 2 +  # v_up
        batch * seq_len * (num_heads * config['v_head_dim']) * emb_dim * 2  # o_proj
    )
    attn_flops = batch * num_heads * seq_len * seq_len * head_dim * 4  # QK + PV

    total_flops = proj_flops + attn_flops
    tflops = total_flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'],
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
    }


def compute_kv_cache_savings(config: dict):
    """
    Compute KV cache memory savings from MLA compression.

    Standard attention: stores full K, V per layer
    MLA: stores compressed kv_latent (kv_lora_rank) + k_rope_raw (qk_rope_head_dim)
    """
    num_heads = config['num_heads']
    standard_kv_dim = num_heads * (config['qk_nope_head_dim'] + config['qk_rope_head_dim'] + config['v_head_dim'])
    mla_kv_dim = config['kv_lora_rank'] + config['qk_rope_head_dim']

    compression_ratio = standard_kv_dim / mla_kv_dim
    return compression_ratio


if __name__ == '__main__':
    print("=" * 80)
    print("DEEPSEEK V3 MULTI-HEAD LATENT ATTENTION (MLA) BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [DEEPSEEK_V3_SMALL, DEEPSEEK_V3]

    print("KV Cache Compression Analysis:")
    print("-" * 40)
    for config in configs:
        ratio = compute_kv_cache_savings(config)
        print(f"  {config['name']}: {ratio:.1f}x compression")
    print()

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | Output Shape")
    print("-" * 80)

    for config in configs:
        try:
            result = benchmark_mla(config)
            print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f} | {result['shape']}")
        except Exception as e:
            print(f"{config['name']:<25} | ERROR: {e}")
