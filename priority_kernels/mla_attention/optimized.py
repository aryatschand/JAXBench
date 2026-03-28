"""MLA Attention (Optimized) — DeepSeek-V3-671B with jax.nn.dot_product_attention.

Multi-head Latent Attention keeps the LoRA projection structure but replaces
the manual attention computation with jax.nn.dot_product_attention.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'deepseek_v3_mla_optimized',
    'model': 'DeepSeek-V3-671B',
    'operator': 'mla_attention',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 7168,
    'num_heads': 128,
    'q_lora_rank': 1536,
    'kv_lora_rank': 512,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'rope_theta': 10000,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o, rope_cos, rope_sin)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 9)
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    H = CONFIG['num_heads']
    q_lora = CONFIG['q_lora_rank']
    kv_lora = CONFIG['kv_lora_rank']
    d_nope = CONFIG['qk_nope_head_dim']
    d_rope = CONFIG['qk_rope_head_dim']
    d_v = CONFIG['v_head_dim']
    x = jax.random.normal(keys[0], (B, S, D), dtype=dtype)
    W_dq = jax.random.normal(keys[1], (D, q_lora), dtype=dtype) * 0.01
    W_uq = jax.random.normal(keys[2], (q_lora, H * (d_nope + d_rope)), dtype=dtype) * 0.01
    W_dkv = jax.random.normal(keys[3], (D, kv_lora), dtype=dtype) * 0.01
    W_uk = jax.random.normal(keys[4], (kv_lora, H * d_nope), dtype=dtype) * 0.01
    W_uv = jax.random.normal(keys[5], (kv_lora, H * d_v), dtype=dtype) * 0.01
    W_o = jax.random.normal(keys[6], (H * d_v, D), dtype=dtype) * 0.01
    pos = jnp.arange(S)
    freqs = 1.0 / (CONFIG['rope_theta'] ** (jnp.arange(0, d_rope, 2, dtype=jnp.float32) / d_rope))
    angles = jnp.outer(pos, freqs)
    rope_cos = jnp.cos(angles).astype(dtype)
    rope_sin = jnp.sin(angles).astype(dtype)
    return x, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o, rope_cos, rope_sin


def _apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def workload(x, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o, rope_cos, rope_sin):
    """MLA with optimized dot_product_attention for the core attention."""
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H = CONFIG['num_heads']
    d_nope = CONFIG['qk_nope_head_dim']
    d_rope = CONFIG['qk_rope_head_dim']
    d_v = CONFIG['v_head_dim']

    # LoRA projections (same as baseline)
    q_compressed = jnp.dot(x, W_dq)
    q_full = jnp.dot(q_compressed, W_uq).reshape(B, S, H, d_nope + d_rope)
    q_nope, q_rope = q_full[..., :d_nope], q_full[..., d_nope:]

    kv_compressed = jnp.dot(x, W_dkv)
    k_nope = jnp.dot(kv_compressed, W_uk).reshape(B, S, H, d_nope)
    v = jnp.dot(kv_compressed, W_uv).reshape(B, S, H, d_v)

    # Apply RoPE
    q_rope = _apply_rope(q_rope, rope_cos[None, :, None, :], rope_sin[None, :, None, :])
    k_rope = _apply_rope(
        jnp.broadcast_to(kv_compressed[:, :, None, :d_rope], (B, S, H, d_rope)),
        rope_cos[None, :, None, :], rope_sin[None, :, None, :]
    )

    # Concatenate q and k components — keep (B, S, H, D) layout for dot_product_attention
    q = jnp.concatenate([q_nope, q_rope], axis=-1)  # (B, S, H, d_nope+d_rope)
    k = jnp.concatenate([k_nope, k_rope], axis=-1)  # (B, S, H, d_nope+d_rope)
    # v already (B, S, H, d_v)

    # Optimized attention
    attn_out = jax.nn.dot_product_attention(
        q, k, v,
        is_causal=True,
    )  # (B, S, H, d_v)

    attn_out = attn_out.reshape(B, S, H * d_v)
    return jnp.dot(attn_out, W_o)


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
    B, S, D, H = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['num_heads']
    d_qk = CONFIG['qk_nope_head_dim'] + CONFIG['qk_rope_head_dim']
    d_v = CONFIG['v_head_dim']
    proj_flops = 2 * B * S * (D * CONFIG['q_lora_rank'] + CONFIG['q_lora_rank'] * H * d_qk +
                                D * CONFIG['kv_lora_rank'] + CONFIG['kv_lora_rank'] * H * CONFIG['qk_nope_head_dim'] +
                                CONFIG['kv_lora_rank'] * H * d_v + H * d_v * D)
    attn_flops = B * H * (2 * S * S * d_qk + 2 * S * S * d_v)
    flops = proj_flops + attn_flops
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
