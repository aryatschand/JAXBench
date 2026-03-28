"""GQA Attention (Optimized) — Llama-3.1-405B with jax.nn.dot_product_attention.

Uses JAX's built-in dot_product_attention which handles GQA natively via
broadcasting when num_query_heads is a multiple of num_kv_heads.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_405b_gqa_optimized',
    'model': 'Llama-3.1-405B',
    'operator': 'gqa_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_query_heads': 128,
    'num_kv_heads': 8,
    'head_dim': 128,
    'emb_dim': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, Wq, Wk, Wv, Wo) — same as baseline."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    D = CONFIG['emb_dim']
    H_q, H_kv, D_h = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    x = jax.random.normal(keys[0], (B, S, D), dtype=dtype)
    Wq = jax.random.normal(keys[1], (D, H_q * D_h), dtype=dtype) * 0.02
    Wk = jax.random.normal(keys[2], (D, H_kv * D_h), dtype=dtype) * 0.02
    Wv = jax.random.normal(keys[3], (D, H_kv * D_h), dtype=dtype) * 0.02
    Wo = jax.random.normal(keys[4], (H_q * D_h, D), dtype=dtype) * 0.02
    return x, Wq, Wk, Wv, Wo


def workload(x, Wq, Wk, Wv, Wo):
    """GQA with dot_product_attention for the attention sub-op."""
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H_q, H_kv, D_h = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    num_q_per_kv = H_q // H_kv

    # dot_product_attention expects (B, S, H, D) layout
    q = jnp.dot(x, Wq).reshape(B, S, H_q, D_h)        # (B, S, H_q, D)
    k = jnp.dot(x, Wk).reshape(B, S, H_kv, D_h)       # (B, S, H_kv, D)
    v = jnp.dot(x, Wv).reshape(B, S, H_kv, D_h)       # (B, S, H_kv, D)

    # Expand KV heads to match query heads
    k = jnp.repeat(k, num_q_per_kv, axis=2)  # (B, S, H_q, D)
    v = jnp.repeat(v, num_q_per_kv, axis=2)

    # Use optimized attention
    attn_out = jax.nn.dot_product_attention(
        q, k, v,
        is_causal=True,
    )  # (B, S, H_q, D)

    attn_out = attn_out.reshape(B, S, H_q * D_h)
    return jnp.dot(attn_out, Wo)


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
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    H_q, H_kv, D_h = CONFIG['num_query_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    proj_flops = 2 * B * S * D * (H_q + 2 * H_kv) * D_h + 2 * B * S * H_q * D_h * D
    attn_flops = 4 * B * H_q * S * S * D_h
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
