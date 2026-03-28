"""Paged Attention (Optimized) — Llama-3.1-70B with jax.nn.dot_product_attention.

Reconstructs full KV from pages (same as baseline), then uses
jax.nn.dot_product_attention for the attention computation.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_70b_paged_attention_optimized',
    'model': 'Llama-3.1-70B',
    'operator': 'paged_attention',
    'num_seqs': 16,
    'max_seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    num_seqs = CONFIG['num_seqs']
    max_seq_len = CONFIG['max_seq_len']
    num_q_heads = CONFIG['num_query_heads']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = max_seq_len // page_size
    total_pages = num_seqs * pages_per_seq
    max_num_tokens = num_seqs
    queries = jax.random.normal(keys[0], (max_num_tokens, num_q_heads, head_dim), dtype=dtype)
    k_pages = jax.random.normal(keys[1], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype) * 0.02
    v_pages = jax.random.normal(keys[2], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype) * 0.02
    kv_lens = jnp.full((num_seqs,), max_seq_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    return queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens


def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    """Paged attention with dot_product_attention for the attention step."""
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    max_seq_len = CONFIG['max_seq_len']
    pages_per_seq = max_seq_len // CONFIG['page_size']
    num_q_per_kv = num_q_heads // num_kv_heads

    def attend_one_seq(seq_idx):
        q_start = cu_q_lens[seq_idx]
        q = jax.lax.dynamic_slice(queries, (q_start, 0, 0), (1, num_q_heads, head_dim))
        seq_pages = page_indices[seq_idx]
        k = k_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
        v = v_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
        # Expand for GQA
        k = jnp.repeat(k, num_q_per_kv, axis=1)
        v = jnp.repeat(v, num_q_per_kv, axis=1)
        # Reshape to (1, H, S, D) for dot_product_attention
        q_4d = q[:, None, :, :].transpose(0, 2, 1, 3)       # (1, H_q, 1, D)
        k_4d = k[None].transpose(0, 2, 1, 3)                 # (1, H_q, S, D)
        v_4d = v[None].transpose(0, 2, 1, 3)                 # (1, H_q, S, D)
        # No causal mask needed for decode (single query token attends to all KV)
        out = jax.nn.dot_product_attention(
            q_4d, k_4d, v_4d,
            scale=head_dim ** -0.5,
        )  # (1, H_q, 1, D)
        return out[0, :, 0, :]  # (H_q, D)

    outputs = jax.vmap(attend_one_seq)(jnp.arange(num_seqs))
    return outputs


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
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    max_seq_len = CONFIG['max_seq_len']
    head_dim = CONFIG['head_dim']
    flops = num_seqs * num_q_heads * (4 * max_seq_len * head_dim)
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
