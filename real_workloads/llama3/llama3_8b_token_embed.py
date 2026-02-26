"""Token Embedding Lookup — Llama 3.1 8B. Based on google/maxtext.

Token embedding is the first operation in every LLM forward pass.
Converts integer token IDs to dense vectors via embedding table lookup.
This is a gather/scatter operation that is purely memory-bandwidth bound.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'llama3_8b_token_embed',
    'model': 'Llama-3.1-8B',
    'operator': 'token_embed',
    'batch': 1,
    'seq_len': 2048,
    'vocab_size': 128256,
    'emb_dim': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (token_ids, embed_table)."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    C = CONFIG
    B, S, V, E = C['batch'], C['seq_len'], C['vocab_size'], C['emb_dim']
    token_ids = jax.random.randint(k1, (B, S), 0, V)
    embed_table = jax.random.normal(k2, (V, E), dtype=dtype) * 0.02
    return token_ids, embed_table


def workload(token_ids, embed_table):
    """Token embedding: look up token IDs in embedding table and scale."""
    emb_dim = CONFIG['emb_dim']
    embeddings = embed_table[token_ids]
    # Scale by sqrt(emb_dim) as in many LLMs
    return embeddings * (emb_dim ** 0.5)


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
    C = CONFIG
    B, S, E = C['batch'], C['seq_len'], C['emb_dim']
    # Memory-bound: read S token embeddings from table, write output
    total_bytes = B * S * E * 2 + C['vocab_size'] * E * 2  # bfloat16
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(total_bytes / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
