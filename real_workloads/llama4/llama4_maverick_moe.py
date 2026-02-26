"""Sparse MoE — Llama 4 Maverick 400B. Based on AI-Hypercomputer/maxtext.

Llama 4 Maverick is the larger MoE variant with 128 experts and top-1
routing. With emb_dim=5120 and per-expert intermediate=4096, total params
~400B but only ~17B active per token due to top-1 expert selection.

This tests a larger expert pool (128) with sparse routing, which creates
different optimization opportunities vs Mixtral's 8 experts.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'llama4_maverick_moe',
    'model': 'Llama-4-Maverick-400B',
    'operator': 'sparse_moe_top1',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 5120,
    'mlp_dim': 4096,
    'num_experts': 128,
    'num_experts_per_tok': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)
    C = CONFIG
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    N = C['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    eg = jax.random.normal(keys[2], (N, E, M), dtype=dtype) * 0.02
    eu = jax.random.normal(keys[3], (N, E, M), dtype=dtype) * 0.02
    ed = jax.random.normal(keys[4], (N, M, E), dtype=dtype) * 0.02
    return x, router, eg, eu, ed


def workload(x, router_weights, expert_gate, expert_up, expert_down):
    """Llama 4 Maverick MoE: top-1 routing over 128 experts."""
    B, S, E = x.shape
    C = CONFIG
    N, K = C['num_experts'], C['num_experts_per_tok']

    # Router: softmax-based top-1
    logits = jnp.dot(x, router_weights)
    top_k_logits, top_k_idx = jax.lax.top_k(logits, K)
    probs = jax.nn.softmax(top_k_logits, axis=-1)

    # Expert computation (einsum-based batched)
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, expert_gate))
    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, expert_down)

    # Weighted combination (top-1)
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    return jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)


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
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    N = C['num_experts']
    flops = B * S * N * E * M * 2 * 3
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
