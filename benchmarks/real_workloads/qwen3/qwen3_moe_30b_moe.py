"""Sparse MoE — Qwen 3 MoE-30B (A3B). Based on AI-Hypercomputer/maxtext.

Qwen 3 MoE uses 128 routed experts with top-8 routing and a shared expert.
The architecture is similar to DeepSeek V3 but with different dimensions:
emb_dim=2048, per-expert intermediate=1024, shared expert intermediate=4096.

Total params ~30B but only ~3B active per token (A3B = Active 3B).
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'qwen3_moe_30b_moe',
    'model': 'Qwen3-MoE-30B-A3B',
    'operator': 'sparse_moe_shared',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 2048,
    'mlp_dim': 1024,
    'num_experts': 128,
    'num_experts_per_tok': 8,
    'shared_mlp_dim': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router, shared_gate, shared_up, shared_down, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    C = CONFIG
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    SM, N = C['shared_mlp_dim'], C['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    sg = jax.random.normal(keys[2], (E, SM), dtype=dtype) * 0.02
    su = jax.random.normal(keys[3], (E, SM), dtype=dtype) * 0.02
    sd = jax.random.normal(keys[4], (SM, E), dtype=dtype) * 0.02
    eg = jax.random.normal(keys[5], (N, E, M), dtype=dtype) * 0.02
    eu = jax.random.normal(keys[6], (N, E, M), dtype=dtype) * 0.02
    ed = jax.random.normal(keys[7], (N, M, E), dtype=dtype) * 0.02
    return x, router, sg, su, sd, eg, eu, ed


def workload(x, router_weights, shared_gate, shared_up, shared_down, expert_gate, expert_up, expert_down):
    """Qwen3 MoE: shared expert + top-k routed experts with sigmoid routing."""
    B, S, E = x.shape
    C = CONFIG
    N, K = C['num_experts'], C['num_experts_per_tok']

    # Shared expert (always active)
    s_out = jnp.dot(jax.nn.silu(jnp.dot(x, shared_gate)) * jnp.dot(x, shared_up), shared_down)

    # Router: sigmoid-based scoring (like DeepSeek V3)
    scores = jax.nn.sigmoid(jnp.dot(x, router_weights))
    top_k_scores, top_k_idx = jax.lax.top_k(scores, K)
    probs = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-6)

    # Routed experts (einsum-based batched computation)
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, expert_gate))
    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, expert_down)

    # Weighted combination of selected experts
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    r_out = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)

    return s_out + r_out


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
    SM, N, K = C['shared_mlp_dim'], C['num_experts'], C['num_experts_per_tok']
    # Shared expert: 3 matmuls
    shared_flops = B * S * E * SM * 2 * 3
    # All experts computed (einsum): 3 matmuls * N experts
    expert_flops = B * S * N * E * M * 2 * 3
    flops = shared_flops + expert_flops
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
