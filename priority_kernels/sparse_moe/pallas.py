"""Sparse MoE (Pallas) — Mixtral-8x7B using megablox GMM for expert matmuls.

Uses jax.experimental.pallas.ops.tpu.megablox.gmm for the grouped expert
matrix multiplications, keeping the same routing logic as baseline.
"""
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

CONFIG = {
    'name': 'mixtral_8x7b_moe_pallas',
    'model': 'Mixtral-8x7B',
    'operator': 'sparse_moe',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 4096,
    'mlp_dim': 14336,
    'num_experts': 8,
    'num_experts_per_tok': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_weights, expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    mlp = CONFIG['mlp_dim']
    E = CONFIG['num_experts']
    x = jax.random.normal(keys[0], (B, S, D), dtype=dtype)
    router = jax.random.normal(keys[1], (D, E), dtype=dtype) * 0.02
    gate = jax.random.normal(keys[2], (E, D, mlp), dtype=dtype) * 0.02
    up = jax.random.normal(keys[3], (E, D, mlp), dtype=dtype) * 0.02
    down = jax.random.normal(keys[4], (E, mlp, D), dtype=dtype) * 0.02
    return x, router, gate, up, down


def workload(x, router_weights, expert_gate, expert_up, expert_down):
    """MoE with Pallas megablox GMM for expert computation."""
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    E = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    mlp = CONFIG['mlp_dim']

    x_flat = x.reshape(-1, D)  # (T, D)
    T = x_flat.shape[0]

    # Router
    logits = jnp.dot(x_flat, router_weights)  # (T, E)
    top_k_vals, top_k_indices = jax.lax.top_k(logits, top_k)  # (T, top_k)
    weights = jax.nn.softmax(top_k_vals, axis=-1)  # (T, top_k)

    # Sort tokens by expert for grouped matmul
    expert_assignments = top_k_indices.reshape(-1)  # (T*top_k,)
    token_indices = jnp.repeat(jnp.arange(T), top_k)  # (T*top_k,)
    sort_order = jnp.argsort(expert_assignments)
    sorted_experts = expert_assignments[sort_order]
    sorted_tokens = token_indices[sort_order]

    # Build group_sizes
    group_sizes = jnp.zeros(E, dtype=jnp.int32)
    for e in range(E):
        group_sizes = group_sizes.at[e].set(jnp.sum(sorted_experts == e))

    # Gather sorted token embeddings
    sorted_x = x_flat[sorted_tokens]  # (T*top_k, D)

    # GMM for gate, up, down projections
    gate_out = gmm(sorted_x, expert_gate, group_sizes)  # (T*top_k, mlp)
    up_out = gmm(sorted_x, expert_up, group_sizes)      # (T*top_k, mlp)
    hidden = jax.nn.silu(gate_out) * up_out
    expert_out = gmm(hidden, expert_down, group_sizes)   # (T*top_k, D)

    # Scatter back and weight
    weight_flat = weights.reshape(-1)[sort_order]  # (T*top_k,)
    weighted_out = expert_out * weight_flat[:, None]

    # Accumulate per-token
    output = jnp.zeros((T, D), dtype=x.dtype)
    output = output.at[sorted_tokens].add(weighted_out)

    return output.reshape(B, S, D)


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
    mlp = CONFIG['mlp_dim']
    E = CONFIG['num_experts']
    top_k = CONFIG['num_experts_per_tok']
    T = B * S
    router_flops = 2 * T * D * E
    expert_flops = T * top_k * (2 * D * mlp + 2 * D * mlp + 2 * mlp * D)
    flops = router_flops + expert_flops
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
