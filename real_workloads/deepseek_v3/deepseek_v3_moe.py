"""MoE with Shared Experts — DeepSeek V3 671B. Extracted from MaxText.

NOTE: ~18GB weights with 256 experts. May OOM on v6e-1 (single chip).
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'deepseek_v3_moe',
    'model': 'DeepSeek-V3-671B',
    'operator': 'moe_shared_experts',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 7168,
    'mlp_dim': 2048,
    'num_experts': 256,
    'num_experts_per_tok': 8,
    'shared_experts': 1,
    'shared_mlp_dim': 18432,
    'routed_scaling_factor': 2.5,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, router_w, router_bias, shared_gate, shared_up, shared_down,
    expert_gate, expert_up, expert_down)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    C = CONFIG
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    SM, N = C['shared_mlp_dim'], C['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=dtype)
    router_w = jax.random.normal(keys[1], (E, N), dtype=dtype) * 0.02
    router_bias = jnp.zeros((N,), dtype=dtype)
    shared_gate = jax.random.normal(keys[2], (E, SM), dtype=dtype) * 0.02
    shared_up = jax.random.normal(keys[3], (E, SM), dtype=dtype) * 0.02
    shared_down = jax.random.normal(keys[4], (SM, E), dtype=dtype) * 0.02
    expert_gate = jax.random.normal(keys[5], (N, E, M), dtype=dtype) * 0.02
    expert_up = jax.random.normal(keys[6], (N, E, M), dtype=dtype) * 0.02
    expert_down = jax.random.normal(keys[7], (N, M, E), dtype=dtype) * 0.02
    return (x, router_w, router_bias, shared_gate, shared_up, shared_down,
            expert_gate, expert_up, expert_down)


def workload(x, router_weights, router_bias, shared_gate, shared_up, shared_down,
             expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    """DeepSeek V3 MoE: shared expert + sigmoid-routed sparse experts."""
    C = CONFIG
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = C['num_experts_per_tok']
    sf = C['routed_scaling_factor']
    # Shared expert
    s_out = jnp.dot(jax.nn.silu(jnp.dot(x, shared_gate)) * jnp.dot(x, shared_up), shared_down)
    # Sigmoid routing
    scores = jax.nn.sigmoid(jnp.dot(x, router_weights)) + router_bias
    top_k_scores, top_k_idx = jax.lax.top_k(scores, K)
    probs = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-6)
    # All experts in parallel
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, expert_gate_kernels))
    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up_kernels)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, expert_down_kernels)
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    r_out = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights) * sf
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
    SM, K, N = C['shared_mlp_dim'], C['num_experts_per_tok'], C['num_experts']
    shared_flops = B * S * E * SM * 2 * 3
    routed_flops = B * S * K * (E * M * 2 * 3)
    routing_flops = B * S * E * N * 2
    flops = shared_flops + routed_flops + routing_flops
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
