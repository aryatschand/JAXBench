"""Gated Linear Attention (GLA) — Yang et al.

Extends linear attention with data-dependent forget gates per head. The gate
controls how quickly each head forgets past context, enabling adaptive memory.
Parallel form materializes the gated causal mask for hardware efficiency.

Paper: "Gated Linear Attention Transformers with Hardware-Efficient Training"
       (Yang et al., ICML 2024)
Used in GLA-Transformer models, gaining traction as efficient attention alternative.

Config based on GLA-1.3B from the paper.
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'gla_1_3b_attention',
    'model': 'GLA-1.3B',
    'operator': 'gated_linear_attention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 16,
    'head_dim': 128,
    'gate_dim': 16,
    'd_model': 2048,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, gate_logits)."""
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, 5)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    # Per-head, per-position gate logits (low rank: gate_dim -> 1)
    # In practice, gate = sigmoid(x @ W_g), producing a scalar per head per position
    gate_logits = jax.random.normal(keys[3], (B, H, S), dtype=jnp.float32) * 2.0
    return query, key_t, value, gate_logits


def workload(query, key, value, gate_logits):
    """Gated linear attention with data-dependent forget gates.

    Recurrent: S_t = G_t * S_{t-1} + K_t^T V_t,  O_t = Q_t S_t
    Parallel:  O = (M ⊙ QK^T) V
    where M[i,j] = Π_{k=j+1}^{i} g_k  (cumulative product of gates)
    """
    B, H, S, D = query.shape

    # Gate: sigmoid gives values in (0, 1) as forget rates
    gate = jax.nn.sigmoid(gate_logits)  # (B, H, S)

    # Build gated causal mask via cumulative log-sum
    log_gate = jnp.log(gate + 1e-8)  # (B, H, S)
    log_gate_cumsum = jnp.cumsum(log_gate, axis=-1)  # (B, H, S)

    # M[i,j] = exp(Σ_{k=j+1}^{i} log(g_k)) = exp(cumsum[i] - cumsum[j])
    M = jnp.exp(log_gate_cumsum[:, :, :, None] - log_gate_cumsum[:, :, None, :])
    # Causal: only i >= j
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
    M = M * causal[None, None, :, :]

    # Gated linear attention scores: M ⊙ QK^T
    scores = jnp.einsum('bhsd,bhtd->bhst',
                        query.astype(jnp.float32),
                        key.astype(jnp.float32))
    scores = scores * M

    # Normalize (optional but stabilizes training)
    norm = jnp.sum(jnp.abs(scores), axis=-1, keepdims=True)
    norm = jnp.maximum(norm, 1.0)
    scores = scores / norm

    # Output
    output = jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)
    return output


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
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    # QK^T: 2*B*H*S*S*D, output AV: 2*B*H*S*S*D
    flops = 2 * B * H * S * S * D * 2
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
