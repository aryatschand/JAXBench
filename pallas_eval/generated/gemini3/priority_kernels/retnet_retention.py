"""Multi-Scale Retention — Microsoft RetNet.

Replaces softmax attention with retention: a causal linear attention mechanism
with fixed exponential decay per head. Different heads use different decay rates
(multi-scale), giving each head a different "memory horizon".

Paper: "Retentive Network: A Successor to Transformer" (Sun et al., 2023)
Used in RetNet models and influenced Mamba-2, GLA, and other recent architectures.

Config based on RetNet-6.7B from the paper.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from functools import partial

CONFIG = {
    'name': 'retnet_6_7b_retention',
    'model': 'RetNet-6.7B',
    'operator': 'multi_scale_retention',
    'batch': 1,
    'seq_len': 2048,
    'num_heads': 16,
    'head_dim': 256,
    'd_model': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(keys[0], (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(keys[1], (B, H, S, D), dtype=dtype)
    value = jax.random.normal(keys[2], (B, H, S, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    B, H, S, D = query.shape
    
    # Tile over the sequence dimension for queries.
    # S=2048 fits entirely in VMEM for K and V, so we load the full sequence
    # for K and V and compute the attention for a block of Q at a time.
    B_S = min(256, S)
    while S % B_S != 0:
        B_S //= 2

    def retention_kernel(q_ref, k_ref, v_ref, o_ref):
        b = pl.program_id(0)
        h = pl.program_id(1)
        q_idx = pl.program_id(2)
        
        # Compute decay rate for this head
        gamma = 1.0 - jnp.exp2(-5.0 - h.astype(jnp.float32))
        log_g = jnp.log(gamma)
        
        # Load blocks from HBM to VMEM
        q = q_ref[0, 0, :, :]  # (B_S, D)
        k = k_ref[0, 0, :, :]  # (S, D)
        v = v_ref[0, 0, :, :]  # (S, D)
        
        # QK^T
        qk = jnp.dot(
            q.astype(jnp.float32), 
            k.astype(jnp.float32).T, 
            preferred_element_type=jnp.float32
        )  # (B_S, S)
        
        # Compute distances
        q_pos = q_idx * B_S + jnp.arange(B_S, dtype=jnp.float32)
        k_pos = jnp.arange(S, dtype=jnp.float32)
        dist = q_pos[:, None] - k_pos[None, :]
        
        # Apply causal mask and decay
        mask = dist >= 0
        # Mask negative distances before exp to prevent any potential overflow
        safe_dist = jnp.maximum(dist, 0.0)
        decay = jnp.exp(log_g * safe_dist)
        
        qk = jnp.where(mask, qk * decay, 0.0)
        
        # Normalize by retention sum (per-query normalization)
        ret_sum = jnp.sum(jnp.abs(qk), axis=-1, keepdims=True)
        ret_sum = jnp.maximum(ret_sum, 1.0)
        qk = qk / ret_sum
        
        # Output projection
        o = jnp.dot(
            qk.astype(v.dtype), 
            v, 
            preferred_element_type=jnp.float32
        )  # (B_S, D)
        
        # Write back to HBM
        o_ref[0, 0, :, :] = o.astype(o_ref.dtype)

    grid_shape = (B, H, S // B_S)
    
    return pl.pallas_call(
        retention_kernel,
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((1, 1, B_S, D), lambda b, h, q_idx: (b, h, q_idx, 0)),
                pl.BlockSpec((1, 1, S, D), lambda b, h, q_idx: (b, h, 0, 0)),
                pl.BlockSpec((1, 1, S, D), lambda b, h, q_idx: (b, h, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, B_S, D), lambda b, h, q_idx: (b, h, q_idx, 0)),
        ),
    )(query, key, value)


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
    # QK^T: 2*B*H*S*S*D, AV: 2*B*H*S*S*D
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
