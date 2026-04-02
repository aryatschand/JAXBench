import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'alphafold_768_triangle_mult',
    'model': 'AlphaFold2',
    'operator': 'triangle_mult_outgoing',
    'N': 768,
    'C': 64,
    'direction': 'outgoing',
}

def create_inputs(dtype=jnp.bfloat16):
    """Returns (pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_norm_scale, output_proj_w, output_gate_w)."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 9)
    N, C = CONFIG['N'], CONFIG['C']
    pair_act = jax.random.normal(keys[0], (N, N, C), dtype=dtype)
    mask = jnp.ones((N, N, 1), dtype=dtype)
    left_proj = jax.random.normal(keys[1], (C, C), dtype=dtype) * 0.02
    right_proj = jax.random.normal(keys[2], (C, C), dtype=dtype) * 0.02
    left_gate = jax.random.normal(keys[3], (C, C), dtype=dtype) * 0.02
    right_gate = jax.random.normal(keys[4], (C, C), dtype=dtype) * 0.02
    center_scale = jax.random.normal(keys[5], (C,), dtype=dtype) * 0.1 + 1.0
    out_proj = jax.random.normal(keys[6], (C, C), dtype=dtype) * 0.02
    out_gate = jax.random.normal(keys[7], (C, C), dtype=dtype) * 0.02
    return pair_act, mask, left_proj, right_proj, left_gate, right_gate, center_scale, out_proj, out_gate

def einsum_kernel(left_ref, right_ref, pair_act_ref, center_scale_ref, out_proj_w_ref, out_gate_w_ref, out_ref):
    acc = jnp.zeros((64, 64, 64), dtype=jnp.float32)
    
    def body_fn(k, acc):
        left_blk = jax.lax.dynamic_slice(left_ref, (0, 0, k * 128), (64, 64, 128)).astype(jnp.float32)
        right_blk = jax.lax.dynamic_slice(right_ref, (0, k * 128, 0), (64, 128, 64)).astype(jnp.float32)
        
        res_trans = jnp.matmul(left_blk, right_blk)
        acc += res_trans
        return acc

    acc = jax.lax.fori_loop(0, 6, body_fn, acc)
    
    acc_ijc = jnp.transpose(acc, (1, 2, 0))
    
    eps = 1e-6
    mean_sq = jnp.mean(acc_ijc * acc_ijc, axis=-1, keepdims=True)
    rms = jnp.sqrt(mean_sq + eps)
    
    center_scale = center_scale_ref[:]
    normed = (acc_ijc / rms) * center_scale
    normed = normed.astype(out_ref.dtype)
    
    out_proj_w = out_proj_w_ref[:, :]
    normed_flat = normed.reshape((4096, 64))
    output_flat = jnp.dot(normed_flat, out_proj_w)
    output = output_flat.reshape((64, 64, 64))
    
    pair_act_blk = pair_act_ref[:, :, :]
    out_gate_w = out_gate_w_ref[:, :]
    pair_act_flat = pair_act_blk.reshape((4096, 64))
    gate_flat = jax.nn.sigmoid(jnp.dot(pair_act_flat, out_gate_w))
    gate = gate_flat.reshape((64, 64, 64))
    
    res = output * gate
    out_ref[:, :, :] = res.astype(out_ref.dtype)

def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    act = pair_act * mask
    
    left_w = jnp.concatenate([left_proj_w, left_gate_w], axis=-1)
    left_res = jnp.dot(act, left_w)
    left_proj_unact = left_res[..., :64]
    left_gate_unact = left_res[..., 64:]
    left_proj = left_proj_unact * jax.nn.sigmoid(left_gate_unact)
    
    right_w = jnp.concatenate([right_proj_w, right_gate_w], axis=-1)
    right_res = jnp.dot(act, right_w)
    right_proj_unact = right_res[..., :64]
    right_gate_unact = right_res[..., 64:]
    right_proj = right_proj_unact * jax.nn.sigmoid(right_gate_unact)
    
    left_trans = jnp.transpose(left_proj, (2, 0, 1))
    right_trans = jnp.transpose(right_proj, (2, 1, 0))
    
    grid_shape = (768 // 64, 768 // 64)
    
    return pl.pallas_call(
        einsum_kernel,
        out_shape=jax.ShapeDtypeStruct(pair_act.shape, pair_act.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((64, 64, 768), lambda i, j: (0, i, 0)),
                pl.BlockSpec((64, 768, 64), lambda i, j: (0, 0, j)),
                pl.BlockSpec((64, 64, 64), lambda i, j: (i, j, 0)),
                pl.BlockSpec((64,), lambda i, j: (0,)),
                pl.BlockSpec((64, 64), lambda i, j: (0, 0)),
                pl.BlockSpec((64, 64), lambda i, j: (0, 0)),
            ],
            out_specs=pl.BlockSpec((64, 64, 64), lambda i, j: (i, j, 0)),
        ),
    )(left_trans, right_trans, pair_act, center_scale, out_proj_w, out_gate_w)

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
    N, C = CONFIG['N'], CONFIG['C']
    proj_flops = 4 * N * N * C * C * 2
    einsum_flops = N * N * N * C * 2
    out_flops = 2 * N * N * C * C * 2
    flops = proj_flops + einsum_flops + out_flops
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
