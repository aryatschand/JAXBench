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


def kernel_fn(left_ref, right_ref, pair_ref, out_proj_ref, out_gate_ref, scale_ref, o_ref):
    i = pl.program_id(axis=0)

    left = left_ref[i, :, :]      # (N, C)
    right = right_ref[:, :, :]    # (N, C)
    pair = pair_ref[i, :, :]      # (N, C)

    N = left.shape[0]
    C = left.shape[1]

    acc = jnp.zeros((N, C), dtype=jnp.float32)

    def body(k, val):
        l = left[k, :]            # (C,)
        r = right[:, k, :]        # (N, C)
        return val + (l * r)

    acc = jax.lax.fori_loop(0, N, body, acc)

    eps = 1e-6
    rms = jnp.sqrt(jnp.mean(acc * acc, axis=-1, keepdims=True) + eps)
    normed = acc / rms * scale_ref[...]

    out = jnp.dot(normed, out_proj_ref[...])
    gate = jax.nn.sigmoid(jnp.dot(pair, out_gate_ref[...]))
    o_ref[i, :, :] = out * gate


def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    act = pair_act * mask
    left_proj = jnp.dot(act, left_proj_w)
    right_proj = jnp.dot(act, right_proj_w)
    left_gate = jax.nn.sigmoid(jnp.dot(act, left_gate_w))
    right_gate = jax.nn.sigmoid(jnp.dot(act, right_gate_w))
    left_proj = left_proj * left_gate
    right_proj = right_proj * right_gate

    N, _, C = left_proj.shape

    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((N, N, C), pair_act.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(N,),
            in_specs=[
                pl.BlockSpec((1, N, C), lambda i: (i, 0, 0)),
                pl.BlockSpec((N, N, C), lambda i: (0, 0, 0)),
                pl.BlockSpec((1, N, C), lambda i: (i, 0, 0)),
                pl.BlockSpec((C, C), lambda i: (0, 0)),
                pl.BlockSpec((C, C), lambda i: (0, 0)),
                pl.BlockSpec((C,), lambda i: (0,)),
            ],
            out_specs=pl.BlockSpec((1, N, C), lambda i: (i, 0, 0)),
        ),
    )(left_proj, right_proj, pair_act, out_proj_w, out_gate_w, center_scale)


def benchmark(num_warmup=5, num_iters=100):
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
