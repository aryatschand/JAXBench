"""Dense bf16 GEMM — Llama-70B FFN dimensions.

Pallas TPU kernel implementation.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'gemm_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'dense_matmul',
    'M': 8192,
    'K': 8192,
    'N': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    return A, B


def gemm_kernel(a_ref, b_ref, c_ref):
    a = a_ref[...]  # (block_m, K)
    b = b_ref[...]  # (K, block_n)
    c = jnp.dot(a.astype(jnp.float32), b.astype(jnp.float32))
    c_ref[...] = c.astype(a.dtype)


def workload(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    bm = 128
    bn = 128

    grid_m = M // bm
    grid_n = N // bn

    return pl.pallas_call(
        gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(grid_m, grid_n),
            in_specs=[
                pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                pl.BlockSpec((K, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        ),
    )(A, B)


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
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    flops = 2 * M * K * N
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
