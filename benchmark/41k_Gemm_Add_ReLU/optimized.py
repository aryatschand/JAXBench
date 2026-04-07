"""76_Gemm_Add_ReLU — JAXBench fused operator workload (Pallas)."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': '76_Gemm_Add_ReLU_pallas',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
}

def create_inputs(dtype=jnp.bfloat16):
    """Create all inputs including weights."""
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (4096, 8192), dtype=dtype)
    weight = jnp.zeros((8192, 8192), dtype=dtype)
    bias = jnp.zeros(8192, dtype=dtype)
    return x, weight, bias

def fused_kernel(x_ref, w_ref, bias_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    
    K = 8192
    block_k = 512
    @pl.when(pl.program_id(2) == K // block_k - 1)
    def epilogue():
        result = acc_ref[...] + bias_ref[...]  # bias add
        result = jax.nn.relu(result)            # activation
        o_ref[...] = result.astype(o_ref.dtype)

def workload(x, w, bias):
    M, K = x.shape
    N = w.shape[1]
    bm, bn, bk = 512, 1024, 512  # block sizes
    return pl.pallas_call(
        fused_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),     # x
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),     # w
                pl.BlockSpec((bn,), lambda i, j, k: (j,)),           # bias
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            grid=(M // bm, N // bn, K // bk),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, w, bias)

def benchmark(num_warmup=5, num_iters=100):
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times_ms = np.array(times) * 1000
    avg = float(np.mean(times_ms))
    return {
        'name': CONFIG['name'],
        'config': {k: v for k, v in CONFIG.items() if k != 'name'},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times_ms)), 4),
        'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
        'status': 'success',
    }

if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
