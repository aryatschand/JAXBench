"""29_Matmul_Mish_Mish — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

CONFIG = {
    'name': '29_Matmul_Mish_Mish_pallas',
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


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def _kernel(x_ref, w_ref, bias_ref, o_ref, acc_ref, *, num_k_blocks):
    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == num_k_blocks - 1)
    def epilogue():
        result = acc_ref[...] + bias_ref[...]
        result = mish(result)
        result = mish(result)
        o_ref[...] = result.astype(o_ref.dtype)


def workload(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    
    bm, bn, bk = 512, 1024, 512
    
    return pl.pallas_call(
        functools.partial(_kernel, num_k_blocks=K // bk),
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                pl.BlockSpec((bn,), lambda i, j, k: (j,)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            grid=(M // bm, N // bn, K // bk),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        )
    )(x, weight, bias)
