import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

CONFIG = {
    'name': '53_Gemm_Scaling_Hardtanh_GELU_pallas',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
    'scaling_factor': 0.5,
    'hardtanh_min': -2,
    'hardtanh_max': 2,
}

def create_inputs(dtype=jnp.bfloat16):
    """Create all inputs including weights."""
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (4096, 8192), dtype=dtype)
    weight = jnp.zeros((8192, 8192), dtype=dtype)
    bias = jnp.zeros(8192, dtype=dtype)
    return x, weight, bias

def fused_kernel(x_ref, w_ref, bias_ref, o_ref, acc_ref, *, K, block_k):
    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    
    @pl.when(pl.program_id(2) == K // block_k - 1)
    def epilogue():
        acc = acc_ref[...]
        b = bias_ref[...]
        # broadcast b from (bn,) to (bm, bn)
        b = jnp.broadcast_to(b[None, :], acc.shape).astype(jnp.float32)
        result = acc + b
        result = result * 0.5
        result = jnp.clip(result, -2.0, 2.0)
        # GELU calculation
        sqrt_2_pi = 0.7978845608028654
        result = result * 0.5 * (1.0 + jnp.tanh(sqrt_2_pi * (result + 0.044715 * result**3)))
        o_ref[...] = result.astype(o_ref.dtype)

def workload(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    bm, bn, bk = 1024, 2048, 512
    
    return pl.pallas_call(
        functools.partial(fused_kernel, K=K, block_k=bk),
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
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, weight, bias)
