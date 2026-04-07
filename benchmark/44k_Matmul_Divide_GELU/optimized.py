"""86_Matmul_Divide_GELU — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': '86_Matmul_Divide_GELU_pallas',
    'batch_size': 4096,
    'input_size': 8192,
    'output_size': 8192,
    'divisor': 10.0,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (4096, 8192), dtype=dtype)
    weight = jnp.zeros((8192, 8192), dtype=dtype)
    bias = jnp.zeros(8192, dtype=dtype)
    return x, weight, bias

def fused_kernel(x_ref, w_ref, bias_ref, o_ref, acc_ref):
    bm, bn, bk = 1024, 1024, 512
    
    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros_like(acc_ref)
        
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    
    @pl.when(pl.program_id(2) == 8192 // 512 - 1)
    def epilogue():
        result = acc_ref[...] + bias_ref[...]
        result = result * 0.1
        result = jax.nn.gelu(result)
        o_ref[...] = result.astype(o_ref.dtype)

def workload(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    bm, bn, bk = 1024, 1024, 512
    
    # ensure bias is 2d
    bias_2d = bias.reshape(1, N)
    
    return pl.pallas_call(
        fused_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                pl.BlockSpec((1, bn), lambda i, j, k: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            grid=(M // bm, N // bn, K // bk),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, weight, bias_2d)
