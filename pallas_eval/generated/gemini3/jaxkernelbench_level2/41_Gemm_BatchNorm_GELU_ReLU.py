```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_epilogue_kernel(eps):
    def epilogue_kernel(x_ref, mean_ref, var_ref, bn_w_ref, bn_b_ref, o_ref):
        x = x_ref[...]
        mean = mean_ref[...]
        var = var_ref[...]
        bn_w = bn_w_ref[...]
        bn_b = bn_b_ref[...]
        
        # BatchNorm
        inv_std = jax.lax.rsqrt(var + eps)
        x_norm = (x - mean) * inv_std * bn_w + bn_b
        
        # GELU
        x_gelu = jax.nn.gelu(x_norm)
        
        # ReLU
        x_relu = jax.nn.relu(x_gelu)
        
        o_ref[...] = x_relu
    return epilogue_kernel

@functools.partial(jax.jit, static_argnames=['eps'])
def fused_epilogue(x, mean, var, bn_w, bn_b, eps=1e-5):
    B, F = x.shape
    block_b = min(B, 512)
    block_f = min(F, 512)
    
    grid = (B // block_b, F // block_f)
    kernel = get_epilogue_kernel(eps)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_b, block_f), lambda i, j: (i, j)),
                pl.BlockSpec((1, block_f), lambda i, j: (0, j)),
                pl.BlockSpec((1, block_f), lambda i, j: (0, j)),
                pl.BlockSpec((1, block_f), lambda i, j: (0, j)),
                pl.BlockSpec((1, block_f), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((block_b, block_f), lambda i, j: (i, j)),
        )
    )(x, mean, var, bn_w, bn_b)

class Model:
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in
