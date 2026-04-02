```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax

def transpose_fwd_kernel(x_ref, o_ref):
    o_ref[...] = jnp.transpose(x_ref[...], (0, 2, 3, 4, 1))

def pallas_transpose_fwd(x):
    N, C, D, H, W = x.shape
    grid = (N, D, H)
    return pl.pallas_call(
        transpose_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((N, D, H, W, C), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, C, 1, 1, W), lambda n, d, h: (n, 0, d, h, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, 1, W, C), lambda n, d, h: (n, d, h, 0, 0))
        )
    )(x)

def bias_transpose_kernel(x_ref, bias_ref, o_ref):
    o_ref[...] = jnp.transpose(x_ref[...] + bias_ref[...], (0, 4, 1, 2, 3))

def pallas_bias_transpose(x, bias):
    N, D, H, W, C = x.shape
    grid = (N, D, H)
    bias_2d = bias.reshape((1, C))
    return pl.pallas_call(
        bias_transpose_kernel,
        out_shape=jax.ShapeDtypeStruct((N, C, D, H, W), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, 1, W, C), lambda n, d, h: (n, d, h, 0, 0)),
                pl.BlockSpec((1, C), lambda n, d, h: (0, 0)),
            ],
            out_specs=pl.BlockSpec((1, C, 1, 1, W), lambda n, d, h: (n, 0, d, h, 0))
        )
    )(x, bias_2d)

def transpose_bwd_kernel(x_ref, o_ref):
    o_ref[...] = jnp.transpose(x_ref[...], (0, 4, 1, 2, 3))

def pallas_transpose_bwd(x):
    N, D, H, W, C = x.shape
    grid = (N, D, H)
    return pl.pallas_call(
        transpose_bwd_kernel,
        out_shape=jax.ShapeDtypeStruct((N, C, D, H, W), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, 1, W, C), lambda n, d, h: (n, d, h, 0, 0)),
            ],
            out_specs
