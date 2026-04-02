```python
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bias_add_kernel(x_ref, bias_ref, o_ref):
    # bias_ref has shape (1, 1)
    o_ref[...] = x_ref[...] + jnp.reshape(bias_ref[...], (1, 1, 1, 1))

def copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

def pad_to_multiple(x, multiple=128):
    N, C, H, W = x.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    return jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w))), pad_h, pad_w

def pallas_bias_add(x, bias):
    # Reshape bias to 2D to satisfy TPU constraints
    C = bias.shape[0]
    bias_2d = bias.reshape(C, 1)
    
    x_padded, pad_h, pad_w = pad_to_multiple(x, 128)
    N, C_pad, H_pad, W_pad = x_padded.shape
    bh, bw = 128, 128
    
    block_x = (1, 1, bh, bw)
    block_bias = (1, 1)
    grid = (N, C_pad, H_pad // bh, W_pad // bw)
    
    out_padded = pl.pallas_call(
        bias_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x_padded.shape, x_padded.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block_x, lambda i, j, k, l: (i, j, k, l)),
                pl.BlockSpec(block_bias, lambda i, j, k, l: (j, 0))
            ],
            out_specs=pl.BlockSpec(block_x, lambda i, j, k, l: (i, j, k, l)),
        )
    )(x_padded, bias_2d)
    
    if pad_h > 0 or pad_w > 0:
