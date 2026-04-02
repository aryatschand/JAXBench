```python
import jax
import jax.numpy as jnp
import jax.lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_block_size(dim, max_size=128):
    for i in [128, 64, 32, 16, 8, 4, 2, 1]:
        if dim % i == 0 and i <= max_size:
            return i
    return 1

def pallas_transpose_nchw_to_nhwc(x):
    N, C, H, W = x.shape
    
    pad_H = (128 - (H % 128)) % 128
    pad_W = (128 - (W % 128)) % 128
    pad_C = (64 - (C % 64)) % 64
    
    if pad_H > 0 or pad_W > 0 or pad_C > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_C), (0, pad_H), (0, pad_W)))
        
    padded_C = C + pad_C
    padded_H = H + pad_H
    padded_W = W + pad_W
    
    block_N = get_block_size(N, 1)
    block_C = get_block_size(padded_C, 64)
    block_H = get_block_size(padded_H, 128)
    block_W = get_block_size(padded_W, 128)
    
    grid = (N // block_N, padded_C // block_C, padded_H // block_H, padded_W // block_W)
    
    def kernel(x_ref, o_ref):
        o_ref[...] = jnp.transpose(x_ref[...], (0, 2, 3, 1))
        
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N, padded_H, padded_W, padded_C), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_N, block_C, block_H, block_W), lambda n, c, h, w: (n, c, h, w)),
            ],
            out_specs=pl.BlockSpec((block_N, block_H, block_W, block_C), lambda n, c, h, w: (n
