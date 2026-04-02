```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_bias_kernel(x_ref, bias_ref, o_ref):
    x_val = x_ref[...]
    bias_val = bias_ref[...]
    # bias_val is (block_c, 1), extract the 1D vector
    bias_1d = bias_val[:, 0]
    # Broadcast bias across the block_n and block_m dimensions
    o_ref[...] = x_val + jnp.expand_dims(bias_1d, axis=(0, 2))

def add_bias_pallas(x, bias):
    N, C, D, H, W = x.shape
    x_flat = x.reshape(N, C, -1)
    M = x_flat.shape[2]
    
    # Pad M to a multiple of 128 for optimal TPU performance
    pad_m = (128 - (M % 128)) % 128
    if pad_m > 0:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, 0), (0, pad_m)))
        
    M_padded = x_flat.shape[2]
    
    # Reshape bias to 2D to satisfy Pallas TPU constraints
    bias_2d = bias.reshape(C, 1)
    
    # Determine block sizes
    block_n = N
    for i in range(min(N, 8), 0, -1):
        if N % i == 0:
            block_n = i
            break
            
    block_c = C
    for i in range(min(C, 128), 0, -1):
        if C % i == 0:
            block_c = i
            break
            
    block_m = 128
    
    grid = (N // block_n, C // block_c, M_padded // block_m)
    
    out_flat = pl.pallas_call(
        add_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_n, block_c, block_m), lambda i, j, k: (i, j, k)),
                pl.BlockSpec((block_c, 1), lambda i, j, k: (j, 0)),
            ],
            out_specs
