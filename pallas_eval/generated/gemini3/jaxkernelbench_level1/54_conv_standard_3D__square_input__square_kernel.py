```python
"""
JAXBench Level 1 - conv_standard_3D__square_input__square_kernel
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bias_add_kernel(x_ref, bias_ref, o_ref):
    o_ref[...] = x_ref[...] + bias_ref[...]

def pallas_bias_add(x, bias):
    # x: (N, C, M)
    # bias: (C,)
    N, C, M = x.shape
    x_flat = x.reshape(N * C, M)
    bias_flat = jnp.tile(bias.reshape(C), N).reshape(N * C, 1)
    
    block_nc = min(N * C, 128)
    while (N * C) % block_nc != 0 and block_nc > 1:
        block_nc //= 2
    if block_nc == 0: block_nc = 1
        
    block_m = min(M, 128)
    while M % block_m != 0 and block_m > 1:
        block_m //= 2
    if block_m == 0: block_m = 1
        
    grid = ((N * C) // block_nc, M // block_m)
    
    out_flat = pl.pallas_call(
        bias_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_nc, block_m), lambda i, j: (i, j)),
                pl.BlockSpec((block_nc, 1), lambda i, j: (i, 0)),
            ],
            out_specs=pl.BlockSpec((block_nc, block_m), lambda i, j: (i, j)),
        )
    )(x_flat, bias_flat)
    
    return out_flat.reshape(N, C, M)

def identity_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

def pallas_identity(x):
    # x: (N, C, M)
    N, C, M = x.shape
    x_flat = x.reshape(N * C, M)
    
    block_nc = min(N * C, 128)
    while (N *
