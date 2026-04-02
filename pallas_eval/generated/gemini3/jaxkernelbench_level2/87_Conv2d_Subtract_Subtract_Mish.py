```python
"""
JAXBench Level 2 - Conv2d_Subtract_Subtract_Mish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mish_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Fused bias addition and Mish activation
    out = x + bias
    mish = out * jnp.tanh(jnp.log1p(jnp.exp(out)))
    
    o_ref[...] = mish

def fused_mish(x, bias):
    orig_shape = x.shape
    orig_c = orig_shape[-1]
    
    # Flatten and reshape to (-1, 128) to strictly satisfy TPU (8, 128) block constraints
    x_flat = x.flatten()
    total_elements = x_flat.size
    
    pad_elements = (128 - (total_elements % 128)) % 128
    if pad_elements > 0:
        x_flat = jnp.pad(x_flat, (0, pad_elements))
        
    x_reshaped = x_flat.reshape(-1, 128)
    M, C = x_reshaped.shape  # C is exactly 128
    
    # Tile bias to match the 128-element chunking
    repeats = (128 + orig_c - 1) // orig_c
    bias_tiled = jnp.tile(bias, repeats)[:128]
    bias_tiled = bias_tiled.reshape(1, 128)
    
    block_m = 512
    block_c = 128
    
    pad_m = (block_m - (M % block_m)) % block_m
    if pad_m > 0:
        x_reshaped = jnp.pad(x_reshaped, ((0, pad_m), (0, 0)))
        
    padded_M = x_reshaped.shape[0]
    grid = (padded_M // block_m, 1)
    
    out = pl.pallas_call(
        mish_kernel,
        out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, x_reshaped.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_m, block_c), lambda i
