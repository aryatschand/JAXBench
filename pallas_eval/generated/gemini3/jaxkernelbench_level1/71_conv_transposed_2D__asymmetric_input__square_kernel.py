import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_bias_kernel(x_ref, bias_ref, o_ref):
    o_ref[0, 0, :] = x_ref[0, 0, :] + bias_ref[0, 0]

def pallas_add_bias(x, bias):
    N, C, H, W = x.shape
    x_flat = x.reshape(N, C, -1)
    seq_len = x_flat.shape[2]
    
    if seq_len <= 128:
        block_size = 128
    elif seq_len <= 256:
        block_size = 256
    elif seq_len <= 512:
        block_size = 512
    elif seq_len <= 1024:
        block_size = 1024
    elif seq_len <= 2048:
        block_size = 2048
    elif seq_len <= 4096:
        block_size = 4096
    else:
        block_size = 8192
        
    pad_size = (block_size - (seq_len % block_size)) % block_size
    
    if pad_size > 0:
        x_flat = jnp.pad(x_flat, ((0, 0), (0, 0), (0, pad_size)))
        
    padded_seq_len = x_flat.shape[2]
    num_blocks = padded_seq_len // block_size
    
    bias_2d = bias.reshape(C, 1)
    
    out = pl.pallas_call(
        add_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(N, C, num_blocks),
            in_specs=[
                pl.BlockSpec((1, 1, block_size), lambda n, c, b: (n, c, b)),
                pl.BlockSpec((1, 1), lambda n, c, b: (c, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, block_size), lambda n, c, b: (n, c, b)),
        )
    )(x_flat, bias_2d)
    
    if pad_size > 0:
        out = out[:, :, :seq_len
