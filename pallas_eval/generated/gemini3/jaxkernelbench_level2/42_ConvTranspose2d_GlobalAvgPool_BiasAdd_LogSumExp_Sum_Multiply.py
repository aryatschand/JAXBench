```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def post_process_kernel(x_ref, bias_ref, o_ref):
    # x_ref: (block_n, 128)
    # bias_ref: (128,)
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Add bias
    x = x + bias[None, :]
    
    # Log-sum-exp over axis 1 (channels)
    max_x = jnp.max(x, axis=1, keepdims=True)
    exp_x = jnp.exp(x - max_x)
    sum_exp = jnp.sum(exp_x, axis=1, keepdims=True)
    lse = max_x + jnp.log(sum_exp)
    
    # Multiply by 10.0
    res = lse * 10.0 # (block_n, 1)
    
    # Repeat to match the output block shape (block_n, 128)
    # This satisfies the TPU requirement that block dims be multiples of (8, 128)
    o_ref[...] = pltpu.repeat(res, 128, axis=1)

def pallas_post_process(x_mean, bias):
    bias_sq = jnp.squeeze(bias) # (128,)
    N = x_mean.shape[0]
    
    # Use a block size of 16 for the batch dimension (matches batch_size and is a multiple of 8)
    block_n = min(N, 16)
    grid_n = N // block_n
    
    out_padded = pl.pallas_call(
        post_process_kernel,
        out_shape=jax.ShapeDtypeStruct((N, 128), x_mean.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(grid_n,),
            in_specs=[
                pl.BlockSpec((block_n, 128), lambda i: (i, 0)),
                pl.BlockSpec((128,), lambda i: (0,)),
            ],
            out_specs=pl.BlockSpec((block_n, 128), lambda i: (i, 0)),
        )
    )(x_mean, bias_sq)
    
    # Slice out the first column to get the final (N, 1) result
    return out_padded[:, 0:1]

class Model:
    def __init__(self, in_channels, out_channels, kernel_
