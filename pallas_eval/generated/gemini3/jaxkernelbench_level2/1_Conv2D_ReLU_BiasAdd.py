```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def epilogue_kernel(x_ref, conv_bias_ref, bias_ref, o_ref):
    # Read the entire block
    x_val = x_ref[:, :]
    cb_val = conv_bias_ref[:, :]
    b_val = bias_ref[:, :]
    
    # Add conv_bias
    val = x_val + cb_val
    
    # ReLU
    val = jnp.maximum(val, 0.0)
    
    # Add bias (which broadcasts over the channel dimension in NHWC)
    val = val + b_val
    
    # Write the entire block
    o_ref[:, :] = val

def epilogue_pallas(x, conv_bias, bias):
    N, H, W, C = x.shape
    M = N * H * W
    
    # Flatten spatial dimensions to make it a 2D array (M, C)
    x_flat = x.reshape((M, C))
    cb_flat = conv_bias.reshape((1, C))
    b_flat = bias.reshape((1, C))
    
    # Pad M to a multiple of 512 for optimal TPU block sizing
    pad_len = (512 - (M % 512)) % 512
    if pad_len > 0:
        x_padded = jnp.pad(x_flat, ((0, pad_len), (0, 0)))
    else:
        x_padded = x_flat
        
    M_padded = M + pad_len
    grid = (M_padded // 512,)
    
    out_padded = pl.pallas_call(
        epilogue_kernel,
        out_shape=jax.ShapeDtypeStruct((M_padded, C), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((512, C), lambda i: (i, 0)),
                pl.BlockSpec((1, C), lambda i: (0, 0)),
                pl.BlockSpec((1, C), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((512, C), lambda i: (i, 0)),
        )
    )(x_padded, cb_flat, b_flat)
    
    if pad_len > 0:
        out_flat = out_padded[:-pad_len, :]
    else:
        out_flat = out_padded
