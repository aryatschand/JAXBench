import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bias_mish_mish_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Add bias
    x = x + bias
    
    # Apply Mish twice
    x = x * jnp.tanh(jnn.softplus(x))
    x = x * jnp.tanh(jnn.softplus(x))
    
    o_ref[...] = x

def pallas_bias_mish_mish(x, bias):
    N, H, W, C = x.shape
    x_flat = x.reshape(-1, C)
    L = x_flat.shape[0]
    
    block_L = 256
    block_C = 128
    
    pad_L = (block_L - (L % block_L)) % block_L
    pad_C = (block_C - (C % block_C)) % block_C
    
    if pad_L > 0 or pad_C > 0:
        x_flat = jnp.pad(x_flat, ((0, pad_L), (0, pad_C)))
        
    grid_L = x_flat.shape[0] // block_L
    grid_C = x_flat.shape[1] // block_C
    
    bias_2d = bias.reshape(1, C)
    if pad_C > 0:
        bias_2d = jnp.pad(bias_2d, ((0, 0), (0, pad_C)))
        
    out_flat = pl.pallas_call(
        bias_mish_mish_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(grid_L, grid_C),
            in_specs=[
                pl.BlockSpec((block_L, block_C), lambda i, j: (i, j)),
                pl.BlockSpec((1, block_C), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((block_L, block_C), lambda i, j: (i, j)),
        )
    )(x_flat, bias_2d)
    
    if pad_L > 0 or pad_C > 0:
        out_flat = out_flat[:L, :C]
        
    return out_flat.reshape(N, H, W, C)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (out,in,H,W) to (H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding
        x = jax.lax.conv_general_dilated(
            x, kernel, 
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Fused bias and double Mish activation
        x = pallas_bias_mish_mish(x, self.bias)
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
