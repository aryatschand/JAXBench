import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def act_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Add bias (automatically broadcasts (C,) to (B, C))
    x = x + bias
    
    # HardSwish: x * relu6(x+3)/6
    hs = x * jnp.minimum(jnp.maximum(x + 3.0, 0.0), 6.0) / 6.0
    
    # ReLU
    out = jnp.maximum(0.0, hs)
    
    o_ref[...] = out

def pallas_bias_act(x, bias):
    N, H, W, C = x.shape
    # Flatten spatial dimensions to use a 1D grid
    x_flat = x.reshape(-1, C)
    
    L = x_flat.shape[0]
    
    # Find a suitable block size that divides the flattened length
    B = 512
    while L % B != 0 and B > 1:
        B //= 2
        
    grid_shape = (L // B,)
    
    out_flat = pl.pallas_call(
        act_kernel,
        out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((B, C), lambda i: (i, 0)),
                pl.BlockSpec((C,), lambda i: (0,)),
            ],
            out_specs=pl.BlockSpec((B, C), lambda i: (i, 0)),
        ),
    )(x_flat, bias)
    
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
        
        # Transpose kernel for JAX conv
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding (no padding) to match PyTorch default
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Fused Bias Add, HardSwish, and ReLU using Pallas
        x = pallas_bias_act(x, self.bias)
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
