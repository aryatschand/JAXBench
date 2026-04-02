```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def post_conv_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Repeat bias to match the block size (128)
    bias_rep = pltpu.repeat(bias, 128, axis=0)
    x = x + bias_rep
    
    # Min along channel dimension
    x = jnp.min(x, axis=1, keepdims=True)
    
    # Double tanh
    x = jnp.tanh(x)
    x = jnp.tanh(x)
    
    o_ref[...] = x

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        # Initialize with PyTorch Conv2d weight shape (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (out,in,H,W) to (H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Fuse bias addition, min reduction, and double tanh into a Pallas kernel
        N, H, W, C = x.shape
        M = N * H * W
        
        x_flat = x.reshape((M, C))
        bias_2d = self.bias.reshape((1, C))
        
        block_m = 128
        pad_m = (block_m - (M % block_m)) % block_m
        
        if pad_m > 0:
            x_flat = jnp.pad(x_flat, ((0, pad_m), (0, 0)), constant_values=jnp.inf)
