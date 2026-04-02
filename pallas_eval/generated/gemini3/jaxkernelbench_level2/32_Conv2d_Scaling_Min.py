```python
"""
JAXBench Level 2 - Conv2d_Scaling_Min
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def post_process_kernel(x_ref, bias_ref, scale_ref, out_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    scale = scale_ref[...]
    
    # Add bias
    x = x + bias
    
    # Scale
    x = x * scale
    
    # Min along channel dimension
    out = jnp.min(x, axis=1, keepdims=True)
    
    out_ref[...] = out

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel for JAX conv
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        N, H, W, C = x.shape
        M = N * H * W
        
        # Flatten spatial dimensions to map over them efficiently
        x_flat = x.reshape((M, C))
        bias_flat = self.bias.reshape((1, C))
        scale_flat = jnp.full((1, C), self.scale_factor, dtype=x.dtype)
        
        # Use a block size of 1024 for the flattened spatial dimension
        M_block = 1024
        pad_size = (M_block - (M % M_block)) % M_block
        
        if pad_size > 0:
            x_flat = jnp.pad(x_flat, ((0, pad_size),
