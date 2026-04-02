```python
"""
JAXBench Level 2 - Conv2d_GELU_GlobalAvgPool
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
        
        # Convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        N, H_prime, W_prime, O = x.shape
        
        # Block sizes for the Pallas kernel
        block_H = 64
        block_W = 64
        block_O = 64
        
        # Pad dimensions to be multiples of block sizes
        pad_H = (H_prime + block_H - 1) // block_H * block_H
        pad_W = (W_prime + block_W - 1) // block_W * block_W
        pad_O = (O + block_O - 1) // block_O * block_O
        
        x_padded = jnp.pad(x, ((0, 0), (0, pad_H - H_prime), (0, pad_W - W_prime), (0, pad_O - O)))
        bias_padded = jnp.pad(self.bias, ((0, pad_O - O),))
        
        grid = (N, pad_H // block_H, pad_W // block_W, pad_O // block_O)
        
        def bias_gelu_pool_kernel(x_ref, bias_ref, o_ref):
            i_start = pl.program_id(1) * block_H
            j_start
