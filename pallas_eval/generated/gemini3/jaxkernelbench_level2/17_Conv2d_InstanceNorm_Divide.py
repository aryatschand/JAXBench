```python
"""
JAXBench Level 2 - Conv2d_InstanceNorm_Divide
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.instance_norm_weight = jnp.ones((out_channels,))
        self.instance_norm_bias = jnp.zeros((out_channels,))
        self.divide_by = divide_by

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d with VALID padding (no padding, like PyTorch default)
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # OIHW -> HWIO
        x_conv = jax.lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',  # PyTorch Conv2d default is no padding
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Transpose to NCHW before Pallas kernel for contiguous memory access
        x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))

        N, C, H_out, W_out = x_conv.shape
        seq_len = H_out * W_out
        
        # Find the next power of 2 (at least 128) for optimal TPU block size
        padded_seq_len = 128
        while padded_seq_len < seq_len:
            padded_seq_len *= 2
            
        x_flat = x_conv.reshape(N, C, seq_len)
        pad_size = padded_seq_len - seq_len
        if pad_size > 0:
            x_padded = jnp.pad(x_flat, ((0, 0), (0, 0), (0, pad_size)))
        else:
            x_padded = x_flat
            
        div_arr = jnp.array([
