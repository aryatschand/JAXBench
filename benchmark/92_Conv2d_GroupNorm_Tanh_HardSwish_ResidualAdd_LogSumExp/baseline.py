"""
JAXBench Level 2 - Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import numpy as np

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        self.groups = groups
        self.eps = eps
        # Conv2d weights
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        
        # GroupNorm weights
        self.group_norm_weight = jnp.ones(out_channels)
        self.group_norm_bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d with VALID padding (no padding) to match PyTorch default
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.conv_weight, (2, 3, 1, 0))  # OIHW -> HWIO
        x_conv = jax.lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',  # Changed from 'SAME' to 'VALID' to match PyTorch default
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        x_conv = x_conv + self.conv_bias.reshape(1, 1, 1, -1)
        x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))  # NHWC -> NCHW

        # Group Normalization
        N, C, H, W = x_conv.shape
        x = x_conv.reshape(N, self.groups, C // self.groups, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        x_norm = x * self.group_norm_weight.reshape(1, -1, 1, 1) + self.group_norm_bias.reshape(1, -1, 1, 1)

        # Tanh
        x_tanh = jnp.tanh(x_norm)

        # HardSwish
        x_hard_swish = x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3, 0), 6) / 6

        # Residual Addition
        x_res = x_conv + x_hard_swish

        # LogSumExp
        x_logsumexp = jax.scipy.special.logsumexp(x_res, axis=1, keepdims=True)
        
        return x_logsumexp

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]