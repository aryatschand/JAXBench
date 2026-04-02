```python
"""
JAXBench Level 2 - Conv2d_GroupNorm_Scale_MaxPool_Clamp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
Optimized with JAX/Pallas TPU kernel.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        # Conv2d weights
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        
        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))
        self.num_groups = num_groups
        self.out_channels = out_channels
        
        # Scale parameter
        self.scale = jnp.ones(scale_shape)
        
        # MaxPool params
        self.maxpool_kernel_size = maxpool_kernel_size
        
        # Clamp params
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # 1. Conv2d (kept in JAX as it is highly optimized)
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # OIHW -> HWIO
        x = jax.lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        N, C, H, W = x.shape

        # 2. Precompute effective scale and bias for GroupNorm + Scale
        x_grouped = x.reshape(N, self.num_groups, -1)
        mean = jnp.mean(x_grouped, axis=2,
