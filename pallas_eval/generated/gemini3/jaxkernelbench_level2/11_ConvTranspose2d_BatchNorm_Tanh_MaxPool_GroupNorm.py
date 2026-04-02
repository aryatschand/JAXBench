```python
"""
JAXBench Level 2 - Task 11: ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
Pallas TPU Kernel Implementation
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_groups = num_groups
        self.eps = 1e-5

        # Conv transpose weights - PyTorch shape: (in_channels, out_channels, kH, kW)
        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))

        # BatchNorm parameters (learnable only - training mode uses batch statistics)
        self.batch_norm_weight = jnp.ones((out_channels,))
        self.batch_norm_bias = jnp.zeros((out_channels,))

        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, H, W) in PyTorch format
        # Convert to NHWC for JAX
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # ConvTranspose2d
        # PyTorch weight: (in_channels, out_channels, kH, kW)
        # JAX conv_transpose expects: (kH, kW, out_channels, in_channels)
        weight = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))

        # Correct padding for JAX conv_transpose to match PyTorch
        k = self.kernel_
