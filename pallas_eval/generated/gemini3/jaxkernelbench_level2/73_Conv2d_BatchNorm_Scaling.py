```python
"""
JAXBench Level 2 - Conv2d_BatchNorm_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bn_scale_kernel(x_ref, w_ref, b_ref, out_ref):
    x_block = x_ref[:, :]
    w_block = w_ref[:, :]
    b_block = b_ref[:, :]
    out_ref[:, :] = x_block * w_block + b_block

class Model:
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.eps = 1e-5

        # Conv2d parameters
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        self.conv_weight = jax.random.normal(key1, (out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jax.random.normal(key2, (out_channels,))

        # BatchNorm2d parameters (learnable only)
        self.bn_weight = jnp.ones((out_channels,))
        self.bn_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, H, W) in PyTorch format
        # Convert to NHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # Conv2d (valid padding - no padding)
        weight = jnp.transpose(self.conv_weight, (2, 3, 1, 0))

        x = lax.conv_general_dilated(
            x, weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        # Note: We skip adding conv
