"""
JAXBench Level 1 - Task 50: conv_standard_2D__square_input__square_kernel
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.137963
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    def __init__(self, num_classes=1000):
        # Initialize conv1 weights with same shape as PyTorch
        self.conv1_weight = None
        self.conv1_bias = None
        self.weight_shape = (96, 3, 11, 11) # out_channels, in_channels, kH, kW
        self.bias_shape = (96,)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Convert conv weights from PyTorch (O,I,H,W) to JAX (H,W,I,O)
        weight = jnp.transpose(self.conv1_weight, (2, 3, 1, 0))
        
        x = lax.conv_general_dilated(
            x,
            weight,
            window_strides=(4, 4),
            padding=((2, 2), (2, 2)),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=1
        )

        if self.conv1_bias is not None:
            x = x + self.conv1_bias.reshape(1, 1, 1, -1)

        # Convert back NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x

batch_size = 256
num_classes = 1000

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]