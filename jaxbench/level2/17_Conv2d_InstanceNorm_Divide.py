"""
JAXBench Level 2 - Conv2d_InstanceNorm_Divide
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

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
        x = jax.lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',  # PyTorch Conv2d default is no padding
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # Instance Normalization
        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x * self.instance_norm_weight.reshape(1, -1, 1, 1) + self.instance_norm_bias.reshape(1, -1, 1, 1)

        # Divide
        x = x / self.divide_by
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]