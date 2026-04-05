"""
JAXBench Level 2 - Conv2D_ReLU_BiasAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        # Initialize with zeros - weights will be set via set_weights()
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (out,in,H,W) to (H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        # Perform convolution with VALID padding (PyTorch default is no padding)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        # Add conv bias
        x = x + self.conv_bias.reshape(1, 1, 1, -1)

        # ReLU
        x = jax.nn.relu(x)

        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Add bias term
        x = x + self.bias

        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]