"""
JAXBench Level 2 - ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # (in,out,H,W) -> (H,W,out,in)
        padding = ((2, 2), (2, 2))  # kernel_size - 1 - pytorch_padding
        out = jax.lax.conv_transpose(
            x_nhwc, kernel, 
            strides=(1, 1),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        x = jnp.transpose(out, (0, 3, 1, 2))  # NHWC -> NCHW

        # Global average pooling
        x = jnp.mean(x, axis=(2, 3), keepdims=True)

        # Add bias
        x = x + self.bias

        # Log-sum-exp
        x = jax.nn.logsumexp(x, axis=1, keepdims=True)

        # Sum
        x = jnp.sum(x, axis=(2, 3))

        # Multiplication
        x = x * 10.0

        return x

batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]