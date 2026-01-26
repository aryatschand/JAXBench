"""
JAXBench Level 2 - ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        # Initialize weights with same shape as PyTorch (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier
        self.kernel_size = kernel_size  # Store kernel_size as instance attribute

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel (in, out, H, W) -> (H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        # Calculate padding for conv_transpose
        # For ConvTranspose2d with output_padding, we need to handle it specially
        # JAX conv_transpose padding: pad = kernel_size - 1 - pytorch_padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        
        # ConvTranspose2d
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=((pad_h, pad_h + self.output_padding), (pad_w, pad_w + self.output_padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        # Add bias
        x = x + self.bias.reshape(1, 1, 1, -1)

        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Multiply by scalar
        x = x * self.multiplier

        # First global average pooling
        x = jnp.mean(x, axis=(2, 3), keepdims=True)

        # Second global average pooling 
        x = jnp.mean(x, axis=(2, 3), keepdims=True)

        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]