"""
JAXBench Level 2 - ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel (in,out,H,W) -> (H,W,out,in) for JAX conv_transpose
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        # For conv_transpose with output_padding, we need to handle it specially
        # JAX conv_transpose padding calculation:
        # For PyTorch ConvTranspose2d with padding=p and output_padding=op:
        # The output size is: (input - 1) * stride - 2*padding + kernel_size + output_padding
        # 
        # In JAX, we use padding='SAME' or explicit padding
        # To match PyTorch's output_padding, we need to adjust the padding
        
        k = self._kernel_size
        
        # Use padding that accounts for PyTorch's padding parameter
        # For transposed conv: pad = kernel_size - 1 - pytorch_padding
        pad_h = k - 1 - self.padding
        pad_w = k - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding), (pad_w, pad_w + self.output_padding))

        # ConvTranspose2d
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        # Convert back NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Add conv_transpose bias (reshape to broadcast over spatial dims)
        x = x + self.conv_transpose_bias.reshape(1, -1, 1, 1)

        # Add the additional bias parameter
        x = x + self.bias

        # Clamp, scale, clamp, divide
        x = jnp.clip(x, 0.0, 1.0)
        x = x * self.scaling_factor
        x = jnp.clip(x, 0.0, 1.0)
        x = x / self.scaling_factor

        return x

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]