"""
JAXBench Level 2 - ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide
Translated from KernelBench PyTorch to JAX using bedrock/opus.
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        self.scaling_factor = scaling_factor
        
        # Initialize weights for ConvTranspose2d
        # PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, kernel_size, kernel_size)
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Initialize conv_transpose weight
        self.conv_transpose_weight = jax.random.normal(key1, (in_channels, out_channels, kernel_size, kernel_size)) * 0.01
        self.conv_transpose_bias = jax.random.normal(key2, (out_channels,)) * 0.01
        
        # Initialize the additional bias parameter
        self.bias = jax.random.normal(key3, bias_shape)
    
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Transposed convolution using jax.lax.conv_transpose
        # x shape: (batch, in_channels, height, width)
        # weight shape: (in_channels, out_channels, kernel_height, kernel_width)
        
        # For conv_transpose, we need to rearrange dimensions
        # JAX conv_transpose expects: input (N, H, W, C), kernel (H, W, I, O)
        # We're using NCHW format, so we need to transpose
        
        # Transpose input from NCHW to NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose weight from (I, O, H, W) to (H, W, O, I) for conv_transpose
        # Note: PyTorch ConvTranspose2d weight is (in_channels, out_channels, kH, kW)
        weight_hwoi = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))
        
        # Perform transposed convolution
        x_conv = lax.conv_transpose(
            x_nhwc,
            weight_hwoi,
            strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )
        
        # Handle output_padding by padding the output
        if self.output_padding > 0:
            x_conv = jnp.pad(x_conv, ((0, 0), (0, self.output_padding), (0, self.output_padding), (0, 0)))
        
        # Add conv_transpose bias (broadcast over spatial dimensions)
        x_conv = x_conv + self.conv_transpose_bias.reshape(1, 1, 1, -1)
        
        # Transpose back to NCHW
        x = jnp.transpose(x_conv, (0, 3, 1, 2))
        
        # Add the additional bias term
        x = x + self.bias
        
        # Clamp between 0 and 1
        x = jnp.clip(x, 0.0, 1.0)
        
        # Scale
        x = x * self.scaling_factor
        
        # Clamp again
        x = jnp.clip(x, 0.0, 1.0)
        
        # Divide by scaling factor
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
    key = jax.random.PRNGKey(42)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]