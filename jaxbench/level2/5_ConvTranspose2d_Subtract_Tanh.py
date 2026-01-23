"""
JAXBench Level 2 - ConvTranspose2d_Subtract_Tanh
Translated from KernelBench PyTorch to JAX using bedrock/opus.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

class Model:
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        
        # Initialize weights for ConvTranspose2d
        # PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, kernel_size, kernel_size)
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Initialize conv transpose weights
        self.conv_transpose_weight = jax.random.normal(key1, (in_channels, out_channels, kernel_size, kernel_size)) * 0.01
        self.conv_transpose_bias = jax.random.normal(key2, (out_channels,)) * 0.01
        
        # Initialize the bias parameter
        self.bias = jax.random.normal(key3, bias_shape)
    
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # x is in NCHW format
        # Convert to NHWC for JAX convolution
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        
        # PyTorch ConvTranspose2d weight is (in_channels, out_channels, kH, kW)
        # For JAX conv_transpose, we need (kH, kW, out_channels, in_channels) for NHWC
        weight = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))  # (kH, kW, out_channels, in_channels)
        
        # Calculate padding for JAX conv_transpose
        # PyTorch output: H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
        # JAX padding calculation: pad = kernel_size - 1 - pytorch_padding
        pad = self.kernel_size - 1 - self.padding
        
        # Perform transposed convolution
        x_conv = lax.conv_transpose(
            x_nhwc,
            weight,
            strides=(self.stride, self.stride),
            padding=((pad, pad + self.output_padding), (pad, pad + self.output_padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )
        
        # Add conv transpose bias (broadcast over spatial dimensions)
        x_conv = x_conv + self.conv_transpose_bias.reshape(1, 1, 1, -1)
        
        # Convert back to NCHW
        x_nchw = jnp.transpose(x_conv, (0, 3, 1, 2))  # NHWC -> NCHW
        
        # Subtract bias (bias_shape is (out_channels, 1, 1))
        x_nchw = x_nchw - self.bias
        
        # Apply tanh activation
        x_nchw = jnp.tanh(x_nchw)
        
        return x_nchw

batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(42)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]