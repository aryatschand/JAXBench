"""
JAXBench Level 2 - ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU
Translated from KernelBench PyTorch to JAX using bedrock/opus.
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.norm_shape = norm_shape
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights
        key = jax.random.PRNGKey(0)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # ConvTranspose3d weight: PyTorch shape is (in_channels, out_channels, kD, kH, kW)
        self.conv_transpose_weight = jax.random.normal(key1, (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])) * 0.01
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        
        # Sum weight parameter
        self.sum_weight = jnp.array(sum_weight)
        
        # LayerNorm parameters
        self.norm_weight = jnp.ones(norm_shape)
        self.norm_bias = jnp.zeros(norm_shape)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        # PyTorch input: NCDHW, JAX needs NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        
        # PyTorch ConvTranspose3d weight: (in_channels, out_channels, kD, kH, kW)
        # For transposed convolution with DHWOI format, we need (kD, kH, kW, out_channels, in_channels)
        # The kernel connects in_channels (input) to out_channels (output)
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        
        # Calculate padding for transposed convolution
        # pad = kernel_size - 1 - pytorch_padding
        pad_d = self.kernel_size[0] - 1 - self.padding[0]
        pad_h = self.kernel_size[1] - 1 - self.padding[1]
        pad_w = self.kernel_size[2] - 1 - self.padding[2]
        
        padding_spec = (
            (pad_d, pad_d + self.output_padding[0]),
            (pad_h, pad_h + self.output_padding[1]),
            (pad_w, pad_w + self.output_padding[2])
        )
        
        x = lax.conv_transpose(
            x,
            kernel,
            strides=self.stride,
            padding=padding_spec,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
            transpose_kernel=True
        )
        
        # Add bias
        x = x + self.conv_transpose_bias
        
        # Convert back to NCDHW for consistency, then back to NDHWC for operations
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
        
        # Add sum_weight
        x = x + self.sum_weight
        
        # Layer normalization over channels (last dim after transpose)
        # norm_shape is (out_channels,), so we normalize over channel dimension
        # For NCDHW, channels is dim 1
        mean = jnp.mean(x, axis=1, keepdims=True)
        var = jnp.var(x, axis=1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        # Apply weight and bias - reshape for broadcasting over NCDHW
        x = x * self.norm_weight.reshape(1, -1, 1, 1, 1) + self.norm_bias.reshape(1, -1, 1, 1, 1)
        
        # Average pooling 3D
        # Convert to NDHWC for pooling
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        
        window_shape = (1, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2], 1)
        strides = (1, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2], 1)
        
        x = lax.reduce_window(
            x,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        )
        x = x / (self.pool_kernel_size[0] * self.pool_kernel_size[1] * self.pool_kernel_size[2])
        
        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
        
        # GELU activation
        x = jax.nn.gelu(x)
        
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]