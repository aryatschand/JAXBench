"""
JAXBench Level 2 - ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        # ConvTranspose3d weight shape: (in_channels, out_channels, D, H, W)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract = jnp.zeros(out_channels)
        
        self.stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.padding_val = padding
        self.output_padding = output_padding
        
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in, out, D, H, W) -> (D, H, W, out, in)
        
        # Calculate padding for conv_transpose
        pad_val = self.kernel_size - 1 - self.padding_val
        padding = ((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val))
        
        x = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=self.stride,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
        
        if self.output_padding:
            # Pad configuration for jax.lax.pad: (low, high, interior) for each dimension
            x = jax.lax.pad(x, 0.0, 
                          ((0, 0, 0), 
                           (0, self.output_padding, 0),
                           (0, self.output_padding, 0),
                           (0, self.output_padding, 0),
                           (0, 0, 0)))
        
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # MaxPool3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        
        pool_window = (1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1)
        pool_strides = (1, self.pool_stride, self.pool_stride, self.pool_stride, 1)
        
        # For reduce_window, padding is a sequence of (low, high) pairs
        pool_padding = ((0, 0), 
                       (self.pool_padding, self.pool_padding),
                       (self.pool_padding, self.pool_padding),
                       (self.pool_padding, self.pool_padding),
                       (0, 0))
        
        x = jax.lax.reduce_window(x_ndhwc, -jnp.inf, jax.lax.max,
                                window_dimensions=pool_window,
                                window_strides=pool_strides,
                                padding=pool_padding)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # Softmax across channels
        x = jax.nn.softmax(x, axis=1)
        
        # Subtract
        x = x - self.subtract.reshape(1, -1, 1, 1, 1)
        
        # Swish activation
        x = x * jax.nn.sigmoid(x)
        
        # Max across channels
        x = jnp.max(x, axis=1)
        
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 3, 16, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, 1, 2, 2, 0]