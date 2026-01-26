"""
JAXBench Level 2 - ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        # Initialize ConvTranspose2d weight with PyTorch shape (in_channels, out_channels, kH, kW)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size  # Store kernel_size as instance attribute
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # (in,out,H,W) -> (H,W,out,in)
        pad_val = self.kernel_size - 1 - self.padding
        padding = ((pad_val, pad_val), (pad_val, pad_val))
        x = jax.lax.conv_transpose(x_nhwc, kernel, 
                                 strides=(self.stride, self.stride),
                                 padding=padding,
                                 dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # MaxPool2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        window_shape = (1, self.maxpool_kernel_size, self.maxpool_kernel_size, 1)
        strides = (1, self.maxpool_stride, self.maxpool_stride, 1)
        x = jax.lax.reduce_window(x_nhwc, -jnp.inf, jax.lax.max, 
                                window_dimensions=window_shape,
                                window_strides=strides, 
                                padding='VALID')
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Hardtanh
        x = jnp.clip(x, self.hardtanh_min, self.hardtanh_max)

        # Mean and tanh
        x = jnp.mean(x, axis=(2, 3), keepdims=True)
        x = jnp.tanh(x)
        
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 64, 256, 256))]

def get_init_inputs():
    return [64, 64, 3, 1, 1, 2, 2, -1, 1]