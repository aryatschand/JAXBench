"""
JAXBench Level 2 - ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        # For ConvTranspose3d, PyTorch weight shape is (in_channels, out_channels, k, k, k)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel (in, out, D, H, W) -> (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Calculate padding: kernel_size - 1 - pytorch_padding
        pad = self.weight.shape[2] - 1 - self.padding
        padding = ((pad, pad), (pad, pad), (pad, pad))
        
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Scale
        x = x * self.scale
        
        # MaxPool3d
        window_shape = (1, self.maxpool_kernel_size, self.maxpool_kernel_size, self.maxpool_kernel_size, 1)
        strides = window_shape
        x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window_shape, strides, 'VALID')
        
        # Global average pooling - reduce all spatial dims to 1
        spatial_shape = x.shape[1:4]
        x = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        
        # Convert back NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Clamp
        x = jnp.clip(x, self.clamp_min, self.clamp_max)
        
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 3, 16, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, 0.5, 2]