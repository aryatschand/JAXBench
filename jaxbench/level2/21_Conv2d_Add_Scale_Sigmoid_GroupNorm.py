"""
JAXBench Level 2 - Conv2d_Add_Scale_Sigmoid_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        # Initialize weights with same shapes as PyTorch
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.scale = jnp.zeros(scale_shape)
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))
        self.num_groups = num_groups
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d: NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.conv_weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding (PyTorch default is no padding)
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add conv bias
        x = x + self.conv_bias.reshape(1, 1, 1, -1)
        
        # Back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Add bias and scale
        x = x + self.bias
        x = x * self.scale
        
        # Sigmoid
        x = jax.nn.sigmoid(x)
        
        # Group normalization
        N, C, H, W = x.shape
        G = self.num_groups
        x = jnp.reshape(x, (N, G, C // G, H, W))
        
        mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        
        x = jnp.reshape(x, (N, C, H, W))
        x = x * self.group_norm_weight.reshape(1, -1, 1, 1)
        x = x + self.group_norm_bias.reshape(1, -1, 1, 1)
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]