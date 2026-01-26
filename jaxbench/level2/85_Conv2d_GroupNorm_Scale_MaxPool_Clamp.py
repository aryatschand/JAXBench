"""
JAXBench Level 2 - Conv2d_GroupNorm_Scale_MaxPool_Clamp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        # Conv2d weights
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        
        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))
        self.num_groups = num_groups
        self.out_channels = out_channels
        
        # Scale parameter
        self.scale = jnp.ones(scale_shape)
        
        # MaxPool params
        self.maxpool_kernel_size = maxpool_kernel_size
        
        # Clamp params
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # OIHW -> HWIO
        x = jax.lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # GroupNorm
        N, C, H, W = x.shape
        x = x.reshape(N, self.num_groups, -1)
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.var(x, axis=2, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, H, W)
        x = x * self.group_norm_weight.reshape(1, -1, 1, 1) + self.group_norm_bias.reshape(1, -1, 1, 1)

        # Scale
        x = x * self.scale

        # MaxPool
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        window_shape = (1, self.maxpool_kernel_size, self.maxpool_kernel_size, 1)
        x = jax.lax.reduce_window(
            x_nhwc,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=window_shape,
            window_strides=window_shape,
            padding='VALID')
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # Clamp
        x = jnp.clip(x, self.clamp_min, self.clamp_max)
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 128, 128
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]