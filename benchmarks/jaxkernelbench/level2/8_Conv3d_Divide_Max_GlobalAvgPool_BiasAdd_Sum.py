"""
JAXBench Level 2 - Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        # Conv3d weight shape: (out_channels, in_channels, kD, kH, kW)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.bias_conv = jnp.zeros(out_channels)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = jnp.zeros(bias_shape)
        self.sum_dim = sum_dim

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv3d: NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        # Transpose kernel: (out,in,D,H,W) -> (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        # Add conv bias
        x = x + self.bias_conv.reshape(1, 1, 1, 1, -1)
        
        # Division
        x = x / self.divisor
        
        # Max pooling
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1,) + self.pool_size + (1,),
            window_strides=(1,) + self.pool_size + (1,),
            padding='VALID'
        )
        
        # Global average pooling - reduce to (N, 1, 1, 1, C)
        for axis, size in enumerate(x.shape[1:-1], 1):
            x = jnp.mean(x, axis=axis, keepdims=True)
            
        # Back to NCDHW format
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Add bias and sum
        x = x + self.bias
        x = jnp.sum(x, axis=self.sum_dim)
        
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 8, 16, 64, 64))]

def get_init_inputs():
    return [8, 16, (3, 3, 3), 2.0, (2, 2, 2), (16, 1, 1, 1), 1]