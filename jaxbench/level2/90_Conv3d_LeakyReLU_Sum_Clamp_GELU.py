"""
JAXBench Level 2 - Conv3d_LeakyReLU_Sum_Clamp_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import gelu, leaky_relu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.sum_tensor = jnp.zeros(sum_tensor_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (out,in,D,H,W) to (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Rest of operations
        x = leaky_relu(x, negative_slope=0.2)
        x = x + self.sum_tensor
        x = jnp.clip(x, -1.0, 1.0)
        x = gelu(x)
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]