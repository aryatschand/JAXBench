"""
JAXBench Level 2 - Conv3d_Min_Softmax
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        self.dim = dim
        # Initialize conv weights with PyTorch shape (out_channels, in_channels, D, H, W)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (out,in,D,H,W) -> (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW for min operation
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Min along specified dimension
        x = jnp.min(x, axis=self.dim)
        
        # Softmax along channel dimension (dim=1)
        x = jax.nn.softmax(x, axis=1)
        
        return x

batch_size = 128
in_channels = 3  
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]