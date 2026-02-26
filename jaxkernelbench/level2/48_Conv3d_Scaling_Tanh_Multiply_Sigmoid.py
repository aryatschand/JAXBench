"""
JAXBench Level 2 - Conv3d_Scaling_Tanh_Multiply_Sigmoid
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        # Initialize Conv3d weights with PyTorch shape (out_channels, in_channels, D, H, W)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias_conv = jnp.zeros(out_channels)
        self.scaling_factor = jnp.zeros(bias_shape)
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel (out,in,D,H,W) -> (D,H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D Convolution with VALID padding (PyTorch default is no padding)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        # Add conv bias
        x = x + self.bias_conv.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        x = x * self.scaling_factor
        x = jnp.tanh(x)
        x = x * self.bias
        x = jax.nn.sigmoid(x)
        return x

batch_size = 128
in_channels = 3  
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]