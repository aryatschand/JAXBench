"""
JAXBench Level 2 - Conv3d_Max_LogSumExp_ReLU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv3d: NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(self.stride, self.stride, self.stride),
            padding=[(self.padding, self.padding)] * 3,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
        
        # Add bias
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        
        # MaxPool3d
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID')
        
        # Convert back to NCDHW for remaining ops
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # logsumexp along channel dim
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
        
        # ReLU
        x = jax.nn.relu(x)
        
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]