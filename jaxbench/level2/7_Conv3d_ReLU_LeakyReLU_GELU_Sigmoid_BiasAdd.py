"""
JAXBench Level 2 - Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.nn import relu, sigmoid, gelu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        # Initialize with zeros - weights will be set via set_weights()
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (out,in,d,h,w) to (d,h,w,in,out)
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))
        
        # 3D convolution with VALID padding (PyTorch default is no padding)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        # Add conv bias
        x = x + self.conv_bias.reshape(1, 1, 1, 1, -1)
        
        # Convert back to NCDHW for activations
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Apply activations
        x = relu(x)
        x = jnp.where(x >= 0, x, 0.01 * x)  # leaky relu
        x = gelu(x)
        x = sigmoid(x)
        
        # Add bias
        x = x + self.bias
        
        return x

batch_size = 64
in_channels = 8  
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]