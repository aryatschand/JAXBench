"""
JAXBench Level 2 - Conv2d_Tanh_Scaling_BiasAdd_Max
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        # Initialize conv weights with same shape as PyTorch
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias_conv = jnp.zeros(out_channels)
        self.scaling_factor = scaling_factor
        self.bias = jnp.zeros(bias_shape)
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Prepare conv kernel
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Convolution with VALID padding (PyTorch default for nn.Conv2d is no padding)
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        
        # Add conv bias
        x = x + self.bias_conv.reshape(1, 1, 1, -1)
        
        # Tanh activation
        x = jnp.tanh(x)
        
        # Scaling
        x = x * self.scaling_factor
        
        # Bias addition (convert bias to NHWC format)
        bias_nhwc = jnp.transpose(self.bias, (1, 2, 0))
        x = x + bias_nhwc
        
        # Max pooling
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, self.pool_kernel_size, self.pool_kernel_size, 1),
            window_strides=(1, self.pool_kernel_size, self.pool_kernel_size, 1),
            padding='VALID')
            
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]