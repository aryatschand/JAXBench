"""
JAXBench Level 1 - Task 44: Average_Pooling_1D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.136252
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x: Input array of shape (batch_size, in_channels, input_length).

        Returns:
            Output array with 1D Average Pooling applied.
        """
        # Convert NCL to NLC for JAX pooling
        x = jnp.transpose(x, (0, 2, 1))
        
        # Pad the input manually
        if self.padding > 0:
            x = jnp.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)
        
        # Use lax.reduce_window for average pooling
        # Window shape is (1, kernel_size, 1) for NLC format
        window_shape = (1, self.kernel_size, 1)
        strides = (1, self.stride, 1)
        
        # Sum pooling
        out = lax.reduce_window(x, 0.0, lax.add, window_shape, strides, 'VALID')
        
        # Divide by kernel size to get average
        out = out / self.kernel_size
        
        # Transpose back to NCL
        out = jnp.transpose(out, (0, 2, 1))
        
        return out

    def set_weights(self, weights_dict):
        # No learnable parameters
        pass

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, input_length))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]