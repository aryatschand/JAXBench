"""
JAXBench Level 1 - Task 46: Average_Pooling_3D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.136579
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import List

class Model:
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        if stride is None:
            stride = kernel_size
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        Applies Average Pooling to the input tensor.

        Args:
            x: Input array of shape (batch_size, channels, depth, height, width).

        Returns:
            Output array with Average Pooling applied.
        """
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Define window dimensions for NDHWC format (1, D, H, W, 1)
        window_shape = (1, self.kernel_size, self.kernel_size, self.kernel_size, 1)
        strides = (1, self.stride, self.stride, self.stride, 1)
        
        # Define padding for each dimension
        padding = [(0, 0), (self.padding, self.padding), (self.padding, self.padding), 
                   (self.padding, self.padding), (0, 0)]
        
        # Use lax.reduce_window for average pooling
        # Sum pooling
        pooled_sum = lax.reduce_window(
            x, 
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding=padding
        )
        
        # Divide by window size to get average
        window_size = self.kernel_size ** 3
        x = pooled_sum / window_size
                    
        # Convert back from NDHWC to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

    def set_weights(self, weights_dict):
        # No learnable parameters for AvgPool
        pass

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]