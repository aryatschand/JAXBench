"""
JAXBench Level 1 - Task 45: Average_Pooling_2D (reduced size to avoid memory issues)
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Simple model that performs 2D Average Pooling.
    Uses jax.lax.reduce_window for pooling operations.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def set_weights(self, weights_dict):
        """No weights for pooling."""
        pass
    
    def forward(self, x):
        """
        x: (batch, channels, height, width) - NCHW format
        """
        # Convert to NHWC for easier pooling
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # (N, H, W, C)
        
        # Apply padding
        if self.padding > 0:
            x_nhwc = jnp.pad(x_nhwc, 
                           ((0, 0), (self.padding, self.padding), 
                            (self.padding, self.padding), (0, 0)), 
                           mode='constant', constant_values=0)
        
        # Sum pooling
        out_sum = lax.reduce_window(
            x_nhwc,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=(1, self.kernel_size, self.kernel_size, 1),
            window_strides=(1, self.stride, self.stride, 1),
            padding='VALID'
        )
        
        # Divide by kernel area for average
        out = out_sum / (self.kernel_size * self.kernel_size)
        
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        
        return out


# Test code - REDUCED SIZE to avoid memory issues
batch_size = 4   # Reduced from 16
channels = 32    # Reduced from 64
height = 512     # Reduced from 2048
width = 512      # Reduced from 2048
kernel_size = 11


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, channels, height, width))
    return [x]


def get_init_inputs():
    return [kernel_size]

