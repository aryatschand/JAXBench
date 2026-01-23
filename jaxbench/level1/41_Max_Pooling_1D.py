"""
JAXBench Level 1 - Task 41: Max_Pooling_1D (reduced size to avoid memory issues)
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax


class Model:
    """
    Simple model that performs Max Pooling 1D.
    Uses jax.lax.reduce_window for pooling operations.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
    
    def set_weights(self, weights_dict):
        """No weights for pooling."""
        pass
    
    def forward(self, x):
        """
        x: (batch, channels, length) - NCL format
        """
        # Convert to NLC for easier pooling
        x_nlc = jnp.transpose(x, (0, 2, 1))  # (N, L, C)
        
        # Apply padding
        if self.padding > 0:
            # Pad the length dimension
            x_nlc = jnp.pad(x_nlc, ((0, 0), (self.padding, self.padding), (0, 0)), 
                          mode='constant', constant_values=-jnp.inf)
        
        # For dilated pooling, we need to handle it differently
        # Dilation in pooling means we sample every dilation-th element
        if self.dilation > 1:
            # Effective kernel size with dilation
            effective_kernel = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
            
            # Use reduce_window with proper window shape
            out = lax.reduce_window(
                x_nlc,
                init_value=-jnp.inf,
                computation=lax.max,
                window_dimensions=(1, effective_kernel, 1),
                window_strides=(1, self.stride, 1),
                padding='VALID'
            )
        else:
            out = lax.reduce_window(
                x_nlc,
                init_value=-jnp.inf,
                computation=lax.max,
                window_dimensions=(1, self.kernel_size, 1),
                window_strides=(1, self.stride, 1),
                padding='VALID'
            )
        
        # Convert back to NCL
        out = jnp.transpose(out, (0, 2, 1))
        
        return out


# Test code - REDUCED SIZE to avoid memory issues
batch_size = 16  # Reduced from 64
features = 64    # Reduced from 192
sequence_length = 8192  # Reduced from 65536

kernel_size = 8
stride = 1
padding = 4
dilation = 3

return_indices = False


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, features, sequence_length))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

