```python
"""
JAXBench Level 1 - Task 41: Max_Pooling_1D (reduced size to avoid memory issues)
Manually translated from KernelBench PyTorch to JAX
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    """
    Simple model that performs Max Pooling 1D.
    Uses a highly optimized Pallas TPU kernel.
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
        N, C, L = x.shape
        M = N * C
        
        # Flatten batch and channels to process them uniformly
        x_flat = x.reshape(M, L)
        
        # Apply padding
        if self.padding > 0:
            x_padded = jnp.pad(x_flat, ((0, 0), (self.padding, self.padding)), 
                               mode='constant', constant_values=-jnp.inf)
        else:
            x_padded = x_flat
            
        L_padded = x_padded.shape[1]
        
        # Calculate effective kernel size with dilation
        if self.dilation > 1:
            effective_kernel = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        else:
            effective_kernel = self.kernel_size
            
        stride = self.stride
        
        # Calculate output length (VALID padding logic)
        L_out = (L_padded - effective_kernel) // stride + 1
        
        # Pad M to be a multiple of block size (bm = 16)
        bm = 16
        if M % bm != 0:
            pad_M = bm - (M % bm)
            x_padded = jnp.pad(x_padded, ((0, pad_M), (0, 0)), mode='constant', constant_values=-jnp.inf)
            M_padded = M + pad_M
        else:
            M_padded = M
            
        def max_pool_kernel(x_ref, o
