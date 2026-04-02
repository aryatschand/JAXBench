```python
"""
JAXBench Level 1 - Task 23: Softmax
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.129344
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def max_kernel(x_ref, out_ref):
    # x_ref: (8, 1024), out_ref: (8, 128)
    m = jnp.max(x_ref[...], axis=1, keepdims=True)
    out_ref[...] = pltpu.repeat(m, 128, axis=1)

def sum_kernel(x_ref, max_ref, out_ref):
    # x_ref: (8, 1024), max_ref: (8, 128), out_ref: (8, 128)
    x = x_ref[...]
    m = max_ref[:, 0:1]
    s = jnp.sum(jnp.exp(x - m), axis=1, keepdims=True)
    out_ref[...] = pltpu.repeat(s, 128, axis=1)

def softmax_kernel(x_ref, max_ref, sum_ref, out_ref):
    # x_ref: (8, 1024), max_ref: (8, 128), sum_ref: (8, 128), out_ref: (8, 1024)
    x = x_ref[...]
    m = max_ref[:, 0:1]
    s = sum_ref[:, 0:1]
    out_ref[...] = jnp.exp(x - m) / s

class Model:
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies Softmax activation to the input tensor.

        Args:
            x: Input array of shape (batch_size, num_features).

        Returns:
            Output array with Softmax applied, same shape as input.
        """
        batch_size, dim = x.shape
        
        # We tile the computation into blocks of (8, 1024) to satisfy TPU constraints
        # and fit comfortably within VMEM.
        grid_x = batch_size // 8
        grid_y = dim // 1024
        grid_shape = (grid_x, grid_y)
        
        # Step 1: Compute local max per block
        max_out_shape =
