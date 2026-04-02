```python
"""
JAXBench Level 1 - Task 51: Argmax_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.138352
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        self.dim = dim

    def forward(self, x):
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x: Input JAX array.

        Returns:
            JAX array: Output array with argmax applied, with the specified dimension removed.
        """
        dim = self.dim % x.ndim
        orig_shape = x.shape
        
        # Pad all dimensions to be multiples of 128 to satisfy TPU block size constraints
        pad_sizes = []
        padded_shape = []
        for s in orig_shape:
            rem = s % 128
            if rem == 0:
                pad_sizes.append((0, 0))
                padded_shape.append(s)
            else:
                pad = 128 - rem
                pad_sizes.append((0, pad))
                padded_shape.append(s + pad)
                
        needs_padding = any(p[1] > 0 for p in pad_sizes)
        if needs_padding:
            x_pad = jnp.pad(x, pad_sizes, constant_values=-jnp.inf)
        else:
            x_pad = x
            
        non_reduce_dims = [i for i in range(x_pad.ndim) if i != dim]
        
        # For 3D input, we have 2 non-reduce dimensions mapped to the grid
        dim_A, dim_B = non_reduce_dims[0], non_reduce_dims[1]
        
        # Choose block sizes to keep VMEM usage well within the 16MB limit
        # while ensuring the minor dimensions are multiples of 8 and 128.
        if dim == 0:
            block_A = 8
            block_B = 128
        else:
            block_A = 1
            block_B = 128
            
        grid_A = padded_shape[dim_A] // block_A
