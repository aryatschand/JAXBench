"""
JAXBench Level 1 - Task 91: cumsum_reverse
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.148605
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        # Fallback to vanilla JAX if conditions for the optimized Pallas kernel are not met
        if self.dim != 1 or x.ndim != 2:
            return jnp.flip(jnp.cumsum(jnp.flip(x, axis=self.dim), axis=self.dim), axis=self.dim)
        
        block_rows = 8
        block_cols = x.shape[1]
        
        # Ensure block dimensions are valid for TPU f32 (multiples of 8 and 128)
        if x.shape[0] % block_rows != 0 or block_cols % 128 != 0:
            return jnp.flip(jnp.cumsum(jnp.flip(x, axis=self.dim), axis=self.dim), axis=self.dim)

        grid_shape = (x.shape[0] // block_rows, 1)

        def kernel(x_ref, o_ref):
            # Load the entire block into VMEM
            x_val = x_ref[...]
            
            # Perform reverse cumulative sum along axis 1
            rev_x = jnp.flip(x_val, axis=1)
            csum = jnp.cumsum(rev_x, axis=1)
            res = jnp.flip(csum, axis=1)
            
            # Store the result back to HBM
            o_ref[...] = res

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec((block_rows, block_cols), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_rows, block_cols), lambda i, j: (i, j)),
            ),
        )(x)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]
