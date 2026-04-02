"""
JAXBench Level 1 - Task 89: cumsum
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.147886
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape), where `*input_shape` 
                            can vary depending on the use case.

        Returns:
            jnp.ndarray: Array of the same shape as `x` after applying cumulative sum along `dim`.
        """
        dim = self.dim
        if dim < 0:
            dim += x.ndim

        # Fallback to vanilla JAX for non-2D arrays or unsupported dimensions
        if x.ndim != 2 or dim not in (0, 1):
            return jnp.cumsum(x, axis=self.dim)

        if dim == 0:
            # Process full columns. Block width is up to 128.
            b1 = 128 if x.shape[1] % 128 == 0 else (8 if x.shape[1] % 8 == 0 else 1)
            block_shape = (x.shape[0], b1)
            grid_shape = (1, x.shape[1] // b1)
            def index_map(i, j): return (0, j)
        else:  # dim == 1
            # Process full rows. Block height is up to 128.
            b0 = 128 if x.shape[0] % 128 == 0 else (8 if x.shape[0] % 8 == 0 else 1)
            block_shape = (b0, x.shape[1])
            grid_shape = (x.shape[0] // b0, 1)
            def index_map(i, j): return (i, 0)

        def cumsum_kernel(x_ref, o_ref):
            o_ref[...] = jnp.cumsum(x_ref[...], axis=dim)

        return pl.pallas_call(
            cumsum_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[pl.BlockSpec(block_shape, index_map)],
                out_specs=pl.BlockSpec(block_shape, index_map),
            ),
        )(x)

    def set_weights(self, weights_dict):
        """
        No-op since this model has no learnable parameters.
        """
        pass

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated array with shape 
              (batch_size, *input_shape).
    """
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]
