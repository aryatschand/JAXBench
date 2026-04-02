"""
JAXBench Level 1 - Task 90: cumprod
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.148274
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def cumprod_kernel_dim1(x_ref, o_ref):
    o_ref[...] = jnp.cumprod(x_ref[...], axis=1)

def cumprod_kernel_dim0(x_ref, o_ref):
    o_ref[...] = jnp.cumprod(x_ref[...], axis=0)

def get_block_size(dim_to_split, dim_to_keep, itemsize):
    # Target about 4MB per buffer to fit comfortably in VMEM with intermediates
    max_B = max(1, (4 * 1024 * 1024) // (dim_to_keep * itemsize))
    max_B = min(max_B, 128)
    B = min(max_B, dim_to_split)
    while B > 0:
        if dim_to_split % B == 0:
            return B
        B -= 1
    return 1

class Model:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape).

        Returns:
            jnp.ndarray: Array of the same shape as `x` after applying cumulative product along `dim`.
        """
        if x.ndim != 2:
            return jnp.cumprod(x, axis=self.dim)
            
        shape = x.shape
        itemsize = x.dtype.itemsize
        actual_dim = self.dim % x.ndim

        if actual_dim == 1:
            B = get_block_size(shape[0], shape[1], itemsize)
            block_shape = (B, shape[1])
            grid = (shape[0] // B,)
            return pl.pallas_call(
                cumprod_kernel_dim1,
                out_shape=jax.ShapeDtypeStruct(shape, x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid,
                    in_specs=[pl.BlockSpec(block_shape, lambda i: (i, 0))],
                    out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0)),
                )
            )(x)
        elif actual_dim == 0:
            B = get_block_size(shape[1], shape[0], itemsize)
            block_shape = (shape[0], B)
            grid = (shape[1] // B,)
            return pl.pallas_call(
                cumprod_kernel_dim0,
                out_shape=jax.ShapeDtypeStruct(shape, x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid,
                    in_specs=[pl.BlockSpec(block_shape, lambda i: (0, i))],
                    out_specs=pl.BlockSpec(block_shape, lambda i: (0, i)),
                )
            )(x)
        else:
            return jnp.cumprod(x, axis=self.dim)

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]
