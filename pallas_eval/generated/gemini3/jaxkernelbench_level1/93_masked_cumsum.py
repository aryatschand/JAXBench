"""
JAXBench Level 1 - Task 93: masked_cumsum
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.149306
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape).
            mask (jnp.ndarray): Boolean mask of the same shape as x.

        Returns:
            jnp.ndarray: Cumulative sum of elements where mask is True.
        """
        original_shape = x.shape
        dim = self.dim
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
            mask = mask.reshape(1, -1)
            dim = 1 if dim == 0 or dim == -1 else dim
            
        if x.ndim == 2:
            dim = dim % 2
            if dim == 1:
                block_0 = 8
                while x.shape[0] % block_0 != 0 and block_0 > 1:
                    block_0 -= 1
                block_1 = x.shape[1]
                
                grid = (x.shape[0] // block_0,)
                
                def kernel_dim1(x_ref, mask_ref, o_ref):
                    x_val = x_ref[...]
                    mask_val = mask_ref[...]
                    masked_x = x_val * mask_val.astype(x_val.dtype)
                    o_ref[...] = jnp.cumsum(masked_x, axis=1)
                    
                out = pl.pallas_call(
                    kernel_dim1,
                    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        grid=grid,
                        in_specs=[
                            pl.BlockSpec((block_0, block_1), lambda i: (i, 0)),
                            pl.BlockSpec((block_0, block_1), lambda i: (i, 0)),
                        ],
                        out_specs=pl.BlockSpec((block_0, block_1), lambda i: (i, 0)),
                    )
                )(x, mask)
                return out.reshape(original_shape)
                
            elif dim == 0:
                block_1 = 8
                while x.shape[1] % block_1 != 0 and block_1 > 1:
                    block_1 -= 1
                block_0 = x.shape[0]
                
                grid = (x.shape[1] // block_1,)
                
                def kernel_dim0(x_ref, mask_ref, o_ref):
                    x_val = x_ref[...]
                    mask_val = mask_ref[...]
                    masked_x = x_val * mask_val.astype(x_val.dtype)
                    o_ref[...] = jnp.cumsum(masked_x, axis=0)
                    
                out = pl.pallas_call(
                    kernel_dim0,
                    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        grid=grid,
                        in_specs=[
                            pl.BlockSpec((block_0, block_1), lambda i: (0, i)),
                            pl.BlockSpec((block_0, block_1), lambda i: (0, i)),
                        ],
                        out_specs=pl.BlockSpec((block_0, block_1), lambda i: (0, i)),
                    )
                )(x, mask)
                return out.reshape(original_shape)
                
        # Fallback for >2D or unsupported cases
        return jnp.cumsum(x * mask, axis=self.dim).reshape(original_shape)

    def set_weights(self, weights_dict):
        pass

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.uniform(key1, shape=(batch_size, *input_shape))
    mask = jax.random.bernoulli(key2, shape=(batch_size, *input_shape))
    return [x, mask]

def get_init_inputs():
    return [dim]
