"""
JAXBench Level 1 - Task 47: Sum_reduction_over_a_dimension
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.136904
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self.dim == 1, "This optimized kernel currently supports dim=1 only."

        B, M, N = x.shape

        def kernel(x_ref, o_ref):
            x_block = x_ref[...]            # (1, M, N)
            summed = jnp.sum(x_block, axis=1, keepdims=True)  # (1, 1, N)
            o_ref[...] = summed

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, 1, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(B, 1, 1),
                in_specs=[
                    pl.BlockSpec((1, M, N), lambda i, j, k: (i, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, N), lambda i, j, k: (i, 0, 0)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]

def get_init_inputs():
    return [reduce_dim]
