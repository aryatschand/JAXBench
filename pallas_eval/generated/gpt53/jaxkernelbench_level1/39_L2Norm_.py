"""
JAXBench Level 1 - Task 39: L2Norm_
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.134901
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        pass

    def forward(self, x):
        def kernel_fn(x_ref, o_ref):
            x_block = x_ref[0, :]
            sumsq = jnp.sum(x_block * x_block)
            norm = jnp.sqrt(sumsq)
            o_ref[0, :] = x_block / norm

        B, D = x.shape

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(B,),
                in_specs=[
                    pl.BlockSpec((1, D), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((1, D), lambda i: (i, 0)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
