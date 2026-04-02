"""
JAXBench Level 1 - Task 23: Softmax
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.129344
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def softmax_kernel(x_ref, o_ref):
    x = x_ref[0, :]
    x_max = jnp.max(x)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    denom = jnp.sum(exp_x)
    o_ref[0, :] = exp_x / denom


class Model:
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        batch, dim = x.shape

        return pl.pallas_call(
            softmax_kernel,
            out_shape=jax.ShapeDtypeStruct((batch, dim), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch,),
                in_specs=[pl.BlockSpec((1, dim), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((1, dim), lambda i: (i, 0)),
            ),
        )(x)


batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
