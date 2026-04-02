"""
JAXBench Level 1 - Task 27: SELU_
Pallas TPU optimized version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def selu_kernel(x_ref, o_ref):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    x = x_ref[:, :]
    out = scale * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
    o_ref[:, :] = out

class Model:
    """
    Simple model that performs a SELU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        block = (8, 128)
        grid = (x.shape[0] // block[0], x.shape[1] // block[1])

        return pl.pallas_call(
            selu_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
