"""
JAXBench Level 1 - Task 96: HuberLoss (Pallas TPU optimized)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def huber_kernel(pred_ref, tgt_ref, out_ref):
    diff = pred_ref[...] - tgt_ref[...]
    abs_diff = jnp.abs(diff)
    loss = jnp.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    out_ref[...] = loss


class Model:
    def __init__(self):
        pass
    
    def set_weights(self, weights_dict):
        pass
    
    def forward(self, predictions, targets):
        n, m = predictions.shape

        block_m = 128
        block_n = 128

        grid = (n // block_m, m // block_n)

        loss = pl.pallas_call(
            huber_kernel,
            out_shape=jax.ShapeDtypeStruct(predictions.shape, predictions.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(predictions, targets)

        return jnp.mean(loss)


batch_size = 4096
input_shape = (4096,)
dim = 1


def get_inputs():
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    scale = jax.random.uniform(key1, ())
    predictions = jax.random.uniform(key2, (batch_size, *input_shape)) * scale
    targets = jax.random.uniform(key3, (batch_size, *input_shape))
    return [predictions, targets]


def get_init_inputs():
    return []
