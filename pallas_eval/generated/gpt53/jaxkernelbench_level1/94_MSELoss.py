import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def mse_kernel(pred_ref, tgt_ref, out_ref):
    diff = pred_ref[...] - tgt_ref[...]
    out_ref[...] = diff * diff


def mse_pallas(predictions, targets):
    n, m = predictions.shape
    block = (128, 128)
    grid = (n // block[0], m // block[1])

    squared = pl.pallas_call(
        mse_kernel,
        out_shape=jax.ShapeDtypeStruct(predictions.shape, predictions.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i, j: (i, j)),
                pl.BlockSpec(block, lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(predictions, targets)

    return jnp.mean(squared)


class Model:
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        return mse_pallas(predictions, targets)


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    scale = jax.random.uniform(key1, shape=())
    return [jax.random.uniform(key2, shape=(batch_size, *input_shape)) * scale,
            jax.random.uniform(key3, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return []
