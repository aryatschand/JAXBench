import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        # Ensure 2D
        preds = predictions
        t = targets.reshape(-1, 1)

        def kernel_fn(x_ref, t_ref, o_ref):
            x = x_ref[:, :]
            tt = t_ref[:, :]
            tt_b = pltpu.repeat(tt, x.shape[1], axis=1)
            out = jnp.maximum(1.0 - x * tt_b, 0.0)
            o_ref[:, :] = out

        block = (128, 128)
        grid = (preds.shape[0] // block[0], preds.shape[1] // block[1])

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(preds.shape, preds.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec((block[0], 1), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(preds, t)

        return jnp.mean(out)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    return [
        jax.random.uniform(key1, shape=(batch_size, *input_shape)),
        jax.random.randint(key2, shape=(batch_size,), minval=0, maxval=2) * 2 - 1
    ]

def get_init_inputs():
    return []
