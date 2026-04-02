import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import softmax


def kldiv_kernel(pred_ref, tgt_ref, out_ref):
    pred = pred_ref[:, :]
    tgt = tgt_ref[:, :]
    val = tgt * (jnp.log(tgt) - jnp.log(pred))
    out_ref[0, 0] = jnp.sum(val)


class Model:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        # ensure 2D
        pred = predictions
        tgt = targets

        m, n = pred.shape

        bm = 128
        bn = 128

        gm = m // bm
        gn = n // bn

        partial = pl.pallas_call(
            kldiv_kernel,
            out_shape=jax.ShapeDtypeStruct((gm, gn), pred.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(gm, gn),
                in_specs=[
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((1, 1), lambda i, j: (i, j)),
            ),
        )(pred, tgt)

        total = jnp.sum(partial)
        return total / (m * n)


batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1


def get_inputs():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    scale = jax.random.uniform(key1)
    pred = softmax(jax.random.uniform(key2, (batch_size, *input_shape)) * scale, axis=-1)
    tgt = softmax(jax.random.uniform(key3, (batch_size, *input_shape)), axis=-1)
    return [pred, tgt]


def get_init_inputs():
    return []
