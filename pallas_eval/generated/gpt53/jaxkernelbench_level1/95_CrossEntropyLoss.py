import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cross_entropy_kernel(x_ref, t_ref, o_ref):
    x = x_ref[...]  # (B, C)
    t = t_ref[:, 0]  # (B,)

    # log-sum-exp
    row_max = jnp.max(x, axis=1, keepdims=True)
    exp_shifted = jnp.exp(x - row_max)
    sum_exp = jnp.sum(exp_shifted, axis=1, keepdims=True)
    lse = row_max + jnp.log(sum_exp)

    # gather target logits
    tgt = jnp.take_along_axis(x, t[:, None], axis=1)

    loss = lse - tgt  # (B, 1)
    o_ref[...] = loss


class Model:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        # reshape targets to 2D for TPU constraint
        targets_2d = targets[:, None]

        B, C = predictions.shape
        block = (128, C)
        grid = (B // block[0], 1)

        losses = pl.pallas_call(
            cross_entropy_kernel,
            out_shape=jax.ShapeDtypeStruct((B, 1), predictions.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, 0)),
                    pl.BlockSpec((block[0], 1), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block[0], 1), lambda i, j: (i, 0)),
            ),
        )(predictions, targets_2d)

        return jnp.mean(losses)


batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1


def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    return [
        jax.random.uniform(key1, shape=(batch_size, *input_shape)),
        jax.random.randint(key2, shape=(batch_size,), minval=0, maxval=num_classes),
    ]


def get_init_inputs():
    return []
