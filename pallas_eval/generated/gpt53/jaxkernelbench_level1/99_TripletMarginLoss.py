import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, margin=1.0):
        self.margin = margin

    def forward(self, anchor, positive, negative):
        B, D = anchor.shape

        # ensure 2D output (B, 1)
        def kernel(a_ref, p_ref, n_ref, o_ref):
            a = a_ref[:, :]
            p = p_ref[:, :]
            n = n_ref[:, :]

            diff_pos = a - p
            diff_neg = a - n

            d_pos = jnp.sqrt(jnp.sum(diff_pos * diff_pos, axis=1, keepdims=True) + 1e-12)
            d_neg = jnp.sqrt(jnp.sum(diff_neg * diff_neg, axis=1, keepdims=True) + 1e-12)

            loss = jnp.maximum(0.0, d_pos - d_neg + self.margin)
            o_ref[:, :] = loss

        block_m = 128
        block = (block_m, D)
        grid = (B // block_m,)

        losses = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, 1), anchor.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec(block, lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(anchor, positive, negative)

        return jnp.mean(losses)


batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    scale = jax.random.uniform(key1)
    anchor = jax.random.uniform(key2, (batch_size,) + input_shape) * scale
    positive = jax.random.uniform(key3, (batch_size,) + input_shape)
    negative = jax.random.uniform(key4, (batch_size,) + input_shape)

    return [anchor, positive, negative]

def get_init_inputs():
    return [1.0]
