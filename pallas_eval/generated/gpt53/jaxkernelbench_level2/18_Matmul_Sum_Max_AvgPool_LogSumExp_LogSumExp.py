import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, wsum_ref, bsum_ref, o_ref):
    x = x_ref[...]            # (bm, K)
    w = wsum_ref[...]         # (K, 1)
    b = bsum_ref[...]         # (1, 1)

    # broadcast multiply then reduce
    y = jnp.sum(x * w.T, axis=1, keepdims=True) + b
    o_ref[...] = y


class Model:
    def __init__(self, in_features, out_features):
        self.linear_weight = jnp.zeros((out_features, in_features))
        self.linear_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Precompute fused weights
        wsum = jnp.sum(self.linear_weight, axis=0).reshape(-1, 1)  # (in_features, 1)
        bsum = jnp.sum(self.linear_bias).reshape(1, 1)             # (1, 1)

        B, K = x.shape

        bm = 128
        block = (bm, K)
        grid = (B // bm,)

        return pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((B, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec((K, 1), lambda i: (0, 0)),
                    pl.BlockSpec((1, 1), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((bm, 1), lambda i: (i, 0)),
            ),
        )(x, wsum, bsum)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features]
