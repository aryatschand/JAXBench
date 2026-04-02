import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import gelu


def fused_kernel(x_ref, w_ref, b_ref, o_ref, pool_kernel_size, scale_factor):
    x = x_ref[...]                    # (bm, in_features)
    w = w_ref[...]                    # (in_features, out_features)
    b = b_ref[...]                    # (out_features,)

    # Matmul + bias
    y = jnp.dot(x, w) + b

    # AvgPool along feature dim
    B, F = y.shape
    K = pool_kernel_size
    new_F = F // K
    y = jnp.reshape(y, (B, new_F, K))
    y = jnp.sum(y, axis=2) / K

    # GELU + scale
    y = gelu(y)
    y = y * scale_factor

    # Max over features
    y = jnp.max(y, axis=1)

    # write as (bm, 1)
    o_ref[...] = y[:, None]


class Model:
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B = x.shape[0]
        bm = 128
        grid = (B // bm, 1)

        return pl.pallas_call(
            lambda x_ref, w_ref, b_ref, o_ref: fused_kernel(
                x_ref, w_ref, b_ref, o_ref,
                self.pool_kernel_size,
                self.scale_factor
            ),
            out_shape=jax.ShapeDtypeStruct((B, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, x.shape[1]), lambda i, j: (i, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i, j: (0, 0)),
                    pl.BlockSpec((self.bias.shape[0],), lambda i, j: (0,)),
                ],
                out_specs=pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
            ),
        )(x, self.weight, self.bias).squeeze(-1)


batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
