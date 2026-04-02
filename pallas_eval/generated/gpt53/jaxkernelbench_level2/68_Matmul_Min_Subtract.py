import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, w_ref, b_ref, o_ref, constant):
    x_block = x_ref[:, :]
    w_block = w_ref[:, :]
    b_block = b_ref[:, :]

    acc = jnp.dot(x_block, w_block)
    acc = acc + b_block
    acc = jnp.minimum(acc, constant)
    acc = acc - constant

    o_ref[:, :] = acc


class Model:
    def __init__(self, in_features, out_features, constant):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))
        self.constant = jnp.array(constant)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            arr = jnp.array(value)
            if name == "bias":
                arr = arr.reshape(1, -1)
            setattr(self, name.replace('.', '_'), arr)

    def forward(self, x):
        M, K = x.shape
        K2, N = self.weight.shape

        bm = 128
        bn = 128

        grid = (M // bm, N // bn)

        return pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, bn), lambda i, j: (0, j)),
                    pl.BlockSpec((1, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias, self.constant)


batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, constant]
