import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kernel_fn(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[:, :]            # (bm, K)
    w = w_ref[:, :]            # (K, bn)
    b = b_ref[:]               # (bn,)

    acc = jnp.matmul(x, w, preferred_element_type=jnp.float32)
    acc = acc + b
    acc = jax.nn.relu(acc)
    acc = acc / 2.0

    o_ref[:, :] = acc


class Model:
    def __init__(self, in_features, out_features, divisor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        M, K = x.shape
        _, N = self.weight.shape

        bm = 128
        bn = 128

        grid = (M // bm, N // bn)

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),   # x
                    pl.BlockSpec((K, bn), lambda i, j: (0, j)),   # w
                    pl.BlockSpec((bn,), lambda i, j: (j,)),       # b
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias)


batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, divisor]
