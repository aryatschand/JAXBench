import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kernel_fn(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[:, :]            # (bm, K)
    w = w_ref[:, :]            # (K, N)
    b = b_ref[:, :]            # (1, N)

    y = jnp.matmul(x, w) + b   # (bm, N)
    y = 1.0 / (1.0 + jnp.exp(-y))
    y = jnp.sum(y, axis=1, keepdims=True)

    o_ref[:, :] = y


class Model:
    def __init__(self, input_size, hidden_size):
        self.weight = jnp.zeros((input_size, hidden_size))
        self.bias = jnp.zeros(hidden_size)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        batch_size, input_size = x.shape
        hidden_size = self.weight.shape[1]

        # reshape bias to 2D
        bias_2d = self.bias.reshape(1, hidden_size)

        bm = min(batch_size, 128)

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch_size // bm,),
                in_specs=[
                    pl.BlockSpec((bm, input_size), lambda i: (i, 0)),
                    pl.BlockSpec((input_size, hidden_size), lambda i: (0, 0)),
                    pl.BlockSpec((1, hidden_size), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((bm, 1), lambda i: (i, 0)),
            ),
        )(x, self.weight, bias_2d)


batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]


def get_init_inputs():
    return [input_size, hidden_size]
