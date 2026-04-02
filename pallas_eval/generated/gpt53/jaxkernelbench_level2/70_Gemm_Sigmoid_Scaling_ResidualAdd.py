import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, input_size, hidden_size, scaling_factor):
        self.weight = jnp.zeros((input_size, hidden_size))
        self.bias = jnp.zeros(hidden_size)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, I = x.shape
        H = self.weight.shape[1]

        bm = 128
        bn = 128

        assert B % bm == 0
        assert H % bn == 0

        bias_2d = self.bias.reshape(1, H)

        def kernel(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[:, :]
            w_block = w_ref[:, :]
            b_block = b_ref[0, :]

            acc = jnp.dot(x_block, w_block)
            acc = acc + b_block

            original = acc
            acc = jax.nn.sigmoid(acc)
            acc = acc * self.scaling_factor
            acc = acc + original

            o_ref[:, :] = acc

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, H), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(B // bm, H // bn),
                in_specs=[
                    pl.BlockSpec((bm, I), lambda i, j: (i, 0)),
                    pl.BlockSpec((I, bn), lambda i, j: (0, j)),
                    pl.BlockSpec((1, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias_2d)

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
