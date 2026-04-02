import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, K = x.shape
        N = self.weight.shape[1]

        bm = 128
        bn = 128
        bk = 128

        grid = (B // bm, N // bn, K // bk)

        def kernel(x_ref, w_ref, b_ref, o_ref):
            k_id = pl.program_id(axis=2)

            x_block = x_ref[:, :]
            w_block = w_ref[:, :]

            partial = jnp.matmul(x_block, w_block, preferred_element_type=jnp.float32)

            def init():
                o_ref[...] = partial

            def accumulate():
                o_ref[...] = o_ref[...] + partial

            pl.when(k_id == 0, init)
            pl.when(k_id != 0, accumulate)

            def finalize():
                out = o_ref[...]

                bias = b_ref[0, :]
                out = out + bias

                out = out * jnp.tanh(jax.nn.softplus(out))
                out = out * jnp.tanh(jax.nn.softplus(out))

                o_ref[...] = out

            pl.when(k_id == (K // bk) - 1, finalize)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                    pl.BlockSpec((1, bn), lambda i, j, k: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            ),
        )(x, self.weight, self.bias.reshape(1, -1))


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
