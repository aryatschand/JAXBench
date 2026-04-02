import jax
import jax.numpy as jnp
from jax.nn import gelu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, max_dim):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.max_dim = max_dim

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight) + self.bias
        B, O = x.shape

        block_m = 128
        block_k = 128
        grid_m = B // block_m
        num_k_tiles = O // block_k

        def kernel(x_ref, o_ref):
            row_max = jnp.full((block_m, 1), -jnp.inf, dtype=x_ref.dtype)
            row_sum = jnp.zeros((block_m, 1), dtype=x_ref.dtype)

            def body(k, carry):
                row_max, row_sum = carry
                x_block = x_ref[:, k * block_k:(k + 1) * block_k]

                block_max = jnp.max(x_block, axis=1, keepdims=True)
                block_sum = jnp.sum(x_block, axis=1, keepdims=True)

                row_max = jnp.maximum(row_max, block_max)
                row_sum = row_sum + block_sum
                return row_max, row_sum

            row_max, row_sum = jax.lax.fori_loop(0, num_k_tiles, body, (row_max, row_sum))
            row_mean = row_sum / O
            out = gelu(row_max - row_mean)

            o_ref[:, :] = out

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[pl.BlockSpec((block_m, block_k), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(x)

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, max_dim]
