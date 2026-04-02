import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            arr = jnp.array(value)
            if name == "bias":
                arr = arr.reshape(1, -1)
            setattr(self, name.replace('.', '_'), arr)

    def forward(self, x):
        def kernel(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[...]                      # (B, K)
            w_full = w_ref[...]                       # (K, N)
            b_full = b_ref[...]                       # (1, N)

            y = jnp.matmul(x_block, w_full) + b_full
            y = y * self.multiplier
            y = jnp.where(y >= 0, y, y * self.negative_slope)

            o_ref[...] = y

        B, K = x.shape
        N = self.weight.shape[1]

        block_b = 128
        grid_b = B // block_b

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_b,),
                in_specs=[
                    pl.BlockSpec((block_b, K), lambda i: (i, 0)),
                    pl.BlockSpec((K, N), lambda i: (0, 0)),
                    pl.BlockSpec((1, N), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_b, N), lambda i: (i, 0)),
            ),
        )(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
