import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            arr = jnp.array(value)
            if name == "bias":
                arr = arr.reshape(1, -1)
            setattr(self, name.replace('.', '_'), arr)

    def forward(self, x):
        def kernel(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[...]          # (BM, K)
            w_block = w_ref[...]          # (K, BN)
            b_block = b_ref[...]          # (1, BN)

            acc = jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)
            acc = acc + b_block
            acc = acc - self.subtract_value
            acc = acc * self.multiply_value
            acc = jnp.maximum(acc, 0.0)

            o_ref[...] = acc

        M, K = x.shape
        K2, N = self.weight.shape

        BM = 128
        BN = 128

        grid = (M // BM, N // BN)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias)

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
