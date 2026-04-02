import jax
import jax.numpy as jnp
from jax.nn import gelu, softmax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 128
BN = 128

def kernel_fn(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]          # (BM, K)
    w = w_ref[...]          # (K, BN)
    b = b_ref[...]          # (1, BN)

    acc = jnp.dot(x, w, preferred_element_type=jnp.float32)
    acc = acc + b

    acc = gelu(acc)

    row_max = jnp.max(acc, axis=1, keepdims=True)
    exp = jnp.exp(acc - row_max)
    row_sum = jnp.sum(exp, axis=1, keepdims=True)
    out = exp / row_sum

    o_ref[...] = out

class Model:
    def __init__(self, in_features, out_features):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            arr = jnp.array(value)
            if name == "bias":
                arr = arr.reshape(1, -1)
            setattr(self, name.replace('.', '_'), arr)

    def forward(self, x):
        M, K = x.shape
        _, N = self.weight.shape

        grid = (M // BM, N // BN)

        return pl.pallas_call(
            kernel_fn,
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

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
