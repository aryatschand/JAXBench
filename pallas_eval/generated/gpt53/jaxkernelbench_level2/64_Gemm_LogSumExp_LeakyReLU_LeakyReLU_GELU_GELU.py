import jax
import jax.numpy as jnp
from jax.nn import gelu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def post_kernel(x_ref, o_ref):
    x = x_ref[...]

    # LeakyReLU twice
    x = jnp.where(x > 0, x, 0.01 * x)
    x = jnp.where(x > 0, x, 0.01 * x)

    # GELU twice
    x = gelu(x)
    x = gelu(x)

    o_ref[...] = x

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Gemm
        x = jnp.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias

        # LogSumExp (batch, 1)
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)

        batch = x.shape[0]
        block = (128, 1)
        grid = (batch // 128, 1)

        x = pl.pallas_call(
            post_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x)

        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
