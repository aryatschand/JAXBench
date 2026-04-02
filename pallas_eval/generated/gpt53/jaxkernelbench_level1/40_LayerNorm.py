import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def layernorm_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]  # (1, F, D1, D2)
    w = w_ref[...]  # (F, D1, D2)
    b = b_ref[...]  # (F, D1, D2)

    # Expand weight/bias to match batch block
    w = pltpu.repeat(w, 0, 1)  # (1, F, D1, D2)
    b = pltpu.repeat(b, 0, 1)

    mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    var = jnp.mean((x - mean) * (x - mean), axis=(1, 2, 3), keepdims=True)

    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    o_ref[...] = x_norm * w + b


class Model:
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        self.normalized_shape = normalized_shape
        self.weight = jnp.ones(normalized_shape)
        self.bias = jnp.zeros(normalized_shape)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, F, D1, D2 = x.shape

        return pl.pallas_call(
            layernorm_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(B,),
                in_specs=[
                    pl.BlockSpec((1, F, D1, D2), lambda i: (i, 0, 0, 0)),
                    pl.BlockSpec((F, D1, D2), lambda i: (0, 0, 0)),
                    pl.BlockSpec((F, D1, D2), lambda i: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, F, D1, D2), lambda i: (i, 0, 0, 0)),
            ),
        )(x, self.weight, self.bias)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
