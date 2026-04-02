import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        eps = self.eps

        def kernel(x_ref, o_ref):
            x_block = x_ref[...]
            rms = jnp.sqrt(jnp.mean(x_block * x_block, axis=1, keepdims=True) + eps)
            o_ref[...] = x_block / rms

        block = (8, self.num_features, 128, 128)
        grid = (
            x.shape[0] // block[0],
            x.shape[2] // block[2],
            x.shape[3] // block[3],
        )

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j, k: (i, 0, j, k)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j, k: (i, 0, j, k)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
