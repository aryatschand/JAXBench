import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, H, W = x.shape
        HW = H * W

        x_2d = x.reshape(N * C, HW)

        def kernel_fn(x_ref, o_ref):
            row = x_ref[...]  # (1, HW)
            mean = jnp.mean(row, axis=1, keepdims=True)
            var = jnp.var(row, axis=1, keepdims=True)
            out = (row - mean) / jnp.sqrt(var + self.eps)
            o_ref[...] = out

        block = (1, HW)
        grid = (N * C,)

        out_2d = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i: (i, 0))],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x_2d)

        return out_2d.reshape(N, C, H, W)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
