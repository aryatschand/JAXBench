import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, num_features: int, num_groups: int):
        self.num_groups = num_groups
        self.num_features = num_features
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        input_shape = x.shape
        N, C, H, W = input_shape
        G = self.num_groups

        x_reshaped = x.reshape((N * G, (C // G) * H * W))

        def kernel_fn(x_ref, o_ref):
            x_block = x_ref[...]
            mean = jnp.mean(x_block, axis=1, keepdims=True)
            var = jnp.mean((x_block - mean) ** 2, axis=1, keepdims=True)
            out = (x_block - mean) / jnp.sqrt(var + self.eps)
            o_ref[...] = out

        M, K = x_reshaped.shape

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((M, K), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(M,),
                in_specs=[pl.BlockSpec((1, K), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((1, K), lambda i: (i, 0)),
            ),
        )(x_reshaped)

        out = out.reshape(input_shape)

        return out * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)


batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features, num_groups]
