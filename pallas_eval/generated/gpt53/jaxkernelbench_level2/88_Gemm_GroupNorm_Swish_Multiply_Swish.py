import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, gn_w_ref, gn_b_ref, mul_w_ref, o_ref):
    x = x_ref[...]  # (B, G)

    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.var(x, axis=1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)

    gn_w = gn_w_ref[...]
    gn_b = gn_b_ref[...]

    x = x * gn_w + gn_b

    x = x * jax.nn.sigmoid(x)
    x = x * mul_w_ref[...]
    x = x * jax.nn.sigmoid(x)

    o_ref[...] = x


class Model:
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        self.gemm_weight = jnp.zeros((out_features, in_features))
        self.gemm_bias = jnp.zeros((out_features,))
        self.group_norm_weight = jnp.ones((out_features,))
        self.group_norm_bias = jnp.zeros((out_features,))
        self.num_groups = num_groups
        self.out_features = out_features
        self.multiply_weight = jnp.zeros(multiply_weight_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.gemm_weight.T) + self.gemm_bias

        batch_size = x.shape[0]
        group_size = self.out_features // self.num_groups

        x = x.reshape(batch_size, self.num_groups, group_size)

        block_b = 8
        block_g = group_size

        grid = (batch_size // block_b, self.num_groups)

        def gn_w_map(i, j):
            return (j * group_size,)

        def gn_b_map(i, j):
            return (j * group_size,)

        def mul_w_map(i, j):
            return (j * group_size,)

        x = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_b, block_g), lambda i, j: (i, j)),
                    pl.BlockSpec((block_g,), gn_w_map),
                    pl.BlockSpec((block_g,), gn_b_map),
                    pl.BlockSpec((block_g,), mul_w_map),
                ],
                out_specs=pl.BlockSpec((block_b, block_g), lambda i, j: (i, j)),
            ),
        )(x, self.group_norm_weight, self.group_norm_bias, self.multiply_weight)

        x = x.reshape(batch_size, self.out_features)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
