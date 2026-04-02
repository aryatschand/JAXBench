import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref, eps):
    x = x_ref[:, :]
    mean = mean_ref[:, :]
    var = var_ref[:, :]
    gamma = gamma_ref[:, :]
    beta = beta_ref[:, :]

    y = (x - mean) / jnp.sqrt(var + eps)
    y = y * gamma + beta
    y = jax.nn.gelu(y)
    y = jax.nn.relu(y)

    o_ref[:, :] = y


class Model:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = 1e-5

        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        self.gemm_weight = jax.random.normal(key1, (out_features, in_features))
        self.gemm_bias = jax.random.normal(key2, (out_features,))

        self.batch_norm_weight = jnp.ones((out_features,))
        self.batch_norm_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.gemm_weight.T) + self.gemm_bias

        mean = jnp.mean(x, axis=0, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)

        gamma = self.batch_norm_weight.reshape(1, -1)
        beta = self.batch_norm_bias.reshape(1, -1)

        B, C = x.shape
        block_m = 128
        block_n = 128

        grid = (B // block_m, C // block_n)

        return pl.pallas_call(
            lambda x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref: fused_kernel(
                x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref, self.eps
            ),
            out_shape=jax.ShapeDtypeStruct((B, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x, mean, var, gamma, beta)


batch_size = 16384
in_features = 4096
out_features = 4096


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_features))
    return [x]


def get_init_inputs():
    return [in_features, out_features]
