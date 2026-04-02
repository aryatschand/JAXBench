import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(x_ref, w_ref, o_ref):
    x = x_ref[:, :]
    w = w_ref[:, :]
    o_ref[:, :] = jnp.dot(x, w)


def pallas_matmul(x, w):
    M, K = x.shape
    K2, N = w.shape
    assert K == K2

    bm = 128
    bn = 128

    grid = (M // bm, N // bn)

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                pl.BlockSpec((K, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        ),
    )(x, w)


class Model:
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)
        self.num_groups = num_groups
        self.out_features = out_features

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = pallas_matmul(x, self.weight)

        x = jax.nn.sigmoid(x) * x
        x = x + self.bias

        group_size = self.out_features // self.num_groups
        x = x.reshape(-1, self.num_groups, group_size)

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)

        x = x.reshape(-1, self.out_features)
        x = x * self.group_norm_weight + self.group_norm_bias

        return x


batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
