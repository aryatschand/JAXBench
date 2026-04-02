import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, gw_ref, gb_ref, bias_ref, o_ref):
    x = x_ref[...]  # (B, C)
    gw = gw_ref[:, 0]  # (C,)
    gb = gb_ref[:, 0]  # (C,)
    bias = bias_ref[:, 0]  # (C,)

    B, C = x.shape
    G = 512
    group_size = C // G

    xg = x.reshape(B, G, group_size)

    mean = jnp.mean(xg, axis=2, keepdims=True)
    var = jnp.var(xg, axis=2, keepdims=True)

    xg = (xg - mean) / jnp.sqrt(var + 1e-5)
    x = xg.reshape(B, C)

    x = x * gw + gb

    min_vals = jnp.min(x, axis=1, keepdims=True)  # (B, 1)

    out = min_vals + bias  # broadcast to (B, C)

    o_ref[...] = out


class Model:
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        self.weight = jnp.zeros((out_features, in_features))
        self.bias_linear = jnp.zeros(out_features)

        self.num_groups = num_groups
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)
        self.eps = 1e-5

        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight.T) + self.bias_linear  # (N, C)

        N, C = x.shape
        block = (128, C)
        grid = (N // 128,)

        gw = self.group_norm_weight.reshape(C, 1)
        gb = self.group_norm_bias.reshape(C, 1)
        bias = self.bias.reshape(C, 1)

        y = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((N, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x, gw, gb, bias)

        y = y.T.reshape(1, C, N, 1)
        return y


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
