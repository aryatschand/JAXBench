"""
JAXBench Level 2 - Matmul_GroupNorm_LeakyReLU_Sum
Pallas TPU optimized version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, gn_weight_ref, gn_bias_ref, o_ref, num_groups, eps, negative_slope):
    x = x_ref[0, :]  # (C,)

    C = x.shape[0]
    group_size = C // num_groups

    xg = x.reshape((num_groups, group_size))
    mean = jnp.mean(xg, axis=1, keepdims=True)
    var = jnp.var(xg, axis=1, keepdims=True)
    xg = (xg - mean) / jnp.sqrt(var + eps)
    x = xg.reshape((C,))

    x = x * gn_weight_ref[0, :] + gn_bias_ref[0, :]

    x = jnp.where(x > 0, x, x * negative_slope)

    x = x + x

    o_ref[0, :] = x


class Model:
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        self.fc_weight = jnp.zeros((hidden_size, input_size))
        self.fc_bias = jnp.zeros(hidden_size)

        self.num_groups = num_groups
        self.num_channels = hidden_size
        self.eps = eps
        self.gn_weight = jnp.ones(hidden_size)
        self.gn_bias = jnp.zeros(hidden_size)

        self.negative_slope = negative_slope

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.fc_weight.T) + self.fc_bias  # (N, C)

        N, C = x.shape

        # reshape params to 2D for Pallas
        gn_weight = self.gn_weight.reshape(1, C)
        gn_bias = self.gn_bias.reshape(1, C)

        def kernel(x_ref, gn_w_ref, gn_b_ref, o_ref):
            fused_kernel(
                x_ref,
                gn_w_ref,
                gn_b_ref,
                o_ref,
                self.num_groups,
                self.eps,
                self.negative_slope,
            )

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N,),
                in_specs=[
                    pl.BlockSpec((1, C), lambda i: (i, 0)),
                    pl.BlockSpec((1, C), lambda i: (0, 0)),
                    pl.BlockSpec((1, C), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((1, C), lambda i: (i, 0)),
            ),
        )(x, gn_weight, gn_bias)


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]


def get_init_inputs():
    return [input_size, hidden_size, num_groups]
