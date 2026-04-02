"""
JAXBench Level 2 - Gemm_Scaling_Hardtanh_GELU
Pallas TPU optimized version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, w_ref, b_ref, o_ref, scaling_factor, ht_min, ht_max):
    x = x_ref[...]          # (bm, K)
    w = w_ref[...]          # (K, bn)
    b = b_ref[...]          # (bn,)

    acc = jnp.matmul(x, w) + b

    x = acc * scaling_factor
    x = jnp.clip(x, ht_min, ht_max)

    gelu = 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))
    o_ref[...] = x * gelu


class Model:
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))  # make 2D
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == "bias":
                value = jnp.array(value)[None, :]
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]

        B, K = x.shape
        _, N = self.weight.shape

        bm = 128
        bn = 128

        grid = (B // bm, N // bn)

        return pl.pallas_call(
            lambda x_ref, w_ref, b_ref, o_ref: fused_kernel(
                x_ref, w_ref, b_ref, o_ref,
                self.scaling_factor,
                self.hardtanh_min,
                self.hardtanh_max,
            ),
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, bn), lambda i, j: (0, j)),
                    pl.BlockSpec((1, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias)


batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
