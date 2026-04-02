import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from typing import Tuple

def fused_kernel(x_ref, mean_ref, var_ref, mult_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]  # (B, C)
    mean = mean_ref[...]  # (B, C)
    var = var_ref[...]  # (B, C)
    mult = mult_ref[...]  # (1, C)
    w = w_ref[...]  # (1, C)
    b = b_ref[...]  # (1, C)

    x = x * mult
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    x = x * w + b
    x = jnp.clip(x, -1.0, 1.0)
    x = x * mult
    out = jnp.max(x, axis=1, keepdims=True)

    o_ref[...] = out

def pallas_fused(x, mean, var, mult, w, b):
    B, C = x.shape
    block = (min(B, 128), C)
    grid = (B // block[0],)

    return pl.pallas_call(
        fused_kernel,
        out_shape=jax.ShapeDtypeStruct((B, 1), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i: (i, 0)),
                pl.BlockSpec(block, lambda i: (i, 0)),
                pl.BlockSpec(block, lambda i: (i, 0)),
                pl.BlockSpec((1, C), lambda i: (0, 0)),
                pl.BlockSpec((1, C), lambda i: (0, 0)),
                pl.BlockSpec((1, C), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((block[0], 1), lambda i: (i, 0)),
        ),
    )(x, mean, var, mult, w, b)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)
        self.instance_norm_weight = jnp.ones(out_channels)
        self.instance_norm_bias = jnp.zeros(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        N, C, D, H, W = x.shape
        x_reshaped = jnp.transpose(x, (0, 2, 3, 4, 1)).reshape(-1, C)

        mean = jnp.mean(x, axis=(2,3,4), keepdims=True)
        var = jnp.var(x, axis=(2,3,4), keepdims=True)

        mean = jnp.transpose(mean, (0, 2, 3, 4, 1)).reshape(-1, C)
        var = jnp.transpose(var, (0, 2, 3, 4, 1)).reshape(-1, C)

        mult = self.multiplier.reshape(1, C)
        w = self.instance_norm_weight.reshape(1, C)
        b = self.instance_norm_bias.reshape(1, C)

        out = pallas_fused(x_reshaped, mean, var, mult, w, b)
        out = out.reshape(N, D, H, W)

        return out

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
