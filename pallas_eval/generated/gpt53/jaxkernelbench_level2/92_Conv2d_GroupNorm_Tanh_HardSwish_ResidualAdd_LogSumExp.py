import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(xc_ref, xn_ref, o_ref):
    xc = xc_ref[0, :]
    xn = xn_ref[0, :]

    x_tanh = jnp.tanh(xn)
    x_hs = x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3.0, 0.0), 6.0) / 6.0
    x_res = xc + x_hs

    m = jnp.max(x_res)
    lse = m + jnp.log(jnp.sum(jnp.exp(x_res - m)))

    o_ref[0, 0] = lse


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        self.groups = groups
        self.eps = eps
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        self.group_norm_weight = jnp.ones(out_channels)
        self.group_norm_bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.conv_weight, (2, 3, 1, 0))
        x_conv = jax.lax.conv_general_dilated(
            x_nhwc,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        x_conv = x_conv + self.conv_bias.reshape(1, 1, 1, -1)
        x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))

        N, C, H, W = x_conv.shape

        xg = x_conv.reshape(N, self.groups, C // self.groups, H, W)
        mean = jnp.mean(xg, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(xg, axis=(2, 3, 4), keepdims=True)
        xg = (xg - mean) / jnp.sqrt(var + self.eps)
        xg = xg.reshape(N, C, H, W)
        x_norm = xg * self.group_norm_weight.reshape(1, -1, 1, 1) + \
                 self.group_norm_bias.reshape(1, -1, 1, 1)

        NHW = N * H * W
        xc_flat = jnp.transpose(x_conv, (0, 2, 3, 1)).reshape(NHW, C)
        xn_flat = jnp.transpose(x_norm, (0, 2, 3, 1)).reshape(NHW, C)

        out = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((NHW, 1), xc_flat.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(NHW,),
                in_specs=[
                    pl.BlockSpec((1, C), lambda i: (i, 0)),
                    pl.BlockSpec((1, C), lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
            ),
        )(xc_flat, xn_flat)

        out = out.reshape(N, H, W, 1)
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
