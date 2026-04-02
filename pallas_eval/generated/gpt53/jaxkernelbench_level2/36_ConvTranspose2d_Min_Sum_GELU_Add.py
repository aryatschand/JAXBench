"""
JAXBench Level 2 - ConvTranspose2d_Min_Sum_GELU_Add
Pallas TPU optimized version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def post_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]  # (1, C, H, BW)

    # Min over channel axis
    x = jnp.min(x, axis=1, keepdims=True)  # (1,1,H,BW)

    # Sum over height
    x = jnp.sum(x, axis=2, keepdims=True)  # (1,1,1,BW)

    # GELU
    x = 0.5 * x * (1.0 + jnp.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    # Add bias
    x = x + bias_ref[...]

    o_ref[...] = x


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        pad = kernel.shape[0] - 1 - self.padding
        padding = ((pad, pad), (pad, pad))

        out = jax.lax.conv_transpose(
            x_nhwc,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        if self.output_padding > 0:
            out = jax.lax.pad(
                out,
                padding_value=0.0,
                padding_config=((0, 0, 0), (0, self.output_padding, 0),
                                (0, self.output_padding, 0), (0, 0, 0))
            )

        out = jnp.transpose(out, (0, 3, 1, 2))  # (N,C,H,W)

        N, C, H, W = out.shape

        BW = 128
        assert W % BW == 0

        result = pl.pallas_call(
            post_kernel,
            out_shape=jax.ShapeDtypeStruct((N, 1, 1, W), out.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N, W // BW),
                in_specs=[
                    pl.BlockSpec((1, C, H, BW), lambda i, j: (i, 0, 0, j)),
                    pl.BlockSpec((1, 1, 1, 1), lambda i, j: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, 1, BW), lambda i, j: (i, 0, 0, j)),
            ),
        )(out, self.bias)

        return result


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(16, 64, 128, 128))]


def get_init_inputs():
    return [64, 128, 3, 2, 1, 1, (1, 1, 1)]
