import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, bias_ref, add_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    add = add_ref[...]

    y = x + bias + add
    hardswish = y * jnp.minimum(jnp.maximum(y + 3.0, 0.0), 6.0) / 6.0
    out = y * hardswish

    o_ref[...] = out


def fused_pallas(x, bias, add):
    # reshape to 2D
    orig_shape = x.shape
    x2 = x.reshape(x.shape[0] * x.shape[1], -1)
    add2 = add.reshape(x2.shape)

    # broadcast bias to match x
    bias_full = jnp.broadcast_to(bias, orig_shape).reshape(x2.shape)

    M, N = x2.shape
    bm = min(128, M)
    bn = min(128, N)

    grid = (M // bm, N // bn)

    out = pl.pallas_call(
        fused_kernel,
        out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        ),
    )(x2, bias_full, add2)

    return out.reshape(orig_shape)


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        self.conv_transpose_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x, add_input):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))

        pad_d = self.kernel_size - 1 - self.padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding

        padding = (
            (pad_d, pad_d + self.output_padding),
            (pad_h, pad_h + self.output_padding),
            (pad_w, pad_w + self.output_padding),
        )

        x = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=("NDHWC", "DHWOI", "NDHWC"),
        )

        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        bias = self.conv_transpose_bias.reshape(1, -1, 1, 1, 1)

        x = fused_pallas(x, bias, add_input)

        return x


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    return [
        jax.random.uniform(key1, (batch_size, in_channels, D, H, W)),
        jax.random.uniform(key2, (batch_size, out_channels, D * stride, H * stride, W * stride)),
    ]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
