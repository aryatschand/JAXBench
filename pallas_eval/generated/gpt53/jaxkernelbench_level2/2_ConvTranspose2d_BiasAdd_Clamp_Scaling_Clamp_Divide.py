import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_post_kernel(x_ref, bias1_ref, bias2_ref, o_ref, scaling_factor):
    x = x_ref[...]

    block_m, block_n = x.shape

    pid_m = pl.program_id(axis=0)
    pid_n = pl.program_id(axis=1)

    row_start = pid_m * block_m
    rows = row_start + jnp.arange(block_m)

    C = bias1_ref.shape[0]
    channel_idx = rows % C

    b1 = bias1_ref[channel_idx]
    b2 = bias2_ref[channel_idx, 0]

    b1 = b1[:, None]
    b2 = b2[:, None]

    x = x + b1 + b2
    x = jnp.clip(x, 0.0, 1.0)
    x = x * scaling_factor
    x = jnp.clip(x, 0.0, 1.0)
    x = x / scaling_factor

    o_ref[...] = x


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        k = self._kernel_size
        pad_h = k - 1 - self.padding
        pad_w = k - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding), (pad_w, pad_w + self.output_padding))

        x = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape
        x2d = x.reshape(N * C, H * W)

        block_m = 128
        block_n = 128

        grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

        out = pl.pallas_call(
            fused_post_kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((C,), lambda i, j: (0,)),
                    pl.BlockSpec((C, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2d, self.conv_transpose_bias, self.bias, self.scaling_factor)

        out = out.reshape(N, C, H, W)
        return out


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
