import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, b_ref, o_ref):
    x = x_ref[:, :]
    b = b_ref[:, :]
    # broadcast bias along columns
    b_full = b + jnp.zeros_like(x)
    o_ref[:, :] = 2.0 * x * x + b_full * x + x


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

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))

        padding = ((1, 1), (1, 1), (1, 1))

        x_conv = jax.lax.conv_transpose(
            x_ndhwc,
            kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        if self.output_padding > 0:
            x_conv = jnp.pad(
                x_conv,
                ((0, 0),
                 (0, self.output_padding),
                 (0, self.output_padding),
                 (0, self.output_padding),
                 (0, 0))
            )

        x_conv = x_conv + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x_conv, (0, 4, 1, 2, 3))

        N, C, D, H, W = x.shape
        x_flat = jnp.reshape(x, (N * C, D * H * W))

        # prepare bias per (N*C, 1)
        bias_c = self.bias.reshape(C)
        bias_nc = jnp.tile(bias_c, (N,))
        bias_flat = bias_nc.reshape(N * C, 1)

        block_m = min(128, x_flat.shape[0])
        block_n = min(128, x_flat.shape[1])

        grid = (
            x_flat.shape[0] // block_m,
            x_flat.shape[1] // block_n,
        )

        out_flat = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_flat, bias_flat)

        x_out = jnp.reshape(out_flat, (N, C, D, H, W))
        return x_out


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(16, 32, 16, 32, 32))]


def get_init_inputs():
    return [32, 64, 3, 2, 1, 1, (64, 1, 1, 1)]
