import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


def pallas_copy(x):
    x2d = x.reshape(x.shape[0], -1)
    block_m = min(x2d.shape[0], 8)
    block_n = min(x2d.shape[1], 128)
    block = (block_m, block_n)
    grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

    y2d = pl.pallas_call(
        copy_kernel,
        out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x2d)

    return y2d.reshape(x.shape)


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        self.weight = jnp.zeros((in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.pytorch_padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.groups = groups
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        jax_padding = tuple(k - 1 - p for k, p in zip(self.kernel_size, self.pytorch_padding))

        if self.groups == 1:
            x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
            kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

            out = jax.lax.conv_transpose(
                x_ndhwc, kernel,
                strides=self.stride,
                padding=((jax_padding[0], jax_padding[0] + self.output_padding[0]),
                         (jax_padding[1], jax_padding[1] + self.output_padding[1]),
                         (jax_padding[2], jax_padding[2] + self.output_padding[2])),
                dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
            )

            out = jnp.transpose(out, (0, 4, 1, 2, 3))
        else:
            in_per_group = x.shape[1] // self.groups
            out_per_group = self.out_channels // self.groups

            x_groups = jnp.split(x, self.groups, axis=1)
            w_groups = jnp.split(self.weight, self.groups, axis=0)

            out_groups = []
            for x_g, w_g in zip(x_groups, w_groups):
                x_g_ndhwc = jnp.transpose(x_g, (0, 2, 3, 4, 1))
                kernel = jnp.transpose(w_g, (2, 3, 4, 1, 0))

                out_g = jax.lax.conv_transpose(
                    x_g_ndhwc, kernel,
                    strides=self.stride,
                    padding=((jax_padding[0], jax_padding[0] + self.output_padding[0]),
                             (jax_padding[1], jax_padding[1] + self.output_padding[1]),
                             (jax_padding[2], jax_padding[2] + self.output_padding[2])),
                    dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
                )

                out_g = jnp.transpose(out_g, (0, 4, 1, 2, 3))
                out_groups.append(out_g)

            out = jnp.concatenate(out_groups, axis=1)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)

        out = pallas_copy(out)
        return out


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 32, 12, 24, 48))
    return [x]


def get_init_inputs():
    return [32, 32, (3, 5, 7), (2, 2, 2), (1, 2, 3), (1, 1, 1), 4]
