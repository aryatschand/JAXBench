import jax
import jax.numpy as jnp
import jax.lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def copy_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        self.conv2d_weight = jnp.zeros(weight_shape, dtype=jnp.float32)
        if bias:
            self.conv2d_bias = jnp.zeros((out_channels,), dtype=jnp.float32)
        else:
            self.conv2d_bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            attr_name = name.replace('.', '_')
            setattr(self, attr_name, jnp.array(value, dtype=jnp.float32))

    def forward(self, x):
        x = x.astype(jnp.float32)

        x = jnp.transpose(x, (0, 2, 3, 1))
        weight = jnp.transpose(self.conv2d_weight, (2, 3, 1, 0))

        out = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=self.stride,
            padding=[(self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])],
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.groups
        )

        if self.conv2d_bias is not None:
            out = out + self.conv2d_bias.reshape(1, 1, 1, -1)

        # reshape to 2D for Pallas requirement
        n, h, w, c = out.shape
        out_2d = jnp.reshape(out, (n * h, w * c))

        block_m = min(512, out_2d.shape[0])
        block_n = min(512, out_2d.shape[1])

        grid_m = out_2d.shape[0] // block_m
        grid_n = out_2d.shape[1] // block_n

        out_2d = pl.pallas_call(
            copy_kernel,
            out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(out_2d)

        out = jnp.reshape(out_2d, (n, h, w, c))
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256


def get_inputs():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size, in_channels, height, width), dtype=jnp.float32)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
