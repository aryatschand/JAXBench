import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_elemwise_kernel(x_ref, m_ref, o_ref):
    x = x_ref[...]
    m = m_ref[...]
    y = jnp.where(x > 0, x, x * 0.2)
    y = y * m
    y = jnp.where(y > 0, y, y * 0.2)
    o_ref[...] = y


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        kernel_size = self.weight.shape[2]
        pad_amount = kernel_size - 1 - self.padding
        padding = (
            (pad_amount, pad_amount + self.output_padding),
            (pad_amount, pad_amount + self.output_padding),
            (pad_amount, pad_amount + self.output_padding),
        )

        out = jax.lax.conv_transpose(
            x_ndhwc,
            kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
        )

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)

        out = jnp.transpose(out, (0, 4, 1, 2, 3))  # NCDHW

        # reshape to 2D for Pallas (N*D*H*W, C)
        n, c, d, h, w = out.shape
        out_2d = jnp.reshape(jnp.transpose(out, (0, 2, 3, 4, 1)), (-1, c))

        m = self.multiplier.reshape((1, c))

        block_m = min(out_2d.shape[0], 512)
        block_n = min(c, 128)

        grid = (out_2d.shape[0] // block_m, c // block_n)

        out_2d = pl.pallas_call(
            fused_elemwise_kernel,
            out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(out_2d, m)

        # reshape back
        out = jnp.reshape(out_2d, (n, d, h, w, c))
        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        # MaxPool3d
        out = jnp.transpose(out, (0, 2, 3, 4, 1))
        out = jax.lax.reduce_window(
            out,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID',
        )
        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        return out


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(16, 16, 16, 32, 32))]


def get_init_inputs():
    return [16, 32, 3, 2, 1, 1, (32, 1, 1, 1)]
