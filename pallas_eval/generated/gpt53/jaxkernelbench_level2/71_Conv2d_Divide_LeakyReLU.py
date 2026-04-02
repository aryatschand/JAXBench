import jax
import jax.numpy as jnp
from jax.nn import leaky_relu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_post_kernel(x_ref, b_ref, o_ref):
    x = x_ref[:, :]
    b = b_ref[0, :]
    y = x + b
    y = y / 2.0
    y = jnp.where(y >= 0, y, 0.01 * y)
    o_ref[:, :] = y


def fused_post(x, bias, divisor):
    NHW, C = x.shape

    block_m = 128
    block_n = 128

    grid_m = NHW // block_m
    grid_n = C // block_n

    return pl.pallas_call(
        fused_post_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(grid_m, grid_n),
            in_specs=[
                pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        ),
    )(x, bias.reshape(1, -1))


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        N, H, W, C = x.shape
        x_flat = jnp.reshape(x, (N * H * W, C))

        x_flat = fused_post(x_flat, self.bias, self.divisor)

        x = jnp.reshape(x_flat, (N, H, W, C))
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
