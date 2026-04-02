import jax
import jax.numpy as jnp
from jax.nn import gelu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, s_ref, o_ref):
    x = x_ref[:, :]
    s = s_ref[:, :]
    y = jnp.where(x >= 0, x, 0.2 * x)
    y = y + s
    y = jnp.clip(y, -1.0, 1.0)
    y = gelu(y)
    o_ref[:, :] = y


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.sum_tensor = jnp.zeros(sum_tensor_shape)

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
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NCDHW

        N, C, D, H, W = x.shape
        x2d = x.reshape(N * C, D * H * W)

        sum_broadcast = jnp.broadcast_to(self.sum_tensor, (C, D, H, W))
        sum_broadcast = jnp.tile(sum_broadcast[None, ...], (N, 1, 1, 1, 1))
        s2d = sum_broadcast.reshape(N * C, D * H * W)

        block_m = min(128, x2d.shape[0])
        block_n = min(128, x2d.shape[1])

        grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

        out2d = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2d, s2d)

        return out2d.reshape(N, C, D, H, W)


batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
