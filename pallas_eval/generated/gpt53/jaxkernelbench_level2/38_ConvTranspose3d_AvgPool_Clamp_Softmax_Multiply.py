import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def scale_kernel(x_ref, scale_ref, o_ref):
    x = x_ref[:, :]
    scale = scale_ref[:, :]
    o_ref[:, :] = x * scale


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        self.conv_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros(out_channels)

        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = jnp.ones((1, out_channels, 1, 1, 1))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # AvgPool
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        k = self.pool_kernel_size
        x = jax.lax.reduce_window(
            x,
            0.0,
            jax.lax.add,
            (1, k, k, k, 1),
            (1, k, k, k, 1),
            'VALID'
        )
        x = x / (k ** 3)

        # ConvTranspose3d
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))
        padding = [(self.kernel_size - 1 - self.padding) for _ in range(3)]
        padding = [(p, p + self.output_padding) for p in padding]

        x = jax.lax.conv_transpose(
            x,
            kernel,
            (self.stride, self.stride, self.stride),
            padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        x = x + self.conv_bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # Clamp
        x = jnp.clip(x, self.clamp_min, self.clamp_max)

        # Softmax (spatial)
        b, c, d, h, w = x.shape
        x = x.reshape(b, c, -1)
        x = jax.nn.softmax(x, axis=2)

        # Pallas scale (fused multiply)
        B, C, S = x.shape
        x_2d = x.reshape(B * C, S)

        scale = self.scale.reshape(C)
        scale_2d = jnp.repeat(scale[:, None], S, axis=1)
        scale_2d = jnp.tile(scale_2d, (B, 1))

        block_m = min(128, B * C)
        block_n = min(128, S)

        grid = ((B * C) // block_m, S // block_n)

        x_scaled = pl.pallas_call(
            scale_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_2d, scale_2d)

        x = x_scaled.reshape(b, c, d, h, w)
        return x


batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]
