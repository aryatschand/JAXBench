import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def elem_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    y = x - elem_kernel.subtract_value
    y = y * jnp.minimum(jnp.maximum(y + 3.0, 0.0), 6.0) / 6.0
    o_ref[:, :] = y


def mish_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    y = x * jnp.tanh(jax.nn.softplus(x))
    o_ref[:, :] = y


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv (same as baseline)
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        x = x + self.bias.reshape(1, 1, 1, -1)

        # Flatten to 2D for Pallas
        n, h, w, c = x.shape
        x2 = x.reshape(n * h * w, c)

        # First fused kernel: subtract + hardswish
        elem_kernel.subtract_value = self.subtract_value

        block_m = 128
        block_n = 128
        grid = (x2.shape[0] // block_m, x2.shape[1] // block_n)

        x2 = pl.pallas_call(
            elem_kernel,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2)

        x = x2.reshape(n, h, w, c)

        # MaxPool
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, self.pool_kernel_size, self.pool_kernel_size, 1),
            window_strides=(1, self.pool_kernel_size, self.pool_kernel_size, 1),
            padding="VALID",
        )

        # Flatten again
        n, h, w, c = x.shape
        x2 = x.reshape(n * h * w, c)

        # Mish kernel
        grid = (x2.shape[0] // block_m, x2.shape[1] // block_n)

        x2 = pl.pallas_call(
            mish_kernel,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2)

        x = x2.reshape(n, h, w, c)

        # Back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
