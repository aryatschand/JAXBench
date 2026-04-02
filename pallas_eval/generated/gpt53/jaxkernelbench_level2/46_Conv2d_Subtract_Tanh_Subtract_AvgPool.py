import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape
        x_2d = x.reshape(N * C, H * W)

        def kernel_fn(x_ref, o_ref):
            val = x_ref[...]
            val = val - self.subtract1_value
            val = jnp.tanh(val)
            val = val - self.subtract2_value
            o_ref[...] = val

        block_m = min(512, x_2d.shape[0])
        block_n = min(512, x_2d.shape[1])

        grid = (x_2d.shape[0] // block_m, x_2d.shape[1] // block_n)

        x_2d = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_2d)

        x = x_2d.reshape(N, C, H, W)

        x = jax.lax.reduce_window(
            x,
            init_value=0.,
            computation=jax.lax.add,
            window_dimensions=(1, 1, self.kernel_size_pool, self.kernel_size_pool),
            window_strides=(1, 1, self.kernel_size_pool, self.kernel_size_pool),
            padding='VALID'
        ) / (self.kernel_size_pool * self.kernel_size_pool)

        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
