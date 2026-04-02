import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _sum_kernel(self, x_ref, o_ref):
        x = x_ref[...]
        o_ref[...] = jnp.sum(x, axis=1, keepdims=True)

    def _pallas_sum(self, x_2d):
        M, C = x_2d.shape
        block = (M, C)
        return pl.pallas_call(
            self._sum_kernel,
            out_shape=jax.ShapeDtypeStruct((M, 1), x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(block, lambda i: (0, 0))],
                out_specs=pl.BlockSpec((M, 1), lambda i: (0, 0)),
            ),
        )(x_2d)

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        pad_size = self.kernel_size - 1 - self.padding
        padding = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        )

        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 3, 3, 3, 1),
            window_strides=(1, 3, 3, 3, 1),
            padding='VALID'
        )

        N, D, H, W, C = x.shape
        x_2d = x.reshape(N * D * H * W, C)

        summed = self._pallas_sum(x_2d)

        x = summed.reshape(N, D, H, W, 1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
