import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _fused_kernel(self, x, bias, scale):
        n, d, h, w, c = x.shape
        x2d = x.reshape(n * d * h * w, c)

        def kernel(x_ref, b_ref, o_ref):
            x_block = x_ref[:, :]
            b = b_ref[0, :]
            out = x_block + b
            out = out * scale
            o_ref[:, :] = out

        block_m = min(256, x2d.shape[0])
        block_n = min(128, x2d.shape[1])

        grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2d, bias.reshape(1, -1))

        return out.reshape(n, d, h, w, c)

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        pad = self.weight.shape[2] - 1 - self.padding
        padding = ((pad, pad), (pad, pad), (pad, pad))

        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        x = self._fused_kernel(x, self.bias, self.scale)

        window_shape = (1, self.maxpool_kernel_size, self.maxpool_kernel_size, self.maxpool_kernel_size, 1)
        strides = window_shape
        x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window_shape, strides, 'VALID')

        x = jnp.mean(x, axis=(1, 2, 3), keepdims=True)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        x = jnp.clip(x, self.clamp_min, self.clamp_max)

        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 3, 16, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, 0.5, 2]
