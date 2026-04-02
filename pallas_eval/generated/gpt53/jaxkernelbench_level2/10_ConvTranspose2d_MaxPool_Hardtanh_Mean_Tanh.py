import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _mean_tanh_kernel(self, x_ref, o_ref):
        x = x_ref[...]  # (block_m, block_k)
        x = jnp.clip(x, self.hardtanh_min, self.hardtanh_max)
        mean = jnp.mean(x, axis=1, keepdims=True)
        o_ref[...] = jnp.tanh(mean)

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        pad_val = self.kernel_size - 1 - self.padding
        padding = ((pad_val, pad_val), (pad_val, pad_val))

        x = jax.lax.conv_transpose(
            x_nhwc,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))

        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        window_shape = (1, self.maxpool_kernel_size, self.maxpool_kernel_size, 1)
        strides = (1, self.maxpool_stride, self.maxpool_stride, 1)

        x = jax.lax.reduce_window(
            x_nhwc,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        )

        x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW

        # reshape to 2D for Pallas: (N*C, H*W)
        N, C, H, W = x.shape
        x2d = x.reshape(N * C, H * W)

        block_m = 128
        block_k = 1024

        M = (N * C) // block_m
        K = (H * W) // block_k

        out = pl.pallas_call(
            self._mean_tanh_kernel,
            out_shape=jax.ShapeDtypeStruct((N * C, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(M, K),
                in_specs=[
                    pl.BlockSpec((block_m, block_k), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i, j: (i, 0)),
            ),
        )(x2d)

        out = out.reshape(N, C, 1, 1)
        return out


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 64, 256, 256))]


def get_init_inputs():
    return [64, 64, 3, 1, 1, 2, 2, -1, 1]
