import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding),
                   (pad_w, pad_w + self.output_padding))

        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW

        N, C, H, W = x.shape
        x2 = jnp.reshape(x, (N * H * W, C))

        bias_vec = jnp.reshape(self.bias, (1, C))

        def kernel_fn(x_ref, b_ref, o_ref):
            x_block = x_ref[...]
            b_block = b_ref[...]

            m = jnp.max(x_block, axis=1, keepdims=True)
            e = jnp.exp(x_block - m)
            s = jnp.sum(e, axis=1, keepdims=True)
            sm = e / s

            y = sm + b_block
            y = y * self.scaling_factor
            y = sigmoid(y)

            o_ref[...] = y

        block_m = 128
        block_n = C  # full channel

        grid_m = x2.shape[0] // block_m
        grid_n = x2.shape[1] // block_n

        y2 = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2, bias_vec)

        y = jnp.reshape(y2, (N, C, H, W))
        return y

    @property
    def kernel_size(self):
        return self.weight.shape[2]

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
