import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def softmax_sigmoid_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    x_max = jnp.max(x, axis=1, keepdims=True)
    x_exp = jnp.exp(x - x_max)
    x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
    softmax = x_exp / x_sum
    o_ref[:, :] = jnn.sigmoid(softmax)

def pallas_softmax_sigmoid(x):
    n, c = x.shape
    block_m = 128
    block_n = 128
    grid = (n // block_m, c // block_n)

    return pl.pallas_call(
        softmax_sigmoid_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        ),
    )(x)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        k = kernel_size
        if isinstance(k, int):
            k = (k, k, k)
        self.kernel_size = k
        self.weight = jnp.zeros((in_channels, out_channels, k[0], k[1], k[2]))
        if bias:
            self.bias = jnp.zeros((out_channels,))
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.has_bias = bias

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        pad_d = self.kernel_size[0] - 1 - self.padding[0]
        pad_h = self.kernel_size[1] - 1 - self.padding[1]
        pad_w = self.kernel_size[2] - 1 - self.padding[2]
        padding = (
            (pad_d, pad_d + self.output_padding[0]),
            (pad_h, pad_h + self.output_padding[1]),
            (pad_w, pad_w + self.output_padding[2]),
        )

        x = jax.lax.conv_transpose(
            x,
            kernel,
            strides=self.stride,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
        )

        if self.has_bias:
            x = x + self.bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        n, c, d, h, w = x.shape
        x2d = jnp.reshape(x, (n * d * h * w, c))

        x2d = pallas_softmax_sigmoid(x2d)

        x = jnp.reshape(x2d, (n, d, h, w, c))
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
