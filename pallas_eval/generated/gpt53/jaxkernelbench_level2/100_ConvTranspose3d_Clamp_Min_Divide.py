import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def clamp_div_kernel(x_ref, o_ref, min_val, divisor):
    x = x_ref[...]
    y = jnp.maximum(x, min_val) / divisor
    o_ref[...] = y

def pallas_clamp_div(x, min_val, divisor):
    orig_shape = x.shape
    # flatten to 2D (M, N)
    M = orig_shape[0] * orig_shape[1] * orig_shape[2] * orig_shape[3]
    N = orig_shape[4]
    x2d = x.reshape(M, N)

    block_m = 128
    block_n = 128

    grid = (M // block_m, N // block_n)

    def kernel(x_ref, o_ref):
        clamp_div_kernel(x_ref, o_ref, min_val, divisor)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        ),
    )(x2d)

    return out.reshape(orig_shape)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.pytorch_padding = padding
        self.min_value = min_value
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        pad_val = self.kernel_size - 1 - self.pytorch_padding
        padding = ((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val))

        x = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        x = pallas_clamp_div(x, self.min_value, self.divisor)

        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
