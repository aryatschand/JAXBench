import jax
import jax.numpy as jnp
from jax.nn import gelu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, o_ref, add_value, multiply_value):
    x = x_ref[...]
    x = x + add_value
    x = jnp.minimum(x, 0.0)
    x = gelu(x)
    x = x * multiply_value
    o_ref[...] = x

def fused_elementwise(x, add_value, multiply_value):
    n, m = x.shape
    block = (min(n, 512), min(m, 512))
    grid = (n // block[0], m // block[1])
    return pl.pallas_call(
        lambda x_ref, o_ref: fused_kernel(x_ref, o_ref, add_value, multiply_value),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.kernel_size = kernel_size
        self.add_value = add_value
        self.multiply_value = multiply_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        padding = ((self.kernel_size - 1, self.kernel_size - 1),
                  (self.kernel_size - 1, self.kernel_size - 1))
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))

        # reshape to 2D for Pallas
        orig_shape = x.shape
        x2d = x.reshape(orig_shape[0], -1)
        x2d = fused_elementwise(x2d, self.add_value, self.multiply_value)
        x = x2d.reshape(orig_shape)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
