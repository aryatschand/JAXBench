import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import relu


def hardswish_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x * jnp.minimum(jnp.maximum(x + 3.0, 0.0), 6.0) / 6.0


def pallas_hardswish(x):
    # flatten to 2D for TPU constraint
    n = x.shape[0]
    rest = x.size // n
    x2d = jnp.reshape(x, (n, rest))

    block = (1, min(rest, 1024))
    grid = (n, rest // block[1])

    out2d = pl.pallas_call(
        hardswish_kernel,
        out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x2d)

    return jnp.reshape(out2d, x.shape)


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = jnp.zeros(out_channels)

        self.num_groups = num_groups
        self.gamma = jnp.ones(out_channels)
        self.beta = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))

        if hasattr(self, 'bias'):
            x = x + self.bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # Pallas HardSwish
        x = pallas_hardswish(x)

        # GroupNorm
        N, C, D, H, W = x.shape
        x = x.reshape(N, self.num_groups, C // self.num_groups, D, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, D, H, W)
        x = self.gamma.reshape(1, -1, 1, 1, 1) * x + self.beta.reshape(1, -1, 1, 1, 1)

        x = jnp.mean(x, axis=(2, 3, 4))

        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]


batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4
