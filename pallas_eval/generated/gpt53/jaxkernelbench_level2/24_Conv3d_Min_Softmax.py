import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def post_kernel(x_ref, b_ref, o_ref):
    x = x_ref[...]  # (1, D, 1, 1, C)
    b = b_ref[...]  # (1, C)

    x = x + b.reshape(1, 1, 1, 1, -1)
    x = jnp.min(x, axis=1)  # (1, 1, 1, C)

    x = x - jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x)
    x = exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    o_ref[...] = x


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        self.dim = dim
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        N, D, H, W, C = x.shape

        bias_2d = self.bias.reshape(1, C)

        out = pl.pallas_call(
            post_kernel,
            out_shape=jax.ShapeDtypeStruct((N, H, W, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N, H, W),
                in_specs=[
                    pl.BlockSpec((1, D, 1, 1, C), lambda i, j, k: (i, 0, j, k, 0)),
                    pl.BlockSpec((1, C), lambda i, j, k: (0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, 1, C), lambda i, j, k: (i, j, k, 0)),
            ),
        )(x, bias_2d)

        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
