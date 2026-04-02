import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pool_avg_kernel(x_ref, o_ref):
    x = x_ref[...]  # (N, D, H, W, C)
    # Max pool with stride = window (assumes divisible)
    N, D, H, W, C = x.shape
    d2 = D // 2
    h2 = H // 2
    w2 = W // 2

    x = x.reshape(N, d2, 2, h2, 2, w2, 2, C)
    x = jnp.max(x, axis=(2, 4, 6))  # (N, d2, h2, w2, C)

    # Global average pool
    x = jnp.mean(x, axis=(1, 2, 3), keepdims=True)  # (N,1,1,1,C)

    o_ref[...] = x


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.bias_conv = jnp.zeros(out_channels)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = jnp.zeros(bias_shape)
        self.sum_dim = sum_dim

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
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        x = x + self.bias_conv.reshape(1, 1, 1, 1, -1)
        x = x / self.divisor

        N, D, H, W, C = x.shape
        out_shape = (N, 1, 1, 1, C)

        result = pl.pallas_call(
            pool_avg_kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(x.shape, lambda i: (0, 0, 0, 0, 0))],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
            ),
        )(x)

        x = jnp.transpose(result, (0, 4, 1, 2, 3))
        x = x + self.bias
        x = jnp.sum(x, axis=self.sum_dim)

        return x


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 8, 16, 64, 64))]


def get_init_inputs():
    return [8, 16, (3, 3, 3), 2.0, (2, 2, 2), (16, 1, 1, 1), 1]
