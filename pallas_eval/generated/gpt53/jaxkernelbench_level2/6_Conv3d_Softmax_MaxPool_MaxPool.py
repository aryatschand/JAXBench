import jax
import jax.numpy as jnp
from jax.nn import softmax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _softmax_kernel(self, x_ref, o_ref):
        x = x_ref[:, :]
        x_max = jnp.max(x, axis=1, keepdims=True)
        x_exp = jnp.exp(x - x_max)
        x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
        o_ref[:, :] = x_exp / x_sum

    def _pallas_softmax(self, x):
        # x: (N, C, D, H, W) -> reshape to (N*D*H*W, C)
        N, C, D, H, W = x.shape
        x2 = jnp.transpose(x, (0, 2, 3, 4, 1)).reshape(-1, C)

        M = x2.shape[0]
        block_m = min(128, M)
        block_n = C

        grid = (M // block_m,)

        out = pl.pallas_call(
            self._softmax_kernel,
            out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i: (i, 0)),
            ),
        )(x2)

        out = out.reshape(N, D, H, W, C)
        return jnp.transpose(out, (0, 4, 1, 2, 3))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        x = self._pallas_softmax(x)

        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        for _ in range(2):
            x = jax.lax.reduce_window(
                x,
                init_value=-jnp.inf,
                computation=jax.lax.max,
                window_dimensions=(1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1),
                window_strides=(1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1),
                padding='VALID'
            )

        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
