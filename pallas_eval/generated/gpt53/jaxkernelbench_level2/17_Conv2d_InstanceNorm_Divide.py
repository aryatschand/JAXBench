import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def norm_div_kernel(x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref):
    x = x_ref[:, :]
    mean = mean_ref[:, :]
    var = var_ref[:, :]
    gamma = gamma_ref[:, :]
    beta = beta_ref[:, :]

    y = (x - mean) / jnp.sqrt(var + 1e-5)
    y = y * gamma + beta
    y = y / 2.0

    o_ref[:, :] = y

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.instance_norm_weight = jnp.ones((out_channels,))
        self.instance_norm_bias = jnp.zeros((out_channels,))
        self.divide_by = divide_by

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        x = jax.lax.conv_general_dilated(
            x_nhwc,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))

        mean = jnp.mean(x, axis=(2, 3), keepdims=True)
        var = jnp.var(x, axis=(2, 3), keepdims=True)

        N, C, H, W = x.shape
        x2d = x.reshape(N * C, H * W)
        mean2d = jnp.broadcast_to(mean, (N, C, H, W)).reshape(N * C, H * W)
        var2d = jnp.broadcast_to(var, (N, C, H, W)).reshape(N * C, H * W)
        gamma2d = jnp.broadcast_to(
            self.instance_norm_weight.reshape(1, C, 1, 1),
            (N, C, H, W)
        ).reshape(N * C, H * W)
        beta2d = jnp.broadcast_to(
            self.instance_norm_bias.reshape(1, C, 1, 1),
            (N, C, H, W)
        ).reshape(N * C, H * W)

        block_m = 128
        block_n = 128
        grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

        def kernel(x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref):
            i = pl.program_id(axis=0)
            j = pl.program_id(axis=1)
            xs = x_ref[:, :]
            ms = mean_ref[:, :]
            vs = var_ref[:, :]
            gs = gamma_ref[:, :]
            bs = beta_ref[:, :]
            y = (xs - ms) / jnp.sqrt(vs + 1e-5)
            y = y * gs + bs
            y = y / self.divide_by
            o_ref[:, :] = y

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x2d, mean2d, var2d, gamma2d, beta2d)

        return out.reshape(N, C, H, W)

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
