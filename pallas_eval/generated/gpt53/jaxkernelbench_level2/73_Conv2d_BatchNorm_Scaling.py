import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.eps = 1e-5

        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        self.conv_weight = jax.random.normal(key1, (out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jax.random.normal(key2, (out_channels,))

        self.bn_weight = jnp.ones((out_channels,))
        self.bn_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def _bn_scale_kernel(self, x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref):
        x = x_ref[...]

        bm = x.shape[0]
        bn = x.shape[1]

        pid = pl.program_id(axis=0)
        row_start = pid * bm

        row_ids = row_start + jnp.arange(bm)
        C = mean_ref.shape[0]

        ch_idx = row_ids % C
        ch_idx = ch_idx.reshape(-1, 1)

        mean = mean_ref[ch_idx, 0]
        var = var_ref[ch_idx, 0]
        gamma = gamma_ref[ch_idx, 0]
        beta = beta_ref[ch_idx, 0]

        y = (x - mean) / jnp.sqrt(var + self.eps)
        y = y * gamma + beta
        y = y * self.scaling_factor

        o_ref[...] = y

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))

        weight = jnp.transpose(self.conv_weight, (2, 3, 1, 0))

        x = lax.conv_general_dilated(
            x, weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        x = x + self.conv_bias.reshape(1, 1, 1, -1)

        x = jnp.transpose(x, (0, 3, 1, 2))

        mean = jnp.mean(x, axis=(0, 2, 3))
        var = jnp.mean((x - mean.reshape(1, -1, 1, 1)) ** 2, axis=(0, 2, 3))

        N, C, H, W = x.shape
        x2d = x.reshape(N * C, H * W)

        mean2d = mean.reshape(C, 1)
        var2d = var.reshape(C, 1)
        gamma2d = self.bn_weight.reshape(C, 1)
        beta2d = self.bn_bias.reshape(C, 1)

        bm = 128
        bn = 128

        grid_m = (x2d.shape[0] // bm)
        grid_n = (x2d.shape[1] // bn)

        def kernel(x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref):
            self._bn_scale_kernel(x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[
                    pl.BlockSpec((bm, bn), lambda i: (i, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                    pl.BlockSpec((C, 1), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i: (i, 0)),
            ),
        )(x2d, mean2d, var2d, gamma2d, beta2d)

        x = out.reshape(N, C, H, W)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
