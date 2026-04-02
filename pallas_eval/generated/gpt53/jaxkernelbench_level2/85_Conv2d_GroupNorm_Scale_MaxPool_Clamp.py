import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def groupnorm_kernel(x_ref, gamma_ref, beta_ref, o_ref):
    x = x_ref[...]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)

    gamma = gamma_ref[...]
    beta = beta_ref[...]

    o_ref[...] = x_norm * gamma + beta


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))

        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))
        self.num_groups = num_groups
        self.out_channels = out_channels

        self.scale = jnp.ones(scale_shape)

        self.maxpool_kernel_size = maxpool_kernel_size

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv (keep as JAX primitive)
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        x = jax.lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))

        # GroupNorm via Pallas
        N, C, H, W = x.shape
        G = self.num_groups
        group_size = (C // G) * H * W

        x_grouped = x.reshape(N, G, group_size)

        gamma = self.group_norm_weight.reshape(1, C, 1, 1)
        beta = self.group_norm_bias.reshape(1, C, 1, 1)
        gamma = gamma.reshape(N, G, group_size)
        beta = beta.reshape(N, G, group_size)

        block = (1, 1, group_size)
        grid = (N, G)

        x_norm = pl.pallas_call(
            groupnorm_kernel,
            out_shape=jax.ShapeDtypeStruct(x_grouped.shape, x_grouped.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j, 0)),
                    pl.BlockSpec(block, lambda i, j: (i, j, 0)),
                    pl.BlockSpec(block, lambda i, j: (i, j, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j, 0)),
            ),
        )(x_grouped, gamma, beta)

        x = x_norm.reshape(N, C, H, W)

        # Scale (fused cheap op)
        x = x * self.scale

        # MaxPool
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        k = self.maxpool_kernel_size
        window_shape = (1, k, k, 1)
        x = jax.lax.reduce_window(
            x_nhwc,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=window_shape,
            window_strides=window_shape,
            padding='VALID')
        x = jnp.transpose(x, (0, 3, 1, 2))

        # Clamp
        x = jnp.clip(x, self.clamp_min, self.clamp_max)

        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
