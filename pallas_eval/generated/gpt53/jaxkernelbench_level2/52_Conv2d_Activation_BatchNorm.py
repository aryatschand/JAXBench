import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, mean_ref, var_ref, weight_ref, bias_ref, o_ref):
    x = x_ref[...]

    # Mish activation: x * tanh(softplus(x))
    softplus_x = jnp.log1p(jnp.exp(x))
    mish = jnp.tanh(softplus_x) * x

    mean = mean_ref[...]
    var = var_ref[...]
    weight = weight_ref[...]
    bias = bias_ref[...]

    y = (mish - mean) / jnp.sqrt(var + 1e-5) * weight + bias
    o_ref[...] = y


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

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

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        weight = jnp.transpose(self.conv_weight, (2, 3, 1, 0))

        x = lax.conv_general_dilated(
            x,
            weight,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        x = x + self.conv_bias.reshape(1, 1, 1, -1)

        # NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape
        NHW = N * H * W

        # Compute BN stats
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)

        # Broadcast everything to full shape
        mean_b = jnp.broadcast_to(mean, x.shape)
        var_b = jnp.broadcast_to(var, x.shape)
        weight_b = jnp.broadcast_to(self.bn_weight.reshape(1, -1, 1, 1), x.shape)
        bias_b = jnp.broadcast_to(self.bn_bias.reshape(1, -1, 1, 1), x.shape)

        # Reshape to 2D (C, NHW)
        x_2d = jnp.reshape(jnp.transpose(x, (1, 0, 2, 3)), (C, NHW))
        mean_2d = jnp.reshape(jnp.transpose(mean_b, (1, 0, 2, 3)), (C, NHW))
        var_2d = jnp.reshape(jnp.transpose(var_b, (1, 0, 2, 3)), (C, NHW))
        weight_2d = jnp.reshape(jnp.transpose(weight_b, (1, 0, 2, 3)), (C, NHW))
        bias_2d = jnp.reshape(jnp.transpose(bias_b, (1, 0, 2, 3)), (C, NHW))

        block = (128, 64)
        grid = (C // 128, NHW // 64)

        out_2d = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((C, NHW), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x_2d, mean_2d, var_2d, weight_2d, bias_2d)

        # Reshape back to NCHW
        out = jnp.transpose(jnp.reshape(out_2d, (C, N, H, W)), (1, 0, 2, 3))
        return out


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
