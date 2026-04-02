import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def post_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]  # (1, H, W, C)
    bias = bias_ref[...]  # (C,)

    x = x + bias.reshape((1, 1, 1, -1))
    x = jnn.gelu(x)
    x = jnp.mean(x, axis=(1, 2))  # (1, C)

    o_ref[...] = x


class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Kernel transform
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        # Convolution (kept in XLA, very efficient)
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        N, H, W, C = x.shape

        # Pallas fused: bias + GELU + global avg pool
        result = pl.pallas_call(
            post_kernel,
            out_shape=jax.ShapeDtypeStruct((N, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N,),
                in_specs=[
                    pl.BlockSpec((1, H, W, C), lambda i: (i, 0, 0, 0)),
                    pl.BlockSpec((C,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((1, C), lambda i: (i, 0)),
            ),
        )(x, self.bias)

        return result


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
