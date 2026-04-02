import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, b_ref, o_ref):
    x = x_ref[...]  # (1, C, H, W)
    b = b_ref[...]  # (C, 1, 1)

    mean_hw = jnp.mean(x, axis=(2, 3), keepdims=True)
    y = mean_hw + b
    lse = jax.nn.logsumexp(y, axis=1, keepdims=True)
    s = jnp.sum(lse, axis=(2, 3))
    o_ref[...] = s * 10.0


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose2d (kept in JAX)
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        padding = ((2, 2), (2, 2))
        out = jax.lax.conv_transpose(
            x_nhwc,
            kernel,
            strides=(1, 1),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        x = jnp.transpose(out, (0, 3, 1, 2))  # (N, C, H, W)

        N, C, H, W = x.shape

        result = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((N, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N,),
                in_specs=[
                    pl.BlockSpec((1, C, H, W), lambda i: (i, 0, 0, 0)),
                    pl.BlockSpec((C, 1, 1), lambda i: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
            ),
        )(x, self.bias)

        return result


batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
