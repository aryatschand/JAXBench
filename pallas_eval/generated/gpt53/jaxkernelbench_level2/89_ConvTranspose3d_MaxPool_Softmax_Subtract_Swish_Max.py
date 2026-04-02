import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, subtract_ref, o_ref):
    x = x_ref[...]  # (B, C)
    sub = subtract_ref[...]  # (C,)

    # softmax over channels
    x_max = jnp.max(x, axis=1, keepdims=True)
    x_exp = jnp.exp(x - x_max)
    x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
    soft = x_exp / x_sum

    # subtract
    y = soft - sub

    # swish
    y = y * jax.nn.sigmoid(y)

    # max over channels
    out = jnp.max(y, axis=1, keepdims=True)

    o_ref[...] = out


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.subtract = jnp.zeros(out_channels)

        self.stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.padding_val = padding
        self.output_padding = output_padding

        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        pad_val = self.kernel_size - 1 - self.padding_val
        padding = ((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val))

        x = jax.lax.conv_transpose(
            x_ndhwc,
            kernel,
            strides=self.stride,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
        )

        if self.output_padding:
            x = jax.lax.pad(
                x,
                0.0,
                (
                    (0, 0, 0),
                    (0, self.output_padding, 0),
                    (0, self.output_padding, 0),
                    (0, self.output_padding, 0),
                    (0, 0, 0),
                ),
            )

        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # MaxPool3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))

        pool_window = (1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1)
        pool_strides = (1, self.pool_stride, self.pool_stride, self.pool_stride, 1)

        pool_padding = (
            (0, 0),
            (self.pool_padding, self.pool_padding),
            (self.pool_padding, self.pool_padding),
            (self.pool_padding, self.pool_padding),
            (0, 0),
        )

        x = jax.lax.reduce_window(
            x_ndhwc,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=pool_window,
            window_strides=pool_strides,
            padding=pool_padding,
        )

        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NCDHW

        # reshape to 2D for Pallas
        N, C, D, H, W = x.shape
        rows = N * D * H * W
        x2d = jnp.reshape(jnp.transpose(x, (0, 2, 3, 4, 1)), (rows, C))

        block_m = 128
        grid_m = rows // block_m

        out2d = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((rows, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[
                    pl.BlockSpec((block_m, C), lambda i: (i, 0)),
                    pl.BlockSpec((C,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(x2d, self.subtract)

        out = jnp.reshape(out2d, (N, D, H, W))
        return out


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (128, 3, 16, 32, 32))]


def get_init_inputs():
    return [3, 16, 3, 2, 1, 1, 2, 2, 0]
