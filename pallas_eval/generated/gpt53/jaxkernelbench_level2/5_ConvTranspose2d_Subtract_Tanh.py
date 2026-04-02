import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_bias_tanh_kernel(x_ref, conv_bias_ref, bias_ref, o_ref):
    x = x_ref[...]  # (B, C) where B = NHW flattened
    conv_b = conv_bias_ref[...]  # (C,)
    bias = bias_ref[...]  # (C,)

    y = x + conv_b[None, :] - bias[None, :]
    y = jnp.tanh(y)

    o_ref[...] = y

def fused_bias_tanh(x, conv_bias, bias):
    B, C = x.shape
    block_b = min(B, 128)
    block_c = min(C, 128)

    grid_b = B // block_b
    grid_c = C // block_c

    return pl.pallas_call(
        fused_bias_tanh_kernel,
        out_shape=jax.ShapeDtypeStruct((B, C), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(grid_b, grid_c),
            in_specs=[
                pl.BlockSpec((block_b, block_c), lambda i, j: (i, j)),
                pl.BlockSpec((block_c,), lambda i, j: (j,)),
                pl.BlockSpec((block_c,), lambda i, j: (j,)),
            ],
            out_specs=pl.BlockSpec((block_b, block_c), lambda i, j: (i, j)),
        ),
    )(x, conv_bias, bias)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape

        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)

        self.conv_transpose_weight = jax.random.normal(key1, (in_channels, out_channels, kernel_size, kernel_size)) * 0.01
        self.conv_transpose_bias = jax.random.normal(key2, (out_channels,)) * 0.01
        self.bias = jax.random.normal(key3, bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

        weight = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))

        pad = self.kernel_size - 1 - self.padding

        x_conv = lax.conv_transpose(
            x_nhwc,
            weight,
            strides=(self.stride, self.stride),
            padding=((pad, pad + self.output_padding), (pad, pad + self.output_padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )

        N, H, W, C = x_conv.shape
        x_flat = x_conv.reshape(N * H * W, C)

        bias_flat = self.bias.reshape(C)

        y_flat = fused_bias_tanh(x_flat, self.conv_transpose_bias, bias_flat)

        y_nhwc = y_flat.reshape(N, H, W, C)

        y_nchw = jnp.transpose(y_nhwc, (0, 3, 1, 2))

        return y_nchw

batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(42)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
