import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_bias_relu_hswish_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[:, :]
    b = bias_ref[0, :]
    y = x + b
    y = jnp.maximum(y, 0)
    y = y * jnp.clip((y + 3.0) / 6.0, 0.0, 1.0)
    o_ref[:, :] = y

def fused_activation(x, bias):
    M, C = x.shape
    block_m = 128
    block_c = 128
    grid = (M // block_m, C // block_c)

    return pl.pallas_call(
        fused_bias_relu_hswish_kernel,
        out_shape=jax.ShapeDtypeStruct((M, C), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_m, block_c), lambda i, j: (i, j)),
                pl.BlockSpec((1, block_c), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_c), lambda i, j: (i, j)),
        ),
    )(x, bias.reshape(1, -1))

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        N, H, W, C = x.shape
        x_2d = x.reshape(N * H * W, C)

        x_2d = fused_activation(x_2d, self.bias)

        x = x_2d.reshape(N, H, W, C)
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
