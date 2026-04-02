import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding

        x = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride),
            padding=((pad_h, pad_h + self.output_padding), (pad_w, pad_w + self.output_padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, -1)

        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x * self.multiplier

        N, C, H, W = x.shape
        x_flat = jnp.reshape(x, (N * C, H * W))

        def mean_kernel(x_ref, o_ref):
            vals = x_ref[0, :]
            s = jnp.sum(vals)
            o_ref[0, 0] = s / vals.shape[0]

        out = pl.pallas_call(
            mean_kernel,
            out_shape=jax.ShapeDtypeStruct((N * C, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N * C,),
                in_specs=[pl.BlockSpec((1, H * W), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
            ),
        )(x_flat)

        out = jnp.reshape(out, (N, C, 1, 1))
        return out


batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]
