import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, o_ref, add_value, scale):
    x = x_ref[...]

    # Mish: x * tanh(softplus(x))
    softplus = jnp.log1p(jnp.exp(x))
    mish = x * jnp.tanh(softplus)

    # Add value
    y = mish + add_value

    # Hardtanh
    y = jnp.clip(y, -1.0, 1.0)

    # Scale
    y = y * scale

    o_ref[...] = y

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding),
                   (pad_w, pad_w + self.output_padding))

        x = jax.lax.conv_transpose(
            x_nhwc,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW

        N, C, H, W = x.shape

        block = (1, C, 8, 128)
        grid = (N, H // 8, W // 128)

        def kernel(x_ref, o_ref):
            fused_kernel(x_ref, o_ref, self.add_value, self.scale)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda n, h, w: (n, 0, h, w))],
                out_specs=pl.BlockSpec(block, lambda n, h, w: (n, 0, h, w)),
            ),
        )(x)

        return out

    @property
    def kernel_size(self):
        return self.weight.shape[2]

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
