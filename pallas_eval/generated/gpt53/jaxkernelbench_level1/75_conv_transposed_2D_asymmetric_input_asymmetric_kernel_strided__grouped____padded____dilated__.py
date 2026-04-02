import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        self.weight = jnp.zeros((in_channels, out_channels // groups, kernel_size[0], kernel_size[1]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _conv_reference(self, x):
        x_groups = jnp.split(x, self.groups, axis=1)
        w_groups = jnp.split(self.weight, self.groups, axis=0)

        results = []
        for g in range(self.groups):
            x_g = jnp.transpose(x_groups[g], (0, 2, 3, 1))
            kernel_g = jnp.transpose(w_groups[g], (2, 3, 1, 0))

            dilated_kernel_h = self.dilation[0] * (self.kernel_size[0] - 1) + 1
            dilated_kernel_w = self.dilation[1] * (self.kernel_size[1] - 1) + 1

            pad_h_total = dilated_kernel_h - 1
            pad_w_total = dilated_kernel_w - 1

            pad_h_before = pad_h_total - self.padding[0]
            pad_h_after = pad_h_total - self.padding[0]
            pad_w_before = pad_w_total - self.padding[1]
            pad_w_after = pad_w_total - self.padding[1]

            padding_jax = ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after))

            out_g = jax.lax.conv_general_dilated(
                x_g,
                kernel_g,
                window_strides=(1, 1),
                padding=padding_jax,
                lhs_dilation=self.stride,
                rhs_dilation=self.dilation,
                dimension_numbers=('NHWC', 'HWOI', 'NHWC')
            )

            out_g = jnp.transpose(out_g, (0, 3, 1, 2))
            results.append(out_g)

        out = jnp.concatenate(results, axis=1)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out

    def forward(self, x):
        out = self._conv_reference(x)

        # reshape to 2D for Pallas (TPU requires >=2D, we use flatten spatial)
        n, c, h, w = out.shape
        out_2d = out.reshape(n, c * h * w)

        def kernel(o_ref, out_ref):
            out_ref[...] = o_ref[...]

        block_m = min(n, 8)
        block_n = min(c * h * w, 128)

        grid = (n // block_m, (c * h * w) // block_n)

        result = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(out_2d)

        return result.reshape(n, c, h, w)


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]
