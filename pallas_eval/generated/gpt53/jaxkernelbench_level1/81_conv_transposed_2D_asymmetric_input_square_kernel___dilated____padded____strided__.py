import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def _kernel_copy(self, x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        effective_kernel_h = self.dilation * (self._kernel_size - 1) + 1
        effective_kernel_w = self.dilation * (self._kernel_size - 1) + 1

        pad_h = effective_kernel_h - 1 - self.padding
        pad_w = effective_kernel_w - 1 - self.padding
        padding = ((pad_h, pad_h), (pad_w, pad_w))

        if self.dilation > 1:
            kh, kw, out_c, in_c = kernel.shape
            dilated_kh = (kh - 1) * self.dilation + 1
            dilated_kw = (kw - 1) * self.dilation + 1
            dilated_kernel = jnp.zeros((dilated_kh, dilated_kw, out_c, in_c))
            dilated_kernel = dilated_kernel.at[::self.dilation, ::self.dilation, :, :].set(kernel)
            kernel = dilated_kernel

        out = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, -1)

        # reshape to 2D for Pallas
        n, h, w, c = out.shape
        out_2d = out.reshape(n * h, w * c)

        block_h = min(512, out_2d.shape[0])
        block_w = min(512, out_2d.shape[1])

        grid = (
            out_2d.shape[0] // block_h,
            out_2d.shape[1] // block_w,
        )

        result_2d = pl.pallas_call(
            self._kernel_copy,
            out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_h, block_w), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_h, block_w), lambda i, j: (i, j)),
            ),
        )(out_2d)

        out = result_2d.reshape(n, h, w, c)
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

    @property
    def kernel_size(self):
        return self._kernel_size


batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
