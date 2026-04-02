import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    @property
    def kernel_size(self):
        return self.weight.shape[2]

    def _copy_kernel(self, x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        pad_d = self.kernel_size - 1 - self.padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_d, pad_d + self.output_padding),
                   (pad_h, pad_h + self.output_padding),
                   (pad_w, pad_w + self.output_padding))

        if self.groups == 1:
            out = jax.lax.conv_transpose(
                x, kernel,
                strides=(self.stride, self.stride, self.stride),
                padding=padding,
                dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
        else:
            x_groups = jnp.split(x, self.groups, axis=-1)
            k_groups = jnp.split(kernel, self.groups, axis=-1)
            out_groups = []
            for xg, kg in zip(x_groups, k_groups):
                out_groups.append(
                    jax.lax.conv_transpose(
                        xg, kg,
                        strides=(self.stride, self.stride, self.stride),
                        padding=padding,
                        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))
                )
            out = jnp.concatenate(out_groups, axis=-1)

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)

        # reshape to 2D for Pallas (TPU requires >=2D)
        n, d, h, w, c = out.shape
        out_2d = jnp.reshape(out, (n * d * h, w * c))

        block_m = min(512, out_2d.shape[0])
        block_n = min(512, out_2d.shape[1])

        grid = (out_2d.shape[0] // block_m, out_2d.shape[1] // block_n)

        result_2d = pl.pallas_call(
            self._copy_kernel,
            out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(out_2d)

        out = jnp.reshape(result_2d, (n, d, h, w, c))
        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(8, 48, 64, 64, 64))
    return [x]


def get_init_inputs():
    return [48, 48, 3]
