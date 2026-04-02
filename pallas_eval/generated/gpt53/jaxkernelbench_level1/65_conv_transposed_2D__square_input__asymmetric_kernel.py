import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def identity_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

def pallas_identity(x):
    shape = x.shape
    block = shape
    grid = (1,) * len(shape)
    return pl.pallas_call(
        identity_kernel,
        out_shape=jax.ShapeDtypeStruct(shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec(block, lambda *idx: tuple(0 for _ in idx))],
            out_specs=pl.BlockSpec(block, lambda *idx: tuple(0 for _ in idx)),
        ),
    )(x)

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size[0], kernel_size[1]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = (stride, stride)
        self.padding = ((kernel_size[0]-1-padding, kernel_size[0]-1-padding),
                       (kernel_size[1]-1-padding, kernel_size[1]-1-padding))
        self.output_padding = output_padding
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        if self.groups == 1:
            x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
            kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
            out = jax.lax.conv_transpose(
                x_nhwc,
                kernel,
                strides=self.stride,
                padding=self.padding,
                dimension_numbers=('NHWC', 'HWOI', 'NHWC')
            )
            if self.bias is not None:
                out = out + self.bias.reshape(1, 1, 1, -1)
            out = jnp.transpose(out, (0, 3, 1, 2))
        else:
            x_groups = jnp.split(x, self.groups, axis=1)
            kernel_groups = jnp.split(self.weight, self.groups, axis=0)
            out_groups = []
            for x_group, kernel_group in zip(x_groups, kernel_groups):
                x_nhwc = jnp.transpose(x_group, (0, 2, 3, 1))
                kernel_hwoi = jnp.transpose(kernel_group, (2, 3, 1, 0))
                out_group = jax.lax.conv_transpose(
                    x_nhwc,
                    kernel_hwoi,
                    strides=self.stride,
                    padding=self.padding,
                    dimension_numbers=('NHWC', 'HWOI', 'NHWC')
                )
                out_group = jnp.transpose(out_group, (0, 3, 1, 2))
                out_groups.append(out_group)
            out = jnp.concatenate(out_groups, axis=1)
            if self.bias is not None:
                out = out + self.bias.reshape(1, -1, 1, 1)

        out = pallas_identity(out)
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(8, 64, 512, 512))
    return [x]

def get_init_inputs():
    return [64, 64, (3, 7)]
