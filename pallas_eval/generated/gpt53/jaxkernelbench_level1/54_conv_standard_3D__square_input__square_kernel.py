import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        padding = [(self.padding, self.padding)] * 3

        def kernel_fn(x_ref, w_ref, o_ref):
            x_val = x_ref[...]
            w_val = w_ref[...]

            out = jax.lax.conv_general_dilated(
                x_val,
                w_val,
                window_strides=(self.stride,) * 3,
                padding=padding,
                lhs_dilation=(1,) * 3,
                rhs_dilation=(self.dilation,) * 3,
                dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
                feature_group_count=self.groups
            )

            if self.bias is not None:
                out = out + self.bias.reshape(1, 1, 1, 1, -1)

            o_ref[...] = out

        out_shape = jax.eval_shape(
            lambda a, b: jax.lax.conv_general_dilated(
                a, b,
                window_strides=(self.stride,) * 3,
                padding=padding,
                lhs_dilation=(1,) * 3,
                rhs_dilation=(self.dilation,) * 3,
                dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
                feature_group_count=self.groups
            ),
            x_ndhwc,
            kernel
        ).shape

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec(x_ndhwc.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(kernel.shape, lambda i: (0, 0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
            ),
        )(x_ndhwc, kernel)

        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, depth, width, height))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
