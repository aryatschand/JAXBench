import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def bn_kernel(x_ref, mean_ref, var_ref, scale_ref, bias_ref, o_ref):
    x = x_ref[...]
    mean = mean_ref[...]
    var = var_ref[...]
    scale = scale_ref[...]
    bias = bias_ref[...]
    o_ref[...] = scale * (x - mean) / jnp.sqrt(var + 1e-5) + bias


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        self.conv_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros((out_channels,))

        self.bn_scale = jnp.ones((out_channels,))
        self.bn_bias = jnp.zeros((out_channels,))
        self.bn_mean = jnp.zeros((out_channels,))
        self.bn_var = jnp.ones((out_channels,))

        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))

        pad_size = self.conv_weight.shape[2] - 1 - self.padding
        padding = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))

        conv_out = jax.lax.conv_transpose(
            x_ndhwc,
            kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias.reshape(1, 1, 1, 1, -1)

        conv_out = jnp.transpose(conv_out, (0, 4, 1, 2, 3))

        mean = self.bn_mean.reshape(1, -1, 1, 1, 1)
        var = self.bn_var.reshape(1, -1, 1, 1, 1)
        scale = self.bn_scale.reshape(1, -1, 1, 1, 1)
        bias = self.bn_bias.reshape(1, -1, 1, 1, 1)

        out_shape = conv_out.shape

        bn_out = pl.pallas_call(
            bn_kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, conv_out.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(mean.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(var.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(scale.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(bias.shape, lambda i: (0, 0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
            ),
        )(conv_out, mean, var, scale, bias)

        window_shape = (1, 1, 2, 2, 2)
        strides = (1, 1, 2, 2, 2)

        pool1 = jax.lax.reduce_window(
            bn_out, 0.0, jax.lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        ) / 8.0

        pool2 = jax.lax.reduce_window(
            pool1, 0.0, jax.lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        ) / 8.0

        return pool2


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (64, 3, 32, 32, 32))]


def get_init_inputs():
    return [3, 16, 3, 2, 1, (16, 1, 1, 1)]
