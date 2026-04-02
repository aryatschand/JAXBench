import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from typing import List

class Model:
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        window_shape = (1, self.kernel_size, self.kernel_size, self.kernel_size, 1)
        strides = (1, self.stride, self.stride, self.stride, 1)
        padding = [(0, 0),
                   (self.padding, self.padding),
                   (self.padding, self.padding),
                   (self.padding, self.padding),
                   (0, 0)]

        def kernel_fn(x_ref, o_ref):
            x_val = x_ref[...]
            pooled_sum = lax.reduce_window(
                x_val,
                init_value=0.0,
                computation=lax.add,
                window_dimensions=window_shape,
                window_strides=strides,
                padding=padding,
            )
            o_ref[...] = pooled_sum / (self.kernel_size ** 3)

        # compute output shape using reference op
        out = lax.reduce_window(
            x,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding=padding,
        )
        out_shape = out.shape

        result = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(x.shape, lambda i: (0, 0, 0, 0, 0))],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
            ),
        )(x)

        result = jnp.transpose(result, (0, 4, 1, 2, 3))
        return result

    def set_weights(self, weights_dict):
        pass

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
