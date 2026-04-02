import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.scale_factor = scale_factor

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

        n, h, w, c = x.shape
        x_2d = jnp.reshape(x, (n * h * w, c))

        def kernel_fn(x_ref, b_ref, o_ref):
            vals = x_ref[...] + b_ref[...]
            vals = vals * self.scale_factor
            mins = jnp.min(vals, axis=1, keepdims=True)
            o_ref[...] = mins

        block_m = 1024
        block_c = c
        grid_m = (n * h * w) // block_m

        out_2d = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((n * h * w, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[
                    pl.BlockSpec((block_m, block_c), lambda i: (i, 0)),
                    pl.BlockSpec((block_c,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(x_2d, self.bias)

        out = jnp.reshape(out_2d, (n, h, w, 1))
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
