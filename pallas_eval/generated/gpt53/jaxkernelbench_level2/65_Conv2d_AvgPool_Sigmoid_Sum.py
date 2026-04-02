import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, H, W = x.shape
        O = self.weight.shape[0]

        def kernel_fn(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[...]  # (1, C, H, W)
            w = w_ref[...]
            b = b_ref[...]

            # NCHW -> NHWC
            x_nhwc = jnp.transpose(x_block, (0, 2, 3, 1))
            kernel = jnp.transpose(w, (2, 3, 1, 0))

            y = jax.lax.conv_general_dilated(
                x_nhwc,
                kernel,
                window_strides=(1, 1),
                padding='VALID',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )
            y = y + b.reshape(1, 1, 1, -1)

            k = self.pool_kernel_size
            y = jax.lax.reduce_window(
                y,
                0.0,
                jax.lax.add,
                window_dimensions=(1, k, k, 1),
                window_strides=(1, k, k, 1),
                padding='VALID'
            )
            y = y / (k * k)

            y = jnp.transpose(y, (0, 3, 1, 2))
            y = jax.nn.sigmoid(y)
            y = jnp.sum(y, axis=(1, 2, 3))  # (1,)

            o_ref[...] = y.reshape(1, 1)

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((N, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N,),
                in_specs=[
                    pl.BlockSpec((1, C, H, W), lambda i: (i, 0, 0, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec((O,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
            ),
        )(x, self.weight, self.bias)

        return out.reshape(N,)

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
