import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv3d: NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(self.stride, self.stride, self.stride),
            padding=[(self.padding, self.padding)] * 3,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )

        x = x + self.bias.reshape(1, 1, 1, 1, -1)

        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        )

        # NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # reshape to (rows, channels)
        n, c, d, h, w = x.shape
        rows = n * d * h * w
        x2 = jnp.reshape(jnp.transpose(x, (0, 2, 3, 4, 1)), (rows, c))

        def kernel_fn(x_ref, o_ref):
            vals = x_ref[:, :]
            m = jnp.max(vals, axis=1, keepdims=True)
            s = jnp.sum(jnp.exp(vals - m), axis=1, keepdims=True)
            out = m + jnp.log(s)
            o_ref[:, :] = jnp.maximum(out, 0.0)

        block_m = 128
        block_n = c

        grid_m = rows // block_m

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((rows, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(x2)

        out = jnp.reshape(out, (n, d, h, w, 1))
        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        return out


batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
