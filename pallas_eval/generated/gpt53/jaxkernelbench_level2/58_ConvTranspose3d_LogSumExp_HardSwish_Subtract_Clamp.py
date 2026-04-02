import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))

        kernel_size = self.weight.shape[2]
        pad_size = kernel_size - 2
        padding = [(pad_size, pad_size)] * 3

        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(2, 2, 2),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NCDHW

        N, C, D, H, W = x.shape

        def kernel_fn(x_ref, bias_ref, o_ref):
            vals = x_ref[...]  # (1, C, 1, 1, block_w)

            m = jnp.max(vals, axis=1, keepdims=True)
            lse = m + jnp.log(jnp.sum(jnp.exp(vals - m), axis=1, keepdims=True))

            hs = lse * jax.nn.sigmoid(lse + 3.0) / 6.0
            out = hs - bias_ref[...]

            out = jnp.clip(out, -1.0, 1.0)
            o_ref[...] = out

        block_w = 32
        grid = (N, D, H, W // block_w)

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((N, 1, D, H, W), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((1, C, 1, 1, block_w), lambda n, d, h, w: (n, 0, d, h, w)),
                    pl.BlockSpec((1, 1, 1, 1), lambda n, d, h, w: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, 1, 1, block_w), lambda n, d, h, w: (n, 0, d, h, w)),
            ),
        )(x, self.bias)

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
