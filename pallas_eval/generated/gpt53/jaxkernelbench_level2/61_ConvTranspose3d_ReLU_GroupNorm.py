import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def relu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.maximum(x, 0)

def pallas_relu(x):
    # Ensure at least 2D
    orig_shape = x.shape
    x2d = x.reshape(x.shape[0], -1)

    block_m = min(x2d.shape[0], 128)
    block_n = min(x2d.shape[1], 128)

    grid = (x2d.shape[0] // block_m, x2d.shape[1] // block_n)

    out = pl.pallas_call(
        relu_kernel,
        out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        ),
    )(x2d)

    return out.reshape(orig_shape)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.use_bias = bias
        self.eps = 1e-5

        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(
            key, (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        self.conv_transpose_bias = jnp.zeros((out_channels,)) if bias else None

        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        k = self.kernel_size
        pad = k - 1
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
        )

        if self.conv_transpose_bias is not None:
            x = x + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # Pallas ReLU
        x = pallas_relu(x)

        # GroupNorm
        N, C, D, H, W = x.shape
        G = self.groups
        x = x.reshape(N, G, C // G, D, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C, D, H, W)
        x = x * self.group_norm_weight.reshape(1, -1, 1, 1, 1) + \
            self.group_norm_bias.reshape(1, -1, 1, 1, 1)

        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
