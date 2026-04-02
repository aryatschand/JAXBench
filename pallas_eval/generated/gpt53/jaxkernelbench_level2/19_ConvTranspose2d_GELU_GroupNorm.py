import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def gn_kernel(x_ref, gamma_ref, beta_ref, o_ref):
    x = x_ref[...]

    # GELU
    x = gelu(x)

    # compute mean/var per row
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=1, keepdims=True)

    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)

    # apply affine (broadcast)
    o_ref[...] = x_norm * gamma_ref[...] + beta_ref[...]


class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.num_groups = num_groups
        self.eps = 1e-5

        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(
            key, (in_channels, out_channels, kernel_size, kernel_size)
        )
        self.conv_transpose_bias = jnp.zeros((out_channels,))

        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1))

        k = self.kernel_size
        pad = k - 1
        jax_padding = ((pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding=jax_padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
        )

        x = x + self.conv_transpose_bias.reshape(1, 1, 1, -1)

        # NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape
        G = self.num_groups
        group_size = C // G

        # reshape for groupnorm: (N*G, group_size*H*W)
        x_reshaped = x.reshape(N, G, group_size, H, W)
        x_reshaped = x_reshaped.reshape(N * G, group_size * H * W)

        # gamma/beta per group expanded
        gamma = self.group_norm_weight.reshape(G, group_size)
        beta = self.group_norm_bias.reshape(G, group_size)

        gamma = jnp.repeat(gamma, H * W, axis=1).reshape(G, group_size * H * W)
        beta = jnp.repeat(beta, H * W, axis=1).reshape(G, group_size * H * W)

        gamma = jnp.tile(gamma, (N, 1))
        beta = jnp.tile(beta, (N, 1))

        # ensure 2D
        block_m = min(128, x_reshaped.shape[0])
        block_n = min(1024, x_reshaped.shape[1])

        grid = (
            x_reshaped.shape[0] // block_m,
            x_reshaped.shape[1] // block_n,
        )

        def kernel(x_ref, g_ref, b_ref, o_ref):
            gn_kernel(x_ref, g_ref, b_ref, o_ref)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, x_reshaped.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_reshaped, gamma, beta)

        # reshape back
        out = out.reshape(N, G, group_size, H, W)
        out = out.reshape(N, C, H, W)

        return out


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
groups = 8
num_groups = 8


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
