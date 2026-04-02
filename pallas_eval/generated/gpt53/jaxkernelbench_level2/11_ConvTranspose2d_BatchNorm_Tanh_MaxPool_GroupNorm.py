import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bn_tanh_kernel(x_ref, mean_ref, var_ref, gamma_ref, beta_ref, o_ref, eps):
    x = x_ref[:, :]
    mean = mean_ref[0, :]
    var = var_ref[0, :]
    gamma = gamma_ref[0, :]
    beta = beta_ref[0, :]

    y = (x - mean) / jnp.sqrt(var + eps)
    y = y * gamma + beta
    y = jnp.tanh(y)

    o_ref[:, :] = y

def bn_tanh_pallas(x, mean, var, gamma, beta, eps):
    n, c = x.shape
    block = (min(n, 1024), min(c, 128))
    grid = (n // block[0], c // block[1])

    return pl.pallas_call(
        lambda xr, mr, vr, gr, br, or_: bn_tanh_kernel(xr, mr, vr, gr, br, or_, eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i, j: (i, j)),
                pl.BlockSpec((1, block[1]), lambda i, j: (0, j)),
                pl.BlockSpec((1, block[1]), lambda i, j: (0, j)),
                pl.BlockSpec((1, block[1]), lambda i, j: (0, j)),
                pl.BlockSpec((1, block[1]), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x, mean, var, gamma, beta)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_groups = num_groups
        self.eps = 1e-5

        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.batch_norm_weight = jnp.ones((out_channels,))
        self.batch_norm_bias = jnp.zeros((out_channels,))
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))

        weight = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))
        k = self.kernel_size
        pad = k - 1 - self.padding
        jax_padding = ((pad, pad), (pad, pad))

        x = lax.conv_transpose(
            x, weight,
            strides=(self.stride, self.stride),
            padding=jax_padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        x = x + self.conv_transpose_bias.reshape(1, 1, 1, -1)

        x = jnp.transpose(x, (0, 3, 1, 2))

        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)

        N, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)

        mean_flat = mean.reshape(1, C)
        var_flat = var.reshape(1, C)
        gamma = self.batch_norm_weight.reshape(1, C)
        beta = self.batch_norm_bias.reshape(1, C)

        x_flat = bn_tanh_pallas(x_flat, mean_flat, var_flat, gamma, beta, self.eps)

        x = x_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        x = jnp.transpose(x, (0, 2, 3, 1))
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
        x = jnp.transpose(x, (0, 3, 1, 2))

        N, C, H, W = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        x = x * self.group_norm_weight.reshape(1, -1, 1, 1) + self.group_norm_bias.reshape(1, -1, 1, 1)

        return x

batch_size = 512
in_channels = 64
out_channels = 128
height = width = 32
kernel_size = 5
stride = 1
padding = 1
groups = 8
num_groups = 8

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
