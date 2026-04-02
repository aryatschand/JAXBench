import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.eps = 1e-5

        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(
            key, (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        self.conv_transpose_bias = jnp.zeros((out_channels,)) if bias else None

        self.batch_norm_weight = jnp.ones((out_channels,))
        self.batch_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        batch_size, d_in, h_in, w_in, channels = x.shape
        k = self.kernel_size

        if self.stride > 1:
            d_dilated = d_in + (d_in - 1) * (self.stride - 1)
            h_dilated = h_in + (h_in - 1) * (self.stride - 1)
            w_dilated = w_in + (w_in - 1) * (self.stride - 1)

            x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
            x_dilated = x_dilated.at[:, ::self.stride, ::self.stride, ::self.stride, :].set(x)
            x = x_dilated

        pad = k - 1 - self.padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        if self.conv_transpose_bias is not None:
            x = x + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)

        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NCDHW

        mean = jnp.mean(x, axis=(0, 2, 3, 4), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3, 4), keepdims=True)

        bn_weight = self.batch_norm_weight.reshape(1, -1, 1, 1, 1)
        bn_bias = self.batch_norm_bias.reshape(1, -1, 1, 1, 1)

        spatial_mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)

        # Flatten to 2D for Pallas: (N*C, D*H*W)
        N, C, D, H, W = x.shape
        NC = N * C
        S = D * H * W

        x_flat = x.reshape(NC, S)

        mean_flat = mean.reshape(1, C, 1, 1, 1).repeat(N, axis=0).reshape(NC, 1)
        var_flat = var.reshape(1, C, 1, 1, 1).repeat(N, axis=0).reshape(NC, 1)
        w_flat = bn_weight.reshape(1, C, 1, 1, 1).repeat(N, axis=0).reshape(NC, 1)
        b_flat = bn_bias.reshape(1, C, 1, 1, 1).repeat(N, axis=0).reshape(NC, 1)
        sm_flat = spatial_mean.reshape(NC, 1)

        def kernel_fn(x_ref, mean_ref, var_ref, w_ref, b_ref, sm_ref, o_ref):
            x_block = x_ref[:, :]
            mean_block = mean_ref[:, :]
            var_block = var_ref[:, :]
            w_block = w_ref[:, :]
            b_block = b_ref[:, :]
            sm_block = sm_ref[:, :]

            inv_std = 1.0 / jnp.sqrt(var_block + self.eps)
            y = (x_block - mean_block) * inv_std
            y = y * w_block + b_block
            y = y - sm_block

            o_ref[:, :] = y

        bm = 128
        bn = 128

        grid = (NC // bm, S // bn)

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((NC, S), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
                    pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
                    pl.BlockSpec((bm, 1), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x_flat, mean_flat, var_flat, w_flat, b_flat, sm_flat)

        x = out.reshape(N, C, D, H, W)
        return x


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
