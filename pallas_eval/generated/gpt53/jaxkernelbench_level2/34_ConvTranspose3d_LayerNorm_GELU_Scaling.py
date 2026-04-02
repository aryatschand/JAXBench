import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor

        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,)) if bias else None

        self.layer_norm_weight = jnp.ones((out_channels,))
        self.layer_norm_bias = jnp.zeros((out_channels,))

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

        N, C, D, H, W = x.shape
        x_2d = x.reshape(N * C * D * H, W)

        def ln_gelu_kernel(x_ref, w_ref, b_ref, o_ref):
            val = x_ref[...]
            mean = jnp.mean(val, axis=1, keepdims=True)
            var = jnp.mean((val - mean) ** 2, axis=1, keepdims=True)
            norm = (val - mean) / jnp.sqrt(var + self.eps)
            norm = norm * w_ref[...] + b_ref[...]
            out = jax.nn.gelu(norm) * self.scaling_factor
            o_ref[...] = out

        rows = x_2d.shape[0]
        block_rows = 128
        grid = (rows // block_rows,)

        out_2d = pl.pallas_call(
            ln_gelu_kernel,
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_rows, W), lambda i: (i, 0)),
                    pl.BlockSpec((W,), lambda i: (0,)),
                    pl.BlockSpec((W,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((block_rows, W), lambda i: (i, 0)),
            ),
        )(x_2d, self.layer_norm_weight, self.layer_norm_bias)

        x = out_2d.reshape(N, C, D, H, W)
        return x


batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
