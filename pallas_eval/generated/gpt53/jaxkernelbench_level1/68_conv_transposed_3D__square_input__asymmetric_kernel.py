import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def conv_transpose_kernel(x_ref, w_ref, o_ref):
    x = x_ref[...]
    w = w_ref[...]

    N, D, H, W, Cin = x.shape
    kd, kh, kw, Cout, _ = w.shape

    out = jnp.zeros((N, D + kd - 1, H + kh - 1, W + kw - 1, Cout), dtype=x.dtype)

    for dz in range(kd):
        for dy in range(kh):
            for dx in range(kw):
                x_slice = x
                w_slice = w[dz, dy, dx]  # (Cout, Cin)
                contrib = jnp.einsum('ndhwc,oc->ndhwo', x_slice, w_slice)
                out = out.at[:, dz:dz + D, dy:dy + H, dx:dx + W, :].add(contrib)

    o_ref[...] = out


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias

        kd, kh, kw = self.kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (kd, kh, kw, out_channels, in_channels))

        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 4, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))

        N, D, H, W, Cin = x_ndhwc.shape
        kd, kh, kw = self.kernel_size
        Cout = self.out_channels

        out_shape = (N, D + kd - 1, H + kh - 1, W + kw - 1, Cout)

        block = out_shape
        grid = (1,)

        out = pl.pallas_call(
            conv_transpose_kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, x_ndhwc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(x_ndhwc.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i: (0, 0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
            ),
        )(x_ndhwc, self.weight)

        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)

        return out


batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth, width, height))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]
