import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def conv_kernel(x_ref, w_ref, o_ref, H_out, W_out, O, K):
    bm, bn = o_ref.shape

    i = pl.program_id(0)
    j = pl.program_id(1)

    row_start = i * bm
    col_start = j * bn

    rows = row_start + jnp.arange(bm)
    cols = col_start + jnp.arange(bn)

    N = x_ref.shape[0]
    H = x_ref.shape[1]
    W = x_ref.shape[2]
    C = x_ref.shape[3]

    out = jnp.zeros((bm, bn), dtype=jnp.float32)

    for kh in range(K):
        for kw in range(K):
            for c in range(C):
                r = rows // H_out
                h = rows % H_out

                w_out = cols // O
                o = cols % O

                h_in = h + kh
                w_in = w_out + kw

                x_val = x_ref[r, h_in, w_in, c]
                w_val = w_ref[kh, kw, c, o]

                out += x_val[:, None] * w_val[None, :]

    o_ref[...] = out


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = bias

        k = kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (k, k, in_channels, out_channels))

        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))

        p = self.padding
        if p > 0:
            x = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))

        N, H, W, C = x.shape
        K = self.kernel_size
        O = self.out_channels

        H_out = H - K + 1
        W_out = W - K + 1

        out_shape_2d = (N * H_out, W_out * O)

        bm = 8
        bn = 128

        grid = (out_shape_2d[0] // bm, out_shape_2d[1] // bn)

        out_2d = pl.pallas_call(
            lambda x_ref, w_ref, o_ref: conv_kernel(x_ref, w_ref, o_ref, H_out, W_out, O, K),
            out_shape=jax.ShapeDtypeStruct(out_shape_2d, jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(x.shape, lambda i, j: (0, 0, 0, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i, j: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight)

        out = out_2d.reshape(N, H_out, W_out, O)
        out = jnp.transpose(out, (0, 3, 1, 2))

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
