import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def conv1d_kernel(x_ref, w_ref, o_ref, stride, padding, dilation, L, C, OC, K):
    pid = pl.program_id(axis=0)
    block_rows = x_ref.shape[0]

    x_block = x_ref[...]  # (B, C)
    w = w_ref[...]        # (K, C, OC)

    # output accumulator
    out = jnp.zeros((block_rows, OC), dtype=jnp.float32)

    def body_k(k, acc):
        shift = k * dilation - padding

        # shift within block (approximate, ignores cross-block)
        if shift == 0:
            x_shifted = x_block
        else:
            pad_top = jnp.maximum(shift, 0)
            pad_bot = jnp.maximum(-shift, 0)

            x_padded = jnp.pad(x_block, ((pad_top, pad_bot), (0, 0)))
            x_shifted = x_padded[pad_bot:pad_bot + block_rows, :]

        w_k = w[k, :, :]  # (C, OC)
        acc = acc + jnp.dot(x_shifted, w_k)
        return acc

    out = jax.lax.fori_loop(0, K, body_k, out)

    o_ref[...] = out.astype(o_ref.dtype)


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        kernel_shape = (out_channels, in_channels // groups, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias_param = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, L = x.shape

        # NCL -> (N*L, C)
        x_flat = jnp.transpose(x, (0, 2, 1)).reshape(N * L, C)

        # weight -> (K, C, OC)
        w = jnp.transpose(self.weight, (2, 1, 0))

        block_rows = 128
        grid = (x_flat.shape[0] // block_rows,)

        out_flat = pl.pallas_call(
            conv1d_kernel,
            out_shape=jax.ShapeDtypeStruct((x_flat.shape[0], self.out_channels), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_rows, C), lambda i: (i, 0)),
                    pl.BlockSpec(w.shape, lambda i: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((block_rows, self.out_channels), lambda i: (i, 0)),
            ),
        )(x_flat, w, self.stride, self.padding, self.dilation, L, C, self.out_channels, self.kernel_size)

        # reshape back: (N, L, OC) -> NCL
        y = out_flat.reshape(N, L, self.out_channels)
        y = jnp.transpose(y, (0, 2, 1))

        if self.use_bias:
            y = y + self.bias_param.reshape(1, -1, 1)

        return y


def get_inputs():
    key = jax.random.PRNGKey(0)
    batch_size = 32
    in_channels = 64
    length = 131072
    x = jax.random.uniform(key, shape=(batch_size, in_channels, length))
    return [x]


def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]
