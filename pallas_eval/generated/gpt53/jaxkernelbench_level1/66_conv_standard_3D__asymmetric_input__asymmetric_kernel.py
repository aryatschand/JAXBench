import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.groups = groups
        self.use_bias = bias

        kd, kh, kw = self.kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (kd, kh, kw, in_channels // groups, out_channels))

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

        N, D, H, W, C = x_ndhwc.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        D_out = (D + 2 * pd - kd) // sd + 1
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1

        x_pad = jnp.pad(x_ndhwc,
                        ((0, 0), (pd, pd), (ph, ph), (pw, pw), (0, 0)))

        P = N * D_out * H_out * W_out

        def kernel(x_ref, w_ref, o_ref):
            p_idx = pl.program_id(axis=0)
            c_idx = pl.program_id(axis=1)

            p_block = x_ref.shape[0]
            c_block = w_ref.shape[-1]

            p = p_idx * p_block + jnp.arange(p_block)[:, None]
            c = c_idx * c_block + jnp.arange(c_block)[None, :]

            n = p // (D_out * H_out * W_out)
            rem = p % (D_out * H_out * W_out)
            d = rem // (H_out * W_out)
            rem2 = rem % (H_out * W_out)
            h = rem2 // W_out
            w = rem2 % W_out

            acc = jnp.zeros((p_block, c_block), dtype=jnp.float32)

            for kz in range(kd):
                for ky in range(kh):
                    for kx in range(kw):
                        in_d = d * sd + kz
                        in_h = h * sh + ky
                        in_w = w * sw + kx

                        x_val = x_ref[n, in_d, in_h, in_w, :]
                        w_val = w_ref[kz, ky, kx, :, c]

                        acc += jnp.sum(x_val[..., None] * w_val, axis=1)

            o_ref[...] = acc

        block_p = 128
        block_c = 128

        grid_p = (P + block_p - 1) // block_p
        grid_c = (self.out_channels + block_c - 1) // block_c

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((P, self.out_channels), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_p, grid_c),
                in_specs=[
                    pl.BlockSpec((block_p, 1, 1, 1, C),
                                 lambda i, j: (i, 0, 0, 0, 0)),
                    pl.BlockSpec((kd, kh, kw, C, block_c),
                                 lambda i, j: (0, 0, 0, 0, j)),
                ],
                out_specs=pl.BlockSpec((block_p, block_c),
                                       lambda i, j: (i, j)),
            ),
        )(x_pad, self.weight)

        out = out.reshape(N, D_out, H_out, W_out, self.out_channels)
        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)

        return out


batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 128
width = 128


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
