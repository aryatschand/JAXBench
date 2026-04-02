import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.use_bias = bias

        kh, kw = self.kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (kh, kw, out_channels // groups, in_channels // groups))

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
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

        B, H, W, IC = x_nhwc.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        ph, pw = self.padding

        pad_h = kh - 1 - ph
        pad_w = kw - 1 - pw

        OH = (H - 1) * sh - 2 * pad_h + dh * (kh - 1) + 1
        OW = (W - 1) * sw - 2 * pad_w + dw * (kw - 1) + 1

        OC = self.out_channels

        # flatten output to 2D (B*OC*OH, OW)
        def kernel(x_ref, w_ref, o_ref):
            row = pl.program_id(axis=0)

            OW_block = o_ref.shape[1]

            b = row // (OC * OH)
            rem = row % (OC * OH)
            oc = rem // OH
            oh = rem % OH

            acc = jnp.zeros((OW_block,), dtype=jnp.float32)

            x_block = x_ref[b, :, :, :]  # (H, W, IC)
            w_block = w_ref[:, :, oc, :]  # (kh, kw, IC)

            for kh_i in range(kh):
                for kw_i in range(kw):
                    ih = oh + pad_h - kh_i * dh
                    cond_h = (ih % sh == 0)
                    ih = ih // sh

                    valid_h = (ih >= 0) & (ih < H) & cond_h

                    for iw in range(W):
                        ow_base = iw * sw - pad_w + kw_i * dw
                        ow_idx = jnp.arange(OW_block)
                        match = ow_idx == ow_base

                        valid = valid_h & match

                        val = jnp.where(valid, x_block[ih, iw, :], 0.0)
                        wv = w_block[kh_i, kw_i, :]

                        acc += jnp.sum(val * wv)

            o_ref[row, :] = acc

        total_rows = B * OC * OH
        block = (1, min(OW, 128))
        grid = (total_rows,)

        out_flat = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((total_rows, OW), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((1, H, W, IC), lambda i: (i, 0, 0, 0)),
                    pl.BlockSpec((kh, kw, OC, IC), lambda i: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x_nhwc, self.weight)

        out = out_flat.reshape(B, OC, OH, OW)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out


batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
height_in = 128
width_in = 256


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, height_in, width_in))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
