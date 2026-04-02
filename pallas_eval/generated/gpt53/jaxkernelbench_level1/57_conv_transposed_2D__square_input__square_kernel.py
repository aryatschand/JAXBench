import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def conv_transpose_kernel(x_ref, w_ref, o_ref,
                          stride, pad, out_pad,
                          kH, kW,
                          in_ch, out_ch):
    x = x_ref[...]        # (BH, BW, IC)
    w = w_ref[...]        # (kH, kW, OC, IC)

    BH, BW, IC = x.shape
    OH = (BH - 1) * stride - 2 * pad + kH + out_pad
    OW = (BW - 1) * stride - 2 * pad + kW + out_pad

    out = jnp.zeros((OH, OW, out_ch), dtype=x.dtype)

    def body_h(i, out):
        def body_w(j, out):
            val = x[i, j]  # (IC,)
            def body_kh(kh, out):
                def body_kw(kw, out):
                    oh = i * stride + kh - pad
                    ow = j * stride + kw - pad

                    valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)

                    contrib = jnp.dot(w[kh, kw], val)  # (OC,)

                    def write(out):
                        return out.at[oh, ow].add(contrib)

                    out = jnp.where(valid, write(out), out)
                    return out
                return jax.lax.fori_loop(0, kW, body_kw, out)
            return jax.lax.fori_loop(0, kH, body_kh, out)
        return jax.lax.fori_loop(0, BW, body_w, out)

    out = jax.lax.fori_loop(0, BH, body_h, out)

    o_ref[...] = out


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.kernel_shape = kernel_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels

        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, kernel_shape)
        if bias:
            self.bias_param = jax.random.normal(key, (out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                self.weight = jnp.array(value)
            elif 'bias' in name:
                self.bias_param = jnp.array(value)

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        weight = jnp.transpose(self.weight, (2, 3, 1, 0))  # (kH, kW, OC, IC)

        N, H, W, C = x.shape

        OH = (H - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        OW = (W - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        def run_single(xi):
            xi_2d = xi  # (H, W, C)

            block_h = H
            block_w = W

            return pl.pallas_call(
                lambda x_ref, w_ref, o_ref: conv_transpose_kernel(
                    x_ref, w_ref, o_ref,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.kernel_size,
                    self.kernel_size,
                    self.in_channels,
                    self.out_channels
                ),
                out_shape=jax.ShapeDtypeStruct((OH, OW, self.out_channels), xi.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(1,),
                    in_specs=[
                        pl.BlockSpec((H, W, C), lambda i: (0, 0, 0)),
                        pl.BlockSpec((self.kernel_size, self.kernel_size, self.out_channels, self.in_channels), lambda i: (0, 0, 0, 0)),
                    ],
                    out_specs=pl.BlockSpec((OH, OW, self.out_channels), lambda i: (0, 0, 0)),
                ),
            )(xi_2d, weight)

        y = jax.vmap(run_single)(x)

        if self.use_bias:
            y = y + self.bias_param.reshape(1, 1, 1, -1)

        # NHWC -> NCHW
        y = jnp.transpose(y, (0, 3, 1, 2))
        return y


batch_size = 8
in_channels = 64
out_channels = 64
kernel_size = 3
height = 1024
width = 1024


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
