import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, w_ref, o_ref,
            N, D, H, W, Cin,
            Dout, Hout, Wout, Cout,
            kd, kh, kw,
            pd, ph, pw):
    x = x_ref[...]
    w = w_ref[...]

    out = jnp.zeros((N, Dout, Hout, Wout, Cout), dtype=x.dtype)

    for n in range(N):
        for od in range(Dout):
            for oh in range(Hout):
                for ow in range(Wout):
                    for co in range(Cout):
                        acc = 0.0
                        for kd_i in range(kd):
                            id_in = od - (kd - 1) + pd + kd_i
                            if id_in < 0 or id_in >= D:
                                continue
                            for kh_i in range(kh):
                                ih_in = oh - (kh - 1) + ph + kh_i
                                if ih_in < 0 or ih_in >= H:
                                    continue
                                for kw_i in range(kw):
                                    iw_in = ow - (kw - 1) + pw + kw_i
                                    if iw_in < 0 or iw_in >= W:
                                        continue
                                    for ci in range(Cin):
                                        acc += x[n, id_in, ih_in, iw_in, ci] * w[kd_i, kh_i, kw_i, co, ci]
                        out = out.at[n, od, oh, ow, co].set(acc)

    o_ref[...] = out


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.dilation = (dilation, dilation, dilation)
        self.groups = groups
        self.use_bias = bias

        k = kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (k, k, k, out_channels // groups, in_channels // groups))

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
        pd, ph, pw = self.padding

        Dout = D + 2 * (kd - 1 - pd) - (kd - 1)
        Hout = H + 2 * (kh - 1 - ph) - (kh - 1)
        Wout = W + 2 * (kw - 1 - pw) - (kw - 1)

        out = pl.pallas_call(
            lambda x_ref, w_ref, o_ref: _kernel(
                x_ref, w_ref, o_ref,
                N, D, H, W, Cin,
                Dout, Hout, Wout, self.out_channels,
                kd, kh, kw,
                pd, ph, pw
            ),
            out_shape=jax.ShapeDtypeStruct((N, Dout, Hout, Wout, self.out_channels), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec(x_ndhwc.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i: (0, 0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((N, Dout, Hout, Wout, self.out_channels), lambda i: (0, 0, 0, 0, 0)),
            ),
        )(x_ndhwc, self.weight)

        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1, 1)

        return out


batch_size = 8
in_channels = 48
out_channels = 24
kernel_size = 3
depth = 96
height = 96
width = 96


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, depth, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
