"""
JAXBench Level 2 - Conv2d_Min_Tanh_Tanh
Pallas TPU kernel version
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def kernel_fn(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]        # (1, H+2, W+2, Cin)
    w = w_ref[...]        # (Kh, Kw, Cin, Cout)
    b = b_ref[...]        # (Cout,)

    H = o_ref.shape[1]
    W = o_ref.shape[2]
    Kh, Kw = w.shape[0], w.shape[1]
    Cout = w.shape[3]

    def body_hw(idx, out):
        i = idx // W
        j = idx % W

        def body_cout(c, acc_min):
            val = 0.0

            def body_kh(kh, val_inner):
                def body_kw(kw, val_inner2):
                    x_slice = x[0, i + kh, j + kw, :]
                    w_slice = w[kh, kw, :, c]
                    return val_inner2 + jnp.sum(x_slice * w_slice)
                return jax.lax.fori_loop(0, Kw, body_kw, val_inner)

            val = jax.lax.fori_loop(0, Kh, body_kh, val)
            val = val + b[c]
            acc_min = jnp.minimum(acc_min, val)
            return acc_min

        min_val = jax.lax.fori_loop(0, Cout, body_cout, jnp.inf)
        min_val = jnp.tanh(min_val)
        min_val = jnp.tanh(min_val)

        out = out.at[0, i, j, 0].set(min_val)
        return out

    out = jnp.zeros_like(o_ref[...])
    out = jax.lax.fori_loop(0, H * W, body_hw, out)
    o_ref[...] = out


class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Pad for VALID conv handling inside kernel
        k = self.weight.shape[2]
        x = jnp.pad(x, ((0,0),(0,0),(0,0),(0,0)))

        # HWIO
        w = jnp.transpose(self.weight, (2, 3, 1, 0))
        b = self.bias

        N, H, W, C = x.shape
        Kh = Kw = k
        Ho = H - Kh + 1
        Wo = W - Kw + 1

        out_shape = (N, Ho, Wo, 1)

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N,),
                in_specs=[
                    pl.BlockSpec((1, H, W, C), lambda n: (n, 0, 0, 0)),
                    pl.BlockSpec((Kh, Kw, C, w.shape[3]), lambda n: (0, 0, 0, 0)),
                    pl.BlockSpec((w.shape[3],), lambda n: (0,)),
                ],
                out_specs=pl.BlockSpec((1, Ho, Wo, 1), lambda n: (n, 0, 0, 0)),
            ),
        )(x, w, b)


batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
