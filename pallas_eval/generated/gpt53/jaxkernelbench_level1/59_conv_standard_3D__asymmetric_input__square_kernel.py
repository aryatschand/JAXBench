import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.use_bias = bias

        kernel_shape = (kernel_size, kernel_size, 1, in_channels, out_channels)
        k = 1.0 / (in_channels * kernel_size * kernel_size)
        rng = jax.random.PRNGKey(0)
        self.weight = jax.random.uniform(rng, kernel_shape) * jnp.sqrt(k)

        if bias:
            self.bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv3d.weight':
                value = jnp.transpose(value, (2, 3, 4, 1, 0))
            elif name == 'conv3d.bias':
                value = jnp.array(value)
            setattr(self, name.replace('conv3d.', ''), value)

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NDHWC

        N, D, H, W, C = x.shape
        kD, kH, kW = self.kernel_size, self.kernel_size, 1
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        dD, dH, dW = self.dilation

        out_D = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
        out_H = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        out_W = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        M = N * out_D * out_H * out_W
        OC = self.out_channels

        x_flat = x
        w = self.weight

        def kernel(x_ref, w_ref, o_ref):
            pid = pl.program_id(axis=0)

            row = pid
            n = row // (out_D * out_H * out_W)
            rem = row % (out_D * out_H * out_W)
            od = rem // (out_H * out_W)
            rem2 = rem % (out_H * out_W)
            oh = rem2 // out_W
            ow = rem2 % out_W

            acc = jnp.zeros((OC,), dtype=jnp.float32)

            for kd in range(kD):
                for kh in range(kH):
                    for kw in range(kW):
                        id = od * sD + kd * dD - pD
                        ih = oh * sH + kh * dH - pH
                        iw = ow * sW + kw * dW - pW

                        valid = (id >= 0) & (id < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                        val = jnp.where(
                            valid,
                            x_ref[n, id, ih, iw, :],
                            jnp.zeros((C,), dtype=x_ref.dtype)
                        )

                        w_slice = w_ref[kd, kh, kw, :, :]
                        acc = acc + jnp.dot(val, w_slice)

            o_ref[row, :] = acc.astype(o_ref.dtype)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, OC), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(M,),
                in_specs=[
                    pl.BlockSpec(x.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(w.shape, lambda i: (0, 0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, OC), lambda i: (i, 0)),
            ),
        )(x_flat, w)

        out = out.reshape((N, out_D, out_H, out_W, OC))

        if self.use_bias:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)

        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width, depth))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
