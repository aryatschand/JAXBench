import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        rng = jax.random.PRNGKey(0)
        weight_shape = (out_channels, in_channels, *kernel_size)
        k1, k2 = jax.random.split(rng)

        weight = jax.random.normal(k1, weight_shape) * (1.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])) ** 0.5
        self.weight = jnp.transpose(weight, (2, 3, 4, 1, 0))

        if bias:
            self.bias = jax.random.normal(k2, (out_channels,))
        else:
            self.bias = None

        self.stride = (stride, stride, stride)
        self.padding = padding
        self.dilation = (dilation, dilation, dilation)
        self.groups = groups
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv3d.weight':
                value = jnp.transpose(jnp.array(value), (2,3,4,1,0))
                setattr(self, 'weight', value)
            elif name == 'conv3d.bias':
                setattr(self, 'bias', jnp.array(value))
            else:
                setattr(self, name.replace('conv3d.', '').replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0,2,3,4,1))  # NDHWC

        N, D, H, W, C = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pad = self.padding
        out_channels = self.weight.shape[-1]

        x_padded = jnp.pad(x, ((0,0),(pad,pad),(pad,pad),(pad,pad),(0,0)))

        out_D = (D + 2*pad - kD)//sD + 1
        out_H = (H + 2*pad - kH)//sH + 1
        out_W = (W + 2*pad - kW)//sW + 1

        total_out = N * out_D * out_H * out_W

        def kernel_fn(x_ref, w_ref, o_ref):
            idx = pl.program_id(axis=0)

            n = idx // (out_D * out_H * out_W)
            rem = idx % (out_D * out_H * out_W)
            od = rem // (out_H * out_W)
            rem2 = rem % (out_H * out_W)
            oh = rem2 // out_W
            ow = rem2 % out_W

            acc = jnp.zeros((out_channels,), dtype=jnp.float32)

            def kd_loop(kd, acc):
                def kh_loop(kh, acc):
                    def kw_loop(kw, acc):
                        d = od * sD + kd
                        h = oh * sH + kh
                        w = ow * sW + kw

                        x_val = x_ref[n, d, h, w, :]
                        w_val = w_ref[kd, kh, kw, :, :]

                        acc = acc + jnp.dot(x_val, w_val)
                        return acc
                    return lax.fori_loop(0, kW, kw_loop, acc)
                return lax.fori_loop(0, kH, kh_loop, acc)

            acc = lax.fori_loop(0, kD, kd_loop, acc)

            if self.bias is not None:
                acc = acc + self.bias

            o_ref[idx, :] = acc

        out_flat = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((total_out, out_channels), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(total_out,),
                in_specs=[
                    pl.BlockSpec(x_padded.shape, lambda i: (0,0,0,0,0)),
                    pl.BlockSpec(self.weight.shape, lambda i: (0,0,0,0,0)),
                ],
                out_specs=pl.BlockSpec((1, out_channels), lambda i: (i, 0)),
            ),
        )(x_padded, self.weight)

        out = jnp.reshape(out_flat, (N, out_D, out_H, out_W, out_channels))

        out = jnp.transpose(out, (0,4,1,2,3))
        return out


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
width = 64
height = 64
depth = 64

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, width, height, depth))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
