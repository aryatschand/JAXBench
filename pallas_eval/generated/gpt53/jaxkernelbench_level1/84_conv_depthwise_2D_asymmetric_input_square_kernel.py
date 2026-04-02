import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.groups = in_channels

        key = jax.random.PRNGKey(0)
        weight_shape = (kernel_size, kernel_size, 1, in_channels)
        self.weight = jax.random.normal(key, weight_shape) * 0.02

        if bias:
            self.bias_param = jnp.zeros((out_channels,))
        else:
            self.bias_param = None

    def set_weights(self, weights_dict):
        w = weights_dict['conv2d.weight']
        w = jnp.transpose(w, (2, 3, 1, 0))
        self.weight = jnp.array(w)
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NHWC

        N, H, W, C = x.shape
        k = self.kernel_size
        stride = self.stride
        pad = self.padding

        H_out = (H + 2 * pad - k) // stride + 1
        W_out = (W + 2 * pad - k) // stride + 1

        x_padded = jnp.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)))

        out_shape_2d = (N * H_out, W_out * C)

        block = (8, 128)
        grid = (out_shape_2d[0] // block[0], out_shape_2d[1] // block[1])

        def kernel(x_ref, w_ref, o_ref):
            pid0 = pl.program_id(axis=0)
            pid1 = pl.program_id(axis=1)

            row_start = pid0 * block[0]
            col_start = pid1 * block[1]

            rows = row_start + jnp.arange(block[0])[:, None]
            cols = col_start + jnp.arange(block[1])[None, :]

            n = rows // H_out
            h = rows % H_out

            w_idx = cols // C
            c = cols % C

            acc = jnp.zeros((block[0], block[1]), dtype=jnp.float32)

            def body(kh, acc):
                def inner(kw, acc_inner):
                    h_in = h * stride + kh
                    w_in = w_idx * stride + kw
                    val = x_ref[n, h_in, w_in, c]
                    weight = w_ref[kh, kw, 0, c]
                    return acc_inner + val * weight
                return jax.lax.fori_loop(0, k, inner, acc)
            acc = jax.lax.fori_loop(0, k, body, acc)

            o_ref[...] = acc

        out_2d = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape_2d, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(x_padded.shape, lambda i, j: (0,0,0,0)),
                    pl.BlockSpec(self.weight.shape, lambda i, j: (0,0,0,0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x_padded, self.weight)

        out = out_2d.reshape(N, H_out, W_out, C)

        if self.bias_param is not None:
            out = out + self.bias_param.reshape(1,1,1,-1)

        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
