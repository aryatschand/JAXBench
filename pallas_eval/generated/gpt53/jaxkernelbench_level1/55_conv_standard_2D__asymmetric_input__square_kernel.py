import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
        self.weight = None
        if bias:
            self.bias_param = None

    def set_weights(self, weights_dict):
        weight = weights_dict['conv2d.weight']
        self.weight = jnp.transpose(jnp.array(weight), (2, 3, 1, 0))
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))

        N, H, W, C = x.shape
        KH, KW, _, OC = self.weight.shape

        pad = self.padding
        stride = self.stride

        H_out = (H + 2 * pad - KH) // stride + 1
        W_out = (W + 2 * pad - KW) // stride + 1

        x_padded = jnp.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)))

        block = (1, 128, 128, 32)

        def kernel(x_ref, w_ref, o_ref):
            x_block = x_ref[...]
            w = w_ref[...]

            acc = jnp.zeros_like(o_ref[...])

            for kh in range(KH):
                for kw in range(KW):
                    h_idx = kh
                    w_idx = kw
                    x_slice = x_block[:, h_idx:h_idx+128, w_idx:w_idx+128, :]
                    w_slice = w[kh, kw, :, :]
                    acc += jnp.einsum('nhwc,co->nhwo', x_slice, w_slice)

            o_ref[...] = acc

        grid = (
            N // block[0],
            H_out // block[1],
            W_out // block[2],
            OC // block[3],
        )

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, H_out, W_out, OC), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda n, h, w, c: (n, h, w, 0)),
                    pl.BlockSpec((KH, KW, C, OC), lambda n, h, w, c: (0, 0, 0, c)),
                ],
                out_specs=pl.BlockSpec(block, lambda n, h, w, c: (n, h, w, c)),
            ),
        )(x_padded, self.weight)

        if self.use_bias:
            out = out + self.bias_param.reshape(1, 1, 1, -1)

        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 8
height = 512
width = 1024
in_channels = 64
out_channels = 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
