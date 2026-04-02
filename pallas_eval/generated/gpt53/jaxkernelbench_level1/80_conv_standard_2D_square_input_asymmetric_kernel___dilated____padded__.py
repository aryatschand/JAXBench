import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        kernel_shape = (kernel_size[0], kernel_size[1], in_channels, out_channels)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias_param = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        weight = weights_dict['conv2d.weight']
        self.weight = jnp.transpose(jnp.array(weight), (2, 3, 1, 0))
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NHWC

        N, H, W, C = x.shape
        kH, kW = self.kernel_size
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        stride = self.stride

        H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride + 1
        W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride + 1

        x_padded = jnp.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)))

        # reshape to fuse batch + height for 2D grid
        x_reshaped = x_padded.reshape(N * x_padded.shape[1], x_padded.shape[2], C)
        out_shape = (N * H_out, W_out, self.out_channels)

        bh = 128
        bw = 128

        bh = min(bh, out_shape[0])
        bw = min(bw, out_shape[1])

        grid = (out_shape[0] // bh, out_shape[1] // bw)

        def kernel(x_ref, w_ref, o_ref):
            x_block = x_ref[:, :, :]
            w = w_ref[:, :, :, :]

            bh_, bw_, _ = o_ref.shape

            acc = jnp.zeros((bh_, bw_, self.out_channels), dtype=jnp.float32)

            for kh in range(kH):
                for kw in range(kW):
                    ih = kh * dil_h
                    iw = kw * dil_w

                    x_slice = x_block[ih:ih+bh_, iw:iw+bw_, :]

                    for c in range(self.in_channels):
                        acc += x_slice[:, :, c:c+1] * w[kh, kw, c:c+1, :]

            o_ref[...] = acc.astype(o_ref.dtype)

        in_block_h = bh + (kH - 1) * dil_h
        in_block_w = bw + (kW - 1) * dil_w

        y = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((in_block_h, in_block_w, C), lambda i, j: (i, j, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i, j: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((bh, bw, self.out_channels), lambda i, j: (i, j, 0)),
            ),
        )(x_reshaped, self.weight)

        if self.use_bias:
            y = y + self.bias_param[None, None, :]

        y = y.reshape(N, H_out, W_out, self.out_channels)
        y = jnp.transpose(y, (0, 3, 1, 2))
        return y


# Test parameters
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
