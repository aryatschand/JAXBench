import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = ((padding_h, padding_h), (padding_w, padding_w))
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.use_bias = bias
        
        weight_shape = (in_channels, 1, kernel_size_h, kernel_size_w)
        k = jax.random.PRNGKey(0)
        weight = jax.random.normal(k, weight_shape) * 0.02
        self.weight = jnp.transpose(weight, (2, 3, 1, 0))  # HWIO
        
        if bias:
            self.bias = jnp.zeros(in_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                setattr(self, 'weight', value)
            elif 'bias' in name:
                setattr(self, 'bias', jnp.array(value))
            else:
                setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NHWC

        N, H, W, C = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding[0][0], self.padding[1][0]

        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

        x_padded = jnp.pad(x, ((0,0),(PH,PH),(PW,PW),(0,0)))

        def kernel_fn(x_ref, w_ref, o_ref):
            x_val = x_ref[...]
            w_val = w_ref[...]

            N_, H_, W_, C_ = x_val.shape
            KH_, KW_, _, _ = w_val.shape

            OH_ = (H_ - KH_) // SH + 1
            OW_ = (W_ - KW_) // SW + 1

            out = jnp.zeros((N_, OH_, OW_, C_), dtype=x_val.dtype)

            def body_n(n, out_n):
                def body_h(h, out_h):
                    def body_w(w, out_w):
                        h_start = h * SH
                        w_start = w * SW

                        window = x_val[n, h_start:h_start+KH_, w_start:w_start+KW_, :]
                        # depthwise: multiply per channel
                        val = jnp.sum(window * w_val[:, :, 0, :], axis=(0,1))
                        out_w = out_w.at[w].set(val)
                        return out_w
                    row = jax.lax.fori_loop(0, OW_, body_w, jnp.zeros((OW_, C_), dtype=x_val.dtype))
                    out_h = out_h.at[h].set(row)
                    return out_h
                plane = jax.lax.fori_loop(0, OH_, body_h, jnp.zeros((OH_, OW_, C_), dtype=x_val.dtype))
                out_n = out_n.at[n].set(plane)
                return out_n

            out = jax.lax.fori_loop(0, N_, body_n, out)
            o_ref[...] = out

        y = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((N, OH, OW, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec(x_padded.shape, lambda i: (0,)),
                    pl.BlockSpec(self.weight.shape, lambda i: (0,))
                ],
                out_specs=pl.BlockSpec((N, OH, OW, C), lambda i: (0,))
            ),
        )(x_padded, self.weight)

        if self.use_bias:
            y = y + self.bias

        return jnp.transpose(y, (0, 3, 1, 2))


batch_size = 32
in_channels = 128  
out_channels = 128
kernel_size_h = 3
kernel_size_w = 7
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]
