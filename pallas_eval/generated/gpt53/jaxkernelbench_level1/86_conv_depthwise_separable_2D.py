import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        depthwise_shape = (kernel_size, kernel_size, 1, in_channels)
        self.depthwise_weight = jnp.zeros(depthwise_shape)
        self.depthwise_bias = jnp.zeros((in_channels,)) if bias else None

        pointwise_shape = (1, 1, in_channels, out_channels)
        self.pointwise_weight = jnp.zeros(pointwise_shape)
        self.pointwise_bias = jnp.zeros((out_channels,)) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'depthwise.weight' in name:
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                self.depthwise_weight = value
            elif 'pointwise.weight' in name:
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                self.pointwise_weight = value
            elif 'depthwise.bias' in name:
                self.depthwise_bias = jnp.array(value)
            elif 'pointwise.bias' in name:
                self.pointwise_bias = jnp.array(value)

    def forward(self, x):
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

        N, H, W, C = x_nhwc.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        D = self.dilation
        OC = self.out_channels

        H_out = (H + 2 * P - D * (K - 1) - 1) // S + 1
        W_out = (W + 2 * P - D * (K - 1) - 1) // S + 1

        def kernel_fn(x_ref, dw_ref, pw_ref, db_ref, pb_ref, o_ref):
            x_val = x_ref[...]
            dw = dw_ref[...]
            pw = pw_ref[...]

            if db_ref.shape[0] > 0:
                db = db_ref[...]
            else:
                db = jnp.zeros((C,), dtype=x_val.dtype)

            if pb_ref.shape[0] > 0:
                pb = pb_ref[...]
            else:
                pb = jnp.zeros((OC,), dtype=x_val.dtype)

            out = jnp.zeros((N, H_out, W_out, OC), dtype=x_val.dtype)

            for n in range(N):
                for h in range(H_out):
                    for w in range(W_out):
                        dw_acc = jnp.zeros((C,), dtype=x_val.dtype)
                        for kh in range(K):
                            for kw in range(K):
                                ih = h * S + kh * D - P
                                iw = w * S + kw * D - P
                                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                                x_slice = jnp.where(valid, x_val[n, ih, iw, :], 0.0)
                                w_slice = dw[kh, kw, 0, :]
                                dw_acc = dw_acc + x_slice * w_slice
                        dw_acc = dw_acc + db

                        pw_acc = jnp.dot(dw_acc, pw[0, 0, :, :]) + pb
                        out = out.at[n, h, w, :].set(pw_acc)

            o_ref[...] = out

        out_shape = (N, H_out, W_out, OC)

        result = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_shape, x_nhwc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec(x_nhwc.shape, lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec(self.depthwise_weight.shape, lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec(self.pointwise_weight.shape, lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec((0,) if self.depthwise_bias is None else self.depthwise_bias.shape, lambda i: (0,)),
                    pl.BlockSpec((0,) if self.pointwise_bias is None else self.pointwise_bias.shape, lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0)),
            ),
        )(x_nhwc,
          self.depthwise_weight,
          self.pointwise_weight,
          jnp.zeros((0,)) if self.depthwise_bias is None else self.depthwise_bias,
          jnp.zeros((0,)) if self.pointwise_bias is None else self.pointwise_bias)

        return jnp.transpose(result, (0, 3, 1, 2))


# Test parameters
batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
