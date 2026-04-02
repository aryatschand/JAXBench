import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from typing import List

class Model:
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        k = self.kernel_size
        s = self.stride
        p = self.padding
        d = self.dilation

        # NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # compute output shape
        N, D, H, W, C = x.shape
        D_out = (D + 2 * p - d * (k - 1) - 1) // s + 1
        H_out = (H + 2 * p - d * (k - 1) - 1) // s + 1
        W_out = (W + 2 * p - d * (k - 1) - 1) // s + 1

        def kernel_fn(x_ref, o_ref):
            x_val = x_ref[...]

            x_pad = jnp.pad(
                x_val,
                ((0, 0), (p, p), (p, p), (p, p), (0, 0)),
                constant_values=-jnp.inf,
            )

            out = jnp.full((N, D_out, H_out, W_out, C), -jnp.inf, dtype=x_val.dtype)

            for kd in range(k):
                for kh in range(k):
                    for kw in range(k):
                        d_idx = kd * d
                        h_idx = kh * d
                        w_idx = kw * d

                        slice_val = x_pad[
                            :,
                            d_idx : d_idx + s * D_out : s,
                            h_idx : h_idx + s * H_out : s,
                            w_idx : w_idx + s * W_out : s,
                            :
                        ]
                        out = jnp.maximum(out, slice_val)

            o_ref[...] = out

        out_shape = (N, D_out, H_out, W_out, C)

        result = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[pl.BlockSpec(x.shape, lambda i: (0, 0, 0, 0, 0))],
                out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0, 0, 0, 0)),
            ),
        )(x)

        # NDHWC -> NCDHW
        result = jnp.transpose(result, (0, 4, 1, 2, 3))
        return result

    def set_weights(self, weights_dict):
        pass

batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, dim1, dim2, dim3))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
