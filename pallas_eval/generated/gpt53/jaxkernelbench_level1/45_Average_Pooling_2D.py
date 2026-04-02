import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def set_weights(self, weights_dict):
        pass

    def forward(self, x):
        N, C, H, W = x.shape

        # NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        if self.padding > 0:
            x = jnp.pad(
                x,
                ((0, 0), (self.padding, self.padding),
                 (self.padding, self.padding), (0, 0)),
            )

        H_p, W_p = x.shape[1], x.shape[2]

        k = self.kernel_size
        s = self.stride

        H_out = (H_p - k) // s + 1
        W_out = (W_p - k) // s + 1

        # Flatten to 2D: rows = N * H_out, cols = W_out * C
        x_flat = x
        out_shape = (N * H_out, W_out * C)

        def kernel(x_ref, o_ref):
            pid = pl.program_id(axis=0)

            row = pid  # maps to (n, h_out)
            n = row // H_out
            h_out = row % H_out

            h_start = h_out * s

            x_block = x_ref[n, :, :, :]  # (H_p, W_p, C)

            # process full row (W_out * C)
            out_row = jnp.zeros((W_out * C,), dtype=x_block.dtype)

            for w_out in range(W_out):
                w_start = w_out * s

                window = x_block[
                    h_start:h_start + k,
                    w_start:w_start + k,
                    :
                ]  # (k, k, C)

                val = jnp.sum(window, axis=(0, 1)) / (k * k)

                start = w_out * C
                out_row = out_row.at[start:start + C].set(val)

            o_ref[row, :] = out_row

        grid = (N * H_out,)
        block = (1, out_shape[1])

        out_flat = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec((1, H_p, W_p, C), lambda i: (i // H_out, 0, 0, 0))],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x)

        out = out_flat.reshape(N, H_out, W_out, C)
        out = jnp.transpose(out, (0, 3, 1, 2))

        return out


batch_size = 4
channels = 32
height = 512
width = 512
kernel_size = 11


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, channels, height, width))
    return [x]


def get_init_inputs():
    return [kernel_size]
