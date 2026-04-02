import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        N, C, H, W = x.shape

        # NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

        k = self.kernel_size
        s = self.stride
        p = self.padding
        d = self.dilation

        H_out = (H + 2 * p - d * (k - 1) - 1) // s + 1
        W_out = (W + 2 * p - d * (k - 1) - 1) // s + 1

        # Pad input manually
        x_pad = jnp.pad(
            x_nhwc,
            ((0, 0), (p, p), (p, p), (0, 0)),
            constant_values=-jnp.inf,
        )

        total_rows = N * H_out * C
        total_cols = W_out

        x_flat = x_pad
        out_shape = (total_rows, total_cols)

        def kernel(x_ref, o_ref):
            row_id = pl.program_id(axis=0)
            col_id = pl.program_id(axis=1)

            row_start = row_id * BLOCK_M
            col_start = col_id * BLOCK_N

            rows = jnp.arange(row_start, row_start + BLOCK_M)
            cols = jnp.arange(col_start, col_start + BLOCK_N)

            rows = rows[:, None]
            cols = cols[None, :]

            n = rows // (H_out * C)
            rem = rows % (H_out * C)
            h_out = rem // C
            c = rem % C

            w_out = cols

            h_in = h_out * s
            w_in = w_out * s

            max_val = jnp.full((BLOCK_M, BLOCK_N), -jnp.inf, dtype=x_ref.dtype)

            def body_i(i, acc):
                def body_j(j, acc2):
                    h_idx = h_in + i * d
                    w_idx = w_in + j * d

                    val = x_ref[n, h_idx, w_idx, c]
                    return jnp.maximum(acc2, val)

                return jax.lax.fori_loop(0, k, body_j, acc)

            max_val = jax.lax.fori_loop(0, k, body_i, max_val)

            o_ref[...] = max_val

        BLOCK_M = 128
        BLOCK_N = 128

        grid_m = (total_rows + BLOCK_M - 1) // BLOCK_M
        grid_n = (total_cols + BLOCK_N - 1) // BLOCK_N

        out_flat = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec(x_flat.shape, lambda i, j: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
            ),
        )(x_flat)

        out = out_flat.reshape(N, H_out, W_out, C)
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, height, width))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
