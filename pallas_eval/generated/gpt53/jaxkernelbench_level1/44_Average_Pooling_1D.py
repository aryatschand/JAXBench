import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        n, c, l = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # reshape to (N*C, L)
        x2 = jnp.reshape(x, (n * c, l))

        # pad
        if p > 0:
            x2 = jnp.pad(x2, ((0, 0), (p, p)))

        lp = x2.shape[1]
        l_out = (lp - k) // s + 1

        def kernel_fn(x_ref, o_ref):
            x_block = x_ref[:, :]  # (B, Lp)

            # prefix sum
            cs = jnp.cumsum(x_block, axis=1)
            zero = jnp.zeros((cs.shape[0], 1), cs.dtype)
            cs = jnp.concatenate([zero, cs], axis=1)

            # sliding window average
            out = (cs[:, k:k + l_out] - cs[:, :l_out]) / k

            o_ref[:, :] = out

        block_rows = 128
        block = (block_rows, lp)

        grid = (x2.shape[0] // block_rows,)

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((x2.shape[0], l_out), x2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_rows, l_out), lambda i: (i, 0)),
            ),
        )(x2)

        # reshape back to (N, C, L_out)
        out = jnp.reshape(out, (n, c, l_out))
        return out

    def set_weights(self, weights_dict):
        pass


batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, input_length))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
