import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
    
    def set_weights(self, weights_dict):
        pass

    def forward(self, x):
        # x: (N, C, L)
        N, C, L = x.shape

        # Convert to NLC
        x = jnp.transpose(x, (0, 2, 1))  # (N, L, C)

        # Padding
        if self.padding > 0:
            x = jnp.pad(
                x,
                ((0, 0), (self.padding, self.padding), (0, 0)),
                mode='constant',
                constant_values=-jnp.inf
            )

        N, Lp, C = x.shape

        # Flatten to 2D (B, L)
        x2d = jnp.reshape(x, (N * C, Lp))

        # Effective kernel
        effective_kernel = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)

        # Output length
        out_L = (Lp - effective_kernel) // self.stride + 1

        B = x2d.shape[0]

        # Ensure divisible tiling
        block_b = 128
        block_l = 128

        grid_b = B // block_b
        grid_l = out_L // block_l

        def kernel(x_ref, o_ref):
            b_idx = pl.program_id(axis=0)
            l_idx = pl.program_id(axis=1)

            x_block = x_ref[:, :]
            out = jnp.full((block_b, block_l), -jnp.inf, dtype=x_block.dtype)

            def body(k, acc):
                idx = l_idx * block_l * self.stride + k * self.dilation
                cols = idx + jnp.arange(block_l) * self.stride
                vals = x_block[:, cols]
                return jnp.maximum(acc, vals)

            out = lax.fori_loop(0, self.kernel_size, body, out)
            o_ref[:, :] = out

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, out_L), x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_b, grid_l),
                in_specs=[
                    pl.BlockSpec((block_b, Lp), lambda i, j: (i, 0)),
                ],
                out_specs=pl.BlockSpec((block_b, block_l), lambda i, j: (i, j)),
            ),
        )(x2d)

        # Reshape back
        out = jnp.reshape(out, (N, C, out_L))
        return out


# Test code
batch_size = 16
features = 64
sequence_length = 8192

kernel_size = 8
stride = 1
padding = 4
dilation = 3

return_indices = False


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, features, sequence_length))
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
