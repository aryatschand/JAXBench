import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        pass
    
    def forward(self, A: jnp.ndarray, s: float) -> jnp.ndarray:
        def kernel(a_ref, s_ref, o_ref):
            a = a_ref[:, :]
            scalar = s_ref[0, 0]
            o_ref[:, :] = a * scalar

        M, N = A.shape
        block = (128, 128)
        grid = (M // block[0], N // block[1])

        s_arr = jnp.array([[s]], dtype=A.dtype)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(A, s_arr)

M = 16384 * 4
N = 4096 * 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    A = jax.random.uniform(key, shape=(M, N))
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []
