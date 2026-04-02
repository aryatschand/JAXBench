import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mul_kernel(A_ref, s_ref, C_ref):
    C_ref[...] = A_ref[...] * s_ref[0, 0]

class Model:
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        pass
    
    def forward(self, A: jnp.ndarray, s: float) -> jnp.ndarray:
        s_arr = jnp.asarray(s, dtype=A.dtype).reshape(1, 1)
        
        block_M = min(A.shape[0], 1024)
        block_N = min(A.shape[1], 1024)
        
        grid_shape = (A.shape[0] // block_M, A.shape[1] // block_N)
        
        return pl.pallas_call(
            mul_kernel,
            out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_M, block_N), lambda i, j: (i, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_M, block_N), lambda i, j: (i, j)),
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
