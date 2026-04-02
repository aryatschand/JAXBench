import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref):
    res = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
    c_ref[...] = res.astype(a_ref.dtype)

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        m, k = A.shape
        _, n = B.shape
        
        bm = min(m, 512)
        bn = min(n, 512)
        
        grid_shape = (m // bm, n // bn)
        
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
                    pl.BlockSpec((k, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, N))
    B = jax.random.uniform(key2, shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []
