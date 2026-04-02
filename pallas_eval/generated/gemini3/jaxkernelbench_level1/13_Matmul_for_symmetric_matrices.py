import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref):
    c_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)

class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices using a Pallas TPU kernel.

        Args:
            A (jnp.ndarray): Input matrix A, shape (N, N), symmetric.
            B (jnp.ndarray): Input matrix B, shape (N, N), symmetric.

        Returns:
            jnp.ndarray: Output matrix C, shape (N, N).
        """
        BM = min(A.shape[0], 256)
        BN = min(B.shape[1], 256)
        grid = (A.shape[0] // BM, B.shape[1] // BN)
        
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((A.shape[0], B.shape[1]), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((BM, A.shape[1]), lambda i, j: (i, 0)),
                    pl.BlockSpec((A.shape[1], BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(A, B)

    def set_weights(self, weights_dict):
        """No weights to set for this model."""
        pass

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric arrays A and B.
    """
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    
    A = jax.random.uniform(key1, shape=(N, N))
    A = (A + A.T) / 2  # Ensure symmetry
    
    B = jax.random.uniform(key2, shape=(N, N))
    B = (B + B.T) / 2  # Ensure symmetry
    
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []
