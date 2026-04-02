"""
JAXBench Level 1 - Task 13: Matmul_for_symmetric_matrices
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.125745
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    bm, bn = o_ref.shape
    N = a_ref.shape[1]
    bk = 128

    acc = jnp.zeros((bm, bn), dtype=jnp.float32)

    def body(k, acc):
        k0 = k * bk
        a = a_ref[:, k0:k0 + bk]
        b = b_ref[k0:k0 + bk, :]
        acc = acc + jnp.matmul(a, b, preferred_element_type=jnp.float32)
        return acc

    num_k = N // bk
    acc = jax.lax.fori_loop(0, num_k, body, acc)

    o_ref[...] = acc.astype(o_ref.dtype)


class Model:
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        pass
    
    def forward(self, A, B):
        N = A.shape[0]
        bm = 128
        bn = 128

        grid = (N // bm, N // bn)

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((N, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, N), lambda i, j: (i, 0)),
                    pl.BlockSpec((N, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
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
    A = (A + A.T) / 2
    
    B = jax.random.uniform(key2, shape=(N, N))
    B = (B + B.T) / 2
    
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []
