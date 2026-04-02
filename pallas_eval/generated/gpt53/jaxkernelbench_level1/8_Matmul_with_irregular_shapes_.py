import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(a_ref, b_ref, o_ref):
    acc = jnp.zeros((a_ref.shape[0], b_ref.shape[1]), dtype=jnp.float32)

    K = a_ref.shape[1]
    tile_k = 3
    num_tiles = K // tile_k

    def body(i, acc):
        k_start = i * tile_k
        a_tile = a_ref[:, k_start:k_start + tile_k]
        b_tile = b_ref[k_start:k_start + tile_k, :]
        acc = acc + jnp.matmul(a_tile, b_tile, preferred_element_type=jnp.float32)
        return acc

    acc = jax.lax.fori_loop(0, num_tiles, body, acc)
    o_ref[...] = acc.astype(o_ref.dtype)


class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        block_m = 5
        block_n = 1

        grid = (M // block_m, N // block_n)

        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(A, B)


M = 8205
K = 2949
N = 5921


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(M, K))
    B = jax.random.uniform(key2, shape=(K, N))
    return [A, B]


def get_init_inputs():
    return []
