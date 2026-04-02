import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        N = A.shape[0]

        BM = 128
        BN = 128
        BK = 128

        def kernel(a_ref, b_ref, o_ref):
            i = pl.program_id(axis=0)
            j = pl.program_id(axis=1)

            row_start = i * BM
            col_start = j * BN

            acc = jnp.zeros((BM, BN), dtype=jnp.float32)

            def body(k, acc):
                a_block = a_ref[:, k * BK:(k + 1) * BK]
                b_block = b_ref[k * BK:(k + 1) * BK, :]
                acc = acc + jnp.matmul(a_block, b_block, preferred_element_type=jnp.float32)
                return acc

            K_tiles = N // BK
            acc = jax.lax.fori_loop(0, K_tiles, body, acc)

            acc = acc.astype(o_ref.dtype)

            rows = row_start + jnp.arange(BM)[:, None]
            cols = col_start + jnp.arange(BN)[None, :]
            mask = rows <= cols

            o_ref[...] = jnp.where(mask, acc, 0)

        grid = (N // BM, N // BN)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((BM, N), lambda i, j: (i, 0)),
                    pl.BlockSpec((N, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(A, B)

    def set_weights(self, weights_dict):
        pass


N = 4096

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jnp.triu(jax.random.uniform(key1, (N, N)))
    B = jnp.triu(jax.random.uniform(key2, (N, N)))
    return [A, B]

def get_init_inputs():
    return []
