import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, A, B):
        b, i, j, l = A.shape
        k = B.shape[1]

        M = b * i * j
        N = k
        K = l

        A2 = A.reshape(M, K)

        block_m = 128
        block_n = 128
        block_k = 128

        def kernel(a_ref, b_ref, o_ref):
            acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)

            def body(k_idx, acc):
                a_block = a_ref[:, k_idx * block_k:(k_idx + 1) * block_k]
                b_block = b_ref[k_idx * block_k:(k_idx + 1) * block_k, :]
                acc = acc + jnp.dot(a_block, b_block, preferred_element_type=jnp.float32)
                return acc

            num_k = K // block_k
            acc = jax.lax.fori_loop(0, num_k, body, acc)

            o_ref[...] = acc.astype(o_ref.dtype)

        grid_m = M // block_m
        grid_n = N // block_n

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(A2, B)

        return out.reshape(b, i, j, k)


b = 8
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(b, i, j, l))
    B = jax.random.uniform(key2, shape=(l, k))
    return [A, B]

def get_init_inputs():
    return []
