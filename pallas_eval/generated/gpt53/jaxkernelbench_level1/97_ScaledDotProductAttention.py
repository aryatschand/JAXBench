import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, Q, K, V):
        B, H, S, D = Q.shape
        BH = B * H

        Qr = Q.reshape(BH, S, D)
        Kr = K.reshape(BH, S, D)
        Vr = V.reshape(BH, S, D)

        def kernel(q_ref, k_ref, v_ref, o_ref):
            q = q_ref[...]
            k = k_ref[...]
            v = v_ref[...]

            d_k = q.shape[-1]
            scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(d_k)
            weights = jax.nn.softmax(scores, axis=-1)
            out = jnp.matmul(weights, v)

            o_ref[...] = out

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((BH, S, D), Q.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec((BH, S, D), lambda i: (0, 0, 0)),
                    pl.BlockSpec((BH, S, D), lambda i: (0, 0, 0)),
                    pl.BlockSpec((BH, S, D), lambda i: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((BH, S, D), lambda i: (0, 0, 0)),
            ),
        )(Qr, Kr, Vr)

        return out.reshape(B, H, S, D)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    
    Q = jax.random.uniform(key1, shape=(batch_size, num_heads, sequence_length, embedding_dimension))
    K = jax.random.uniform(key2, shape=(batch_size, num_heads, sequence_length, embedding_dimension))
    V = jax.random.uniform(key3, shape=(batch_size, num_heads, sequence_length, embedding_dimension))
    return [Q, K, V]

def get_init_inputs():
    return []
