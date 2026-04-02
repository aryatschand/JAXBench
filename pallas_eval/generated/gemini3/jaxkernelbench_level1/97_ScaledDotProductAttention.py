import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self):
        pass

    def forward(self, Q, K, V):
        B, H, S, D = Q.shape
        
        def attn_kernel(q_ref, k_ref, v_ref, o_ref):
            # Load blocks into VMEM
            q = q_ref[0, 0, :, :]
            k = k_ref[0, 0, :, :]
            v = v_ref[0, 0, :, :]
            
            d_k = q.shape[-1]
            
            # Compute scaled dot-product attention
            scores = jnp.matmul(q, k.T, preferred_element_type=jnp.float32) / jnp.sqrt(jnp.float32(d_k))
            
            # Softmax
            scores_max = jnp.max(scores, axis=-1, keepdims=True)
            exp_scores = jnp.exp(scores - scores_max)
            attn = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
            
            # Output projection
            out = jnp.matmul(attn, v, preferred_element_type=jnp.float32)
            
            # Store result
            o_ref[0, 0, :, :] = out

        return pl.pallas_call(
            attn_kernel,
            out_shape=jax.ShapeDtypeStruct(Q.shape, Q.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(B, H),
                in_specs=[
                    pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
                    pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
                    pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
            ),
        )(Q, K, V)

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
