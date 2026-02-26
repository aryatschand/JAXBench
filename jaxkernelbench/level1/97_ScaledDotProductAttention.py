"""
JAXBench Level 1 - Task 97: ScaledDotProductAttention
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.150614
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self):
        pass

    def forward(self, Q, K, V):
        # Compute scaled dot product attention
        d_k = Q.shape[-1]
        scores = jnp.matmul(Q, jnp.transpose(K, (0, 1, 3, 2))) / jnp.sqrt(d_k)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attention_weights, V)
        return out

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