```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import gelu

def partial_lse_kernel(x_ref, weight_ref, bias_ref, m_out_ref, s_out_ref):
    B_M = x_ref.shape[0]
    B_N = weight_ref.shape[1]
    B_K = 256
    
    acc = jnp.zeros((B_M, B_N), dtype=jnp.float32)
    
    def loop_body(k, acc):
        x_block = x_ref[:, k*B_K : (k+1)*B_K]
        w_block = weight_ref[k*B_K : (k+1)*B_K, :]
        return acc + jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)
    
    acc = jax.lax.fori_loop(0, x_ref.shape[1] // B_K, loop_body, acc)
    
    bias_val = bias_ref[...]
    bias_val = pltpu.repeat(bias_val, B_M, axis=0)
    acc = acc + bias_val
    
    m = jnp.max(acc, axis=1, keepdims=True)
    s = jnp.sum(jnp.exp(acc - m), axis=1, keepdims=True)
    
    m_padded = jnp.zeros((B_M, 128), dtype=jnp.float32)
    m_padded = m_padded.at[:, :1].set(m)
    m_out_ref[...] = m_padded
    
    s_padded = jnp.zeros((B_M, 128), dtype=jnp.float32)
    s_padded = s_padded.at[:, :1].set(s)
    s_out_ref[...] = s_padded

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        batch_size = x.shape[0]
        in_features = self.weight.shape[0]
