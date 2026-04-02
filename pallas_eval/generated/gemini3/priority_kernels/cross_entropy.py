` or any markdown is allowed.
"Output ONLY the complete Python file. No explanation, no markdown fences."
"Do NOT include any text before or after the Python code."
I must output raw Python code.

Let's refine the code structure.
```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_8b_cross_entropy',
    'model': 'Llama-3.1-8B',
    'operator': 'fused_cross_entropy',
    'batch_tokens': 4096,
    'hidden_dim': 4096,
    'vocab_size': 128256,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, H, V = CONFIG['batch_tokens'], CONFIG['hidden_dim'], CONFIG['vocab_size']
    hidden = jax.random.normal(k1, (B, H), dtype=dtype)
    weight = jax.random.normal(k2, (H, V), dtype=dtype) * 0.02
    labels = jax.random.randint(k3, (B,), 0, V)
    return hidden, weight, labels

def ce_kernel(hidden_ref, weight_ref, max_ref, sum_exp_ref):
    h = hidden_ref[...]
    w = weight_ref[...]
    
    logits = jnp.dot(h, w, preferred_element_type=jnp.float32)
    
    m = jnp.max(logits, axis=-1, keepdims=True)
    s = jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True)
    
    max_ref[...] = m
    sum_exp_ref[...] = s

def workload(hidden, weight, labels):
    B, H = hidden.shape
    _, V = weight.shape
    
    B_block = 256
    V_block = 256
    
    grid_shape = (B // B_block, V // V_block)
    
    out_shape_max = jax.ShapeDtypeStruct((B, V // V_block), jnp.float32)
    out_shape_sum = jax.ShapeDtypeStruct((B, V // V_block), jnp.float32)
    
    max_out, sum_exp_out = pl.pallas_call(
        ce_kernel,
        out_shape=(out
