```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_relu_div_kernel(x_ref, w_ref, b_ref, div_ref, o_ref):
    BM = x_ref.shape[0]
    BN = w_ref.shape[1]
    K = x_ref.shape[1]
    
    # Tile over K with accumulator in scratch VMEM
    BK = min(256, K)
    num_k = K // BK
    
    x_val = x_ref[...]
    w_val = w_ref[...]
    
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    def body_fn(i, acc):
        x_chunk = jax.lax.dynamic_slice(x_val, (0, i * BK), (BM, BK))
        w_chunk = jax.lax.dynamic_slice(w_val, (i * BK, 0), (BK, BN))
        return acc + jnp.dot(x_chunk, w_chunk, preferred_element_type=jnp.float32)
    
    acc = jax.lax.fori_loop(0, num_k, body_fn, acc)
    
    # Add bias
    b_val = b_ref[...]
    b_val = pltpu.repeat(b_val, BM, axis=0)
    acc = acc + b_val
    
    # ReLU
    acc = jnp.maximum(acc, 0.0)
    
    # Divide
    div_val = div_ref[...]
    div_val = pltpu.repeat(div_val, BM, axis=0)
    div_val = pltpu.repeat(div_val, BN, axis=1)
    acc = acc / div_val
    
    o_ref[...] = acc.astype(o_ref.dtype)

class Model:
    def __init__(self, in_features, out_features, divisor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Reshape 1D/scalar inputs to 2D as required by Pallas
        bias_2d = self.bias.reshape(1, -1)
        divisor_2d = jnp.array([[self.divisor]], dtype=x.
