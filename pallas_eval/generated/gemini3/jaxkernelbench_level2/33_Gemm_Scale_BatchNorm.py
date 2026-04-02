```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_epilogue_kernel(x_ref, w_ref, a_ref, b_ref, o_ref):
    BM = x_ref.shape[0]
    BN = w_ref.shape[1]
    K = x_ref.shape[1]
    BK = 256 if K >= 256 else K
    
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    def loop_body(k, acc):
        x_val = x_ref[:, k*BK : (k+1)*BK]
        w_val = w_ref[k*BK : (k+1)*BK, :]
        return acc + jnp.dot(x_val, w_val, preferred_element_type=jnp.float32)
        
    acc = jax.lax.fori_loop(0, K // BK, loop_body, acc)
    
    a_val = a_ref[:, :]
    b_val = b_ref[:, :]
    
    a_val = pltpu.repeat(a_val, BM, axis=0)
    b_val = pltpu.repeat(b_val, BM, axis=0)
    
    res = acc * a_val + b_val
    o_ref[:, :] = res.astype(o_ref.dtype)

class Model:
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        self.gemm_weight = jnp.zeros((out_features, in_features))
        self.gemm_bias = jnp.zeros((out_features,))
        self.scale = jnp.zeros(scale_shape)
        self.bn_weight = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))
        self.bn_running_mean = jnp.zeros((out_features,))
        self.bn_running_var = jnp.ones((out_features,))
        self.eps = eps

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        W_T = self.gemm_weight.T
        inv_std = 1.0 / jnp.sqrt(self.bn_running_var + self.eps)
        
        # Fuse all epilogue element-wise operations into a single scale (A) and
