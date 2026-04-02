```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_gelu_kernel(x_ref, x_pool_ref, o_ref):
    # Read blocks from HBM to VMEM
    x_val = x_ref[...]
    x_pool_val = x_pool_ref[...]
    
    # Apply GELU to the pooled values
    gelu_val = jax.nn.gelu(x_pool_val)
    
    # Residual Add (implicit broadcasting of (128, 1) to (128, 512))
    o_ref[...] = gelu_val + x_val

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros(out_features)
        else:
            self.bias = None
        self.subtract = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # -------------------------------------------------------------------
        # Mathematical Simplification:
        # The original operations are:
        #   x_gemm = x @ weight + bias - subtract
        #   x_pool = mean(x_gemm, axis=1)
        #   x_lse = logsumexp(x_pool, axis=1)
        #   out = gelu(x_lse) + x
        #
        # 1. mean(x @ weight + bias - subtract, axis=1) is mathematically equivalent to:
        #    (x @ sum(weight, axis=1) + sum(bias - subtract)) / out_features
        #    This reduces an O(B * in * out) matmul to an O(B * in) matrix-vector product.
        #
        # 2. logsumexp on a (B, 1) array is exactly the identity function.
        #    log(sum(exp(x_pool), axis=1)) = log(exp(x_pool)) = x_pool
        # -------------------------------------------------------------------
        
        out_features = self.weight.shape[1]
        
        # Precompute sums over the output dimension
        weight_sum = jnp.sum(self.weight, axis=1, keepdims=True) # Shape: (in_features, 1)
        
        if self.bias is not None:
            bias_sub_sum = jnp
