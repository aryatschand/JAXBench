```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_forward_kernel(x_ref, w_ref, b_ref, o_ref):
    # Load full blocks into VMEM
    x_val = x_ref[...]
    w_val = w_ref[...]
    
    # Matmul: Mosaic automatically tiles this over K and uses scratch VMEM for the accumulator
    acc = jnp.dot(x_val, w_val)
    
    # Load combined bias (shape: 1, BN) and add (broadcasts to BM, BN)
    b_val = b_ref[...]
    acc = acc + b_val
    
    # Swish
    acc = jax.nn.swish(acc)
    
    # Tanh
    acc = jnp.tanh(acc)
    
    # GELU
    acc = jax.nn.gelu(acc)
    
    # Hardtanh
    acc = jnp.clip(acc, -1.0, 1.0)
    
    # Store result
    o_ref[...] = acc

class Model:
    def __init__(self, in_features, out_features, add_value_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.add_value = jnp.zeros(add_value_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Pre-compute combined bias and reshape to 2D to satisfy TPU constraints
        combined_bias = (self.bias + self.add_value).reshape(1, -1)
        
        BM = 128
        BN = 128
        K = self.weight.shape[0]
        
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        out_shape = (x.shape[0], self.weight.shape[1])
        
        return pl.pallas_call(
            fused_forward_kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, BN), lambda i, j:
