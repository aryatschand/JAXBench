```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def forward_kernel(x_ref, w_ref, b_ref, c_ref, o_ref):
    x = x_ref[...]
    w = w_ref[...]
    b = b_ref[...]
    c = c_ref[...]
    
    # Matmul with f32 accumulator
    acc = jnp.dot(x, w, preferred_element_type=jnp.float32)
    
    # Add bias
    b_rep = pltpu.repeat(b, acc.shape[0], axis=0)
    acc = acc + b_rep
    
    # Minimum and Subtract constant
    c_rep = pltpu.repeat(c, acc.shape[0], axis=0)
    c_rep = pltpu.repeat(c_rep, acc.shape[1], axis=1)
    
    acc = jnp.minimum(acc, c_rep)
    acc = acc - c_rep
    
    o_ref[...] = acc

class Model:
    def __init__(self, in_features, out_features, constant):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.constant = jnp.array(constant)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Reshape 1D tensors to 2D as required by Pallas
        b_2d = self.bias.reshape(1, -1)
        c_2d = self.constant.reshape(1, 1)
        
        # Block sizes optimized for TPU v6e VMEM (32MB)
        BM = 128
        BN = 256
        BK = 16384
        
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        
        return pl.pallas_call(
            forward_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, BK), lambda i, j: (i, 0)),
                    pl.BlockSpec((BK, BN), lambda i, j: (0, j)),
