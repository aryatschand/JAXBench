```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_forward_kernel(bm, bn, bk, K):
    def forward_kernel(x_ref, w_ref, b_ref, scale_ref, o_ref):
        # Load the entire (bm, K) and (K, bn) blocks into VMEM
        x_val = x_ref[...]
        w_val = w_ref[...]
        
        # Initialize accumulator in scratch VMEM
        acc = jnp.zeros((bm, bn), dtype=jnp.float32)
        
        # Tile over K dimension inside the kernel
        def loop_body(i, acc):
            x_block = jax.lax.dynamic_slice(x_val, (0, i * bk), (bm, bk))
            w_block = jax.lax.dynamic_slice(w_val, (i * bk, 0), (bk, bn))
            return acc + jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)
        
        acc = jax.lax.fori_loop(0, K // bk, loop_body, acc)
        
        # Add bias (broadcasting 1D to 2D)
        b_val = b_ref[...]
        out = acc + b_val
        
        # Apply sigmoid
        sig = jax.nn.sigmoid(out)
        
        # Scale and add residual
        scale = scale_ref[...]
        res = sig * scale + out
        
        # Write back to HBM
        o_ref[...] = res.astype(o_ref.dtype)
        
    return forward_kernel

class Model:
    def __init__(self, input_size, hidden_size, scaling_factor):
        self.weight = jnp.zeros((input_size, hidden_size))
        self.bias = jnp.zeros(hidden_size)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        M, K = x.shape
        _, N = self.weight.shape
        
        # Block sizes optimized for TPU MXU
        bm = 128
        bn = 128
        bk = 128
        
        # Reshape 1D/scalar inputs to 2D to satisfy Pallas constraints
        bias_2d = self.bias.reshape(1, N)
        scale_2d = jnp.array([[self.
