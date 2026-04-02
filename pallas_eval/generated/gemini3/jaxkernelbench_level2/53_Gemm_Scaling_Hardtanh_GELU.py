```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_fused_kernel(x_ref, w_ref, b_ref, scale_ref, min_ref, max_ref, o_ref):
    # Load blocks into VMEM
    x = x_ref[...]
    w = w_ref[...]
    b = b_ref[...]
    
    # Matmul with f32 accumulator
    acc = jnp.dot(x, w, preferred_element_type=jnp.float32)
    
    # Add bias (using pltpu.repeat to avoid implicit broadcast_to)
    b_rep = pltpu.repeat(b, acc.shape[0], axis=0)
    acc = acc + b_rep
    
    # Scale
    scale_val = scale_ref[0, 0]
    acc = acc * scale_val
    
    # Hardtanh
    min_val = min_ref[0, 0]
    max_val = max_ref[0, 0]
    acc = jnp.clip(acc, min_val, max_val)
    
    # GELU
    sqrt_2_over_pi = jnp.sqrt(2.0 / jnp.pi)
    gelu = acc * 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (acc + 0.044715 * acc**3)))
    
    # Store result
    o_ref[...] = gelu

class Model:
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Block sizes tuned for TPU VMEM and MXU
        BM = 256
        BN = 256
        BK = self.weight.shape[0]
        
        M = x.shape[0]
        N = self.weight.shape[1]
        
        grid_shape = (M // BM, N // BN)
        
        # Ensure inputs are at least 2D for Pallas
        bias_2d = jnp.atleast_2d(self
