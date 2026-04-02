```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def pool_kernel(x_curr_ref, x_next_ref, W1_ref, W2_ref, o_ref):
    # x_curr_ref: (block_NC, step)
    # x_next_ref: (block_NC, step)
    # W1_ref: (step, block_L_out)
    # W2_ref: (step, block_L_out)
    # o_ref: (block_NC, block_L_out)
    
    res1 = jax.lax.dot_general(
        x_curr_ref[...], W1_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32
    )
    res2 = jax.lax.dot_general(
        x_next_ref[...], W2_ref[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32
    )
    o_ref[...] = res1 + res2

class Model:
    """
    Simple model that performs 1D Average Pooling using a Pallas TPU kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        N, C, L = x.shape
        NC = N * C
        
        L_p = L + 2 * self.padding
        L_out = (L_p - self.kernel_size) // self.stride + 1
        
        block_NC = min(NC, 64)
        if block_NC < 8:
            block_NC = 8
            
        pad_NC = (block_NC - (NC % block_NC)) % block_NC
        NC_padded = NC + pad_NC
        num_blocks_NC = NC_padded // block_NC
        
        block_L_out = 512
        while block_L_out * self.stride + self.stride < self.kernel_size:
            block_L_out *= 2
            
        step = block_L_out * self.stride
        num_blocks_L = (L_out + block_L_out - 1) // block_L_out
        
        total_len_required = (num_blocks_L + 1) * step
        pad_left = self.padding
        pad_right = max(0,
