```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_epilogue_kernel(group_size, eps):
    def epilogue_kernel(x_ref, bias_lin_ref, gn_w_ref, gn_b_ref, out_ref):
        # Read blocks from HBM to VMEM
        x = x_ref[...]
        bias_lin = bias_lin_ref[...]
        gn_w = gn_w_ref[...]
        gn_b = gn_b_ref[...]
        
        # 1. Add linear bias
        x = x + bias_lin
        
        # 2. Group Normalization
        # x has shape (block_x, block_y)
        block_x, block_y = x.shape
        
        # Reshape to compute mean and var over each group
        # Shape: (block_x, block_y // group_size, group_size)
        x_reshaped = x.reshape((block_x, block_y // group_size, group_size))
        
        mean = jnp.mean(x_reshaped, axis=2, keepdims=True)
        diff = x_reshaped - mean
        var = jnp.mean(diff * diff, axis=2, keepdims=True)
        
        x_norm = diff / jnp.sqrt(var + eps)
        x_norm = x_norm.reshape((block_x, block_y))
        
        # Apply group norm weight and bias
        x_out = x_norm * gn_w + gn_b
        
        # 3. Min operation along features (axis=1)
        out_ref[...] = jnp.min(x_out, axis=1, keepdims=True)
        
    return epilogue_kernel

class Model:
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        self.weight = jnp.zeros((out_features, in_features))
        self.bias_linear = jnp.zeros(out_features)
        
        self.num_groups = num_groups
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)
        self.eps = 1e-5
        
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # 1. Linear layer matmul (highly optimized in vanilla
