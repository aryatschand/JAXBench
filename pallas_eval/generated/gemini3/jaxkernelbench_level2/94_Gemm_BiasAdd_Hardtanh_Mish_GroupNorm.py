```python
"""
JAXBench Level 2 - Gemm_BiasAdd_Hardtanh_Mish_GroupNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_fused_kernel(batch_block, channel_block, channels_per_group):
    groups_in_block = channel_block // channels_per_group
    
    def kernel(x_ref, bias_ref, gn_w_ref, gn_b_ref, out_ref):
        # Read blocks from HBM to VMEM
        x = x_ref[...]
        bias = bias_ref[...]
        gn_w = gn_w_ref[...]
        gn_b = gn_b_ref[...]
        
        # Reshape to expose the groups for GroupNorm
        x = x.reshape((batch_block, groups_in_block, channels_per_group))
        bias = bias.reshape((1, groups_in_block, channels_per_group))
        gn_w = gn_w.reshape((1, groups_in_block, channels_per_group))
        gn_b = gn_b.reshape((1, groups_in_block, channels_per_group))
        
        # Bias add
        x = x + bias
        
        # Hardtanh
        x = jnp.clip(x, -1.0, 1.0)
        
        # Mish
        sp = jax.nn.softplus(x)
        x = x * jnp.tanh(sp)
        
        # GroupNorm
        mean = jnp.mean(x, axis=2, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=2, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        
        # Scale and shift
        x = x * gn_w + gn_b
        
        # Reshape back and write to HBM
        out_ref[...] = x.reshape((batch_block, channel_block))
        
    return kernel

class Model:
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)
        self.groupnorm_weight = jnp.ones(out_features)
        self.groupnorm_bias = jnp.zeros(out_features)
        self.num_groups = num_groups
        self.num_
