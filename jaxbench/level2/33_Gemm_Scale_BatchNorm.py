"""
JAXBench Level 2 - Gemm_Scale_BatchNorm
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        # Linear layer weights - PyTorch stores as (out_features, in_features)
        self.gemm_weight = jnp.zeros((out_features, in_features))
        self.gemm_bias = jnp.zeros((out_features,))
        
        # Scale parameter
        self.scale = jnp.zeros(scale_shape)
        
        # BatchNorm parameters
        self.bn_weight = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))
        self.bn_running_mean = jnp.zeros((out_features,))
        self.bn_running_var = jnp.ones((out_features,))
        self.eps = eps

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer - PyTorch Linear stores weight as (out_features, in_features)
        # So we need to transpose it for the matmul: x @ weight.T + bias
        x = x @ self.gemm_weight.T + self.gemm_bias
        
        # Scale
        x = x * self.scale
        
        # BatchNorm (using running mean/var for inference mode)
        x_centered = x - self.bn_running_mean
        x_normalized = x_centered / jnp.sqrt(self.bn_running_var + self.eps)
        x = self.bn_weight * x_normalized + self.bn_bias
        
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scale_shape]