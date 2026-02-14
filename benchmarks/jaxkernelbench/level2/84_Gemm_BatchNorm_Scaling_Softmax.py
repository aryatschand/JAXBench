"""
JAXBench Level 2 - Gemm_BatchNorm_Scaling_Softmax
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
import numpy as np

class Model:
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        # Linear layer weights
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        
        # BatchNorm parameters
        self.bn_scale = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))
        self.bn_mean = jnp.zeros((out_features,))
        self.bn_var = jnp.ones((out_features,))
        self.bn_eps = bn_eps
        
        # Scale parameter
        self.scale = jnp.ones(scale_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'gemm.weight':
                # Transpose weight for JAX convention
                setattr(self, 'weight', jnp.array(value.T))
            elif name == 'gemm.bias':
                setattr(self, 'bias', jnp.array(value))
            elif name == 'bn.weight':
                setattr(self, 'bn_scale', jnp.array(value))
            elif name == 'bn.bias':
                setattr(self, 'bn_bias', jnp.array(value))
            elif name == 'bn.running_mean':
                setattr(self, 'bn_mean', jnp.array(value))
            elif name == 'bn.running_var':
                setattr(self, 'bn_var', jnp.array(value))
            elif name == 'scale':
                setattr(self, 'scale', jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.weight) + self.bias

        # BatchNorm
        mean = self.bn_mean
        var = self.bn_var
        x_normalized = (x - mean) / jnp.sqrt(var + self.bn_eps)
        x = self.bn_scale * x_normalized + self.bn_bias
        
        # Scale
        x = self.scale * x
        
        # Softmax
        x = jax.nn.softmax(x, axis=1)
        
        return x

batch_size = 1024
in_features = 8192  
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]