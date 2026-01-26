"""
JAXBench Level 2 - Matmul_BatchNorm_BiasAdd_Divide_Swish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        # Linear layer weights
        self.weight = jnp.zeros((in_features, out_features))
        self.linear_bias = jnp.zeros(out_features)
        
        # BatchNorm parameters
        self.bn_scale = jnp.ones(out_features)
        self.bn_bias = jnp.zeros(out_features)
        self.bn_mean = jnp.zeros(out_features)
        self.bn_var = jnp.ones(out_features)
        self.bn_eps = bn_eps
        
        # Additional bias
        self.bias = jnp.zeros(bias_shape)
        self.divide_value = divide_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'matmul.weight':
                setattr(self, 'weight', jnp.array(value.T))  # Transpose for JAX convention
            elif name == 'matmul.bias':
                setattr(self, 'linear_bias', jnp.array(value))
            elif name == 'bn.weight':
                setattr(self, 'bn_scale', jnp.array(value))
            elif name == 'bn.bias':
                setattr(self, 'bn_bias', jnp.array(value))
            elif name == 'bn.running_mean':
                setattr(self, 'bn_mean', jnp.array(value))
            elif name == 'bn.running_var':
                setattr(self, 'bn_var', jnp.array(value))
            elif name == 'bias':
                setattr(self, 'bias', jnp.array(value))

    def forward(self, x):
        # Linear layer
        x = jnp.matmul(x, self.weight) + self.linear_bias
        
        # BatchNorm
        x_normalized = (x - self.bn_mean) / jnp.sqrt(self.bn_var + self.bn_eps)
        x = self.bn_scale * x_normalized + self.bn_bias
        
        # Bias and divide
        x = x + self.bias
        x = x / self.divide_value
        
        # Swish activation
        x = x * jax.nn.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]