"""
JAXBench Level 2 - Matmul_Dropout_Softmax
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax import random

class Model:
    def __init__(self, in_features, out_features, dropout_p):
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Initialize weights and bias with zeros (will be set via set_weights)
        # PyTorch Linear weight shape is (out_features, in_features)
        self.matmul_weight = jnp.zeros((out_features, in_features))
        self.matmul_bias = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer - PyTorch stores weight as (out_features, in_features)
        # So we need to transpose it for x @ weight.T + bias
        x = x @ self.matmul_weight.T + self.matmul_bias
        
        # Skip dropout during inference (no rng needed)
        # In inference mode, dropout is typically disabled
        # If training mode is needed, rng should be passed differently
            
        # Softmax
        x = jax.nn.softmax(x, axis=1)
        return x

batch_size = 128
in_features = 16384  
out_features = 16384
dropout_p = 0.2

def get_inputs():
    key = random.PRNGKey(0)
    return [random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, dropout_p]