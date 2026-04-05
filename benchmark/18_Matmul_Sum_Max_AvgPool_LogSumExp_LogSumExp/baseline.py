"""
JAXBench Level 2 - Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_features, out_features):
        # PyTorch Linear weight is (out_features, in_features), need to transpose for matmul
        self.linear_weight = jnp.zeros((out_features, in_features))
        self.linear_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, in_features)
        Returns:
            Array of shape (batch_size, 1)
        """
        # PyTorch Linear: x @ weight.T + bias
        # weight shape is (out_features, in_features), so we transpose it
        x = jnp.matmul(x, self.linear_weight.T) + self.linear_bias  # (batch_size, out_features)
        x = jnp.sum(x, axis=1, keepdims=True)  # (batch_size, 1)
        x = jnp.max(x, axis=1, keepdims=True)  # (batch_size, 1)
        x = jnp.mean(x, axis=1, keepdims=True)  # (batch_size, 1)
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)  # (batch_size, 1)
        x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)  # (batch_size, 1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]