"""
JAXBench Level 2 - Gemm_BatchNorm_GELU_ReLU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 41: Gemm_BatchNorm_GELU_ReLU
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp

class Model:
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = 1e-5

        # Linear layer parameters
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        self.gemm_weight = jax.random.normal(key1, (out_features, in_features))
        self.gemm_bias = jax.random.normal(key2, (out_features,))

        # BatchNorm1d parameters (learnable only)
        self.batch_norm_weight = jnp.ones((out_features,))
        self.batch_norm_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features).
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Linear: x @ weight.T + bias
        x = jnp.matmul(x, self.gemm_weight.T) + self.gemm_bias

        # BatchNorm1d (training mode - use batch statistics)
        mean = jnp.mean(x, axis=0, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps) * self.batch_norm_weight + self.batch_norm_bias

        # GELU activation
        x = jax.nn.gelu(x)

        # ReLU activation
        x = jax.nn.relu(x)

        return x

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_features))
    return [x]

def get_init_inputs():
    return [in_features, out_features]
