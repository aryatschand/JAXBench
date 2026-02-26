"""
JAXBench Level 2 - Gemm_Divide_Sum_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 14: Gemm_Divide_Sum_Scaling
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor

        # Weight matrix (hidden_size, input_size)
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (hidden_size, input_size))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).
        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Use highest precision matmul to match CPU computation
        # TPU can use bfloat16 internally which causes large numerical differences
        x = lax.dot_general(
            x, self.weight.T,
            dimension_numbers=(((1,), (0,)), ((), ())),
            precision=lax.Precision.HIGHEST
        )
        # Divide by 2
        x = x / 2.0
        # Sum along dim=1 with keepdim
        x = jnp.sum(x, axis=1, keepdims=True)
        # Scale
        x = x * self.scaling_factor
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, input_size))
    return [x]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
