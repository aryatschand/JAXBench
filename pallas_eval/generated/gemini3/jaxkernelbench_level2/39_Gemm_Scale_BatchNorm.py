```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Linear layer parameters
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        self.gemm_weight = jax.random.normal(key1, (out_features, in_features))
        self.gemm_bias = jax.random.normal(key2, (out_features,))

        # Scale parameter
        self.scale = jax.random.normal(key3, scale_shape)

        # BatchNorm1d parameters (learnable only)
        self.bn_weight = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))

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
        # Linear: x @ weight.T
        x = jnp.matmul(x, self.gemm_weight.T)

        # Reshape 1D parameters to 2D for Pallas
        gemm_bias = self.gemm_bias.reshape(1, -1)
        scale = self.scale.reshape(1, -1)
        bn_weight = self.bn_weight.reshape(1, -1)
        bn_bias = self.bn_bias.reshape(1, -1)

        batch_size, out_features = x.shape
        block_col = 128
        grid_dim = out_features // block_col

        def get_epilogue_kernel(eps):
            def epilogue_kernel(x_ref, gemm_bias_ref, scale_ref, bn_weight_ref, bn_bias_ref, out_ref):
                x_val = x_ref[...]
                bias_val = gemm_bias_ref[...]
