```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
        BM = 128 if x.shape[0] % 128 == 0 else x.shape[0]
        BN = 128 if self.weight.shape[1] % 128 == 0 else self.weight.shape[1]
        K = x.shape[1]
        BK = 256 if K % 256 == 0 else K
        
        # Reshape 1D parameters to 2D for block specs
        linear_bias = self.linear_bias.reshape(1, -1)
        bn_mean = self.bn_mean.reshape(1, -1)
        bn_var
