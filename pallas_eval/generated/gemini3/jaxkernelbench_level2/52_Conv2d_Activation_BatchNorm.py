```python
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mish_bias_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    b = bias_ref[...]
    x = x + b
    softplus_x = jnp.logaddexp(0.0, x)
    o_ref[...] = x * jnp.tanh(softplus_x)

def bn_kernel(x_ref, mean_ref, inv_std_ref, weight_ref, bias_ref, o_ref):
    x = x_ref[...]
    mean = mean_ref[...]
    inv_std = inv_std_ref[...]
    w = weight_ref[...]
    b = bias_ref[...]
    o_ref[...] = (x - mean) * inv_std * w + b

class Model:
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    Activation: multiply(tanh(softplus(x)), x) - which is Mish activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

        # Conv2d parameters
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        self.conv_weight = jax.random.normal(key1, (out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jax.random.normal(key2, (out_channels,))

        # BatchNorm2d parameters (learnable only)
        self.bn_weight = jnp.ones((out_channels,))
        self.bn_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, H, W) in PyTorch format
        # Convert to NHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # Conv2d (valid padding - no padding)
        # PyTorch weight: (out_channels,
