```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import softmax, sigmoid

def get_fused_kernel(scaling_factor):
    def fused_kernel(x_ref, bias_ref, o_ref):
        x = x_ref[:, :]
        bias = bias_ref[:]
        
        # Softmax over C (axis 1)
        x_max = jnp.max(x, axis=1, keepdims=True)
        x_exp = jnp.exp(x - x_max)
        x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
        x_softmax = x_exp / x_sum
        
        # Bias add
        bias_bcast = pltpu.repeat(bias[None, :], x.shape[0], axis=0)
        x_bias = x_softmax + bias_bcast
        
        # Scaling
        x_scaled = x_bias * scaling_factor
        
        # Sigmoid
        x_sig = 1.0 / (1.0 + jnp.exp(-x_scaled))
        
        o_ref[:, :] = x_sig
    return fused_kernel

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (in, out, H, W) to (H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Calculate padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding), 
                  (pad_w, pad_w +
