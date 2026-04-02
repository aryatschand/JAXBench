```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def logsumexp_relu_bias_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Broadcast bias to match x shape
    bias_rep = pltpu.repeat(bias, x.shape[0], axis=0)
    x = x + bias_rep
    
    # LogSumExp along channel dim (axis=1)
    max_x = jnp.max(x, axis=1, keepdims=True)
    max_x_rep = pltpu.repeat(max_x, x.shape[1], axis=1)
    
    exp_x = jnp.exp(x - max_x_rep)
    sum_exp_x = jnp.sum(exp_x, axis=1, keepdims=True)
    
    lse = max_x + jnp.log(sum_exp_x)
    
    # ReLU
    out = jnp.maximum(lse, 0.0)
    
    o_ref[...] = out

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv3d: NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # 3D convolution
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(self.stride, self.stride, self.stride),
            padding=[(self.padding, self.padding)] * 3,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
        
        # MaxPool3d
        # We swap bias addition and MaxPool3d. Since bias is constant over spatial dims,
        # max(x + bias) == max(x) + bias. Doing MaxPool3d first reduces the tensor size by 8x
        # before we apply the bias addition, saving significant memory bandwidth.
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.
