"""
JAXBench Level 2 - ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        # ConvTranspose3d weights shape: (in_channels, out_channels, kD, kH, kW)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        
        # BatchNorm3d parameters
        self.bn_weight = jnp.ones(out_channels)
        self.bn_bias = jnp.zeros(out_channels) 
        self.bn_running_mean = jnp.zeros(out_channels)
        self.bn_running_var = jnp.ones(out_channels)
        
        self.scale_factor = scale_factor
        self.eps = eps
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        padding = ((self.kernel_size-1, self.kernel_size-1),
                  (self.kernel_size-1, self.kernel_size-1),
                  (self.kernel_size-1, self.kernel_size-1))
        x = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(1, 1, 1),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # Scale
        x = x * self.scale_factor

        # BatchNorm3d
        mean = self.bn_running_mean.reshape(1, -1, 1, 1, 1)
        var = self.bn_running_var.reshape(1, -1, 1, 1, 1)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        x = x * self.bn_weight.reshape(1, -1, 1, 1, 1) + self.bn_bias.reshape(1, -1, 1, 1, 1)

        # Global Average Pooling
        x = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
        
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]