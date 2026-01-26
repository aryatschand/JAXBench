"""
JAXBench Level 2 - ConvTranspose3d_BatchNorm_AvgPool_AvgPool
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        # ConvTranspose3d weights - note in/out channels are swapped from Conv3d
        self.conv_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros((out_channels,))
        
        # BatchNorm3d parameters
        self.bn_scale = jnp.ones((out_channels,))
        self.bn_bias = jnp.zeros((out_channels,))
        self.bn_mean = jnp.zeros((out_channels,))
        self.bn_var = jnp.ones((out_channels,))
        
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        
        # Padding is kernel_size - 1 - pytorch_padding for each dimension
        pad_size = self.conv_weight.shape[2] - 1 - self.padding
        padding = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))
        
        conv_out = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias.reshape(1, 1, 1, 1, -1)
        
        # Back to NCDHW
        conv_out = jnp.transpose(conv_out, (0, 4, 1, 2, 3))
        
        # BatchNorm3d
        mean = self.bn_mean.reshape(1, -1, 1, 1, 1)
        var = self.bn_var.reshape(1, -1, 1, 1, 1)
        scale = self.bn_scale.reshape(1, -1, 1, 1, 1)
        bias = self.bn_bias.reshape(1, -1, 1, 1, 1)
        
        x_normalized = (conv_out - mean) / jnp.sqrt(var + 1e-5)
        bn_out = scale * x_normalized + bias
        
        # AvgPool3d (first)
        window_shape = (1, 1, 2, 2, 2)
        strides = (1, 1, 2, 2, 2)
        pool1 = jax.lax.reduce_window(
            bn_out, 0.0, jax.lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        ) / 8.0  # Divide by window size (2*2*2)
        
        # AvgPool3d (second)
        pool2 = jax.lax.reduce_window(
            pool1, 0.0, jax.lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        ) / 8.0
        
        return pool2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (64, 3, 32, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, (16, 1, 1, 1)]