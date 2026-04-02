import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def epilogue_kernel(x_ref, bias_ref, mult_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    mult = mult_ref[...]
    
    # Add bias
    x = x + bias.reshape(1, 1, 1, 1, -1)
    
    # LeakyReLU
    x = jnp.where(x > 0, x, x * 0.2)
    
    # Multiply
    x = x * mult.reshape(1, 1, 1, 1, -1)
    
    # LeakyReLU
    x = jnp.where(x > 0, x, x * 0.2)
    
    # MaxPool3d (2x2x2 window, 2x2x2 stride)
    D_blk = x.shape[1]
    H_blk = x.shape[2]
    W_blk = x.shape[3]
    C_blk = x.shape[4]
    
    x_reshaped = x.reshape(1, D_blk // 2, 2, H_blk // 2, 2, W_blk // 2, 2, C_blk)
    x_pooled = jnp.max(x_reshaped, axis=(2, 4, 6))
    
    o_ref[...] = x_pooled

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.multiplier = jnp.zeros(multiplier_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        
        kernel_size = self.
