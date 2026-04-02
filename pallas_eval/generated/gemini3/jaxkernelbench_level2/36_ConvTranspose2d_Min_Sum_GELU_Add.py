```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def partial_reduce_kernel(x_ref, o_ref):
    x_val = x_ref[...]
    # Minimum operation along channel dimension (axis 3 in NHWC)
    min_c = jnp.min(x_val, axis=3)
    # Sum operation along height dimension (axis 1 in NHWC block)
    sum_h = jnp.sum(min_c, axis=1)
    o_ref[0, 0, :] = sum_h

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # (in,out,H,W) -> (H,W,out,in)
        
        # Calculate padding
        pad = kernel.shape[0] - 1 - self.padding
        padding = ((pad, pad), (pad, pad))
        
        out = jax.lax.conv_transpose(
            x_nhwc, 
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        # Handle output padding if needed
        if self.output_padding > 0:
            out = jax.lax.pad(
                out,
                padding_value=0.0,
                padding_config=((0, 0, 0), (0, self.output_padding, 0), 
                              (0, self.output_padding, 0), (0, 0, 0))
            )
            
        N, H, W, C = out.shape
        
        # Determine safe block sizes for VMEM
        block_W = 32
        while
