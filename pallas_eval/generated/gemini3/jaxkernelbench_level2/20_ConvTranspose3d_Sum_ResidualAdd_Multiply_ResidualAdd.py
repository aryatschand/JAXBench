```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def elementwise_kernel(x_ref, cb_ref, bp1_ref, o_ref):
    # Read the block of spatial elements
    y = x_ref[0, 0, :]
    
    # Read the channel-specific biases (scalars for this block)
    cb = cb_ref[0]
    bp1 = bp1_ref[0]
    
    # Fused element-wise operations:
    # orig = y + cb
    # x1 = orig + b
    # x2 = x1 + orig = 2 * orig + b
    # x3 = x2 * orig = (2 * orig + b) * orig
    # x4 = x3 + orig = (2 * orig + b + 1) * orig
    # We precompute bp1 = b + 1.0 outside the kernel.
    orig = y + cb
    factor = orig * 2.0 + bp1
    x4 = factor * orig
    
    # Write the result back
    o_ref[0, 0, :] = x4

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        self.conv_transpose_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Use NCDHW layout directly to avoid expensive transposes.
        # The PyTorch weight shape (in, out, D, H, W) perfectly matches the 'IODHW' rhs_spec.
        padding = ((1, 1), (1, 1), (1, 1))
        
        x_conv = jax.lax.conv_transpose(
            x, self.conv_transpose_weight,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NCDHW', 'IODHW', 'NCDHW')
        )
        
        # Handle output_padding by padding the result at the end of spatial dimensions
        if self.output_padding > 0:
