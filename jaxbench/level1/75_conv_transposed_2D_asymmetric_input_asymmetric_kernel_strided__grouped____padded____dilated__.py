"""
JAXBench Level 1 - conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        # Initialize weights with PyTorch shapes (in_channels, out_channels/groups, kH, kW)
        self.weight = jnp.zeros((in_channels, out_channels // groups, kernel_size[0], kernel_size[1]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # For dilated transposed convolution, we need to use conv_general_dilated
        # with transposed=True approach, or manually implement it
        
        # Handle grouped convolution by splitting
        x_groups = jnp.split(x, self.groups, axis=1)
        w_groups = jnp.split(self.weight, self.groups, axis=0)
        
        results = []
        for g in range(self.groups):
            # Convert NCHW -> NHWC
            x_g = jnp.transpose(x_groups[g], (0, 2, 3, 1))
            
            # Convert kernel (in/groups, out/groups, H, W) -> (H, W, out/groups, in/groups)
            kernel_g = jnp.transpose(w_groups[g], (2, 3, 1, 0))
            
            # For transposed convolution with dilation, we use conv_general_dilated
            # The "transposed" conv is achieved by swapping lhs_dilation and rhs_dilation
            # lhs_dilation corresponds to stride in transposed conv
            # rhs_dilation corresponds to dilation in transposed conv
            
            # Calculate padding for transposed convolution
            # Output size formula for conv_transpose:
            # out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
            
            # For conv_general_dilated with lhs_dilation (transposed conv):
            # We need to compute the appropriate padding
            dilated_kernel_h = self.dilation[0] * (self.kernel_size[0] - 1) + 1
            dilated_kernel_w = self.dilation[1] * (self.kernel_size[1] - 1) + 1
            
            # Padding calculation for transposed convolution
            pad_h_total = dilated_kernel_h - 1
            pad_w_total = dilated_kernel_w - 1
            
            # Adjust for PyTorch padding
            pad_h_before = pad_h_total - self.padding[0]
            pad_h_after = pad_h_total - self.padding[0]
            pad_w_before = pad_w_total - self.padding[1]
            pad_w_after = pad_w_total - self.padding[1]
            
            padding_jax = ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after))
            
            out_g = jax.lax.conv_general_dilated(
                x_g, 
                kernel_g,
                window_strides=(1, 1),
                padding=padding_jax,
                lhs_dilation=self.stride,  # This creates the transposed effect
                rhs_dilation=self.dilation,  # Kernel dilation
                dimension_numbers=('NHWC', 'HWOI', 'NHWC')
            )
            
            # Convert back NHWC -> NCHW
            out_g = jnp.transpose(out_g, (0, 3, 1, 2))
            results.append(out_g)
            
        out = jnp.concatenate(results, axis=1)
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
            
        return out

# Test parameters
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]