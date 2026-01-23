"""
JAXBench Level 1 - conv_transposed_1D_dilated
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # PyTorch ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._kernel_size = kernel_size
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # Convert NCW -> NWC
        x = jnp.transpose(x, (0, 2, 1))
        
        # For dilated transposed convolution, we need to use conv_general_dilated
        # with transposed=True approach, or manually handle dilation
        
        # Transpose kernel from (in, out, W) -> (W, out, in)
        kernel = jnp.transpose(self.weight, (2, 1, 0))
        
        # For transposed convolution with dilation, we use conv_general_dilated
        # The "lhs_dilation" parameter effectively creates a transposed convolution
        # and "rhs_dilation" handles the kernel dilation
        
        # Calculate output padding for transposed conv
        # For conv_transpose: output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        
        # Use conv_general_dilated with lhs_dilation for transposed convolution
        # lhs_dilation = stride makes it a transposed convolution
        # rhs_dilation = dilation handles the kernel dilation
        
        # Padding calculation for transposed conv
        # We need to compute the appropriate padding
        effective_kernel_size = self.dilation * (self._kernel_size - 1) + 1
        pad_total = effective_kernel_size - 1 - self.padding
        
        out = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1,),
            padding=((pad_total, pad_total),),
            lhs_dilation=(self.stride,),  # This makes it transposed convolution
            rhs_dilation=(self.dilation,),  # This handles kernel dilation
            dimension_numbers=('NWC', 'WOI', 'NWC')
        )
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
            
        # Convert back NWC -> NCW
        out = jnp.transpose(out, (0, 2, 1))
        
        return out
    
    @property
    def kernel_size(self):
        return self.weight.shape[2]

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (32, 32, 131072))
    return [x]

def get_init_inputs():
    return [32, 64, 5, 1, 0, 3]