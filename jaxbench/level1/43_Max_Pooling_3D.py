"""
JAXBench Level 1 - Task 43: Max_Pooling_3D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.135964
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import List

class Model:
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        if stride is None:
            stride = kernel_size
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            jnp.ndarray: Output tensor with Max Pooling 3D applied.
        """
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Window shape for NDHWC format: (1, D, H, W, 1)
        window_shape = (1, self.kernel_size, self.kernel_size, self.kernel_size, 1)
        strides = (1, self.stride, self.stride, self.stride, 1)
        
        # Padding for NDHWC format: ((0, 0), (pad, pad), (pad, pad), (pad, pad), (0, 0))
        padding = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        
        # Apply max pooling using lax.reduce_window
        # For dilation, we need to use window_dilation parameter
        window_dilation = (1, self.dilation, self.dilation, self.dilation, 1)
        
        x = lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=window_shape,
            window_strides=strides,
            padding=padding,
            window_dilation=window_dilation
        )
        
        # Convert back from NDHWC to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

    def set_weights(self, weights_dict):
        """No weights to set for MaxPool3d"""
        pass

batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, dim1, dim2, dim3))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]