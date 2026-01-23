"""
JAXBench Level 1 - Task 42: Max_Pooling_2D
Auto-translated from KernelBench PyTorch to JAX
Generated: 2026-01-22T22:58:39.135630
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

class Model:
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x: Input array of shape (batch_size, channels, height, width).

        Returns:
            Output array after Max Pooling 2D.
        """
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Apply max pooling using lax.reduce_window
        window_shape = (1, self.kernel_size, self.kernel_size, 1)
        strides = (1, self.stride, self.stride, 1)
        
        # Padding format for reduce_window: ((before, after), ...) for each dimension
        padding = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        
        # Window dilation for dilated pooling
        window_dilation = (1, self.dilation, self.dilation, 1)
        
        out = lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=window_shape,
            window_strides=strides,
            padding=padding,
            window_dilation=window_dilation
        )
        
        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, channels, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]