import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
        Applies Max Pooling 2D to the input tensor using a Pallas TPU kernel.

        Args:
            x: Input array of shape (batch_size, channels, height, width).

        Returns:
            Output array after Max Pooling 2D.
        """
        batch_size, channels, height, width = x.shape
        
        out_height = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        def pool_kernel(x_ref, o_ref):
            # Load the entire spatial block for the current batch and channel
            x_val = x_ref[...][0, 0, :, :]
            
            # Pad the spatial dimensions in VMEM
            if self.padding > 0:
                pad_val = jnp.array(-jnp.inf, dtype=x_val.dtype)
                x_padded = jax.lax.pad(
                    x_val, 
                    pad_val, 
                    ((self.padding, self.padding, 0), (self.padding, self.padding, 0))
                )
            else:
                x_padded = x_val
                
            out = None
            
            # Unroll the pooling window loop at trace time
            for ky in range(self.kernel_size):
                for kx in range(self.kernel_size):
                    y_offset = ky * self.dilation
                    x_offset = kx * self.dilation
                    
                    # Calculate exact stop indices to ensure static slice shapes
                    stop_y = y_offset + (out_height - 1) * self.stride + 1
                    stop_x = x_offset + (out_width - 1) * self.stride + 1
                    
                    slice_y = slice(y_offset, stop_y, self.stride)
                    slice_x = slice(x_offset, stop_x, self.stride)
                    
                    window_val = x_padded[slice_y, slice_x]
                    
                    if out is None:
                        out = window_val
                    else:
                        out = jnp.maximum(out, window_val)
                        
            # Write the result back to HBM
            o_ref[...] = out[None, None, :, :]
            
        grid = (batch_size, channels)
        out_shape = jax.ShapeDtypeStruct((batch_size, channels, out_height, out_width), x.dtype)
        
        return pl.pallas_call(
            pool_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((1, 1, height, width), lambda b, c: (b, c, 0, 0))
                ],
                out_specs=pl.BlockSpec((1, 1, out_height, out_width), lambda b, c: (b, c, 0, 0))
            )
        )(x)

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
