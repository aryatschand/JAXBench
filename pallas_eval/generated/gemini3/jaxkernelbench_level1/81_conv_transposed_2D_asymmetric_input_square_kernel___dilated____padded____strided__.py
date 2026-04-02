import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_bias_kernel(x_ref, bias_ref, o_ref):
    # bias_ref has shape (1, block_C)
    # x_ref has shape (block_L, block_C)
    # Broadcasting works natively in Pallas
    o_ref[...] = x_ref[...] + bias_ref[...]

def pallas_add_bias(x, bias):
    # Reshape bias to 2D to satisfy TPU constraints: (1, C)
    bias = jnp.reshape(bias, (1, -1))
    
    L, C = x.shape
    
    # Pad dimensions to multiples of 128 for optimal Pallas block sizes
    pad_L = (128 - (L % 128)) % 128
    pad_C = (128 - (C % 128)) % 128
    
    if pad_L > 0 or pad_C > 0:
        x = jnp.pad(x, ((0, pad_L), (0, pad_C)))
    if pad_C > 0:
        bias = jnp.pad(bias, ((0, 0), (0, pad_C)))
        
    padded_L, padded_C = x.shape
    block_L = min(padded_L, 128)
    block_C = min(padded_C, 128)
    
    grid = (padded_L // block_L, padded_C // block_C)
    
    out = pl.pallas_call(
        add_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_L, block_C), lambda i, j: (i, j)),
                pl.BlockSpec((1, block_C), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((block_L, block_C), lambda i, j: (i, j)),
        )
    )(x, bias)
    
    # Slice back to original shape
    return out[:L, :C]

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        # Initialize weights with PyTorch shape (in_channels, out_channels, kH, kW)
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (in, out, H, W) to (H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # For transposed convolution with dilation, we compute the effective kernel size
        effective_kernel_h = self.dilation * (self._kernel_size - 1) + 1
        effective_kernel_w = self.dilation * (self._kernel_size - 1) + 1
        
        # Calculate padding for conv_transpose
        pad_h = effective_kernel_h - 1 - self.padding
        pad_w = effective_kernel_w - 1 - self.padding
        padding = ((pad_h, pad_h), (pad_w, pad_w))

        # Perform transposed convolution using native rhs_dilation
        # This avoids the memory and compute overhead of manually dilating the kernel with zeros
        out = jax.lax.conv_transpose(
            x,
            kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            rhs_dilation=(self.dilation, self.dilation),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        # Add bias using the optimized Pallas kernel
        bias = self.bias if self.bias is not None else jnp.zeros(out.shape[-1], dtype=out.dtype)
        
        orig_shape = out.shape
        out_flat = out.reshape(-1, orig_shape[-1])
        out_flat = pallas_add_bias(out_flat, bias)
        out = out_flat.reshape(orig_shape)

        # Convert back from NHWC to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

    @property
    def kernel_size(self):
        return self._kernel_size


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
