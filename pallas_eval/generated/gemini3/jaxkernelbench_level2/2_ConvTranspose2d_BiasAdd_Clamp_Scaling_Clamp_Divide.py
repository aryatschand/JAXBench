```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        # Initialize weights with same shapes as PyTorch
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel (in,out,H,W) -> (H,W,out,in) for JAX conv_transpose
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        k = self._kernel_size
        
        # Use padding that accounts for PyTorch's padding parameter
        pad_h = k - 1 - self.padding
        pad_w = k - 1 - self.padding
        padding = ((pad_h, pad_h + self.output_padding), (pad_w, pad_w + self.output_padding))

        # ConvTranspose2d
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )

        # Convert back NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))

        B, C, H, W = x.shape
        M = H * W

        def get_block_size(dim, max_size):
            if dim >= max_size:
                return max_size
            return 1 if dim == 0 else 2**(dim - 1).bit_length()

        block_C = get_block_size(C, 16)
        block_M = get_block_size(M, 1024)

        pad_C = (block_C - (C % block_C)) % block_C
        pad_M = (block_M - (M % block_M)) % block_M

        x_flat = x.reshape(B, C, M)
        if pad_C > 0 or pad_M > 0:
            x_flat = jnp.pad(x_flat, ((0, 0), (0, pad_C), (0, pad_M)))

        C_padded = C + pad_C
        M_padded = M + pad_M

        # Combine biases and ensure 2D shape for Pallas
        total_bias = (self.conv_transpose_bias.reshape(-1) + self.bias.reshape(-1)).astype(x_flat.dtype)
        if pad_C > 0:
            total_bias = jnp.pad(total_bias, ((0, pad_C),))
        total_bias = total_bias.reshape(C_padded, 1)

        # Ensure scaling factor is a 2D array for Pallas
        scale_arr = jnp.atleast_1d(self.scaling_factor).astype(x_flat.dtype).reshape(1, 1)

        def epilogue_kernel(x_ref, bias_ref, scale_ref, o_ref):
            x_val = x_ref[...]
            b_val = bias_ref[...]
            s_val = scale_ref[...]
            
            # Reshape for broadcasting
            b_val = b_val.reshape(1, b_val.shape[0], 1)
            s_val = s_val.reshape(1, 1, 1)
            
            # Fused epilogue operations
            x_val = x_val + b_val
            x_val = jnp.clip(x_val, 0.0, 1.0)
            x_val = x_val * s_val
            x_val = jnp.clip(x_val, 0.0, 1.0)
            x_val = x_val / s_val
            
            o_ref[...] = x_val

        grid = (B, C_padded // block_C, M_padded // block_M)

        out_flat = pl.pallas_call(
            epilogue_kernel,
            out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
            grid
