"""
JAXBench Level 1 - Task 63: conv_standard_2D__square_input__square_kernel
Manually translated from KernelBench PyTorch to JAX
"""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@functools.lru_cache(maxsize=None)
def make_conv_kernel(k_h, k_w):
    def conv_kernel(*refs):
        # refs contains k_h * k_w input refs, 1 weight ref, 1 out ref
        x_refs = refs[:k_h * k_w]
        w_ref = refs[k_h * k_w]
        out_ref = refs[-1]
        
        B_b, H_b, W_b, O = out_ref.shape
        C = x_refs[0].shape[-1]
        
        acc = jnp.zeros((B_b * H_b * W_b, O), dtype=jnp.float32)
        w = w_ref[...] # shape (k_h, k_w, C, O)
        
        for i in range(k_h):
            for j in range(k_w):
                idx = i * k_w + j
                x = x_refs[idx][...] # shape (B_b, H_b, W_b, C)
                x_flat = x.reshape((B_b * H_b * W_b, C))
                w_ij = w[i, j] # shape (C, O)
                
                # dot product
                acc += jnp.dot(x_flat, w_ij, preferred_element_type=jnp.float32)
                
        out_ref[...] = acc.reshape((B_b, H_b, W_b, O)).astype(out_ref.dtype)
    return conv_kernel


class Model:
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    
    PyTorch Conv2d: kernel shape (out_channels, in_channels, H, W)
    JAX conv_general_dilated with NHWC: kernel shape (H, W, in_channels, out_channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.use_bias = bias
        
        k = kernel_size
        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (k, k, in_channels // groups, out_channels))
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
    
    def set_weights(self, weights_dict):
        """Set weights from PyTorch model."""
        for name, value in weights_dict.items():
            if 'weight' in name:
                # PyTorch Conv2d: (out_channels, in_channels, H, W)
                # JAX: (H, W, in_channels, out_channels)
                pt_weight = jnp.array(value)
                self.weight = jnp.transpose(pt_weight, (2, 3, 1, 0))
            elif 'bias' in name:
                self.bias = jnp.array(value)
    
    def forward(self, x):
        """x: (batch, in_channels, H, W) - NCHW format"""
        # Convert to NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        
        ph, pw = self.padding
        k_h, k_w = self.kernel_size
        
        # Check if we can use Pallas fast path
        if (self.stride == (1, 1) and self.dilation == (1, 1) and 
            self.groups == 1 and k_h <= 5 and k_w <= 5 and 
            x_nhwc.shape[-1] % 8 == 0 and self.out_channels % 8 == 0):
            
            N, H, W, C = x_nhwc.shape
            
            # 1. Initial padding based on self.padding
            x_pad = jnp.pad(x_nhwc, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
            
            # 2. Calculate valid output shape
            H_out = H + 2 * ph - k_h + 1
            W_out = W + 2 * pw - k_w + 1
            
            # 3. Determine block sizes
            def get_block_size(size):
                if size <= 16: return 16
                elif size <= 32: return 32
                else: return 64
            
            H_block = get_block_size(H_out)
            W_block = get_block_size(W_out)
            
            # 4. Calculate padded output shape to be multiple of block sizes
            H_out_pad = (H_out + H_block - 1) // H_block * H_block
            W_out_pad = (W_out + W_block - 1) // W_block * W_block
            
            # 5. Calculate required input shape to produce padded output
            H_req = H_out_pad + k_h - 1
            W_req = W_out_pad + k_w - 1
            
            # 6. Pad x_pad further if necessary
            pad_h = max(0, H_req - x_pad.shape[1])
            pad_w = max(0, W_req - x_pad.shape[2])
            x_pad2 = jnp.pad(x_pad, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
            
            # 7. Create shifted tensors
            shifted_xs = []
            for i in range(k_h):
                for j in range(k_w):
                    shifted_xs.append(x_pad2[:, i:i+H_out_pad, j:j+W_out_pad, :])
            
            grid = (N, H_out_pad // H_block, W_out_pad // W_block)
            
            in_specs = [
                pl.BlockSpec((1, H_block, W_block, C), lambda n, h, w: (n, h, w, 0))
                for _ in range(k_h * k_w)
            ]
            in_specs.append(
                pl.BlockSpec((k_h, k_w, C, self.out_channels), lambda n, h, w: (0, 0, 0, 0))
            )
            
            out_spec = pl.BlockSpec((1, H_block, W_block, self.out_channels), lambda n, h, w: (n, h, w, 0))
            
            kernel_fn = make_conv_kernel(k_h, k_w)
            
            out_padded = pl.pallas_call(
                kernel_fn,
                out_shape=jax.ShapeDtypeStruct((N, H_out_pad, W_out_pad, self.out_channels), x_nhwc.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid,
                    in_specs=in_specs,
                    out_specs=out_spec
                )
            )(*shifted_xs, self.weight)
            
            out = out_padded[:, :H_out, :W_out, :]
            
        else:
            # Fallback to lax.conv_general_dilated
            out = lax.conv_general_dilated(
                x_nhwc,
                self.weight,
                window_strides=self.stride,
                padding=((ph, ph), (pw, pw)),
                lhs_dilation=(1, 1),
                rhs_dilation=self.dilation,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=self.groups
            )
            
        # Convert back to NCHW
        out = jnp.transpose(out, (0, 3, 1, 2))
        
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        
        return out


# Test code
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024


def get_inputs():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (batch_size, in_channels, height, width))
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
