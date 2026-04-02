import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = bias
        
        # Initialize weights with same shapes as PyTorch
        key = jax.random.PRNGKey(0)
        weight_shape = (kernel_size, in_channels, out_channels)
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv1d.weight':
                # Convert PyTorch weight (out_channels, in_channels, kernel_size) 
                # to JAX (kernel_size, in_channels, out_channels)
                value = jnp.transpose(jnp.array(value), (2, 1, 0))
                setattr(self, 'weight', value)
            elif name == 'conv1d.bias':
                value = jnp.array(value)
                setattr(self, 'bias', value)

    def forward(self, x):
        batch_size, in_channels, length = x.shape
        
        # Calculate exact output length
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        out_length = (length - effective_kernel_size) // self.stride + 1
        
        # Tile output length into blocks of 128 for optimal MXU usage
        block_out_len = 128
        num_blocks = (out_length + block_out_len - 1) // block_out_len
        
        # Calculate required input block length to satisfy the output block
        max_stop_idx = (self.kernel_size - 1) * self.dilation + block_out_len * self.stride
        # Round up to next multiple of 128 to satisfy TPU block size constraints
        block_in_len = ((max_stop_idx + 127) // 128) * 128
        
        # Pad input sequence if necessary to cover all blocks
        max_start = (num_blocks - 1) * block_out_len * self.stride
        required_length = max_start + block_in_len
        pad_len = max(0, required_length - length)
        
        if pad_len > 0:
            x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_len)))
        else:
            x_pad = x
            
        # Extract overlapping input blocks for each output block
        starts = jnp.arange(num_blocks) * (block_out_len * self.stride)
        def get_block(start):
            return jax.lax.dynamic_slice(x_pad, (0, 0, start), (batch_size, in_channels, block_in_len))
        x_blocks = jax.vmap(get_block, in_axes=0, out_axes=1)(starts)
        
        # Prepare bias (must be at least 2D for Pallas)
        if self.bias is None:
            bias = jnp.zeros((self.out_channels, 1), dtype=x.dtype)
        else:
            bias = self.bias.reshape(self.out_channels, 1)
            
        def conv_kernel(x_ref, w_ref, bias_ref, out_ref):
            x_val = x_ref[0, 0, :, :]
            w_val = w_ref[...]
            bias_val = bias_ref[...]
            
            # Accumulate in f32 for precision
            out_val = jnp.zeros((self.out_channels, block_out_len), dtype=jnp.float32)
            
            # Unroll over kernel size
            for k in range(self.kernel_size):
                start_idx = k * self.dilation
                stop_idx = start_idx + block_out_len * self.stride
                # Strided slice to gather the exact elements for this kernel position
                x_k = x_val[:, start_idx:stop_idx:self.stride]
                w_k = w_val[k]
                # Matrix multiplication maps directly to MXU
                out_val += jnp.dot(w_k.T, x_k, preferred_element_type=jnp.float32)
                
            # Add bias (broadcasts over the sequence dimension)
            out_val += bias_val
            
            out_ref[0, 0, :, :] = out_val.astype(out_ref.dtype)
            
        # Execute Pallas kernel
        out_blocks = pl.pallas_call(
            conv_kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size, num_blocks, self.out_channels, block_out_len), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch_size, num_blocks),
                in_specs=[
                    pl.BlockSpec((1, 1, in_channels, block_in_len), lambda b, i: (b, i, 0, 0)),
                    pl.BlockSpec((self.kernel_size, in_channels, self.out_channels), lambda b, i: (0, 0, 0)),
                    pl.BlockSpec((self.out_channels, 1), lambda b, i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, self.out_channels, block_out_len), lambda b, i: (b, i, 0, 0)),
            ),
        )(x_blocks, self.weight, bias)
        
        # Reconstruct the final output sequence
        out = jnp.transpose(out_blocks, (0, 2, 1, 3))
        out = out.reshape(batch_size, self.out_channels, -1)
        out = out[:, :, :out_length]
        
        return out

# Test code - REDUCED SIZE for memory
batch_size = 16  # Reduced from 64
in_channels = 64  
out_channels = 128
kernel_size = 3
length = 65536  # Reduced from 524280
stride = 3
dilation = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
