import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax

class Model:
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        
        # Initialize weights with same shape as PyTorch but transposed for JAX
        weight_shape = (in_channels, 1, kernel_size, kernel_size) # PyTorch shape
        key = jax.random.PRNGKey(0)
        weight = jax.random.normal(key, weight_shape) * 0.02
        # Transpose from (C_out, C_in, H, W) to (H, W, C_in, C_out) for JAX
        self.conv2d_weight = jnp.transpose(weight, (2, 3, 1, 0))
        
        if bias:
            self.conv2d_bias = jnp.zeros(in_channels)
        else:
            self.conv2d_bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                # Transpose weight from PyTorch to JAX format
                value = jnp.transpose(value, (2, 3, 1, 0))
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        # 1. Transpose x from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        N, H, W, C = x.shape
        
        # 2. Calculate original output spatial dimensions
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 3. Determine block sizes for Pallas kernel
        H_block = 32
        W_block = 32
        C_block = 128  # Multiple of 128 for optimal TPU performance
        
        # 4. Calculate padded dimensions to ensure they are multiples of block sizes
        C_pad = (C + C_block - 1) // C_block * C_block
        H_out_pad = (H_out + H_block - 1) // H_block * H_block
        W_out_pad = (W_out + W_block - 1) // W_block * W_block
        
        # 5. Calculate required input spatial dimensions for the padded output
        H_req = (H_out_pad - 1) * self.stride + self.kernel_size
        W_req = (W_out_pad - 1) * self.stride + self.kernel_size
        
        # 6. Pad input tensor
        pad_h_top = self.padding
        pad_h_bottom = max(0, H_req - H - pad_h_top)
        pad_w_left = self.padding
        pad_w_right = max(0, W_req - W - pad_w_left)
        
        x_padded = jnp.pad(x, ((0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, C_pad - C)))
        # Slice to exact required size in case stride > 1 drops elements
        x_padded = x_padded[:, :H_req, :W_req, :]
        
        # 7. Create sliced inputs to bypass BlockSpec overlapping limitations
        sliced_inputs = []
        for dy in range(self.kernel_size):
            for dx in range(self.kernel_size):
                sliced = x_padded[:, dy : dy + H_out_pad * self.stride : self.stride, 
                                     dx : dx + W_out_pad * self.stride : self.stride, :]
                sliced_inputs.append(sliced)
                
        # 8. Pad weight and bias to match C_pad
        w = self.conv2d_weight
        if C_pad > C:
            w = jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, C_pad - C)))
            
        if self.has_bias:
            b = self.conv2d_bias
            if C_pad > C:
                b = jnp.pad(b, (0, C_pad - C))
                
        # 9. Define Grid and BlockSpecs
        N_block = 1
        grid = (N, H_out_pad // H_block, W_out_pad // W_block, C_pad // C_block)
        
        x_block_shape = (N_block, H_block, W_block, C_block)
        x_index_map = lambda n, h, w_idx, c: (n, h, w_idx, c)
        
        in_specs = [pl.BlockSpec(x_block_shape, x_index_map) for _ in range(self.kernel_size * self.kernel_size)]
        
        w_block_shape = (self.kernel_size, self.kernel_size, 1, C_block)
        w_index_map = lambda n, h, w_idx, c: (0, 0, 0, c)
        in_specs.append(pl.BlockSpec(w_block_shape, w_index_map))
        
        if self.has_bias:
            b_block_shape = (C_block,)
            b_index_map = lambda n, h, w_idx, c: (c,)
            in_specs.append(pl.BlockSpec(b_block_shape, b_index_map))
            
        out_block_shape = (N_block, H_block, W_block, C_block)
        out_index_map = lambda n, h, w_idx, c: (n, h, w_idx, c)
        out_specs = pl.BlockSpec(out_block_shape, out_index_map)
        
        # 10. Define Pallas kernel function
        K = self.kernel_size
        has_bias = self.has_bias
        
        def kernel_fn(*args):
            if has_bias:
                x_refs = args[:K*K]
                w_ref = args[K*K]
                b_ref = args[K*K+1]
                o_ref = args[K*K+2]
            else:
                x_refs = args[:K*K]
                w_ref = args[K*K]
                o_ref = args[K*K+1]
                
            # Use f32 accumulator for precision and performance
            acc = jnp.zeros(x_refs[0].shape, dtype=jnp.float32)
            
            idx = 0
            for dy in range(K):
                for dx in range(K):
                    w_val = w_ref[dy, dx, 0, :]
                    w_val = w_val.reshape((1, 1, 1, -1))
                    acc += x_refs[idx][...] * w_val
                    idx += 1
                    
            if has_bias:
                b_val = b_ref[...]
                b_val = b_val.reshape((1, 1, 1, -1))
                acc += b_val
                
            o_ref[...] = acc.astype(o_ref.dtype)
            
        # 11. Execute Pallas kernel
        out_shape = jax.ShapeDtypeStruct((N, H_out_pad, W_out_pad, C_pad), x.dtype)
        
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
        )
        
        if self.has_bias:
            out_padded = pl.pallas_call(
                kernel_fn,
                out_shape=out_shape,
                grid_spec=grid_spec
            )(*sliced_inputs, w, b)
        else:
            out_padded = pl.pallas_call(
                kernel_fn,
                out_shape=out_shape,
                grid_spec=grid_spec
            )(*sliced_inputs, w)
            
        # 12. Slice back to original output dimensions and transpose to NCHW
        out = out_padded[:, :H_out, :W_out, :C]
        out = jnp.transpose(out, (0, 3, 1, 2))
        
        return out

# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
