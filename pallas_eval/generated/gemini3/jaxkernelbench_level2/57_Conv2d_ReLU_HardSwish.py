import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def act_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # bias has shape (8, 128), we only need the first row
    b = bias[0, :]
    
    # Add bias
    x = x + b
    
    # ReLU
    x = jnp.maximum(0.0, x)
    
    # HardSwish
    x = x * jnp.clip((x + 3.0) / 6.0, 0.0, 1.0)
    
    o_ref[...] = x

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        # Initialize with PyTorch conv2d weight shape (out_channels, in_channels, kH, kW)
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel from (out,in,H,W) to (H,W,in,out)
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        
        # Perform convolution
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        N, H, W, C = x.shape
        
        # Fuse Bias + ReLU + HardSwish into a single Pallas kernel
        # We reshape the tensor to (-1, 128) to satisfy the (8, 128) block shape rule
        if 128 % C == 0:
            E = x.size
            pad_E = (128 - (E % 128)) % 128
            if pad_E > 0:
                x_flat = jnp.pad(x.reshape(-1), (0, pad_E))
                x_2d = x_flat.reshape((-1, 128))
            else:
                x_2d = x.reshape((-1, 128))
            
            # Tile bias to match the 128 dimension
            bias_128 = jnp.tile(self.bias, 128 // C)
            bias_2d = jnp.tile(bias_128, (8, 1))
            
            block_K = 1024
            K = x_2d.shape[0]
            pad_K = (block_K - (K % block_K)) % block_K
            if pad_K > 0:
                x_2d = jnp.pad(x_2d, ((0, pad_K), (0, 0)))
                
            grid = (x_2d.shape[0] // block_K,)
            
            out_2d = pl.pallas_call(
                act_kernel,
                out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid,
                    in_specs=[
                        pl.BlockSpec((block_K, 128), lambda i: (i, 0)),
                        pl.BlockSpec((8, 128), lambda i: (0, 0)),
                    ],
                    out_specs=pl.BlockSpec((block_K, 128), lambda i: (i, 0)),
                ),
            )(x_2d, bias_2d)
            
            if pad_K > 0:
                out_2d = out_2d[:-pad_K]
            out_flat = out_2d.reshape(-1)
            if pad_E > 0:
                out_flat = out_flat[:-pad_E]
            x = out_flat.reshape((N, H, W, C))
        else:
            # Fallback for arbitrary channels
            x = x + self.bias.reshape(1, 1, 1, -1)
            x = jnp.maximum(0.0, x)
            x = x * jnp.clip((x + 3.0) / 6.0, 0.0, 1.0)
            
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        return x

batch_size = 128
in_channels = 8  
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
