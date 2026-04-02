import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def epilogue_kernel(x_ref, bias_ref, scale_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    scale = scale_ref[0]
    
    # Mean over D (axis 1)
    x_mean = jnp.sum(x, axis=1, keepdims=True) / x_ref.shape[1]
    
    # Add bias
    x_bias = x_mean + bias.reshape(1, 1, 1, 1, -1)
    
    # Softmax over C (axis 4)
    x_max = jnp.max(x_bias, axis=4, keepdims=True)
    x_exp = jnp.exp(x_bias - x_max)
    x_sum = jnp.sum(x_exp, axis=4, keepdims=True)
    x_soft = x_exp / x_sum
    
    # Tanh
    x_tanh = jnp.tanh(x_soft)
    
    # Scale
    x_out = x_tanh * scale
    
    o_ref[...] = x_out

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros((1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose3d
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))  # (in,out,D,H,W) -> (D,H,W,out,in)
        padding = ((self.kernel_size-1-self.padding,)*2,)*3  # For each spatial dim
        conv_out = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(self.stride,)*3,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        N, D, H, W, C = conv_out.shape
        
        def get_block_size(dim, target):
            for i in range(target, 0, -1):
                if dim % i == 0:
                    return i
            return 1
            
        bh = get_block_size(H, 8)
        bw = get_block_size(W, 32)
        
        grid = (N, H // bh, W // bw)
        
        bias_1d = self.bias.reshape(-1)
        scale_1d = jnp.array([self.scaling_factor], dtype=conv_out.dtype)
        
        out_shape = jax.ShapeDtypeStruct((N, 1, H, W, C), conv_out.dtype)
        
        epilogue_out = pl.pallas_call(
            epilogue_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((1, D, bh, bw, C), lambda n, h, w: (n, 0, h, w, 0)),
                    pl.BlockSpec((C,), lambda n, h, w: (0,)),
                    pl.BlockSpec((1,), lambda n, h, w: (0,))
                ],
                out_specs=pl.BlockSpec((1, 1, bh, bw, C), lambda n, h, w: (n, 0, h, w, 0))
            )
        )(conv_out, bias_1d, scale_1d)
        
        # Transpose back to NCDHW
        out = jnp.transpose(epilogue_out, (0, 4, 1, 2, 3))
        
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (16, 16, 32, 128, 128))]

def get_init_inputs():
    return [16, 64, 3, 1, 1, 2.0]
