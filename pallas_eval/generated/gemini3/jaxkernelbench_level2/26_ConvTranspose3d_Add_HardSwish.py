import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def elementwise_kernel(x_ref, bias_ref, add_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    add = add_ref[...]
    
    # Add bias and add_input
    val = x + bias + add
    
    # HardSwish: val * min(max(val + 3, 0), 6) / 6
    # Then multiply by val again as per original code: x = x * hardswish_x
    hw = val * jnp.minimum(jnp.maximum(val + 3.0, 0.0), 6.0) / 6.0
    res = val * hw
    
    o_ref[...] = res

def pallas_elementwise(x, bias, add_input):
    N, C, D, H, W = x.shape
    M = D * H * W
    
    x_flat = x.reshape((N, C, M))
    add_flat = add_input.reshape((N, C, M))
    bias_flat = bias.reshape((C, 1))
    
    block_C = C
    block_M = 512
    
    # Ensure block_M perfectly divides M
    for b in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        if M % b == 0:
            block_M = b
            break
            
    grid = (N, C // block_C, M // block_M)
    
    out_flat = pl.pallas_call(
        elementwise_kernel,
        out_shape=jax.ShapeDtypeStruct((N, C, M), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, block_C, block_M), lambda i, j, k: (i, j, k)),
                pl.BlockSpec((block_C, 1), lambda i, j, k: (j, 0)),
                pl.BlockSpec((1, block_C, block_M), lambda i, j, k: (i, j, k)),
            ],
            out_specs=pl.BlockSpec((1, block_C, block_M), lambda i, j, k: (i, j, k)),
        )
    )(x_flat, bias_flat, add_flat)
    
    return out_flat.reshape((N, C, D, H, W))

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        # For ConvTranspose3d, weight shape is (in_channels, out_channels, D, H, W)
        self.conv_transpose_weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x, add_input):
        # Convert NCDHW -> NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel (in, out, D, H, W) -> (D, H, W, out, in)
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))

        # For conv_transpose with output_padding, we need to handle it specially
        # PyTorch output size = (input - 1) * stride - 2 * padding + kernel_size + output_padding
        # JAX conv_transpose padding calculation
        pad_d = self.kernel_size - 1 - self.padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        
        # Add output_padding to the high side of padding
        padding = ((pad_d, pad_d + self.output_padding), 
                   (pad_h, pad_h + self.output_padding), 
                   (pad_w, pad_w + self.output_padding))

        # ConvTranspose3d
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        # Convert back NDHWC -> NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # Fuse bias addition, add_input addition, and HardSwish into a single Pallas kernel
        x = pallas_elementwise(x, self.conv_transpose_bias, add_input)

        return x


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    return [
        jax.random.uniform(key1, (batch_size, in_channels, D, H, W)),
        jax.random.uniform(key2, (batch_size, out_channels, D*stride, H*stride, W*stride))
    ]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
