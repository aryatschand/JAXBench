import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bias_tanh_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    res = x + bias
    res = jnp.tanh(res)
    o_ref[...] = res

class Model:
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape
        
        # Initialize weights for ConvTranspose2d
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Initialize conv transpose weights
        self.conv_transpose_weight = jax.random.normal(key1, (in_channels, out_channels, kernel_size, kernel_size)) * 0.01
        self.conv_transpose_bias = jax.random.normal(key2, (out_channels,)) * 0.01
        
        # Initialize the bias parameter
        self.bias = jax.random.normal(key3, bias_shape)
    
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # x is in NCHW format
        # Convert to NHWC for JAX convolution
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        
        # PyTorch ConvTranspose2d weight is (in_channels, out_channels, kH, kW)
        # For JAX conv_transpose, we need (kH, kW, out_channels, in_channels) for NHWC
        weight = jnp.transpose(self.conv_transpose_weight, (2, 3, 1, 0))  # (kH, kW, out_channels, in_channels)
        
        # Calculate padding for JAX conv_transpose
        pad = self.kernel_size - 1 - self.padding
        
        # Perform transposed convolution
        x_conv = lax.conv_transpose(
            x_nhwc,
            weight,
            strides=(self.stride, self.stride),
            padding=((pad, pad + self.output_padding), (pad, pad + self.output_padding)),
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )
        
        N, H, W, C = x_conv.shape
        
        # Combine the conv_transpose_bias and the subtraction bias into a single vector
        combined_bias = self.conv_transpose_bias - self.bias.reshape(-1)
        
        # Tile the bias to match the flattened N * C dimension
        repeated_bias = jnp.tile(combined_bias, N).reshape(1, 1, N * C)
        
        # Reshape x_conv to group the N and C dimensions together.
        # This allows us to use a highly efficient block size where the innermost dimension is a multiple of 128.
        x_reshaped = x_conv.reshape(H, W, N * C)
        
        grid = (H,)
        block_shape = (1, W, N * C)
        
        # Fuse bias addition and tanh activation into a single Pallas kernel
        out_reshaped = pl.pallas_call(
            bias_tanh_kernel,
            out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, x_reshaped.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block_shape, lambda i: (i, 0, 0)),
                    pl.BlockSpec((1, 1, N * C), lambda i: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0, 0)),
            ),
        )(x_reshaped, repeated_bias)
        
        # Reshape back to NHWC and transpose to NCHW
        x_conv_new = out_reshaped.reshape(N, H, W, C)
        x_nchw = jnp.transpose(x_conv_new, (0, 3, 1, 2))
        
        return x_nchw

batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(42)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
