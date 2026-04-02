import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, bias_ref, o_ref, scale1, scale2):
    x = x_ref[...]
    b = bias_ref[...]
    x = x + b
    x = x * scale1
    x = x * scale2
    o_ref[...] = x

def fused_op(x, bias, scale1, scale2):
    n = x.shape[0]
    c = x.shape[1]
    block = (min(n, 1024), min(c, 128))
    grid = (n // block[0], c // block[1])
    return pl.pallas_call(
        lambda x_ref, b_ref, o_ref: fused_kernel(x_ref, b_ref, o_ref, scale1, scale2),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i, j: (i, j)),
                pl.BlockSpec((1, block[1]), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x, bias)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        self.scale1 = jnp.array(scale1)
        self.bias = jnp.zeros(bias_shape)
        self.scale2 = jnp.array(scale2)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        pad_val = self.kernel_size - 1 - self.padding
        padding = ((pad_val, pad_val), (pad_val, pad_val), (pad_val, pad_val))
        
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=(self.stride, self.stride, self.stride),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        
        x = x + self.conv_transpose_bias
        
        x = jax.lax.reduce_window(
            x,
            init_value=0.,
            computation=jax.lax.add,
            window_dimensions=(1, 2, 2, 2, 1),
            window_strides=(1, 2, 2, 2, 1),
            padding='VALID'
        ) / 8.0
        
        bias_reshaped = self.bias.reshape(1, 1, 1, 1, -1)
        
        # reshape to 2D for Pallas: (N*D*H*W, C)
        n, d, h, w, c = x.shape
        x2 = x.reshape(n * d * h * w, c)
        b2 = bias_reshaped.reshape(1, c)
        
        x2 = fused_op(x2, b2, self.scale1, self.scale2)
        
        x = x2.reshape(n, d, h, w, c)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(128, 3, 16, 32, 32))]

def get_init_inputs():
    return [3, 16, 3, 2, 1, 0.5, 1.0, (16, 1, 1, 1)]
