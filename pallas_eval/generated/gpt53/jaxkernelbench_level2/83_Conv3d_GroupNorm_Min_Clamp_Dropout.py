import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def clamp_kernel(x_ref, o_ref, min_val, max_val):
    x = x_ref[...]
    x = jnp.minimum(x, min_val)
    x = jnp.clip(x, min_val, max_val)
    o_ref[...] = x

def pallas_clamp(x, min_val, max_val):
    n, m = x.shape
    block0 = min(n, 128)
    block1 = min(m, 128)
    grid = (n // block0, m // block1)

    return pl.pallas_call(
        lambda x_ref, o_ref: clamp_kernel(x_ref, o_ref, min_val, max_val),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec((block0, block1), lambda i, j: (i, j))],
            out_specs=pl.BlockSpec((block0, block1), lambda i, j: (i, j)),
        ),
    )(x)

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        self.conv_weight = jnp.zeros((out_channels, in_channels, *self.kernel_size))
        self.conv_bias = jnp.zeros(out_channels)
        
        self.norm_weight = jnp.ones(out_channels)
        self.norm_bias = jnp.zeros(out_channels)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.conv_weight, (2, 3, 4, 1, 0))
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
        x = x + self.conv_bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        N, C, D, H, W = x.shape
        G = self.groups
        x = x.reshape(N, G, C//G, D, H, W)
        mean = jnp.mean(x, axis=(2,3,4,5), keepdims=True)
        var = jnp.var(x, axis=(2,3,4,5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, D, H, W)
        x = x * self.norm_weight.reshape(1,-1,1,1,1) + self.norm_bias.reshape(1,-1,1,1,1)

        # Flatten for Pallas kernel
        x_flat = x.reshape(N, -1)
        x_flat = pallas_clamp(x_flat, self.min_value, self.max_value)
        x = x_flat.reshape(N, C, D, H, W)

        return x

batch_size = 128
in_channels = 3  
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
