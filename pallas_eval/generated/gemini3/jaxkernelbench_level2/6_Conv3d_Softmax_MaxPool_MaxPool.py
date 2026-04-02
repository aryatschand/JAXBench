import jax
import jax.numpy as jnp
from jax.nn import softmax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    C = bias.shape[1]
    x = x.reshape((64, C))
    
    bias_rep = pltpu.repeat(bias, 64, axis=0)
    x = x + bias_rep
    
    x_max = jnp.max(x, axis=1, keepdims=True)
    x_exp = jnp.exp(x - x_max)
    x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
    x_softmax = x_exp / x_sum
    
    out = jnp.max(x_softmax, axis=0)
    
    o_ref[...] = out.reshape((1, 1, C))

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros((out_channels,))
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        if self.pool_kernel_size != 2:
            x = jnp.transpose(x, (0, 2, 3, 4, 1))
            kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
            
            x = jax.lax.conv_general_dilated(
                x, kernel,
                window_strides=(1, 1, 1),
                padding='VALID',
                dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
            )
            x = x + self.bias.reshape(1, 1, 1, 1, -1)
            
            x = jnp.transpose(x, (0, 4, 1, 2, 3))
            x = softmax(x, axis=1)
            
            x = jnp.transpose(x, (0, 2, 3, 4, 1))
            
            for _ in range(2):
                x = jax.lax.reduce_window(
                    x,
                    init_value=-jnp.inf,
                    computation=jax.lax.max,
                    window_dimensions=(1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1),
                    window_strides=(1, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size, 1),
                    padding='VALID'
                )

            x = jnp.transpose(x, (0, 4, 1, 2, 3))
            return x

        N = x.shape[0]
        
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
        )
        
        D_out, H_out, W_out = x.shape[1], x.shape[2], x.shape[3]
        C = x.shape[4]
        
        D_slice = (D_out // 4) * 4
        H_slice = (H_out // 4) * 4
        W_slice = (W_out // 4) * 4
        
        x_sliced = x[:, :D_slice, :H_slice, :W_slice, :]
        
        D_grid = D_slice // 4
        H_grid = H_slice // 4
        W_grid = W_slice // 4
        grid_spatial = D_grid * H_grid * W_grid
        
        x_reshaped = x_sliced.reshape((N, D_grid, 4, H_grid, 4, W_grid, 4, C))
        x_transposed = jnp.transpose(x_reshaped, (0, 1, 3, 5, 2, 4, 6, 7))
        
        block_in = 64 * C
        x_flat = x_transposed.reshape((N, grid_spatial, block_in))
        
        bias = self.bias.reshape((1, C))
        
        out_shape = jax.ShapeDtypeStruct((N, grid_spatial, C), x_flat.dtype)
        
        out_flat = pl.pallas_call(
            fused_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N, grid_spatial),
                in_specs=[
                    pl.BlockSpec((1, 1, block_in), lambda i, j: (i, j, 0)),
                    pl.BlockSpec((1, C), lambda i, j: (0, 0))
                ],
                out_specs=pl.BlockSpec((1, 1, C), lambda i, j: (i, j, 0))
            )
        )(x_flat, bias)
        
        out_reshaped = out_flat.reshape((N, D_grid, H_grid, W_grid, C))
        out = jnp.transpose(out_reshaped, (0, 4, 1, 2, 3))
        
        return out

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
