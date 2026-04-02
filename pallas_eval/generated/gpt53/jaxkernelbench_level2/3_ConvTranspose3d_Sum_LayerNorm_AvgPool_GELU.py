import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.norm_shape = norm_shape
        self.pool_kernel_size = pool_kernel_size
        
        key = jax.random.PRNGKey(0)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        self.conv_transpose_weight = jax.random.normal(
            key1,
            (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        ) * 0.01
        self.conv_transpose_bias = jnp.zeros((out_channels,))
        
        self.sum_weight = jnp.array(sum_weight)
        self.norm_weight = jnp.ones(norm_shape)
        self.norm_bias = jnp.zeros(norm_shape)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        
        pad_d = self.kernel_size[0] - 1 - self.padding[0]
        pad_h = self.kernel_size[1] - 1 - self.padding[1]
        pad_w = self.kernel_size[2] - 1 - self.padding[2]
        
        padding_spec = (
            (pad_d, pad_d + self.output_padding[0]),
            (pad_h, pad_h + self.output_padding[1]),
            (pad_w, pad_w + self.output_padding[2])
        )
        
        x = lax.conv_transpose(
            x,
            kernel,
            strides=self.stride,
            padding=padding_spec,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
            transpose_kernel=True
        )
        
        x = x + self.conv_transpose_bias
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NCDHW
        
        # Flatten for Pallas: (N*D*H*W, C)
        N, C, D, H, W = x.shape
        x_flat = jnp.transpose(x, (0, 2, 3, 4, 1)).reshape(-1, C)

        def kernel_fn(x_ref, o_ref):
            vals = x_ref[...]  # (B, C)
            
            vals = vals + self.sum_weight
            
            mean = jnp.mean(vals, axis=1, keepdims=True)
            var = jnp.mean((vals - mean) * (vals - mean), axis=1, keepdims=True)
            vals = (vals - mean) / jnp.sqrt(var + 1e-5)
            
            vals = vals * self.norm_weight + self.norm_bias
            
            # GELU
            vals = 0.5 * vals * (1.0 + jnp.tanh(
                jnp.sqrt(2.0 / jnp.pi) * (vals + 0.044715 * vals * vals * vals)
            ))
            
            o_ref[...] = vals

        block_rows = 128
        total_rows = x_flat.shape[0]
        assert total_rows % block_rows == 0

        x_flat = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(total_rows // block_rows,),
                in_specs=[pl.BlockSpec((block_rows, C), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((block_rows, C), lambda i: (i, 0)),
            ),
        )(x_flat)

        # Restore shape
        x = x_flat.reshape(N, D, H, W, C)
        
        # AvgPool
        window_shape = (1, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2], 1)
        strides = (1, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2], 1)
        
        x = lax.reduce_window(
            x,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_shape,
            window_strides=strides,
            padding='VALID'
        )
        
        x = x / (self.pool_kernel_size[0] * self.pool_kernel_size[1] * self.pool_kernel_size[2])
        
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x


batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
