import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        
        self.bn_weight = jnp.ones(out_channels)
        self.bn_bias = jnp.zeros(out_channels) 
        self.bn_running_mean = jnp.zeros(out_channels)
        self.bn_running_var = jnp.ones(out_channels)
        
        self.scale_factor = scale_factor
        self.eps = eps
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        padding = ((self.kernel_size-1, self.kernel_size-1),
                  (self.kernel_size-1, self.kernel_size-1),
                  (self.kernel_size-1, self.kernel_size-1))
        x = jax.lax.conv_transpose(
            x_ndhwc, kernel,
            strides=(1, 1, 1),
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        x = x + self.bias.reshape(1, 1, 1, 1, -1)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        N, C, D, H, W = x.shape
        x2d = x.reshape(N * C, D * H * W)

        mean = self.bn_running_mean
        var = self.bn_running_var
        gamma = self.bn_weight
        beta = self.bn_bias

        def kernel_fn(x_ref, o_ref):
            i = pl.program_id(axis=0)
            j = pl.program_id(axis=1)

            x_block = x_ref[...]

            c_idx = i * BLOCK_M + jnp.arange(BLOCK_M)
            c_idx = c_idx % C

            m = mean[c_idx][:, None]
            v = var[c_idx][:, None]
            g = gamma[c_idx][:, None]
            b = beta[c_idx][:, None]

            y = x_block * self.scale_factor
            y = (y - m) / jnp.sqrt(v + self.eps)
            y = y * g + b

            o_ref[...] = y

        BLOCK_M = 8
        BLOCK_N = 128

        grid_m = x2d.shape[0] // BLOCK_M
        grid_n = x2d.shape[1] // BLOCK_N

        y2d = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
            ),
        )(x2d)

        y = y2d.reshape(N, C, D, H, W)

        y = jnp.mean(y, axis=(2, 3, 4), keepdims=True)
        return y

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
