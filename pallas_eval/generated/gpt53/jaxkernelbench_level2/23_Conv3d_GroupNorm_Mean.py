import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.num_groups = num_groups
        self.gamma = jnp.ones(out_channels)
        self.beta = jnp.zeros(out_channels)
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Conv3d (keep JAX primitive for efficiency)
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
        
        # GroupNorm (JAX)
        N, C, D, H, W = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G, D, H, W)
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        x = x.reshape(N, C, D, H, W)
        x = x * self.gamma.reshape(1, -1, 1, 1, 1) + self.beta.reshape(1, -1, 1, 1, 1)

        # Flatten for reduction
        x_flat = x.reshape(N, -1)

        # Pallas kernel for fast per-batch mean
        def mean_kernel(x_ref, o_ref):
            vals = x_ref[...]
            s = jnp.sum(vals, axis=1, keepdims=True)
            o_ref[...] = s / vals.shape[1]

        block_m = 1
        block_n = x_flat.shape[1]

        result = pl.pallas_call(
            mean_kernel,
            out_shape=jax.ShapeDtypeStruct((N, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N,),
                in_specs=[pl.BlockSpec((block_m, block_n), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
            ),
        )(x_flat)

        return result.squeeze(-1)

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
