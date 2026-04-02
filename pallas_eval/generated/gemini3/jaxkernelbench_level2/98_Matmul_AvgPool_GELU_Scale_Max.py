import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import gelu

def get_kernel(scale_factor, pool_kernel_size):
    def kernel(x_ref, w_ref, bias_ref, out_ref):
        x = x_ref[...]
        w = w_ref[...]
        bias = bias_ref[...]
        
        # Matmul with f32 accumulator
        acc = jnp.dot(x, w, preferred_element_type=jnp.float32) + bias
        
        BM = x.shape[0]
        BN = w.shape[1]
        
        # AvgPool1d
        acc_reshaped = acc.reshape((BM, BN // pool_kernel_size, pool_kernel_size))
        pooled = jnp.sum(acc_reshaped, axis=2) / pool_kernel_size
        
        # GELU
        activated = gelu(pooled)
        
        # Scale
        scaled = activated * scale_factor
        
        # Max along the pooled dimension
        local_max = jnp.max(scaled, axis=1)
        
        # Write to output reference
        out_ref[...] = local_max.reshape((BM, 1))
    return kernel

class Model:
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Tile sizes optimized for TPU v6e VMEM (16MB)
        BM = min(128, x.shape[0])
        BN = min(256, self.weight.shape[1])
        
        grid = (x.shape[0] // BM, self.weight.shape[1] // BN)
        
        out_shape = jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1] // BN), x.dtype)
        
        kernel = get_kernel(self.scale_factor, self.pool_kernel_size)
        
        # Reshape bias to 2D to satisfy Pallas constraints
        bias_2d = jnp.expand_dims(self.bias, axis=0)
        
        partial_max = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((BM, self.weight.shape[0]), lambda i, j: (i, 0)),
                    pl.BlockSpec((self.weight.shape[0], BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, 1), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias_2d)
        
        # Final reduction over the BN blocks
        return jnp.max(partial_max, axis=1)

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
