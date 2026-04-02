import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def kernel_fn(y_ref, bias_ref, o_ref):
    y = y_ref[...]
    bias = bias_ref[...]
    o_ref[...] = (bias + y) * y

class Model:
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.bmm_weight = jnp.zeros((out_features, in_features))
        self.bmm_bias = jnp.zeros((out_features,))
        
        self.instance_norm_weight = jnp.ones((out_features,))
        self.instance_norm_bias = jnp.zeros((out_features,))
        self.eps = eps
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x, y):
        # As noted in the original code, instance norm over 1x1 spatial dimensions
        # results in a variance of 0 and mean equal to the input.
        # Thus, (x - mean) / sqrt(var + eps) is exactly 0.
        # The output of the instance norm is simply the instance_norm_bias.
        # The entire linear layer and instance norm simplify to just the bias.
        # We can skip the expensive matmul and norm, and directly compute (bias + y) * y.
        
        bias_2d = jnp.reshape(self.instance_norm_bias, (1, -1))
        
        block_n = min(y.shape[0], 512)
        block_c = min(y.shape[1], 512)
        grid_shape = (y.shape[0] // block_n, y.shape[1] // block_c)
        
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(y.shape, y.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_n, block_c), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_c), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_n, block_c), lambda i, j: (i, j)),
            ),
        )(y, bias_2d)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    return [
        jax.random.uniform(key1, (batch_size, in_features), dtype=jnp.float32),
        jax.random.uniform(key2, (batch_size, out_features), dtype=jnp.float32)
    ]

def get_init_inputs():
    return [in_features, out_features]
