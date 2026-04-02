import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import math

def new_gelu_kernel(x_ref, o_ref):
    x = x_ref[...]
    c1 = math.sqrt(2.0 / math.pi)
    c2 = 0.044715
    o_ref[...] = 0.5 * x * (1.0 + jnp.tanh(c1 * (x + c2 * jnp.power(x, 3.0))))

class Model:
    def __init__(self):
        pass
    
    def forward(self, x):
        block_0 = min(x.shape[0], 512)
        block_1 = min(x.shape[1], 512)
        grid_0 = x.shape[0] // block_0
        grid_1 = x.shape[1] // block_1
        
        return pl.pallas_call(
            new_gelu_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_0, grid_1),
                in_specs=[pl.BlockSpec((block_0, block_1), lambda i, j: (i, j))],
                out_specs=pl.BlockSpec((block_0, block_1), lambda i, j: (i, j)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        pass

batch_size = 8192
dim = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, dim))]

def get_init_inputs():
    return []
