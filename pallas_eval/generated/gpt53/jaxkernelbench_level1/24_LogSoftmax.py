import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        self.dim = dim
    
    def forward(self, x):
        def kernel_fn(x_ref, o_ref):
            row = x_ref[...]  # shape (1, D)
            m = jnp.max(row, axis=1, keepdims=True)
            y = row - m
            exp_y = jnp.exp(y)
            s = jnp.sum(exp_y, axis=1, keepdims=True)
            logsum = jnp.log(s)
            o_ref[...] = y - logsum

        batch, dim = x.shape

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch,),
                in_specs=[pl.BlockSpec((1, dim), lambda i: (i, 0))],
                out_specs=pl.BlockSpec((1, dim), lambda i: (i, 0)),
            ),
        )(x)
    
    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
