import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def relu_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    o_ref[:, :] = jnp.maximum(x, 0)

class Model:
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Applies ReLU activation to the input array.

        Args:
            x: Input JAX array of any shape.

        Returns:
            JAX array: Output array with ReLU applied, same shape as input.
        """
        block = (128, 128)
        grid = (x.shape[0] // block[0], x.shape[1] // block[1])

        return pl.pallas_call(
            relu_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
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
