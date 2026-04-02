import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x):
        alpha = self.alpha

        def elu_kernel(x_ref, o_ref):
            x_val = x_ref[:, :]
            o_ref[:, :] = jnp.where(x_val > 0, x_val, alpha * (jnp.exp(x_val) - 1))

        # Ensure 2D
        if x.ndim == 1:
            x = x[:, None]

        block = (128, 128)
        grid = (x.shape[0] // block[0], x.shape[1] // block[1])

        out = pl.pallas_call(
            elu_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
                out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
            ),
        )(x)

        return out

    def set_weights(self, weights_dict):
        pass

batch_size = 4096
dim = 393216

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return [1.0]
