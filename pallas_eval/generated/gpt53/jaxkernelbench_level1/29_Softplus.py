import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def softplus_kernel(x_ref, o_ref):
    x = x_ref[:, :]
    o_ref[:, :] = jnp.logaddexp(x, 0.0)


class Model:
    """
    Simple model that performs a Softplus activation.
    """
    def __init__(self):
        pass

    def forward(self, x):
        """
        Applies Softplus activation to the input tensor.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with Softplus applied, same shape as input.
        """
        # Ensure 2D (TPU constraint already satisfied here)
        M, N = x.shape

        block_m = 128
        block_n = 128

        grid = (M // block_m, N // block_n)

        return pl.pallas_call(
            softplus_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x)

    def set_weights(self, weights_dict):
        # No weights to set for this model
        pass


batch_size = 4096
dim = 393216


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim))
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
