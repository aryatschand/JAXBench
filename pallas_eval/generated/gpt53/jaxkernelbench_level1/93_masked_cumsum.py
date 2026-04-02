import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def forward(self, x, mask):
        assert self.dim == 1, "This optimized kernel only supports dim=1"

        def kernel_fn(x_ref, m_ref, o_ref):
            x_block = x_ref[...]
            m_block = m_ref[...]
            masked = x_block * m_block
            o_ref[...] = jnp.cumsum(masked, axis=1)

        batch, width = x.shape
        block = (1, width)

        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch,),
                in_specs=[
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec(block, lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x, mask)

    def set_weights(self, weights_dict):
        pass


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.uniform(key1, shape=(batch_size, *input_shape))
    mask = jax.random.bernoulli(key2, shape=(batch_size, *input_shape))
    return [x, mask]


def get_init_inputs():
    return [dim]
